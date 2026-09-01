"""同源构造历史训练特征与递归预测特征。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand_forecast.data.schemas import validate_daily_sales
from demand_forecast.features.definitions import WINDOWS, feature_names_for


@dataclass(frozen=True)
class SingleFeatureResult:
    """单个目标日的特征结果；不可用时保留机器可读原因。"""

    values: dict[str, float] | None
    unavailable_reason: str | None

    @property
    def is_available(self) -> bool:
        """调用方无需检查 values 是否为空。"""
        return self.values is not None


@dataclass(frozen=True)
class FeatureBuildResult:
    """批量构造结果；features 可直接送入训练，unavailable 用于诊断。"""

    features: pd.DataFrame
    unavailable: pd.DataFrame


class FeatureBuilder:
    """只计算确定性原始特征，不负责标准化、训练或递归状态更新。"""

    def build_historical(
        self,
        daily_sales: pd.DataFrame,
        feature_set: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> FeatureBuildResult:
        """为每个可观测目标日构建特征；窗口不完整时只记录原因，不填补。"""
        daily = validate_daily_sales(daily_sales)
        required_names = feature_names_for(feature_set)
        start = pd.Timestamp(start_date).normalize() if start_date else daily["date"].min()
        end = pd.Timestamp(end_date).normalize() if end_date else daily["date"].max()
        if start > end:
            raise ValueError("start_date 不能晚于 end_date")

        feature_rows: list[dict[str, object]] = []
        unavailable_rows: list[dict[str, object]] = []
        for sku, sku_daily in daily.groupby("sku", sort=True):
            # 每个 SKU 单独切历史，物理上避免 groupby/rolling 跨 SKU 串值。
            history_source = self._daily_to_history(sku_daily)
            prepared_history = self._validate_history(
                history_source,
                sku_daily["date"].max() + pd.Timedelta(days=1),
            )
            target_rows = sku_daily.loc[
                (sku_daily["date"] >= start) & (sku_daily["date"] <= end)
            ].sort_values("date")
            for target in target_rows.itertuples(index=False):
                target_date = pd.Timestamp(target.date)
                if not bool(target.is_observed):
                    unavailable_rows.append(
                        self._unavailable_row(sku, target_date, "target_date_unobserved", feature_set)
                    )
                    continue

                # 只切取 t-1 及以前的索引，批量模式与单步模式共用同一计算核心。
                prior_history = prepared_history.loc[: target_date - pd.Timedelta(days=1)]
                result = self._build_from_prepared(prior_history, target_date, required_names)
                if result.is_available:
                    row: dict[str, object] = {
                        "sku": sku,
                        "date": target_date,
                        "target_quantity": float(target.quantity),
                    }
                    row.update(result.values or {})
                    feature_rows.append(row)
                else:
                    unavailable_rows.append(
                        self._unavailable_row(sku, target_date, result.unavailable_reason or "unknown", feature_set)
                    )

        feature_columns = ["sku", "date", "target_quantity", *required_names]
        features = pd.DataFrame(feature_rows, columns=feature_columns)
        unavailable = pd.DataFrame(
            unavailable_rows,
            columns=["sku", "date", "feature_set", "reason"],
        )
        return FeatureBuildResult(features=features, unavailable=unavailable)

    def build_next(
        self,
        history: pd.DataFrame,
        target_date: str | pd.Timestamp,
        feature_set: str,
    ) -> SingleFeatureResult:
        """使用 target_date 前的单 SKU 历史，构造一日特征。

        未来递归预测由调用方追加预测销量和发生概率后，再调用本方法。
        """
        target = pd.Timestamp(target_date).normalize()
        required_names = feature_names_for(feature_set)
        prepared = self._validate_history(history, target)
        return self._build_from_prepared(prepared, target, required_names)

    def _build_from_prepared(
        self,
        prepared: pd.DataFrame,
        target: pd.Timestamp,
        required_names: tuple[str, ...],
    ) -> SingleFeatureResult:
        """从已校验的 t 前历史计算特征；供批量和单步入口共同调用。"""
        values: dict[str, float] = {}

        needs_current = any(name.startswith("current_mean_") for name in required_names)
        needs_occurrence = any(name.startswith("occurrence_rate_") for name in required_names)
        needs_yoy = any(name.startswith("yoy_mean_") for name in required_names)

        if needs_current or needs_occurrence:
            for window in WINDOWS:
                dates = pd.date_range(target - pd.Timedelta(days=window), periods=window, freq="D")
                window_values, reason = self._read_window(prepared, dates)
                if reason:
                    return SingleFeatureResult(None, f"current_window_{window}_{reason}")
                if needs_current and f"current_mean_{window}" in required_names:
                    values[f"current_mean_{window}"] = float(window_values["quantity"].mean())
                if needs_occurrence and f"occurrence_rate_{window}" in required_names:
                    values[f"occurrence_rate_{window}"] = float(window_values["occurrence"].mean())

        if needs_yoy:
            # 严格使用 t - 365 个自然日，不采用“上一自然年同日”偏移。
            yoy_anchor = target - pd.Timedelta(days=365)
            for window in WINDOWS:
                dates = pd.date_range(yoy_anchor - pd.Timedelta(days=window), periods=window, freq="D")
                window_values, reason = self._read_window(prepared, dates)
                if reason:
                    return SingleFeatureResult(None, f"yoy_window_{window}_{reason}")
                if f"yoy_mean_{window}" in required_names:
                    values[f"yoy_mean_{window}"] = float(window_values["quantity"].mean())

        if "dow_sin" in required_names:
            weekday = target.dayofweek  # 周一为 0，周日为 6。
            values["dow_sin"] = float(np.sin(2 * np.pi * weekday / 7))
            values["dow_cos"] = float(np.cos(2 * np.pi * weekday / 7))

        return SingleFeatureResult(values=values, unavailable_reason=None)

    def _daily_to_history(self, sku_daily: pd.DataFrame) -> pd.DataFrame:
        """把实际销量日序列转换为统一历史格式；实际发生率由 y>0 得到。"""
        history = sku_daily[["date", "quantity", "is_observed"]].copy()
        history["occurrence"] = pd.Series(pd.NA, index=history.index, dtype="Float64")
        observed = history["is_observed"].astype(bool)
        history.loc[observed, "occurrence"] = (history.loc[observed, "quantity"] > 0).astype(float)
        return history

    def _validate_history(self, history: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
        """校验单 SKU 历史，并拒绝把目标日或未来值偷偷带进特征计算。"""
        required = {"date", "quantity", "occurrence"}
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"history 缺少必填字段: {sorted(missing)}")
        result = history.copy()
        result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
        result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce").astype("Float64")
        result["occurrence"] = pd.to_numeric(result["occurrence"], errors="coerce").astype("Float64")
        if "sku" in result.columns and result["sku"].nunique(dropna=True) > 1:
            raise ValueError("build_next 的 history 只能包含一个 SKU")
        if result["date"].isna().any() or result["date"].duplicated().any():
            raise ValueError("history 的 date 必须可解析且每天仅一行")
        if (result["date"] >= target_date).any():
            raise ValueError("history 只能包含 target_date 之前的数据")
        if (result["quantity"].dropna() < 0).any():
            raise ValueError("history 的 quantity 不能为负")
        if ((result["occurrence"].dropna() < 0) | (result["occurrence"].dropna() > 1)).any():
            raise ValueError("history 的 occurrence 必须位于 [0, 1]")
        if "is_observed" in result.columns:
            result["is_observed"] = result["is_observed"].astype("boolean")
        return result.set_index("date", drop=False).sort_index()

    def _read_window(
        self,
        history: pd.DataFrame,
        required_dates: pd.DatetimeIndex,
    ) -> tuple[pd.DataFrame | None, str | None]:
        """读取完整窗口；缺日期、不可观测或空值都会使该特征不可用。"""
        window = history.reindex(required_dates)
        if window["date"].isna().any():
            return None, "insufficient_history"
        if "is_observed" in window.columns and (~window["is_observed"].fillna(False)).any():
            return None, "unobserved_history"
        if window[["quantity", "occurrence"]].isna().any().any():
            return None, "unobserved_history"
        return window, None

    @staticmethod
    def _unavailable_row(sku: str, target_date: pd.Timestamp, reason: str, feature_set: str) -> dict[str, object]:
        """统一诊断表结构，后续模型训练器可据此说明跳过原因。"""
        return {"sku": sku, "date": target_date, "feature_set": feature_set, "reason": reason}
