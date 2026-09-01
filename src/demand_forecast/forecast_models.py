"""四模型的统一 ForecastModel 实现。

核心数学模型继续保持独立；本模块只处理训练边界、递归历史和持久化契约。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import pandas as pd

from demand_forecast.backtesting.adapters import (
    BacktestModelAdapter,
    Direct10BacktestAdapter,
    FivePeriodBacktestAdapter,
    HurdleBacktestAdapter,
    TSBBacktestAdapter,
)
from demand_forecast.backtesting.contracts import DailyForecast, ModelUnavailableError
from demand_forecast.data.schemas import validate_daily_sales
from demand_forecast.model_contracts import DailySeries


FORECAST_ARTIFACT_SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class AdapterFittedModel:
    """统一模型的冻结状态：核心参数加上递归预测所需的历史尾部。"""

    model_name: str
    sku: str
    trained_through: str
    core_fitted: object
    inference_history: pd.DataFrame
    feature_version: str
    data_policy: Mapping[str, object]


class AdapterForecastModel:
    """把既有 adapter 升格为文档规定的 fit/predict/serialize 公共接口。"""

    name: str
    history_days: int

    def __init__(self, adapter: BacktestModelAdapter, history_days: int) -> None:
        self.adapter = adapter
        self.name = adapter.name
        self.history_days = history_days

    def fit(self, series: DailySeries, train_end: date) -> AdapterFittedModel:
        """显式 cutoff 后训练，并保存刚好足够递归预测的历史尾部。"""
        cutoff = pd.Timestamp(train_end).normalize()
        if pd.isna(cutoff):
            raise ValueError("train_end 必须是可解析日期")
        daily = validate_daily_sales(series)
        self._validate_single_sorted_sku(daily)
        training = daily.loc[daily["date"] <= cutoff].copy()
        if training.empty or cutoff not in set(training["date"]):
            raise ValueError("train_end 必须存在于输入日序列")
        if (training["date"] > cutoff).any():  # 防御性断言，避免未来数据绕过调用边界。
            raise ValueError("训练数据不能包含 train_end 之后的日期")

        core_fitted = self.adapter.fit(training.copy(), cutoff)
        if getattr(core_fitted, "converged", True) is False:
            raise ModelUnavailableError("optimizer_not_converged")

        sku = str(training["sku"].iloc[0])
        return AdapterFittedModel(
            model_name=self.name,
            sku=sku,
            trained_through=cutoff.strftime("%Y-%m-%d"),
            core_fitted=core_fitted,
            inference_history=self._history_tail(training),
            feature_version=self._feature_version(core_fitted),
            data_policy=self._data_policy(training),
        )

    def predict(self, fitted: AdapterFittedModel, horizon: int) -> list[DailyForecast]:
        """只从 fitted 内的历史递归预测；不接收也不读取未来实际销量。"""
        self._validate_fitted(fitted)
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon 必须是正整数")
        dates = pd.date_range(
            pd.Timestamp(fitted.trained_through).normalize() + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        # adapter 的 forecast 实现会复制并递归追加预测值，fitted 本身保持不可变。
        return list(self.adapter.forecast(fitted.core_fitted, fitted.inference_history.copy(), dates))

    def serialize(self, fitted: AdapterFittedModel) -> dict[str, object]:
        """保存模型参数、特征版本、数据策略和最小递归历史。"""
        self._validate_fitted(fitted)
        core_payload = dict(self.adapter.serialize(fitted.core_fitted))
        return {
            "schema_version": FORECAST_ARTIFACT_SCHEMA_VERSION,
            "model_name": self.name,
            "model_version": core_payload.get("implementation_version", "v1"),
            "sku": fitted.sku,
            "trained_through": fitted.trained_through,
            "feature_version": fitted.feature_version,
            "fitted_model": core_payload,
            "inference_history": self._history_to_records(fitted.inference_history),
            "data_policy": dict(fitted.data_policy),
        }

    def deserialize(self, payload: Mapping[str, object]) -> AdapterFittedModel:
        """恢复一个可直接 predict 的统一模型状态。"""
        if payload.get("schema_version") != FORECAST_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("不支持的 ForecastModel artifact 版本")
        if payload.get("model_name") != self.name:
            raise ValueError("artifact 的 model_name 与当前模型不匹配")
        core_payload = payload.get("fitted_model")
        records = payload.get("inference_history")
        data_policy = payload.get("data_policy")
        if not isinstance(core_payload, Mapping) or not isinstance(records, list) or not isinstance(data_policy, Mapping):
            raise ValueError("ForecastModel artifact 缺少必要字段")
        core_model = getattr(self.adapter, "model", None)
        if core_model is None or not hasattr(core_model, "deserialize"):
            raise ValueError("当前模型不支持反序列化")
        history = self._records_to_history(records)
        fitted = AdapterFittedModel(
            model_name=self.name,
            sku=str(payload["sku"]),
            trained_through=str(payload["trained_through"]),
            core_fitted=core_model.deserialize(core_payload),
            inference_history=history,
            feature_version=str(payload["feature_version"]),
            data_policy=dict(data_policy),
        )
        self._validate_fitted(fitted)
        return fitted

    def _history_tail(self, training: pd.DataFrame) -> pd.DataFrame:
        """保存当前模型真正需要的尾部，而不是重复保存完整训练集。"""
        return training.tail(self.history_days).copy().reset_index(drop=True)

    @staticmethod
    def _feature_version(core_fitted: object) -> str:
        """TSB 没有特征矩阵，也必须记录其原始日序列版本。"""
        return str(getattr(core_fitted, "feature_version", "raw_daily_v1"))

    @staticmethod
    def _data_policy(training: pd.DataFrame) -> dict[str, object]:
        """把零销量、不可观测日和上市状态语义固化到 artifact。"""
        launch = pd.Timestamp(training["launch_date"].iloc[-1]).strftime("%Y-%m-%d")
        return {
            "missing_observed_day": "zero",
            "unobserved_day": "null_and_excluded",
            "launch_date": launch,
            "yoy_offset_days": 365,
            "timezone_policy": "normalized_business_date",
        }

    @staticmethod
    def _history_to_records(history: pd.DataFrame) -> list[dict[str, object]]:
        """把 pandas 的 NA 和时间类型转换为 JSON 友好的最小记录。"""
        rows: list[dict[str, object]] = []
        for row in history.itertuples(index=False):
            rows.append(
                {
                    "sku": str(row.sku),
                    "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                    "quantity": None if pd.isna(row.quantity) else float(row.quantity),
                    "is_observed": bool(row.is_observed),
                    "launch_date": pd.Timestamp(row.launch_date).strftime("%Y-%m-%d"),
                    "observation_reason": None if pd.isna(row.observation_reason) else str(row.observation_reason),
                }
            )
        return rows

    @staticmethod
    def _records_to_history(records: list[object]) -> pd.DataFrame:
        """恢复并复用标准 DailySeries 校验，防止损坏 artifact 静默上线。"""
        if not all(isinstance(row, Mapping) for row in records):
            raise ValueError("inference_history 必须是记录列表")
        history = pd.DataFrame([dict(row) for row in records])
        if history.empty:
            raise ValueError("inference_history 不能为空")
        return validate_daily_sales(history)

    def _validate_fitted(self, fitted: AdapterFittedModel) -> None:
        """检查模型、历史和截止日彼此一致。"""
        if not isinstance(fitted, AdapterFittedModel) or fitted.model_name != self.name:
            raise ValueError("fitted model 与 ForecastModel 不匹配")
        if pd.isna(pd.Timestamp(fitted.trained_through)) or not fitted.sku.strip():
            raise ValueError("fitted model 的 SKU 或 trained_through 非法")
        history = validate_daily_sales(fitted.inference_history)
        self._validate_single_sorted_sku(history)
        if str(history["sku"].iloc[-1]) != fitted.sku:
            raise ValueError("inference_history 的 SKU 与 fitted model 不匹配")
        if pd.Timestamp(history["date"].iloc[-1]) != pd.Timestamp(fitted.trained_through):
            raise ValueError("inference_history 必须截至 trained_through")
        if len(history) > self.history_days:
            raise ValueError("inference_history 超过模型允许的历史尾部")

    @staticmethod
    def _validate_single_sorted_sku(series: pd.DataFrame) -> None:
        sku_values = series["sku"].dropna().astype(str).unique()
        if len(sku_values) != 1 or not str(sku_values[0]).strip():
            raise ValueError("ForecastModel 每次只能处理一个非空 SKU")
        if not series["date"].is_monotonic_increasing:
            raise ValueError("ForecastModel 输入必须按日期升序")


class FivePeriodForecastModel(AdapterForecastModel):
    """文档 API 下的五周期基准模型。"""

    def __init__(self, adapter: FivePeriodBacktestAdapter | None = None) -> None:
        super().__init__(adapter or FivePeriodBacktestAdapter(), history_days=90)


class Direct10ForecastModel(AdapterForecastModel):
    """文档 API 下的 Direct10 模型。"""

    def __init__(self, adapter: Direct10BacktestAdapter | None = None) -> None:
        super().__init__(adapter or Direct10BacktestAdapter(), history_days=455)


class TSBForecastModel(AdapterForecastModel):
    """文档 API 下的 TSB 模型。"""

    def __init__(self, adapter: TSBBacktestAdapter | None = None) -> None:
        # TSB 多步预测只依赖冻结状态；保留截止日一行用于统一 artifact 校验。
        super().__init__(adapter or TSBBacktestAdapter(), history_days=1)


class HurdleForecastModel(AdapterForecastModel):
    """文档 API 下的两段式模型。"""

    def __init__(self, adapter: HurdleBacktestAdapter | None = None) -> None:
        super().__init__(adapter or HurdleBacktestAdapter(), history_days=90)


def default_forecast_model_factories() -> dict[str, type[AdapterForecastModel]]:
    """生产和服务共用的四模型工厂映射。"""
    return {
        "five_period": FivePeriodForecastModel,
        "direct10": Direct10ForecastModel,
        "tsb": TSBForecastModel,
        "hurdle": HurdleForecastModel,
    }
