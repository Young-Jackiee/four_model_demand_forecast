"""Backtester 的日期、预测和结果数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, NotRequired, TypedDict

import numpy as np
import pandas as pd


class ModelUnavailableError(ValueError):
    """模型因历史或特征不足而无法完成本次回测。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"模型当前不可回测: {reason}")


@dataclass(frozen=True)
class BacktestSplit:
    """唯一表达正式训练和测试边界的不可变日期配置。"""

    train_start: pd.Timestamp | str
    train_end: pd.Timestamp | str
    test_start: pd.Timestamp | str
    test_end: pd.Timestamp | str
    expected_test_days: int | None = None

    def __post_init__(self) -> None:
        """统一日期精度，并在入口阻断最常见的边界错误。"""
        for field_name in ("train_start", "train_end", "test_start", "test_end"):
            value = pd.Timestamp(getattr(self, field_name)).normalize()
            if pd.isna(value):
                raise ValueError(f"{field_name} 必须是可解析日期")
            object.__setattr__(self, field_name, value)
        if self.train_start > self.train_end:
            raise ValueError("train_start 不能晚于 train_end")
        if self.test_end < self.test_start:
            raise ValueError("test_end 不能早于 test_start")
        if self.test_start != self.train_end + pd.Timedelta(days=1):
            raise ValueError("test_start 必须紧接 train_end，不能存在重叠或空档")
        if self.expected_test_days is not None:
            if self.expected_test_days < 1:
                raise ValueError("expected_test_days 必须为正整数或 None")
            if self.horizon != self.expected_test_days:
                raise ValueError("测试日期范围与 expected_test_days 不一致")

    @property
    def horizon(self) -> int:
        """从日期边界推导测试天数，避免在算法中写入魔法数字。"""
        return len(self.test_dates)

    @property
    def train_dates(self) -> pd.DatetimeIndex:
        """返回包含两端的正式训练日历。"""
        return pd.date_range(self.train_start, self.train_end, freq="D")

    @property
    def test_dates(self) -> pd.DatetimeIndex:
        """返回包含两端的正式预测日历。"""
        return pd.date_range(self.test_start, self.test_end, freq="D")


class DailyForecast(TypedDict):
    """文档规定的公共日预测接口；date 使用 Python 自然日而非 pandas 类型。"""

    sku: str
    date: date
    prediction: float
    components: NotRequired[dict[str, float]]


def make_daily_forecast(
    sku: str,
    forecast_date: date | pd.Timestamp | str,
    prediction: float,
    components: Mapping[str, float] | None = None,
) -> DailyForecast:
    """创建并校验公共预测字典，阻止非法值跨越模型边界。"""
    normalized_sku = str(sku).strip()
    normalized_date = pd.Timestamp(forecast_date).normalize()
    normalized_prediction = float(prediction)
    if not normalized_sku:
        raise ValueError("DailyForecast 的 sku 不能为空")
    if pd.isna(normalized_date):
        raise ValueError("DailyForecast 的 date 必须可解析")
    if not np.isfinite(normalized_prediction) or normalized_prediction < 0.0:
        raise ValueError("DailyForecast 的 prediction 必须是有限非负数")
    result: DailyForecast = {
        "sku": normalized_sku,
        "date": normalized_date.date(),
        "prediction": normalized_prediction,
    }
    if components is not None:
        normalized_components: dict[str, float] = {}
        for name, value in components.items():
            normalized_name = str(name).strip()
            normalized_value = float(value)
            if not normalized_name or not np.isfinite(normalized_value):
                raise ValueError("DailyForecast 的 components 必须是名称和有限数值")
            normalized_components[normalized_name] = normalized_value
        result["components"] = normalized_components
    return result


def validate_daily_forecast(value: object) -> DailyForecast:
    """验证外部传入的预测记录，并返回字段类型已标准化的副本。"""
    if not isinstance(value, Mapping):
        raise ValueError("预测结果必须全部是 DailyForecast 字典")
    required = {"sku", "date", "prediction"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"DailyForecast 缺少字段: {sorted(missing)}")
    raw_components = value.get("components")
    if raw_components is not None and not isinstance(raw_components, Mapping):
        raise ValueError("DailyForecast 的 components 必须是字典或省略")
    return make_daily_forecast(
        str(value["sku"]),
        value["date"],
        float(value["prediction"]),
        raw_components,
    )


@dataclass(frozen=True)
class BacktestMetrics:
    """仅基于可观测测试标签计算的统一指标。"""

    mae: float
    mse: float
    rmse: float
    wape: float | None
    actual_total: float
    prediction_total: float
    cumulative_bias: float
    n_evaluated_days: int
    evaluated_dates: tuple[pd.Timestamp, ...]

    def __post_init__(self) -> None:
        """保存实际参与指标计算的日期，供后续 ModelSelector 验证可比性。"""
        dates = tuple(pd.Timestamp(date).normalize() for date in self.evaluated_dates)
        if len(dates) != self.n_evaluated_days or not dates:
            raise ValueError("evaluated_dates 必须非空且数量等于 n_evaluated_days")
        if len(set(dates)) != len(dates) or list(dates) != sorted(dates):
            raise ValueError("evaluated_dates 必须唯一且按日期升序")
        numeric_values = (self.mae, self.mse, self.rmse, self.actual_total, self.prediction_total, self.cumulative_bias)
        if not all(np.isfinite(value) for value in numeric_values):
            raise ValueError("BacktestMetrics 数值必须有限")
        if self.wape is not None and not np.isfinite(self.wape):
            raise ValueError("WAPE 必须为有限数或 None")
        if self.mae < 0.0 or self.mse < 0.0 or self.rmse < 0.0 or (self.wape is not None and self.wape < 0.0):
            raise ValueError("误差指标不能为负")
        object.__setattr__(self, "evaluated_dates", dates)


@dataclass(frozen=True)
class BacktestResult:
    """单 SKU、单模型、单一正式时间窗口的完整回测结果。"""

    sku: str
    model_name: str
    split: BacktestSplit
    status: str
    unavailable_reason: str | None
    forecasts: tuple[DailyForecast, ...]
    metrics: BacktestMetrics | None
    fitted_metadata: Mapping[str, object] | None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        """状态字段决定指标是否存在，避免半完成结果被误用于模型选择。"""
        if self.status not in {"completed", "unavailable", "failed"}:
            raise ValueError("BacktestResult.status 只能为 completed、unavailable 或 failed")
        if self.status == "completed" and self.metrics is None:
            raise ValueError("completed 回测必须包含 metrics")
        if self.status == "unavailable" and not self.unavailable_reason:
            raise ValueError("unavailable 回测必须包含 unavailable_reason")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed 回测必须包含 failure_reason")
