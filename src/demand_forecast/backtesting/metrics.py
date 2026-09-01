"""统一且纯函数化的回测指标计算。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from demand_forecast.backtesting.contracts import (
    BacktestMetrics,
    DailyForecast,
    ModelUnavailableError,
    validate_daily_forecast,
)
from demand_forecast.data.schemas import validate_daily_sales


def evaluate_forecasts(
    test_actuals: pd.DataFrame,
    forecasts: Sequence[DailyForecast],
    expected_sku: str,
    expected_dates: pd.DatetimeIndex,
) -> BacktestMetrics:
    """在完整预测完成后，按日期键对齐可观测实际值并计算统一指标。"""
    actuals = validate_daily_sales(test_actuals)
    _validate_actuals(actuals, expected_sku, expected_dates)
    normalized_forecasts = _validate_forecasts(forecasts, expected_sku, expected_dates)
    prediction_by_date = {pd.Timestamp(forecast["date"]): forecast["prediction"] for forecast in normalized_forecasts}
    observed = actuals.loc[actuals["is_observed"].astype(bool), ["date", "quantity"]]
    if observed.empty:
        raise ModelUnavailableError("no_observed_test_targets")

    actual_values = observed["quantity"].to_numpy(dtype=float)
    predicted_values = np.asarray([prediction_by_date[date] for date in observed["date"]], dtype=float)
    residual = predicted_values - actual_values
    absolute_error = np.abs(residual)
    actual_total = float(actual_values.sum())
    prediction_total = float(predicted_values.sum())
    wape = None if np.isclose(actual_total, 0.0, atol=1e-12) else float(absolute_error.sum() / actual_total)
    mse = float(np.mean(residual**2))
    return BacktestMetrics(
        mae=float(np.mean(absolute_error)),
        mse=mse,
        rmse=float(np.sqrt(mse)),
        wape=wape,
        actual_total=actual_total,
        prediction_total=prediction_total,
        cumulative_bias=float(prediction_total - actual_total),
        n_evaluated_days=len(observed),
        evaluated_dates=tuple(pd.Timestamp(date).normalize() for date in observed["date"]),
    )


def _validate_actuals(actuals: pd.DataFrame, expected_sku: str, expected_dates: pd.DatetimeIndex) -> None:
    """指标层只接受完整测试日历中的一个 SKU，禁止按行位置猜测对齐。"""
    skus = actuals["sku"].dropna().unique()
    if len(skus) != 1 or str(skus[0]) != expected_sku:
        raise ValueError("测试实际值的 SKU 与回测结果不匹配")
    dates = pd.DatetimeIndex(actuals["date"])
    if len(actuals) != len(expected_dates) or set(dates) != set(expected_dates):
        raise ValueError("测试实际值日期必须与完整测试日历一一对应")


def _validate_forecasts(
    forecasts: Sequence[DailyForecast], expected_sku: str, expected_dates: pd.DatetimeIndex
) -> list[DailyForecast]:
    """预测必须覆盖每个测试自然日一次，且不可依赖输入顺序碰巧正确。"""
    if len(forecasts) != len(expected_dates):
        raise ValueError("预测数量必须等于测试 horizon")
    normalized = [validate_daily_forecast(forecast) for forecast in forecasts]
    if any(forecast["sku"] != expected_sku for forecast in normalized):
        raise ValueError("预测结果包含错误 SKU")
    dates = [pd.Timestamp(forecast["date"]) for forecast in normalized]
    if len(set(dates)) != len(dates):
        raise ValueError("预测日期不能重复")
    if list(dates) != list(expected_dates):
        raise ValueError("预测日期必须按完整测试日历连续且一一对应")
    return normalized
