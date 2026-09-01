"""四模型正式回测的轻量编排与结果汇总。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from demand_forecast.backtesting.adapters import BacktestModelAdapter
from demand_forecast.backtesting.backtester import Backtester
from demand_forecast.backtesting.contracts import BacktestResult
from demand_forecast.data.schemas import validate_daily_sales
from demand_forecast.model_contracts import ForecastModel


BacktestAdapterFactory = Callable[[], ForecastModel | BacktestModelAdapter]


class FormalBacktestRunner:
    """按 SKU × 模型执行正式回测；每一次调用都创建独立 adapter 与 fitted state。"""

    def __init__(self, backtester: Backtester) -> None:
        self.backtester = backtester

    def run(
        self,
        daily_sales: pd.DataFrame,
        adapter_factories: Sequence[BacktestAdapterFactory],
    ) -> list[BacktestResult]:
        """返回完整结果，不持久化、不选 winner，也不吞掉未知程序异常。"""
        if not adapter_factories:
            raise ValueError("adapter_factories 不能为空")
        daily = validate_daily_sales(daily_sales)
        results: list[BacktestResult] = []
        for _, sku_series in daily.groupby("sku", sort=True):
            # 每个模型拿到独立副本；即使未来模型误修改输入，也不能污染同 SKU 的其他模型。
            for factory in adapter_factories:
                model = factory()
                results.append(self.backtester.backtest_one_sku(sku_series.copy(), model))
        return results


def results_to_metrics_frame(results: Sequence[BacktestResult]) -> pd.DataFrame:
    """将 BacktestResult 展平为 ModelSelector 可消费的内存指标表。"""
    rows: list[dict[str, object]] = []
    for result in results:
        metrics = result.metrics
        rows.append(
            {
                "sku": result.sku,
                "model_name": result.model_name,
                "status": result.status,
                "unavailable_reason": result.unavailable_reason,
                "train_start": result.split.train_start,
                "train_end": result.split.train_end,
                "test_start": result.split.test_start,
                "test_end": result.split.test_end,
                "forecast_days": len(result.forecasts),
                "evaluated_days": metrics.n_evaluated_days if metrics else 0,
                "evaluated_dates": tuple(date.strftime("%Y-%m-%d") for date in metrics.evaluated_dates) if metrics else tuple(),
                "mae": metrics.mae if metrics else None,
                "mse": metrics.mse if metrics else None,
                "rmse": metrics.rmse if metrics else None,
                "wape": metrics.wape if metrics else None,
                "actual_total": metrics.actual_total if metrics else None,
                "prediction_total": metrics.prediction_total if metrics else None,
                "cumulative_bias": metrics.cumulative_bias if metrics else None,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "sku",
            "model_name",
            "status",
            "unavailable_reason",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "forecast_days",
            "evaluated_days",
            "evaluated_dates",
            "mae",
            "mse",
            "rmse",
            "wape",
            "actual_total",
            "prediction_total",
            "cumulative_bias",
        ],
    )
