"""统一的严格时间回测组件。"""

from demand_forecast.backtesting.adapters import Direct10BacktestAdapter, FivePeriodBacktestAdapter, HurdleBacktestAdapter, TSBBacktestAdapter
from demand_forecast.backtesting.backtester import Backtester
from demand_forecast.backtesting.experiment import FormalBacktestRunner, results_to_metrics_frame
from demand_forecast.model_selection import (
    ModelSelector,
    SelectionInputError,
    SelectionResult,
    read_selection_results,
    write_selection_results,
)
from demand_forecast.backtesting.contracts import (
    BacktestMetrics,
    BacktestResult,
    BacktestSplit,
    DailyForecast,
    ModelUnavailableError,
)
from demand_forecast.model_contracts import ForecastModel

__all__ = [
    "BacktestMetrics",
    "BacktestResult",
    "BacktestSplit",
    "Backtester",
    "DailyForecast",
    "ForecastModel",
    "Direct10BacktestAdapter",
    "FivePeriodBacktestAdapter",
    "FormalBacktestRunner",
    "HurdleBacktestAdapter",
    "ModelUnavailableError",
    "ModelSelector",
    "SelectionInputError",
    "SelectionResult",
    "TSBBacktestAdapter",
    "results_to_metrics_frame",
    "read_selection_results",
    "write_selection_results",
]
