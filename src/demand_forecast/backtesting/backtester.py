"""严格分离训练、预测和评价阶段的统一回测器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from demand_forecast.backtesting.adapters import BacktestModelAdapter
from demand_forecast.backtesting.contracts import (
    BacktestMetrics,
    BacktestResult,
    BacktestSplit,
    DailyForecast,
    ModelUnavailableError,
)
from demand_forecast.backtesting.metrics import evaluate_forecasts
from demand_forecast.data.schemas import validate_daily_sales
from demand_forecast.model_contracts import ForecastModel


Evaluator = Callable[[pd.DataFrame, Sequence[DailyForecast], str, pd.DatetimeIndex], BacktestMetrics]


class Backtester:
    """统一执行“训练切片 → 完整预测 → 延后评价”的单 SKU 正式回测。"""

    def __init__(self, split: BacktestSplit, evaluator: Evaluator = evaluate_forecasts) -> None:
        self.split = split
        self.evaluator = evaluator

    def backtest_one_sku(
        self,
        daily_series: pd.DataFrame,
        model: ForecastModel | BacktestModelAdapter,
    ) -> BacktestResult:
        """回测一个 SKU；测试 actual 在 adapter 预测返回前不会被切出或传递。"""
        series, sku = self._validate_single_series(daily_series)
        train_series = series.loc[
            (series["date"] >= self.split.train_start) & (series["date"] <= self.split.train_end)
        ].copy()
        try:
            # Phase A：adapter 只能拿到训练实际值和训练截止日。
            fitted = model.fit(train_series, self.split.train_end)
            fitted_metadata = model.serialize(fitted)

            # 正式 API 只接收 horizon；兼容分支只服务旧测试和迁移期内部 adapter。
            if hasattr(model, "predict"):
                forecasts = tuple(model.predict(fitted, self.split.horizon))
            else:
                forecasts = tuple(model.forecast(fitted, train_series.copy(), self.split.test_dates))
        except ModelUnavailableError as error:
            return BacktestResult(
                sku=sku,
                model_name=model.name,
                split=self.split,
                status="unavailable",
                unavailable_reason=error.reason,
                forecasts=tuple(),
                metrics=None,
                fitted_metadata=None,
            )

        # Phase C：完整 horizon 预测结束后，才从原始日序列取出测试实际值并评价。
        test_actuals = series.loc[
            (series["date"] >= self.split.test_start) & (series["date"] <= self.split.test_end)
        ].copy()
        try:
            metrics = self.evaluator(test_actuals, forecasts, sku, self.split.test_dates)
        except ModelUnavailableError as error:
            # 即使没有可评价标签，完整预测仍有诊断价值，必须保留。
            return BacktestResult(
                sku=sku,
                model_name=model.name,
                split=self.split,
                status="unavailable",
                unavailable_reason=error.reason,
                forecasts=forecasts,
                metrics=None,
                fitted_metadata=fitted_metadata,
            )

        return BacktestResult(
            sku=sku,
            model_name=model.name,
            split=self.split,
            status="completed",
            unavailable_reason=None,
            forecasts=forecasts,
            metrics=metrics,
            fitted_metadata=fitted_metadata,
        )

    def backtest_all_skus(
        self,
        daily_sales: pd.DataFrame,
        model: ForecastModel | BacktestModelAdapter,
    ) -> list[BacktestResult]:
        """按 SKU 独立调用核心方法，绝不共享 fitted state 或递归历史。"""
        daily = validate_daily_sales(daily_sales)
        results: list[BacktestResult] = []
        for _, sku_series in daily.groupby("sku", sort=True):
            results.append(self.backtest_one_sku(sku_series, model))
        return results

    def _validate_single_series(self, daily_series: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """检查正式窗口日历完整、唯一、有序，严重数据问题不自动修复。"""
        series = validate_daily_sales(daily_series)
        sku_values = series["sku"].dropna().unique()
        if len(sku_values) != 1 or series["sku"].isna().any() or (series["sku"].str.strip() == "").any():
            raise ValueError("backtest_one_sku 必须接收一个非空 SKU 的日序列")
        if not series["date"].is_monotonic_increasing:
            raise ValueError("日序列必须按日期升序，Backtester 不会静默排序")
        required_dates = pd.date_range(self.split.train_start, self.split.test_end, freq="D")
        in_range = series.loc[
            (series["date"] >= self.split.train_start) & (series["date"] <= self.split.test_end), "date"
        ]
        if len(in_range) != len(required_dates) or set(in_range) != set(required_dates):
            raise ValueError("日序列必须完整覆盖 train_start 至 test_end 的每个自然日")
        return series, str(sku_values[0])
