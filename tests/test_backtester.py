"""统一 Backtester 的时间边界、无泄漏和指标契约测试。"""

import math
import unittest

import pandas as pd

from demand_forecast.backtesting.adapters import FivePeriodBacktestAdapter, TSBBacktestAdapter
from demand_forecast.backtesting.backtester import Backtester
from demand_forecast.backtesting.contracts import BacktestSplit, DailyForecast, ModelUnavailableError
from demand_forecast.backtesting.metrics import evaluate_forecasts
from demand_forecast.models.five_period import FivePeriodConfig, FivePeriodFittedModel
from demand_forecast.models.tsb import TSBConfig, TSBModel


def make_daily(
    sku: str,
    quantities: list[float | None],
    observed: list[bool] | None = None,
    start: str = "2025-01-01",
) -> pd.DataFrame:
    """构造连续、已排序的单 SKU 标准日序列。"""
    observed = observed or [True] * len(quantities)
    return pd.DataFrame(
        {
            "sku": sku,
            "date": pd.date_range(start, periods=len(quantities), freq="D"),
            "quantity": pd.Series(quantities, dtype="Float64"),
            "is_observed": observed,
            "launch_date": pd.Timestamp(start),
            "observation_reason": [pd.NA if value else "source_missing" for value in observed],
        }
    )


class StaticAdapter:
    """不访问测试 actual 的简化模型，用于验证 Backtester 的边界。"""

    name = "static"

    def __init__(self, prediction: float = 3.0) -> None:
        self.prediction = prediction
        self.fit_dates: list[pd.Timestamp] = []
        self.forecast_called = False

    def fit(self, train_series: pd.DataFrame, train_end: pd.Timestamp) -> dict[str, float]:
        self.fit_dates = list(train_series["date"])
        if train_series["date"].max() != train_end:
            raise AssertionError("Backtester 没有传入精确训练切片")
        return {"prediction": self.prediction}

    def forecast(
        self, fitted: object, train_series: pd.DataFrame, forecast_dates: pd.DatetimeIndex
    ) -> list[DailyForecast]:
        self.forecast_called = True
        if train_series["date"].max() > self.fit_dates[-1]:
            raise AssertionError("测试期数据被传入预测模型")
        sku = str(train_series["sku"].iloc[0])
        return [DailyForecast(sku, date, self.prediction) for date in forecast_dates]

    def serialize(self, fitted: object) -> dict[str, object]:
        return {"model_name": self.name, "prediction": self.prediction}


class UnavailableAdapter(StaticAdapter):
    """模拟模型历史不足，验证批量回测不会伪造指标。"""

    name = "unavailable"

    def fit(self, train_series: pd.DataFrame, train_end: pd.Timestamp) -> object:
        raise ModelUnavailableError("insufficient_history")


class FixedFivePeriodModel:
    """固定 R7 权重，专门验证 adapter 确实把预测追加到递归历史。"""

    name = "five_period"

    def fit(self, training_features: pd.DataFrame, trained_through: str) -> FivePeriodFittedModel:
        return FivePeriodFittedModel(
            weights=(1.0, 0.0, 0.0, 0.0, 0.0),
            feature_names=("current_mean_7", "current_mean_14", "current_mean_30", "current_mean_60", "current_mean_90"),
            feature_version="v1",
            trained_through=trained_through,
            config=FivePeriodConfig(),
            effective_step_size=1.0,
            n_training_rows=len(training_features),
            iterations=1,
            converged=True,
            final_objective=0.0,
        )

    def predict_one(self, fitted: FivePeriodFittedModel, row: dict[str, float]) -> float:
        return float(row["current_mean_7"])

    def serialize(self, fitted: FivePeriodFittedModel) -> dict[str, object]:
        return {"model_name": self.name, "weights": list(fitted.weights)}


class BacktesterTests(unittest.TestCase):
    """验证回测器不泄漏测试销量，且所有模型接受相同日历。"""

    def setUp(self) -> None:
        self.split = BacktestSplit("2025-01-01", "2025-01-04", "2025-01-05", "2025-01-06", expected_test_days=2)

    def test_exact_split_and_horizon(self) -> None:
        """训练含 train_end，测试从后一日开始，horizon 由日期推导。"""
        adapter = StaticAdapter()
        result = Backtester(self.split).backtest_one_sku(make_daily("A", [1, 2, 3, 4, 5, 6]), adapter)
        self.assertEqual(adapter.fit_dates[-1], pd.Timestamp("2025-01-04"))
        self.assertEqual([forecast.date for forecast in result.forecasts], list(self.split.test_dates))
        self.assertEqual(self.split.horizon, 2)

    def test_invalid_split_rejects_overlap_gap_and_wrong_expected_horizon(self) -> None:
        """边界配置错误必须在运行模型前失败。"""
        with self.assertRaisesRegex(ValueError, "紧接"):
            BacktestSplit("2025-01-01", "2025-01-04", "2025-01-04", "2025-01-06")
        with self.assertRaisesRegex(ValueError, "expected_test_days"):
            BacktestSplit("2025-01-01", "2025-01-04", "2025-01-05", "2025-01-06", expected_test_days=52)

    def test_test_target_mutation_changes_metrics_but_not_forecasts(self) -> None:
        """测试 actual 只在评价阶段出现，修改它不能影响完整 horizon 预测。"""
        original = make_daily("A", [1, 2, 3, 4, 5, 6])
        changed = original.copy()
        changed.loc[changed["date"] >= self.split.test_start, "quantity"] = [500.0, 600.0]
        first = Backtester(self.split).backtest_one_sku(original, StaticAdapter(3.0))
        second = Backtester(self.split).backtest_one_sku(changed, StaticAdapter(3.0))
        self.assertEqual(first.forecasts, second.forecasts)
        self.assertNotEqual(first.metrics.mae, second.metrics.mae)

    def test_evaluator_runs_only_after_forecast(self) -> None:
        """自定义 evaluator 可观察到 forecast 已完成，防止变成滚动一步评估。"""
        adapter = StaticAdapter()

        def evaluator(actuals: pd.DataFrame, forecasts: tuple[DailyForecast, ...], sku: str, dates: pd.DatetimeIndex):
            self.assertTrue(adapter.forecast_called)
            return evaluate_forecasts(actuals, forecasts, sku, dates)

        result = Backtester(self.split, evaluator=evaluator).backtest_one_sku(make_daily("A", [1, 2, 3, 4, 5, 6]), adapter)
        self.assertEqual(result.status, "completed")

    def test_metrics_exactness_and_bias_sign(self) -> None:
        """指标按规范公式计算，累计偏差固定为预测合计减实际合计。"""
        actuals = make_daily("A", [2.0, 4.0], start="2025-01-05")
        dates = pd.date_range("2025-01-05", periods=2, freq="D")
        metrics = evaluate_forecasts(actuals, [DailyForecast("A", dates[0], 1.0), DailyForecast("A", dates[1], 6.0)], "A", dates)
        self.assertEqual(metrics.mae, 1.5)
        self.assertEqual(metrics.mse, 2.5)
        self.assertAlmostEqual(metrics.rmse, math.sqrt(2.5))
        self.assertEqual(metrics.wape, 0.5)
        self.assertEqual(metrics.actual_total, 6.0)
        self.assertEqual(metrics.prediction_total, 7.0)
        self.assertEqual(metrics.cumulative_bias, 1.0)

    def test_zero_wape_and_unobserved_metric_mask(self) -> None:
        """全零实际 WAPE 未定义；不可观测日保留预测但不进入任何指标合计。"""
        dates = pd.date_range("2025-01-05", periods=2, freq="D")
        zero_metrics = evaluate_forecasts(
            make_daily("A", [0.0, 0.0], start="2025-01-05"),
            [DailyForecast("A", dates[0], 1.0), DailyForecast("A", dates[1], 0.0)],
            "A",
            dates,
        )
        self.assertIsNone(zero_metrics.wape)
        masked_metrics = evaluate_forecasts(
            make_daily("A", [5.0, None], [True, False], start="2025-01-05"),
            [DailyForecast("A", dates[0], 7.0), DailyForecast("A", dates[1], 100.0)],
            "A",
            dates,
        )
        self.assertEqual(masked_metrics.n_evaluated_days, 1)
        self.assertEqual(masked_metrics.actual_total, 5.0)
        self.assertEqual(masked_metrics.prediction_total, 7.0)

    def test_invalid_forecast_dates_and_missing_test_calendar_fail_fast(self) -> None:
        """不能用行顺序凑指标：预测和实际都必须精确覆盖测试日历。"""
        dates = self.split.test_dates
        with self.assertRaisesRegex(ValueError, "重复"):
            evaluate_forecasts(
                make_daily("A", [1.0, 2.0], start="2025-01-05"),
                [DailyForecast("A", dates[0], 1.0), DailyForecast("A", dates[0], 2.0)],
                "A",
                dates,
            )
        incomplete = make_daily("A", [1.0, 2.0, 3.0, 4.0, 5.0])
        with self.assertRaisesRegex(ValueError, "完整覆盖"):
            Backtester(self.split).backtest_one_sku(incomplete, StaticAdapter())

    def test_unavailable_model_and_no_observed_test_target_are_explicit(self) -> None:
        """历史不足和无测试标签都不能伪造完成状态。"""
        daily = make_daily("A", [1, 2, 3, 4, 5, 6])
        unavailable = Backtester(self.split).backtest_one_sku(daily, UnavailableAdapter())
        self.assertEqual(unavailable.status, "unavailable")
        self.assertEqual(unavailable.unavailable_reason, "insufficient_history")
        no_labels = make_daily("A", [1, 2, 3, 4, None, None], [True, True, True, True, False, False])
        no_label_result = Backtester(self.split).backtest_one_sku(no_labels, StaticAdapter())
        self.assertEqual(no_label_result.unavailable_reason, "no_observed_test_targets")
        self.assertEqual(len(no_label_result.forecasts), 2)

    def test_sku_isolation_in_batch_mode(self) -> None:
        """批量循环必须为每个 SKU 单独 fit，不能串用训练历史。"""
        daily = pd.concat([make_daily("A", [1, 1, 1, 1, 1, 1]), make_daily("B", [9, 9, 9, 9, 9, 9])], ignore_index=True)
        adapter = StaticAdapter()
        results = Backtester(self.split).backtest_all_skus(daily, adapter)
        self.assertEqual([result.sku for result in results], ["A", "B"])
        self.assertEqual(len(results), 2)

    def test_five_period_adapter_recursively_uses_prediction_history(self) -> None:
        """第二日 R7 必须包含第一日预测，而不是偷偷读取第一日测试实际。"""
        split = BacktestSplit("2025-01-01", "2025-04-01", "2025-04-02", "2025-04-03", expected_test_days=2)
        daily = make_daily("A", [float(value) for value in range(1, 94)])
        adapter = FivePeriodBacktestAdapter(model=FixedFivePeriodModel())
        result = Backtester(split).backtest_one_sku(daily, adapter)
        self.assertAlmostEqual(result.forecasts[0].prediction, 88.0)
        self.assertAlmostEqual(result.forecasts[1].prediction, 619.0 / 7.0)

    def test_tsb_adapter_keeps_constant_horizon(self) -> None:
        """TSB 正式测试期没有实际更新，所有预测必须保持 p×z 常数。"""
        split = BacktestSplit("2025-01-01", "2025-01-05", "2025-01-06", "2025-01-07", expected_test_days=2)
        adapter = TSBBacktestAdapter(TSBModel(TSBConfig(smoothing_values=(0.1,), validation_days=2, initialization_observed_days=2)))
        original = make_daily("A", [2, 0, 3, 0, 4, 100, 200])
        changed = original.copy()
        changed.loc[changed["date"] >= split.test_start, "quantity"] = [1000.0, 2000.0]
        result = Backtester(split).backtest_one_sku(original, adapter)
        changed_result = Backtester(split).backtest_one_sku(changed, adapter)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.forecasts[0].prediction, result.forecasts[1].prediction)
        self.assertEqual(result.forecasts, changed_result.forecasts)
        self.assertNotEqual(result.metrics.mae, changed_result.metrics.mae)


if __name__ == "__main__":
    unittest.main()
