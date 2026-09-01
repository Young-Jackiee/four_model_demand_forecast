"""四模型统一正式回测的时间、泄漏、隔离与汇总集成测试。"""

import unittest

import pandas as pd

from demand_forecast.backtesting import (
    BacktestSplit,
    Backtester,
    DailyForecast,
    Direct10BacktestAdapter,
    FivePeriodBacktestAdapter,
    FormalBacktestRunner,
    HurdleBacktestAdapter,
    TSBBacktestAdapter,
    results_to_metrics_frame,
)
from demand_forecast.backtesting.metrics import evaluate_forecasts
from demand_forecast.backtesting.contracts import ModelUnavailableError
from demand_forecast.models.direct10 import Direct10Config, Direct10Model
from demand_forecast.models.five_period import FivePeriodConfig, FivePeriodModel
from demand_forecast.models.hurdle import HurdleConfig, HurdleModel
from demand_forecast.models.tsb import TSBConfig, TSBModel


FORMAL_SPLIT = BacktestSplit(
    "2025-01-01",
    "2026-06-30",
    "2026-07-01",
    "2026-08-21",
    expected_test_days=52,
)


def make_daily(sku: str = "A") -> pd.DataFrame:
    """生成满足 Direct10、TSB 与 Hurdle 训练条件的完整正式实验日历。"""
    quantities = [0.0 if index % 4 == 0 else float((index % 7) + 1) for index in range(598)]
    return pd.DataFrame(
        {
            "sku": sku,
            "date": pd.date_range("2025-01-01", periods=598, freq="D"),
            "quantity": quantities,
            "is_observed": True,
            "launch_date": pd.Timestamp("2025-01-01"),
            "observation_reason": pd.NA,
        }
    )


def model_factories():
    """缩小网格仅为测试提速；正式边界和四种预测语义保持不变。"""
    return (
        lambda: FivePeriodBacktestAdapter(FivePeriodModel(FivePeriodConfig(max_iter=5_000))),
        lambda: TSBBacktestAdapter(
            TSBModel(TSBConfig(smoothing_values=(0.1,), validation_days=90, initialization_observed_days=7))
        ),
        lambda: HurdleBacktestAdapter(
            HurdleModel(HurdleConfig(lambda_values=(0.1,), validation_days=60, max_iter=100, tol=1e-8))
        ),
        lambda: Direct10BacktestAdapter(Direct10Model(Direct10Config(max_iter=5_000, tol=1e-8))),
    )


class MeanAdapter:
    """轻量 adapter，用于只验证 runner 的 SKU/模型状态隔离而非模型数学。"""

    def __init__(self, name: str) -> None:
        self.name = name

    def fit(self, train_series: pd.DataFrame, train_end: pd.Timestamp) -> dict[str, float]:
        return {"prediction": float(train_series.loc[train_series["is_observed"], "quantity"].mean())}

    def forecast(self, fitted: object, train_series: pd.DataFrame, forecast_dates: pd.DatetimeIndex) -> list[DailyForecast]:
        sku = str(train_series["sku"].iloc[0])
        prediction = float(fitted["prediction"])
        return [DailyForecast(sku, date, prediction) for date in forecast_dates]

    def serialize(self, fitted: object) -> dict[str, object]:
        return {"model_name": self.name, **fitted}


class UnavailableAdapter(MeanAdapter):
    """模拟正常业务不可适用，确认 runner 不会中断同 SKU 的其他模型。"""

    def fit(self, train_series: pd.DataFrame, train_end: pd.Timestamp) -> dict[str, float]:
        raise ModelUnavailableError("insufficient_history")


class FormalBacktestIntegrationTests(unittest.TestCase):
    """覆盖正式窗口的统一契约，而不是重复各模型的内部数学单测。"""

    def test_all_four_models_ignore_formal_test_actuals_and_share_observation_mask(self) -> None:
        """改 52 天测试标签只能改指标；每个模型仍预测完整 52 天并统一评价 51 天。"""
        daily = make_daily()
        masked_date = FORMAL_SPLIT.test_start + pd.Timedelta(days=9)
        daily.loc[daily["date"] == masked_date, ["is_observed", "quantity"]] = [False, pd.NA]
        daily.loc[daily["date"] == masked_date, "observation_reason"] = "source_missing"
        changed = daily.copy()
        observed_test = (changed["date"] >= FORMAL_SPLIT.test_start) & changed["is_observed"]
        changed.loc[observed_test, "quantity"] = 500.0

        first = FormalBacktestRunner(Backtester(FORMAL_SPLIT)).run(daily, model_factories())
        second = FormalBacktestRunner(Backtester(FORMAL_SPLIT)).run(changed, model_factories())
        self.assertEqual({result.model_name for result in first}, {"five_period", "tsb", "hurdle", "direct10"})
        for before, after in zip(first, second, strict=True):
            self.assertEqual(before.model_name, after.model_name)
            self.assertEqual(before.status, "completed")
            self.assertEqual(before.forecasts, after.forecasts)
            self.assertEqual(len(before.forecasts), 52)
            self.assertEqual(before.metrics.n_evaluated_days, 51)
            self.assertNotEqual(before.metrics.mae, after.metrics.mae)

    def test_runner_is_sku_isolated_order_independent_and_returns_selector_ready_frame(self) -> None:
        """factory + defensive copy 使 SKU 改动和模型顺序都不会污染无关结果。"""
        daily_a = make_daily("A")
        daily_b = make_daily("B")
        daily_b["quantity"] = 1.0
        daily = pd.concat([daily_a, daily_b], ignore_index=True)
        factories = (lambda: MeanAdapter("first"), lambda: MeanAdapter("second"))
        runner = FormalBacktestRunner(Backtester(FORMAL_SPLIT))
        normal = runner.run(daily, factories)
        reversed_order = runner.run(daily, tuple(reversed(factories)))
        changed = daily.copy()
        changed.loc[(changed["sku"] == "A") & (changed["date"] <= FORMAL_SPLIT.train_end), "quantity"] = 999.0
        changed_results = runner.run(changed, factories)

        def as_map(results):
            return {(result.sku, result.model_name): result for result in results}

        normal_map = as_map(normal)
        reversed_map = as_map(reversed_order)
        changed_map = as_map(changed_results)
        self.assertEqual(normal_map, reversed_map)
        for model_name in ("first", "second"):
            self.assertEqual(normal_map[("B", model_name)], changed_map[("B", model_name)])
            self.assertNotEqual(normal_map[("A", model_name)].forecasts, changed_map[("A", model_name)].forecasts)

        frame = results_to_metrics_frame(normal)
        self.assertEqual(len(frame), 4)
        self.assertEqual(set(frame.columns), {
            "sku", "model_name", "status", "unavailable_reason", "train_start", "train_end", "test_start",
            "test_end", "forecast_days", "evaluated_days", "evaluated_dates", "mae", "mse", "rmse", "wape", "actual_total",
            "prediction_total", "cumulative_bias",
        })
        self.assertTrue((frame["forecast_days"] == 52).all())

    def test_evaluator_rejects_out_of_order_forecasts(self) -> None:
        """即使日期集合正确，预测记录也必须按正式测试日历顺序输出。"""
        dates = pd.date_range("2026-07-01", periods=2, freq="D")
        actuals = pd.DataFrame(
            {
                "sku": "A",
                "date": dates,
                "quantity": [1.0, 2.0],
                "is_observed": [True, True],
                "launch_date": pd.Timestamp("2025-01-01"),
                "observation_reason": pd.NA,
            }
        )
        forecasts = [DailyForecast("A", dates[1], 1.0), DailyForecast("A", dates[0], 1.0)]
        with self.assertRaisesRegex(ValueError, "连续"):
            evaluate_forecasts(actuals, forecasts, "A", dates)

    def test_unavailable_one_model_does_not_abort_other_models(self) -> None:
        """正常不可适用属于单模型结果，不能让同 SKU 的其他候选模型失败。"""
        runner = FormalBacktestRunner(Backtester(FORMAL_SPLIT))
        results = runner.run(make_daily(), (lambda: UnavailableAdapter("unavailable"), lambda: MeanAdapter("mean")))
        by_name = {result.model_name: result for result in results}
        self.assertEqual(by_name["unavailable"].status, "unavailable")
        self.assertEqual(by_name["unavailable"].unavailable_reason, "insufficient_history")
        self.assertEqual(by_name["mean"].status, "completed")


if __name__ == "__main__":
    unittest.main()
