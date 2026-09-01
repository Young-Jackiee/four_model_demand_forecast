"""TSB 间歇性需求模型的状态、调参与数据边界测试。"""

import unittest

import numpy as np
import pandas as pd

from demand_forecast.models.tsb import (
    TSBConfig,
    TSBModel,
    TSBState,
    TSBTrainingUnavailableError,
)


def make_daily(quantities: list[float | None], observed: list[bool] | None = None) -> pd.DataFrame:
    """构造符合标准日序列契约的单 SKU 测试数据。"""
    if observed is None:
        observed = [True] * len(quantities)
    return pd.DataFrame(
        {
            "sku": "SKU_TEST",
            "date": pd.date_range("2025-01-01", periods=len(quantities), freq="D"),
            "quantity": quantities,
            "is_observed": observed,
            "launch_date": pd.Timestamp("2025-01-01"),
            "observation_reason": pd.NA,
        }
    )


class TSBModelTests(unittest.TestCase):
    """验证 TSB 公式、冻结预测和无泄漏的内部验证。"""

    def test_update_state_follows_tsb_formula(self) -> None:
        """发生率每天更新，需求规模仅在发生需求时更新。"""
        model = TSBModel()
        initial = TSBState(0.2, 10.0)
        positive = model.update_state(initial, quantity=20.0, alpha=0.5, beta=0.1)
        self.assertAlmostEqual(positive.occurrence_probability, 0.28)
        self.assertAlmostEqual(positive.demand_size, 15.0)
        zero = model.update_state(positive, quantity=0.0, alpha=0.5, beta=0.1)
        self.assertAlmostEqual(zero.occurrence_probability, 0.252)
        self.assertAlmostEqual(zero.demand_size, 15.0)
        self.assertEqual(initial, TSBState(0.2, 10.0))

    def test_future_forecast_is_constant_and_does_not_mutate_state(self) -> None:
        """正式测试期没有真实销量更新时，多日预测必须保持同一数值。"""
        model = TSBModel(TSBConfig(smoothing_values=(0.1,), validation_days=2, initialization_observed_days=2))
        fitted = model.fit(make_daily([2.0, 0.0, 2.0, 0.0, 4.0]), "2025-01-05")
        before = fitted.state
        forecasts = model.forecast_many(fitted, 4)
        self.assertEqual(forecasts, [model.predict_one(fitted)] * 4)
        self.assertEqual(fitted.state, before)

    def test_internal_validation_predicts_before_updating_actual(self) -> None:
        """验证首日的误差必须来自核心历史状态，而不是当天真实销量。"""
        model = TSBModel(TSBConfig(smoothing_values=(1.0,), validation_days=2, initialization_observed_days=1))
        # 核心日为 0；验证日为 10、0。若先更新再预测，首日误差会被错误降低。
        fitted = model.fit(make_daily([0.0, 0.0, 0.0, 10.0, 0.0]), "2025-01-05")
        self.assertAlmostEqual(fitted.validation_mae, 10.0)

    def test_unobserved_date_is_not_used_as_zero_or_validation_label(self) -> None:
        """不可观测日期既不更新状态，也不参与验证 MAE。"""
        model = TSBModel(TSBConfig(smoothing_values=(1.0,), validation_days=2, initialization_observed_days=1))
        daily = make_daily([4.0, 0.0, 4.0, None, 0.0], [True, True, True, False, True])
        fitted = model.fit(daily, "2025-01-05")
        self.assertEqual(fitted.validation_observed_days, 1)
        self.assertAlmostEqual(fitted.validation_mae, 4.0)

    def test_grid_search_tie_is_deterministic(self) -> None:
        """所有组合平局时，按配置声明顺序保留第一组参数。"""
        config = TSBConfig(smoothing_values=(0.30, 0.10), validation_days=2, initialization_observed_days=1)
        fitted = TSBModel(config).fit(make_daily([0.0, 0.0, 0.0, 0.0]), "2025-01-04")
        self.assertEqual((fitted.alpha, fitted.beta), (0.30, 0.30))
        self.assertEqual(fitted.state, TSBState(0.0, 0.0))

    def test_all_zero_history_is_a_valid_zero_model(self) -> None:
        """全零需求不是异常，TSB 应得到零状态和零预测。"""
        fitted = TSBModel(TSBConfig(smoothing_values=(0.1,), validation_days=2, initialization_observed_days=2)).fit(
            make_daily([0.0] * 6), "2025-01-06"
        )
        self.assertEqual(fitted.state, TSBState(0.0, 0.0))
        self.assertEqual(TSBModel().predict_one(fitted), 0.0)

    def test_insufficient_core_history_has_machine_readable_reason(self) -> None:
        """只有验证窗口、没有此前状态历史时，必须明确跳过 TSB。"""
        model = TSBModel(TSBConfig(smoothing_values=(0.1,), validation_days=3, initialization_observed_days=1))
        with self.assertRaises(TSBTrainingUnavailableError) as error:
            model.fit(make_daily([1.0, 0.0, 2.0]), "2025-01-03")
        self.assertEqual(error.exception.reason, "insufficient_history_for_tsb_validation")

    def test_multi_sku_input_is_rejected(self) -> None:
        """每个 SKU 必须独立建模，不能把多个序列静默拼接。"""
        daily = make_daily([1.0] * 5)
        daily.loc[0, "sku"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "一个 SKU"):
            TSBModel().fit(daily, "2025-01-05")

    def test_fit_rejects_rows_after_trained_through(self) -> None:
        """TSB 不再静默截断 future 行，和其他模型一样在调用边界 fail-fast。"""
        model = TSBModel(TSBConfig(smoothing_values=(0.1,), validation_days=2, initialization_observed_days=1))
        with self.assertRaisesRegex(ValueError, "之后的数据"):
            model.fit(make_daily([1.0, 0.0, 2.0]), "2025-01-02")

    def test_serialization_round_trip_preserves_forecast(self) -> None:
        """加载后的状态、参数与未来预测必须完全一致。"""
        model = TSBModel(TSBConfig(smoothing_values=(0.1, 0.2), validation_days=2, initialization_observed_days=2))
        fitted = model.fit(make_daily([2.0, 0.0, 3.0, 0.0, 5.0]), "2025-01-05")
        restored = model.deserialize(model.serialize(fitted))
        self.assertEqual(restored, fitted)
        self.assertAlmostEqual(model.predict_one(restored), model.predict_one(fitted))

    def test_forecast_horizon_must_be_nonnegative_integer(self) -> None:
        """预测天数不能让 bool、负数或小数悄悄通过。"""
        model = TSBModel(TSBConfig(smoothing_values=(0.1,), validation_days=2, initialization_observed_days=1))
        fitted = model.fit(make_daily([1.0, 0.0, 1.0, 0.0]), "2025-01-04")
        for invalid in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                model.forecast_many(fitted, invalid)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
