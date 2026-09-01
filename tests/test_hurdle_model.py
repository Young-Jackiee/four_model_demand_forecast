"""Hurdle 的数学、双状态递归、无泄漏与持久化测试。"""

import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from demand_forecast.backtesting import BacktestSplit, Backtester, HurdleBacktestAdapter
from demand_forecast.features.builder import FeatureBuilder
from demand_forecast.models.hurdle import (
    FEATURE_VERSION,
    HURDLE_FEATURES,
    INTERNAL_VALIDATION_MODE,
    OBSERVATION_POLICY,
    FeatureStandardizer,
    HurdleConfig,
    HurdleFittedModel,
    HurdleModel,
    HurdleParameters,
    HurdleTrainingUnavailableError,
    LogisticFitDiagnostics,
)


def make_daily(quantities: list[float | None], observed: list[bool] | None = None) -> pd.DataFrame:
    """创建连续、可传入 DailySeries 契约的单 SKU 测试序列。"""
    observed = observed or [True] * len(quantities)
    return pd.DataFrame(
        {
            "sku": "A",
            "date": pd.date_range("2025-01-01", periods=len(quantities), freq="D"),
            "quantity": pd.Series(quantities, dtype="Float64"),
            "is_observed": observed,
            "launch_date": pd.Timestamp("2025-01-01"),
            "observation_reason": [pd.NA if value else "source_missing" for value in observed],
        }
    )


def alternating_daily(length: int) -> pd.DataFrame:
    """产生同时含零和正销量的可训练序列。"""
    quantities = [float((index % 5) + 2) if index % 3 else 0.0 for index in range(length)]
    return make_daily(quantities)


class HurdleModelTests(unittest.TestCase):
    """所有测试聚焦于最容易产生 silent bug 的 Hurdle 不变量。"""

    def setUp(self) -> None:
        self.config = HurdleConfig(lambda_values=(0.1, 1.0), validation_days=3, max_iter=100, tol=1e-8)
        self.model = HurdleModel(self.config)

    def test_fit_predict_and_serialization_round_trip(self) -> None:
        """最终预测必须等于 p×q，且加载后 p、q、ŷ 完全一致。"""
        daily = alternating_daily(105)
        fitted = self.model.fit(daily, "2025-04-15")
        history = self.model._to_forecast_history(daily, fitted)
        features = FeatureBuilder().build_next(history, "2025-04-16", "hurdle").values
        prediction = self.model.predict_one(fitted, features or {})
        restored = self.model.deserialize(self.model.serialize(fitted))
        restored_prediction = self.model.predict_one(restored, features or {})
        self.assertGreaterEqual(prediction.occurrence_probability, 0.0)
        self.assertLessEqual(prediction.occurrence_probability, 1.0)
        self.assertGreaterEqual(prediction.conditional_quantity, 0.0)
        self.assertAlmostEqual(prediction.prediction, prediction.occurrence_probability * prediction.conditional_quantity)
        self.assertEqual(prediction, restored_prediction)
        self.assertEqual(fitted.feature_names, HURDLE_FEATURES)
        self.assertEqual(fitted.internal_validation_mode, INTERNAL_VALIDATION_MODE)

    def test_logistic_and_quantity_match_specification_math(self) -> None:
        """发生模型截距不惩罚；数量模型的目标是 log1p 而非原始销量。"""
        values = np.zeros((3, 12), dtype=float)
        beta0, beta, diagnostics = self.model._fit_logistic(values, np.asarray([0.0, 0.0, 1.0]), lambda_p=50.0)
        self.assertTrue(diagnostics.converged)
        self.assertAlmostEqual(beta0, math.log(0.5), places=6)
        np.testing.assert_allclose(beta, 0.0, atol=1e-10)
        gamma0, gamma, _ = self.model._fit_quantity(
            np.zeros((2, 12), dtype=float), np.log1p(np.asarray([1.0, 3.0])), lambda_q=10.0
        )
        self.assertAlmostEqual(gamma0, float(np.mean(np.log1p([1.0, 3.0]))), places=12)
        np.testing.assert_allclose(gamma, 0.0, atol=1e-12)

    def test_single_class_occurrence_is_explicitly_unavailable(self) -> None:
        """未惩罚截距下单类别 logistic 没有有限最优解，不能伪造 Hurdle。"""
        with self.assertRaises(HurdleTrainingUnavailableError) as zero_error:
            self.model.fit(make_daily([0.0] * 105), "2025-04-15")
        self.assertEqual(zero_error.exception.reason, "single_class_occurrence_internal_training")
        with self.assertRaises(HurdleTrainingUnavailableError) as positive_error:
            self.model.fit(make_daily([2.0] * 105), "2025-04-15")
        self.assertEqual(positive_error.exception.reason, "single_class_occurrence_internal_training")

    def test_fit_rejects_any_data_after_trained_through(self) -> None:
        """Hurdle.fit 不静默过滤未来行，防止调用方误把 formal test 传进模型。"""
        with self.assertRaisesRegex(ValueError, "之后的数据"):
            self.model.fit(alternating_daily(105), "2025-04-14")

    def test_recursive_occurrence_uses_probability_not_binary_prediction(self) -> None:
        """第二日 F 窗口应纳入首日 p；用 1[ŷ>0] 会得到不同的 p2。"""
        fitted = self._manual_fitted()
        daily = alternating_daily(100)
        forecast_dates = pd.date_range("2025-04-11", periods=2, freq="D")
        steps = self.model.forecast_many(fitted, daily, forecast_dates)
        history = self.model._to_forecast_history(daily, fitted)
        first_features = self.model.feature_builder.build_next(history, forecast_dates[0], "hurdle").values or {}
        first = self.model.predict_one(fitted, first_features)
        history.loc[len(history)] = {"date": forecast_dates[0], "quantity": first.prediction, "occurrence": first.occurrence_probability}
        probability_features = self.model.feature_builder.build_next(history, forecast_dates[1], "hurdle").values or {}
        expected_second = self.model.predict_one(fitted, probability_features)
        binary_history = self.model._to_forecast_history(daily, fitted)
        binary_history.loc[len(binary_history)] = {"date": forecast_dates[0], "quantity": first.prediction, "occurrence": float(first.prediction > 0.0)}
        binary_features = self.model.feature_builder.build_next(binary_history, forecast_dates[1], "hurdle").values or {}
        binary_second = self.model.predict_one(fitted, binary_features)
        self.assertAlmostEqual(steps[1].result.occurrence_probability, expected_second.occurrence_probability)
        self.assertNotAlmostEqual(steps[1].result.occurrence_probability, binary_second.occurrence_probability)

    def test_unobserved_history_is_not_imputed(self) -> None:
        """训练末期窗口含不可观测日时，Hurdle 不得把它解释为零销量。"""
        daily = alternating_daily(105)
        daily.loc[99, "is_observed"] = False
        daily.loc[99, "quantity"] = pd.NA
        daily.loc[99, "observation_reason"] = "stockout"
        with self.assertRaises(HurdleTrainingUnavailableError) as error:
            self.model.fit(daily, "2025-04-15")
        self.assertIn("hurdle", error.exception.reason)

    def test_backtester_hurdle_forecast_ignores_formal_test_actuals(self) -> None:
        """修改正式测试标签不能改变 fitted metadata 或完整递归 forecast。"""
        split = BacktestSplit("2025-01-01", "2025-04-15", "2025-04-16", "2025-04-20", expected_test_days=5)
        daily = alternating_daily(110)
        changed = daily.copy()
        changed.loc[changed["date"] >= split.test_start, "quantity"] = [1000.0] * 5
        adapter = HurdleBacktestAdapter(HurdleModel(self.config))
        first = Backtester(split).backtest_one_sku(daily, adapter)
        second = Backtester(split).backtest_one_sku(changed, HurdleBacktestAdapter(HurdleModel(self.config)))
        self.assertEqual(first.status, "completed")
        self.assertEqual(first.forecasts, second.forecasts)
        self.assertEqual(first.fitted_metadata, second.fitted_metadata)
        self.assertNotEqual(first.metrics.mae, second.metrics.mae)
        self.assertEqual(len(first.forecasts), 5)
        self.assertIn("p", first.forecasts[0].components or {})
        self.assertIn("q", first.forecasts[0].components or {})

    def test_lambda_grid_is_jointly_evaluated_as_all_pairs(self) -> None:
        """四个 lambda 值必须形成 4×4 组合，而不是分别挑选两个单模型最优值。"""
        model = HurdleModel(HurdleConfig(validation_days=3, max_iter=100, tol=1e-8))
        daily = alternating_daily(105)
        internal_train = daily.iloc[:-3].copy()
        internal_validation = daily.iloc[-3:].copy()
        features = model._build_training_features(internal_train)
        scaler = model._fit_standardizer(features)
        with patch.object(model, "_fit_parameters", wraps=model._fit_parameters) as fit_parameters:
            model._select_lambdas("A", features, scaler, internal_train, internal_validation)
        self.assertEqual(fit_parameters.call_count, 16)

    def _manual_fitted(self) -> HurdleFittedModel:
        """构造受控参数，使 p 对 occurrence_rate_7 敏感，检验递归状态来源。"""
        beta = np.zeros(12)
        beta[5] = 2.0  # occurrence_rate_7 的固定位置。
        parameters = HurdleParameters(
            standardizer=FeatureStandardizer(mean=(0.0,) * 12, scale=(1.0,) * 12),
            beta0=-2.0,
            beta=tuple(beta),
            gamma0=math.log1p(4.0),
            gamma=(0.0,) * 12,
            occurrence_diagnostics=LogisticFitDiagnostics(iterations=1, converged=True, final_objective=0.0),
            quantity_objective=0.0,
        )
        return HurdleFittedModel(
            sku="A",
            lambda_p=0.1,
            lambda_q=0.1,
            feature_names=HURDLE_FEATURES,
            feature_version=FEATURE_VERSION,
            parameters=parameters,
            trained_through="2025-04-10",
            config=self.config,
            validation_mae=0.0,
            validation_observed_days=1,
            n_full_training_rows=1,
            n_positive_training_rows=1,
            observation_policy=OBSERVATION_POLICY,
        )


if __name__ == "__main__":
    unittest.main()
