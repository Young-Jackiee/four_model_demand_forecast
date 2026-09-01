"""五周期基准模型的优化、预测和序列化测试。"""

import unittest

import numpy as np
import pandas as pd

from demand_forecast.features.builder import FeatureBuilder
from demand_forecast.features.definitions import CURRENT_MEAN_FEATURES, FEATURE_VERSION
from demand_forecast.models.five_period import (
    FivePeriodConfig,
    FivePeriodFittedModel,
    FivePeriodModel,
    project_to_simplex,
)


def make_training_frame(x_values: np.ndarray, y_values: np.ndarray) -> pd.DataFrame:
    """把数值矩阵包装为 FeatureBuilder 输出兼容的训练表。"""
    frame = pd.DataFrame(x_values, columns=CURRENT_MEAN_FEATURES)
    frame.insert(0, "target_quantity", y_values)
    return frame


def make_fitted(weights: tuple[float, ...]) -> FivePeriodFittedModel:
    """构造固定权重模型，用于测试按特征名预测。"""
    return FivePeriodFittedModel(
        weights=weights,
        feature_names=CURRENT_MEAN_FEATURES,
        feature_version=FEATURE_VERSION,
        trained_through="2026-06-30",
        config=FivePeriodConfig(),
        effective_step_size=0.01,
        n_training_rows=1,
        iterations=1,
        converged=True,
        final_objective=0.0,
    )


class FivePeriodModelTests(unittest.TestCase):
    """验证单纯形约束、确定性和模型契约。"""

    def setUp(self) -> None:
        self.model = FivePeriodModel()

    def test_equal_weight_prediction(self) -> None:
        """均匀权重作用于 1..5 时，预测必须等于 3。"""
        fitted = make_fitted((0.2, 0.2, 0.2, 0.2, 0.2))
        row = dict(zip(CURRENT_MEAN_FEATURES, [1.0, 2.0, 3.0, 4.0, 5.0], strict=True))
        self.assertEqual(self.model.predict_one(fitted, row), 3.0)

    def test_projection_has_expected_result_and_constraints(self) -> None:
        """投影不仅满足约束，还应得到标准算法的确定结果。"""
        projected = project_to_simplex(np.array([-1.0, -1.0, 0.9, 0.5, -1.0]))
        np.testing.assert_allclose(projected, [0.0, 0.0, 0.7, 0.3, 0.0], atol=1e-12)
        self.assertTrue((projected >= 0).all())
        self.assertAlmostEqual(projected.sum(), 1.0, places=12)

    def test_fit_keeps_weights_on_simplex_and_is_deterministic(self) -> None:
        """相同输入重复拟合应得到相同的非负、和为 1 的权重。"""
        rng = np.random.default_rng(7)
        x_values = rng.uniform(0.0, 10.0, size=(80, 5))
        target = 0.55 * x_values[:, 0] + 0.45 * x_values[:, 4]
        training = make_training_frame(x_values, target)
        first = self.model.fit(training, "2026-06-30")
        second = self.model.fit(training, "2026-06-30")
        np.testing.assert_allclose(first.weights, second.weights, atol=1e-12)
        self.assertTrue(np.all(np.asarray(first.weights) >= 0))
        self.assertAlmostEqual(sum(first.weights), 1.0, places=12)

    def test_constant_series_predicts_the_constant(self) -> None:
        """所有窗口和目标均为常数时，任何可行权重都应预测该常数。"""
        training = make_training_frame(np.full((20, 5), 6.0), np.full(20, 6.0))
        fitted = self.model.fit(training, "2026-06-30")
        prediction = self.model.predict_one(fitted, dict(zip(CURRENT_MEAN_FEATURES, [6.0] * 5, strict=True)))
        self.assertAlmostEqual(prediction, 6.0, places=12)

    def test_prediction_uses_feature_names_not_mapping_order(self) -> None:
        """字典插入顺序改变时，按名称预测的结果必须保持不变。"""
        fitted = make_fitted((1.0, 0.0, 0.0, 0.0, 0.0))
        reversed_row = {name: value for name, value in reversed(list(zip(CURRENT_MEAN_FEATURES, [8, 7, 6, 5, 4], strict=True)))}
        self.assertEqual(self.model.predict_one(fitted, reversed_row), 8.0)

    def test_missing_feature_and_invalid_training_data_are_rejected(self) -> None:
        """缺列、NaN 或负数不能被静默送入优化器。"""
        fitted = make_fitted((0.2, 0.2, 0.2, 0.2, 0.2))
        with self.assertRaisesRegex(ValueError, "缺少字段"):
            self.model.predict_one(fitted, {CURRENT_MEAN_FEATURES[0]: 1.0})

        invalid = make_training_frame(np.ones((2, 5)), np.array([1.0, np.nan]))
        with self.assertRaisesRegex(ValueError, "必须有限"):
            self.model.fit(invalid, "2026-06-30")

    def test_serialization_round_trip_preserves_prediction(self) -> None:
        """JSON 友好字典加载后，对相同特征的预测必须一致。"""
        x_values = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]], dtype=float)
        training = make_training_frame(x_values, np.array([2.0, 3.0, 4.0]))
        fitted = self.model.fit(training, "2026-06-30")
        payload = self.model.serialize(fitted)
        restored = self.model.deserialize(payload)
        row = dict(zip(CURRENT_MEAN_FEATURES, [4.0, 5.0, 6.0, 7.0, 8.0], strict=True))
        self.assertAlmostEqual(self.model.predict_one(fitted, row), self.model.predict_one(restored, row), places=12)

    def test_short_optimization_run_reports_not_converged(self) -> None:
        """迭代上限太小时必须记录未收敛，而不是伪造成功。"""
        model = FivePeriodModel(FivePeriodConfig(max_iter=1, tol=1e-15))
        x_values = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=float)
        fitted = model.fit(make_training_frame(x_values, np.array([1.0, 0.0])), "2026-06-30")
        self.assertFalse(fitted.converged)
        self.assertEqual(fitted.iterations, 1)

    def test_future_training_rows_are_rejected(self) -> None:
        """核心模型也必须拒绝 trained_through 之后的特征，不能只依赖外层回测器。"""
        training = make_training_frame(np.ones((2, 5)), np.array([1.0, 2.0]))
        training["date"] = [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-07-01")]
        with self.assertRaisesRegex(ValueError, "之后的日期"):
            self.model.fit(training, "2026-06-30")

    def test_trains_on_feature_builder_output(self) -> None:
        """模型应能直接消费 FeatureBuilder 的五周期训练表，无需重算滚动窗口。"""
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        daily = pd.DataFrame(
            {
                "sku": "A",
                "date": dates,
                "quantity": [float((index % 4) + 2) for index in range(100)],
                "is_observed": True,
                "launch_date": pd.Timestamp("2025-01-01"),
                "observation_reason": pd.NA,
            }
        )
        features = FeatureBuilder().build_historical(daily, "five_period").features
        fitted = self.model.fit(features, "2025-04-10")
        self.assertEqual(fitted.n_training_rows, len(features))
        self.assertTrue(np.isclose(sum(fitted.weights), 1.0))


if __name__ == "__main__":
    unittest.main()
