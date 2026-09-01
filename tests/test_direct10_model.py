"""Direct10 的十特征契约、约束优化、递归与无泄漏测试。"""

import unittest

import numpy as np
import pandas as pd

from demand_forecast.backtesting import BacktestSplit, Backtester, Direct10BacktestAdapter
from demand_forecast.features.definitions import CURRENT_MEAN_FEATURES, FEATURE_VERSION, YOY_MEAN_FEATURES
from demand_forecast.models.direct10 import (
    DIRECT10_FEATURES,
    Direct10Config,
    Direct10FittedModel,
    Direct10Model,
    Direct10TrainingUnavailableError,
)


def make_training(values: np.ndarray, target: np.ndarray, end: str = "2026-06-30") -> pd.DataFrame:
    """构造与 FeatureBuilder 输出同字段的训练表。"""
    frame = pd.DataFrame(values, columns=DIRECT10_FEATURES)
    frame.insert(0, "target_quantity", target)
    frame.insert(0, "date", pd.date_range(end=end, periods=len(frame), freq="D"))
    frame.insert(0, "sku", "A")
    return frame


def make_daily(length: int, start: str = "2025-01-01") -> pd.DataFrame:
    """构造连续、全可观测的单 SKU 日序列。"""
    return pd.DataFrame(
        {
            "sku": "A",
            "date": pd.date_range(start, periods=length, freq="D"),
            "quantity": [float((index % 9) + 1) for index in range(length)],
            "is_observed": True,
            "launch_date": pd.Timestamp(start),
            "observation_reason": pd.NA,
        }
    )


def make_fitted(weights: tuple[float, ...]) -> Direct10FittedModel:
    """构造可控模型，以验证特征顺序和递归来源。"""
    return Direct10FittedModel(
        weights=weights,
        feature_names=DIRECT10_FEATURES,
        feature_version=FEATURE_VERSION,
        trained_through="2026-03-31",
        config=Direct10Config(),
        effective_step_size=0.1,
        n_training_rows=1,
        iterations=1,
        converged=True,
        final_objective=0.0,
    )


class FixedDirect10Model:
    """固定 R7 权重，专门观察适配器是否真正递归追加预测。"""

    name = "direct10"

    def __init__(self, weights: tuple[float, ...] | None = None) -> None:
        self.weights = weights or ((1.0,) + (0.0,) * 9)

    def fit(self, training_features: pd.DataFrame, trained_through: str) -> Direct10FittedModel:
        return Direct10FittedModel(
            weights=self.weights,
            feature_names=DIRECT10_FEATURES,
            feature_version=FEATURE_VERSION,
            trained_through=trained_through,
            config=Direct10Config(),
            effective_step_size=1.0,
            n_training_rows=len(training_features),
            iterations=1,
            converged=True,
            final_objective=0.0,
        )

    def predict_one(self, fitted: Direct10FittedModel, feature_row: dict[str, float]) -> float:
        return float(sum(weight * feature_row[name] for weight, name in zip(fitted.weights, DIRECT10_FEATURES, strict=True)))

    def serialize(self, fitted: Direct10FittedModel) -> dict[str, object]:
        return {"model_name": self.name, "weights": list(fitted.weights)}


class Direct10ModelTests(unittest.TestCase):
    """测试所有 Direct10 特有的数学与时间序列不变量。"""

    def setUp(self) -> None:
        self.model = Direct10Model(Direct10Config(max_iter=20_000, tol=1e-9))

    def test_feature_contract_and_equal_weight_prediction(self) -> None:
        """固定十特征顺序，均匀权重作用于 1..10 必须等于 5.5。"""
        self.assertEqual(DIRECT10_FEATURES, CURRENT_MEAN_FEATURES + YOY_MEAN_FEATURES)
        fitted = make_fitted((0.1,) * 10)
        row = dict(zip(DIRECT10_FEATURES, range(1, 11), strict=True))
        self.assertAlmostEqual(self.model.predict_one(fitted, row), 5.5, places=12)

    def test_fit_is_deterministic_nonnegative_and_on_simplex(self) -> None:
        """同一训练数据应得到相同的十个非负权重，且总和严格为一。"""
        rng = np.random.default_rng(42)
        values = rng.uniform(0.0, 10.0, size=(60, 10))
        target = 0.7 * values[:, 0] + 0.3 * values[:, 8]
        first = self.model.fit(make_training(values, target), "2026-06-30")
        second = self.model.fit(make_training(values, target), "2026-06-30")
        np.testing.assert_allclose(first.weights, second.weights, atol=1e-12)
        self.assertEqual(len(first.weights), 10)
        self.assertTrue((np.asarray(first.weights) >= 0.0).all())
        self.assertAlmostEqual(sum(first.weights), 1.0, places=12)

    def test_regularization_center_is_exactly_point_one(self) -> None:
        """目标函数的 ridge 中心必须是 10 维 0.1，而不是 FivePeriod 的 0.2。"""
        values = np.zeros((3, 10))
        target = np.zeros(3)
        weights = np.zeros(10)
        objective = self.model._objective(values, target, weights, np.full(10, 0.1))
        self.assertAlmostEqual(objective, self.model.config.ridge * 0.1, places=15)

    def test_constant_series_and_serialization_round_trip(self) -> None:
        """常数特征应给出常数预测，保存加载后预测不能改变。"""
        training = make_training(np.full((12, 10), 6.0), np.full(12, 6.0))
        fitted = self.model.fit(training, "2026-06-30")
        row = dict(zip(DIRECT10_FEATURES, [6.0] * 10, strict=True))
        restored = self.model.deserialize(self.model.serialize(fitted))
        self.assertAlmostEqual(self.model.predict_one(fitted, row), 6.0, places=12)
        self.assertAlmostEqual(self.model.predict_one(fitted, row), self.model.predict_one(restored, row), places=12)

    def test_feature_lookup_is_name_based_and_invalid_models_fail_fast(self) -> None:
        """输入映射重排安全；缺特征和错误持久化顺序必须立即报错。"""
        fitted = make_fitted((1.0,) + (0.0,) * 9)
        reversed_row = {name: value for name, value in reversed(list(zip(DIRECT10_FEATURES, range(1, 11), strict=True)))}
        self.assertEqual(self.model.predict_one(fitted, reversed_row), 1.0)
        with self.assertRaisesRegex(ValueError, "缺少字段"):
            self.model.predict_one(fitted, {DIRECT10_FEATURES[0]: 1.0})
        broken = Direct10FittedModel(**{**fitted.__dict__, "feature_names": tuple(reversed(DIRECT10_FEATURES))})
        with self.assertRaisesRegex(ValueError, "顺序"):
            self.model.predict_one(broken, reversed_row)

    def test_empty_or_future_training_rows_are_rejected(self) -> None:
        """模型不应静默训练不足样本，也不应接收 trained_through 后的特征行。"""
        empty = pd.DataFrame(columns=["target_quantity", *DIRECT10_FEATURES])
        with self.assertRaises(Direct10TrainingUnavailableError):
            self.model.fit(empty, "2026-06-30")
        future = make_training(np.ones((1, 10)), np.ones(1), end="2026-07-01")
        with self.assertRaisesRegex(ValueError, "之后的日期"):
            self.model.fit(future, "2026-06-30")

    def test_adapter_requires_455_day_effective_history(self) -> None:
        """454 个前置自然日不能产生 Direct10 训练行，455 天边界可以。"""
        short = make_daily(456)
        split_short = BacktestSplit("2025-01-01", "2026-03-31", "2026-04-01", "2026-04-01", expected_test_days=1)
        unavailable = Backtester(split_short).backtest_one_sku(short, Direct10BacktestAdapter())
        self.assertEqual(unavailable.status, "unavailable")
        self.assertEqual(unavailable.unavailable_reason, "insufficient_direct10_history")

        enough = make_daily(457)
        split_enough = BacktestSplit("2025-01-01", "2026-04-01", "2026-04-02", "2026-04-02", expected_test_days=1)
        completed = Backtester(split_enough).backtest_one_sku(enough, Direct10BacktestAdapter())
        self.assertEqual(completed.status, "completed")

    def test_recursive_current_window_changes_but_yoy_uses_calendar_history(self) -> None:
        """第 2 天 current R7 纳入第 1 天预测；YOY R7 仍读取 365 天前的真实日历。"""
        daily = make_daily(458)
        daily["quantity"] = 0.0
        daily.loc[455, "quantity"] = 7.0
        # 目标 2026-04-02 的 YOY R7（2025-03-26..04-01）设为 9，方便核对锚点来源。
        daily.loc[84:91, "quantity"] = 9.0
        split = BacktestSplit("2025-01-01", "2026-04-01", "2026-04-02", "2026-04-03", expected_test_days=2)
        result = Backtester(split).backtest_one_sku(daily, Direct10BacktestAdapter(FixedDirect10Model()))
        self.assertEqual(result.status, "completed")
        self.assertAlmostEqual(result.forecasts[0].prediction, 1.0)
        self.assertAlmostEqual(result.forecasts[1].prediction, 8.0 / 7.0)

        yoy_model = FixedDirect10Model((0.0,) * 5 + (1.0,) + (0.0,) * 4)
        yoy_fitted = yoy_model.fit(pd.DataFrame({"target_quantity": [0.0], **{name: [0.0] for name in DIRECT10_FEATURES}}), "2026-04-01")
        adapter = Direct10BacktestAdapter(yoy_model)
        forecasts = adapter.forecast(yoy_fitted, daily.iloc[:456], pd.date_range("2026-04-02", periods=2, freq="D"))
        self.assertAlmostEqual(forecasts[0].prediction, 9.0)
        self.assertAlmostEqual(forecasts[1].prediction, 9.0)

    def test_formal_target_mutation_cannot_change_forecast_but_yoy_history_can(self) -> None:
        """测试标签不是特征源；而真正的去年窗口改动必须能影响 Direct10 预测。"""
        daily = make_daily(598)
        split = BacktestSplit("2025-01-01", "2026-06-30", "2026-07-01", "2026-08-21", expected_test_days=52)
        adapter = Direct10BacktestAdapter(FixedDirect10Model())
        first = Backtester(split).backtest_one_sku(daily, adapter)
        changed_test = daily.copy()
        changed_test.loc[changed_test["date"] >= split.test_start, "quantity"] = 9999.0
        second = Backtester(split).backtest_one_sku(changed_test, Direct10BacktestAdapter(FixedDirect10Model()))
        self.assertEqual(first.forecasts, second.forecasts)
        self.assertNotEqual(first.metrics.mae, second.metrics.mae)

        # 2026-07-01 的 YOY R7 使用 2025-06-24..30；这是真正允许影响预测的训练历史。
        changed_yoy = daily.copy()
        changed_yoy.loc[changed_yoy["date"].between("2025-06-24", "2025-06-30"), "quantity"] = 100.0
        yoy_only = (0.0,) * 5 + (1.0,) + (0.0,) * 4
        yoy_adapter = Direct10BacktestAdapter(FixedDirect10Model(yoy_only))
        first_yoy = Backtester(split).backtest_one_sku(daily, yoy_adapter)
        third = Backtester(split).backtest_one_sku(changed_yoy, Direct10BacktestAdapter(FixedDirect10Model(yoy_only)))
        self.assertNotEqual(first_yoy.forecasts[0].prediction, third.forecasts[0].prediction)

    def test_unobserved_history_is_never_imputed_and_no_residual_stage_exists(self) -> None:
        """不可观测日使训练行失效；模型输出仅来自一次十权重点积。"""
        daily = make_daily(456)
        daily.loc[454, "is_observed"] = False
        daily.loc[454, "quantity"] = pd.NA
        split = BacktestSplit("2025-01-01", "2026-03-31", "2026-04-01", "2026-04-01", expected_test_days=1)
        result = Backtester(split).backtest_one_sku(daily, Direct10BacktestAdapter())
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.unavailable_reason, "no_available_direct10_training_features")
        self.assertFalse(hasattr(self.model, "residual_model"))


if __name__ == "__main__":
    unittest.main()
