"""FeatureBuilder 的边界、防泄漏和训练-推理一致性测试。"""

import unittest

import pandas as pd

from demand_forecast.features.builder import FeatureBuilder


def make_daily(
    sku: str,
    start: str,
    quantities: list[float | None],
    observed: list[bool] | None = None,
) -> pd.DataFrame:
    """构造满足 DailySeries 契约的小型日序列，便于人工验证预期值。"""
    dates = pd.date_range(start, periods=len(quantities), freq="D")
    observed = observed or [True] * len(quantities)
    reasons = [pd.NA if flag else "source_missing" for flag in observed]
    return pd.DataFrame(
        {
            "sku": sku,
            "date": dates,
            "quantity": pd.Series(quantities, dtype="Float64"),
            "is_observed": observed,
            "launch_date": pd.Timestamp(start),
            "observation_reason": reasons,
        }
    )


def make_history(start: str, quantities: list[float | None], observed: list[bool] | None = None) -> pd.DataFrame:
    """构造单 SKU 的预测历史；实际阶段 occurrence 等于 1[quantity>0]。"""
    observed = observed or [True] * len(quantities)
    occurrence = [float(value > 0) if flag and value is not None else None for value, flag in zip(quantities, observed)]
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=len(quantities), freq="D"),
            "quantity": quantities,
            "occurrence": occurrence,
            "is_observed": observed,
        }
    )


class FeatureBuilderTests(unittest.TestCase):
    """每项测试都对应一个容易静默出错的时间序列不变量。"""

    def setUp(self) -> None:
        self.builder = FeatureBuilder()

    def test_exact_current_window_boundary(self) -> None:
        """目标日只使用前 90 天；R7 必须是最后 7 天 84..90 的均值。"""
        history = make_history("2025-01-01", [float(value) for value in range(1, 91)])
        result = self.builder.build_next(history, "2025-04-01", "five_period")
        self.assertTrue(result.is_available)
        self.assertEqual(result.values["current_mean_7"], 87.0)
        self.assertEqual(result.values["current_mean_90"], 45.5)

    def test_target_and_future_values_cannot_change_features(self) -> None:
        """修改 y[t] 或未来销量后，历史批量特征中 t 日特征必须不变。"""
        daily = make_daily("A", "2025-01-01", [float(value) for value in range(1, 101)])
        target_date = pd.Timestamp("2025-04-01")
        first = self.builder.build_historical(daily, "five_period")
        before = first.features.loc[first.features["date"] == target_date].iloc[0]

        changed = daily.copy()
        changed.loc[changed["date"] >= target_date, "quantity"] = 9999.0
        second = self.builder.build_historical(changed, "five_period")
        after = second.features.loc[second.features["date"] == target_date].iloc[0]
        self.assertEqual(before["current_mean_7"], after["current_mean_7"])
        self.assertEqual(before["current_mean_90"], after["current_mean_90"])

    def test_sku_isolation(self) -> None:
        """SKU A 的高销量绝不能进入 SKU B 的滚动窗口。"""
        daily = pd.concat(
            [
                make_daily("A", "2025-01-01", [100.0] * 91),
                make_daily("B", "2025-01-01", [1.0] * 91),
            ],
            ignore_index=True,
        )
        result = self.builder.build_historical(daily, "five_period")
        row_b = result.features.loc[
            (result.features["sku"] == "B") & (result.features["date"] == pd.Timestamp("2025-04-01"))
        ].iloc[0]
        self.assertEqual(row_b["current_mean_7"], 1.0)
        self.assertEqual(row_b["current_mean_90"], 1.0)

    def test_next_step_rejects_multiple_skus(self) -> None:
        """单步预测历史必须是单 SKU，避免调用方意外拼接不同商品。"""
        history = pd.concat(
            [
                make_history("2025-01-01", [1.0] * 45).assign(sku="A"),
                make_history("2025-02-15", [1.0] * 45).assign(sku="B"),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "一个 SKU"):
            self.builder.build_next(history, "2025-04-01", "five_period")

    def test_insufficient_history_and_direct10_requirement(self) -> None:
        """R90 不足 90 个前置日不可用；Direct10 必须有完整 455 日历史。"""
        short_history = make_history("2025-01-01", [1.0] * 89)
        short = self.builder.build_next(short_history, "2025-03-31", "five_period")
        self.assertEqual(short.unavailable_reason, "current_window_90_insufficient_history")

        direct_short = make_history("2025-01-01", [1.0] * 454)
        direct = self.builder.build_next(direct_short, "2026-03-31", "direct10")
        self.assertEqual(direct.unavailable_reason, "yoy_window_90_insufficient_history")

    def test_yoy_is_exactly_365_elapsed_days_in_leap_year(self) -> None:
        """闰年策略固定为 t-365 天；不使用 DateOffset(years=1)。"""
        target = pd.Timestamp("2025-02-28")
        anchor = target - pd.Timedelta(days=365)
        self.assertEqual(anchor, pd.Timestamp("2024-02-29"))
        dates = pd.date_range(target - pd.Timedelta(days=455), periods=455, freq="D")
        quantities = [0.0] * len(dates)
        yoy_dates = pd.date_range(anchor - pd.Timedelta(days=7), periods=7, freq="D")
        for value, current_date in enumerate(yoy_dates, start=10):
            quantities[dates.get_loc(current_date)] = float(value)
        history = make_history(str(dates[0].date()), quantities)
        result = self.builder.build_next(history, target, "direct10")
        self.assertTrue(result.is_available)
        self.assertEqual(result.values["yoy_mean_7"], 13.0)

    def test_occurrence_and_weekday_features(self) -> None:
        """发生率使用实际 0/非零状态；周一对应 sin=0、cos=1。"""
        quantities = [0.0] * 83 + [0.0, 2.0, 0.0, 4.0, 0.0, 1.0, 0.0]
        history = make_history("2024-10-08", quantities)
        result = self.builder.build_next(history, "2025-01-06", "hurdle")
        self.assertTrue(result.is_available)
        self.assertEqual(result.values["occurrence_rate_7"], 3 / 7)
        self.assertAlmostEqual(result.values["dow_sin"], 0.0, places=12)
        self.assertAlmostEqual(result.values["dow_cos"], 1.0, places=12)

    def test_unobserved_window_is_not_imputed(self) -> None:
        """窗口内任一不可观测日都应使特征不可用，而不是补零或忽略。"""
        observed = [True] * 90
        observed[-3] = False
        quantities: list[float | None] = [1.0] * 90
        quantities[-3] = None
        history = make_history("2025-01-01", quantities, observed)
        result = self.builder.build_next(history, "2025-04-01", "five_period")
        self.assertEqual(result.unavailable_reason, "current_window_7_unobserved_history")

    def test_historical_and_next_step_are_identical(self) -> None:
        """同一目标日的批量特征与单步特征必须同源且数值一致。"""
        daily = make_daily("A", "2025-01-01", [float((value % 5) + 1) for value in range(100)])
        target_date = pd.Timestamp("2025-04-01")
        batch = self.builder.build_historical(daily, "hurdle")
        batch_row = batch.features.loc[batch.features["date"] == target_date].iloc[0]
        prior = daily.loc[daily["date"] < target_date, ["date", "quantity", "is_observed"]].copy()
        prior["occurrence"] = (prior["quantity"] > 0).astype(float)
        single = self.builder.build_next(prior, target_date, "hurdle")
        self.assertTrue(single.is_available)
        for feature_name, value in single.values.items():
            self.assertAlmostEqual(batch_row[feature_name], value, places=12)

    def test_target_unobserved_is_reported(self) -> None:
        """不可观测目标日没有标签，训练批量中必须单独记录而非参与训练。"""
        daily = make_daily("A", "2025-01-01", [1.0] * 91)
        daily.loc[90, "is_observed"] = False
        daily.loc[90, "quantity"] = pd.NA
        daily.loc[90, "observation_reason"] = "source_missing"
        result = self.builder.build_historical(daily, "five_period")
        row = result.unavailable.loc[result.unavailable["date"] == pd.Timestamp("2025-04-01")].iloc[0]
        self.assertEqual(row["reason"], "target_date_unobserved")


if __name__ == "__main__":
    unittest.main()
