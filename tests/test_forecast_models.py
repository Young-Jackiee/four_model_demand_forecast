"""文档统一 ForecastModel 接口、序列化和服务加载测试。"""

from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from demand_forecast.forecast_models import (
    Direct10ForecastModel,
    FivePeriodForecastModel,
    HurdleForecastModel,
    TSBForecastModel,
)
from demand_forecast.forecast_service import ForecastService
from demand_forecast.production import ProductionCandidate, ProductionPublisher


def make_daily(sku: str, periods: int) -> pd.DataFrame:
    """构造连续且有零销量的标准日序列。"""
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    quantities = [0.0 if index % 5 == 0 else float((index % 7) + 1) for index in range(periods)]
    return pd.DataFrame(
        {
            "sku": sku,
            "date": dates,
            "quantity": quantities,
            "is_observed": True,
            "launch_date": pd.Timestamp("2025-01-01"),
            "observation_reason": pd.NA,
        }
    )


class ForecastModelTests(unittest.TestCase):
    """锁定文档 API 的无泄漏、递归和加载行为。"""

    def test_fit_uses_explicit_cutoff_and_predict_needs_no_actual_series(self) -> None:
        """cutoff 后销量改动不能影响 fitted artifact 或未来预测。"""
        daily = make_daily("A", 120)
        cutoff = pd.Timestamp("2025-04-30")
        changed = daily.copy()
        changed.loc[changed["date"] > cutoff, "quantity"] = 9999.0
        model = FivePeriodForecastModel()
        first = model.fit(daily, cutoff)
        second = model.fit(changed, cutoff)
        self.assertEqual(model.serialize(first), model.serialize(second))
        self.assertEqual(model.predict(first, 3), model.predict(second, 3))
        self.assertEqual(model.predict(model.deserialize(model.serialize(first)), 3), model.predict(first, 3))
        self.assertEqual(
            [item["date"] for item in model.predict(first, 3)],
            [item.date() for item in pd.date_range("2025-05-01", periods=3, freq="D")],
        )

    def test_tsb_round_trip_and_forecast_service(self) -> None:
        """保存、发布、重新加载后，服务预测必须与训练期模型完全一致。"""
        daily = make_daily("A", 150)
        cutoff = pd.Timestamp("2025-05-30")
        model = TSBForecastModel()
        fitted = model.fit(daily, cutoff)
        expected = tuple(model.predict(fitted, 3))
        artifact = model.serialize(fitted)
        restored = model.deserialize(artifact)
        self.assertEqual(model.predict(restored, 3), list(expected))

        with tempfile.TemporaryDirectory() as directory:
            candidate = ProductionCandidate(
                artifact_id="A__tsb__20250530__fixture",
                sku="A",
                selected_model="tsb",
                production_train_end=cutoff,
                generated_at=pd.Timestamp("2025-05-30T12:00:00Z"),
                model_artifact=artifact,
                forecasts=expected,
                selection_evidence={},
                data_fingerprint="fixture",
            )
            ProductionPublisher(directory).publish(candidate)
            self.assertEqual(ForecastService(directory).predict_active("A", 3), list(expected))

    def test_direct10_and_hurdle_public_artifacts_restore_exact_forecasts(self) -> None:
        """四模型公共层都必须能保存、加载，再得到相同的递归日预测。"""
        cases = (
            (Direct10ForecastModel(), make_daily("A", 500), pd.Timestamp("2026-05-15")),
            (HurdleForecastModel(), make_daily("B", 200), pd.Timestamp("2025-07-19")),
        )
        for model, daily, cutoff in cases:
            fitted = model.fit(daily, cutoff)
            expected = model.predict(fitted, 3)
            restored = model.deserialize(model.serialize(fitted))
            self.assertEqual(model.predict(restored, 3), expected)
            self.assertTrue(all(isinstance(item, dict) for item in expected))


if __name__ == "__main__":
    unittest.main()
