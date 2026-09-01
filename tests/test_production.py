"""生产重训与发布的关键契约测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from demand_forecast.backtesting.contracts import BacktestSplit, DailyForecast, ModelUnavailableError
from demand_forecast.backtesting.adapters import _to_quantity_forecast_history
from demand_forecast.model_selection import ModelSelectionAudit, SelectionResult
from demand_forecast.production import ProductionPipeline, ProductionPublisher, ProductionTrainer


SKU = "SKU-PROD"
CUTOFF = pd.Timestamp("2026-08-21")


def make_daily() -> pd.DataFrame:
    """提供包含 cutoff 后行的日序列，测试生产训练不会读取未来输入。"""
    dates = pd.date_range("2026-01-01", "2026-08-25", freq="D")
    return pd.DataFrame(
        {
            "sku": SKU,
            "date": dates,
            "quantity": [float(index % 5) for index in range(len(dates))],
            "is_observed": True,
            "launch_date": pd.Timestamp("2026-01-01"),
            "observation_reason": "observed",
        }
    )


def make_selection(status: str = "selected") -> SelectionResult:
    """构造已完成的正式选择结果；生产层不应重新计算这些指标。"""
    split = BacktestSplit("2026-06-01", "2026-06-30", "2026-07-01", "2026-07-02", expected_test_days=2)
    audits = tuple(
        ModelSelectionAudit(name, "completed", 1.0, 0.0, 0.0, name == "tsb", "baseline", "fixture")
        for name in ("five_period", "direct10", "tsb", "hurdle")
    )
    return SelectionResult(
        sku=SKU,
        status=status,
        winner_model="tsb" if status == "selected" else None,
        split=split,
        evaluated_dates=tuple(split.test_dates),
        baseline_model="five_period",
        audits=audits,
    )


@dataclass(frozen=True)
class FakeFitted:
    """测试用最小 fitted 状态。"""

    trained_through: str
    total: float
    converged: bool = True


class FakeAdapter:
    """记录实际训练切片，验证生产编排复用 adapter fit/forecast 边界。"""

    name = "tsb"

    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode
        self.fit_dates: tuple[pd.Timestamp, ...] | None = None

    def fit(self, train_series: pd.DataFrame, train_end: pd.Timestamp) -> FakeFitted:
        self.fit_dates = tuple(train_series["date"])
        if self.mode == "unavailable":
            raise ModelUnavailableError("fixture_not_applicable")
        if self.mode == "fit_error":
            raise RuntimeError("fixture_fit_error")
        return FakeFitted(train_end.strftime("%Y-%m-%d"), float(train_series["quantity"].sum()), self.mode != "not_converged")

    def serialize(self, fitted: FakeFitted) -> dict[str, object]:
        if self.mode == "serialize_error":
            raise RuntimeError("fixture_serialize_error")
        return {"model_name": "tsb", "implementation_version": "fixture", "trained_through": fitted.trained_through}

    def forecast(self, fitted: FakeFitted, train_series: pd.DataFrame, dates: pd.DatetimeIndex):
        if self.mode == "forecast_error":
            raise RuntimeError("fixture_forecast_error")
        actual_dates = dates[:-1] if self.mode == "short" else dates
        if self.mode == "wrong_date":
            actual_dates = pd.date_range(dates[0] + pd.Timedelta(days=1), periods=len(dates), freq="D")
        return [DailyForecast(SKU, date, fitted.total / 100.0) for date in actual_dates]


def factories(adapter: FakeAdapter) -> dict[str, object]:
    """ProductionTrainer 要求完整四模型映射；本测试只选择 TSB。"""
    return {
        "five_period": lambda: FakeAdapter(),
        "direct10": lambda: FakeAdapter(),
        "tsb": lambda: adapter,
        "hurdle": lambda: FakeAdapter(),
    }


class ProductionTests(unittest.TestCase):
    """覆盖显式 cutoff、失败分类、完整 horizon 与 active 语义。"""

    def test_explicit_cutoff_excludes_later_rows_and_publishes_complete_forecast(self) -> None:
        """训练只看到传入 cutoff 前日期；发布后 artifact 与 50 天记录完整存在。"""
        adapter = FakeAdapter()
        trainer = ProductionTrainer(factories(adapter))
        with tempfile.TemporaryDirectory() as directory:
            pipeline = ProductionPipeline(trainer, ProductionPublisher(directory))
            result = pipeline.run(
                make_daily(), make_selection(), CUTOFF, 50,
                generated_at="2026-08-21T12:00:00Z", selection_model_metadata={"alpha": 0.1},
            )
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(adapter.fit_dates[-1], CUTOFF)
            self.assertNotIn(pd.Timestamp("2026-08-22"), adapter.fit_dates)
            sku_dir = Path(directory) / "skus" / SKU
            active = json.loads((sku_dir / "active.json").read_text(encoding="utf-8"))
            artifact_dir = sku_dir / "runs" / active["artifact_id"]
            forecasts = json.loads((artifact_dir / "forecasts.json").read_text(encoding="utf-8"))
            artifact = json.loads((artifact_dir / "model_artifact.json").read_text(encoding="utf-8"))
            self.assertEqual(len(forecasts), 50)
            self.assertEqual(forecasts[0]["date"], "2026-08-22")
            self.assertEqual(forecasts[-1]["date"], "2026-10-10")
            self.assertEqual(artifact["production_fit"]["trained_through"], "2026-08-21")
            self.assertEqual(artifact["selection_evidence"]["selection_time_model_metadata"], {"alpha": 0.1})

    def test_future_input_mutation_cannot_change_candidate(self) -> None:
        """生产 cutoff 后数据即使存在于输入中，也不能进入 fit 或影响候选。"""
        first_adapter = FakeAdapter()
        second_adapter = FakeAdapter()
        first = ProductionTrainer(factories(first_adapter)).train_and_forecast(
            make_daily(), make_selection(), CUTOFF, 3, generated_at="2026-08-21T12:00:00Z"
        )
        changed = make_daily()
        changed.loc[changed["date"] > CUTOFF, "quantity"] = 9999.0
        second = ProductionTrainer(factories(second_adapter)).train_and_forecast(
            changed, make_selection(), CUTOFF, 3, generated_at="2026-08-21T12:00:00Z"
        )
        self.assertEqual(first.candidate.data_fingerprint, second.candidate.data_fingerprint)
        self.assertEqual(first.candidate.forecasts, second.candidate.forecasts)

    def test_recursive_history_resets_filtered_dataframe_index(self) -> None:
        """不同 SKU 留下的非零原索引不能让第一条预测覆盖真实训练历史。"""
        indexed = make_daily().iloc[:100].copy()
        indexed.index = pd.RangeIndex(start=500, stop=600)
        history = _to_quantity_forecast_history(indexed)
        history.loc[len(history)] = {"date": pd.Timestamp("2026-04-11"), "quantity": 9.0, "occurrence": 1.0}
        self.assertEqual(len(history), 101)
        self.assertEqual(history.iloc[0]["date"], pd.Timestamp("2026-01-01"))

    def test_unselected_and_model_failures_are_explicit_without_fallback(self) -> None:
        """无 winner、不可适用与训练失败都不能改写为另一模型。"""
        no_winner = ProductionTrainer(factories(FakeAdapter())).train_and_forecast(make_daily(), make_selection("unselectable_baseline_unavailable"), CUTOFF, 3)
        unavailable = ProductionTrainer(factories(FakeAdapter("unavailable"))).train_and_forecast(make_daily(), make_selection(), CUTOFF, 3)
        failed = ProductionTrainer(factories(FakeAdapter("fit_error"))).train_and_forecast(make_daily(), make_selection(), CUTOFF, 3)
        self.assertEqual((no_winner.status, unavailable.status, failed.status), ("no_selected_model", "not_applicable", "training_failed"))
        self.assertIsNone(no_winner.deployed_model)
        self.assertIsNone(unavailable.candidate)
        self.assertIsNone(failed.candidate)

    def test_serialization_nonconvergence_and_bad_horizon_are_rejected(self) -> None:
        """不能把未收敛 fitted model、不可保存 artifact 或部分预测发布为成功。"""
        not_converged = ProductionTrainer(factories(FakeAdapter("not_converged"))).train_and_forecast(make_daily(), make_selection(), CUTOFF, 3)
        serialization = ProductionTrainer(factories(FakeAdapter("serialize_error"))).train_and_forecast(make_daily(), make_selection(), CUTOFF, 3)
        short = ProductionTrainer(factories(FakeAdapter("short"))).train_and_forecast(make_daily(), make_selection(), CUTOFF, 3)
        wrong_date = ProductionTrainer(factories(FakeAdapter("wrong_date"))).train_and_forecast(make_daily(), make_selection(), CUTOFF, 3)
        self.assertEqual(not_converged.status, "training_failed")
        self.assertEqual(serialization.status, "serialization_failed")
        self.assertEqual(short.status, "forecast_failed")
        self.assertEqual(wrong_date.status, "forecast_failed")

    def test_failed_new_run_keeps_previous_active_semantics(self) -> None:
        """已有 active 后失败只保留其标识，不会把 selected winner 改成旧模型。"""
        with tempfile.TemporaryDirectory() as directory:
            publisher = ProductionPublisher(directory)
            success = ProductionPipeline(ProductionTrainer(factories(FakeAdapter())), publisher).run(
                make_daily(), make_selection(), CUTOFF, 3, generated_at="2026-08-21T12:00:00Z"
            )
            failed = ProductionPipeline(ProductionTrainer(factories(FakeAdapter("fit_error"))), publisher).run(
                make_daily(), make_selection(), CUTOFF, 3, generated_at="2026-08-22T12:00:00Z"
            )
            self.assertEqual(success.status, "succeeded")
            self.assertEqual(failed.status, "training_failed")
            self.assertEqual(failed.deployed_model, "tsb")
            self.assertIsNotNone(failed.previous_active_artifact_id)
            self.assertEqual(publisher.active_deployment(SKU).artifact_id, failed.previous_active_artifact_id)


if __name__ == "__main__":
    unittest.main()
