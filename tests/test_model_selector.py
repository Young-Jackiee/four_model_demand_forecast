"""ModelSelector 的规范规则、可比较性、确定性与持久化测试。"""

import tempfile
import unittest

import pandas as pd

from demand_forecast.backtesting.contracts import BacktestMetrics, BacktestResult, BacktestSplit, DailyForecast
from demand_forecast.model_selection import (
    BASELINE_MODEL,
    ModelSelector,
    SelectionInputError,
    read_selection_results,
    write_selection_results,
)


SPLIT = BacktestSplit("2025-01-01", "2025-01-03", "2025-01-04", "2025-01-05", expected_test_days=2)
DATES = tuple(SPLIT.test_dates)
MODELS = ("five_period", "direct10", "tsb", "hurdle")


def completed(
    model_name: str,
    mae: float,
    bias: float,
    *,
    sku: str = "A",
    split: BacktestSplit = SPLIT,
    dates: tuple[pd.Timestamp, ...] = DATES,
    mse: float = 1.0,
    rmse: float = 1.0,
    wape: float | None = 0.5,
    prediction_total: float = 10.0,
) -> BacktestResult:
    """构造已经由统一 evaluator 得到的 completed 回测结果。"""
    forecasts = tuple(DailyForecast(sku, date, 1.0) for date in split.test_dates)
    metrics = BacktestMetrics(
        mae=mae,
        mse=mse,
        rmse=rmse,
        wape=wape,
        actual_total=10.0,
        prediction_total=prediction_total,
        cumulative_bias=bias,
        n_evaluated_days=len(dates),
        evaluated_dates=dates,
    )
    return BacktestResult(sku, model_name, split, "completed", None, forecasts, metrics, {})


def unavailable(model_name: str, reason: str = "insufficient_history", sku: str = "A") -> BacktestResult:
    """构造正常业务不可适用结果。"""
    return BacktestResult(sku, model_name, SPLIT, "unavailable", reason, tuple(), None, None)


def failed(model_name: str, reason: str = "optimizer_failed", sku: str = "A") -> BacktestResult:
    """构造显式失败结果，确认 selector 不会将其伪装成不可适用。"""
    return BacktestResult(sku, model_name, SPLIT, "failed", None, tuple(), None, None, failure_reason=reason)


def result_set(base: BacktestResult, direct: BacktestResult, tsb: BacktestResult, hurdle: BacktestResult) -> list[BacktestResult]:
    """按任意输入顺序传递前的完整四模型结果集合。"""
    return [base, direct, tsb, hurdle]


class ModelSelectorTests(unittest.TestCase):
    """每项测试直接锁定一条选择规则，避免规则随重构而漂移。"""

    def setUp(self) -> None:
        self.selector = ModelSelector()

    def test_specification_example_uses_two_strict_gates_then_lowest_eligible_mae(self) -> None:
        """MAE 或 |Bias| 任一未改善都不能挑战 baseline。"""
        result = self.selector.select_one_sku(
            result_set(
                completed("five_period", 1.5, 40.0),
                completed("direct10", 1.6, 10.0),
                completed("tsb", 1.3, 25.0),
                completed("hurdle", 1.1, 60.0),
            )
        )
        audits = {audit.model_name: audit for audit in result.audits}
        self.assertEqual(result.winner_model, "tsb")
        self.assertTrue(audits["tsb"].eligible)
        self.assertEqual(audits["hurdle"].reason, "absolute_bias_not_strictly_improved")
        self.assertEqual(audits["direct10"].reason, "mae_not_strictly_improved")

    def test_strict_equality_sign_changes_and_zero_baseline_bias(self) -> None:
        """规则是 <；Bias 仅比较绝对值，baseline Bias=0 时 challenger 不可能合格。"""
        equal_mae = self.selector.select_one_sku(
            result_set(completed("five_period", 1.5, 10.0), completed("direct10", 9.0, 99.0), completed("tsb", 1.5, 1.0), completed("hurdle", 9.0, 99.0))
        )
        equal_bias = self.selector.select_one_sku(
            result_set(completed("five_period", 1.5, -10.0), completed("direct10", 9.0, 99.0), completed("tsb", 1.0, 10.0), completed("hurdle", 9.0, 99.0))
        )
        sign_change = self.selector.select_one_sku(
            result_set(completed("five_period", 1.5, 20.0), completed("direct10", 9.0, 99.0), completed("tsb", 1.0, -10.0), completed("hurdle", 9.0, 99.0))
        )
        zero_bias = self.selector.select_one_sku(
            result_set(completed("five_period", 2.0, 0.0), completed("direct10", 9.0, 99.0), completed("tsb", 1.0, 0.01), completed("hurdle", 9.0, 99.0))
        )
        self.assertEqual(equal_mae.winner_model, BASELINE_MODEL)
        self.assertEqual(equal_bias.winner_model, BASELINE_MODEL)
        self.assertEqual(sign_change.winner_model, "tsb")
        self.assertEqual(zero_bias.winner_model, BASELINE_MODEL)

    def test_multiple_eligible_tie_break_and_irrelevant_metrics_do_not_change_winner(self) -> None:
        """eligible 内只比较 MAE；完全并列时使用固定 canonical order。"""
        base = completed("five_period", 2.0, 100.0)
        direct = completed("direct10", 1.2, 50.0, mse=999.0, rmse=99.0, wape=9.0, prediction_total=999.0)
        tsb = completed("tsb", 1.2, 60.0, mse=0.0001, rmse=0.01, wape=0.001, prediction_total=0.1)
        hurdle = completed("hurdle", 1.3, 40.0)
        first = self.selector.select_one_sku(result_set(base, direct, tsb, hurdle))
        second = self.selector.select_one_sku([hurdle, tsb, base, direct])
        self.assertEqual(first.winner_model, "direct10")
        self.assertEqual(first, second)
        self.assertEqual({audit.model_name for audit in first.audits if audit.eligible}, {"direct10", "tsb", "hurdle"})

    def test_unavailable_and_failed_have_different_selection_outcomes(self) -> None:
        """challenger unavailable 可审计地退出；baseline unavailable 与 failed 均不产生 winner。"""
        challenger_unavailable = self.selector.select_one_sku(
            result_set(completed("five_period", 2.0, 10.0), unavailable("direct10"), completed("tsb", 1.0, 5.0), unavailable("hurdle"))
        )
        base_unavailable = self.selector.select_one_sku(
            result_set(unavailable("five_period"), completed("direct10", 1.0, 1.0), completed("tsb", 1.0, 1.0), completed("hurdle", 1.0, 1.0))
        )
        candidate_failed = self.selector.select_one_sku(
            result_set(completed("five_period", 2.0, 10.0), failed("direct10"), completed("tsb", 1.0, 5.0), unavailable("hurdle"))
        )
        self.assertEqual(challenger_unavailable.winner_model, "tsb")
        self.assertEqual(base_unavailable.status, "unselectable_baseline_unavailable")
        self.assertIsNone(base_unavailable.winner_model)
        self.assertEqual({audit.model_name: audit.decision for audit in base_unavailable.audits}["tsb"], "not_compared")
        self.assertEqual(candidate_failed.status, "incomplete_backtest")
        self.assertIsNone(candidate_failed.winner_model)
        self.assertEqual({audit.model_name: audit.decision for audit in candidate_failed.audits}["direct10"], "failed")

    def test_invalid_missing_duplicate_sku_split_mask_and_non_finite_results_fail_fast(self) -> None:
        """不可审计输入不能被静默选择，避免把不同实验或损坏结果混在一起。"""
        good = result_set(
            completed("five_period", 2.0, 10.0), completed("direct10", 1.0, 5.0), completed("tsb", 1.2, 6.0), completed("hurdle", 1.3, 7.0)
        )
        with self.assertRaises(SelectionInputError):
            self.selector.select_one_sku(good[:-1])
        with self.assertRaises(SelectionInputError):
            self.selector.select_one_sku(good + [completed("five_period", 1.0, 1.0)])
        with self.assertRaises(SelectionInputError):
            self.selector.select_one_sku([*good[:3], completed("hurdle", 1.3, 7.0, sku="B")])
        other_split = BacktestSplit("2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", expected_test_days=2)
        with self.assertRaises(SelectionInputError):
            self.selector.select_one_sku([*good[:3], completed("hurdle", 1.3, 7.0, split=other_split, dates=tuple(other_split.test_dates))])
        with self.assertRaises(SelectionInputError):
            self.selector.select_one_sku([*good[:3], completed("hurdle", 1.3, 7.0, dates=(DATES[0],))])

        object.__setattr__(good[1].metrics, "mae", float("nan"))
        with self.assertRaises(SelectionInputError):
            self.selector.select_one_sku(good)

    def test_batch_isolation_determinism_and_json_round_trip(self) -> None:
        """不同 SKU 独立选择；相同输入重复选择与保存加载后均保持一致。"""
        a = result_set(completed("five_period", 2.0, 10.0), completed("direct10", 1.0, 5.0), completed("tsb", 1.2, 4.0), completed("hurdle", 1.3, 3.0))
        b = [completed(result.model_name, result.metrics.mae, result.metrics.cumulative_bias, sku="B") for result in a]
        selected = self.selector.select_many([*reversed(a), *reversed(b)])
        self.assertEqual([result.sku for result in selected], ["A", "B"])
        self.assertTrue(all(result.winner_model == "direct10" for result in selected))
        self.assertEqual([self.selector.select_one_sku(a) for _ in range(100)], [selected[0]] * 100)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/selection_results.json"
            write_selection_results(selected, path)
            self.assertEqual(read_selection_results(path), selected)


if __name__ == "__main__":
    unittest.main()
