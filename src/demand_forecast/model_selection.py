"""按技术规范从统一正式回测结果中选择每个 SKU 的 winner_model。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from demand_forecast.backtesting.contracts import BacktestResult, BacktestSplit


IMPLEMENTATION_VERSION = "v1"
BASELINE_MODEL = "five_period"
# 该顺序只处理规范未定义的完全同 MAE；不参与任何非并列情形的评分。
MODEL_TIE_BREAK_ORDER = ("five_period", "direct10", "tsb", "hurdle")
CHALLENGER_MODELS = tuple(name for name in MODEL_TIE_BREAK_ORDER if name != BASELINE_MODEL)


class SelectionInputError(ValueError):
    """选择输入不完整、重复或不可比较时抛出，禁止静默产生 winner。"""


@dataclass(frozen=True)
class ModelSelectionAudit:
    """单模型在一次选择中的指标、资格与原因，供业务审计而非重新评分。"""

    model_name: str
    backtest_status: str
    mae: float | None
    cumulative_bias: float | None
    absolute_bias: float | None
    eligible: bool
    decision: str
    reason: str


@dataclass(frozen=True)
class SelectionResult:
    """一个 SKU 的纯选择结果；不持有模型、不训练、不预测。"""

    sku: str
    status: str
    winner_model: str | None
    split: BacktestSplit
    evaluated_dates: tuple[pd.Timestamp, ...]
    baseline_model: str
    audits: tuple[ModelSelectionAudit, ...]
    implementation_version: str = IMPLEMENTATION_VERSION

    def __post_init__(self) -> None:
        """阻止不完整结果被误当作可用于重训的 winner。"""
        if self.status not in {"selected", "unselectable_baseline_unavailable", "incomplete_backtest"}:
            raise ValueError("SelectionResult.status 非法")
        if self.status == "selected" and self.winner_model not in MODEL_TIE_BREAK_ORDER:
            raise ValueError("selected 必须包含 canonical winner_model")
        if self.status != "selected" and self.winner_model is not None:
            raise ValueError("不可选择结果不能包含 winner_model")
        if self.baseline_model != BASELINE_MODEL:
            raise ValueError("baseline_model 必须是 five_period")
        if tuple(audit.model_name for audit in self.audits) != MODEL_TIE_BREAK_ORDER:
            raise ValueError("audits 必须按 canonical model order 完整保存")


class ModelSelector:
    """严格执行“相对 FivePeriod 同时改善 MAE 与 |Bias|”的纯决策层。"""

    def select_one_sku(self, results: Sequence[BacktestResult]) -> SelectionResult:
        """选择一个 SKU；输入顺序、scenario 和其他非规范指标均不影响结果。"""
        by_name, sku, split = self._validate_result_set(results)
        completed = [result for result in by_name.values() if result.status == "completed"]
        evaluated_dates = self._validate_completed_comparability(completed, split)
        audits = {name: self._initial_audit(result) for name, result in by_name.items()}

        # failed 表示明确程序/数值故障；绝不能被当作 unavailable 后继续选模。
        if any(result.status == "failed" for result in by_name.values()):
            self._mark_not_compared_audits(by_name, audits, "incomplete_backtest")
            return self._result(sku, "incomplete_backtest", None, split, evaluated_dates, audits)

        baseline = by_name[BASELINE_MODEL]
        if baseline.status == "unavailable":
            audits[BASELINE_MODEL] = self._replace_audit(
                audits[BASELINE_MODEL], decision="baseline_unavailable", reason=baseline.unavailable_reason or "unknown"
            )
            self._mark_not_compared_audits(by_name, audits, "baseline_unavailable")
            return self._result(sku, "unselectable_baseline_unavailable", None, split, evaluated_dates, audits)
        if baseline.status != "completed" or baseline.metrics is None:
            raise SelectionInputError("FivePeriod baseline 状态或指标非法")

        base_mae = baseline.metrics.mae
        base_abs_bias = abs(baseline.metrics.cumulative_bias)
        audits[BASELINE_MODEL] = self._replace_audit(audits[BASELINE_MODEL], decision="baseline", reason="fixed_specification_baseline")
        eligible = [baseline]
        for model_name in CHALLENGER_MODELS:
            candidate = by_name[model_name]
            if candidate.status == "unavailable":
                audits[model_name] = self._replace_audit(
                    audits[model_name], decision="not_applicable", reason=candidate.unavailable_reason or "unknown"
                )
                continue
            if candidate.status != "completed" or candidate.metrics is None:
                raise SelectionInputError(f"{model_name} 状态或指标非法")
            candidate_mae = candidate.metrics.mae
            candidate_abs_bias = abs(candidate.metrics.cumulative_bias)
            if not candidate_mae < base_mae:
                audits[model_name] = self._replace_audit(
                    audits[model_name], decision="rejected", reason="mae_not_strictly_improved"
                )
            elif not candidate_abs_bias < base_abs_bias:
                audits[model_name] = self._replace_audit(
                    audits[model_name], decision="rejected", reason="absolute_bias_not_strictly_improved"
                )
            else:
                eligible.append(candidate)
                audits[model_name] = self._replace_audit(
                    audits[model_name], decision="eligible", reason="mae_and_absolute_bias_strictly_improved", eligible=True
                )

        rank = {name: index for index, name in enumerate(MODEL_TIE_BREAK_ORDER)}
        winner = min(eligible, key=lambda result: (result.metrics.mae, rank[result.model_name]))
        return self._result(sku, "selected", winner.model_name, split, evaluated_dates, audits)

    def select_many(self, results: Sequence[BacktestResult]) -> list[SelectionResult]:
        """按 SKU 分组选择；某个合法 unselectable SKU 不污染其他 SKU 结果。"""
        grouped: dict[str, list[BacktestResult]] = {}
        for result in results:
            grouped.setdefault(result.sku, []).append(result)
        if not grouped:
            raise SelectionInputError("results 不能为空")
        return [self.select_one_sku(grouped[sku]) for sku in sorted(grouped)]

    @staticmethod
    def _validate_result_set(results: Sequence[BacktestResult]) -> tuple[dict[str, BacktestResult], str, BacktestSplit]:
        """要求四个 canonical 模型各有一条明确结果，拒绝缺失、重复、混 SKU 或混窗口。"""
        if not results:
            raise SelectionInputError("一个 SKU 至少需要四个 BacktestResult")
        sku_values = {result.sku for result in results}
        if len(sku_values) != 1:
            raise SelectionInputError("select_one_sku 不能混合多个 SKU")
        by_name: dict[str, BacktestResult] = {}
        for result in results:
            if result.model_name not in MODEL_TIE_BREAK_ORDER:
                raise SelectionInputError(f"未知模型名: {result.model_name}")
            if result.model_name in by_name:
                raise SelectionInputError(f"同一 SKU 存在重复模型结果: {result.model_name}")
            by_name[result.model_name] = result
        missing = set(MODEL_TIE_BREAK_ORDER) - set(by_name)
        if missing:
            raise SelectionInputError(f"缺少模型回测结果: {sorted(missing)}")
        split = next(iter(by_name.values())).split
        if any(result.split != split for result in by_name.values()):
            raise SelectionInputError("同一 SKU 的回测时间窗口不一致")
        return by_name, next(iter(sku_values)), split

    @staticmethod
    def _validate_completed_comparability(
        completed: Sequence[BacktestResult], split: BacktestSplit
    ) -> tuple[pd.Timestamp, ...]:
        """所有 success 模型必须使用同一评价日期集合，而不只是不严谨地比较天数。"""
        reference: tuple[pd.Timestamp, ...] | None = None
        test_dates = set(split.test_dates)
        for result in completed:
            if result.metrics is None or len(result.forecasts) != split.horizon:
                raise SelectionInputError(f"{result.model_name} 的 completed 回测不完整")
            metrics = result.metrics
            numeric = (metrics.mae, metrics.mse, metrics.rmse, metrics.actual_total, metrics.prediction_total, metrics.cumulative_bias)
            if not all(np.isfinite(value) for value in numeric):
                raise SelectionInputError(f"{result.model_name} 包含非有限指标")
            if metrics.wape is not None and not np.isfinite(metrics.wape):
                raise SelectionInputError(f"{result.model_name} 的 WAPE 非法")
            if metrics.mae < 0.0 or metrics.mse < 0.0 or metrics.rmse < 0.0 or (metrics.wape is not None and metrics.wape < 0.0):
                raise SelectionInputError(f"{result.model_name} 包含负误差指标")
            dates = metrics.evaluated_dates
            if not dates or len(dates) != metrics.n_evaluated_days or not set(dates).issubset(test_dates):
                raise SelectionInputError(f"{result.model_name} 的 evaluated_dates 非法")
            if reference is None:
                reference = dates
            elif dates != reference:
                raise SelectionInputError("同一 SKU 的 completed 模型评价日期集合不一致")
        return reference or tuple()

    @staticmethod
    def _initial_audit(result: BacktestResult) -> ModelSelectionAudit:
        """从已计算 metrics 复制审计值；Selector 永远不重新计算它们。"""
        metrics = result.metrics
        return ModelSelectionAudit(
            model_name=result.model_name,
            backtest_status=result.status,
            mae=metrics.mae if metrics else None,
            cumulative_bias=metrics.cumulative_bias if metrics else None,
            absolute_bias=abs(metrics.cumulative_bias) if metrics else None,
            eligible=False,
            decision="pending",
            reason="pending",
        )

    @staticmethod
    def _replace_audit(audit: ModelSelectionAudit, **changes: object) -> ModelSelectionAudit:
        """保持不可变审计记录，避免后续分支意外修改已记录的指标。"""
        values = asdict(audit)
        values.update(changes)
        return ModelSelectionAudit(**values)

    def _mark_not_compared_audits(
        self,
        results: dict[str, BacktestResult],
        audits: dict[str, ModelSelectionAudit],
        reason: str,
    ) -> None:
        """无 baseline 或存在 failed 时仍完整保留每个模型的业务状态与原因。"""
        for model_name, result in results.items():
            if model_name == BASELINE_MODEL:
                continue
            if result.status == "unavailable":
                audits[model_name] = self._replace_audit(
                    audits[model_name], decision="not_applicable", reason=result.unavailable_reason or "unknown"
                )
            elif result.status == "failed":
                audits[model_name] = self._replace_audit(
                    audits[model_name], decision="failed", reason=result.failure_reason or "unknown"
                )
            else:
                audits[model_name] = self._replace_audit(audits[model_name], decision="not_compared", reason=reason)

    @staticmethod
    def _result(
        sku: str,
        status: str,
        winner_model: str | None,
        split: BacktestSplit,
        evaluated_dates: tuple[pd.Timestamp, ...],
        audits: dict[str, ModelSelectionAudit],
    ) -> SelectionResult:
        """按 canonical order 输出审计，使相同输入始终得到相同字段顺序。"""
        return SelectionResult(
            sku=sku,
            status=status,
            winner_model=winner_model,
            split=split,
            evaluated_dates=evaluated_dates,
            baseline_model=BASELINE_MODEL,
            audits=tuple(audits[name] for name in MODEL_TIE_BREAK_ORDER),
        )


def selection_result_to_dict(result: SelectionResult) -> dict[str, object]:
    """把选择结果转为可读 JSON，不保存模型对象或任何预测状态。"""
    return {
        "sku": result.sku,
        "status": result.status,
        "winner_model": result.winner_model,
        "baseline_model": result.baseline_model,
        "split": {
            "train_start": result.split.train_start.strftime("%Y-%m-%d"),
            "train_end": result.split.train_end.strftime("%Y-%m-%d"),
            "test_start": result.split.test_start.strftime("%Y-%m-%d"),
            "test_end": result.split.test_end.strftime("%Y-%m-%d"),
            "expected_test_days": result.split.expected_test_days,
        },
        "evaluated_dates": [date.strftime("%Y-%m-%d") for date in result.evaluated_dates],
        "audits": [asdict(audit) for audit in result.audits],
        "implementation_version": result.implementation_version,
    }


def selection_result_from_dict(payload: dict[str, object]) -> SelectionResult:
    """从 JSON 读回选择结果，重新验证 split、模型名和状态契约。"""
    split_payload = payload.get("split")
    audits_payload = payload.get("audits")
    if not isinstance(split_payload, dict) or not isinstance(audits_payload, list):
        raise SelectionInputError("selection JSON 缺少 split 或 audits")
    split = BacktestSplit(**split_payload)
    audits = tuple(ModelSelectionAudit(**dict(audit)) for audit in audits_payload if isinstance(audit, dict))
    if len(audits) != len(MODEL_TIE_BREAK_ORDER):
        raise SelectionInputError("selection JSON 的 audits 非法")
    return SelectionResult(
        sku=str(payload["sku"]),
        status=str(payload["status"]),
        winner_model=str(payload["winner_model"]) if payload.get("winner_model") is not None else None,
        split=split,
        evaluated_dates=tuple(pd.Timestamp(date).normalize() for date in payload["evaluated_dates"]),
        baseline_model=str(payload["baseline_model"]),
        audits=audits,
        implementation_version=str(payload.get("implementation_version", IMPLEMENTATION_VERSION)),
    )


def write_selection_results(results: Sequence[SelectionResult], output_file: str | Path) -> None:
    """由上层显式调用的轻量持久化函数；Selector 本身保持纯逻辑。"""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "v1", "selections": [selection_result_to_dict(result) for result in results]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_selection_results(input_file: str | Path) -> list[SelectionResult]:
    """读取 writer 生成的 JSON，供后续 winner 重训阶段使用。"""
    payload = json.loads(Path(input_file).read_text(encoding="utf-8"))
    selections = payload.get("selections") if isinstance(payload, dict) else None
    if not isinstance(selections, list):
        raise SelectionInputError("selection JSON 缺少 selections")
    if any(not isinstance(item, dict) for item in selections):
        raise SelectionInputError("selection JSON 的每个 selection 必须是对象")
    return [selection_result_from_dict(item) for item in selections]
