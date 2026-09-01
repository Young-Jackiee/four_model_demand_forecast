"""winner 全量重训、未来预测与最小原子发布。

本模块不重新选模，也不提供自动 fallback；它只把已选 winner 变成可审计的生产候选。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from demand_forecast.backtesting.adapters import BacktestModelAdapter
from demand_forecast.backtesting.contracts import DailyForecast, ModelUnavailableError, validate_daily_forecast
from demand_forecast.data.schemas import DataContractError, validate_daily_sales
from demand_forecast.forecast_models import default_forecast_model_factories
from demand_forecast.model_contracts import ForecastModel
from demand_forecast.model_selection import SelectionResult


PRODUCTION_SCHEMA_VERSION = "v1"
PRODUCTION_STATUSES = {
    "ready_to_publish",
    "succeeded",
    "no_selected_model",
    "not_applicable",
    "data_invalid",
    "training_failed",
    "serialization_failed",
    "forecast_failed",
    "publish_failed",
}

ModelFactory = Callable[[], ForecastModel | BacktestModelAdapter]


class ProductionInputError(ValueError):
    """生产编排输入违反契约时抛出；不可被误判为模型不适用。"""


class ProductionPublishError(RuntimeError):
    """候选已生成但无法完整发布时抛出。"""


@dataclass(frozen=True)
class ActiveDeployment:
    """active.json 中最小的上一版语义，不承担完整 model registry 职责。"""

    artifact_id: str
    model_name: str
    trained_through: str


@dataclass(frozen=True)
class ProductionCandidate:
    """尚未发布的完整生产候选；只有通过全部校验后才允许交给 Publisher。"""

    artifact_id: str
    sku: str
    selected_model: str
    production_train_end: pd.Timestamp
    generated_at: pd.Timestamp
    model_artifact: Mapping[str, object]
    forecasts: tuple[DailyForecast, ...]
    selection_evidence: Mapping[str, object]
    data_fingerprint: str


@dataclass(frozen=True)
class ProductionRunResult:
    """一次生产尝试的审计结果；失败时 candidate 必须为 None。"""

    sku: str
    status: str
    selected_model: str | None
    deployed_model: str | None
    production_train_end: pd.Timestamp
    generated_at: pd.Timestamp
    reason_code: str | None
    reason_detail: str | None
    candidate: ProductionCandidate | None
    previous_active_artifact_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in PRODUCTION_STATUSES:
            raise ValueError("ProductionRunResult.status 非法")
        if self.status == "ready_to_publish" and self.candidate is None:
            raise ValueError("ready_to_publish 必须包含 candidate")
        if self.status != "ready_to_publish" and self.candidate is not None:
            raise ValueError("非 ready_to_publish 结果不能包含 candidate")
        if self.status in {"succeeded", "ready_to_publish"} and not self.selected_model:
            raise ValueError("成功候选必须记录 selected_model")


def default_model_factories() -> dict[str, ModelFactory]:
    """返回四个统一 ForecastModel 工厂；每次训练创建独立冻结状态。"""
    return default_forecast_model_factories()


class ProductionTrainer:
    """复用正式回测 adapter 的完整 fit/forecast contract 构造生产候选。"""

    def __init__(self, model_factories: Mapping[str, ModelFactory] | None = None) -> None:
        self.model_factories = dict(model_factories or default_model_factories())
        if set(self.model_factories) != {"five_period", "direct10", "tsb", "hurdle"}:
            raise ProductionInputError("model_factories 必须完整覆盖四个规范模型")

    def train_and_forecast(
        self,
        daily_series: pd.DataFrame,
        selection: SelectionResult,
        production_train_end: pd.Timestamp | str,
        forecast_horizon: int,
        *,
        generated_at: pd.Timestamp | str | None = None,
        selection_model_metadata: Mapping[str, object] | None = None,
    ) -> ProductionRunResult:
        """显式 cutoff 重训 winner 并预测；绝不从输入数据最大日期推断 cutoff。"""
        cutoff = self._parse_production_train_end(production_train_end)
        timestamp = self._parse_generated_at(generated_at)
        sku = selection.sku

        # 未选出 winner 是正式回测的正常结果，不应伪造一个生产模型。
        if selection.status != "selected" or selection.winner_model is None:
            return self._result(
                sku, "no_selected_model", None, cutoff, timestamp,
                "selection_not_available", selection.status,
            )

        try:
            training_series = self._prepare_training_series(daily_series, selection.sku, cutoff)
            self._validate_horizon(forecast_horizon)
        except (DataContractError, ProductionInputError, ValueError) as error:
            return self._result(
                sku, "data_invalid", selection.winner_model, cutoff, timestamp,
                "production_input_invalid", str(error),
            )

        model = self.model_factories[selection.winner_model]()
        try:
            # 与 Backtester 相同：仅把 cutoff 前实际日序列交给完整 fit procedure。
            fitted = model.fit(training_series.copy(), cutoff)
            self._validate_fitted_completion(fitted)
        except ModelUnavailableError as error:
            return self._result(
                sku, "not_applicable", selection.winner_model, cutoff, timestamp,
                "model_not_applicable", error.reason,
            )
        except Exception as error:  # 仅在训练边界分类；不做 fallback，也不吞掉来源信息。
            return self._result(
                sku, "training_failed", selection.winner_model, cutoff, timestamp,
                "fit_exception", self._exception_detail(error),
            )

        try:
            model_artifact = dict(model.serialize(fitted))
            self._validate_model_artifact(model_artifact, selection.winner_model, cutoff)
            # 先验证 JSON 可保存，避免“模型训练成功但 artifact 不可交付”。
            json.dumps(model_artifact, ensure_ascii=False, allow_nan=False)
        except Exception as error:
            return self._result(
                sku, "serialization_failed", selection.winner_model, cutoff, timestamp,
                "serialize_exception", self._exception_detail(error),
            )

        dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
        try:
            if hasattr(model, "predict"):
                forecasts = tuple(model.predict(fitted, forecast_horizon))
            else:
                forecasts = tuple(model.forecast(fitted, training_series.copy(), dates))
            self._validate_forecasts(forecasts, sku, dates)
        except Exception as error:
            return self._result(
                sku, "forecast_failed", selection.winner_model, cutoff, timestamp,
                "forecast_exception", self._exception_detail(error),
            )

        artifact_id = self._make_artifact_id(sku, selection.winner_model, cutoff, timestamp)
        evidence = self._selection_evidence(selection, selection_model_metadata)
        candidate = ProductionCandidate(
            artifact_id=artifact_id,
            sku=sku,
            selected_model=selection.winner_model,
            production_train_end=cutoff,
            generated_at=timestamp,
            model_artifact=model_artifact,
            forecasts=forecasts,
            selection_evidence=evidence,
            data_fingerprint=self._fingerprint(training_series),
        )
        return self._result(
            sku, "ready_to_publish", selection.winner_model, cutoff, timestamp,
            None, None, candidate=candidate,
        )

    @staticmethod
    def _parse_production_train_end(value: pd.Timestamp | str) -> pd.Timestamp:
        """cutoff 必须由上层传入；此函数只解析，不从 DataFrame 推断。"""
        result = pd.Timestamp(value).normalize()
        if pd.isna(result):
            raise ProductionInputError("production_train_end 必须是可解析日期")
        if result.tzinfo is not None:
            raise ProductionInputError("production_train_end 必须是无时区的业务自然日")
        return result

    @staticmethod
    def _parse_generated_at(value: pd.Timestamp | str | None) -> pd.Timestamp:
        """允许测试传入固定时间；默认记录 UTC 当前时刻。"""
        result = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
        if pd.isna(result):
            raise ProductionInputError("generated_at 必须是可解析时间")
        return result

    @staticmethod
    def _validate_horizon(horizon: int) -> None:
        """生产窗口必须是正整数，防止 bool 或零天被误当成功。"""
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ProductionInputError("forecast_horizon 必须是正整数")

    @staticmethod
    def _prepare_training_series(daily_series: pd.DataFrame, sku: str, cutoff: pd.Timestamp) -> pd.DataFrame:
        """验证单 SKU 连续日历，并显式切到 cutoff；未来行只能保留在输入外侧。"""
        daily = validate_daily_sales(daily_series)
        sku_values = daily["sku"].dropna().astype(str).unique()
        if len(sku_values) != 1 or str(sku_values[0]) != sku:
            raise ProductionInputError("生产训练数据必须只包含 SelectionResult 对应的一个 SKU")
        if not daily["date"].is_monotonic_increasing:
            raise ProductionInputError("生产训练数据必须按日期升序，禁止静默排序")
        training = daily.loc[daily["date"] <= cutoff].copy()
        if training.empty or cutoff not in set(training["date"]):
            raise ProductionInputError("production_train_end 在输入日序列中不存在")
        expected_dates = pd.date_range(training["date"].iloc[0], cutoff, freq="D")
        if len(training) != len(expected_dates) or list(training["date"]) != list(expected_dates):
            raise ProductionInputError("截至 production_train_end 的日序列必须连续且无重复")
        return training

    @staticmethod
    def _validate_fitted_completion(fitted: object) -> None:
        """五周期和 Direct10 显式报告未收敛时，第一版禁止把近似解静默上线。"""
        if getattr(fitted, "converged", True) is False:
            raise RuntimeError("optimizer_not_converged")

    @staticmethod
    def _validate_forecasts(
        forecasts: Sequence[DailyForecast],
        expected_sku: str,
        expected_dates: pd.DatetimeIndex,
    ) -> None:
        """在发布前检查完整 50 天（或调用方给定 horizon）而非仅检查聚合数值。"""
        if len(forecasts) != len(expected_dates):
            raise ProductionInputError("forecast_count_mismatch")
        actual_dates: list[pd.Timestamp] = []
        for forecast in forecasts:
            try:
                normalized = validate_daily_forecast(forecast)
            except ValueError as error:
                raise ProductionInputError("forecast_item_not_daily_forecast") from error
            if normalized["sku"] != expected_sku:
                raise ProductionInputError("forecast_sku_mismatch")
            actual_dates.append(pd.Timestamp(normalized["date"]))
        if len(set(actual_dates)) != len(actual_dates) or actual_dates != list(expected_dates):
            raise ProductionInputError("forecast_dates_must_be_contiguous_and_exact")

    @staticmethod
    def _validate_model_artifact(
        model_artifact: Mapping[str, object],
        expected_model_name: str,
        cutoff: pd.Timestamp,
    ) -> None:
        """发布前阻止缺少模型版本或 cutoff 不一致的 artifact。"""
        if model_artifact.get("model_name") != expected_model_name:
            raise ProductionInputError("model_artifact_name_mismatch")
        if not str(model_artifact.get("model_version", "")).strip():
            raise ProductionInputError("model_artifact_missing_model_version")
        if str(model_artifact.get("trained_through")) != cutoff.strftime("%Y-%m-%d"):
            raise ProductionInputError("model_artifact_trained_through_mismatch")

    @staticmethod
    def _selection_evidence(
        selection: SelectionResult,
        selection_model_metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        """分开保留“为何被选中”和“本次重训得到什么参数”。"""
        winner_audit = next(audit for audit in selection.audits if audit.model_name == selection.winner_model)
        return {
            "selection_status": selection.status,
            "selected_model": selection.winner_model,
            "formal_backtest": {
                "train_start": selection.split.train_start.strftime("%Y-%m-%d"),
                "train_end": selection.split.train_end.strftime("%Y-%m-%d"),
                "test_start": selection.split.test_start.strftime("%Y-%m-%d"),
                "test_end": selection.split.test_end.strftime("%Y-%m-%d"),
                "mae": winner_audit.mae,
                "cumulative_bias": winner_audit.cumulative_bias,
            },
            # 当前 selection JSON 不自带 fitted metadata；同轮运行时由上层显式传入。
            "selection_time_model_metadata": dict(selection_model_metadata) if selection_model_metadata else None,
        }

    @staticmethod
    def _fingerprint(training_series: pd.DataFrame) -> str:
        """对真正进入本次 fit 的标准日序列求摘要，便于日后复现而不保存副本。"""
        columns = ["sku", "date", "quantity", "is_observed", "launch_date", "observation_reason"]
        stable = training_series.loc[:, columns].copy()
        stable["date"] = stable["date"].dt.strftime("%Y-%m-%d")
        stable["launch_date"] = stable["launch_date"].dt.strftime("%Y-%m-%d")
        payload = stable.to_csv(index=False, lineterminator="\n").encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _make_artifact_id(sku: str, model_name: str, cutoff: pd.Timestamp, generated_at: pd.Timestamp) -> str:
        """文件夹名只使用跨平台安全字符；同一秒重复运行仍由 Publisher 拒绝覆盖。"""
        stamp = generated_at.tz_convert("UTC").strftime("%Y%m%dT%H%M%S%fZ") if generated_at.tzinfo else generated_at.strftime("%Y%m%dT%H%M%S%f")
        return f"{sku}__{model_name}__{cutoff.strftime('%Y%m%d')}__{stamp}"

    @staticmethod
    def _exception_detail(error: Exception) -> str:
        """保存类型和消息，便于审计但不把 traceback 直接写进业务 artifact。"""
        return f"{type(error).__name__}: {error}"

    @staticmethod
    def _result(
        sku: str,
        status: str,
        selected_model: str | None,
        cutoff: pd.Timestamp,
        generated_at: pd.Timestamp,
        reason_code: str | None,
        reason_detail: str | None,
        *,
        candidate: ProductionCandidate | None = None,
    ) -> ProductionRunResult:
        """集中构造不可变结果，保证失败路径不会意外携带半成品 candidate。"""
        return ProductionRunResult(
            sku=sku,
            status=status,
            selected_model=selected_model,
            deployed_model=None,
            production_train_end=cutoff,
            generated_at=generated_at,
            reason_code=reason_code,
            reason_detail=reason_detail,
            candidate=candidate,
        )


class ProductionPublisher:
    """将完整 candidate 写入独立 run 目录，最后才原子替换 active 指针。"""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def active_deployment(self, sku: str) -> ActiveDeployment | None:
        """读取指定 SKU 的上一版 active 指针；不同 SKU 的状态绝不能共用。"""
        path = self._sku_dir(sku) / "active.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return ActiveDeployment(
                artifact_id=str(payload["artifact_id"]),
                model_name=str(payload["model_name"]),
                trained_through=str(payload["trained_through"]),
            )
        except Exception as error:
            raise ProductionPublishError(f"active_pointer_invalid: {type(error).__name__}: {error}") from error

    def publish(self, candidate: ProductionCandidate) -> ActiveDeployment:
        """先完成 SKU 专属 run 文件夹，再更新该 SKU 的 active.json。"""
        sku_dir = self._sku_dir(candidate.sku)
        runs_dir = sku_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        final_dir = runs_dir / candidate.artifact_id
        if final_dir.exists():
            raise ProductionPublishError("artifact_id_already_exists")
        temporary_dir = Path(tempfile.mkdtemp(prefix=f".{candidate.artifact_id}.", dir=runs_dir))
        try:
            artifact_payload = self._artifact_payload(candidate)
            forecast_payload = self._forecast_payload(candidate)
            self._write_json(temporary_dir / "model_artifact.json", artifact_payload)
            self._write_json(temporary_dir / "forecasts.json", forecast_payload)
            self._write_json(temporary_dir / "manifest.json", self._manifest_payload(candidate))
            # 同一磁盘内 rename 是原子目录切换；此时 active 仍指向上一版。
            os.replace(temporary_dir, final_dir)
            active = ActiveDeployment(candidate.artifact_id, candidate.selected_model, candidate.production_train_end.strftime("%Y-%m-%d"))
            self._replace_active_pointer(candidate.sku, active)
            return active
        except Exception as error:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)
            if isinstance(error, ProductionPublishError):
                raise
            raise ProductionPublishError(f"publish_exception: {type(error).__name__}: {error}") from error

    def _replace_active_pointer(self, sku: str, active: ActiveDeployment) -> None:
        """用 SKU 专属临时文件替换指针，防止并行 SKU 互相覆盖。"""
        sku_dir = self._sku_dir(sku)
        sku_dir.mkdir(parents=True, exist_ok=True)
        temporary = sku_dir / ".active.json.tmp"
        try:
            self._write_json(
                temporary,
                {
                    "schema_version": PRODUCTION_SCHEMA_VERSION,
                    "artifact_id": active.artifact_id,
                    "model_name": active.model_name,
                    "trained_through": active.trained_through,
                },
            )
            os.replace(temporary, sku_dir / "active.json")
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    def _sku_dir(self, sku: str) -> Path:
        """SKU 已通过日序列契约校验；仍拒绝路径分隔符，避免用户标识逃逸目录。"""
        if sku in {"", ".", ".."} or any(character in sku for character in ("/", "\\", ":")):
            raise ProductionPublishError("sku_not_safe_for_artifact_path")
        return self.output_dir / "skus" / sku

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, object] | Sequence[Mapping[str, object]]) -> None:
        """禁止 NaN/Inf 进入 JSON，写入后立即 flush，尽早暴露磁盘错误。"""
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _artifact_payload(candidate: ProductionCandidate) -> dict[str, object]:
        """artifact 同时保存选择证据和生产重训状态，但二者使用不同字段避免混淆。"""
        return {
            "schema_version": PRODUCTION_SCHEMA_VERSION,
            "artifact_id": candidate.artifact_id,
            "sku": candidate.sku,
            "selected_model": candidate.selected_model,
            "deployed_model": candidate.selected_model,
            "deployment_status": "succeeded",
            "generated_at": candidate.generated_at.isoformat(),
            "feature_version": candidate.model_artifact.get("feature_version"),
            "data_policy": candidate.model_artifact.get("data_policy"),
            "selection_evidence": dict(candidate.selection_evidence),
            "production_fit": {
                "trained_through": candidate.production_train_end.strftime("%Y-%m-%d"),
                "input_data_fingerprint": candidate.data_fingerprint,
                "fitted_model": dict(candidate.model_artifact),
            },
        }

    @staticmethod
    def _forecast_payload(candidate: ProductionCandidate) -> list[dict[str, object]]:
        """预测记录保留必要 provenance；components 仅在模型提供时保存。"""
        rows: list[dict[str, object]] = []
        for forecast in candidate.forecasts:
            rows.append(
                {
                    "artifact_id": candidate.artifact_id,
                    "sku": forecast["sku"],
                    "date": forecast["date"].isoformat(),
                    "prediction": forecast["prediction"],
                    "model_name": candidate.selected_model,
                    "model_version": candidate.model_artifact["model_version"],
                    "trained_through": candidate.production_train_end.strftime("%Y-%m-%d"),
                    "generated_at": candidate.generated_at.isoformat(),
                    "components": dict(forecast["components"]) if "components" in forecast else None,
                }
            )
        return rows

    @staticmethod
    def _manifest_payload(candidate: ProductionCandidate) -> dict[str, object]:
        """manifest 仅描述该 run 是否完整，不能被误当成模型参数文件。"""
        return {
            "schema_version": PRODUCTION_SCHEMA_VERSION,
            "artifact_id": candidate.artifact_id,
            "sku": candidate.sku,
            "model_name": candidate.selected_model,
            "trained_through": candidate.production_train_end.strftime("%Y-%m-%d"),
            "generated_at": candidate.generated_at.isoformat(),
            "forecast_days": len(candidate.forecasts),
            "forecast_start": candidate.forecasts[0]["date"].isoformat(),
            "forecast_end": candidate.forecasts[-1]["date"].isoformat(),
        }


class ProductionPipeline:
    """将训练候选与发布串联；失败只记录状态，绝不自动更换 selected_model。"""

    def __init__(self, trainer: ProductionTrainer, publisher: ProductionPublisher) -> None:
        self.trainer = trainer
        self.publisher = publisher

    def run(
        self,
        daily_series: pd.DataFrame,
        selection: SelectionResult,
        production_train_end: pd.Timestamp | str,
        forecast_horizon: int,
        *,
        generated_at: pd.Timestamp | str | None = None,
        selection_model_metadata: Mapping[str, object] | None = None,
    ) -> ProductionRunResult:
        """失败时保留上一版 active 语义；首次运行则 deployed_model 为 None。"""
        previous = self.publisher.active_deployment(selection.sku)
        result = self.trainer.train_and_forecast(
            daily_series,
            selection,
            production_train_end,
            forecast_horizon,
            generated_at=generated_at,
            selection_model_metadata=selection_model_metadata,
        )
        if result.status != "ready_to_publish":
            return replace(
                result,
                deployed_model=previous.model_name if previous else None,
                previous_active_artifact_id=previous.artifact_id if previous else None,
            )
        assert result.candidate is not None
        try:
            deployed = self.publisher.publish(result.candidate)
        except Exception as error:
            return ProductionRunResult(
                sku=result.sku,
                status="publish_failed",
                selected_model=result.selected_model,
                deployed_model=previous.model_name if previous else None,
                production_train_end=result.production_train_end,
                generated_at=result.generated_at,
                reason_code="publish_exception",
                reason_detail=f"{type(error).__name__}: {error}",
                candidate=None,
                previous_active_artifact_id=previous.artifact_id if previous else None,
            )
        return ProductionRunResult(
            sku=result.sku,
            status="succeeded",
            selected_model=result.selected_model,
            deployed_model=deployed.model_name,
            production_train_end=result.production_train_end,
            generated_at=result.generated_at,
            reason_code=None,
            reason_detail=None,
            candidate=None,
            previous_active_artifact_id=previous.artifact_id if previous else None,
        )


def write_production_run_results(results: Sequence[ProductionRunResult], output_file: str | Path) -> None:
    """保存所有 SKU 的成功或失败状态；失败也必须可审计。"""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "runs": [
            {
                "sku": result.sku,
                "status": result.status,
                "selected_model": result.selected_model,
                "deployed_model": result.deployed_model,
                "production_train_end": result.production_train_end.strftime("%Y-%m-%d"),
                "generated_at": result.generated_at.isoformat(),
                "reason_code": result.reason_code,
                "reason_detail": result.reason_detail,
                "previous_active_artifact_id": result.previous_active_artifact_id,
            }
            for result in results
        ],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
