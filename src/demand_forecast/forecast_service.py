"""加载 active 模型并生成日预测的轻量服务层。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from demand_forecast.backtesting.contracts import DailyForecast
from demand_forecast.forecast_models import AdapterForecastModel, default_forecast_model_factories


class ForecastServiceError(RuntimeError):
    """active 指针或模型 artifact 无法安全用于预测时抛出。"""


class ForecastService:
    """只负责加载当前 active artifact 和预测，不承担训练或模型注册职责。"""

    def __init__(
        self,
        production_dir: str | Path,
        model_factories: Mapping[str, type[AdapterForecastModel]] | None = None,
    ) -> None:
        self.production_dir = Path(production_dir)
        self.model_factories = dict(model_factories or default_forecast_model_factories())

    def predict_active(self, sku: str, horizon: int) -> list[DailyForecast]:
        """读取 SKU 当前 active 模型，并从其 trained_through 次日开始预测。"""
        active = self._read_active(sku)
        artifact_path = self._sku_dir(sku) / "runs" / active["artifact_id"] / "model_artifact.json"
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            model_payload = artifact["production_fit"]["fitted_model"]
        except Exception as error:
            raise ForecastServiceError(f"model_artifact_invalid: {type(error).__name__}: {error}") from error
        if not isinstance(model_payload, Mapping):
            raise ForecastServiceError("model_artifact_missing_fitted_model")
        model_name = str(model_payload.get("model_name", ""))
        if active["model_name"] != model_name:
            raise ForecastServiceError("active_model_name_and_artifact_model_name_mismatch")
        factory = self.model_factories.get(model_name)
        if factory is None:
            raise ForecastServiceError(f"unsupported_model_name: {model_name}")
        model = factory()
        try:
            fitted = model.deserialize(model_payload)
            if fitted.sku != sku:
                raise ForecastServiceError("active_sku_and_artifact_sku_mismatch")
            return model.predict(fitted, horizon)
        except ForecastServiceError:
            raise
        except Exception as error:
            raise ForecastServiceError(f"active_prediction_failed: {type(error).__name__}: {error}") from error

    def _read_active(self, sku: str) -> dict[str, str]:
        """校验最小 active 指针，不接受路径穿越形式的 SKU。"""
        path = self._sku_dir(sku) / "active.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            artifact_id = str(payload["artifact_id"])
            model_name = str(payload["model_name"])
        except FileNotFoundError as error:
            raise ForecastServiceError("active_deployment_not_found") from error
        except Exception as error:
            raise ForecastServiceError(f"active_pointer_invalid: {type(error).__name__}: {error}") from error
        if not artifact_id or not model_name:
            raise ForecastServiceError("active_pointer_missing_required_fields")
        return {"artifact_id": artifact_id, "model_name": model_name}

    def _sku_dir(self, sku: str) -> Path:
        """与 Publisher 相同地拒绝会逃逸输出目录的 SKU。"""
        if not sku or sku in {".", ".."} or any(character in sku for character in ("/", "\\", ":")):
            raise ForecastServiceError("sku_not_safe_for_artifact_path")
        return self.production_dir / "skus" / sku
