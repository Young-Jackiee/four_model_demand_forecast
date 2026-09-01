"""规范定义的五周期基准模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from demand_forecast.features.definitions import CURRENT_MEAN_FEATURES, FEATURE_VERSION


IMPLEMENTATION_VERSION = "v1"
MODEL_NAME = "five_period"


@dataclass(frozen=True)
class FivePeriodConfig:
    """投影梯度下降的可复现配置；默认值是实现决策，不是规范指定数值。"""

    ridge: float = 1e-3
    # 滚动窗口高度相关，需允许足够迭代后再以稳定的权重变化阈值停止。
    max_iter: int = 100_000
    # 相关滚动窗口下 1e-7 在 10 万次投影后仍可能不停止；1e-6 仍远小于业务指标容差。
    tol: float = 1e-6
    step_size: float | None = None

    def validate(self) -> None:
        """在训练前拒绝无意义的优化参数。"""
        if self.ridge < 0:
            raise ValueError("ridge 不能为负")
        if self.max_iter < 1:
            raise ValueError("max_iter 至少为 1")
        if self.tol <= 0:
            raise ValueError("tol 必须大于 0")
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError("step_size 必须为正数或 None")


@dataclass(frozen=True)
class FivePeriodFittedModel:
    """一次训练后的不可变模型状态，可被保存和重新加载。"""

    weights: tuple[float, ...]
    feature_names: tuple[str, ...]
    feature_version: str
    trained_through: str
    config: FivePeriodConfig
    effective_step_size: float
    n_training_rows: int
    iterations: int
    converged: bool
    final_objective: float
    implementation_version: str = IMPLEMENTATION_VERSION


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """投影到非负且和为 1 的概率单纯形，使用标准排序阈值算法。"""
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("待投影向量必须是一维、非空且有限")

    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values)
    indices = np.arange(1, values.size + 1)
    valid = sorted_values - (cumulative - 1.0) / indices > 0
    rho = np.nonzero(valid)[0][-1]
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(values - theta, 0.0)


class FivePeriodModel:
    """学习五个滚动窗口的非负且和为 1 的加权系数。"""

    name = MODEL_NAME

    def __init__(self, config: FivePeriodConfig | None = None) -> None:
        self.config = config or FivePeriodConfig()
        self.config.validate()

    def fit(self, training_features: pd.DataFrame, trained_through: str) -> FivePeriodFittedModel:
        """对 FeatureBuilder 输出的训练表执行确定性的投影梯度下降。"""
        trained_date = pd.Timestamp(trained_through).normalize()
        if pd.isna(trained_date):
            raise ValueError("trained_through 必须是可解析日期")
        x_values, y_values = self._validate_training_features(training_features, trained_date)

        uniform = np.full(len(CURRENT_MEAN_FEATURES), 1.0 / len(CURRENT_MEAN_FEATURES))
        step_size = self._resolve_step_size(x_values)
        weights = uniform.copy()
        converged = False
        iterations = self.config.max_iter

        for iteration in range(1, self.config.max_iter + 1):
            gradient = self._gradient(x_values, y_values, weights, uniform)
            updated = project_to_simplex(weights - step_size * gradient)
            if np.max(np.abs(updated - weights)) <= self.config.tol:
                weights = updated
                converged = True
                iterations = iteration
                break
            weights = updated

        # 最后再投影一次，消除浮点误差，保证序列化状态严格满足约束。
        weights = project_to_simplex(weights)
        objective = self._objective(x_values, y_values, weights, uniform)
        return FivePeriodFittedModel(
            weights=tuple(float(value) for value in weights),
            feature_names=CURRENT_MEAN_FEATURES,
            feature_version=FEATURE_VERSION,
            trained_through=trained_date.strftime("%Y-%m-%d"),
            config=self.config,
            effective_step_size=float(step_size),
            n_training_rows=len(training_features),
            iterations=iterations,
            converged=converged,
            final_objective=float(objective),
        )

    def predict_one(
        self,
        fitted: FivePeriodFittedModel,
        feature_row: Mapping[str, float] | pd.Series,
    ) -> float:
        """按特征名称而非列位置取值，避免调用方重排列后静默预测错误。"""
        self._validate_fitted(fitted)
        values = self._extract_feature_row(feature_row, fitted.feature_names)
        prediction = float(np.dot(np.asarray(fitted.weights), values))
        # 理论上凸组合已保证非负；此处只消除极小的浮点负数。
        return max(0.0, prediction)

    def serialize(self, fitted: FivePeriodFittedModel) -> dict[str, object]:
        """输出 JSON 可直接保存的纯 Python 字典。"""
        self._validate_fitted(fitted)
        return {
            "model_name": self.name,
            "weights": list(fitted.weights),
            "feature_names": list(fitted.feature_names),
            "feature_version": fitted.feature_version,
            "trained_through": fitted.trained_through,
            "hyperparameters": asdict(fitted.config),
            "effective_step_size": fitted.effective_step_size,
            "n_training_rows": fitted.n_training_rows,
            "iterations": fitted.iterations,
            "converged": fitted.converged,
            "final_objective": fitted.final_objective,
            "implementation_version": fitted.implementation_version,
        }

    def deserialize(self, payload: Mapping[str, object]) -> FivePeriodFittedModel:
        """从序列化字典重建模型，并重新校验参数和特征契约。"""
        if payload.get("model_name") != self.name:
            raise ValueError("序列化内容不是 five_period 模型")
        config_payload = payload.get("hyperparameters")
        if not isinstance(config_payload, Mapping):
            raise ValueError("序列化内容缺少 hyperparameters")
        config = FivePeriodConfig(**dict(config_payload))
        config.validate()
        fitted = FivePeriodFittedModel(
            weights=tuple(float(value) for value in payload["weights"]),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            feature_version=str(payload["feature_version"]),
            trained_through=str(payload["trained_through"]),
            config=config,
            effective_step_size=float(payload["effective_step_size"]),
            n_training_rows=int(payload["n_training_rows"]),
            iterations=int(payload["iterations"]),
            converged=bool(payload["converged"]),
            final_objective=float(payload["final_objective"]),
            implementation_version=str(payload.get("implementation_version", IMPLEMENTATION_VERSION)),
        )
        self._validate_fitted(fitted)
        return fitted

    def _resolve_step_size(self, x_values: np.ndarray) -> float:
        """未显式配置步长时，按二次目标的 Lipschitz 常数选择稳定步长。"""
        if self.config.step_size is not None:
            return self.config.step_size
        hessian = 2.0 * (x_values.T @ x_values) / len(x_values)
        hessian += 2.0 * self.config.ridge * np.eye(x_values.shape[1])
        largest_eigenvalue = float(np.linalg.eigvalsh(hessian).max())
        return 1.0 / largest_eigenvalue if largest_eigenvalue > 1e-15 else 1.0

    def _gradient(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        weights: np.ndarray,
        uniform: np.ndarray,
    ) -> np.ndarray:
        """计算规范目标函数对权重的解析梯度。"""
        residual = x_values @ weights - y_values
        return 2.0 * (x_values.T @ residual) / len(x_values) + 2.0 * self.config.ridge * (weights - uniform)

    def _objective(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        weights: np.ndarray,
        uniform: np.ndarray,
    ) -> float:
        """计算最终目标值，用于诊断和复现训练状态。"""
        residual = x_values @ weights - y_values
        return float(np.mean(residual**2) + self.config.ridge * np.sum((weights - uniform) ** 2))

    @staticmethod
    def _validate_training_features(
        training_features: pd.DataFrame,
        trained_through: pd.Timestamp | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """确保模型只接收完整的五周期特征和非负监督目标。"""
        required = {"target_quantity", *CURRENT_MEAN_FEATURES}
        missing = required - set(training_features.columns)
        if missing:
            raise ValueError(f"训练特征缺少字段: {sorted(missing)}")
        if training_features.empty:
            raise ValueError("训练特征不能为空")
        # 直接调用核心模型时也拒绝未来标签，不能只依赖外层 Backtester 防泄漏。
        if trained_through is not None and "date" in training_features.columns:
            dates = pd.to_datetime(training_features["date"], errors="coerce").dt.normalize()
            if dates.isna().any() or (dates > trained_through).any():
                raise ValueError("FivePeriod 训练特征不能包含 trained_through 之后的日期")
        x_values = training_features.loc[:, CURRENT_MEAN_FEATURES].to_numpy(dtype=float)
        y_values = training_features["target_quantity"].to_numpy(dtype=float)
        if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            raise ValueError("训练特征和目标值必须有限")
        if (x_values < 0).any() or (y_values < 0).any():
            raise ValueError("训练特征和目标值不能为负")
        return x_values, y_values

    @staticmethod
    def _extract_feature_row(
        feature_row: Mapping[str, float] | pd.Series,
        feature_names: tuple[str, ...],
    ) -> np.ndarray:
        """按已保存的特征顺序提取数值，并检查预测输入完整性。"""
        missing = [name for name in feature_names if name not in feature_row]
        if missing:
            raise ValueError(f"预测特征缺少字段: {missing}")
        values = np.asarray([feature_row[name] for name in feature_names], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("预测特征必须是有限的非负数")
        return values

    @staticmethod
    def _validate_fitted(fitted: FivePeriodFittedModel) -> None:
        """检查加载或调用的拟合状态仍符合五周期模型契约。"""
        weights = np.asarray(fitted.weights, dtype=float)
        if fitted.feature_names != CURRENT_MEAN_FEATURES:
            raise ValueError("FivePeriod 的特征名称或顺序不匹配")
        if fitted.feature_version != FEATURE_VERSION:
            raise ValueError("FivePeriod 的 feature_version 不受支持")
        if weights.shape != (len(CURRENT_MEAN_FEATURES),) or not np.isfinite(weights).all():
            raise ValueError("FivePeriod 权重数量或数值非法")
        if (weights < -1e-12).any() or not np.isclose(weights.sum(), 1.0, atol=1e-10):
            raise ValueError("FivePeriod 权重必须位于概率单纯形")
        if fitted.effective_step_size <= 0 or fitted.n_training_rows < 1:
            raise ValueError("FivePeriod 拟合元信息非法")
