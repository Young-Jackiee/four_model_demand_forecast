"""规范定义的 Direct10（当前滚动均值 + 年度同期滚动均值）模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from demand_forecast.features.definitions import FEATURE_VERSION, feature_names_for
from demand_forecast.models.five_period import project_to_simplex


IMPLEMENTATION_VERSION = "v1"
MODEL_NAME = "direct10"
DIRECT10_FEATURES = feature_names_for("direct10")


class Direct10TrainingUnavailableError(ValueError):
    """当没有满足 455 日历史要求的有效 Direct10 训练行时抛出。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Direct10 当前不可训练: {reason}")


@dataclass(frozen=True)
class Direct10Config:
    """投影梯度下降配置；ridge 数值是 V1 实现决策，并非规范给定网格。"""

    ridge: float = 1e-3
    max_iter: int = 100_000
    # 与 FivePeriod 保持相同的可复现收敛阈值，避免生产重训在合理近似解上失败。
    tol: float = 1e-6
    step_size: float | None = None

    def validate(self) -> None:
        """拒绝会破坏优化目标或确定性的配置。"""
        if not np.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge 必须是有限非负数")
        if self.max_iter < 1:
            raise ValueError("max_iter 至少为 1")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol 必须是有限正数")
        if self.step_size is not None and (not np.isfinite(self.step_size) or self.step_size <= 0.0):
            raise ValueError("step_size 必须是有限正数或 None")


@dataclass(frozen=True)
class Direct10FittedModel:
    """完整训练期拟合后的不可变模型状态。"""

    weights: tuple[float, ...]
    feature_names: tuple[str, ...]
    feature_version: str
    trained_through: str
    config: Direct10Config
    effective_step_size: float
    n_training_rows: int
    iterations: int
    converged: bool
    final_objective: float
    implementation_version: str = IMPLEMENTATION_VERSION


class Direct10Model:
    """用十个均值特征进行单阶段、约束线性回归的 Direct10。"""

    name = MODEL_NAME

    def __init__(self, config: Direct10Config | None = None) -> None:
        self.config = config or Direct10Config()
        self.config.validate()

    def fit(self, training_features: pd.DataFrame, trained_through: str) -> Direct10FittedModel:
        """仅拟合 FeatureBuilder 已筛选出的有效监督训练行。"""
        values, target = self._validate_training_features(training_features, trained_through)
        trained_date = pd.Timestamp(trained_through).normalize()
        center = np.full(len(DIRECT10_FEATURES), 1.0 / len(DIRECT10_FEATURES))
        step_size = self._resolve_step_size(values)
        weights = center.copy()
        converged = False
        iterations = self.config.max_iter

        for iteration in range(1, self.config.max_iter + 1):
            gradient = self._gradient(values, target, weights, center)
            updated = project_to_simplex(weights - step_size * gradient)
            if np.max(np.abs(updated - weights)) <= self.config.tol:
                weights = updated
                converged = True
                iterations = iteration
                break
            weights = updated

        weights = project_to_simplex(weights)
        fitted = Direct10FittedModel(
            weights=tuple(float(value) for value in weights),
            feature_names=DIRECT10_FEATURES,
            feature_version=FEATURE_VERSION,
            trained_through=trained_date.strftime("%Y-%m-%d"),
            config=self.config,
            effective_step_size=float(step_size),
            n_training_rows=len(values),
            iterations=iterations,
            converged=converged,
            final_objective=float(self._objective(values, target, weights, center)),
        )
        self._validate_fitted(fitted)
        return fitted

    def predict_one(self, fitted: Direct10FittedModel, feature_row: Mapping[str, float] | pd.Series) -> float:
        """按保存的名称和顺序读取十个特征，防止列位置错配。"""
        self._validate_fitted(fitted)
        values = self._extract_feature_row(feature_row, fitted.feature_names)
        return max(0.0, float(np.dot(np.asarray(fitted.weights), values)))

    def serialize(self, fitted: Direct10FittedModel) -> dict[str, object]:
        """输出 JSON 可保存的参数、特征契约与优化诊断。"""
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

    def deserialize(self, payload: Mapping[str, object]) -> Direct10FittedModel:
        """从 JSON 友好字典恢复模型，并重新验证单纯形约束。"""
        if payload.get("model_name") != self.name:
            raise ValueError("序列化内容不是 direct10 模型")
        config_payload = payload.get("hyperparameters")
        if not isinstance(config_payload, Mapping):
            raise ValueError("序列化内容缺少 hyperparameters")
        config = Direct10Config(**dict(config_payload))
        config.validate()
        fitted = Direct10FittedModel(
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

    def _resolve_step_size(self, values: np.ndarray) -> float:
        """从二次目标的 Hessian 上界推导稳定步长；可由配置覆盖。"""
        if self.config.step_size is not None:
            return self.config.step_size
        hessian = 2.0 * (values.T @ values) / len(values)
        hessian += 2.0 * self.config.ridge * np.eye(values.shape[1])
        largest_eigenvalue = float(np.linalg.eigvalsh(hessian).max())
        return 1.0 / largest_eigenvalue if largest_eigenvalue > 1e-15 else 1.0

    def _gradient(self, values: np.ndarray, target: np.ndarray, weights: np.ndarray, center: np.ndarray) -> np.ndarray:
        """计算 mean squared error + ridge||w-0.1||² 的解析梯度。"""
        residual = values @ weights - target
        return 2.0 * (values.T @ residual) / len(values) + 2.0 * self.config.ridge * (weights - center)

    def _objective(self, values: np.ndarray, target: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
        """计算训练后的规范目标值，供诊断和复现。"""
        residual = values @ weights - target
        return float(np.mean(residual**2) + self.config.ridge * np.sum((weights - center) ** 2))

    @staticmethod
    def _validate_training_features(training_features: pd.DataFrame, trained_through: str) -> tuple[np.ndarray, np.ndarray]:
        """拒绝空表、未来训练行、缺列、NaN 与负销量，绝不隐式补值。"""
        required = {"target_quantity", *DIRECT10_FEATURES}
        missing = required - set(training_features.columns)
        if missing:
            raise ValueError(f"Direct10 训练特征缺少字段: {sorted(missing)}")
        if training_features.empty:
            raise Direct10TrainingUnavailableError("no_available_direct10_training_features")
        if "date" in training_features.columns:
            dates = pd.to_datetime(training_features["date"], errors="coerce").dt.normalize()
            trained_date = pd.Timestamp(trained_through).normalize()
            if dates.isna().any() or (dates > trained_date).any():
                raise ValueError("Direct10 训练特征不能包含 trained_through 之后的日期")
        values = training_features.loc[:, DIRECT10_FEATURES].to_numpy(dtype=float)
        target = training_features["target_quantity"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            raise ValueError("Direct10 训练特征和目标值必须有限")
        if (values < 0.0).any() or (target < 0.0).any():
            raise ValueError("Direct10 训练特征和目标值不能为负")
        return values, target

    @staticmethod
    def _extract_feature_row(feature_row: Mapping[str, float] | pd.Series, feature_names: tuple[str, ...]) -> np.ndarray:
        """通过名称提取，允许输入映射任意排序但拒绝缺失特征。"""
        missing = [name for name in feature_names if name not in feature_row]
        if missing:
            raise ValueError(f"Direct10 预测特征缺少字段: {missing}")
        values = np.asarray([feature_row[name] for name in feature_names], dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("Direct10 预测特征必须是有限非负数")
        return values

    @staticmethod
    def _validate_fitted(fitted: Direct10FittedModel) -> None:
        """检查载入模型仍是十维、版本匹配的概率单纯形回归。"""
        if not isinstance(fitted, Direct10FittedModel):
            raise ValueError("Direct10 fitted model 类型非法")
        fitted.config.validate()
        weights = np.asarray(fitted.weights, dtype=float)
        if fitted.feature_names != DIRECT10_FEATURES or fitted.feature_version != FEATURE_VERSION:
            raise ValueError("Direct10 特征名称、顺序或版本不匹配")
        if weights.shape != (len(DIRECT10_FEATURES),) or not np.isfinite(weights).all():
            raise ValueError("Direct10 权重数量或数值非法")
        if (weights < -1e-12).any() or not np.isclose(weights.sum(), 1.0, atol=1e-10):
            raise ValueError("Direct10 权重必须位于概率单纯形")
        if fitted.effective_step_size <= 0.0 or fitted.n_training_rows < 1:
            raise ValueError("Direct10 拟合元信息非法")
        if not np.isfinite(fitted.final_objective) or fitted.final_objective < 0.0:
            raise ValueError("Direct10 最终目标值非法")
        if pd.isna(pd.Timestamp(fitted.trained_through)):
            raise ValueError("Direct10 trained_through 非法")
