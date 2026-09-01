"""TSB（Teunter-Syntetos-Babai）间歇性需求预测模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from demand_forecast.data.schemas import validate_daily_sales


IMPLEMENTATION_VERSION = "v1"
MODEL_NAME = "tsb"
OBSERVATION_POLICY = "skip_unobserved"


class TSBTrainingUnavailableError(ValueError):
    """当某个 SKU 的历史不足以按既定规则训练 TSB 时抛出。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"TSB 当前不可训练: {reason}")


@dataclass(frozen=True)
class TSBConfig:
    """TSB 的确定性调参和初始化配置。"""

    smoothing_values: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.30)
    validation_days: int = 90
    initialization_observed_days: int = 90

    def validate(self) -> None:
        """拒绝会产生无效状态或无法复现网格搜索的配置。"""
        if not self.smoothing_values:
            raise ValueError("smoothing_values 不能为空")
        if len(set(self.smoothing_values)) != len(self.smoothing_values):
            raise ValueError("smoothing_values 不能包含重复值")
        if any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in self.smoothing_values):
            raise ValueError("平滑参数必须是 (0, 1] 内的有限数")
        if self.validation_days < 1:
            raise ValueError("validation_days 至少为 1")
        if self.initialization_observed_days < 1:
            raise ValueError("initialization_observed_days 至少为 1")


@dataclass(frozen=True)
class TSBState:
    """某一时点的发生率 p 与发生需求时的规模 z。"""

    occurrence_probability: float
    demand_size: float


@dataclass(frozen=True)
class TSBFittedModel:
    """一次训练后的不可变 TSB 状态，可直接用于未来多日预测。"""

    sku: str
    alpha: float
    beta: float
    state: TSBState
    trained_through: str
    config: TSBConfig
    validation_mae: float
    validation_observed_days: int
    n_observed_training_days: int
    initialization_observed_days_used: int
    implementation_version: str = IMPLEMENTATION_VERSION


class TSBModel:
    """使用发生率与正需求规模分解预测间歇性日需求。"""

    name = MODEL_NAME

    def __init__(self, config: TSBConfig | None = None) -> None:
        self.config = config or TSBConfig()
        self.config.validate()

    def fit(self, daily_series: pd.DataFrame, trained_through: str) -> TSBFittedModel:
        """在训练末 90 个自然日选参后，用全部训练历史重新得到最终状态。"""
        series, sku, trained_date = self._prepare_training_series(daily_series, trained_through)
        validation_start = trained_date - pd.Timedelta(days=self.config.validation_days - 1)
        core = series.loc[series["date"] < validation_start]
        validation = series.loc[series["date"] >= validation_start]
        core_observed = core.loc[core["is_observed"]]
        validation_observed = validation.loc[validation["is_observed"]]

        # 无法建立历史状态或无法计算验证误差时，不用臆造参数继续训练。
        if core_observed.empty:
            raise TSBTrainingUnavailableError("insufficient_history_for_tsb_validation")
        if validation_observed.empty:
            raise TSBTrainingUnavailableError("no_observed_target_in_tsb_validation")

        alpha, beta, validation_mae = self._select_hyperparameters(core_observed, validation_observed)
        all_observed = series.loc[series["is_observed"]]
        final_state, initialization_days_used = self._fit_state(all_observed, alpha, beta)

        fitted = TSBFittedModel(
            sku=sku,
            alpha=alpha,
            beta=beta,
            state=final_state,
            trained_through=trained_date.strftime("%Y-%m-%d"),
            config=self.config,
            validation_mae=validation_mae,
            validation_observed_days=len(validation_observed),
            n_observed_training_days=len(all_observed),
            initialization_observed_days_used=initialization_days_used,
        )
        self._validate_fitted(fitted)
        return fitted

    def predict_one(self, fitted: TSBFittedModel) -> float:
        """使用冻结状态预测下一天；预测本身绝不读取或修改真实销量。"""
        self._validate_fitted(fitted)
        return max(0.0, fitted.state.occurrence_probability * fitted.state.demand_size)

    def forecast_many(self, fitted: TSBFittedModel, horizon: int) -> list[float]:
        """没有新实际销量时，TSB 状态不变，故未来预测为同一常数。"""
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValueError("horizon 必须是非负整数")
        return [self.predict_one(fitted)] * horizon

    def update_state(self, state: TSBState, quantity: float, alpha: float, beta: float) -> TSBState:
        """收到一个可观测日的真实销量后，返回新状态而不修改传入状态。"""
        self._validate_state(state)
        self._validate_smoothing(alpha, "alpha")
        self._validate_smoothing(beta, "beta")
        observed_quantity = self._validate_quantity(quantity)
        occurrence = 1.0 if observed_quantity > 0.0 else 0.0
        probability = state.occurrence_probability + beta * (occurrence - state.occurrence_probability)
        demand_size = state.demand_size
        if occurrence == 1.0:
            demand_size = demand_size + alpha * (observed_quantity - demand_size)
        # 合法参数下 p 天然在 [0, 1]；裁剪只用于消除极小浮点误差。
        return TSBState(float(np.clip(probability, 0.0, 1.0)), float(max(0.0, demand_size)))

    def serialize(self, fitted: TSBFittedModel) -> dict[str, object]:
        """输出可直接 JSON 保存的模型状态和必要训练诊断。"""
        self._validate_fitted(fitted)
        return {
            "model_name": self.name,
            "sku": fitted.sku,
            "selected_hyperparameters": {"alpha": fitted.alpha, "beta": fitted.beta},
            "state": {
                "occurrence_probability": fitted.state.occurrence_probability,
                "demand_size": fitted.state.demand_size,
            },
            "trained_through": fitted.trained_through,
            "config": {
                "smoothing_values": list(fitted.config.smoothing_values),
                "validation_days": fitted.config.validation_days,
                "initialization_observed_days": fitted.config.initialization_observed_days,
            },
            "validation_mae": fitted.validation_mae,
            "validation_observed_days": fitted.validation_observed_days,
            "n_observed_training_days": fitted.n_observed_training_days,
            "initialization_observed_days_used": fitted.initialization_observed_days_used,
            "observation_policy": OBSERVATION_POLICY,
            "implementation_version": fitted.implementation_version,
        }

    def deserialize(self, payload: Mapping[str, object]) -> TSBFittedModel:
        """从保存字典重建冻结状态，并再次校验数据契约。"""
        if payload.get("model_name") != self.name:
            raise ValueError("序列化内容不是 tsb 模型")
        if payload.get("observation_policy") != OBSERVATION_POLICY:
            raise ValueError("TSB 的不可观测日期处理策略不匹配")
        config_payload = payload.get("config")
        hyperparameters = payload.get("selected_hyperparameters")
        state_payload = payload.get("state")
        if not isinstance(config_payload, Mapping) or not isinstance(hyperparameters, Mapping) or not isinstance(state_payload, Mapping):
            raise ValueError("TSB 序列化内容缺少 config、selected_hyperparameters 或 state")
        config = TSBConfig(
            smoothing_values=tuple(float(value) for value in config_payload["smoothing_values"]),
            validation_days=int(config_payload["validation_days"]),
            initialization_observed_days=int(config_payload["initialization_observed_days"]),
        )
        config.validate()
        fitted = TSBFittedModel(
            sku=str(payload["sku"]),
            alpha=float(hyperparameters["alpha"]),
            beta=float(hyperparameters["beta"]),
            state=TSBState(
                occurrence_probability=float(state_payload["occurrence_probability"]),
                demand_size=float(state_payload["demand_size"]),
            ),
            trained_through=str(payload["trained_through"]),
            config=config,
            validation_mae=float(payload["validation_mae"]),
            validation_observed_days=int(payload["validation_observed_days"]),
            n_observed_training_days=int(payload["n_observed_training_days"]),
            initialization_observed_days_used=int(payload["initialization_observed_days_used"]),
            implementation_version=str(payload.get("implementation_version", IMPLEMENTATION_VERSION)),
        )
        self._validate_fitted(fitted)
        return fitted

    def _select_hyperparameters(
        self,
        core_observed: pd.DataFrame,
        validation_observed: pd.DataFrame,
    ) -> tuple[float, float, float]:
        """逐组参数在验证日执行“先预测、后用真实值更新”的在线模拟。"""
        best: tuple[float, float, float] | None = None
        for alpha in self.config.smoothing_values:
            for beta in self.config.smoothing_values:
                state, _ = self._fit_state(core_observed, alpha, beta)
                absolute_errors: list[float] = []
                for quantity in validation_observed["quantity"].to_numpy(dtype=float):
                    absolute_errors.append(abs(state.occurrence_probability * state.demand_size - quantity))
                    state = self.update_state(state, quantity, alpha, beta)
                mae = float(np.mean(absolute_errors))
                # 使用严格小于：网格顺序即为可复现的平局处理规则。
                if best is None or mae < best[2]:
                    best = (float(alpha), float(beta), mae)
        assert best is not None  # 调用方已保证验证期存在可观测标签。
        return best

    def _fit_state(self, observed_series: pd.DataFrame, alpha: float, beta: float) -> tuple[TSBState, int]:
        """用一个时间段的可观测日初始化并顺序更新状态。"""
        if observed_series.empty:
            raise TSBTrainingUnavailableError("no_observed_history")
        quantities = observed_series["quantity"].to_numpy(dtype=float)
        initialization_days_used = min(self.config.initialization_observed_days, len(quantities))
        initialization_quantities = quantities[:initialization_days_used]
        first_positive = quantities[quantities > 0.0]
        # 历史中还没有正销量时 z=0；后续真实正销量会自然将其更新为正。
        initial_size = float(first_positive[0]) if len(first_positive) else 0.0
        state = TSBState(
            occurrence_probability=float(np.mean(initialization_quantities > 0.0)),
            demand_size=initial_size,
        )
        for quantity in quantities[initialization_days_used:]:
            state = self.update_state(state, quantity, alpha, beta)
        return state, initialization_days_used

    @staticmethod
    def _prepare_training_series(
        daily_series: pd.DataFrame,
        trained_through: str,
    ) -> tuple[pd.DataFrame, str, pd.Timestamp]:
        """校验单 SKU 日序列，截取训练截止日，保留不可观测日期供边界划分。"""
        series = validate_daily_sales(daily_series)
        if series["sku"].isna().any() or (series["sku"].str.strip() == "").any():
            raise ValueError("TSB 日序列的 sku 不能为空")
        sku_values = series["sku"].dropna().unique()
        if len(sku_values) != 1:
            raise ValueError("TSB 每次 fit 只能接收一个 SKU 的日序列")
        trained_date = pd.Timestamp(trained_through).normalize()
        if pd.isna(trained_date):
            raise ValueError("trained_through 必须是可解析日期")
        if (series["date"] > trained_date).any():
            raise ValueError("TSB.fit 不能接收 trained_through 之后的数据")
        series = series.sort_values("date", ignore_index=True)
        if series.empty:
            raise TSBTrainingUnavailableError("no_history_on_or_before_trained_through")
        return series, str(sku_values[0]), trained_date

    @staticmethod
    def _validate_quantity(quantity: float) -> float:
        """TSB 只接受可观测日的有限非负销量。"""
        value = float(quantity)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("quantity 必须是有限的非负数")
        return value

    @staticmethod
    def _validate_smoothing(value: float, name: str) -> None:
        """单独校验 alpha 或 beta，便于公开状态更新函数安全复用。"""
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} 必须位于 (0, 1]")

    @classmethod
    def _validate_state(cls, state: TSBState) -> None:
        """保证概率和需求规模的基本物理含义。"""
        if not isinstance(state, TSBState):
            raise ValueError("state 必须是 TSBState")
        if not np.isfinite(state.occurrence_probability) or not 0.0 <= state.occurrence_probability <= 1.0:
            raise ValueError("occurrence_probability 必须位于 [0, 1]")
        cls._validate_quantity(state.demand_size)

    @classmethod
    def _validate_fitted(cls, fitted: TSBFittedModel) -> None:
        """检查加载后的状态、诊断信息和参数组合仍满足 TSB 契约。"""
        if not isinstance(fitted, TSBFittedModel) or not fitted.sku.strip():
            raise ValueError("TSB 拟合模型的 sku 非法")
        fitted.config.validate()
        cls._validate_smoothing(fitted.alpha, "alpha")
        cls._validate_smoothing(fitted.beta, "beta")
        if fitted.alpha not in fitted.config.smoothing_values or fitted.beta not in fitted.config.smoothing_values:
            raise ValueError("TSB 选中的参数不在配置网格中")
        cls._validate_state(fitted.state)
        if pd.isna(pd.Timestamp(fitted.trained_through)):
            raise ValueError("TSB trained_through 非法")
        if not np.isfinite(fitted.validation_mae) or fitted.validation_mae < 0.0:
            raise ValueError("TSB validation_mae 非法")
        if fitted.validation_observed_days < 1 or fitted.n_observed_training_days < 1:
            raise ValueError("TSB 可观测日统计非法")
        if not 1 <= fitted.initialization_observed_days_used <= fitted.n_observed_training_days:
            raise ValueError("TSB 初始化日统计非法")
