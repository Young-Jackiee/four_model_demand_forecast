"""规范定义的两段式（Hurdle）日需求预测模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from demand_forecast.data.schemas import validate_daily_sales
from demand_forecast.features.builder import FeatureBuilder
from demand_forecast.features.definitions import FEATURE_VERSION, feature_names_for


IMPLEMENTATION_VERSION = "v1"
MODEL_NAME = "hurdle"
INTERNAL_VALIDATION_MODE = "recursive_no_actual_update"
OBSERVATION_POLICY = "exclude_unobserved_targets_and_history"
HURDLE_FEATURES = feature_names_for("hurdle")


class HurdleTrainingUnavailableError(ValueError):
    """当数据无法满足 Hurdle 数学或递归特征契约时抛出。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Hurdle 当前不可训练: {reason}")


@dataclass(frozen=True)
class HurdleConfig:
    """规范参数网格及可复现的数值优化配置。"""

    lambda_values: tuple[float, ...] = (0.1, 1.0, 10.0, 50.0)
    validation_days: int = 60
    max_iter: int = 100
    tol: float = 1e-8
    newton_damping: float = 1e-10
    max_backtracking: int = 25
    max_linear_predictor: float = 700.0

    def validate(self) -> None:
        """拒绝会改变规范网格含义或破坏数值优化的配置。"""
        if not self.lambda_values or len(set(self.lambda_values)) != len(self.lambda_values):
            raise ValueError("lambda_values 必须非空且不重复")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.lambda_values):
            raise ValueError("lambda_values 必须是有限正数")
        if self.validation_days < 1 or self.max_iter < 1 or self.max_backtracking < 1:
            raise ValueError("validation_days、max_iter 和 max_backtracking 必须为正整数")
        if self.tol <= 0.0 or self.newton_damping < 0.0:
            raise ValueError("tol 必须为正，newton_damping 不能为负")
        if not np.isfinite(self.max_linear_predictor) or self.max_linear_predictor <= 0.0:
            raise ValueError("max_linear_predictor 必须为有限正数")


@dataclass(frozen=True)
class FeatureStandardizer:
    """只保存训练期统计量的轻量 StandardScaler。"""

    mean: tuple[float, ...]
    scale: tuple[float, ...]


@dataclass(frozen=True)
class LogisticFitDiagnostics:
    """发生模型优化诊断，便于识别未收敛而非伪造成功。"""

    iterations: int
    converged: bool
    final_objective: float


@dataclass(frozen=True)
class HurdleParameters:
    """内部候选模型参数；用于 16 组 lambda 的递归验证。"""

    standardizer: FeatureStandardizer
    beta0: float
    beta: tuple[float, ...]
    gamma0: float
    gamma: tuple[float, ...]
    occurrence_diagnostics: LogisticFitDiagnostics
    quantity_objective: float


@dataclass(frozen=True)
class HurdlePrediction:
    """单日两段式输出，保留 p、q 供递归和解释使用。"""

    occurrence_probability: float
    conditional_quantity: float
    prediction: float


@dataclass(frozen=True)
class HurdleForecastStep:
    """带日期的 Hurdle 递归预测步骤。"""

    date: pd.Timestamp
    result: HurdlePrediction


@dataclass(frozen=True)
class HurdleFittedModel:
    """最终完整训练期重训后的不可变 Hurdle 模型。"""

    sku: str
    lambda_p: float
    lambda_q: float
    feature_names: tuple[str, ...]
    feature_version: str
    parameters: HurdleParameters
    trained_through: str
    config: HurdleConfig
    validation_mae: float
    validation_observed_days: int
    n_full_training_rows: int
    n_positive_training_rows: int
    internal_validation_mode: str = INTERNAL_VALIDATION_MODE
    observation_policy: str = OBSERVATION_POLICY
    implementation_version: str = IMPLEMENTATION_VERSION


class HurdleModel:
    """用发生概率和条件销量乘积预测日需求，并拥有双状态递归语义。"""

    name = MODEL_NAME

    def __init__(
        self,
        config: HurdleConfig | None = None,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        self.config = config or HurdleConfig()
        self.config.validate()
        self.feature_builder = feature_builder or FeatureBuilder()

    def fit(self, train_series: pd.DataFrame, trained_through: str) -> HurdleFittedModel:
        """在训练末 60 个自然日递归选参后，使用完整训练期重新训练。"""
        series, sku, trained_date = self._prepare_train_series(train_series, trained_through)
        validation_start = trained_date - pd.Timedelta(days=self.config.validation_days - 1)
        internal_train = series.loc[series["date"] < validation_start].copy()
        internal_validation = series.loc[series["date"] >= validation_start].copy()
        if internal_validation.loc[internal_validation["is_observed"].astype(bool)].empty:
            raise HurdleTrainingUnavailableError("no_observed_internal_validation_targets")

        internal_features = self._build_training_features(internal_train)
        self._validate_training_applicability(internal_features, "internal")
        # 16 组 lambda 共用同一份严格只由 internal_train 拟合的 scaler。
        internal_standardizer = self._fit_standardizer(internal_features)
        lambda_p, lambda_q, validation_mae, validation_count = self._select_lambdas(
            sku,
            internal_features,
            internal_standardizer,
            internal_train,
            internal_validation,
        )

        full_features = self._build_training_features(series)
        self._validate_training_applicability(full_features, "full")
        full_standardizer = self._fit_standardizer(full_features)
        final_parameters, positive_count = self._fit_parameters(
            full_features,
            full_standardizer,
            lambda_p,
            lambda_q,
        )
        if not final_parameters.occurrence_diagnostics.converged:
            raise HurdleTrainingUnavailableError("final_occurrence_optimizer_not_converged")

        fitted = HurdleFittedModel(
            sku=sku,
            lambda_p=lambda_p,
            lambda_q=lambda_q,
            feature_names=HURDLE_FEATURES,
            feature_version=FEATURE_VERSION,
            parameters=final_parameters,
            trained_through=trained_date.strftime("%Y-%m-%d"),
            config=self.config,
            validation_mae=validation_mae,
            validation_observed_days=validation_count,
            n_full_training_rows=len(full_features),
            n_positive_training_rows=positive_count,
        )
        self._validate_fitted(fitted)
        return fitted

    def predict_one(
        self,
        fitted: HurdleFittedModel,
        feature_row: Mapping[str, float] | pd.Series,
    ) -> HurdlePrediction:
        """按已保存特征顺序生成 p、q 和 p×q，不允许列位置猜测。"""
        self._validate_fitted(fitted)
        values = self._extract_feature_row(feature_row, fitted.feature_names)
        return self._predict_from_parameters(fitted.parameters, values)

    def forecast_many(
        self,
        fitted: HurdleFittedModel,
        train_series: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex,
    ) -> list[HurdleForecastStep]:
        """递归追加 ŷ 与 p；整个过程只使用训练历史和先前预测。"""
        self._validate_fitted(fitted)
        dates = pd.DatetimeIndex(forecast_dates).normalize()
        expected_dates = pd.date_range(
            pd.Timestamp(fitted.trained_through).normalize() + pd.Timedelta(days=1),
            periods=len(dates),
            freq="D",
        )
        if len(dates) != len(expected_dates) or list(dates) != list(expected_dates):
            raise ValueError("Hurdle forecast_dates 必须从 trained_through 次日开始且连续")
        history = self._to_forecast_history(train_series, fitted)
        steps: list[HurdleForecastStep] = []
        for date in dates:
            feature_result = self.feature_builder.build_next(history, date, feature_set="hurdle")
            if not feature_result.is_available:
                raise HurdleTrainingUnavailableError(
                    f"hurdle_recursive_feature_unavailable:{feature_result.unavailable_reason}"
                )
            result = self.predict_one(fitted, feature_result.values or {})
            steps.append(HurdleForecastStep(date=pd.Timestamp(date).normalize(), result=result))
            # 规范要求：quantity 追加 ŷ，occurrence 追加 p，绝不能追加 1[ŷ>0]。
            history.loc[len(history)] = {
                "date": pd.Timestamp(date).normalize(),
                "quantity": result.prediction,
                "occurrence": result.occurrence_probability,
            }
        return steps

    def serialize(self, fitted: HurdleFittedModel) -> dict[str, object]:
        """输出 JSON 友好字典，保存 scaler、参数、特征顺序与验证诊断。"""
        self._validate_fitted(fitted)
        parameters = fitted.parameters
        return {
            "model_name": self.name,
            "sku": fitted.sku,
            "trained_through": fitted.trained_through,
            "feature_version": fitted.feature_version,
            "feature_names": list(fitted.feature_names),
            "selected_hyperparameters": {"lambda_p": fitted.lambda_p, "lambda_q": fitted.lambda_q},
            "config": {
                "lambda_values": list(fitted.config.lambda_values),
                "validation_days": fitted.config.validation_days,
                "max_iter": fitted.config.max_iter,
                "tol": fitted.config.tol,
                "newton_damping": fitted.config.newton_damping,
                "max_backtracking": fitted.config.max_backtracking,
                "max_linear_predictor": fitted.config.max_linear_predictor,
            },
            "parameters": {
                "scaler_mean": list(parameters.standardizer.mean),
                "scaler_scale": list(parameters.standardizer.scale),
                "beta0": parameters.beta0,
                "beta": list(parameters.beta),
                "gamma0": parameters.gamma0,
                "gamma": list(parameters.gamma),
            },
            "internal_validation": {
                "mae": fitted.validation_mae,
                "observed_days": fitted.validation_observed_days,
                "mode": fitted.internal_validation_mode,
            },
            "training": {
                "n_full_training_rows": fitted.n_full_training_rows,
                "n_positive_training_rows": fitted.n_positive_training_rows,
                "occurrence_iterations": parameters.occurrence_diagnostics.iterations,
                "occurrence_converged": parameters.occurrence_diagnostics.converged,
                "occurrence_final_objective": parameters.occurrence_diagnostics.final_objective,
                "quantity_final_objective": parameters.quantity_objective,
            },
            "observation_policy": fitted.observation_policy,
            "implementation_version": fitted.implementation_version,
        }

    def deserialize(self, payload: Mapping[str, object]) -> HurdleFittedModel:
        """从纯 Python 字典恢复模型，并检查所有持久化契约。"""
        if payload.get("model_name") != self.name:
            raise ValueError("序列化内容不是 hurdle 模型")
        config_payload = payload.get("config")
        selected = payload.get("selected_hyperparameters")
        parameters_payload = payload.get("parameters")
        validation_payload = payload.get("internal_validation")
        training_payload = payload.get("training")
        if not all(isinstance(value, Mapping) for value in (config_payload, selected, parameters_payload, validation_payload, training_payload)):
            raise ValueError("Hurdle 序列化内容缺少必要字段")
        config = HurdleConfig(
            lambda_values=tuple(float(value) for value in config_payload["lambda_values"]),
            validation_days=int(config_payload["validation_days"]),
            max_iter=int(config_payload["max_iter"]),
            tol=float(config_payload["tol"]),
            newton_damping=float(config_payload["newton_damping"]),
            max_backtracking=int(config_payload["max_backtracking"]),
            max_linear_predictor=float(config_payload["max_linear_predictor"]),
        )
        config.validate()
        parameters = HurdleParameters(
            standardizer=FeatureStandardizer(
                mean=tuple(float(value) for value in parameters_payload["scaler_mean"]),
                scale=tuple(float(value) for value in parameters_payload["scaler_scale"]),
            ),
            beta0=float(parameters_payload["beta0"]),
            beta=tuple(float(value) for value in parameters_payload["beta"]),
            gamma0=float(parameters_payload["gamma0"]),
            gamma=tuple(float(value) for value in parameters_payload["gamma"]),
            occurrence_diagnostics=LogisticFitDiagnostics(
                iterations=int(training_payload["occurrence_iterations"]),
                converged=bool(training_payload["occurrence_converged"]),
                final_objective=float(training_payload["occurrence_final_objective"]),
            ),
            quantity_objective=float(training_payload["quantity_final_objective"]),
        )
        fitted = HurdleFittedModel(
            sku=str(payload["sku"]),
            lambda_p=float(selected["lambda_p"]),
            lambda_q=float(selected["lambda_q"]),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            feature_version=str(payload["feature_version"]),
            parameters=parameters,
            trained_through=str(payload["trained_through"]),
            config=config,
            validation_mae=float(validation_payload["mae"]),
            validation_observed_days=int(validation_payload["observed_days"]),
            n_full_training_rows=int(training_payload["n_full_training_rows"]),
            n_positive_training_rows=int(training_payload["n_positive_training_rows"]),
            internal_validation_mode=str(validation_payload["mode"]),
            observation_policy=str(payload["observation_policy"]),
            implementation_version=str(payload.get("implementation_version", IMPLEMENTATION_VERSION)),
        )
        self._validate_fitted(fitted)
        return fitted

    def _select_lambdas(
        self,
        sku: str,
        internal_features: pd.DataFrame,
        standardizer: FeatureStandardizer,
        internal_train: pd.DataFrame,
        internal_validation: pd.DataFrame,
    ) -> tuple[float, float, float, int]:
        """用 16 个参数对的最终销量递归 MAE 联合选参，平局保留网格首个。"""
        best: tuple[float, float, float, int] | None = None
        failure_reasons: set[str] = set()
        validation_dates = pd.DatetimeIndex(internal_validation["date"])
        for lambda_p in self.config.lambda_values:
            for lambda_q in self.config.lambda_values:
                try:
                    parameters, _ = self._fit_parameters(internal_features, standardizer, lambda_p, lambda_q)
                    if not parameters.occurrence_diagnostics.converged:
                        continue
                    candidate = self._make_candidate_fitted(
                        sku,
                        parameters,
                        lambda_p,
                        lambda_q,
                        pd.Timestamp(internal_train["date"].max()),
                    )
                    steps = self.forecast_many(candidate, internal_train, validation_dates)
                    mae, observed_days = self._validation_mae(internal_validation, steps)
                except HurdleTrainingUnavailableError as error:
                    failure_reasons.add(error.reason)
                    continue
                if best is None or mae < best[2]:
                    best = (float(lambda_p), float(lambda_q), mae, observed_days)
        if best is None:
            if len(failure_reasons) == 1:
                raise HurdleTrainingUnavailableError(failure_reasons.pop())
            raise HurdleTrainingUnavailableError("no_valid_hurdle_lambda_candidate")
        return best

    def _make_candidate_fitted(
        self,
        sku: str,
        parameters: HurdleParameters,
        lambda_p: float,
        lambda_q: float,
        trained_through: pd.Timestamp,
    ) -> HurdleFittedModel:
        """构造仅供内部验证使用的冻结参数，不触碰 validation actual。"""
        return HurdleFittedModel(
            sku=sku,
            lambda_p=lambda_p,
            lambda_q=lambda_q,
            feature_names=HURDLE_FEATURES,
            feature_version=FEATURE_VERSION,
            parameters=parameters,
            trained_through=trained_through.strftime("%Y-%m-%d"),
            config=self.config,
            validation_mae=0.0,
            validation_observed_days=1,
            n_full_training_rows=1,
            n_positive_training_rows=1,
        )

    def _build_training_features(self, series: pd.DataFrame) -> pd.DataFrame:
        """只保留有可观测 target 且完整 R/F 历史窗口的 Hurdle 训练行。"""
        result = self.feature_builder.build_historical(series, feature_set="hurdle")
        if result.features.empty:
            raise HurdleTrainingUnavailableError("no_available_hurdle_training_features")
        return result.features

    @staticmethod
    def _validate_training_applicability(features: pd.DataFrame, stage: str) -> None:
        """在网格搜索前报告单类别或无正销量，而不是只给出笼统候选失败。"""
        target = features["target_quantity"].to_numpy(dtype=float)
        occurrence = target > 0.0
        if occurrence.min() == occurrence.max():
            raise HurdleTrainingUnavailableError(f"single_class_occurrence_{stage}_training")
        if not occurrence.any():
            raise HurdleTrainingUnavailableError(f"no_positive_quantity_{stage}_training")

    def _fit_standardizer(self, features: pd.DataFrame) -> FeatureStandardizer:
        """仅从本次训练特征 fit 均值和尺度；零方差列用 1 防止除零。"""
        values = self._training_matrix(features)
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return FeatureStandardizer(tuple(float(value) for value in mean), tuple(float(value) for value in scale))

    def _fit_parameters(
        self,
        features: pd.DataFrame,
        standardizer: FeatureStandardizer,
        lambda_p: float,
        lambda_q: float,
    ) -> tuple[HurdleParameters, int]:
        """在共享 scaler 后分别拟合 occurrence 与 positive-only quantity 两段。"""
        self._validate_lambda(lambda_p, "lambda_p")
        self._validate_lambda(lambda_q, "lambda_q")
        values = self._training_matrix(features)
        target = features["target_quantity"].to_numpy(dtype=float)
        standardized = self._transform(values, standardizer)
        occurrence = (target > 0.0).astype(float)
        if occurrence.min() == occurrence.max():
            raise HurdleTrainingUnavailableError("single_class_occurrence_training")
        positive = target > 0.0
        if not positive.any():
            raise HurdleTrainingUnavailableError("no_positive_quantity_training")
        beta0, beta, logistic_diagnostics = self._fit_logistic(standardized, occurrence, lambda_p)
        gamma0, gamma, quantity_objective = self._fit_quantity(
            standardized[positive],
            np.log1p(target[positive]),
            lambda_q,
        )
        return (
            HurdleParameters(
                standardizer=standardizer,
                beta0=beta0,
                beta=tuple(float(value) for value in beta),
                gamma0=gamma0,
                gamma=tuple(float(value) for value in gamma),
                occurrence_diagnostics=logistic_diagnostics,
                quantity_objective=quantity_objective,
            ),
            int(positive.sum()),
        )

    def _fit_logistic(
        self,
        values: np.ndarray,
        target: np.ndarray,
        lambda_p: float,
    ) -> tuple[float, np.ndarray, LogisticFitDiagnostics]:
        """用阻尼 Newton 法精确最小化规范定义的 sum-BCE 加 ridge 目标。"""
        positive_rate = float(target.mean())
        intercept = float(np.log(positive_rate / (1.0 - positive_rate)))
        coefficients = np.zeros(values.shape[1], dtype=float)
        objective = self._logistic_objective(values, target, intercept, coefficients, lambda_p)
        for iteration in range(1, self.config.max_iter + 1):
            score = intercept + values @ coefficients
            probability = self._sigmoid(score)
            residual = probability - target
            gradient = np.concatenate(([residual.sum()], values.T @ residual + lambda_p * coefficients))
            if np.max(np.abs(gradient)) <= self.config.tol:
                return intercept, coefficients, LogisticFitDiagnostics(iteration, True, objective)
            weight = probability * (1.0 - probability)
            hessian = np.empty((values.shape[1] + 1, values.shape[1] + 1), dtype=float)
            hessian[0, 0] = weight.sum()
            hessian[0, 1:] = values.T @ weight
            hessian[1:, 0] = hessian[0, 1:]
            hessian[1:, 1:] = values.T @ (values * weight[:, None]) + lambda_p * np.eye(values.shape[1])
            hessian += self.config.newton_damping * np.eye(hessian.shape[0])
            try:
                direction = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                return intercept, coefficients, LogisticFitDiagnostics(iteration, False, objective)
            step = 1.0
            accepted = False
            for _ in range(self.config.max_backtracking):
                candidate_intercept = intercept - step * direction[0]
                candidate_coefficients = coefficients - step * direction[1:]
                candidate_objective = self._logistic_objective(
                    values, target, candidate_intercept, candidate_coefficients, lambda_p
                )
                if candidate_objective <= objective:
                    intercept, coefficients, objective = candidate_intercept, candidate_coefficients, candidate_objective
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                return intercept, coefficients, LogisticFitDiagnostics(iteration, False, objective)
        return intercept, coefficients, LogisticFitDiagnostics(self.config.max_iter, False, objective)

    @staticmethod
    def _fit_quantity(values: np.ndarray, target: np.ndarray, lambda_q: float) -> tuple[float, np.ndarray, float]:
        """用未惩罚截距、惩罚系数的 ridge 正规方程拟合 log1p 正销量。"""
        design = np.column_stack((np.ones(len(values)), values))
        penalty = np.diag(np.concatenate(([0.0], np.full(values.shape[1], lambda_q))))
        try:
            parameters = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        except np.linalg.LinAlgError as error:
            raise HurdleTrainingUnavailableError("quantity_ridge_solver_failed") from error
        residual = target - design @ parameters
        objective = float(np.sum(residual**2) + lambda_q * np.sum(parameters[1:] ** 2))
        return float(parameters[0]), parameters[1:], objective

    def _predict_from_parameters(self, parameters: HurdleParameters, values: np.ndarray) -> HurdlePrediction:
        """稳定地恢复 p、q 与最终日销量，避免 expm1 溢出。"""
        standardized = self._transform(values.reshape(1, -1), parameters.standardizer)[0]
        probability = float(self._sigmoid(np.asarray([parameters.beta0 + np.dot(parameters.beta, standardized)]))[0])
        linear_quantity = parameters.gamma0 + np.dot(parameters.gamma, standardized)
        if linear_quantity <= 0.0:
            quantity = 0.0
        else:
            safe_linear = min(float(linear_quantity), self.config.max_linear_predictor)
            quantity = float(np.expm1(safe_linear))
        prediction = float(probability * quantity)
        if not np.isfinite(prediction):
            raise HurdleTrainingUnavailableError("non_finite_hurdle_prediction")
        return HurdlePrediction(probability, quantity, prediction)

    def _validation_mae(
        self,
        validation: pd.DataFrame,
        steps: list[HurdleForecastStep],
    ) -> tuple[float, int]:
        """预测完成后才读取 validation actual，并屏蔽不可观测 target。"""
        prediction_by_date = {step.date: step.result.prediction for step in steps}
        observed = validation.loc[validation["is_observed"].astype(bool), ["date", "quantity"]]
        if observed.empty:
            raise HurdleTrainingUnavailableError("no_observed_internal_validation_targets")
        actual = observed["quantity"].to_numpy(dtype=float)
        predicted = np.asarray([prediction_by_date[pd.Timestamp(date)] for date in observed["date"]], dtype=float)
        return float(np.mean(np.abs(predicted - actual))), len(observed)

    def _to_forecast_history(self, train_series: pd.DataFrame, fitted: HurdleFittedModel) -> pd.DataFrame:
        """真实历史用 0/1 occurrence；预测历史将在 forecast_many 中追加概率 p。"""
        series = validate_daily_sales(train_series)
        sku_values = series["sku"].dropna().unique()
        trained_date = pd.Timestamp(fitted.trained_through).normalize()
        if len(sku_values) != 1 or str(sku_values[0]) != fitted.sku:
            raise ValueError("Hurdle forecast 的训练 SKU 与 fitted model 不匹配")
        if (series["date"] > trained_date).any():
            raise ValueError("Hurdle forecast 只能接收截至 trained_through 的训练历史")
        # 上层按 SKU 切片后索引可能不是 0..n-1；递归 loc[len(history)] 必须追加而非覆盖。
        history = series[["date", "quantity", "is_observed"]].copy().reset_index(drop=True)
        history["occurrence"] = pd.Series(pd.NA, index=history.index, dtype="Float64")
        observed = history["is_observed"].astype(bool)
        history.loc[observed, "occurrence"] = (history.loc[observed, "quantity"] > 0.0).astype(float)
        return history[["date", "quantity", "occurrence"]].copy()

    def _prepare_train_series(
        self,
        train_series: pd.DataFrame,
        trained_through: str,
    ) -> tuple[pd.DataFrame, str, pd.Timestamp]:
        """Hurdle.fit 明确拒绝测试行，防止依赖模型内部过滤造成侥幸安全。"""
        series = validate_daily_sales(train_series)
        sku_values = series["sku"].dropna().unique()
        if len(sku_values) != 1 or series["sku"].isna().any() or (series["sku"].str.strip() == "").any():
            raise ValueError("Hurdle 每次 fit 只能接收一个非空 SKU")
        if not series["date"].is_monotonic_increasing:
            raise ValueError("Hurdle 训练日序列必须按日期升序")
        trained_date = pd.Timestamp(trained_through).normalize()
        if pd.isna(trained_date):
            raise ValueError("trained_through 必须是可解析日期")
        if (series["date"] > trained_date).any():
            raise ValueError("Hurdle.fit 不能接收 trained_through 之后的数据")
        if series.empty:
            raise HurdleTrainingUnavailableError("no_training_history")
        expected_dates = pd.date_range(series["date"].min(), trained_date, freq="D")
        if len(series) != len(expected_dates) or set(series["date"]) != set(expected_dates):
            raise ValueError("Hurdle 训练日序列必须连续覆盖至 trained_through")
        return series.copy(), str(sku_values[0]), trained_date

    @staticmethod
    def _training_matrix(features: pd.DataFrame) -> np.ndarray:
        """提取固定顺序的 12 列训练特征，并拒绝 NaN、负销量等脏输入。"""
        required = {"target_quantity", *HURDLE_FEATURES}
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"Hurdle 训练特征缺少字段: {sorted(missing)}")
        values = features.loc[:, HURDLE_FEATURES].to_numpy(dtype=float)
        target = features["target_quantity"].to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all() or not np.isfinite(target).all():
            raise HurdleTrainingUnavailableError("non_finite_or_empty_training_features")
        if (target < 0.0).any():
            raise ValueError("Hurdle target_quantity 不能为负")
        return values

    @staticmethod
    def _transform(values: np.ndarray, standardizer: FeatureStandardizer) -> np.ndarray:
        """按保存的训练期统计量变换特征，禁止预测期重新 fit。"""
        mean = np.asarray(standardizer.mean, dtype=float)
        scale = np.asarray(standardizer.scale, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(HURDLE_FEATURES):
            raise ValueError("Hurdle 特征维度不匹配")
        if mean.shape != (len(HURDLE_FEATURES),) or scale.shape != mean.shape:
            raise ValueError("Hurdle scaler 维度不匹配")
        if not np.isfinite(values).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0.0).any():
            raise ValueError("Hurdle scaler 或特征数值非法")
        return (values - mean) / scale

    @staticmethod
    def _extract_feature_row(feature_row: Mapping[str, float] | pd.Series, feature_names: tuple[str, ...]) -> np.ndarray:
        """按持久化名称取特征，阻止训练和加载后因列顺序变化静默错位。"""
        missing = [name for name in feature_names if name not in feature_row]
        if missing:
            raise ValueError(f"Hurdle 预测特征缺少字段: {missing}")
        values = np.asarray([feature_row[name] for name in feature_names], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Hurdle 预测特征必须有限")
        return values

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        """分段 sigmoid 避免大负数输入时 exp 溢出。"""
        result = np.empty_like(values, dtype=float)
        positive = values >= 0.0
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponent = np.exp(values[~positive])
        result[~positive] = exponent / (1.0 + exponent)
        return result

    @staticmethod
    def _logistic_objective(
        values: np.ndarray,
        target: np.ndarray,
        intercept: float,
        coefficients: np.ndarray,
        lambda_p: float,
    ) -> float:
        """规范的 sum-BCE + lambda/2 × 系数平方；截距不在 penalty 内。"""
        score = intercept + values @ coefficients
        return float(np.sum(np.logaddexp(0.0, score) - target * score) + (lambda_p / 2.0) * np.sum(coefficients**2))

    @staticmethod
    def _validate_lambda(value: float, name: str) -> None:
        """候选 lambda 必须来自有限正数域。"""
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须是有限正数")

    @classmethod
    def _validate_fitted(cls, fitted: HurdleFittedModel) -> None:
        """验证持久化模型仍符合 12 特征、非负数量及版本契约。"""
        if not isinstance(fitted, HurdleFittedModel) or not fitted.sku.strip():
            raise ValueError("Hurdle fitted model 的 sku 非法")
        fitted.config.validate()
        cls._validate_lambda(fitted.lambda_p, "lambda_p")
        cls._validate_lambda(fitted.lambda_q, "lambda_q")
        if fitted.lambda_p not in fitted.config.lambda_values or fitted.lambda_q not in fitted.config.lambda_values:
            raise ValueError("Hurdle 选中的 lambda 不在配置网格中")
        if fitted.feature_names != HURDLE_FEATURES or fitted.feature_version != FEATURE_VERSION:
            raise ValueError("Hurdle 特征名称、顺序或版本不匹配")
        parameters = fitted.parameters
        cls._transform(np.zeros((1, len(HURDLE_FEATURES))), parameters.standardizer)
        for value in (parameters.beta0, parameters.gamma0, parameters.quantity_objective, parameters.occurrence_diagnostics.final_objective):
            if not np.isfinite(value):
                raise ValueError("Hurdle 参数必须有限")
        if not np.isfinite(np.asarray(parameters.beta)).all() or not np.isfinite(np.asarray(parameters.gamma)).all():
            raise ValueError("Hurdle 系数必须有限")
        if len(parameters.beta) != len(HURDLE_FEATURES) or len(parameters.gamma) != len(HURDLE_FEATURES):
            raise ValueError("Hurdle 系数维度不匹配")
        if parameters.occurrence_diagnostics.iterations < 1 or not parameters.occurrence_diagnostics.converged:
            raise ValueError("Hurdle occurrence 模型未收敛")
        if pd.isna(pd.Timestamp(fitted.trained_through)):
            raise ValueError("Hurdle trained_through 非法")
        if not np.isfinite(fitted.validation_mae) or fitted.validation_mae < 0.0 or fitted.validation_observed_days < 1:
            raise ValueError("Hurdle 内部验证诊断非法")
        if fitted.n_full_training_rows < 1 or fitted.n_positive_training_rows < 1:
            raise ValueError("Hurdle 训练行统计非法")
        if fitted.internal_validation_mode != INTERNAL_VALIDATION_MODE or fitted.observation_policy != OBSERVATION_POLICY:
            raise ValueError("Hurdle 数据或验证策略不受支持")
