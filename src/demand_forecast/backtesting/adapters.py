"""把现有模型接入统一 Backtester 的轻量适配层。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd

from demand_forecast.backtesting.contracts import DailyForecast, ModelUnavailableError
from demand_forecast.features.builder import FeatureBuilder
from demand_forecast.models.direct10 import (
    Direct10FittedModel,
    Direct10Model,
    Direct10TrainingUnavailableError,
)
from demand_forecast.models.five_period import FivePeriodFittedModel, FivePeriodModel
from demand_forecast.models.hurdle import HurdleFittedModel, HurdleModel, HurdleTrainingUnavailableError
from demand_forecast.models.tsb import TSBFittedModel, TSBModel, TSBTrainingUnavailableError


class BacktestModelAdapter(Protocol):
    """Backtester 唯一依赖的模型边界；预测日历中没有实际销量字段。"""

    name: str

    def fit(self, train_series: pd.DataFrame, train_end: pd.Timestamp) -> object: ...

    def forecast(
        self, fitted: object, train_series: pd.DataFrame, forecast_dates: pd.DatetimeIndex
    ) -> Sequence[DailyForecast]: ...

    def serialize(self, fitted: object) -> dict[str, object]: ...


class FivePeriodBacktestAdapter:
    """封装五周期模型的特征训练与数量递归预测语义。"""

    name = FivePeriodModel.name

    def __init__(
        self,
        model: FivePeriodModel | None = None,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        self.model = model or FivePeriodModel()
        self.feature_builder = feature_builder or FeatureBuilder()

    def fit(self, train_series: pd.DataFrame, train_end: pd.Timestamp) -> FivePeriodFittedModel:
        """只从训练切片生成五周期特征，绝不将测试 target 放进训练表。"""
        result = self.feature_builder.build_historical(
            train_series,
            feature_set="five_period",
            end_date=train_end,
        )
        if result.features.empty:
            raise ModelUnavailableError("no_available_five_period_training_features")
        return self.model.fit(result.features, trained_through=train_end.strftime("%Y-%m-%d"))

    def forecast(
        self,
        fitted: object,
        train_series: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex,
    ) -> list[DailyForecast]:
        """逐日追加预测销量，再构造下一日滚动特征。"""
        if not isinstance(fitted, FivePeriodFittedModel):
            raise ValueError("FivePeriod adapter 收到错误的 fitted model")
        _validate_forecast_dates(fitted.trained_through, forecast_dates, "FivePeriod")
        sku_values = train_series["sku"].dropna().unique()
        if len(sku_values) != 1:
            raise ValueError("FivePeriod adapter 只能预测一个 SKU")
        sku = str(sku_values[0])
        history = self._to_forecast_history(train_series)
        forecasts: list[DailyForecast] = []
        for date in forecast_dates:
            feature_result = self.feature_builder.build_next(history, date, feature_set="five_period")
            if not feature_result.is_available:
                raise ModelUnavailableError(f"five_period_recursive_feature_unavailable:{feature_result.unavailable_reason}")
            prediction = self.model.predict_one(fitted, feature_result.values or {})
            forecasts.append(DailyForecast(sku=sku, date=date, prediction=prediction))
            # 这是模型预测历史而不是测试期真实数据；FivePeriod 不使用 occurrence，但保留完整历史契约。
            history.loc[len(history)] = {
                "date": date,
                "quantity": prediction,
                "occurrence": float(prediction > 0.0),
            }
        return forecasts

    def serialize(self, fitted: object) -> dict[str, object]:
        """复用核心模型已有的可复现序列化结果。"""
        if not isinstance(fitted, FivePeriodFittedModel):
            raise ValueError("FivePeriod adapter 收到错误的 fitted model")
        return self.model.serialize(fitted)

    @staticmethod
    def _to_forecast_history(train_series: pd.DataFrame) -> pd.DataFrame:
        """未观测训练日保留空值，使 FeatureBuilder 阻止错误补零的递归预测。"""
        return _to_quantity_forecast_history(train_series)


class Direct10BacktestAdapter:
    """封装 Direct10 的十特征训练与递归数量预测语义。"""

    name = Direct10Model.name

    def __init__(
        self,
        model: Direct10Model | None = None,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        self.model = model or Direct10Model()
        self.feature_builder = feature_builder or FeatureBuilder()

    def fit(self, train_series: pd.DataFrame, train_end: pd.Timestamp) -> Direct10FittedModel:
        """FeatureBuilder 是唯一的 455 日 / is_observed 训练行判定来源。"""
        result = self.feature_builder.build_historical(
            train_series,
            feature_set="direct10",
            end_date=train_end,
        )
        if result.features.empty:
            if not result.unavailable.empty and result.unavailable["reason"].str.contains("insufficient_history").all():
                raise ModelUnavailableError("insufficient_direct10_history")
            raise ModelUnavailableError("no_available_direct10_training_features")
        try:
            return self.model.fit(result.features, trained_through=train_end.strftime("%Y-%m-%d"))
        except Direct10TrainingUnavailableError as error:
            raise ModelUnavailableError(error.reason) from error

    def forecast(
        self,
        fitted: object,
        train_series: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex,
    ) -> list[DailyForecast]:
        """每天只追加前一日预测；YOY 特征仍由 FeatureBuilder 回读历史日历。"""
        if not isinstance(fitted, Direct10FittedModel):
            raise ValueError("Direct10 adapter 收到错误的 fitted model")
        dates = pd.DatetimeIndex(forecast_dates).normalize()
        _validate_forecast_dates(fitted.trained_through, dates, "Direct10")
        # 365 天后 YOY 窗口会开始引用预测值，已不再是“历史年度同期”；V1 显式拒绝。
        if len(dates) > 365:
            raise ModelUnavailableError("direct10_forecast_horizon_exceeds_yoy_history")
        sku_values = train_series["sku"].dropna().unique()
        if len(sku_values) != 1:
            raise ValueError("Direct10 adapter 只能预测一个 SKU")
        history = _to_quantity_forecast_history(train_series)
        forecasts: list[DailyForecast] = []
        for date in dates:
            feature_result = self.feature_builder.build_next(history, date, feature_set="direct10")
            if not feature_result.is_available:
                raise ModelUnavailableError(
                    f"direct10_recursive_feature_unavailable:{feature_result.unavailable_reason}"
                )
            prediction = self.model.predict_one(fitted, feature_result.values or {})
            forecasts.append(DailyForecast(sku=str(sku_values[0]), date=date, prediction=prediction))
            # occurrence 对 Direct10 不参与计算；仍写入合法值以满足共享 history 数据契约。
            history.loc[len(history)] = {"date": date, "quantity": prediction, "occurrence": float(prediction > 0.0)}
        return forecasts

    def serialize(self, fitted: object) -> dict[str, object]:
        """保存 Direct10 的十个权重和优化配置。"""
        if not isinstance(fitted, Direct10FittedModel):
            raise ValueError("Direct10 adapter 收到错误的 fitted model")
        return self.model.serialize(fitted)


def _to_quantity_forecast_history(train_series: pd.DataFrame) -> pd.DataFrame:
    """把实际日序列转成递归历史；重置索引确保 loc[len] 始终是追加。"""
    # groupby 或布尔筛选会保留原索引；若不重置，递归预测可能覆盖一条真实历史。
    history = train_series[["date", "quantity", "is_observed"]].copy().reset_index(drop=True)
    history["occurrence"] = pd.Series(pd.NA, index=history.index, dtype="Float64")
    observed = history["is_observed"].astype(bool)
    history.loc[observed, "occurrence"] = (history.loc[observed, "quantity"] > 0).astype(float)
    return history[["date", "quantity", "occurrence"]].copy()


def _validate_forecast_dates(
    trained_through: str,
    forecast_dates: pd.DatetimeIndex,
    model_name: str,
) -> pd.DatetimeIndex:
    """所有 adapter 的预测日历都必须紧接训练截止日，避免单独调用时错位。"""
    dates = pd.DatetimeIndex(forecast_dates).normalize()
    expected = pd.date_range(
        pd.Timestamp(trained_through).normalize() + pd.Timedelta(days=1),
        periods=len(dates),
        freq="D",
    )
    if list(dates) != list(expected):
        raise ValueError(f"{model_name} forecast_dates 必须从 trained_through 次日开始且连续")
    return dates


class TSBBacktestAdapter:
    """封装 TSB 的冻结状态、多步常数预测语义。"""

    name = TSBModel.name

    def __init__(self, model: TSBModel | None = None) -> None:
        self.model = model or TSBModel()

    def fit(self, train_series: pd.DataFrame, train_end: pd.Timestamp) -> TSBFittedModel:
        """TSB 内部验证只能看到 Backtester 已截取的训练期。"""
        try:
            return self.model.fit(train_series, trained_through=train_end.strftime("%Y-%m-%d"))
        except TSBTrainingUnavailableError as error:
            raise ModelUnavailableError(error.reason) from error

    def forecast(
        self,
        fitted: object,
        train_series: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex,
    ) -> list[DailyForecast]:
        """无新实际值时 p、z 均冻结，测试期每一天使用同一预测。"""
        if not isinstance(fitted, TSBFittedModel):
            raise ValueError("TSB adapter 收到错误的 fitted model")
        dates = _validate_forecast_dates(fitted.trained_through, forecast_dates, "TSB")
        predictions = self.model.forecast_many(fitted, len(dates))
        return [
            DailyForecast(
                sku=fitted.sku,
                date=date,
                prediction=prediction,
                components={
                    "occurrence_probability": fitted.state.occurrence_probability,
                    "demand_size": fitted.state.demand_size,
                },
            )
            for date, prediction in zip(dates, predictions, strict=True)
        ]

    def serialize(self, fitted: object) -> dict[str, object]:
        """复用 TSB 保存的参数网格、状态和内部验证诊断。"""
        if not isinstance(fitted, TSBFittedModel):
            raise ValueError("TSB adapter 收到错误的 fitted model")
        return self.model.serialize(fitted)


class HurdleBacktestAdapter:
    """封装 Hurdle 的内部调参、双状态递归和 p/q 诊断输出。"""

    name = HurdleModel.name

    def __init__(self, model: HurdleModel | None = None) -> None:
        self.model = model or HurdleModel()

    def fit(self, train_series: pd.DataFrame, train_end: pd.Timestamp) -> HurdleFittedModel:
        """Backtester 只传训练切片，Hurdle 内部 60 天验证不可能看到正式测试。"""
        try:
            return self.model.fit(train_series, trained_through=train_end.strftime("%Y-%m-%d"))
        except HurdleTrainingUnavailableError as error:
            raise ModelUnavailableError(error.reason) from error

    def forecast(
        self,
        fitted: object,
        train_series: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex,
    ) -> list[DailyForecast]:
        """将 Hurdle 的 p、q、ŷ 转换为统一预测记录。"""
        if not isinstance(fitted, HurdleFittedModel):
            raise ValueError("Hurdle adapter 收到错误的 fitted model")
        try:
            steps = self.model.forecast_many(fitted, train_series, forecast_dates)
        except HurdleTrainingUnavailableError as error:
            raise ModelUnavailableError(error.reason) from error
        return [
            DailyForecast(
                sku=fitted.sku,
                date=step.date,
                prediction=step.result.prediction,
                components={
                    "p": step.result.occurrence_probability,
                    "q": step.result.conditional_quantity,
                },
            )
            for step in steps
        ]

    def serialize(self, fitted: object) -> dict[str, object]:
        """保存 Hurdle 的 scaler、系数、lambda 与内部验证诊断。"""
        if not isinstance(fitted, HurdleFittedModel):
            raise ValueError("Hurdle adapter 收到错误的 fitted model")
        return self.model.serialize(fitted)
