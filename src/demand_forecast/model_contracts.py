"""文档规定的四模型公共软件接口。"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd

from demand_forecast.backtesting.contracts import DailyForecast


# V1 使用 DataFrame 承载 DailySeries；字段约束统一由 validate_daily_sales 校验。
DailySeries = pd.DataFrame


class FittedModel(Protocol):
    """所有已训练模型必须暴露的最小可追溯信息。"""

    sku: str
    trained_through: str


class ForecastModel(Protocol):
    """技术规范第 9 节定义的统一模型边界。"""

    name: str

    def fit(self, series: DailySeries, train_end: date) -> FittedModel:
        """只使用 train_end 当日及以前的实际日序列训练。"""

    def predict(self, fitted: FittedModel, horizon: int) -> list[DailyForecast]:
        """从训练截止日下一天开始，完整递归预测 horizon 天。"""

    def serialize(self, fitted: FittedModel) -> dict[str, object]:
        """保存参数、特征版本和推理所需的最小历史。"""
