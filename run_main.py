"""一键运行当前阶段的数据流水线。

直接在 PyCharm 运行本文件即可，无需填写命令行参数。
"""

from pathlib import Path

import pandas as pd

from demand_forecast.backtesting import (
    BacktestSplit,
    Backtester,
    FormalBacktestRunner,
    results_to_metrics_frame,
    ModelSelector,
    write_selection_results,
)
from demand_forecast.data.daily_series import DailySeriesBuilder, write_daily_sales
from demand_forecast.data.synthetic import SyntheticConfig, generate_source_data, write_source_data
from demand_forecast.features.builder import FeatureBuilder
from demand_forecast.forecast_models import (
    Direct10ForecastModel,
    FivePeriodForecastModel,
    HurdleForecastModel,
    TSBForecastModel,
)
from demand_forecast.production import (
    ProductionPipeline,
    ProductionPublisher,
    ProductionTrainer,
    write_production_run_results,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "configs" / "synthetic_dataset.json"
SOURCE_DIR = PROJECT_ROOT / "data" / "synthetic" / "source"
DAILY_SALES_FILE = PROJECT_ROOT / "data" / "synthetic" / "derived" / "daily_sales.csv"
SELECTION_RESULTS_FILE = PROJECT_ROOT / "data" / "synthetic" / "derived" / "selection_results.json"
PRODUCTION_DIR = PROJECT_ROOT / "data" / "synthetic" / "production"
PRODUCTION_RESULTS_FILE = PROJECT_ROOT / "data" / "synthetic" / "derived" / "production_run_results.json"


def main() -> None:
    """按固定顺序生成数据、检查特征并运行已实现模型的正式回测。"""
    # 配置统一管理日期和随机种子，避免运行时依赖人工输入。
    config = SyntheticConfig.from_json(CONFIG_FILE)

    # 第一步：模拟商品主数据、订单明细和不可观测日期例外。
    sku_master, raw_sales, exceptions = generate_source_data(config)
    write_source_data(sku_master, raw_sales, exceptions, config, SOURCE_DIR)

    # 第二步：订单明细聚合、补齐自然日，并保留不可观测标记。
    builder = DailySeriesBuilder(
        history_start=str(config.history_start.date()),
        actual_end=str(config.actual_end.date()),
    )
    daily_sales = builder.build(sku_master, raw_sales, exceptions)
    write_daily_sales(daily_sales, DAILY_SALES_FILE)

    # 第三步：特征暂不落盘，只在内存中验证训练期是否可用。
    feature_builder = FeatureBuilder()
    feature_summary: dict[str, tuple[int, int]] = {}
    for feature_set in ("five_period", "hurdle", "direct10"):
        result = feature_builder.build_historical(
            daily_sales,
            feature_set=feature_set,
            end_date=str(config.train_end.date()),
        )
        feature_summary[feature_set] = (len(result.features), len(result.unavailable))

    # 第四步：Backtester 是正式时间切分唯一入口；测试实际值不会传给模型训练或预测。
    split = BacktestSplit(
        train_start=config.history_start,
        train_end=config.train_end,
        test_start=config.test_start,
        test_end=config.actual_end,
        expected_test_days=52,
    )
    runner = FormalBacktestRunner(Backtester(split))
    # factory 保证每个 SKU × 模型拥有独立 adapter、模型对象、fitted state 与预测历史。
    results = runner.run(
        daily_sales,
        (FivePeriodForecastModel, TSBForecastModel, HurdleForecastModel, Direct10ForecastModel),
    )
    metrics_frame = results_to_metrics_frame(results)
    selections = ModelSelector().select_many(results)
    write_selection_results(selections, SELECTION_RESULTS_FILE)

    # 第五步：production_train_end 只能来自上层配置，绝不使用 daily_sales 的 max(date)。
    forecast_horizon = len(pd.date_range(config.forecast_start, config.forecast_end, freq="D"))
    production_pipeline = ProductionPipeline(ProductionTrainer(), ProductionPublisher(PRODUCTION_DIR))
    metadata_by_sku_model = {
        (result.sku, result.model_name): result.fitted_metadata
        for result in results
        if result.status == "completed" and result.fitted_metadata is not None
    }
    production_results = []
    for selection in selections:
        sku_daily = daily_sales.loc[daily_sales["sku"] == selection.sku].copy()
        winner_metadata = (
            metadata_by_sku_model.get((selection.sku, selection.winner_model))
            if selection.winner_model is not None
            else None
        )
        production_results.append(
            production_pipeline.run(
                sku_daily,
                selection,
                production_train_end=config.production_train_end,
                forecast_horizon=forecast_horizon,
                selection_model_metadata=winner_metadata,
            )
        )
    write_production_run_results(production_results, PRODUCTION_RESULTS_FILE)

    print("数据流水线运行完成。")
    print(f"SKU 数量: {len(sku_master)}")
    print(f"原始订单行数: {len(raw_sales)}")
    print(f"标准日序列行数: {len(daily_sales)}")
    print(f"源数据目录: {SOURCE_DIR}")
    print(f"标准数据文件: {DAILY_SALES_FILE}")
    print(f"选模结果文件: {SELECTION_RESULTS_FILE}")
    print(f"生产 artifact 目录: {PRODUCTION_DIR}")
    print(f"生产运行结果文件: {PRODUCTION_RESULTS_FILE}")
    print("训练期特征检查:")
    for feature_set, (available, unavailable) in feature_summary.items():
        print(f"  {feature_set}: 可用 {available} 行，不可用 {unavailable} 行")
    _print_backtest_summary(metrics_frame)
    _print_selection_summary(selections)
    _print_production_summary(production_results)


def _print_backtest_summary(metrics_frame) -> None:
    """按 SKU 展示四模型统一成绩；结果仅保留内存，ModelSelector 后续再消费。"""
    print("四模型正式回测:")
    for sku, sku_rows in metrics_frame.groupby("sku", sort=True):
        print(f"  {sku}:")
        for row in sku_rows.itertuples(index=False):
            if row.status == "completed":
                print(
                    f"    {row.model_name}: completed, MAE={row.mae:.4f}, "
                    f"Bias={row.cumulative_bias:.4f}, 预测天数={row.forecast_days}, 评价天数={row.evaluated_days}"
                )
            else:
                print(f"    {row.model_name}: unavailable, 原因={row.unavailable_reason}")


def _print_selection_summary(selections) -> None:
    """打印纯回测决策结果；不在此处训练 winner 或生成未来预测。"""
    print("ModelSelector:")
    for selection in selections:
        if selection.status == "selected":
            print(f"  {selection.sku}: winner_model={selection.winner_model}")
        else:
            print(f"  {selection.sku}: {selection.status}")


def _print_production_summary(production_results) -> None:
    """展示是否真正发布，不把 selected winner 与 deployed model 混为一谈。"""
    print("Production:")
    for result in production_results:
        message = f"  {result.sku}: status={result.status}, selected={result.selected_model}, deployed={result.deployed_model}"
        if result.reason_code:
            message += f", reason={result.reason_code}"
        print(message)


if __name__ == "__main__":
    main()
