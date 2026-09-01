"""把订单明细转换为模型唯一允许使用的 SKU × 自然日日序列。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from demand_forecast.data.schemas import (
    DataContractError,
    validate_daily_sales,
    validate_observation_exceptions,
    validate_raw_sales,
    validate_sku_master,
)


class DailySeriesBuilder:
    """从源数据构建连续日历，并显式保留不可观测日期。"""

    def __init__(self, history_start: str, actual_end: str) -> None:
        self.history_start = pd.Timestamp(history_start).normalize()
        self.actual_end = pd.Timestamp(actual_end).normalize()
        if self.history_start > self.actual_end:
            raise DataContractError("history_start 不能晚于 actual_end")

    def build(
        self,
        sku_master: pd.DataFrame,
        raw_sales: pd.DataFrame,
        observation_exceptions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """生成完整日历；只有已上市且可观测日期才填充真实零销量。"""
        master = validate_sku_master(sku_master)
        sales = validate_raw_sales(raw_sales)
        exceptions = self._prepare_exceptions(observation_exceptions)
        self._validate_source_relationships(master, sales, exceptions)

        # 订单时间先归到业务自然日，再按 SKU + 日汇总。
        sales["date"] = sales["sold_at"].dt.normalize()
        daily_quantity = (
            sales.groupby(["sku", "date"], as_index=False, sort=True)["quantity"]
            .sum()
            .rename(columns={"quantity": "raw_quantity"})
        )

        # 笛卡尔积确保每个 SKU 在整个历史范围内每天都有一行。
        calendar = pd.MultiIndex.from_product(
            [master["sku"].tolist(), pd.date_range(self.history_start, self.actual_end, freq="D")],
            names=["sku", "date"],
        ).to_frame(index=False)
        result = calendar.merge(master, on="sku", how="left", validate="many_to_one")
        result = result.merge(daily_quantity, on=["sku", "date"], how="left", validate="one_to_one")
        result = result.merge(
            exceptions[["sku", "date", "reason"]],
            on=["sku", "date"],
            how="left",
            validate="one_to_one",
        )

        # 上市前默认不可观测；上市后除例外外默认已观测。
        is_after_launch = result["date"] >= result["launch_date"]
        has_exception = result["reason"].notna()
        result["is_observed"] = (is_after_launch & ~has_exception).astype("boolean")
        result["observation_reason"] = pd.NA
        result.loc[~is_after_launch, "observation_reason"] = "not_launched"
        result.loc[has_exception, "observation_reason"] = result.loc[has_exception, "reason"]

        # 只有可观测日期的无订单记录才代表真实零销量。
        result["quantity"] = result["raw_quantity"].where(result["is_observed"], pd.NA)
        result.loc[result["is_observed"] & result["quantity"].isna(), "quantity"] = 0.0
        result["quantity"] = result["quantity"].astype("Float64")

        result = result[
            ["sku", "date", "quantity", "is_observed", "launch_date", "observation_reason"]
        ].sort_values(["sku", "date"], ignore_index=True)
        return validate_daily_sales(result)

    def _prepare_exceptions(self, exceptions: pd.DataFrame | None) -> pd.DataFrame:
        """没有例外文件时，使用同字段空表，调用方无需写特殊分支。"""
        if exceptions is None:
            exceptions = pd.DataFrame(columns=["sku", "date", "is_observed", "reason"])
        return validate_observation_exceptions(exceptions)

    def _validate_source_relationships(
        self,
        master: pd.DataFrame,
        sales: pd.DataFrame,
        exceptions: pd.DataFrame,
    ) -> None:
        """检查跨表关系，避免把业务矛盾静默加工成看似正常的数据。"""
        known_skus = set(master["sku"])
        unknown_sales = set(sales["sku"]) - known_skus
        unknown_exceptions = set(exceptions["sku"]) - known_skus
        if unknown_sales:
            raise DataContractError(f"raw_sales 出现未登记 SKU: {sorted(unknown_sales)}")
        if unknown_exceptions:
            raise DataContractError(f"observation_exceptions 出现未登记 SKU: {sorted(unknown_exceptions)}")

        sales_dates = sales["sold_at"].dt.normalize()
        if (sales_dates < self.history_start).any() or (sales_dates > self.actual_end).any():
            raise DataContractError("raw_sales 的日期必须位于本次历史范围内")
        if ((exceptions["date"] < self.history_start) | (exceptions["date"] > self.actual_end)).any():
            raise DataContractError("observation_exceptions 的日期必须位于本次历史范围内")

        launch_dates = master.set_index("sku")["launch_date"]
        before_launch = sales_dates < sales["sku"].map(launch_dates)
        if before_launch.any():
            raise DataContractError("raw_sales 不能出现上市日前的订单")

        exception_keys = pd.MultiIndex.from_frame(exceptions[["sku", "date"]])
        sales_keys = pd.MultiIndex.from_arrays([sales["sku"], sales_dates])
        if sales_keys.isin(exception_keys).any():
            raise DataContractError("不可观测日期不能同时存在 raw_sales 订单")


def read_source_files(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取约定文件名，集中管理 CSV 输入位置。"""
    master = pd.read_csv(source_dir / "sku_master.csv")
    sales = pd.read_csv(source_dir / "raw_sales.csv")
    exceptions_path = source_dir / "observation_exceptions.csv"
    exceptions = pd.read_csv(exceptions_path) if exceptions_path.exists() else pd.DataFrame(
        columns=["sku", "date", "is_observed", "reason"]
    )
    return master, sales, exceptions


def write_daily_sales(daily_sales: pd.DataFrame, output_file: Path) -> None:
    """写出稳定列顺序与日期格式，便于版本对比和人工检查。"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output = daily_sales.copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    output["launch_date"] = output["launch_date"].dt.strftime("%Y-%m-%d")
    output.to_csv(output_file, index=False, encoding="utf-8-sig")


def main() -> None:
    """命令行入口：只完成源数据到标准日序列的转换。"""
    parser = argparse.ArgumentParser(description="构建 SKU × 自然日标准销量序列")
    parser.add_argument("--source-dir", type=Path, required=True, help="源数据 CSV 所在目录")
    parser.add_argument("--output-file", type=Path, required=True, help="daily_sales.csv 输出路径")
    parser.add_argument("--history-start", required=True, help="历史开始日期，例如 2025-01-01")
    parser.add_argument("--actual-end", required=True, help="最后一个实际销量日期，例如 2026-08-21")
    args = parser.parse_args()

    master, sales, exceptions = read_source_files(args.source_dir)
    daily_sales = DailySeriesBuilder(args.history_start, args.actual_end).build(master, sales, exceptions)
    write_daily_sales(daily_sales, args.output_file)
    print(f"已生成 {len(daily_sales)} 行标准日序列: {args.output_file}")


if __name__ == "__main__":
    main()
