"""数据契约与输入校验。

模型层只能读取经过本模块校验后的日粒度数据，避免把脏数据问题带进算法。
"""

from __future__ import annotations

import pandas as pd


class DataContractError(ValueError):
    """当输入数据不满足约定字段或业务规则时抛出。"""


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    """检查必填字段，缺字段时给出易定位的错误信息。"""
    missing = required - set(frame.columns)
    if missing:
        raise DataContractError(f"{name} 缺少必填字段: {sorted(missing)}")


def _as_date(values: pd.Series, field: str) -> pd.Series:
    """把日期解析为零点时间；解析失败不允许静默通过。"""
    result = pd.to_datetime(values, errors="coerce").dt.normalize()
    if result.isna().any():
        raise DataContractError(f"字段 {field} 存在无法解析的日期")
    return result


def validate_sku_master(sku_master: pd.DataFrame) -> pd.DataFrame:
    """校验 SKU 主数据，并返回类型统一后的副本。"""
    _require_columns(sku_master, {"sku", "launch_date"}, "sku_master")
    result = sku_master.copy()
    result["sku"] = result["sku"].astype("string").str.strip()
    if result["sku"].isna().any() or (result["sku"] == "").any():
        raise DataContractError("sku_master 的 sku 不能为空")
    if result["sku"].duplicated().any():
        raise DataContractError("sku_master 的 sku 必须唯一")
    result["launch_date"] = _as_date(result["launch_date"], "launch_date")
    return result


def validate_raw_sales(raw_sales: pd.DataFrame) -> pd.DataFrame:
    """校验原始订单行；订单行数量必须为正数。"""
    _require_columns(raw_sales, {"transaction_id", "sku", "sold_at", "quantity"}, "raw_sales")
    result = raw_sales.copy()
    result["transaction_id"] = result["transaction_id"].astype("string").str.strip()
    result["sku"] = result["sku"].astype("string").str.strip()
    if result[["transaction_id", "sku"]].isna().any().any() or (result[["transaction_id", "sku"]] == "").any().any():
        raise DataContractError("raw_sales 的 transaction_id 和 sku 不能为空")
    if result["transaction_id"].duplicated().any():
        raise DataContractError("raw_sales 的 transaction_id 不能重复")
    result["sold_at"] = pd.to_datetime(result["sold_at"], errors="coerce")
    if result["sold_at"].isna().any():
        raise DataContractError("raw_sales 的 sold_at 存在无法解析的时间")
    result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce")
    if result["quantity"].isna().any() or (result["quantity"] <= 0).any():
        raise DataContractError("raw_sales 的 quantity 必须为正数")
    return result


def validate_observation_exceptions(exceptions: pd.DataFrame) -> pd.DataFrame:
    """校验不可观测例外；本文件只允许记录 false，避免成为冗余日历。"""
    _require_columns(exceptions, {"sku", "date", "is_observed", "reason"}, "observation_exceptions")
    result = exceptions.copy()
    result["sku"] = result["sku"].astype("string").str.strip()
    result["date"] = _as_date(result["date"], "date")
    result["is_observed"] = result["is_observed"].astype("boolean")
    result["reason"] = result["reason"].astype("string").str.strip()
    if result["sku"].isna().any() or (result["sku"] == "").any():
        raise DataContractError("observation_exceptions 的 sku 不能为空")
    if result["is_observed"].isna().any() or result["is_observed"].any():
        raise DataContractError("observation_exceptions 只能记录 is_observed=false 的例外")
    if result["reason"].isna().any() or (result["reason"] == "").any():
        raise DataContractError("observation_exceptions 的 reason 不能为空")
    if result.duplicated(["sku", "date"]).any():
        raise DataContractError("observation_exceptions 的 sku + date 不能重复")
    return result


def validate_daily_sales(daily_sales: pd.DataFrame) -> pd.DataFrame:
    """校验标准日序列：仅可观测日必须有非负销量。"""
    _require_columns(
        daily_sales,
        {"sku", "date", "quantity", "is_observed", "launch_date", "observation_reason"},
        "daily_sales",
    )
    result = daily_sales.copy()
    result["sku"] = result["sku"].astype("string")
    result["date"] = _as_date(result["date"], "date")
    result["launch_date"] = _as_date(result["launch_date"], "launch_date")
    result["is_observed"] = result["is_observed"].astype("boolean")
    result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce").astype("Float64")
    if result.duplicated(["sku", "date"]).any():
        raise DataContractError("daily_sales 必须保证每个 sku + date 仅一行")
    observed = result["is_observed"]
    if result.loc[observed, "quantity"].isna().any():
        raise DataContractError("is_observed=true 的日期必须有 quantity")
    if (result.loc[observed, "quantity"] < 0).any():
        raise DataContractError("可观测日的 quantity 不能为负")
    if result.loc[~observed, "quantity"].notna().any():
        raise DataContractError("is_observed=false 的日期不能伪造 quantity=0")
    return result
