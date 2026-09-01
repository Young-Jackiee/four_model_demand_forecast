"""生成可复现的模拟源数据。

本模块只生成主数据、订单明细和不可观测例外；日粒度数据必须由 DailySeriesBuilder 产生。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    """模拟数据的最小配置；日期边界与技术设计规范保持一致。"""

    history_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    actual_end: pd.Timestamp
    production_train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    random_seed: int
    skus_per_scenario: int

    @classmethod
    def from_json(cls, config_file: Path) -> "SyntheticConfig":
        """读取 JSON，避免把随机种子和日期硬编码到生成逻辑。"""
        content = json.loads(config_file.read_text(encoding="utf-8"))
        config = cls(
            history_start=pd.Timestamp(content["history_start"]).normalize(),
            train_end=pd.Timestamp(content["train_end"]).normalize(),
            test_start=pd.Timestamp(content["test_start"]).normalize(),
            actual_end=pd.Timestamp(content["actual_end"]).normalize(),
            production_train_end=pd.Timestamp(content["production_train_end"]).normalize(),
            forecast_start=pd.Timestamp(content["forecast_start"]).normalize(),
            forecast_end=pd.Timestamp(content["forecast_end"]).normalize(),
            random_seed=int(content["random_seed"]),
            skus_per_scenario=int(content["skus_per_scenario"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """确认历史、测试和未来预测窗口首尾连续且长度符合规范。"""
        if self.history_start > self.train_end or self.train_end >= self.actual_end:
            raise ValueError("必须满足 history_start <= train_end < actual_end")
        if self.test_start != self.train_end + pd.Timedelta(days=1):
            raise ValueError("test_start 必须紧接 train_end")
        if self.actual_end < self.test_start:
            raise ValueError("actual_end 不能早于 test_start")
        if self.production_train_end > self.actual_end:
            raise ValueError("production_train_end 不能晚于 actual_end")
        if self.production_train_end < self.test_start:
            raise ValueError("production_train_end 不能早于正式测试开始日")
        if self.forecast_start != self.production_train_end + pd.Timedelta(days=1):
            raise ValueError("forecast_start 必须紧接 production_train_end")
        if self.forecast_end < self.forecast_start:
            raise ValueError("forecast_end 不能早于 forecast_start")
        if self.skus_per_scenario < 1:
            raise ValueError("skus_per_scenario 至少为 1")


SCENARIOS = ("stable", "intermittent", "weekly", "annual", "new_launch", "observation_gap")


def generate_source_data(config: SyntheticConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成三张源表；同一配置和种子始终生成完全相同的结果。"""
    rng = np.random.default_rng(config.random_seed)
    master = _make_sku_master(config)
    exceptions = _make_observation_exceptions(master, config, rng)
    exception_keys = set(zip(exceptions["sku"], pd.to_datetime(exceptions["date"])))
    raw_sales = _make_raw_sales(master, config, exception_keys, rng)
    return master, raw_sales, exceptions


def _make_sku_master(config: SyntheticConfig) -> pd.DataFrame:
    """创建少量目的明确的 SKU，不追求模拟庞大商品目录。"""
    rows: list[dict[str, str]] = []
    for scenario in SCENARIOS:
        for index in range(1, config.skus_per_scenario + 1):
            launch_date = config.history_start
            if scenario == "new_launch":
                # 两个新品分别覆盖较长和较短的上市历史。
                launch_date = pd.Timestamp("2025-07-01") if index % 2 else pd.Timestamp("2026-02-15")
            rows.append(
                {
                    "sku": f"SIM-{scenario.upper()}-{index:02d}",
                    "launch_date": launch_date.strftime("%Y-%m-%d"),
                    # 该字段仅用于模拟数据审阅；标准化和模型不会依赖它。
                    "scenario": scenario,
                }
            )
    return pd.DataFrame(rows).sort_values("sku", ignore_index=True)


def _make_observation_exceptions(
    master: pd.DataFrame,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """只为 observation_gap SKU 写少量例外，模拟缺货和数据同步中断。"""
    rows: list[dict[str, object]] = []
    observed_skus = master.loc[master["scenario"] == "observation_gap", "sku"].tolist()
    all_dates = pd.date_range(config.history_start + pd.Timedelta(days=120), config.actual_end - pd.Timedelta(days=30), freq="D")
    for index, sku in enumerate(observed_skus):
        reason = "stockout" if index % 2 == 0 else "source_missing"
        # 少量离散日期足以测试边界，不人为构造复杂的可用性平台。
        chosen = rng.choice(all_dates.to_numpy(), size=6, replace=False)
        for value in chosen:
            rows.append(
                {
                    "sku": sku,
                    "date": pd.Timestamp(value).strftime("%Y-%m-%d"),
                    "is_observed": False,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=["sku", "date", "is_observed", "reason"]).sort_values(
        ["sku", "date"], ignore_index=True
    )


def _make_raw_sales(
    master: pd.DataFrame,
    config: SyntheticConfig,
    exception_keys: set[tuple[str, pd.Timestamp]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """根据日需求拆成若干订单行，故意保留同日多行这一真实输入特征。"""
    rows: list[dict[str, object]] = []
    transaction_number = 1
    for item in master.itertuples(index=False):
        launch_date = pd.Timestamp(item.launch_date)
        for current_date in pd.date_range(launch_date, config.actual_end, freq="D"):
            if (item.sku, current_date) in exception_keys:
                # 该日潜在需求不写入订单表，Builder 会依据例外标记为不可观测。
                continue
            quantity = _sample_daily_quantity(item.scenario, current_date, launch_date, rng)
            for line_quantity in _split_to_order_lines(quantity, rng):
                sold_at = current_date + pd.Timedelta(hours=int(rng.integers(8, 22)), minutes=int(rng.integers(0, 60)))
                rows.append(
                    {
                        "transaction_id": f"TX-{transaction_number:08d}",
                        "sku": item.sku,
                        "sold_at": sold_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "quantity": int(line_quantity),
                    }
                )
                transaction_number += 1
    return pd.DataFrame(rows, columns=["transaction_id", "sku", "sold_at", "quantity"])


def _sample_daily_quantity(
    scenario: str,
    current_date: pd.Timestamp,
    launch_date: pd.Timestamp,
    rng: np.random.Generator,
) -> int:
    """按场景生成整数销量；规则足够清晰，避免生成器反而比模型复杂。"""
    dow = current_date.dayofweek
    day_of_year = current_date.dayofyear
    days_since_launch = (current_date - launch_date).days

    if scenario == "stable":
        value = 6.0 + (1.0 if dow < 5 else -1.0) + rng.normal(0, 1.2)
    elif scenario == "intermittent":
        occurs = rng.random() < (0.16 if dow < 5 else 0.08)
        value = rng.lognormal(mean=1.55, sigma=0.35) if occurs else 0.0
    elif scenario == "weekly":
        probability = 0.72 if dow in (0, 2, 4) else 0.28
        occurs = rng.random() < probability
        value = rng.lognormal(mean=1.35, sigma=0.30) if occurs else 0.0
    elif scenario == "annual":
        annual_factor = 1.0 + 0.55 * np.sin(2 * np.pi * (day_of_year - 35) / 365.25)
        value = 7.0 * annual_factor + rng.normal(0, 1.1)
    elif scenario == "new_launch":
        ramp = min(days_since_launch / 75.0, 1.0)
        value = 1.5 + 5.0 * ramp + (0.8 if dow < 5 else -0.8) + rng.normal(0, 1.0)
    elif scenario == "observation_gap":
        value = 4.5 + (1.2 if dow < 5 else -0.6) + rng.normal(0, 1.0)
    else:
        raise ValueError(f"未知模拟场景: {scenario}")

    return max(0, int(np.rint(value)))


def _split_to_order_lines(quantity: int, rng: np.random.Generator) -> list[int]:
    """把一天的正销量拆为 1 至 3 笔订单，零销量自然没有订单行。"""
    if quantity <= 0:
        return []
    line_count = min(quantity, int(rng.integers(1, 4)))
    # multinomial 可能把某份分到 0；零数量不应成为一条真实订单行。
    return [part for part in rng.multinomial(quantity, np.repeat(1 / line_count, line_count)).tolist() if part > 0]


def write_source_data(
    sku_master: pd.DataFrame,
    raw_sales: pd.DataFrame,
    exceptions: pd.DataFrame,
    config: SyntheticConfig,
    output_dir: Path,
) -> None:
    """写出三张源表及轻量运行元信息，不额外维护场景数据库表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    sku_master.to_csv(output_dir / "sku_master.csv", index=False, encoding="utf-8-sig")
    raw_sales.to_csv(output_dir / "raw_sales.csv", index=False, encoding="utf-8-sig")
    exceptions.to_csv(output_dir / "observation_exceptions.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "random_seed": config.random_seed,
        "history_start": config.history_start.strftime("%Y-%m-%d"),
        "train_end": config.train_end.strftime("%Y-%m-%d"),
        "test_start": config.test_start.strftime("%Y-%m-%d"),
        "actual_end": config.actual_end.strftime("%Y-%m-%d"),
        "production_train_end": config.production_train_end.strftime("%Y-%m-%d"),
        "forecast_start": config.forecast_start.strftime("%Y-%m-%d"),
        "forecast_end": config.forecast_end.strftime("%Y-%m-%d"),
        "source_rows": {
            "sku_master": len(sku_master),
            "raw_sales": len(raw_sales),
            "observation_exceptions": len(exceptions),
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    """命令行入口：生成三张模拟源表。"""
    parser = argparse.ArgumentParser(description="生成需求预测项目的模拟源数据")
    parser.add_argument("--config", type=Path, required=True, help="模拟数据 JSON 配置文件")
    parser.add_argument("--output-dir", type=Path, required=True, help="源数据输出目录")
    args = parser.parse_args()

    config = SyntheticConfig.from_json(args.config)
    master, sales, exceptions = generate_source_data(config)
    write_source_data(master, sales, exceptions, config, args.output_dir)
    print(f"已生成 {len(master)} 个 SKU、{len(sales)} 条订单行: {args.output_dir}")


if __name__ == "__main__":
    main()
