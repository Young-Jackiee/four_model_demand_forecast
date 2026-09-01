"""模拟数据与技术规范日期窗口的回归测试。"""

import unittest
from pathlib import Path

import pandas as pd

from demand_forecast.data.daily_series import DailySeriesBuilder
from demand_forecast.data.synthetic import SyntheticConfig, generate_source_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SyntheticDataTests(unittest.TestCase):
    """固定随机种子下，数据范围和关键场景必须稳定。"""

    def test_generates_the_required_history_window(self) -> None:
        """规范要求 546 天训练加 52 天测试，共 598 个实际日期。"""
        config = SyntheticConfig.from_json(PROJECT_ROOT / "configs" / "synthetic_dataset.json")
        master, sales, exceptions = generate_source_data(config)
        daily = DailySeriesBuilder(str(config.history_start.date()), str(config.actual_end.date())).build(
            master, sales, exceptions
        )
        self.assertEqual(len(master), 12)
        self.assertEqual(config.test_start, config.train_end + pd.Timedelta(days=1))
        self.assertEqual(daily["date"].nunique(), 598)
        self.assertEqual(len(daily), 12 * 598)
        self.assertGreater((~daily["is_observed"]).sum(), 0)
        self.assertTrue(daily.loc[daily["is_observed"], "quantity"].notna().all())
        self.assertTrue(daily.loc[~daily["is_observed"], "quantity"].isna().all())

    def test_generator_is_reproducible(self) -> None:
        """同一配置重复运行应产生逐行一致的源数据。"""
        config = SyntheticConfig.from_json(PROJECT_ROOT / "configs" / "synthetic_dataset.json")
        first = generate_source_data(config)
        second = generate_source_data(config)
        for left, right in zip(first, second, strict=True):
            self.assertTrue(left.equals(right))


if __name__ == "__main__":
    unittest.main()
