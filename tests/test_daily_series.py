"""日序列构建的关键边界测试。"""

import unittest

import pandas as pd

from demand_forecast.data.daily_series import DailySeriesBuilder
from demand_forecast.data.schemas import DataContractError


class DailySeriesBuilderTests(unittest.TestCase):
    """验证零销量、未上市和不可观测三个状态不会混淆。"""

    def setUp(self) -> None:
        self.master = pd.DataFrame({"sku": ["A"], "launch_date": ["2025-01-02"]})
        self.sales = pd.DataFrame(
            {
                "transaction_id": ["T1", "T2"],
                "sku": ["A", "A"],
                "sold_at": ["2025-01-02 09:00:00", "2025-01-02 14:00:00"],
                "quantity": [2, 3],
            }
        )
        self.exceptions = pd.DataFrame(
            {
                "sku": ["A"],
                "date": ["2025-01-04"],
                "is_observed": [False],
                "reason": ["source_missing"],
            }
        )

    def test_build_distinguishes_zero_from_unobserved(self) -> None:
        """上市后的无订单为 0；上市前和例外日必须保持空值。"""
        actual = DailySeriesBuilder("2025-01-01", "2025-01-04").build(
            self.master, self.sales, self.exceptions
        )
        self.assertFalse(actual.loc[0, "is_observed"])
        self.assertTrue(pd.isna(actual.loc[0, "quantity"]))
        self.assertEqual(actual.loc[1, "quantity"], 5.0)
        self.assertEqual(actual.loc[2, "quantity"], 0.0)
        self.assertFalse(actual.loc[3, "is_observed"])
        self.assertTrue(pd.isna(actual.loc[3, "quantity"]))

    def test_rejects_orders_on_unobserved_day(self) -> None:
        """不可观测日期同时有订单，代表源数据语义冲突，必须报错。"""
        conflicting = self.sales.copy()
        conflicting.loc[0, "sold_at"] = "2025-01-04 09:00:00"
        with self.assertRaises(DataContractError):
            DailySeriesBuilder("2025-01-01", "2025-01-04").build(
                self.master, conflicting, self.exceptions
            )


if __name__ == "__main__":
    unittest.main()
