"""Order safety smoke tests.

These tests must never call real KIS/Toss order functions.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services import broker


class OrderSafetyTests(unittest.TestCase):
    def test_buy_is_blocked_when_trading_disabled(self) -> None:
        with patch("app.services.broker.trading_enabled", return_value=False), \
             patch("app.services.kis_client.place_buy_order") as kis_buy, \
             patch("app.services.toss_client.place_buy_order") as toss_buy:
            res = broker.place_buy_order("005930", 1, price=70000)
        self.assertFalse(res["success"])
        self.assertTrue(res["data"]["blocked"])
        kis_buy.assert_not_called()
        toss_buy.assert_not_called()

    def test_sell_is_blocked_when_trading_disabled(self) -> None:
        with patch("app.services.broker.trading_enabled", return_value=False), \
             patch("app.services.kis_client.place_sell_order") as kis_sell, \
             patch("app.services.toss_client.place_sell_order") as toss_sell:
            res = broker.place_sell_order("005930", 1, price=70000)
        self.assertFalse(res["success"])
        self.assertTrue(res["data"]["blocked"])
        kis_sell.assert_not_called()
        toss_sell.assert_not_called()

    def test_cancel_is_blocked_when_trading_disabled(self) -> None:
        with patch("app.services.broker.trading_enabled", return_value=False), \
             patch("app.services.kis_client.cancel_order") as kis_cancel, \
             patch("app.services.toss_client.cancel_order") as toss_cancel:
            res = broker.cancel_order("005930", "TEST")
        self.assertFalse(res["success"])
        self.assertTrue(res["data"]["blocked"])
        kis_cancel.assert_not_called()
        toss_cancel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
