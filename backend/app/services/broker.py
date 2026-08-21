"""활성 증권사(한투/토스)로 시세·계좌·주문을 동일 인터페이스로 보냅니다."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.services import kis_client, toss_client
from app.services.broker_settings import get_active_broker


def active_broker() -> str:
    return get_active_broker()


def _use_toss() -> bool:
    return active_broker() == "toss"


def is_connected() -> bool:
    if _use_toss():
        return toss_client.is_connected()
    return kis_client.is_kis_connected()


def is_rate_limited() -> bool:
    if _use_toss():
        return toss_client.is_rate_limited()
    return kis_client.is_rate_limited()


def mark_rate_limited(seconds: float = 45.0) -> None:
    if _use_toss():
        toss_client.mark_rate_limited(seconds)
    else:
        kis_client.mark_rate_limited(seconds)


def peek_cached_quote(symbol: str) -> Dict[str, Any]:
    if _use_toss():
        return toss_client.peek_cached_quote(symbol)
    return kis_client.peek_cached_quote(symbol)


def get_stock_quote(symbol: str, market: str = "") -> Dict[str, Any]:
    if _use_toss():
        return toss_client.get_stock_quote(symbol, market)
    return kis_client.get_stock_quote(symbol, market)


def get_stock_orderbook(symbol: str, market: str = "") -> Dict[str, Any]:
    if _use_toss():
        return toss_client.get_stock_orderbook(symbol, market)
    return kis_client.get_stock_orderbook(symbol, market)


def get_account_overview(force: bool = False) -> Dict[str, Any]:
    if _use_toss():
        return toss_client.get_account_overview(force=force)
    return kis_client.get_account_overview(force=force)


def get_account_balance() -> Dict[str, Any]:
    if _use_toss():
        return toss_client.get_account_balance()
    return kis_client.get_account_balance()


def get_account_holdings() -> List[Dict[str, Any]]:
    if _use_toss():
        return toss_client.get_account_holdings()
    return kis_client.get_account_holdings()


def get_pending_orders() -> List[Dict[str, Any]]:
    if _use_toss():
        return toss_client.get_pending_orders()
    return kis_client.get_pending_orders()


def place_buy_order(
    symbol: str,
    qty: int | float,
    price: Optional[int | float] = None,
) -> Dict[str, Any]:
    if _use_toss():
        return toss_client.place_buy_order(symbol, qty, price=price)
    return kis_client.place_buy_order(symbol, qty, price=price)


def place_sell_order(
    symbol: str,
    qty: Optional[int | float] = None,
    price: Optional[int | float] = None,
) -> Dict[str, Any]:
    if _use_toss():
        return toss_client.place_sell_order(symbol, qty, price=price)
    return kis_client.place_sell_order(symbol, qty, price=price)


def cancel_order(symbol: str, order_id: Optional[str] = None) -> Dict[str, Any]:
    if _use_toss():
        return toss_client.cancel_order(symbol, order_id)
    return kis_client.cancel_order(symbol, order_id)


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    minute_interval: int = 1,
    hour_interval: int = 1,
) -> pd.DataFrame:
    if _use_toss():
        return toss_client.fetch_ohlcv(symbol, timeframe, minute_interval, hour_interval)
    return pd.DataFrame()
