import logging
from typing import Any, Dict, List, Optional

import streamlit as st

# Optional pykis import
try:
    from pykis import KisAuth, PyKis
    from pykis import KisQuote
    HAS_PYKIS = True
except Exception:  # pragma: no cover
    PyKis = Any  # type: ignore 
    KisQuote = Any  # type: ignore
    HAS_PYKIS = False

log = logging.getLogger(__name__)


def _api_response(success: bool, message: str, data: Optional[dict] = None) -> Dict[str, Any]:
    """Standardized API response payload."""
    return {"success": success, "message": message, "data": data or {}}


@st.cache_resource(show_spinner=False)
def create_pykis_from_auth(auth_path: str = "secret.json") -> Optional[PyKis]:
    """Create a cached PyKis instance from an auth file."""
    if not HAS_PYKIS:
        st.warning("pykis 미설치: pip install pykis")
        return None
    try:
        kis = PyKis(KisAuth.load(auth_path), keep_token=True)
        return kis
    except Exception as e:  # pragma: no cover
        st.error(f"KIS 인증 실패: {e}")
        log.exception("KIS auth failed")
        return None


def init_pykis(auth_path: str = "secret.json") -> Optional[PyKis]:
    """Initialize the KIS API and show a small toast."""
    if not HAS_PYKIS:
        return None
    try:
        kis = create_pykis_from_auth(auth_path)
        if kis:
            st.toast("✅ KIS API 연결 완료")
        return kis
    except Exception as e:  # pragma: no cover
        st.error(f"KIS 초기화 실패: {e}")
        log.exception("KIS init failed")
        return None


def get_stock_quote(kis_instance: Optional[PyKis], symbol: str) -> Dict[str, Any]:
    """Fetch a stock quote for a symbol."""
    if not kis_instance:
        return {}
    try:
        stock = kis_instance.stock(symbol)
        quote: KisQuote = stock.quote()
        return {
            "symbol": getattr(quote, "symbol", symbol),
            "name": getattr(quote, "name", ""),
            "market": getattr(quote, "market", ""),
            "price": float(getattr(quote, "price", 0)),
            "change": float(getattr(quote, "change", 0)),
            "rate": float(getattr(quote, "rate", 0.0)),
            "volume": float(getattr(quote, "volume", 0)),
            "amount": float(getattr(quote, "amount", 0)),
            "market_cap": float(getattr(quote, "market_cap", 0)),
            "open": float(getattr(quote, "open", 0)),
            "high": float(getattr(quote, "high", 0)),
            "low": float(getattr(quote, "low", 0)),
            "prev_price": float(getattr(quote, "prev_price", 0)),
            "high_limit": float(getattr(quote, "high_limit", 0)),
            "low_limit": float(getattr(quote, "low_limit", 0)),
        }
    except Exception as e:  # pragma: no cover
        st.warning(f"종목 정보 조회 실패: {e}")
        log.exception("quote failed")
        return {}


def get_account_balance(kis_instance: Optional[PyKis]) -> Dict[str, Any]:
    """Return normalized account balance info."""
    empty = {
        "total_amount": 0,
        "deposit": 0,
        "stocks_value": 0,
        "profit_loss": 0,
        "profit_loss_rate": 0.0,
        "market_value": 0,
    }
    if not kis_instance:
        return empty
    try:
        balance = kis_instance.account().balance()
        if not balance:
            return empty

        krw_deposit = 0
        if hasattr(balance, "deposits") and isinstance(balance.deposits, dict):
            krw = balance.deposits.get("KRW")
            if krw is not None:
                krw_deposit = getattr(krw, "amount", 0)

        current_amount = int(getattr(balance, "current_amount", 0))
        total_amount = int(krw_deposit) + int(current_amount)
        profit_loss = int(getattr(balance, "profit", 0))
        profit_loss_rate = float(getattr(balance, "profit_rate", 0.0))
        market_value = int(getattr(balance, "market_value", 0))

        return {
            "total_amount": total_amount,
            "deposit": int(krw_deposit),
            "stocks_value": current_amount,
            "profit_loss": profit_loss,
            "profit_loss_rate": profit_loss_rate,
            "market_value": market_value,
        }
    except Exception as e:  # pragma: no cover
        st.warning(f"계좌 잔고 조회 실패: {e}")
        log.exception("balance failed")
        return empty


def get_account_holdings(kis_instance: Optional[PyKis]) -> List[Dict[str, Any]]:
    """Return current holdings as a list of dicts."""
    if not kis_instance:
        return []
    try:
        balance = kis_instance.account().balance()
        if not balance or not hasattr(balance, "stocks"):
            return []
        holdings: List[Dict[str, Any]] = []
        for stock in balance.stocks:
            holdings.append({
                "name": getattr(stock, "name", ""),
                "symbol": getattr(stock, "symbol", ""),
                "market": getattr(stock, "market", ""),
                "qty": int(getattr(stock, "qty", 0)),
                "price": float(getattr(stock, "price", 0)),
                "amount": float(getattr(stock, "amount", 0)),
                "profit": float(getattr(stock, "profit", 0)),
                "profit_rate": float(getattr(stock, "profit_rate", 0.0)),
            })
        return holdings
    except Exception as e:  # pragma: no cover
        st.warning(f"보유종목 조회 실패: {e}")
        log.exception("holdings failed")
        return []


def place_buy_order(kis_instance: Optional[PyKis], symbol: str, qty: int, price: Optional[int] = None) -> Dict[str, Any]:
    """Place a buy order (market when price=None)."""
    if not kis_instance:
        return _api_response(False, "KIS API 연결이 필요합니다.")
    try:
        stock = kis_instance.stock(symbol)
        if price is None:
            stock.buy(qty=qty)  # market order
        else:
            stock.buy(price=price, qty=qty)
        return _api_response(True, "매수 주문 접수")
    except Exception as e:  # pragma: no cover
        return _api_response(False, f"매수 주문 실패: {e}")


def place_sell_order(kis_instance: Optional[PyKis], symbol: str, qty: Optional[int] = None, price: Optional[int] = None) -> Dict[str, Any]:
    """Place a sell order. qty/price optional to allow full-position or market orders."""
    if not kis_instance:
        return _api_response(False, "KIS API 연결이 필요합니다.")
    try:
        stock = kis_instance.stock(symbol)
        if qty is None and price is None:
            stock.sell()  # full position market
        elif qty is None and price is not None:
            stock.sell(price=price)
        elif qty is not None and price is None:
            stock.sell(qty=qty)
        else:
            stock.sell(price=price, qty=qty)
        return _api_response(True, "매도 주문 접수")
    except Exception as e:  # pragma: no cover
        return _api_response(False, f"매도 주문 실패: {e}")


def get_pending_orders(kis_instance: Optional[PyKis]) -> List[Dict[str, Any]]:
    """Return pending (open) orders."""
    if not kis_instance:
        return []
    try:
        account = kis_instance.account()
        pending_orders = account.pending_orders()
        rows: List[Dict[str, Any]] = []
        for order in pending_orders or []:
            rows.append({
                "name": getattr(order, "name", ""),
                "symbol": getattr(order, "symbol", ""),
                "side": getattr(order, "side", ""),
                "qty": int(getattr(order, "qty", 0)),
                "price": float(getattr(order, "price", 0)),
                "pending_qty": int(getattr(getattr(order, "pending_order", None), "pending_qty", 0)),
            })
        return rows
    except Exception as e:  # pragma: no cover
        st.warning(f"미체결 주문 조회 실패: {e}")
        log.exception("pending failed")
        return []


def cancel_order(kis_instance: Optional[PyKis], symbol: str, order_id: Optional[str] = None) -> Dict[str, Any]:
    """Cancel an order by id or last pending for symbol (best-effort; depends on API)."""
    if not kis_instance:
        return _api_response(False, "KIS API 연결이 필요합니다.")
    try:
        # NOTE: pykis의 취소 API 형태에 따라 수정 필요
        account = kis_instance.account()
        if order_id:
            account.cancel(order_id)  # type: ignore[attr-defined]
            return _api_response(True, "주문 취소 접수", {"order_id": order_id})
        # Fallback: symbol 기준 최근 미체결 취소
        for od in account.pending_orders() or []:
            if getattr(od, "symbol", "") == symbol:
                account.cancel(getattr(od, "id", None))  # type: ignore[attr-defined]
                return _api_response(True, "주문 취소 접수", {"symbol": symbol})
        return _api_response(False, "취소할 주문을 찾지 못했습니다.")
    except Exception as e:  # pragma: no cover
        return _api_response(False, f"주문 취소 실패: {e}")