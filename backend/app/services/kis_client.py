"""한국투자증권(pykis) 클라이언트 래퍼."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import get_settings

log = logging.getLogger(__name__)

try:
    from pykis import KisAuth, PyKis
    HAS_PYKIS = True
except Exception:  # pragma: no cover
    KisAuth = Any  # type: ignore
    PyKis = Any  # type: ignore
    HAS_PYKIS = False

_DOMESTIC_MARKETS = {"KRX", "KOSPI", "KOSDAQ", "KONEX"}
_kis_instance: Optional[Any] = None
_kis_lock = threading.Lock()

# 유량 보호용 TTL 캐시 (초)
_QUOTE_TTL = 20.0
_QUOTE_FAIL_TTL = 60.0  # 실패도 잠시 캐시해 KIS 재호출 폭주 방지
_ORDERBOOK_TTL = 0.8
_ORDERBOOK_FAIL_TTL = 8.0
_ACCOUNT_TTL = 45.0
_RATE_LIMIT_COOLDOWN = 45.0  # 호출 횟수 초과 시 전역 쿨다운
_quote_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_quote_fail_cache: Dict[str, float] = {}
_quote_inflight: Dict[str, threading.Event] = {}
_quote_lock = threading.Lock()
_orderbook_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_orderbook_fail_cache: Dict[str, float] = {}
_orderbook_inflight: Dict[str, threading.Event] = {}
_orderbook_lock = threading.Lock()
_account_cache: Optional[Dict[str, Any]] = None
_account_cache_at = 0.0
_account_lock = threading.Lock()
_account_inflight: Optional[threading.Event] = None
_account_inflight_result: Optional[Dict[str, Any]] = None
_rate_limited_until = 0.0
_rate_limit_lock = threading.Lock()


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc) or ""
    return any(
        tok in msg
        for tok in (
            "호출 횟수",
            "EGW00201",
            "EGW00204",
            "rate limit",
            "Rate Limit",
            "Too Many Requests",
        )
    )


def mark_rate_limited(seconds: float = _RATE_LIMIT_COOLDOWN) -> None:
    """KIS 유량 초과 감지 시 전역 쿨다운 (이미 쿨다운 중이면 로그만 생략)."""
    global _rate_limited_until
    until = time.time() + max(5.0, float(seconds))
    with _rate_limit_lock:
        was_active = time.time() < _rate_limited_until
        _rate_limited_until = max(_rate_limited_until, until)
    if not was_active:
        log.warning("KIS rate-limit cooldown %.0fs", seconds)


def is_rate_limited() -> bool:
    with _rate_limit_lock:
        return time.time() < _rate_limited_until


def _api_response(success: bool, message: str, data: Optional[dict] = None) -> Dict[str, Any]:
    return {"success": success, "message": message, "data": data or {}}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _is_domestic_market(market: str) -> bool:
    m = (market or "").upper()
    if not m:
        return True
    if m in _DOMESTIC_MARKETS:
        return True
    return m.startswith("K") and m not in {"NYSE", "NASDAQ", "AMEX"}


def _empty_account() -> Dict[str, Any]:
    return {
        "connected": False,
        "account": "",
        "virtual": False,
        "total_eval_krw": 0.0,
        "purchase_amount": 0.0,
        "current_amount": 0.0,
        "profit_loss": 0.0,
        "profit_loss_rate": 0.0,
        "deposits": [],
        "domestic": {"deposit_krw": 0.0, "stocks_value": 0.0, "holdings": []},
        "overseas": {"deposit_usd": 0.0, "deposit_krw": 0.0, "stocks_value": 0.0, "holdings": []},
        "holdings": [],
    }


def get_kis(force: bool = False) -> Optional[Any]:
    """KIS 인스턴스 생성. secret.json 우선, 없으면 .env 값 사용."""
    global _kis_instance
    if _kis_instance is not None and not force:
        return _kis_instance
    if not HAS_PYKIS:
        log.warning("pykis is not installed")
        return None

    with _kis_lock:
        if _kis_instance is not None and not force:
            return _kis_instance

        settings = get_settings()
        auth_path = settings.resolved_kis_auth_path

        try:
            if auth_path.exists():
                # 웹소켓은 시세 폴링에 불필요하고 리소스를 잡아먹음
                _kis_instance = PyKis(
                    KisAuth.load(str(auth_path)),
                    keep_token=True,
                    use_websocket=False,
                )
                log.info("KIS connected via auth file: %s", auth_path)
                return _kis_instance

            if settings.kis_app_key and settings.kis_app_secret and settings.kis_account:
                hts_id = (settings.kis_hts_id or "user").lstrip("@").strip()
                # pykis는 첫 번째 auth가 실전(virtual=False)이어야 하며,
                # 모의투자는 virtual_auth로 별도 전달해야 한다.
                if settings.kis_virtual:
                    real_auth = KisAuth(
                        id=hts_id,
                        appkey=settings.kis_app_key,
                        secretkey=settings.kis_app_secret,
                        account=settings.kis_account,
                        virtual=False,
                    )
                    virtual_auth = KisAuth(
                        id=hts_id,
                        appkey=settings.kis_app_key,
                        secretkey=settings.kis_app_secret,
                        account=settings.kis_account,
                        virtual=True,
                    )
                    _kis_instance = PyKis(
                        real_auth,
                        virtual_auth,
                        keep_token=True,
                        use_websocket=False,
                    )
                else:
                    auth = KisAuth(
                        id=hts_id,
                        appkey=settings.kis_app_key,
                        secretkey=settings.kis_app_secret,
                        account=settings.kis_account,
                        virtual=False,
                    )
                    _kis_instance = PyKis(auth, keep_token=True, use_websocket=False)
                log.info("KIS connected via .env credentials (virtual=%s)", settings.kis_virtual)
                return _kis_instance
            log.warning("KIS credentials not found (.env / secret.json)")
        except Exception as e:  # pragma: no cover
            log.exception("KIS init failed: %s", e)
            _kis_instance = None
            return None

    return None


def is_kis_connected() -> bool:
    return get_kis() is not None


def peek_cached_quote(symbol: str) -> Dict[str, Any]:
    """추가 API 호출 없이 캐시된 시세만 반환."""
    key = (symbol or "").strip().upper()
    with _quote_lock:
        cached = _quote_cache.get(key)
        if cached:
            return dict(cached[1])
    return {}


def get_stock_quote(symbol: str, market: str = "") -> Dict[str, Any]:
    from app.services.symbol_utils import is_plausible_symbol, normalize_symbol

    key = normalize_symbol(symbol, market)
    if not key or not is_plausible_symbol(key, market):
        log.debug("quote skipped: invalid symbol input=%r", symbol)
        return {}

    now = time.time()
    wait_ev: Optional[threading.Event] = None
    is_owner = False
    with _quote_lock:
        cached = _quote_cache.get(key)
        if cached and now - cached[0] < _QUOTE_TTL:
            return dict(cached[1])
        fail_at = _quote_fail_cache.get(key)
        if fail_at and now - fail_at < _QUOTE_FAIL_TTL:
            return {}
        inflight = _quote_inflight.get(key)
        if inflight is not None:
            wait_ev = inflight
        else:
            is_owner = True
            _quote_inflight[key] = threading.Event()

    if not is_owner and wait_ev is not None:
        wait_ev.wait(timeout=12.0)
        with _quote_lock:
            cached = _quote_cache.get(key)
            if cached:
                return dict(cached[1])
        return {}

    data: Dict[str, Any] = {}
    try:
        if is_rate_limited():
            with _quote_lock:
                cached = _quote_cache.get(key)
                if cached:
                    return dict(cached[1])
            return {}

        kis = get_kis()
        if not kis:
            with _quote_lock:
                _quote_fail_cache[key] = time.time()
            return {}

        quote = kis.stock(key).quote()
        data = {
            "symbol": getattr(quote, "symbol", key) or key,
            "name": getattr(quote, "name", "") or "",
            "market": getattr(quote, "market", "") or "",
            "price": _to_float(getattr(quote, "price", 0)),
            "change": _to_float(getattr(quote, "change", 0)),
            "rate": _to_float(getattr(quote, "rate", 0.0)),
            "volume": _to_float(getattr(quote, "volume", 0)),
            "amount": _to_float(getattr(quote, "amount", 0)),
            "open": _to_float(getattr(quote, "open", 0)),
            "high": _to_float(getattr(quote, "high", 0)),
            "low": _to_float(getattr(quote, "low", 0)),
            "prev_price": _to_float(getattr(quote, "prev_price", 0)),
        }
        with _quote_lock:
            _quote_cache[key] = (time.time(), data)
            _quote_fail_cache.pop(key, None)
        return data
    except Exception as e:
        if _is_rate_limit_error(e):
            mark_rate_limited()
        log.warning("quote failed for %s: %s", key, e)
        with _quote_lock:
            _quote_fail_cache[key] = time.time()
            cached = _quote_cache.get(key)
            if cached:
                data = dict(cached[1])
        return data
    finally:
        with _quote_lock:
            ev = _quote_inflight.pop(key, None)
        if ev:
            ev.set()


def _empty_orderbook(symbol: str = "", market: str = "", message: str = "") -> Dict[str, Any]:
    return {
        "ok": False,
        "symbol": symbol,
        "market": market,
        "name": "",
        "asks": [],
        "bids": [],
        "ask_volume": 0,
        "bid_volume": 0,
        "decimal_places": 0,
        "cached": False,
        "rate_limited": False,
        "message": message,
        "as_of": 0.0,
    }


def _levels_from_book(items: Any) -> List[Dict[str, Any]]:
    levels: List[Dict[str, Any]] = []
    for item in items or []:
        price = _to_float(getattr(item, "price", 0))
        volume = int(_to_float(getattr(item, "volume", 0)))
        if price <= 0 and volume <= 0:
            continue
        levels.append({"price": price, "volume": max(0, volume)})
    return levels


def get_stock_orderbook(symbol: str, market: str = "") -> Dict[str, Any]:
    """KIS REST 호가 (짧은 TTL 캐시 + rate-limit 시 캐시/스킵)."""
    from app.services.symbol_utils import is_plausible_symbol, normalize_symbol

    key = normalize_symbol(symbol, market)
    mkt = (market or ("KRX" if key.isdigit() else "US")).upper()
    if not key or not is_plausible_symbol(key, market):
        return _empty_orderbook(symbol=str(symbol or "")[:80], market=mkt, message="Invalid symbol")

    now = time.time()
    wait_ev: Optional[threading.Event] = None
    is_owner = False
    with _orderbook_lock:
        cached = _orderbook_cache.get(key)
        if cached and now - cached[0] < _ORDERBOOK_TTL:
            out = dict(cached[1])
            out["cached"] = True
            return out
        fail_at = _orderbook_fail_cache.get(key)
        if fail_at and now - fail_at < _ORDERBOOK_FAIL_TTL:
            if cached:
                out = dict(cached[1])
                out["cached"] = True
                out["message"] = out.get("message") or "Using cached orderbook after recent failure"
                return out
            return _empty_orderbook(key, mkt, "Orderbook temporarily unavailable")
        inflight = _orderbook_inflight.get(key)
        if inflight is not None:
            wait_ev = inflight
        else:
            is_owner = True
            _orderbook_inflight[key] = threading.Event()

    if not is_owner and wait_ev is not None:
        wait_ev.wait(timeout=8.0)
        with _orderbook_lock:
            cached = _orderbook_cache.get(key)
            if cached:
                out = dict(cached[1])
                out["cached"] = True
                return out
        return _empty_orderbook(key, mkt, "Orderbook wait timeout")

    try:
        if is_rate_limited():
            with _orderbook_lock:
                cached = _orderbook_cache.get(key)
            if cached:
                out = dict(cached[1])
                out["cached"] = True
                out["rate_limited"] = True
                out["message"] = "KIS rate-limit cooldown (cached)"
                return out
            empty = _empty_orderbook(key, mkt, "KIS rate-limit cooldown")
            empty["rate_limited"] = True
            return empty

        kis = get_kis()
        if not kis:
            return _empty_orderbook(key, mkt, "KIS is not connected")

        stock = kis.stock(key)
        book = stock.orderbook()
        asks = _levels_from_book(getattr(book, "asks", None))
        bids = _levels_from_book(getattr(book, "bids", None))
        # 0가격 호가 제거 후 best ask/bid만 유지
        asks = [a for a in asks if a["price"] > 0][:10]
        bids = [b for b in bids if b["price"] > 0][:10]
        ask_vol = int(sum(a["volume"] for a in asks))
        bid_vol = int(sum(b["volume"] for b in bids))
        data = {
            "ok": bool(asks or bids),
            "symbol": str(getattr(book, "symbol", key) or key),
            "market": str(getattr(book, "market", mkt) or mkt),
            "name": str(getattr(stock, "name", "") or getattr(book, "name", "") or ""),
            "asks": asks,
            "bids": bids,
            "ask_volume": ask_vol,
            "bid_volume": bid_vol,
            "decimal_places": int(_to_float(getattr(book, "decimal_places", 0))),
            "cached": False,
            "rate_limited": False,
            "message": "" if (asks or bids) else "Empty orderbook (market closed?)",
            "as_of": time.time(),
        }
        with _orderbook_lock:
            _orderbook_cache[key] = (time.time(), data)
            _orderbook_fail_cache.pop(key, None)
        return dict(data)
    except Exception as e:
        if _is_rate_limit_error(e):
            mark_rate_limited()
        log.warning("orderbook failed for %s: %s", key, e)
        with _orderbook_lock:
            _orderbook_fail_cache[key] = time.time()
            cached = _orderbook_cache.get(key)
        if cached:
            out = dict(cached[1])
            out["cached"] = True
            out["rate_limited"] = _is_rate_limit_error(e)
            out["message"] = f"Stale cache after error: {e}"
            return out
        empty = _empty_orderbook(key, mkt, f"Orderbook failed: {e}")
        empty["rate_limited"] = _is_rate_limit_error(e)
        return empty
    finally:
        with _orderbook_lock:
            ev = _orderbook_inflight.pop(key, None)
        if ev:
            ev.set()


def _parse_deposits(balance: Any) -> List[Dict[str, Any]]:
    deposits: List[Dict[str, Any]] = []
    raw = getattr(balance, "deposits", None)
    if not isinstance(raw, dict):
        return deposits
    for currency, dep in raw.items():
        amount = _to_float(getattr(dep, "amount", 0))
        fx = _to_float(getattr(dep, "exchange_rate", 1.0), 1.0)
        cur = str(getattr(dep, "currency", currency) or currency).upper()
        deposits.append({
            "currency": cur,
            "amount": amount,
            "exchange_rate": fx if fx > 0 else 1.0,
            "amount_krw": amount * (fx if cur != "KRW" else 1.0),
            "scope": "domestic" if cur == "KRW" else "overseas",
        })
    return deposits


def _parse_holding(stock: Any) -> Dict[str, Any]:
    market = str(getattr(stock, "market", "") or "")
    qty = _to_float(getattr(stock, "qty", 0))
    price = _to_float(getattr(stock, "price", 0))
    amount = _to_float(getattr(stock, "amount", 0))
    profit = _to_float(getattr(stock, "profit", 0))
    profit_rate = _to_float(getattr(stock, "profit_rate", 0.0))
    currency = "KRW" if _is_domestic_market(market) else "USD"
    for attr in ("currency", "curr_cd", "currency_code"):
        val = getattr(stock, attr, None)
        if val:
            currency = str(val).upper()
            break
    return {
        "name": getattr(stock, "name", "") or "",
        "symbol": getattr(stock, "symbol", "") or "",
        "market": market,
        "qty": qty,
        "price": price,
        "amount": amount,
        "profit": profit,
        "profit_rate": profit_rate,
        "currency": currency,
        "scope": "domestic" if _is_domestic_market(market) else "overseas",
    }


def _fetch_account_overview() -> Dict[str, Any]:
    """실제 잔고 API 호출 (캐시/단일비행 외부에서 사용)."""
    empty = _empty_account()
    if is_rate_limited():
        empty["error"] = "KIS rate limit cooldown"
        return empty
    kis = get_kis()
    if not kis:
        return empty

    settings = get_settings()
    try:
        balance = None
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                balance = kis.account().balance()
                break
            except Exception as e:
                last_err = e
                if _is_rate_limit_error(e) or "초당 거래건수" in str(e):
                    mark_rate_limited()
                    break
                msg = str(e)
                if any(code in msg for code in ("EGW00133", "EGW00215", "1분당 1회")):
                    time.sleep(1.2 * (attempt + 1))
                    continue
                raise
        if balance is None and last_err is not None:
            raise last_err
        if not balance:
            return {
                **empty,
                "connected": True,
                "account": settings.kis_account,
                "virtual": settings.kis_virtual,
            }

        deposits = _parse_deposits(balance)
        holdings = []
        for stock in getattr(balance, "stocks", []) or []:
            item = _parse_holding(stock)
            if item["qty"] <= 0 and item["amount"] <= 0:
                continue
            holdings.append(item)

        domestic_holdings = [h for h in holdings if h["scope"] == "domestic"]
        overseas_holdings = [h for h in holdings if h["scope"] == "overseas"]

        krw_dep = next((d for d in deposits if d["currency"] == "KRW"), None)
        usd_dep = next((d for d in deposits if d["currency"] == "USD"), None)

        domestic_stock_value = sum(h["amount"] for h in domestic_holdings)
        overseas_stock_value = sum(h["amount"] for h in overseas_holdings)

        usd_fx = _to_float(usd_dep["exchange_rate"], 0) if usd_dep else 0.0
        if usd_fx <= 0:
            usd_fx = settings.default_exchange_rate
        overseas_stock_krw = overseas_stock_value * usd_fx

        purchase_amount = _to_float(getattr(balance, "purchase_amount", 0))
        current_amount = _to_float(getattr(balance, "current_amount", 0))
        profit = _to_float(getattr(balance, "profit", 0))
        profit_rate = _to_float(getattr(balance, "profit_rate", 0.0))

        deposit_krw = _to_float(krw_dep["amount"]) if krw_dep else 0.0
        deposit_usd = _to_float(usd_dep["amount"]) if usd_dep else 0.0
        deposit_usd_krw = _to_float(usd_dep["amount_krw"]) if usd_dep else deposit_usd * usd_fx

        total_eval = current_amount
        if total_eval <= 0:
            total_eval = domestic_stock_value + overseas_stock_krw
        total_eval_krw = total_eval + deposit_krw + deposit_usd_krw

        account_no = str(getattr(balance, "account_number", settings.kis_account) or settings.kis_account)

        return {
            "connected": True,
            "account": account_no,
            "virtual": settings.kis_virtual,
            "total_eval_krw": round(total_eval_krw, 2),
            "purchase_amount": round(purchase_amount, 2),
            "current_amount": round(current_amount, 2),
            "profit_loss": round(profit, 2),
            "profit_loss_rate": round(profit_rate, 4),
            "deposits": deposits,
            "domestic": {
                "deposit_krw": round(deposit_krw, 2),
                "stocks_value": round(domestic_stock_value, 2),
                "holdings": domestic_holdings,
            },
            "overseas": {
                "deposit_usd": round(deposit_usd, 2),
                "deposit_krw": round(deposit_usd_krw, 2),
                "stocks_value": round(overseas_stock_value, 2),
                "stocks_value_krw": round(overseas_stock_krw, 2),
                "exchange_rate": usd_fx,
                "holdings": overseas_holdings,
            },
            "holdings": holdings,
        }
    except Exception as e:
        log.exception("account overview failed: %s", e)
        msg = str(e)
        if "모의투자용 앱키가 아닙니다" in msg:
            friendly = "앱키가 실전용입니다. .env의 KIS_VIRTUAL=false 로 설정하세요."
        elif "EGW00133" in msg or "1분당 1회" in msg:
            friendly = "접근토큰 발급 제한입니다. 약 1분 후 다시 시도하세요."
        elif "EGW00215" in msg or "초당 거래건수" in msg or "호출 횟수" in msg:
            friendly = "API 호출이 너무 빠릅니다. 잠시 후 다시 시도하세요."
        else:
            friendly = "계좌 조회에 실패했습니다. KIS 연동 설정을 확인하세요."
        return {
            **empty,
            "connected": True,
            "account": settings.kis_account,
            "virtual": settings.kis_virtual,
            "error": friendly,
        }


def get_account_overview(force: bool = False) -> Dict[str, Any]:
    """국내/해외 예수금·보유종목 통합 조회 (TTL 캐시 + 단일비행)."""
    global _account_cache, _account_cache_at, _account_inflight, _account_inflight_result

    now = time.time()
    if (
        not force
        and _account_cache is not None
        and now - _account_cache_at < _ACCOUNT_TTL
        and not _account_cache.get("error")
    ):
        return dict(_account_cache)

    wait_event: Optional[threading.Event] = None
    leader = False
    with _account_lock:
        now = time.time()
        if (
            not force
            and _account_cache is not None
            and now - _account_cache_at < _ACCOUNT_TTL
            and not _account_cache.get("error")
        ):
            return dict(_account_cache)

        if _account_inflight is not None:
            wait_event = _account_inflight
        else:
            _account_inflight = threading.Event()
            _account_inflight_result = None
            wait_event = _account_inflight
            leader = True

    if not leader and wait_event is not None:
        wait_event.wait(timeout=60)
        if _account_inflight_result is not None:
            return dict(_account_inflight_result)
        if _account_cache is not None:
            return dict(_account_cache)
        return _empty_account()

    try:
        result = _fetch_account_overview()
        # 성공 또는 연결됨 결과는 캐시. 에러만 있고 이전 성공 캐시가 있으면 유지
        if result.get("connected") and not result.get("error"):
            _account_cache = result
            _account_cache_at = time.time()
        elif result.get("error") and _account_cache and _account_cache.get("connected"):
            result = {
                **_account_cache,
                "error": result.get("error"),
                "stale": True,
            }
        elif result.get("connected"):
            _account_cache = result
            _account_cache_at = time.time()
        _account_inflight_result = result
        return dict(result)
    finally:
        with _account_lock:
            if _account_inflight is not None:
                _account_inflight.set()
            _account_inflight = None


def get_account_balance() -> Dict[str, Any]:
    overview = get_account_overview()
    return {
        "total_amount": int(overview.get("total_eval_krw", 0) or 0),
        "deposit": int(overview.get("domestic", {}).get("deposit_krw", 0) or 0),
        "stocks_value": int(overview.get("current_amount", 0) or 0),
        "profit_loss": int(overview.get("profit_loss", 0) or 0),
        "profit_loss_rate": float(overview.get("profit_loss_rate", 0.0) or 0),
        "market_value": int(overview.get("current_amount", 0) or 0),
    }


def get_account_holdings() -> List[Dict[str, Any]]:
    overview = get_account_overview()
    return overview.get("holdings") or []


def place_buy_order(
    symbol: str,
    qty: int | float,
    price: Optional[int | float] = None,
) -> Dict[str, Any]:
    kis = get_kis()
    if not kis:
        return _api_response(False, "KIS API is not connected.")
    try:
        stock = kis.stock(symbol)
        if price is None:
            stock.buy(qty=qty)
        else:
            stock.buy(price=price, qty=qty)
        return _api_response(True, "Buy order accepted")
    except Exception as e:
        return _api_response(False, f"Buy order failed: {e}")


def place_sell_order(
    symbol: str,
    qty: Optional[int | float] = None,
    price: Optional[int | float] = None,
) -> Dict[str, Any]:
    kis = get_kis()
    if not kis:
        return _api_response(False, "KIS API is not connected.")
    try:
        stock = kis.stock(symbol)
        if qty is None and price is None:
            stock.sell()
        elif qty is None:
            stock.sell(price=price)
        elif price is None:
            stock.sell(qty=qty)
        else:
            stock.sell(price=price, qty=qty)
        return _api_response(True, "Sell order accepted")
    except Exception as e:
        return _api_response(False, f"Sell order failed: {e}")


def get_pending_orders() -> List[Dict[str, Any]]:
    kis = get_kis()
    if not kis:
        return []
    try:
        rows = []
        for order in kis.account().pending_orders() or []:
            pending = getattr(order, "pending_order", None)
            rows.append({
                "name": getattr(order, "name", ""),
                "symbol": getattr(order, "symbol", ""),
                "side": str(getattr(order, "side", "")),
                "qty": _to_float(getattr(order, "qty", 0)),
                "price": _to_float(getattr(order, "price", 0)),
                "pending_qty": int(_to_float(getattr(pending, "pending_qty", 0))) if pending else 0,
            })
        return rows
    except Exception as e:
        log.exception("pending failed: %s", e)
        return []


def cancel_order(symbol: str, order_id: Optional[str] = None) -> Dict[str, Any]:
    kis = get_kis()
    if not kis:
        return _api_response(False, "KIS API is not connected.")
    try:
        account = kis.account()
        if order_id:
            account.cancel(order_id)  # type: ignore[attr-defined]
            return _api_response(True, "Cancel accepted", {"order_id": order_id})
        for od in account.pending_orders() or []:
            if getattr(od, "symbol", "") == symbol:
                account.cancel(getattr(od, "id", None))  # type: ignore[attr-defined]
                return _api_response(True, "Cancel accepted", {"symbol": symbol})
        return _api_response(False, "No pending order found.")
    except Exception as e:
        return _api_response(False, f"Cancel failed: {e}")
