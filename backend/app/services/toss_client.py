"""토스증권 Open API 클라이언트. 응답 형태는 KIS 래퍼와 동일하게 맞춥니다."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd

from app.core.config import get_settings

log = logging.getLogger(__name__)

_QUOTE_TTL = 20.0
_QUOTE_FAIL_TTL = 60.0
_ORDERBOOK_TTL = 0.8
_ORDERBOOK_FAIL_TTL = 8.0
_ACCOUNT_TTL = 45.0
_RATE_LIMIT_COOLDOWN = 2.0
_TOKEN_SKEW = 60.0

_http: Optional[httpx.Client] = None
_http_lock = threading.Lock()
_token: Optional[str] = None
_token_exp = 0.0
_token_lock = threading.Lock()
_account_seq: Optional[int] = None
_account_no = ""
_account_lock = threading.Lock()
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
_account_overview_lock = threading.Lock()
_account_inflight: Optional[threading.Event] = None
_account_inflight_result: Optional[Dict[str, Any]] = None
_rate_limited_until = 0.0
_rate_limit_lock = threading.Lock()
_name_cache: Dict[str, str] = {}


class TossAPIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int = 0,
        code: str = "",
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.retry_after = retry_after


def reset_client() -> None:
    """키/활성 증권사 변경 시 토큰·계좌 캐시를 버립니다."""
    global _token, _token_exp, _account_seq, _account_no
    global _account_cache, _account_cache_at
    with _token_lock:
        _token = None
        _token_exp = 0.0
    with _account_lock:
        _account_seq = None
        _account_no = ""
    with _account_overview_lock:
        _account_cache = None
        _account_cache_at = 0.0


def mark_rate_limited(seconds: float = _RATE_LIMIT_COOLDOWN) -> None:
    global _rate_limited_until
    until = time.time() + max(1.0, float(seconds))
    with _rate_limit_lock:
        was_active = time.time() < _rate_limited_until
        _rate_limited_until = max(_rate_limited_until, until)
    if not was_active:
        log.warning("Toss rate-limit cooldown %.0fs", seconds)


def is_rate_limited() -> bool:
    with _rate_limit_lock:
        return time.time() < _rate_limited_until


def _http_client() -> httpx.Client:
    global _http
    if _http is None:
        with _http_lock:
            if _http is None:
                _http = httpx.Client(timeout=15.0)
    return _http


def _base_url() -> str:
    settings = get_settings()
    return (settings.toss_api_base_url or "https://openapi.tossinvest.com").rstrip("/")


def _creds() -> Optional[dict]:
    from app.services.broker_settings import get_toss_override

    return get_toss_override()


def is_configured() -> bool:
    return bool(_creds())


def is_connected() -> bool:
    if not is_configured():
        return False
    try:
        return bool(_access_token())
    except Exception as e:
        log.warning("Toss auth failed: %s", e)
        return False


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _api_response(success: bool, message: str, data: Optional[dict] = None) -> Dict[str, Any]:
    return {"success": success, "message": message, "data": data or {}}


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


def _unwrap(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    err = payload.get("error")
    if err:
        if isinstance(err, dict):
            raise TossAPIError(
                str(err.get("message") or err.get("code") or "Toss API error"),
                code=str(err.get("code") or ""),
            )
        raise TossAPIError(str(err))
    if "result" in payload:
        return payload["result"]
    return payload


def _retry_after(resp: httpx.Response) -> float:
    raw = resp.headers.get("Retry-After") or resp.headers.get("retry-after") or ""
    try:
        return max(1.0, float(raw))
    except Exception:
        return _RATE_LIMIT_COOLDOWN


def _access_token(force: bool = False) -> str:
    global _token, _token_exp
    now = time.time()
    with _token_lock:
        if not force and _token and now < _token_exp - _TOKEN_SKEW:
            return _token
        creds = _creds()
        if not creds:
            raise TossAPIError("Toss API credentials are not configured.")
        resp = _http_client().post(
            f"{_base_url()}/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"],
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code == 429:
            mark_rate_limited(_retry_after(resp))
            raise TossAPIError("Toss auth rate limited", status=429)
        body = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            desc = ""
            if isinstance(body, dict):
                desc = str(body.get("error_description") or body.get("error") or "")
            raise TossAPIError(desc or f"Toss token failed ({resp.status_code})", status=resp.status_code)
        token = str(body.get("access_token") or "")
        if not token:
            raise TossAPIError("Toss token response missing access_token")
        expires = _to_float(body.get("expires_in"), 3600.0)
        _token = token
        _token_exp = time.time() + max(60.0, expires)
        return token


def _request(
    method: str,
    path: str,
    *,
    params: Optional[dict] = None,
    json_body: Any = None,
    account: bool = False,
    retry_auth: bool = True,
) -> Any:
    if is_rate_limited():
        raise TossAPIError("Toss rate-limit cooldown", status=429)

    headers = {
        "Authorization": f"Bearer {_access_token()}",
        "Accept": "application/json",
    }
    if account:
        seq = resolve_account_seq()
        if seq is None:
            raise TossAPIError("Toss accountSeq is not available.")
        headers["X-Tossinvest-Account"] = str(seq)

    resp = _http_client().request(
        method,
        f"{_base_url()}{path}",
        params=params,
        json=json_body,
        headers=headers,
    )
    if resp.status_code == 429:
        wait = _retry_after(resp)
        mark_rate_limited(wait)
        raise TossAPIError("Toss rate limited", status=429, retry_after=wait)
    if resp.status_code == 401 and retry_auth:
        _access_token(force=True)
        return _request(
            method,
            path,
            params=params,
            json_body=json_body,
            account=account,
            retry_auth=False,
        )
    body = resp.json() if resp.content else {}
    if resp.status_code >= 400:
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            raise TossAPIError(
                str(err.get("message") or err.get("code") or f"Toss API {resp.status_code}"),
                status=resp.status_code,
                code=str(err.get("code") or ""),
            )
        raise TossAPIError(f"Toss API {resp.status_code}", status=resp.status_code)
    return _unwrap(body)


def resolve_account_seq(force: bool = False) -> Optional[int]:
    global _account_seq, _account_no
    with _account_lock:
        if _account_seq is not None and not force:
            return _account_seq
    creds = _creds() or {}
    wanted = str(creds.get("account") or "").strip()
    rows = _request("GET", "/api/v1/accounts") or []
    if not isinstance(rows, list):
        rows = []
    picked: Optional[dict] = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        seq = row.get("accountSeq")
        no = str(row.get("accountNo") or "")
        if wanted and (wanted == str(seq) or wanted == no):
            picked = row
            break
        if picked is None and str(row.get("accountType") or "BROKERAGE") == "BROKERAGE":
            picked = row
    if picked is None and rows:
        picked = rows[0] if isinstance(rows[0], dict) else None
    if not picked:
        return None
    seq = picked.get("accountSeq")
    try:
        seq_i = int(seq)
    except Exception:
        return None
    with _account_lock:
        _account_seq = seq_i
        _account_no = str(picked.get("accountNo") or wanted or seq_i)
    return seq_i


def peek_cached_quote(symbol: str) -> Dict[str, Any]:
    key = (symbol or "").strip().upper()
    with _quote_lock:
        cached = _quote_cache.get(key)
        if cached:
            return dict(cached[1])
    return {}


def _stock_name(symbol: str, market: str = "") -> str:
    key = (symbol or "").strip().upper()
    if key in _name_cache:
        return _name_cache[key]
    try:
        rows = _request("GET", "/api/v1/stocks", params={"symbols": key}) or []
        if isinstance(rows, list) and rows:
            name = str(rows[0].get("name") or rows[0].get("englishName") or "")
            if name:
                _name_cache[key] = name
                return name
    except Exception as e:
        log.debug("Toss stock name failed for %s: %s", key, e)
    return ""


def _daily_change(symbol: str, last_price: float) -> Tuple[float, float, float, float, float, float]:
    """전일 대비 등락. (change, rate%, open, high, low, volume)"""
    try:
        page = _request(
            "GET",
            "/api/v1/candles",
            params={"symbol": symbol, "interval": "1d", "count": 2, "adjusted": True},
        ) or {}
        candles = page.get("candles") if isinstance(page, dict) else page
        rows = list(candles or [])
        if not rows:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        latest = rows[0]
        prev = rows[1] if len(rows) > 1 else None
        open_p = _to_float(latest.get("openPrice"))
        high_p = _to_float(latest.get("highPrice"))
        low_p = _to_float(latest.get("lowPrice"))
        vol = _to_float(latest.get("volume"))
        prev_close = _to_float((prev or {}).get("closePrice")) if prev else 0.0
        change = last_price - prev_close if prev_close else 0.0
        rate = (change / prev_close * 100.0) if prev_close else 0.0
        return change, rate, open_p, high_p, low_p, vol
    except Exception as e:
        log.debug("Toss daily candles failed for %s: %s", symbol, e)
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def get_stock_quote(symbol: str, market: str = "") -> Dict[str, Any]:
    from app.services.symbol_utils import is_plausible_symbol, normalize_symbol

    key = normalize_symbol(symbol, market)
    if not key or not is_plausible_symbol(key, market):
        return {}
    if not is_configured():
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
        rows = _request("GET", "/api/v1/prices", params={"symbols": key}) or []
        row = rows[0] if isinstance(rows, list) and rows else {}
        last = _to_float(row.get("lastPrice"))
        currency = str(row.get("currency") or "")
        mkt = (market or ("KRX" if key.isdigit() else "US")).upper()
        if currency == "KRW":
            mkt = "KRX"
        elif currency == "USD":
            mkt = "US"
        change, rate, open_p, high_p, low_p, vol = _daily_change(key, last)
        name = _stock_name(key, mkt)
        data = {
            "symbol": str(row.get("symbol") or key),
            "name": name,
            "market": mkt,
            "price": last,
            "change": change,
            "rate": rate,
            "volume": vol,
            "amount": last * vol if last and vol else 0.0,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "prev_price": last - change if change else last,
        }
        with _quote_lock:
            _quote_cache[key] = (time.time(), data)
            _quote_fail_cache.pop(key, None)
        return data
    except Exception as e:
        if isinstance(e, TossAPIError) and e.status == 429:
            mark_rate_limited(e.retry_after or _RATE_LIMIT_COOLDOWN)
        log.warning("Toss quote failed for %s: %s", key, e)
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


def get_stock_orderbook(symbol: str, market: str = "") -> Dict[str, Any]:
    from app.services.symbol_utils import is_plausible_symbol, normalize_symbol

    key = normalize_symbol(symbol, market)
    mkt = (market or ("KRX" if (key or "").isdigit() else "US")).upper()
    if not key or not is_plausible_symbol(key, market):
        return _empty_orderbook(str(symbol or "")[:80], mkt, "Invalid symbol")
    if not is_configured():
        return _empty_orderbook(key, mkt, "Toss is not connected")

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
                out["message"] = "Toss rate-limit cooldown (cached)"
                return out
            empty = _empty_orderbook(key, mkt, "Toss rate-limit cooldown")
            empty["rate_limited"] = True
            return empty

        book = _request("GET", "/api/v1/orderbook", params={"symbol": key}) or {}
        asks = [
            {"price": _to_float(x.get("price")), "volume": int(_to_float(x.get("volume")))}
            for x in (book.get("asks") or [])
            if _to_float((x or {}).get("price")) > 0
        ]
        bids = [
            {"price": _to_float(x.get("price")), "volume": int(_to_float(x.get("volume")))}
            for x in (book.get("bids") or [])
            if _to_float((x or {}).get("price")) > 0
        ]
        asks.sort(key=lambda x: x["price"])
        bids.sort(key=lambda x: x["price"], reverse=True)
        asks = asks[:10]
        bids = bids[:10]
        data = {
            "ok": bool(asks or bids),
            "symbol": key,
            "market": mkt,
            "name": _name_cache.get(key) or _stock_name(key, mkt),
            "asks": asks,
            "bids": bids,
            "ask_volume": int(sum(a["volume"] for a in asks)),
            "bid_volume": int(sum(b["volume"] for b in bids)),
            "decimal_places": 0 if key.isdigit() else 2,
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
        limited = isinstance(e, TossAPIError) and e.status == 429
        if limited:
            mark_rate_limited(getattr(e, "retry_after", None) or _RATE_LIMIT_COOLDOWN)
        log.warning("Toss orderbook failed for %s: %s", key, e)
        with _orderbook_lock:
            _orderbook_fail_cache[key] = time.time()
            cached = _orderbook_cache.get(key)
        if cached:
            out = dict(cached[1])
            out["cached"] = True
            out["rate_limited"] = limited
            out["message"] = f"Stale cache after error: {e}"
            return out
        empty = _empty_orderbook(key, mkt, f"Orderbook failed: {e}")
        empty["rate_limited"] = limited
        return empty
    finally:
        with _orderbook_lock:
            ev = _orderbook_inflight.pop(key, None)
        if ev:
            ev.set()


def _candles_page(symbol: str, interval: str, count: int, before: Optional[str] = None) -> Tuple[list, Optional[str]]:
    params: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "count": min(200, max(1, count)),
        "adjusted": True,
    }
    if before:
        params["before"] = before
    page = _request("GET", "/api/v1/candles", params=params) or {}
    if isinstance(page, dict):
        return list(page.get("candles") or []), page.get("nextBefore")
    return list(page or []), None


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    minute_interval: int = 1,
    hour_interval: int = 1,
    pages: int = 2,
) -> pd.DataFrame:
    """토스 캔들(1m/1d)을 가져와 요청 봉 단위로 리샘플합니다."""
    from app.services.indicators import ensure_ohlcv

    if not is_configured() or is_rate_limited():
        return pd.DataFrame()
    tf = (timeframe or "day").lower()
    interval = "1m" if tf in {"realtime", "minute", "hour"} else "1d"
    max_pages = 3 if interval == "1d" else max(1, pages)
    rows: list[dict] = []
    before: Optional[str] = None
    seen = set()
    try:
        for _ in range(max_pages):
            chunk, next_before = _candles_page(symbol, interval, 200, before)
            if not chunk:
                break
            for c in chunk:
                ts = c.get("timestamp")
                if ts in seen:
                    continue
                seen.add(ts)
                rows.append(c)
            if not next_before or next_before == before:
                break
            before = str(next_before)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(
            [
                {
                    "datetime": pd.to_datetime(r.get("timestamp")),
                    "open": _to_float(r.get("openPrice")),
                    "high": _to_float(r.get("highPrice")),
                    "low": _to_float(r.get("lowPrice")),
                    "close": _to_float(r.get("closePrice")),
                    "volume": _to_float(r.get("volume")),
                }
                for r in rows
            ]
        )
        frame = frame.dropna(subset=["datetime"]).set_index("datetime").sort_index()
        df = ensure_ohlcv(frame)
        if df.empty:
            return df
        if tf in {"realtime", "minute"} and int(minute_interval) > 1:
            rule = f"{int(minute_interval)}min"
            df = df.resample(rule).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
        elif tf == "hour":
            rule = f"{max(1, int(hour_interval))}h"
            df = df.resample(rule).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
        elif tf == "week":
            df = df.resample("W-FRI").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
        elif tf == "month":
            df = df.resample("ME").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
        return ensure_ohlcv(df)
    except Exception as e:
        if isinstance(e, TossAPIError) and e.status == 429:
            mark_rate_limited(e.retry_after or _RATE_LIMIT_COOLDOWN)
        log.warning("Toss chart failed for %s: %s", symbol, e)
        return pd.DataFrame()


def _pair_amount(obj: Any, key: str = "") -> Tuple[float, float]:
    """{krw, usd} 또는 중첩 amount 객체에서 KRW/USD를 꺼냅니다."""
    if not isinstance(obj, dict):
        return 0.0, 0.0
    node = obj.get(key, obj) if key else obj
    if isinstance(node, dict) and "amount" in node and isinstance(node["amount"], dict):
        node = node["amount"]
    if not isinstance(node, dict):
        return 0.0, 0.0
    return _to_float(node.get("krw")), _to_float(node.get("usd"))


def _parse_holding(item: dict) -> Dict[str, Any]:
    country = str(item.get("marketCountry") or "").upper()
    currency = str(item.get("currency") or ("KRW" if country == "KR" else "USD")).upper()
    domestic = country in {"KR", "KRX"} or currency == "KRW"
    qty = _to_float(item.get("quantity"))
    price = _to_float(item.get("lastPrice"))
    amount = _to_float((item.get("marketValue") or {}).get("amount"), qty * price)
    profit = _to_float((item.get("profitLoss") or {}).get("amount"))
    rate = _to_float((item.get("profitLoss") or {}).get("rate"))
    # Toss rate는 소수(0.1077=10.77%). UI는 % 표기.
    if abs(rate) <= 1.5:
        rate *= 100.0
    return {
        "name": str(item.get("name") or ""),
        "symbol": str(item.get("symbol") or ""),
        "market": "KRX" if domestic else "US",
        "qty": qty,
        "price": price,
        "amount": amount,
        "profit": profit,
        "profit_rate": rate,
        "currency": currency,
        "scope": "domestic" if domestic else "overseas",
    }


def _buying_power(currency: str) -> float:
    try:
        row = _request("GET", "/api/v1/buying-power", params={"currency": currency}, account=True) or {}
        return _to_float(row.get("cashBuyingPower"))
    except Exception as e:
        log.warning("Toss buying-power %s failed: %s", currency, e)
        return 0.0


def _usd_krw() -> float:
    settings = get_settings()
    try:
        row = _request(
            "GET",
            "/api/v1/exchange-rate",
            params={"baseCurrency": "USD", "quoteCurrency": "KRW"},
        ) or {}
        rate = _to_float(row.get("midRate") or row.get("rate"))
        if rate > 0:
            return rate
    except Exception as e:
        log.debug("Toss FX failed: %s", e)
    return settings.default_exchange_rate


def _fetch_account_overview() -> Dict[str, Any]:
    empty = _empty_account()
    if is_rate_limited():
        empty["error"] = "Toss rate limit cooldown"
        return empty
    if not is_configured():
        return empty
    try:
        seq = resolve_account_seq(force=True)
        if seq is None:
            return {**empty, "error": "토스 종합매매 계좌를 찾지 못했습니다."}
        holdings_raw = _request("GET", "/api/v1/holdings", account=True) or {}
        deposit_krw = _buying_power("KRW")
        deposit_usd = _buying_power("USD")
        usd_fx = _usd_krw()

        items = holdings_raw.get("items") if isinstance(holdings_raw, dict) else []
        holdings = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            parsed = _parse_holding(item)
            if parsed["qty"] <= 0 and parsed["amount"] <= 0:
                continue
            holdings.append(parsed)
            if parsed["name"]:
                _name_cache[parsed["symbol"].upper()] = parsed["name"]

        domestic_holdings = [h for h in holdings if h["scope"] == "domestic"]
        overseas_holdings = [h for h in holdings if h["scope"] == "overseas"]
        domestic_stock_value = sum(h["amount"] for h in domestic_holdings)
        overseas_stock_value = sum(h["amount"] for h in overseas_holdings)

        purchase_krw, purchase_usd = _pair_amount(holdings_raw, "totalPurchaseAmount")
        mv_krw, mv_usd = _pair_amount((holdings_raw or {}).get("marketValue") or {}, "amount")
        pl_krw, pl_usd = _pair_amount((holdings_raw or {}).get("profitLoss") or {}, "amount")
        pl_rate = _to_float(((holdings_raw or {}).get("profitLoss") or {}).get("rate"))
        if abs(pl_rate) <= 1.5:
            pl_rate *= 100.0

        if mv_krw <= 0:
            mv_krw = domestic_stock_value
        if mv_usd <= 0:
            mv_usd = overseas_stock_value

        purchase_amount = purchase_krw + purchase_usd * usd_fx
        current_amount = mv_krw + mv_usd * usd_fx
        profit = pl_krw + pl_usd * usd_fx
        deposit_usd_krw = deposit_usd * usd_fx
        total_eval_krw = current_amount + deposit_krw + deposit_usd_krw

        with _account_lock:
            account_no = _account_no or str(seq)

        return {
            "connected": True,
            "account": account_no,
            "virtual": False,
            "total_eval_krw": round(total_eval_krw, 2),
            "purchase_amount": round(purchase_amount, 2),
            "current_amount": round(current_amount, 2),
            "profit_loss": round(profit, 2),
            "profit_loss_rate": round(pl_rate, 4),
            "deposits": [
                {
                    "currency": "KRW",
                    "amount": deposit_krw,
                    "exchange_rate": 1.0,
                    "amount_krw": deposit_krw,
                    "scope": "domestic",
                },
                {
                    "currency": "USD",
                    "amount": deposit_usd,
                    "exchange_rate": usd_fx,
                    "amount_krw": deposit_usd_krw,
                    "scope": "overseas",
                },
            ],
            "domestic": {
                "deposit_krw": round(deposit_krw, 2),
                "stocks_value": round(domestic_stock_value or mv_krw, 2),
                "holdings": domestic_holdings,
            },
            "overseas": {
                "deposit_usd": round(deposit_usd, 2),
                "deposit_krw": round(deposit_usd_krw, 2),
                "stocks_value": round(overseas_stock_value or mv_usd, 2),
                "stocks_value_krw": round((overseas_stock_value or mv_usd) * usd_fx, 2),
                "exchange_rate": usd_fx,
                "holdings": overseas_holdings,
            },
            "holdings": holdings,
        }
    except Exception as e:
        log.exception("Toss account overview failed: %s", e)
        return {
            **empty,
            "connected": True,
            "error": f"토스 계좌 조회에 실패했습니다. {e}",
        }


def get_account_overview(force: bool = False) -> Dict[str, Any]:
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
    with _account_overview_lock:
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
        if result.get("connected") and not result.get("error"):
            _account_cache = result
            _account_cache_at = time.time()
        elif result.get("error") and _account_cache and _account_cache.get("connected"):
            result = {**_account_cache, "error": result.get("error"), "stale": True}
        elif result.get("connected"):
            _account_cache = result
            _account_cache_at = time.time()
        _account_inflight_result = result
        return dict(result)
    finally:
        with _account_overview_lock:
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
    return get_account_overview().get("holdings") or []


def _qty_str(symbol: str, qty: int | float) -> str:
    if (symbol or "").isdigit() or float(qty).is_integer():
        return str(int(qty))
    text = f"{float(qty):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _price_str(symbol: str, price: int | float) -> str:
    if (symbol or "").isdigit():
        return str(int(round(float(price))))
    value = float(price)
    if value < 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _place_order(
    symbol: str,
    side: str,
    qty: int | float,
    price: Optional[int | float],
) -> Dict[str, Any]:
    if not is_configured():
        return _api_response(False, "Toss API is not connected.")
    try:
        body: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "orderType": "MARKET" if price is None else "LIMIT",
            "quantity": _qty_str(symbol, qty),
            "confirmHighValueOrder": True,
        }
        if price is not None:
            body["price"] = _price_str(symbol, price)
        result = _request("POST", "/api/v1/orders", json_body=body, account=True) or {}
        order_id = result.get("orderId") if isinstance(result, dict) else None
        return _api_response(True, f"{side.title()} order accepted", {"order_id": order_id} if order_id else {})
    except Exception as e:
        return _api_response(False, f"{side.title()} order failed: {e}")


def place_buy_order(symbol: str, qty: int | float, price: Optional[int | float] = None) -> Dict[str, Any]:
    return _place_order(symbol, "BUY", qty, price)


def place_sell_order(
    symbol: str,
    qty: Optional[int | float] = None,
    price: Optional[int | float] = None,
) -> Dict[str, Any]:
    if qty is None:
        return _api_response(False, "Sell quantity is required.")
    return _place_order(symbol, "SELL", qty, price)


def get_pending_orders() -> List[Dict[str, Any]]:
    if not is_configured():
        return []
    try:
        page = _request("GET", "/api/v1/orders", params={"status": "OPEN"}, account=True) or {}
        orders = page.get("orders") if isinstance(page, dict) else page
        rows = []
        for order in orders or []:
            qty = _to_float(order.get("quantity"))
            filled = _to_float((order.get("execution") or {}).get("filledQuantity"))
            pending = max(0.0, qty - filled)
            symbol = str(order.get("symbol") or "")
            rows.append({
                "name": _name_cache.get(symbol.upper()) or "",
                "symbol": symbol,
                "side": str(order.get("side") or ""),
                "qty": qty,
                "price": _to_float(order.get("price")),
                "pending_qty": int(pending) if pending.is_integer() else int(pending),
                "order_id": str(order.get("orderId") or ""),
            })
        return rows
    except Exception as e:
        log.exception("Toss pending orders failed: %s", e)
        return []


def cancel_order(symbol: str, order_id: Optional[str] = None) -> Dict[str, Any]:
    if not is_configured():
        return _api_response(False, "Toss API is not connected.")
    try:
        oid = (order_id or "").strip()
        if not oid:
            for od in get_pending_orders():
                if od.get("symbol") == symbol:
                    oid = str(od.get("order_id") or "")
                    break
        if not oid:
            return _api_response(False, "No pending order found.")
        _request("POST", f"/api/v1/orders/{oid}/cancel", json_body={}, account=True)
        return _api_response(True, "Cancel accepted", {"order_id": oid, "symbol": symbol})
    except Exception as e:
        return _api_response(False, f"Cancel failed: {e}")
