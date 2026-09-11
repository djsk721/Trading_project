"""Dividend / ex-dividend calendar helpers.

데이터가 없으면 추정하지 않고 빈 결과를 반환한다.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any

from app.core.config import PROJECT_ROOT, get_settings
from app.services import broker
from app.services.recommend_cache import load_daily, today_stamp

log = logging.getLogger(__name__)

_CACHE_DIR = PROJECT_ROOT / ".cache"
_CACHE_FILE = _CACHE_DIR / "dividend_calendar.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value or 0)
        return out if out == out else default
    except Exception:
        return default


def _parse_date(value: str | None, fallback: date) -> date:
    if not value:
        return fallback
    try:
        return datetime.fromisoformat(value[:10]).date()
    except Exception:
        return fallback


def _load_cache(key: str, ttl: int) -> dict[str, Any] | None:
    try:
        if not _CACHE_FILE.exists():
            return None
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        item = data.get(key)
        if not isinstance(item, dict):
            return None
        if time.time() - float(item.get("cached_at") or 0) > ttl:
            return None
        payload = item.get("payload")
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        log.warning("dividend cache load failed: %s", exc)
        return None


def _save_cache(key: str, payload: dict[str, Any]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8")) if _CACHE_FILE.exists() else {}
        data[key] = {"cached_at": time.time(), "payload": payload}
        _CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("dividend cache save failed: %s", exc)


def _normalize_symbol(symbol: str, market: str) -> str:
    s = str(symbol or "").strip().upper()
    if market == "KRX" and s.isdigit() and not s.endswith(".KS"):
        return f"{s}.KS"
    return s


def _holding_symbols() -> list[dict[str, str]]:
    try:
        account = broker.get_account_overview(force=False)
    except Exception:
        return []
    rows = []
    for h in account.get("holdings") or []:
        sym = str(h.get("symbol") or "").strip().upper()
        if not sym:
            continue
        market = str(h.get("market") or ("KRX" if sym.isdigit() else "US")).upper()
        rows.append({"symbol": sym, "name": str(h.get("name") or sym), "market": market, "source": "holdings"})
    return rows


def _recommend_symbols() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for market in ("ALL", "KRX", "US"):
        payload = load_daily(market, as_of=today_stamp()) or {}
        for item in list(payload.get("items") or []) + list(payload.get("scan_items") or []):
            sym = str(item.get("symbol") or "").strip().upper()
            mkt = str(item.get("market") or ("KRX" if sym.isdigit() else "US")).upper()
            key = (mkt, sym)
            if sym and key not in seen:
                seen.add(key)
                rows.append({"symbol": sym, "name": str(item.get("name") or sym), "market": mkt, "source": "recommend"})
    return rows


def _target_symbols(scope: str, symbols: str) -> list[dict[str, str]]:
    scope = str(scope or "holdings").lower()
    rows: list[dict[str, str]] = []
    if symbols.strip():
        for raw in symbols.split(","):
            sym = raw.strip().upper()
            if sym:
                rows.append({"symbol": sym, "name": sym, "market": "KRX" if sym.isdigit() else "US", "source": "manual"})
    elif scope == "recommend":
        rows = _recommend_symbols()
    elif scope == "all":
        rows = _holding_symbols() + _recommend_symbols()
    else:
        rows = _holding_symbols()

    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (row["market"], row["symbol"])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out[:20]


def _event(row: dict[str, str], ex_date: date, pay_date: date | None, amount: float, source: str) -> dict[str, Any]:
    ex_date = _coerce_date(ex_date) or date.today()
    pay_date = _coerce_date(pay_date) if pay_date else None
    dday = (ex_date - date.today()).days
    return {
        "symbol": row["symbol"],
        "name": row.get("name") or row["symbol"],
        "market": row.get("market") or "US",
        "ex_date": ex_date.isoformat(),
        "pay_date": pay_date.isoformat() if pay_date else None,
        "amount": amount,
        "currency": "KRW" if row.get("market") == "KRX" else "USD",
        "yield_pct": 0.0,
        "d_day": dday,
        "source": source,
        "scope": row.get("source") or "manual",
    }


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime().date()
        except Exception:
            return None
    if isinstance(value, datetime):
        return value.date()
    if type(value) is date:
        return value
    if hasattr(value, "date"):
        try:
            return value.date()
        except Exception:
            return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except Exception:
        return None


def _read_yfinance_dividends(row: dict[str, str], start: date, end: date) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return [], None
    ticker = _normalize_symbol(row["symbol"], row["market"])
    try:
        t = yf.Ticker(ticker)
        events: list[dict[str, Any]] = []
        cal = getattr(t, "calendar", None)
        if isinstance(cal, dict):
            ex_date = cal.get("Ex-Dividend Date") or cal.get("exDividendDate")
            pay_date = cal.get("Dividend Date") or cal.get("Pay Date") or cal.get("dividendDate")
            amount = cal.get("Dividend Rate") or cal.get("Dividend") or cal.get("dividendRate")
            if isinstance(ex_date, (list, tuple)):
                ex_date = ex_date[0] if ex_date else None
            if isinstance(pay_date, (list, tuple)):
                pay_date = pay_date[0] if pay_date else None
            ex_d = _coerce_date(ex_date)
            pay_d = _coerce_date(pay_date)
            if ex_d and start <= ex_d <= end:
                events.append(_event(row, ex_d, pay_d, _safe_float(amount), "yfinance:calendar"))
        divs = getattr(t, "dividends", None)
        if divs is not None and len(divs) > 0:
            last_event = None
            for idx, amount in divs.tail(8).items():
                d = _coerce_date(idx)
                if d:
                    last_event = _event(row, d, None, _safe_float(amount), "yfinance:dividends:last")
                if d and start <= d <= end:
                    events.append(_event(row, d, None, _safe_float(amount), "yfinance:dividends"))
            return events, last_event
        return events, None
    except Exception as exc:
        log.info("dividend fetch skipped %s: %s", ticker, exc)
        return [], None


def build_dividend_calendar(
    start: str | None = None,
    end: str | None = None,
    scope: str = "holdings",
    symbols: str = "",
    force: bool = False,
) -> dict[str, Any]:
    settings = get_settings()
    today = date.today()
    start_d = _parse_date(start, today)
    end_d = _parse_date(end, today + timedelta(days=60))
    if end_d < start_d:
        start_d, end_d = end_d, start_d
    key = f"v2:{start_d}:{end_d}:{scope}:{symbols.strip().upper()}"
    if not force:
        cached = _load_cache(key, int(settings.dividend_cache_ttl_seconds))
        if cached:
            cached["cached"] = True
            return cached

    targets = _target_symbols(scope, symbols)
    events: list[dict[str, Any]] = []
    recent: list[dict[str, Any]] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_read_yfinance_dividends, row, start_d, end_d) for row in targets]
        for fut in as_completed(futs):
            try:
                found, last = fut.result()
            except Exception as exc:
                log.info("dividend worker failed: %s", exc)
                continue
            events.extend(found)
            if last:
                recent.append(last)
    unique = {(e["market"], e["symbol"], e["ex_date"]): e for e in events}
    out_events = sorted(unique.values(), key=lambda e: (e["ex_date"], e["symbol"]))
    recent_unique = {(e["market"], e["symbol"], e["ex_date"]): e for e in recent}
    recent_events = sorted(recent_unique.values(), key=lambda e: (e["ex_date"], e["symbol"]), reverse=True)
    payload = {
        "as_of": today.isoformat(),
        "from": start_d.isoformat(),
        "to": end_d.isoformat(),
        "scope": scope,
        "target_count": len(targets),
        "items": out_events,
        "recent_items": recent_events[:40],
        "cached": False,
        "source_note": "향후 배당락일은 yfinance calendar/dividends에서 확인된 경우만 표시합니다. 향후 일정이 없으면 실제 과거 배당락일을 참고용으로 보여주며, 미래 날짜는 추정하지 않습니다.",
    }
    _save_cache(key, payload)
    return payload
