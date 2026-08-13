"""실시간에 가까운 매크로 지표 스냅샷 (yfinance)."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import PROJECT_ROOT

log = logging.getLogger(__name__)

_CACHE_DIR = PROJECT_ROOT / ".cache"
_DISK_PATH = _CACHE_DIR / "macro_snapshots.json"
_TTL_SEC = 90  # UI 갱신·프롬프트용 짧은 TTL

_lock = threading.Lock()
_mem: Dict[str, Any] = {"fetched_at": 0.0, "payload": None}

# id, yfinance ticker, display label, unit hint
_MACRO_SPECS: List[tuple[str, str, str, str]] = [
    ("kospi", "^KS11", "KOSPI", "pt"),
    ("kosdaq", "^KQ11", "KOSDAQ", "pt"),
    ("spx", "^GSPC", "S&P 500", "pt"),
    ("nasdaq", "^IXIC", "Nasdaq", "pt"),
    ("usdkrw", "KRW=X", "USD/KRW", "fx"),
    ("wti", "CL=F", "WTI", "usd"),
    ("vix", "^VIX", "VIX", "pt"),
    ("us10y", "^TNX", "US 10Y", "pct"),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_price(value: float, unit: str) -> str:
    if value is None:
        return "-"
    if unit == "fx":
        return f"{value:,.2f}"
    if unit == "pct":
        return f"{value:.2f}%"
    if unit == "usd":
        return f"{value:,.2f}"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:.2f}"


def _one_ticker(ticker: str) -> Optional[Dict[str, float]]:
    import yfinance as yf

    hist = yf.Ticker(ticker).history(period="10d", auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    closes = hist["Close"].dropna()
    if len(closes) < 1:
        return None
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) >= 2 else last
    change = last - prev
    change_pct = (change / prev * 100.0) if prev else 0.0
    return {
        "price": last,
        "prev": prev,
        "change": change,
        "change_pct": change_pct,
    }


def _fetch_live() -> dict:
    items: List[dict] = []
    errors: List[str] = []
    for mid, ticker, label, unit in _MACRO_SPECS:
        try:
            row = _one_ticker(ticker)
            if not row:
                errors.append(mid)
                items.append({
                    "id": mid,
                    "ticker": ticker,
                    "label": label,
                    "unit": unit,
                    "price": None,
                    "prev": None,
                    "change": None,
                    "change_pct": None,
                    "price_text": "-",
                    "change_text": "-",
                    "ok": False,
                })
                continue
            ch = row["change_pct"]
            sign = "+" if ch > 0 else ""
            items.append({
                "id": mid,
                "ticker": ticker,
                "label": label,
                "unit": unit,
                "price": round(row["price"], 4),
                "prev": round(row["prev"], 4),
                "change": round(row["change"], 4),
                "change_pct": round(ch, 3),
                "price_text": _fmt_price(row["price"], unit),
                "change_text": f"{sign}{ch:.2f}%",
                "ok": True,
            })
        except Exception as e:
            log.warning("macro fetch failed %s: %s", mid, e)
            errors.append(mid)
            items.append({
                "id": mid,
                "ticker": ticker,
                "label": label,
                "unit": unit,
                "price": None,
                "prev": None,
                "change": None,
                "change_pct": None,
                "price_text": "-",
                "change_text": "-",
                "ok": False,
            })

    return {
        "as_of": _now_iso(),
        "source": "yfinance",
        "ttl_sec": _TTL_SEC,
        "items": items,
        "ok_count": sum(1 for i in items if i.get("ok")),
        "errors": errors,
    }


def _read_disk() -> Optional[dict]:
    try:
        if not _DISK_PATH.exists():
            return None
        data = json.loads(_DISK_PATH.read_text(encoding="utf-8"))
        as_of = data.get("as_of") or ""
        ts = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
        if age <= _TTL_SEC and data.get("items"):
            return data
    except Exception:
        pass
    return None


def _write_disk(payload: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _DISK_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning("macro disk cache save failed: %s", e)


def fetch_macro_snapshots(force: bool = False) -> dict:
    """매크로 스냅샷 (메모리·디스크 TTL 캐시)."""
    now = time.time()
    with _lock:
        mem_payload = _mem.get("payload")
        mem_at = float(_mem.get("fetched_at") or 0)
        if not force and mem_payload and (now - mem_at) < _TTL_SEC:
            return {**mem_payload, "cached": True}

    if not force:
        disk = _read_disk()
        if disk:
            with _lock:
                _mem["payload"] = disk
                _mem["fetched_at"] = now
            return {**disk, "cached": True}

    payload = _fetch_live()
    payload["cached"] = False
    with _lock:
        _mem["payload"] = payload
        _mem["fetched_at"] = time.time()
    _write_disk(payload)
    return payload


def format_macros_for_prompt(macros: Optional[dict] = None) -> str:
    """LLM에 넣을 사실 지표 블록."""
    data = macros if macros is not None else fetch_macro_snapshots(force=False)
    items = data.get("items") or []
    lines = []
    for it in items:
        if not it.get("ok"):
            continue
        lines.append(
            f"- {it['label']}: {it['price_text']} ({it['change_text']} vs prev close)"
        )
    if not lines:
        return "LIVE_MACROS: (unavailable)"
    as_of = (data.get("as_of") or "")[:19].replace("T", " ")
    return (
        "LIVE_MACROS (use ONLY these figures for prices/levels; "
        f"as_of UTC {as_of}):\n" + "\n".join(lines)
    )


def macros_fingerprint(macros: Optional[dict] = None) -> str:
    data = macros if macros is not None else fetch_macro_snapshots(force=False)
    parts = []
    for it in data.get("items") or []:
        if it.get("ok") and it.get("price") is not None:
            # 소수 첫째자리까지로 fingerprint (미세 변동에 다이제스트 폭주 방지)
            parts.append(f"{it['id']}:{round(float(it['price']), 1)}")
    return "|".join(parts)
