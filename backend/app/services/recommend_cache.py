"""일일 추천 결과 캐시 — 시장별·당일 유지, 명시적 업데이트 전까지 재생성하지 않음."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.core.config import PROJECT_ROOT

log = logging.getLogger(__name__)

_CACHE_DIR = PROJECT_ROOT / ".cache"
_lock = threading.Lock()


def normalize_recommend_market(market: str) -> str:
    raw = (market or "ALL").upper().strip()
    if raw in {"KRX", "KR", "KO", "KOSPI", "KOSDAQ", "DOMESTIC"}:
        return "KRX"
    if raw in {"US", "USA", "NASDAQ", "NYSE", "OVERSEAS"}:
        return "US"
    return "ALL"


def markets_for(market: str) -> list[str]:
    out = normalize_recommend_market(market)
    return ["KRX", "US"] if out == "ALL" else [out]


def today_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _cache_path(market: str) -> Path:
    key = normalize_recommend_market(market)
    return _CACHE_DIR / f"recommend_daily_{key}.json"


def load_daily(market: str, as_of: str | None = None) -> Optional[dict]:
    """당일 저장된 추천이 있으면 반환. 날짜가 바뀌면 무효."""
    day = as_of or today_stamp()
    path = _cache_path(market)
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        payload = data.get("payload")
        if not isinstance(payload, dict):
            return None
        cached_day = str(data.get("as_of") or payload.get("as_of") or "")
        if cached_day != day:
            return None
        if not payload.get("items") and not payload.get("scan_items"):
            return None
        return payload
    except Exception as e:
        log.warning("recommend daily cache load failed: %s", e)
        return None


def save_daily(market: str, payload: dict[str, Any]) -> None:
    key = normalize_recommend_market(market)
    as_of = str(payload.get("as_of") or today_stamp())
    body = {
        "market": key,
        "as_of": as_of,
        "cached_at": time.time(),
        "payload": payload,
    }
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with _lock:
            _cache_path(key).write_text(
                json.dumps(body, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception as e:
        log.warning("recommend daily cache save failed: %s", e)
