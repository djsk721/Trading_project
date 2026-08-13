"""일일 추천용 유동 유니버스 — yfinance screen 랭킹 + 일 1회 캐시."""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.config import PROJECT_ROOT, get_settings
from app.services.market_data import POPULAR_KR, POPULAR_US

log = logging.getLogger(__name__)

_CACHE_DIR = PROJECT_ROOT / ".cache"
_TTL_SEC = 24 * 3600
_YF_SUFFIX_RE = re.compile(r"\.(KS|KQ)$", re.IGNORECASE)


def _cache_path(market: str) -> Path:
    return _CACHE_DIR / f"recommend_universe_{market.upper()}.json"


def _load_cache(market: str, ttl: int) -> Optional[dict]:
    path = _cache_path(market)
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        # 구버전(seed_*) 캐시는 무시하고 재구축
        source = str(data.get("source") or "")
        if source.startswith("seed_") or source in {"pykrx", "none"}:
            return None
        if not source.startswith("yfinance"):
            # 예전 pykrx:YYYYMMDD 등도 재구축
            if "yfinance" not in source:
                return None
        ts = float(data.get("cached_at") or 0)
        if time.time() - ts > ttl:
            return None
        items = data.get("items") or []
        if not items:
            return None
        return data
    except Exception as e:
        log.warning("universe cache load failed: %s", e)
        return None


def _save_cache(market: str, payload: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(market).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning("universe cache save failed: %s", e)


def _norm_symbol(symbol: str, market: str) -> str:
    sym = (symbol or "").strip().upper()
    if market == "KRX":
        sym = _YF_SUFFIX_RE.sub("", sym)
        if sym.isdigit() and len(sym) < 6:
            sym = sym.zfill(6)
    return sym


def _is_kr_preferred(symbol: str, name: str) -> bool:
    """우선주 대략 필터 (추천 유니버스에서 제외)."""
    n = (name or "").lower()
    if "prefer" in n or "(1p)" in n or "1p)" in n or "우)" in (name or ""):
        return True
    # 국내 우선주는 끝자리 5/7/8/K 등이 흔함 — 이름에 단서가 있을 때만 제외
    return False


def _screen_quotes(
    region: str,
    *,
    sort_field: str,
    size: int,
    min_market_cap: float,
) -> List[dict]:
    """yfinance EquityQuery screen."""
    try:
        import yfinance as yf
        from yfinance import EquityQuery
    except Exception as e:
        log.warning("yfinance screen import failed: %s", e)
        return []

    try:
        query = EquityQuery(
            "and",
            [
                EquityQuery("eq", ["region", region]),
                EquityQuery("gt", ["intradaymarketcap", float(min_market_cap)]),
            ],
        )
        res = yf.screen(query, sortField=sort_field, sortAsc=False, size=int(size))
        quotes = (res or {}).get("quotes") or []
        return [q for q in quotes if isinstance(q, dict) and q.get("symbol")]
    except Exception as e:
        log.warning("yfinance screen failed region=%s sort=%s: %s", region, sort_field, e)
        return []


def _merge_ranked(
    buckets: List[List[dict]],
    *,
    market: str,
    limit: int,
) -> Dict[str, dict]:
    """symbol -> {name, yf_symbol}"""
    us_exchanges = {"NMS", "NYQ", "NGM", "ASE", "PCX", "BTS"}
    ordered: Dict[str, dict] = {}
    for bucket in buckets:
        for q in bucket:
            raw = str(q.get("symbol") or "").strip()
            sym = _norm_symbol(raw, market)
            if not sym or sym in ordered:
                continue
            name = (
                str(q.get("shortName") or q.get("longName") or q.get("displayName") or sym)
                .strip()
            )
            if market == "KRX" and _is_kr_preferred(sym, name):
                continue
            if market == "US":
                exch = str(q.get("exchange") or "").upper()
                if exch and exch not in us_exchanges:
                    continue
                if len(sym) > 5 or not re.match(r"^[A-Z]{1,5}$", sym):
                    continue
            yf_sym = raw.upper() if market == "KRX" else sym
            # screen이 이미 .KS/.KQ를 주면 그대로 사용
            if market == "KRX" and not _YF_SUFFIX_RE.search(yf_sym):
                yf_sym = f"{sym}.KS"
            ordered[sym] = {"name": name or sym, "yf_symbol": yf_sym}
            if len(ordered) >= limit:
                return ordered
    return ordered


def _build_krx_yfinance(limit: int, head: int) -> Tuple[Dict[str, dict], str]:
    """KR: 시총·거래량 상위 합집합. Yahoo 심볼(.KS/.KQ) 보존."""
    min_cap = 300_000_000_000
    fetch_n = max(head + 20, min(head * 2, 150))
    by_cap = _screen_quotes(
        "kr",
        sort_field="intradaymarketcap",
        size=fetch_n,
        min_market_cap=min_cap,
    )
    by_vol = _screen_quotes(
        "kr",
        sort_field="dayvolume",
        size=fetch_n,
        min_market_cap=min_cap,
    )
    items = _merge_ranked(
        [by_vol[:head], by_cap[:head]],
        market="KRX",
        limit=limit,
    )
    if len(items) >= max(15, limit // 10):
        return items, "yfinance_screen:kr"
    return {}, "none"


def _build_us_yfinance(limit: int, head: int) -> Tuple[Dict[str, dict], str]:
    """US: 시총·거래량 상위 합집합."""
    min_cap_large = 10_000_000_000
    min_cap_liquid = 5_000_000_000
    fetch_n = max(head + 20, min(head * 2, 150))
    by_cap = _screen_quotes(
        "us",
        sort_field="intradaymarketcap",
        size=fetch_n,
        min_market_cap=min_cap_large,
    )
    by_vol = _screen_quotes(
        "us",
        sort_field="dayvolume",
        size=fetch_n,
        min_market_cap=min_cap_liquid,
    )
    items = _merge_ranked(
        [by_vol[:head], by_cap[:head]],
        market="US",
        limit=limit,
    )
    if len(items) >= max(15, limit // 10):
        return items, "yfinance_screen:us"
    return {}, "none"


def get_recommend_universe(market: str = "KRX", force: bool = False) -> dict:
    """
    Returns:
      { market, as_of, source, cached, size,
        items: [{symbol, name, yf_symbol}, ...] }
    """
    settings = get_settings()
    limit = int(getattr(settings, "recommend_universe_size", 100) or 100)
    head = int(getattr(settings, "recommend_universe_head", 80) or 80)
    ttl = int(getattr(settings, "recommend_universe_ttl_seconds", _TTL_SEC) or _TTL_SEC)
    mkt = "KRX" if market.upper() == "KRX" else "US"

    if not force:
        cached = _load_cache(mkt, ttl)
        if cached:
            items = cached.get("items") or []
            # yf_symbol 없는 구캐시는 재구축
            if items and not any(it.get("yf_symbol") for it in items):
                cached = None
            else:
                return {
                    "market": mkt,
                    "as_of": cached.get("as_of") or datetime.now().strftime("%Y-%m-%d"),
                    "source": cached.get("source") or "cache",
                    "cached": True,
                    "size": len(items),
                    "items": items,
                }

    mapping: Dict[str, dict] = {}
    source = "none"
    if mkt == "KRX":
        mapping, source = _build_krx_yfinance(limit=limit, head=head)
    else:
        mapping, source = _build_us_yfinance(limit=limit, head=head)

    if len(mapping) < max(8, limit // 20):
        popular = POPULAR_KR if mkt == "KRX" else POPULAR_US
        mapping = {
            s: {
                "name": n,
                "yf_symbol": f"{s}.KS" if mkt == "KRX" else s,
            }
            for s, n in popular.items()
        }
        source = "popular_fallback"

    trimmed = dict(list(mapping.items())[:limit])
    as_of = datetime.now().strftime("%Y-%m-%d")
    items = [
        {
            "symbol": s,
            "name": meta.get("name") or s,
            "yf_symbol": meta.get("yf_symbol") or s,
        }
        for s, meta in trimmed.items()
    ]
    payload = {
        "market": mkt,
        "as_of": as_of,
        "source": source,
        "cached_at": time.time(),
        "items": items,
    }
    _save_cache(mkt, payload)
    return {
        "market": mkt,
        "as_of": as_of,
        "source": source,
        "cached": False,
        "size": len(items),
        "items": items,
    }


def universe_as_mapping(market: str = "KRX", force: bool = False) -> Tuple[Dict[str, str], dict]:
    meta = get_recommend_universe(market, force=force)
    mapping = {it["symbol"]: it.get("name") or it["symbol"] for it in meta.get("items") or []}
    return mapping, meta


def universe_yf_map(meta: dict) -> Dict[str, str]:
    """symbol -> Yahoo ticker (예: 125490.KQ)."""
    out: Dict[str, str] = {}
    for it in meta.get("items") or []:
        sym = it.get("symbol")
        if not sym:
            continue
        out[str(sym)] = str(it.get("yf_symbol") or sym)
    return out
