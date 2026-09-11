"""종목 관련 최신 뉴스 검색 (블로그 제외, 기사 원문 링크 우선)."""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse
from xml.etree import ElementTree

import httpx

from app.core.config import get_settings
from app.services.market_data import resolve_stock_name
from app.services.news_article_cache import enrich_and_sort_items, prepare_articles_async
from app.services.symbol_utils import looks_like_kr_ticker, normalize_symbol

log = logging.getLogger(__name__)

# 뉴스 매칭용 정식명 캐시 (ticker -> (expires_at, identity))
_IDENTITY_TTL = 3600.0
_identity_cache: Dict[str, Tuple[float, dict]] = {}
_LEGAL_SUFFIX = re.compile(
    r",?\s+(inc\.?|incorporated|corp\.?|corporation|co\.?|ltd\.?|llc\.?|"
    r"holdings|group|plc|sa|ag|nv|주식회사|㈜)\s*\.?$",
    re.I,
)

# 블로그/커뮤니티성 소스 제외
_BLOG_PATTERNS = re.compile(
    r"("
    r"blog\.naver|naver\.blog|tistory\.com|blog\.|wordpress\.|/blogs?/|"
    r"medium\.com|brunch\.co\.kr|velog\.io|tumblr\.com|"
    r"cafe\.naver|cafe\.daum|reddit\.com|dcinside|"
    r"티스토리|네이버\s*블로그|블로그"
    r")",
    re.IGNORECASE,
)

_BLOG_TITLE_HINTS = re.compile(
    r"(블로그|티스토리|브런치|velog|워드프레스|개인\s*칼럼)",
    re.IGNORECASE,
)


def _parse_rss_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except Exception:
            return value


def _clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _extract_original_url(link: str, description_html: str) -> str:
    """Google News 리다이렉트 대신 description 내 게시사 원문 URL을 우선 사용."""
    # description 안의 첫 외부 링크
    for href in re.findall(r'href=["\']([^"\']+)["\']', description_html or "", flags=re.I):
        if "news.google." in href:
            continue
        if href.startswith("http"):
            return href
    return link or ""


def _is_blog_item(title: str, source: str, url: str) -> bool:
    blob = f"{title} {source} {url}"
    if _BLOG_PATTERNS.search(blob):
        return True
    if _BLOG_TITLE_HINTS.search(title or ""):
        return True
    host = urlparse(url).netloc.lower()
    if any(x in host for x in ("blog", "tistory", "medium.com", "brunch", "velog")):
        return True
    return False


def _uniq_keep_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in values:
        val = re.sub(r"\s+", " ", (raw or "").strip())
        if not val:
            continue
        key = val.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def _is_weak_alias(name: str, symbol: str, *, kr_listed: bool) -> bool:
    """짧은 별칭은 타 종목(스냅→바이오스냅)에 부분일치하므로 매칭 키로 쓰지 않습니다."""
    n = (name or "").strip()
    if not n:
        return True
    # 티커 자체는 별도 심볼 매칭을 쓰므로, Snap==SNAP 같은 짧은 영문은 약칭으로 봅니다.
    if kr_listed:
        return False
    if _LEGAL_SUFFIX.search(n):
        return False
    if re.fullmatch(r"[가-힣]{1,4}", n):
        return True
    if re.fullmatch(r"[A-Za-z]{1,5}", n):
        return True
    return False


def _name_variants(name: str) -> List[str]:
    """Snap Inc. / Snap, Inc. 처럼 구두점만 다른 정식명 변형."""
    base = re.sub(r"\s+", " ", (name or "").strip())
    if not base:
        return []
    variants = [base, base.replace(",", "")]
    compact = re.sub(r"[,.]", "", base)
    compact = re.sub(r"\s+", " ", compact).strip()
    if compact:
        variants.append(compact)
    no_suffix = _LEGAL_SUFFIX.sub("", base).strip(" ,.")
    # 접미사 제거한 짧은 영문(Snap)은 변형에 넣지 않음
    if no_suffix and not _is_weak_alias(no_suffix, "", kr_listed=False):
        variants.append(no_suffix)
    return _uniq_keep_order(variants)


def _isolated_pattern(term: str) -> re.Pattern[str]:
    """한글·영문이 붙은 다른 상호(바이오스냅)는 제외하고 토큰 단위로 매칭."""
    escaped = re.escape(term)
    return re.compile(
        rf"(?<![A-Za-z0-9가-힣]){escaped}(?![A-Za-z0-9가-힣])",
        re.IGNORECASE,
    )


def _us_official_names(symbol: str) -> List[str]:
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
    except Exception as exc:
        log.debug("news identity yfinance failed for %s: %s", symbol, exc)
        return []
    names: List[str] = []
    for key in ("longName", "shortName"):
        val = str(info.get(key) or "").strip()
        if val:
            names.append(val)
    return names


def resolve_news_identity(symbol: str, market: str = "KRX", stock_name: Optional[str] = None) -> dict:
    """뉴스 검색·필터에 쓸 종목코드와 정식 회사명."""
    key = normalize_symbol(symbol, market) or (symbol or "").strip().upper()
    cache_key = f"{(market or '').upper()}:{key}"
    now = time.time()
    cached = _identity_cache.get(cache_key)
    if cached and cached[0] > now:
        ident = dict(cached[1])
        if stock_name and stock_name not in ident["display_names"]:
            ident["display_names"] = _uniq_keep_order([stock_name, *ident["display_names"]])
        return ident

    kr_listed = looks_like_kr_ticker(key)
    listed_name = resolve_stock_name(key, market)
    display = (stock_name or "").strip() or listed_name
    official: List[str] = []
    if kr_listed:
        official.append(listed_name)
        # 짧은 별칭(삼성)은 계열사 뉴스까지 끌어오므로 상장 정식명만 사용
        extra = display
        if (
            extra
            and extra.casefold() != listed_name.casefold()
            and extra not in listed_name
            and listed_name not in extra
            and not _is_weak_alias(extra, key, kr_listed=False)
        ):
            official.append(extra)
    else:
        official.extend(_us_official_names(key))
        if display and not _is_weak_alias(display, key, kr_listed=False):
            official.append(display)

    names: List[str] = []
    for raw in official:
        names.extend(_name_variants(raw))
    names = [n for n in _uniq_keep_order(names) if not _is_weak_alias(n, key, kr_listed=kr_listed)]

    ident = {
        "symbol": key,
        "market": (market or "").upper() or ("KRX" if looks_like_kr_ticker(key) else "US"),
        "kr_listed": kr_listed,
        "display": display or key,
        "display_names": _uniq_keep_order([display, key]),
        "names": names,
    }
    _identity_cache[cache_key] = (now + _IDENTITY_TTL, ident)
    return ident


def _item_text(item: dict) -> str:
    return " ".join(
        str(item.get(k) or "")
        for k in ("title", "title_original", "summary", "url")
    )


def news_item_matches_identity(item: dict, identity: dict) -> bool:
    """종목코드 또는 정식 회사명이 토큰으로 있을 때만 해당 종목 뉴스로 인정."""
    blob = _item_text(item)
    if not blob.strip():
        return False
    symbol = str(identity.get("symbol") or "").strip()
    if symbol:
        if _isolated_pattern(symbol).search(blob):
            return True
        # 국내 6자리는 URL/본문에 하이픈이 끼는 경우도 허용
        if looks_like_kr_ticker(symbol) and re.search(
            rf"(?<!\d){symbol[0:3]}-?{symbol[3:6]}(?!\d)", blob
        ):
            return True
    for name in identity.get("names") or []:
        if len(name) < 2:
            continue
        if _isolated_pattern(name).search(blob):
            return True
    return False


def _search_queries(identity: dict) -> List[str]:
    symbol = identity["symbol"]
    names = identity.get("names") or []
    if identity.get("kr_listed"):
        queries = [f'"{symbol}"', f"{symbol} 주식"]
        for name in names[:2]:
            queries.append(f'"{name}" 주식')
            queries.append(f'"{name}" {symbol}')
        return _uniq_keep_order(queries)

    queries = [f'"{symbol}" stock', f"{symbol} earnings", f'"{symbol}" 주식']
    for name in names[:2]:
        queries.append(f'"{name}"')
        queries.append(f'"{name}" stock')
    return _uniq_keep_order(queries)


def _google_news_rss(query: str, language: str = "ko", max_items: int = 40) -> List[dict]:
    """Google News RSS 검색 (기사 위주 쿼리)."""
    hl = "ko" if language.startswith("ko") else "en"
    gl = "KR" if hl == "ko" else "US"
    # 블로그 제외 키워드 + 최근성
    refined = f"{query} when:7d -blog -블로그 -tistory -brunch -velog"
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(refined)}&hl={hl}&gl={gl}&ceid={gl}:{hl}"
    )
    items: List[dict] = []
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
        root = ElementTree.fromstring(resp.text)
        channel = root.find("channel")
        if channel is None:
            return []
        for item in channel.findall("item")[:max_items]:
            raw_desc = item.findtext("description", "") or ""
            title = _clean_html(item.findtext("title", ""))
            # 제목에서 " - Source" 형태 정리
            title = re.sub(r"\s+[-:|]\s+[^-:|]{2,40}$", "", title).strip() or title
            g_link = item.findtext("link", "") or ""
            link = _extract_original_url(g_link, raw_desc)
            desc = _clean_html(raw_desc)
            pub = _parse_rss_date(item.findtext("pubDate"))
            source_el = item.find("source")
            source = (
                source_el.text.strip()
                if source_el is not None and source_el.text
                else "News"
            )
            if not title or not link:
                continue
            if _is_blog_item(title, source, link):
                continue
            items.append({
                "title": title,
                "summary": desc[:400],
                "url": link,
                "source": source,
                "published_at": pub,
            })
    except Exception as e:
        log.warning("Google News RSS failed: %s", e)
    return items


def _yfinance_news(symbol: str, max_items: int = 10) -> List[dict]:
    """해외 종목용 yfinance 뉴스 폴백."""
    if symbol.isdigit():
        return []
    try:
        import yfinance as yf
        news = yf.Ticker(symbol).news or []
        items = []
        for n in news[: max_items * 2]:
            content = n.get("content") if isinstance(n.get("content"), dict) else n
            title = content.get("title") or n.get("title") or ""
            summary = content.get("summary") or n.get("summary") or ""
            link = ""
            if isinstance(content.get("clickThroughUrl"), dict):
                link = content["clickThroughUrl"].get("url", "")
            link = link or n.get("link") or ""
            pub = None
            provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
            source = provider.get("displayName") or n.get("publisher") or "Yahoo Finance"
            if content.get("pubDate"):
                pub = _parse_rss_date(content.get("pubDate"))
            elif n.get("providerPublishTime"):
                pub = datetime.fromtimestamp(n["providerPublishTime"], tz=timezone.utc).isoformat()
            if not title or not link:
                continue
            if _is_blog_item(title, source, link):
                continue
            items.append({
                "title": title,
                "summary": _clean_html(summary)[:400],
                "url": link,
                "source": source,
                "published_at": pub,
            })
            if len(items) >= max_items:
                break
        return items
    except Exception as e:
        log.warning("yfinance news failed: %s", e)
        return []


def fetch_news(symbol: str, market: str = "KRX", stock_name: Optional[str] = None) -> dict:
    settings = get_settings()
    identity = resolve_news_identity(symbol, market, stock_name)
    name = identity["display"]
    ticker = identity["symbol"]
    max_items = settings.news_max_items
    queries = _search_queries(identity)

    collected: List[dict] = []
    seen = set()
    dropped = 0
    for q in queries:
        for item in _google_news_rss(q, language=settings.news_language, max_items=max_items * 2):
            key = (item["title"], item.get("url"))
            if key in seen:
                continue
            seen.add(key)
            item["symbol"] = ticker
            item["stock_name"] = name
            item["title_original"] = item.get("title") or ""
            if not news_item_matches_identity(item, identity):
                dropped += 1
                continue
            collected.append(item)
        if len(collected) >= max_items:
            break

    if len(collected) < 5:
        for item in _yfinance_news(ticker, max_items=max_items):
            key = (item["title"], item.get("url"))
            if key in seen:
                continue
            seen.add(key)
            item["symbol"] = ticker
            item["stock_name"] = name
            item["title_original"] = item.get("title") or ""
            if not news_item_matches_identity(item, identity):
                dropped += 1
                continue
            collected.append(item)

    if dropped:
        log.info(
            "news identity filter %s (%s): kept %s, dropped %s unrelated",
            ticker,
            name,
            len(collected),
            dropped,
        )

    collected = enrich_and_sort_items(collected, sort="importance")
    collected = collected[:max_items]
    return {
        "symbol": ticker,
        "stock_name": name,
        "market": identity["market"] or market,
        "items": collected,
        "count": len(collected),
        "sort": "importance",
    }


def fetch_news_with_prepare(
    symbol: str,
    market: str = "KRX",
    stock_name: Optional[str] = None,
    *,
    prepare: bool = True,
    provider: str = "",
    sort: str = "importance",
) -> dict:
    """종목 뉴스 + 캐시 제목/중요도 반영 + (선택) 백그라운드 요약."""
    payload = fetch_news(symbol=symbol, market=market, stock_name=stock_name)
    items = enrich_and_sort_items(payload.get("items") or [], sort=sort)
    payload["items"] = items
    payload["count"] = len(items)
    payload["sort"] = sort if sort in ("importance", "date") else "importance"
    if prepare:
        pending = [it for it in items if not it.get("has_ai_summary")]
        if pending:
            prepare_articles_async(pending[:12], provider=provider, include_macros=True)
        payload["preparing"] = bool(pending)
    else:
        payload["preparing"] = False
    return payload
