"""종목 관련 최신 뉴스 검색 (블로그 제외, 기사 원문 링크 우선)."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional
from urllib.parse import quote_plus, urlparse
from xml.etree import ElementTree

import httpx

from app.core.config import get_settings
from app.services.market_data import resolve_stock_name
from app.services.news_article_cache import enrich_and_sort_items, prepare_articles_async

log = logging.getLogger(__name__)

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
    name = stock_name or resolve_stock_name(symbol, market)
    max_items = settings.news_max_items

    queries: List[str] = []
    if market.upper() == "KRX" or symbol.isdigit():
        queries.append(f"{name} 주식")
        queries.append(f"{name} 증권")
        queries.append(symbol)
    else:
        queries.append(f"{name} stock news")
        queries.append(f"{symbol} stock")

    collected: List[dict] = []
    seen = set()
    for q in queries:
        for item in _google_news_rss(q, language=settings.news_language, max_items=max_items * 2):
            key = (item["title"], item.get("url"))
            if key in seen:
                continue
            seen.add(key)
            item["symbol"] = symbol
            item["stock_name"] = name
            item["title_original"] = item.get("title") or ""
            collected.append(item)
        if len(collected) >= max_items:
            break

    if len(collected) < 5:
        for item in _yfinance_news(symbol, max_items=max_items):
            key = (item["title"], item.get("url"))
            if key in seen:
                continue
            seen.add(key)
            item["symbol"] = symbol
            item["stock_name"] = name
            item["title_original"] = item.get("title") or ""
            collected.append(item)

    collected = enrich_and_sort_items(collected, sort="importance")
    collected = collected[:max_items]
    return {
        "symbol": symbol,
        "stock_name": name,
        "market": market,
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
