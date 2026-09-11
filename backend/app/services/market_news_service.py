"""시황(매크로) 뉴스 수집 + AI 한글 요약·탭 다이제스트 캐시."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from app.core.config import PROJECT_ROOT, get_settings
from app.services.macro_snapshots import (
    fetch_macro_snapshots,
    format_macros_for_prompt,
)
from app.services.news_article_cache import (
    enrich_and_sort_items,
    get_cached_article,
    prepare_articles_async,
    summarize_article,
    url_key,
)
from app.services.news_service import _google_news_rss
from app.services.rag.llm_router import get_llm_router

# 다이제스트 캐시 스키마 버전 (프롬프트/지표 주입 변경 시 상향)
_DIGEST_SCHEMA = "m4"

log = logging.getLogger(__name__)

_CACHE_DIR = PROJECT_ROOT / ".cache"
_MARKET_LIST_PATH = _CACHE_DIR / "market_news_list.json"
_DIGEST_PATH = _CACHE_DIR / "market_tab_digests.json"

_MARKET_LIST_TTL = 20 * 60  # 시황 목록 20분
_MARKET_LIST_SCHEMA = "c1"  # 탭 구성 변경 시 상향 (crypto 포함)
_DIGEST_TTL = 1 * 3600  # 시황 정리: 1시간 경과 시 재생성 (또는 force 새로고침)

_KST = ZoneInfo("Asia/Seoul")

_lock = threading.Lock()
_digest_inflight: Dict[str, threading.Event] = {}

# 수집 쿼리 → UI 탭 (kr / us / world / risk / crypto)
_MARKET_QUERIES: List[tuple[str, str, str]] = [
    # 국내 증시 + 수급 + 환율
    (
        "kr",
        "코스피 OR 코스닥 OR 한국 증시 OR 외국인 수급 OR 원달러 환율 when:3d",
        "ko",
    ),
    # 국내 경기 / 통화정책
    (
        "kr",
        "한국은행 OR 금통위 OR 기준금리 OR 물가 OR 수출 OR 무역수지 when:3d",
        "ko",
    ),
    # 미국 증시
    (
        "us",
        '"S&P 500" OR Nasdaq OR "Wall Street" OR "US stock market" when:3d',
        "en",
    ),
    # 미국 금리 / 주요 경제지표
    (
        "us",
        '"Federal Reserve" OR FOMC OR "Treasury yield" OR CPI OR PPI OR PCE OR "nonfarm payrolls" when:3d',
        "en",
    ),
    # 중국 + 글로벌 지정학
    (
        "world",
        'China economy OR China stimulus OR Taiwan OR Ukraine OR "Middle East" OR tariffs OR sanctions OR war when:3d',
        "en",
    ),
    # 원자재 + 시장 위험
    (
        "risk",
        'WTI OR Brent OR DXY OR VIX OR "market volatility" OR gold when:3d',
        "en",
    ),
    # 암호화폐 시세·수급
    (
        "crypto",
        "비트코인 OR 이더리움 OR 암호화폐 OR 가상자산 OR 업비트 OR 김치프리미엄 when:3d",
        "ko",
    ),
    # 암호화폐 규제·ETF·글로벌 시장
    (
        "crypto",
        'Bitcoin OR Ethereum OR cryptocurrency OR "crypto market" OR "bitcoin ETF" OR SEC crypto when:3d',
        "en",
    ),
]

_CATEGORY_ORDER = ("kr", "us", "world", "risk", "crypto")

_CATEGORY_LABEL = {
    "kr": "한국",
    "us": "미국",
    "world": "세계·지정학",
    "risk": "리스크·원자재",
    "crypto": "암호화폐",
}


def _categories_payload() -> List[dict]:
    return [{"id": k, "label": _CATEGORY_LABEL[k]} for k in _CATEGORY_ORDER]


def _cache_covers_current_tabs(cached: dict) -> bool:
    if cached.get("schema") != _MARKET_LIST_SCHEMA:
        return False
    ids = {str(c.get("id") or "") for c in (cached.get("categories") or [])}
    return all(cat in ids for cat in _CATEGORY_ORDER)


def _today_kst() -> str:
    return datetime.now(_KST).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("cache load failed %s: %s", path, e)
    return {}


def _save_json(path: Path, data: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("cache save failed %s: %s", path, e)


def _digest_cache_key(category: str, day: Optional[str] = None) -> str:
    return f"{day or _today_kst()}:{category}:{_DIGEST_SCHEMA}"


def get_cached_tab_digest(category: str, day: Optional[str] = None) -> Optional[dict]:
    key = _digest_cache_key(category, day)
    with _lock:
        data = _load_json(_DIGEST_PATH)
    entry = data.get(key)
    if not entry or not entry.get("text"):
        return None
    if not _digest_text_is_korean(str(entry.get("text") or "")):
        return None
    if entry.get("day") and entry.get("day") != (day or _today_kst()):
        return None
    # 이전 정리 시각 기준 1시간 이상이면 만료
    try:
        ts = datetime.fromisoformat(entry.get("updated_at", "").replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
        if age > _DIGEST_TTL:
            return None
    except Exception:
        return None
    return entry


def _put_tab_digest(entry: dict) -> None:
    with _lock:
        data = _load_json(_DIGEST_PATH)
        key = _digest_cache_key(entry.get("category", ""), entry.get("day"))
        data[key] = entry
        # 최근 60개만 유지
        if len(data) > 60:
            items = sorted(
                data.items(),
                key=lambda kv: kv[1].get("updated_at") or "",
                reverse=True,
            )
            data = dict(items[:60])
        _save_json(_DIGEST_PATH, data)


def _is_today_item(item: dict, day: str) -> bool:
    pub = (item.get("published_at") or "").strip()
    if not pub:
        return True  # 날짜 없으면 당일 피드로 취급
    try:
        # 날짜 접두만 비교 (타임존 혼재 허용)
        return pub[:10] == day or pub[:10] >= day
    except Exception:
        return True


def _items_for_digest(items: List[dict], category: str, day: str, limit: int = 8) -> List[dict]:
    if category == "all":
        pool = [it for it in items if _is_today_item(it, day)]
    else:
        pool = [
            it for it in items
            if it.get("category") == category and _is_today_item(it, day)
        ]
    pool = sorted(pool, key=lambda x: x.get("published_at") or "", reverse=True)
    return pool[:limit]


def _article_brief_line(item: dict) -> str:
    cached = get_cached_article(item.get("url") or "")
    # 종합 입력만 한글 제목 우선. 목록 원문은 브리핑 보기에서 번역.
    title = ""
    if cached:
        title = (cached.get("title_ko") or "").strip()
    title = title or (item.get("title") or "").strip()
    body = ""
    if cached and cached.get("summary_ko"):
        body = str(cached["summary_ko"]).replace("\n", " ").strip()
        if len(body) > 280:
            body = body[:280] + "…"
    else:
        body = (item.get("summary") or "").replace("\n", " ").strip()
        if len(body) > 180:
            body = body[:180] + "…"
    src = item.get("source") or ""
    if body:
        return f"- [{src}] {title}\n  {body}"
    return f"- [{src}] {title}"


def _digest_text_is_korean(text: str) -> bool:
    """본문이 한글 위주인지 확인. 영문 초안·영문 캐시는 재생성합니다."""
    body = re.sub(r"\b(LEAD|BODY|NOTE)\s*:", " ", text or "", flags=re.I)
    body = re.sub(r"한\s*줄\s*핵심|배경\s*[/／]?\s*내용|시사점", " ", body)
    hangul = len(re.findall(r"[가-힣]", body))
    latin = len(re.findall(r"[A-Za-z]", body))
    if hangul < 15:
        return False
    if latin > hangul * 2:
        return False
    return True


def _translate_digest_to_korean(text: str, provider: str = "") -> tuple[str, str]:
    """영문 시황 초안을 한국어 브리핑으로 옮깁니다."""
    messages = [
        {
            "role": "system",
            "content": (
                "당신은 한국어 시황 번역 에디터입니다. "
                "입력 브리핑의 의미만 살려 자연스러운 한국어로 다시 씁니다. "
                "문장은 한국어만 사용하고, 티커·BTC·ETF·FOMC 같은 고유명사만 영문을 남깁니다. "
                "새로운 수치를 만들어 넣지 마세요."
            ),
        },
        {
            "role": "user",
            "content": (
                "아래 시황을 한국어로 다시 쓰세요.\n"
                "출력 형식:\n"
                "LEAD: (한 문장, 한국어)\n"
                "BODY: (2~4문장, 한국어)\n"
                "NOTE: (1문장, 한국어)\n\n"
                f"{text}"
            ),
        },
    ]
    out, used = get_llm_router().chat(
        messages,
        provider=provider or None,
        temperature=0.15,
        num_predict=550,
    )
    return (out or "").strip(), used


def _digest_messages(day: str, label: str, macro_block: str, bullets: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "당신은 한국어 트레이딩 데스크 시황 에디터입니다. "
                "반드시 한국어로만 작성합니다. 영어 문장으로 답하지 마세요. "
                "LIVE_MACROS와 제공된 기사만 근거로 짧게 정리합니다. "
                "영문 기사 제목이 있어도 내용을 한글로 옮겨 적습니다. "
                "Bitcoin, ETF, FOMC, BTC 같은 고유명사·티커만 영문을 유지하세요. "
                "학습된 과거 지식의 지수·환율·유가·금리 수치를 절대 쓰지 마세요. "
                "LIVE_MACROS에 없는 구체 수치(포인트, %, 달러)는 만들지 마세요. "
                "기사에만 있는 수치는 '보도 기준'으로만 언급하세요. "
                "번호 목록·이모지·'AI'·'요약하면' 같은 메타 표현은 쓰지 마세요. "
                "투자 권유·확정 전망은 금지입니다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"날짜: {day} (KST)\n"
                f"섹션: {label}\n"
                "언어: 한국어 (필수)\n\n"
                f"{macro_block}\n\n"
                "기사 메모(영어여도 출력은 한글):\n"
                f"{bullets}\n\n"
                "출력은 아래 세 줄 형식을 지키고 본문은 한국어:\n"
                "LEAD: 한 문장 한국어 핵심\n"
                "BODY: 2~4문장 한국어 배경\n"
                "NOTE: 한 문장 한국어 시사점\n"
            ),
        },
    ]


def build_tab_digest(
    *,
    category: str,
    items: List[dict],
    provider: str = "",
    force: bool = False,
    macros: Optional[dict] = None,
) -> dict:
    """탭별 금일 시황 정리 (캐시 우선). LIVE_MACROS + 기사만 근거로 작성."""
    day = _today_kst()
    label = "전체" if category == "all" else _CATEGORY_LABEL.get(category, category)
    macros = macros if macros is not None else fetch_macro_snapshots(force=False)
    macro_block = format_macros_for_prompt(macros)

    if not force:
        cached = get_cached_tab_digest(category, day)
        if cached:
            return {**cached, "cached": True}

    key = _digest_cache_key(category, day)
    wait_ev: Optional[threading.Event] = None
    is_owner = False
    with _lock:
        inflight = _digest_inflight.get(key)
        if inflight is not None:
            wait_ev = inflight
        else:
            is_owner = True
            _digest_inflight[key] = threading.Event()

    if not is_owner and wait_ev is not None:
        wait_ev.wait(timeout=90.0)
        cached = get_cached_tab_digest(category, day)
        if cached:
            return {**cached, "cached": True}
        return {
            "category": category,
            "category_label": label,
            "day": day,
            "text": "시황 정리 대기 중 시간이 초과되었습니다. 다시 시도해 주세요.",
            "provider": "none",
            "source_count": 0,
            "updated_at": _now_iso(),
            "cached": False,
        }

    picks = _items_for_digest(items, category, day, limit=8)
    if not picks:
        entry = {
            "category": category,
            "category_label": label,
            "day": day,
            "text": f"오늘({day}) {label} 관련으로 정리할 기사가 아직 없습니다.",
            "provider": "none",
            "source_count": 0,
            "updated_at": _now_iso(),
            "cached": False,
        }
        with _lock:
            ev = _digest_inflight.pop(key, None)
        if ev:
            ev.set()
        return entry

    bullets = "\n".join(_article_brief_line(it) for it in picks)
    text = ""
    used_provider = "none"
    try:
        text, used_provider = get_llm_router().chat(
            _digest_messages(day, label, macro_block, bullets),
            provider=provider or None,
            temperature=0.25,
            num_predict=550,
        )
        text = (text or "").strip()
        if text and not _digest_text_is_korean(text):
            log.info("tab digest not Korean category=%s, translating", category)
            try:
                translated, trans_provider = _translate_digest_to_korean(text, provider)
                if translated:
                    text = translated
                    used_provider = trans_provider or used_provider
            except Exception as te:
                log.warning("tab digest translate failed category=%s: %s", category, te)
        if text and not _digest_text_is_korean(text):
            log.warning("tab digest still English category=%s", category)
        if not text:
            text = "시황 정리를 생성하지 못했습니다."
    except Exception as e:
        log.exception("tab digest failed category=%s", category)
        text = f"시황 정리 생성에 실패했습니다. ({e})"
        used_provider = "none"
    finally:
        with _lock:
            ev = _digest_inflight.pop(key, None)
        if ev:
            ev.set()

    entry = {
        "category": category,
        "category_label": label,
        "day": day,
        "text": text,
        "provider": used_provider,
        "source_count": len(picks),
        "updated_at": _now_iso(),
        "cached": False,
    }
    if used_provider != "none" and "실패" not in text[:20] and _digest_text_is_korean(text):
        _put_tab_digest(entry)
    return entry


def collect_tab_digests(force: bool = False) -> Dict[str, dict]:
    """캐시된 탭 다이제스트를 모은다."""
    day = _today_kst()
    out: Dict[str, dict] = {}
    for cat in ("all",) + _CATEGORY_ORDER:
        cached = None if force else get_cached_tab_digest(cat, day)
        if cached:
            out[cat] = {**cached, "cached": True, "ready": True}
        else:
            out[cat] = {
                "category": cat,
                "category_label": "전체" if cat == "all" else _CATEGORY_LABEL.get(cat, cat),
                "day": day,
                "text": "",
                "provider": "none",
                "source_count": 0,
                "updated_at": "",
                "cached": False,
                "ready": False,
            }
    return out


def prepare_tab_digests_async(
    items: List[dict],
    provider: str = "",
    force: bool = False,
    macros: Optional[dict] = None,
) -> None:
    """탭별 시황 정리를 백그라운드로 생성."""

    def _worker() -> None:
        time.sleep(1.2)
        snap = macros if macros is not None else fetch_macro_snapshots(force=False)
        for cat in ("all",) + _CATEGORY_ORDER:
            try:
                if not force and get_cached_tab_digest(cat):
                    continue
                build_tab_digest(
                    category=cat,
                    items=items,
                    provider=provider,
                    force=force,
                    macros=snap,
                )
                time.sleep(0.5)
            except Exception as e:
                log.warning("bg tab digest skip %s: %s", cat, e)

    t = threading.Thread(target=_worker, name="market-tab-digest", daemon=True)
    t.start()


def fetch_market_news(max_per_query: int = 8, force: bool = False) -> dict:
    """한국/미국/세계/리스크/암호화폐 시황 뉴스 목록."""
    if not force and _MARKET_LIST_PATH.exists():
        cached = _load_json(_MARKET_LIST_PATH)
        fetched_at = cached.get("fetched_at")
        try:
            if fetched_at and _cache_covers_current_tabs(cached):
                ts = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
                if age < _MARKET_LIST_TTL and cached.get("items"):
                    cached["categories"] = _categories_payload()
                    return cached
        except Exception:
            pass

    settings = get_settings()
    collected: List[dict] = []
    seen = set()
    for category, query, lang in _MARKET_QUERIES:
        for item in _google_news_rss(query, language=lang, max_items=max_per_query):
            url = item.get("url") or ""
            title = item.get("title") or ""
            key = (title, url)
            if not url or key in seen:
                continue
            seen.add(key)
            nid = url_key(url)
            collected.append({
                "id": nid,
                "title": title,
                "title_original": title,
                "summary": item.get("summary") or "",
                "url": url,
                "source": item.get("source") or "News",
                "published_at": item.get("published_at"),
                "category": category,
                "category_label": _CATEGORY_LABEL.get(category, category),
            })

    # 캐시 제목·중요도 반영 후 중요도→날짜 정렬, 카테고리 균형
    collected = enrich_and_sort_items(collected, sort="importance")
    by_cat: Dict[str, List[dict]] = {}
    for it in collected:
        by_cat.setdefault(it["category"], []).append(it)
    per_cat = 8
    balanced: List[dict] = []
    for cat in _CATEGORY_ORDER:
        balanced.extend(by_cat.get(cat, [])[:per_cat])
    balanced = enrich_and_sort_items(balanced, sort="importance")
    cap = max(per_cat * len(_CATEGORY_ORDER), settings.news_max_items)
    balanced = balanced[:cap]

    payload = {
        "fetched_at": _now_iso(),
        "schema": _MARKET_LIST_SCHEMA,
        "items": balanced,
        "count": len(balanced),
        "categories": _categories_payload(),
        "sort": "importance",
    }
    _save_json(_MARKET_LIST_PATH, payload)
    return payload


def summarize_news_item(
    *,
    url: str,
    title: str = "",
    snippet: str = "",
    source: str = "",
    provider: str = "",
    force: bool = False,
) -> dict:
    """단일 뉴스 AI 한글 요약·제목 번역·중요도 (공통 캐시 14일)."""
    return summarize_article(
        url=url,
        title=title,
        snippet=snippet,
        source=source,
        provider=provider,
        force=force,
        include_macros=True,
    )


def market_news_with_prepare(
    prepare: bool = True,
    provider: str = "",
    force: bool = False,
    sort: str = "importance",
) -> dict:
    payload = fetch_market_news(force=force)
    items = enrich_and_sort_items(payload.get("items") or [], sort=sort)
    payload["items"] = items
    payload["categories"] = _categories_payload()
    payload["sort"] = sort if sort in ("importance", "date") else "importance"
    payload["count"] = len(items)

    macros = fetch_macro_snapshots(force=force)

    if prepare:
        pending = [it for it in items if not it.get("has_ai_summary")]
        if pending:
            prepare_articles_async(pending, provider=provider, include_macros=True)

    digests = collect_tab_digests(force=force)
    digest_pending = any(not d.get("ready") for d in digests.values())
    if prepare and digest_pending:
        prepare_tab_digests_async(items, provider=provider, force=force, macros=macros)

    payload["macros"] = macros
    payload["digests"] = digests
    payload["digest_preparing"] = digest_pending
    payload["preparing"] = prepare and (
        any(not it.get("has_ai_summary") for it in items) or digest_pending
    )
    payload["digest_day"] = _today_kst()
    return payload
