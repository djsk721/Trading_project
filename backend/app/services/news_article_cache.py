"""뉴스·시황 공통 기사 캐시 (한글 제목 · 중요도 · 요약, TTL 14일)."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import PROJECT_ROOT

log = logging.getLogger(__name__)

_CACHE_DIR = PROJECT_ROOT / ".cache"
_SUMMARY_PATH = _CACHE_DIR / "news_summaries.json"

# 기본 캐시 14일
_SUMMARY_TTL = 14 * 24 * 3600
_MAX_ENTRIES = 800

_lock = threading.Lock()
_inflight: Dict[str, threading.Event] = {}

_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def url_key(url: str) -> str:
    return hashlib.sha1((url or "").strip().encode("utf-8")).hexdigest()


def looks_english_title(title: str) -> bool:
    """영문 제목 여부 (한글이 거의 없고 라틴 문자가 충분할 때)."""
    t = (title or "").strip()
    if not t:
        return False
    hangul = len(_HANGUL_RE.findall(t))
    latin = len(_LATIN_RE.findall(t))
    return latin >= 6 and hangul < 3


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


def _purge_expired(data: dict) -> dict:
    now = datetime.now(timezone.utc)
    kept: dict = {}
    for k, entry in data.items():
        try:
            ts = datetime.fromisoformat(
                str(entry.get("updated_at", "")).replace("Z", "+00:00")
            )
            age = (now - ts.astimezone(timezone.utc)).total_seconds()
            if age <= _SUMMARY_TTL:
                kept[k] = entry
        except Exception:
            kept[k] = entry
    if len(kept) > _MAX_ENTRIES:
        items = sorted(
            kept.items(),
            key=lambda kv: (
                int(kv[1].get("importance") or 0),
                kv[1].get("updated_at") or "",
            ),
            reverse=True,
        )
        kept = dict(items[:_MAX_ENTRIES])
    return kept


def put_cached_article(entry: dict) -> None:
    with _lock:
        data = _purge_expired(_load_json(_SUMMARY_PATH))
        key = entry.get("id") or url_key(entry.get("url", ""))
        data[key] = entry
        _save_json(_SUMMARY_PATH, data)


def get_cached_article(url: str) -> Optional[dict]:
    key = url_key(url)
    with _lock:
        data = _load_json(_SUMMARY_PATH)
    entry = data.get(key)
    if not entry:
        return None
    try:
        ts = datetime.fromisoformat(str(entry.get("updated_at", "")).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
        if age > _SUMMARY_TTL:
            return None
    except Exception:
        pass
    return entry


def parse_enrichment_fields(raw: str) -> Dict[str, Any]:
    """TITLE_KO / IMPORTANCE / SENTIMENT / LEAD·BODY·NOTE 파싱."""
    text = (raw or "").strip()
    title_ko = ""
    importance = 3
    importance_reason = ""
    sentiment = ""
    sentiment_reason = ""

    m_title = re.search(
        r"(?:^|\n)\s*TITLE_KO\s*[:：]\s*(.+?)(?=\n\s*[A-Z_]+\s*[:：]|\n\s*$|$)",
        text,
        flags=re.I | re.S,
    )
    if m_title:
        title_ko = m_title.group(1).strip().split("\n")[0].strip()

    m_imp = re.search(r"(?:^|\n)\s*IMPORTANCE\s*[:：]\s*([1-5])\b", text, flags=re.I)
    if m_imp:
        importance = int(m_imp.group(1))

    m_reason = re.search(
        r"(?:^|\n)\s*IMPORTANCE_REASON\s*[:：]\s*(.+?)(?=\n\s*[A-Z_]+\s*[:：]|\n\s*$|$)",
        text,
        flags=re.I | re.S,
    )
    if m_reason:
        importance_reason = m_reason.group(1).strip().split("\n")[0].strip()

    m_sent = re.search(
        r"(?:^|\n)\s*SENTIMENT\s*[:：]\s*(bullish|bearish|mixed|neutral)\b",
        text,
        flags=re.I,
    )
    if m_sent:
        sentiment = m_sent.group(1).strip().lower()

    m_sent_reason = re.search(
        r"(?:^|\n)\s*SENTIMENT_REASON\s*[:：]\s*(.+?)(?=\n\s*[A-Z_]+\s*[:：]|\n\s*$|$)",
        text,
        flags=re.I | re.S,
    )
    if m_sent_reason:
        sentiment_reason = m_sent_reason.group(1).strip().split("\n")[0].strip()

    # 요약 본문에서 메타 라벨 제거
    body = re.sub(
        r"(?:^|\n)\s*(?:TITLE_KO|IMPORTANCE|IMPORTANCE_REASON|SENTIMENT|SENTIMENT_REASON)\s*[:：].*",
        "",
        text,
        flags=re.I,
    )
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return {
        "title_ko": title_ko,
        "importance": importance,
        "importance_reason": importance_reason,
        "sentiment": sentiment,
        "sentiment_reason": sentiment_reason,
        "summary_ko": body or text,
    }


def sort_news_items(
    items: List[dict],
    *,
    mode: str = "importance",
) -> List[dict]:
    """importance: 중요도↓ → 날짜↓ / date: 날짜↓ → 중요도↓."""
    mode = (mode or "importance").lower()

    def pub_key(it: dict) -> str:
        return it.get("published_at") or ""

    def imp_key(it: dict) -> int:
        try:
            return int(it.get("importance") or 0)
        except Exception:
            return 0

    if mode == "date":
        return sorted(items, key=lambda it: (pub_key(it), imp_key(it)), reverse=True)
    return sorted(items, key=lambda it: (imp_key(it), pub_key(it)), reverse=True)


def enrich_item_from_cache(item: dict) -> dict:
    """캐시된 한글 제목·중요도·요약 플래그를 항목에 반영."""
    url = item.get("url") or ""
    out = dict(item)
    cached = get_cached_article(url) if url else None
    title_orig = out.get("title_original") or out.get("title") or ""
    out["title_original"] = title_orig
    if cached:
        title_ko = (cached.get("title_ko") or "").strip()
        if title_ko:
            out["title"] = title_ko
        elif cached.get("title"):
            # 구캐시: title 필드가 이미 번역본일 수 있음
            out["title"] = cached["title"]
        out["importance"] = int(cached.get("importance") or 0) or None
        out["importance_reason"] = cached.get("importance_reason") or ""
        out["sentiment"] = cached.get("sentiment") or ""
        out["sentiment_reason"] = cached.get("sentiment_reason") or ""
        out["has_ai_summary"] = bool(cached.get("summary_ko"))
        out["summary_cached"] = True
    else:
        out.setdefault("importance", None)
        out.setdefault("importance_reason", "")
        out.setdefault("sentiment", "")
        out.setdefault("sentiment_reason", "")
        out.setdefault("has_ai_summary", False)
        out["summary_cached"] = False
    return out


def enrich_and_sort_items(items: List[dict], *, sort: str = "importance") -> List[dict]:
    enriched = [enrich_item_from_cache(it) for it in items]
    return sort_news_items(enriched, mode=sort)


def summarize_article(
    *,
    url: str,
    title: str = "",
    snippet: str = "",
    source: str = "",
    provider: str = "",
    force: bool = False,
    include_macros: bool = True,
) -> dict:
    """기사 요약 + 한글 제목 + 중요도 (캐시 14일)."""
    url = (url or "").strip()
    title = (title or "").strip()
    if not url:
        return {
            "id": "",
            "url": "",
            "title": title,
            "title_original": title,
            "title_ko": title,
            "summary_ko": "요약할 기사 URL이 없습니다.",
            "importance": 0,
            "importance_reason": "",
            "sentiment": "",
            "sentiment_reason": "",
            "cached": False,
            "provider": "none",
        }

    if not force:
        cached = get_cached_article(url)
        if cached and cached.get("summary_ko"):
            return {**cached, "cached": True}

    key = url_key(url)
    wait_ev: Optional[threading.Event] = None
    is_owner = False
    with _lock:
        inflight = _inflight.get(key)
        if inflight is not None:
            wait_ev = inflight
        else:
            is_owner = True
            _inflight[key] = threading.Event()

    if not is_owner and wait_ev is not None:
        wait_ev.wait(timeout=90.0)
        cached = get_cached_article(url)
        if cached:
            return {**cached, "cached": True}
        return {
            "id": key,
            "url": url,
            "title": title,
            "title_original": title,
            "title_ko": title,
            "summary_ko": "요약 생성 대기 중 시간이 초과되었습니다. 다시 시도해 주세요.",
            "importance": 0,
            "importance_reason": "",
            "sentiment": "",
            "sentiment_reason": "",
            "cached": False,
            "provider": "none",
        }

    # 지연 import: rag ↔ news_service 순환 참조 방지
    from app.services.macro_snapshots import format_macros_for_prompt
    from app.services.rag.llm_router import get_llm_router

    need_translate = looks_english_title(title)
    macro_block = format_macros_for_prompt() if include_macros else "LIVE_MACROS: (skipped)"
    raw = ""
    used_provider = "none"
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 트레이딩 데스크 뉴스 에디터입니다. "
                    "기사를 한국어 브리핑으로 정리하고, 시장 관점 중요도와 호재/악재 톤을 판별합니다. "
                    "지수·환율·유가·금리 수치는 LIVE_MACROS 또는 기사에 있는 것만 사용하세요. "
                    "학습 지식의 과거 수치를 지어내지 마세요. "
                    "번호 목록·이모지·'AI'·'요약하면' 메타 표현은 금지입니다. "
                    "투자 권유는 하지 마세요. 호재/악재는 참고 태그일 뿐 매수·매도 신호가 아닙니다."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{macro_block}\n\n"
                    f"제목: {title}\n"
                    f"출처: {source or '-'}\n"
                    f"링크: {url}\n"
                    f"발췌: {(snippet or '')[:800]}\n\n"
                    "출력 형식(라벨은 영문 대문자):\n"
                    + (
                        "TITLE_KO: (영문 제목을 자연스러운 한국어 뉴스 제목으로 번역)\n"
                        if need_translate
                        else "TITLE_KO: (원 제목이 한글이면 그대로, 살짝만 다듬기)\n"
                    )
                    + "IMPORTANCE: (1-5 정수만)\n"
                    "  5=시장 전반을 움직일 수 있는 초대형 (금리 결정, 전쟁 격화, 급변)\n"
                    "  4=주요 매크로·섹터에 뚜렷한 영향\n"
                    "  3=일반적이나 데스크가 볼 만한 이슈\n"
                    "  2=국지적·후속 보도\n"
                    "  1=잡음·광고성·관련성 낮음\n"
                    "IMPORTANCE_REASON: (한 줄)\n"
                    "SENTIMENT: (bullish|bearish|mixed|neutral 중 하나)\n"
                    "  bullish=해당 종목·시장에 대체로 호재\n"
                    "  bearish=해당 종목·시장에 대체로 악재\n"
                    "  mixed=호재와 악재가 섞임\n"
                    "  neutral=방향성이 약함\n"
                    "SENTIMENT_REASON: (한 줄, 매수/매도 단정 금지)\n"
                    "LEAD: (한 문장)\n"
                    "BODY: (2~4문장)\n"
                    "NOTE: (1~2문장, 매수/매도 단정 금지)\n"
                ),
            },
        ]
        raw, used_provider = get_llm_router().chat(
            messages,
            provider=provider or None,
            temperature=0.3,
            num_predict=750,
        )
        raw = (raw or "").strip()
        if not raw:
            raw = "요약을 생성하지 못했습니다."
    except Exception as e:
        log.exception("article summarize failed")
        raw = f"요약 생성에 실패했습니다. ({e})"
        used_provider = "none"
    finally:
        with _lock:
            ev = _inflight.pop(key, None)
        if ev:
            ev.set()

    parsed = parse_enrichment_fields(raw)
    title_ko = parsed["title_ko"] or (title if not need_translate else title)
    if need_translate and not parsed["title_ko"]:
        # 번역 라벨 누락 시 원제 유지
        title_ko = title

    entry = {
        "id": key,
        "url": url,
        "title": title_ko,  # 목록 표시용 (한글)
        "title_original": title,
        "title_ko": title_ko,
        "source": source,
        "summary_ko": parsed["summary_ko"],
        "importance": int(parsed["importance"] or 3),
        "importance_reason": parsed["importance_reason"],
        "sentiment": parsed.get("sentiment") or "",
        "sentiment_reason": parsed.get("sentiment_reason") or "",
        "provider": used_provider,
        "updated_at": _now_iso(),
        "cached": False,
        "ttl_days": 14,
    }
    if used_provider != "none" and "실패" not in parsed["summary_ko"][:20]:
        put_cached_article(entry)
    return entry


def prepare_articles_async(items: List[dict], provider: str = "", include_macros: bool = True) -> None:
    """미캐시 기사를 백그라운드 요약."""

    def _worker() -> None:
        for it in items:
            url = it.get("url") or ""
            if not url or get_cached_article(url):
                continue
            try:
                summarize_article(
                    url=url,
                    title=it.get("title_original") or it.get("title") or "",
                    snippet=it.get("summary") or "",
                    source=it.get("source") or "",
                    provider=provider,
                    include_macros=include_macros,
                )
                time.sleep(0.4)
            except Exception as e:
                log.warning("bg article summarize skip: %s", e)

    t = threading.Thread(target=_worker, name="news-article-enrich", daemon=True)
    t.start()
