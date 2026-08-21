"""일일 종목 추천 — 유동 유니버스 기술 스캔 + shortlist 뉴스/AI 선정."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

from app.core.config import get_settings
from app.services.indicators import calculate_indicators, latest_summary
from app.services.market_data import fetch_ohlcv
from app.services.news_service import fetch_news
from app.services.rag.llm_router import get_llm_router
from app.services.recommend_cache import (
    load_daily,
    markets_for,
    normalize_recommend_market,
    save_daily,
    today_stamp,
)
from app.services.recommend_universe import universe_as_mapping, universe_yf_map

log = logging.getLogger(__name__)

RECOMMEND_SYSTEM = (
    "당신은 주식 전략가입니다. 제공된 기술지표와 최근 헤드라인만 사용해 "
    "일일 관심 종목을 선정하세요. 모든 자연어 응답(사유, 시장 코멘트)은 "
    "반드시 한국어로 작성하세요. 가격·뉴스를 지어내지 마세요. "
    "수익을 보장하는 표현은 금지합니다. 유효한 JSON만 출력하세요."
)

RECOMMEND_SUMMARY_SYSTEM = (
    "당신은 증권 리서치 애널리스트입니다. 제공된 종목 정보와 헤드라인만 사용해 "
    "한국어로 간결한 브리핑을 작성하세요. 헤드라인에 없는 사실은 만들지 마세요. "
    "투자 권유·수익 보장 표현은 금지합니다. 유효한 JSON만 출력하세요."
)


def _round_px(price: float, market: str) -> float:
    p = float(price or 0)
    if p <= 0:
        return 0.0
    if str(market).upper() == "KRX":
        return float(int(round(p)))
    if p >= 1:
        return round(p, 2)
    if p >= 0.1:
        return round(p, 3)
    return round(p, 4)


def _suggest_buy_sell(
    price: float,
    market: str,
    bb_lower: float | None = None,
    bb_upper: float | None = None,
) -> tuple[float, float]:
    """현재가·밴드 기반 권장 매수/매도 참고가 (보장 아님)."""
    px = float(price or 0)
    if px <= 0:
        return 0.0, 0.0
    buy = float(bb_lower) if bb_lower and bb_lower > 0 else px * 0.985
    sell = float(bb_upper) if bb_upper and bb_upper > 0 else px * 1.035
    if buy >= px:
        buy = px * 0.99
    if sell <= px:
        sell = px * 1.03
    if buy >= sell:
        buy = px * 0.98
        sell = px * 1.04
    return _round_px(buy, market), _round_px(sell, market)


def _score_row(summary: dict, row) -> tuple[float, List[str]]:
    score = 50.0
    reasons: List[str] = []

    rsi = float(summary.get("rsi", 50))
    if 40 <= rsi <= 60:
        score += 8
        reasons.append("RSI가 중립 모멘텀 구간에 있음")
    elif 30 <= rsi < 40:
        score += 12
        reasons.append("RSI가 과매도 회복 구간에 근접")
    elif rsi < 30:
        score += 6
        reasons.append("RSI 과매도(반등 관찰 구간)")
    elif rsi > 70:
        score -= 10
        reasons.append("RSI 과매수 위험")

    if summary.get("macd_signal") == "bullish":
        score += 12
        reasons.append("MACD 상승 편향")
    else:
        score -= 4
        reasons.append("MACD 하락 편향")

    if str(summary.get("trend_long")) == "UP":
        score += 15
        reasons.append("중기 상승추세(SMA20 > SMA120)")
    else:
        score -= 8
        reasons.append("중기 하락추세")

    vol_ratio = float(summary.get("volume_ratio", 1))
    if vol_ratio >= 1.5:
        score += 8
        reasons.append(f"거래량 급증({vol_ratio:.1f}배)")

    change = float(summary.get("price_change", 0))
    if 0 < change < 3:
        score += 5
        reasons.append("당일 소폭 상승")
    elif change >= 5:
        score -= 3
        reasons.append("당일 급등 후 조정 위험")

    bb_pos = float(row.get("BB_Position", 0.5) or 0.5)
    if 0.2 <= bb_pos <= 0.5:
        score += 6
        reasons.append("가격이 볼린저 밴드 하단~중단에 위치")

    return max(0.0, min(100.0, score)), reasons[:5]


def _score_one(
    symbol: str,
    name: str,
    market: str,
    days: int,
    yf_symbol: str = "",
) -> Dict[str, Any] | None:
    try:
        # 추천 스캔: yfinance만 (KIS/pykrx 미사용), screen의 KS/KQ 접미사 사용
        df = fetch_ohlcv(
            symbol,
            market=market,
            timeframe="day",
            days=days,
            allow_kis=False,
            prefer_yfinance=True,
            yf_symbol=yf_symbol,
        )
        if df.empty or len(df) < 30:
            return None
        ind = calculate_indicators(df)
        summary = latest_summary(ind)
        row = ind.iloc[-1]
        score, reasons = _score_row(summary, row)
        price = float(summary.get("price", 0))
        bb_lower = float(row.get("BB_Lower", 0) or 0)
        bb_upper = float(row.get("BB_Upper", 0) or 0)
        buy_price, sell_price = _suggest_buy_sell(
            price, market, bb_lower=bb_lower, bb_upper=bb_upper,
        )
        return {
            "symbol": symbol,
            "name": name,
            "market": market.upper(),
            "score": round(score, 2),
            "price": price,
            "change_pct": round(float(summary.get("price_change", 0)), 2),
            "reasons": reasons,
            "rsi": round(float(summary.get("rsi", 50)), 2),
            "macd_signal": "BUY" if summary.get("macd_signal") == "bullish" else "SELL",
            "trend": str(summary.get("trend_long", "SIDEWAYS")),
            "buy_price": buy_price,
            "sell_price": sell_price,
            "bb_lower": bb_lower,
            "bb_upper": bb_upper,
        }
    except Exception as e:
        log.warning("recommend skip %s: %s", symbol, e)
        return None


def _collect_candidates(
    market: str,
    days: int,
    universe: Dict[str, str],
    workers: int = 8,
    yf_map: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    workers = max(1, min(int(workers or 8), 16))
    yf_map = yf_map or {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _score_one,
                symbol,
                name,
                market,
                days,
                yf_map.get(symbol, ""),
            ): symbol
            for symbol, name in universe.items()
        }
        for fut in as_completed(futs):
            row = fut.result()
            if row:
                scored.append(row)
    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(scored, start=1):
        item["scan_rank"] = i
    return scored


def _attach_news_briefs(candidates: List[Dict[str, Any]], market: str = "", per_stock: int = 5) -> List[Dict[str, Any]]:
    enriched = []
    for item in candidates:
        headlines: List[str] = []
        item_mkt = str(item.get("market") or market or "KRX").upper()
        try:
            news = fetch_news(item["symbol"], market=item_mkt, stock_name=item["name"])
            for n in (news.get("items") or [])[:per_stock]:
                title = (n.get("title") or "").strip()
                if title:
                    headlines.append(title)
        except Exception as e:
            log.warning("news brief failed for %s: %s", item["symbol"], e)
        enriched.append({**item, "headlines": headlines})
    return enriched


def _clip_text(text: str, max_len: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max(1, max_len - 1)] + "…"


def _trend_status_label(trend: str, rsi: float) -> str:
    if rsi >= 70:
        return "과열 주의"
    if rsi <= 30:
        return "과매도 구간"
    t = str(trend or "").upper()
    if t == "UP":
        return "상승 추세"
    if t == "DOWN":
        return "하락 추세"
    return "관망"


def _rsi_metric_note(rsi: float) -> str:
    r = float(rsi or 50)
    if r >= 70:
        return f"RSI {r:.1f} · 과열 부담"
    if r <= 30:
        return f"RSI {r:.1f} · 과매도"
    if r >= 55:
        return f"RSI {r:.1f} · 과열 부담 낮음"
    return f"RSI {r:.1f} · 중립"


def _normalize_brief_fields(raw: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    """스캔 카드용 구조화 브리핑."""
    rsi = float(item.get("rsi") or 50)
    trend = str(item.get("trend") or "SIDEWAYS")
    highlights = raw.get("highlights") or []
    if isinstance(highlights, str):
        highlights = [highlights]
    highlights = [_clip_text(str(h), 28) for h in highlights if str(h).strip()][:2]
    if not highlights:
        highlights = [_clip_text(str(r), 28) for r in (item.get("reasons") or [])[:2]]

    status_label = str(raw.get("status_label") or "").strip() or _trend_status_label(trend, rsi)
    metric_note = str(raw.get("metric_note") or "").strip() or _rsi_metric_note(rsi)
    detail = str(
        raw.get("detail_summary") or raw.get("summary") or raw.get("ai_summary") or ""
    ).strip()
    sector = str(raw.get("sector") or "").strip()

    return {
        "sector": sector,
        "status_label": _clip_text(status_label, 16),
        "highlights": highlights,
        "metric_note": _clip_text(metric_note, 40),
        "detail_summary": detail[:600],
        "ai_summary": detail[:600],
    }


def _fallback_stock_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    headlines = [h for h in (item.get("headlines") or []) if h][:2]
    hl: List[str] = []
    for h in headlines:
        hl.append(_clip_text(h, 28))
    if not hl:
        hl = [_clip_text(r, 28) for r in (item.get("reasons") or [])[:2]]
    rsi = float(item.get("rsi") or 50)
    trend = str(item.get("trend") or "SIDEWAYS")
    news_line = " · ".join(headlines) if headlines else "최근 헤드라인 수집 제한"
    detail = (
        f"{item.get('name', item.get('symbol', ''))} — {news_line}. "
        f"기술점수 {item.get('score', 0)}, { _trend_status_label(trend, rsi) }."
    )
    return _normalize_brief_fields(
        {
            "sector": "",
            "status_label": _trend_status_label(trend, rsi),
            "highlights": hl,
            "metric_note": _rsi_metric_note(rsi),
            "detail_summary": detail,
        },
        item,
    )


def _summarize_stock_batch(
    batch: List[Dict[str, Any]],
    provider: str = "",
) -> Dict[str, Dict[str, Any]]:
    """LLM으로 종목별 스캔 카드용 구조화 브리핑 생성."""
    if not batch:
        return {}
    settings = get_settings()
    payload = [
        {
            "symbol": c["symbol"],
            "name": c["name"],
            "market": c.get("market", "KRX"),
            "price": c.get("price"),
            "change_pct": c.get("change_pct"),
            "score": c.get("score"),
            "rsi": c.get("rsi"),
            "trend": c.get("trend"),
            "headlines": c.get("headlines") or [],
        }
        for c in batch
    ]
    user_prompt = f"""아래 종목 각각에 대해 **스캔 가능한 짧은 카드 데이터**를 작성하세요.

입력:
{json.dumps(payload, ensure_ascii=False, indent=2)}

반드시 아래 JSON만 출력:
{{
  "summaries": [
    {{
      "symbol": "005930",
      "sector": "반도체 (25자 이내)",
      "status_label": "상승 추세|관망|과열 주의|하락 추세 중 하나",
      "highlights": [
        "핵심 근거 1 — 25자 이내",
        "핵심 근거 2 — 25자 이내"
      ],
      "metric_note": "RSI 56.1 · 과열 부담 낮음 (40자 이내)",
      "detail_summary": "펼칠 때 보는 상세 설명 3~5문장. 분야·뉴스·기술 맥락."
    }}
  ]
}}

규칙:
- summaries는 입력 symbol만 사용
- highlights는 정확히 2개, 각 25자 이내, 한 정보=한 줄
- headlines에 없는 뉴스·실적은 쓰지 말 것
- status_label·metric_note는 스캔용 짧은 라벨
- detail_summary만 긴 문단 허용
- 한국어만
"""

    out: Dict[str, Dict[str, Any]] = {}
    try:
        raw, _ = get_llm_router().chat(
            messages=[
                {"role": "system", "content": RECOMMEND_SUMMARY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            provider=provider or None,
            temperature=0.25,
            num_predict=min(settings.max_new_tokens, 2800),
        )
        parsed = _extract_json(raw)
        if not parsed or "summaries" not in parsed:
            raise ValueError("invalid summary JSON")
        allowed = {c["symbol"] for c in batch}
        for row in parsed.get("summaries") or []:
            sym = str(row.get("symbol", "")).strip()
            if sym not in allowed:
                continue
            base_item = next((c for c in batch if c["symbol"] == sym), {})
            out[sym] = _normalize_brief_fields(row, base_item)
    except Exception as e:
        log.warning("stock summary batch failed (%s items): %s", len(batch), e)

    for item in batch:
        sym = item["symbol"]
        if sym not in out:
            out[sym] = _fallback_stock_summary(item)
    return out


def _attach_ai_summaries(
    candidates: List[Dict[str, Any]],
    provider: str = "",
    batch_size: int = 5,
) -> List[Dict[str, Any]]:
    """상위 shortlist 종목에 sector·ai_summary 부여."""
    if not candidates:
        return []
    batch_size = max(1, min(int(batch_size or 5), 10))
    summary_map: Dict[str, Dict[str, str]] = {}
    for i in range(0, len(candidates), batch_size):
        chunk = candidates[i : i + batch_size]
        summary_map.update(_summarize_stock_batch(chunk, provider=provider))

    enriched: List[Dict[str, Any]] = []
    for item in candidates:
        extra = summary_map.get(item["symbol"]) or _fallback_stock_summary(item)
        enriched.append({**item, **extra})
    return enriched


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _llm_rank(
    candidates: List[Dict[str, Any]],
    market: str,
    top_n: int,
    as_of: str,
    provider: str = "",
) -> tuple[List[Dict[str, Any]], str, bool, str]:
    """LLM으로 최종 순위/사유 생성. 실패 시 기술점수 순으로 폴백."""
    settings = get_settings()
    payload = {
        "as_of": as_of,
        "market": market,
        "top_n": top_n,
        "candidates": [
            {
                "symbol": c["symbol"],
                "name": c["name"],
                "market": c.get("market", market),
                "technical_score": c["score"],
                "price": c["price"],
                "suggested_buy_price": c.get("buy_price"),
                "suggested_sell_price": c.get("sell_price"),
                "bb_lower": c.get("bb_lower"),
                "bb_upper": c.get("bb_upper"),
                "change_pct": c["change_pct"],
                "rsi": c["rsi"],
                "macd_signal": c["macd_signal"],
                "trend": c["trend"],
                "tech_reasons": c["reasons"],
                "headlines": c.get("headlines", []),
            }
            for c in candidates
        ],
    }

    scope = str(market).upper()
    if scope == "KRX":
        market_hint = "국내(KRX) 후보"
        rank_rule = "국내 종목만 대상으로 매력도 순위를 매기세요."
        commentary_hint = "오늘 국내 시장에 대한 2-3문장 한국어 코멘트"
        mix_rule = "- 국내(KRX) 종목만 순위에 포함"
    elif scope == "US":
        market_hint = "해외(US) 후보"
        rank_rule = "해외 종목만 대상으로 매력도 순위를 매기세요."
        commentary_hint = "오늘 해외 시장에 대한 2-3문장 한국어 코멘트"
        mix_rule = "- 해외(US) 종목만 순위에 포함"
    else:
        market_hint = "국내(KRX)·해외(US) 통합 후보"
        rank_rule = "국내·해외를 구분하지 말고 종합 매력도 기준으로 섞어 순위를 매기세요."
        commentary_hint = "오늘 국내·해외 시장에 대한 2-3문장 한국어 코멘트"
        mix_rule = "- KRX와 US를 한 리스트로 통합 순위 작성"
    user_prompt = f"""아래 {market_hint} 중 오늘 관심 가질 만한 종목을 최대 {top_n}개 선정하고 순위를 매기세요.
{rank_rule}
모든 문장(시장 코멘트, reasons)은 반드시 한국어로 작성하세요.

입력:
{json.dumps(payload, ensure_ascii=False, indent=2)}

반드시 아래 JSON 스키마만 출력하세요:
{{
  "market_commentary": "{commentary_hint}",
  "picks": [
    {{
      "symbol": "005930",
      "rank": 1,
      "score": 0-100 숫자,
      "reasons": ["한국어 사유1", "한국어 사유2", "한국어 사유3"],
      "stance": "watch|accumulate|avoid",
      "buy_price": 권장매수가(숫자),
      "sell_price": 권장매도가(숫자)
    }}
  ]
}}

규칙:
- picks는 candidates에 있는 symbol만 사용
- 기술지표와 headlines를 모두 고려
- score는 당신(LLM)의 종합 점수
- 위험 요인도 reasons에 최소 1개 포함
{mix_rule}
- buy_price / sell_price는 참고용 목표가이며, 현재가·볼린저 밴드(suggested_buy_price/suggested_sell_price, bb_lower/bb_upper)를 기준으로 현실적인 범위를 제시
- 일반적으로 buy_price <= 현재가 <= sell_price 를 유지 (이미 급등한 경우 buy_price는 현재가 근처 조정 매수선)
- 가격을 임의로 크게 벗어나게 만들지 말 것
- reasons와 market_commentary는 영어 금지, 한국어만
"""

    try:
        raw, used_provider = get_llm_router().chat(
            messages=[
                {"role": "system", "content": RECOMMEND_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            provider=provider or None,
            temperature=0.2,
            num_predict=min(settings.max_new_tokens, 2200),
        )
        parsed = _extract_json(raw)
        if not parsed or "picks" not in parsed:
            raise ValueError("invalid LLM JSON")

        by_symbol = {c["symbol"]: c for c in candidates}
        picks: List[Dict[str, Any]] = []
        seen = set()
        for p in parsed.get("picks", []):
            symbol = str(p.get("symbol", "")).strip()
            if symbol not in by_symbol or symbol in seen:
                continue
            base = by_symbol[symbol]
            reasons = p.get("reasons") or base["reasons"]
            if isinstance(reasons, str):
                reasons = [reasons]
            stance = str(p.get("stance", "watch"))
            score = p.get("score", base["score"])
            try:
                score = float(score)
            except Exception:
                score = float(base["score"])

            mkt = str(base.get("market") or market)
            fb_buy, fb_sell = _suggest_buy_sell(
                float(base.get("price") or 0),
                mkt,
                bb_lower=float(base.get("bb_lower") or 0) or None,
                bb_upper=float(base.get("bb_upper") or 0) or None,
            )
            buy_price = p.get("buy_price", base.get("buy_price", fb_buy))
            sell_price = p.get("sell_price", base.get("sell_price", fb_sell))
            try:
                buy_price = _round_px(float(buy_price), mkt)
            except Exception:
                buy_price = fb_buy
            try:
                sell_price = _round_px(float(sell_price), mkt)
            except Exception:
                sell_price = fb_sell
            if buy_price <= 0:
                buy_price = fb_buy
            if sell_price <= 0:
                sell_price = fb_sell

            picks.append({
                **base,
                "score": round(max(0.0, min(100.0, score)), 2),
                "reasons": [str(r) for r in reasons][:5],
                "stance": stance,
                "buy_price": buy_price,
                "sell_price": sell_price,
            })
            seen.add(symbol)
            if len(picks) >= top_n:
                break

        if not picks:
            raise ValueError("empty picks")

        picks.sort(key=lambda x: x["score"], reverse=True)
        for i, item in enumerate(picks, start=1):
            item["rank"] = i

        commentary = str(parsed.get("market_commentary") or "").strip()
        return picks, commentary, True, used_provider
    except Exception as e:
        log.warning("LLM recommend failed, fallback to technical rank: %s", e)
        fallback = []
        for i, item in enumerate(candidates[:top_n], start=1):
            mkt = str(item.get("market") or market)
            buy_price = item.get("buy_price")
            sell_price = item.get("sell_price")
            if not buy_price or not sell_price:
                buy_price, sell_price = _suggest_buy_sell(
                    float(item.get("price") or 0),
                    mkt,
                    bb_lower=float(item.get("bb_lower") or 0) or None,
                    bb_upper=float(item.get("bb_upper") or 0) or None,
                )
            fallback.append({
                **item,
                "rank": i,
                "stance": "watch",
                "buy_price": buy_price,
                "sell_price": sell_price,
            })
        return fallback, "기술지표 점수 기준으로 선정했습니다. (AI 응답 실패로 대체)", False, "none"


def _scan_item_view(row: Dict[str, Any]) -> dict:
    return {
        "rank": int(row.get("scan_rank") or 0),
        "symbol": row["symbol"],
        "name": row["name"],
        "market": row["market"],
        "score": row["score"],
        "price": row["price"],
        "change_pct": row["change_pct"],
        "rsi": row["rsi"],
        "macd_signal": row["macd_signal"],
        "trend": row["trend"],
        "reasons": row.get("reasons") or [],
        "sector": str(row.get("sector") or ""),
        "status_label": str(row.get("status_label") or ""),
        "highlights": list(row.get("highlights") or [])[:2],
        "metric_note": str(row.get("metric_note") or ""),
        "detail_summary": str(row.get("detail_summary") or row.get("ai_summary") or ""),
        "ai_summary": str(row.get("ai_summary") or row.get("detail_summary") or ""),
    }


def _empty_daily(market: str, as_of: str) -> dict:
    return {
        "as_of": as_of,
        "market": market,
        "items": [],
        "scan_items": [],
        "universe_size": 0,
        "universe_source": "",
        "shortlist_size": 0,
        "scanned_count": 0,
        "market_commentary": "",
        "used_llm": False,
        "provider": "none",
        "model": "none",
        "cached": False,
        "updated_at": None,
        "disclaimer": (
            "AI·기술지표·뉴스 기반 일일 참고 추천입니다. "
            "권장 매수/매도가는 참고용이며 투자 조언이 아닙니다."
        ),
    }


def build_daily_recommendations(
    market: str = "ALL",
    top_n: int | None = None,
    provider: str = "",
    force: bool = False,
    force_universe: bool = False,
) -> dict:
    settings = get_settings()
    top_n = top_n or settings.recommend_top_n
    days = settings.recommend_lookback_days
    shortlist_n = int(getattr(settings, "recommend_shortlist_size", 20) or 20)
    workers = int(getattr(settings, "recommend_scan_workers", 8) or 8)
    as_of = today_stamp()
    out_market = normalize_recommend_market(market)
    markets = markets_for(out_market)

    if not force:
        cached = load_daily(out_market, as_of=as_of)
        if cached:
            cached = dict(cached)
            cached["cached"] = True
            cached["market"] = out_market
            cached["as_of"] = cached.get("as_of") or as_of
            return cached
        return _empty_daily(out_market, as_of)

    # 0–1) 시장별 유니버스 + 기술지표 스캔 후 통합
    candidates: List[Dict[str, Any]] = []
    sources: List[str] = []
    universe_size = 0
    for mkt in markets:
        universe, uni_meta = universe_as_mapping(mkt, force=force_universe)
        yf_map = universe_yf_map(uni_meta)
        universe_size += int(uni_meta.get("size") or len(universe))
        src = str(uni_meta.get("source") or "")
        if src:
            sources.append(f"{mkt}:{src}")
        candidates.extend(
            _collect_candidates(mkt, days, universe, workers=workers, yf_map=yf_map)
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(candidates, start=1):
        item["scan_rank"] = i

    # 2) 상위 shortlist: 뉴스 + AI 브리핑 (분야·뉴스·개요)
    shortlist = candidates[:shortlist_n]
    shortlist = _attach_news_briefs(shortlist, per_stock=5)
    shortlist = _attach_ai_summaries(shortlist, provider=provider)
    summary_by_symbol = {s["symbol"]: s for s in shortlist}

    scan_items = []
    for c in candidates:
        merged = {**c, **summary_by_symbol.get(c["symbol"], {})}
        scan_items.append(_scan_item_view(merged))

    # 3) 선택 AI 엔진으로 최종 선정 (시장 범위에 맞는 순위)
    picks, commentary, used_llm, used_provider = _llm_rank(
        shortlist,
        out_market,
        top_n,
        as_of,
        provider=provider,
    )

    items = []
    for p in picks:
        extra = summary_by_symbol.get(p["symbol"], {})
        items.append({
            "rank": p["rank"],
            "symbol": p["symbol"],
            "name": p["name"],
            "market": p["market"],
            "score": p["score"],
            "price": p["price"],
            "change_pct": p["change_pct"],
            "reasons": p["reasons"],
            "rsi": p["rsi"],
            "macd_signal": p["macd_signal"],
            "trend": p["trend"],
            "stance": p.get("stance", "watch"),
            "buy_price": float(p.get("buy_price") or 0),
            "sell_price": float(p.get("sell_price") or 0),
            "sector": str(extra.get("sector") or p.get("sector") or ""),
            "status_label": str(extra.get("status_label") or p.get("status_label") or ""),
            "highlights": list(extra.get("highlights") or p.get("highlights") or [])[:2],
            "metric_note": str(extra.get("metric_note") or p.get("metric_note") or ""),
            "detail_summary": str(
                extra.get("detail_summary") or extra.get("ai_summary")
                or p.get("detail_summary") or p.get("ai_summary") or ""
            ),
            "ai_summary": str(
                extra.get("ai_summary") or extra.get("detail_summary")
                or p.get("ai_summary") or p.get("detail_summary") or ""
            ),
        })

    result = {
        "as_of": as_of,
        "market": out_market,
        "items": items,
        "scan_items": scan_items,
        "universe_size": universe_size,
        "universe_source": "+".join(sources) if sources else "",
        "shortlist_size": len(shortlist),
        "scanned_count": len(scan_items),
        "market_commentary": commentary,
        "used_llm": used_llm,
        "provider": used_provider,
        "model": used_provider if used_llm else "technical-only",
        "cached": False,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "disclaimer": (
            "AI·기술지표·뉴스 기반 일일 참고 추천입니다. "
            "권장 매수/매도가는 참고용이며 투자 조언이 아닙니다."
        ),
    }
    save_daily(out_market, result)
    return result
