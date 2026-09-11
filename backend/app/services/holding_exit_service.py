"""보유 종목 매도·축소·보유 점검 서비스."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import PROJECT_ROOT, get_settings
from app.services import broker
from app.services.indicators import calculate_indicators, latest_summary
from app.services.market_data import fetch_ohlcv
from app.services.rag.llm_router import get_llm_router
from app.services.recommend_cache import today_stamp

log = logging.getLogger(__name__)

_CACHE_DIR = PROJECT_ROOT / ".cache"
_CACHE_FILE = _CACHE_DIR / "holdings_exit_daily.json"

HOLDING_EXIT_SYSTEM = (
    "당신은 보유 주식 리스크 점검 애널리스트입니다. 제공된 보유현황과 기술지표만 사용해 "
    "각 종목의 보유/축소/매도 근거와 대응 시나리오를 한국어로 간결하게 보강하세요. 가격, 수익률, 뉴스를 만들지 마세요. "
    "투자 권유나 수익 보장 표현은 금지합니다. 유효한 JSON만 출력하세요."
)


def _load_cache(as_of: str) -> dict | None:
    try:
        if not _CACHE_FILE.exists():
            return None
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        payload = data.get("payload")
        if not isinstance(payload, dict):
            return None
        if str(data.get("as_of") or payload.get("as_of") or "") != as_of:
            return None
        return payload
    except Exception as exc:
        log.warning("holdings exit cache load failed: %s", exc)
        return None


def _save_cache(payload: dict[str, Any]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        body = {"as_of": payload.get("as_of") or today_stamp(), "cached_at": time.time(), "payload": payload}
        _CACHE_FILE.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("holdings exit cache save failed: %s", exc)


def _round_price(value: float, market: str) -> float:
    if str(market).upper() == "KRX":
        return float(int(round(value)))
    if value >= 1:
        return round(value, 2)
    if value >= 0.1:
        return round(value, 3)
    return round(value, 4)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value or 0)
        return out if out == out else default
    except Exception:
        return default


def _holding_cost_basis(row: dict[str, Any]) -> tuple[float, float]:
    qty = _safe_float(row.get("qty"))
    amount = _safe_float(row.get("amount"))
    profit = _safe_float(row.get("profit"))
    cost = max(0.0, amount - profit)
    avg = cost / qty if qty > 0 else 0.0
    return cost, avg


def _holding_currency(row: dict[str, Any]) -> str:
    cur = str(row.get("currency") or "").upper().strip()
    if cur:
        return cur
    return "USD" if str(row.get("scope") or "").lower() == "overseas" or str(row.get("market") or "").upper() == "US" else "KRW"


def _holding_fx(row: dict[str, Any], account: dict[str, Any] | None = None) -> float:
    currency = _holding_currency(row)
    if currency == "KRW":
        return 1.0
    for dep in (account or {}).get("deposits") or []:
        if str(dep.get("currency") or "").upper() == currency:
            fx = _safe_float(dep.get("exchange_rate"), 0.0)
            if fx > 0:
                return fx
    overseas = (account or {}).get("overseas") or {}
    fx = _safe_float(overseas.get("exchange_rate"), 0.0)
    if fx > 0:
        return fx
    return _safe_float((account or {}).get("default_exchange_rate"), 1300.0) or 1300.0


def _with_currency_amounts(row: dict[str, Any], account: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(row)
    currency = _holding_currency(out)
    fx = _holding_fx(out, account)
    amount = _safe_float(out.get("amount"))
    profit = _safe_float(out.get("profit"))
    cost, avg = _holding_cost_basis(out)
    out["currency"] = currency
    out["exchange_rate"] = fx
    out["amount_krw"] = amount * fx
    out["profit_krw"] = profit * fx
    out["cost_amount"] = cost
    out["cost_amount_krw"] = cost * fx
    out["avg_cost"] = avg
    return out


def _decision_from_rules(
    holding: dict[str, Any],
    summary: dict[str, Any],
    latest_row: Any,
    portfolio_weight: float,
) -> tuple[str, float, list[str], list[str]]:
    profit_rate = _safe_float(holding.get("profit_rate"))
    rsi = _safe_float(summary.get("rsi"), 50.0)
    macd = str(summary.get("macd_signal") or "").lower()
    trend = str(summary.get("trend_long") or "SIDEWAYS").upper()
    price = _safe_float(summary.get("price")) or _safe_float(holding.get("price"))
    sma20 = _safe_float(getattr(latest_row, "get", lambda *_: 0)("SMA_20", 0))
    sma60 = _safe_float(getattr(latest_row, "get", lambda *_: 0)("SMA_60", 0))
    sma120 = _safe_float(getattr(latest_row, "get", lambda *_: 0)("SMA_120", 0))
    bb_pos = _safe_float(getattr(latest_row, "get", lambda *_: 0.5)("BB_Position", 0.5), 0.5)
    volatility = _safe_float(getattr(latest_row, "get", lambda *_: 0)("Volatility", 0))

    risk = 0.0
    reasons: list[str] = []
    highlights: list[str] = []

    if profit_rate <= -8:
        risk += 30
        reasons.append(f"손절 기준 접근/이탈: 평가손익률 {profit_rate:.2f}%")
    elif profit_rate <= -4:
        risk += 14
        reasons.append(f"손실 구간: 평가손익률 {profit_rate:.2f}%")
    elif profit_rate >= 25:
        risk += 16
        reasons.append(f"익절 검토 구간: 평가손익률 {profit_rate:.2f}%")
    elif profit_rate >= 12:
        risk += 8
        reasons.append(f"수익 보호 검토: 평가손익률 {profit_rate:.2f}%")

    if rsi >= 75:
        risk += 18
        reasons.append(f"RSI {rsi:.1f}로 과열 부담")
    elif rsi >= 68:
        risk += 10
        reasons.append(f"RSI {rsi:.1f}로 상단권")
    elif rsi <= 30 and profit_rate < 0:
        risk += 12
        reasons.append(f"RSI {rsi:.1f} 약세 구간에서 손실 지속")

    if macd == "bearish":
        risk += 12
        reasons.append("MACD 하락 편향")
    else:
        reasons.append("MACD 상승 편향")
        risk -= 5

    if trend == "DOWN":
        risk += 18
        reasons.append("중기 추세 이탈(SMA20 <= SMA120)")
    else:
        reasons.append("중기 상승 추세 유지")
        risk -= 6

    if price > 0 and sma20 > 0 and price < sma20:
        risk += 10
        reasons.append("현재가가 20일 이동평균 아래")
    if price > 0 and sma60 > 0 and price < sma60:
        risk += 8
        reasons.append("현재가가 60일 이동평균 아래")
    if price > 0 and sma120 > 0 and price < sma120:
        risk += 12
        reasons.append("현재가가 120일 이동평균 아래")

    if bb_pos >= 0.95 and profit_rate > 8:
        risk += 10
        reasons.append("볼린저 밴드 상단권에서 수익 구간")
    elif bb_pos <= 0.08 and profit_rate < 0:
        risk += 8
        reasons.append("볼린저 밴드 하단 이탈권에서 손실 구간")

    if portfolio_weight >= 0.25:
        risk += 10
        reasons.append(f"포트폴리오 비중 {portfolio_weight * 100:.1f}%로 집중도 높음")
    elif portfolio_weight >= 0.15:
        risk += 5
        reasons.append(f"포트폴리오 비중 {portfolio_weight * 100:.1f}%")

    if volatility >= 5:
        risk += 5
        reasons.append(f"최근 변동성 {volatility:.1f}%")

    if risk >= 55:
        decision = "sell"
        highlights.append("매도 검토")
    elif risk >= 32:
        decision = "trim"
        highlights.append("비중 축소")
    else:
        decision = "hold"
        highlights.append("보유 유지")

    if profit_rate <= -8:
        stop = _round_price(price * 0.97, str(holding.get("market") or "KRX"))
        highlights.append(f"손절 방어선 {stop:g}")
    elif profit_rate >= 12:
        take = _round_price(price * 1.03, str(holding.get("market") or "KRX"))
        highlights.append(f"익절 추적선 {take:g}")

    return decision, max(0.0, min(100.0, risk)), reasons[:6], highlights[:3]


def _fallback_item(holding: dict[str, Any], rank: int, error: str) -> dict[str, Any]:
    price = _safe_float(holding.get("price"))
    profit_rate = _safe_float(holding.get("profit_rate"))
    decision = "trim" if profit_rate <= -8 else "hold"
    return {
        "rank": rank,
        "symbol": str(holding.get("symbol") or ""),
        "name": str(holding.get("name") or holding.get("symbol") or ""),
        "market": str(holding.get("market") or "KRX").upper(),
        "decision": decision,
        "decision_label": "비중 축소" if decision == "trim" else "보유",
        "risk_score": 50.0 if decision == "trim" else 20.0,
        "qty": _safe_float(holding.get("qty")),
        "price": price,
        "avg_cost": _safe_float(holding.get("avg_cost")),
        "cost_amount": _safe_float(holding.get("cost_amount")),
        "cost_amount_krw": _safe_float(holding.get("cost_amount_krw")),
        "amount": _safe_float(holding.get("amount")),
        "amount_krw": _safe_float(holding.get("amount_krw")),
        "profit": _safe_float(holding.get("profit")),
        "profit_krw": _safe_float(holding.get("profit_krw")),
        "profit_rate": profit_rate,
        "currency": _holding_currency(holding),
        "exchange_rate": _safe_float(holding.get("exchange_rate"), 1.0),
        "portfolio_weight": 0.0,
        "rsi": 50.0,
        "macd_signal": "NEUTRAL",
        "trend": "UNKNOWN",
        "reasons": [error or "기술지표 데이터를 불러오지 못했습니다."],
        "highlights": ["데이터 확인 필요"],
        "ai_summary": "기술지표 데이터 부족으로 보수적 점검만 가능합니다.",
        "ai_rationale": "차트 데이터가 부족해 손익률 중심의 보수적 판단만 제공합니다.",
        "ai_risk": "데이터 공백으로 추세·모멘텀 리스크를 충분히 확인할 수 없습니다.",
        "ai_action": "차트 데이터가 정상화된 뒤 재점검하고, 손실 확대 여부를 우선 확인하세요.",
        "ai_watchpoints": ["차트 데이터 확보", "손익률 변화", "포트폴리오 비중"],
    }


def _analyze_one(holding: dict[str, Any], rank: int, total_stock_value: float, days: int) -> dict[str, Any]:
    symbol = str(holding.get("symbol") or "").strip().upper()
    market = str(holding.get("market") or "KRX").upper()
    name = str(holding.get("name") or symbol)
    amount = _safe_float(holding.get("amount"))
    amount_krw = _safe_float(holding.get("amount_krw")) or amount
    weight = amount_krw / total_stock_value if total_stock_value > 0 else 0.0
    try:
        df = fetch_ohlcv(symbol, market=market, timeframe="day", days=days, allow_kis=False, prefer_yfinance=True)
        ind = calculate_indicators(df)
        if ind.empty:
            return _fallback_item(holding, rank, "차트 데이터가 없습니다.")
        summary = latest_summary(ind)
        row = ind.iloc[-1]
        decision, risk, reasons, highlights = _decision_from_rules(holding, summary, row, weight)
        label = {"sell": "매도", "trim": "비중 축소", "hold": "보유"}.get(decision, "보유")
        return {
            "rank": rank,
            "symbol": symbol,
            "name": name,
            "market": market,
            "decision": decision,
            "decision_label": label,
            "risk_score": round(risk, 2),
            "qty": _safe_float(holding.get("qty")),
            "price": _safe_float(summary.get("price")) or _safe_float(holding.get("price")),
            "avg_cost": _safe_float(holding.get("avg_cost")),
            "cost_amount": _safe_float(holding.get("cost_amount")),
            "cost_amount_krw": _safe_float(holding.get("cost_amount_krw")),
            "amount": amount,
            "amount_krw": amount_krw,
            "profit": _safe_float(holding.get("profit")),
            "profit_krw": _safe_float(holding.get("profit_krw")),
            "profit_rate": _safe_float(holding.get("profit_rate")),
            "currency": _holding_currency(holding),
            "exchange_rate": _safe_float(holding.get("exchange_rate"), 1.0),
            "portfolio_weight": round(weight, 6),
            "rsi": round(_safe_float(summary.get("rsi"), 50), 2),
            "macd_signal": "BUY" if str(summary.get("macd_signal")) == "bullish" else "SELL",
            "trend": str(summary.get("trend_long") or "SIDEWAYS"),
            "sma20": _safe_float(row.get("SMA_20", 0)),
            "sma60": _safe_float(row.get("SMA_60", 0)),
            "sma120": _safe_float(row.get("SMA_120", 0)),
            "bb_position": round(_safe_float(row.get("BB_Position", 0.5), 0.5), 4),
            "reasons": reasons,
            "highlights": highlights,
            "ai_summary": "",
            "ai_rationale": "",
            "ai_risk": "",
            "ai_action": "",
            "ai_watchpoints": [],
        }
    except Exception as exc:
        log.warning("holding exit analyze failed %s: %s", symbol, exc)
        return _fallback_item(holding, rank, str(exc))


def _attach_ai_context(items: list[dict[str, Any]], provider: str = "") -> tuple[list[dict[str, Any]], bool, str]:
    if not items:
        return items, False, "none"
    settings = get_settings()
    sample = [
        {
            "symbol": it["symbol"],
            "name": it["name"],
            "decision": it["decision"],
            "decision_label": it["decision_label"],
            "risk_score": it["risk_score"],
            "profit_rate": it["profit_rate"],
            "portfolio_weight": it["portfolio_weight"],
            "rsi": it["rsi"],
            "macd_signal": it["macd_signal"],
            "trend": it["trend"],
            "price": it["price"],
            "avg_cost": it["avg_cost"],
            "bb_position": it.get("bb_position"),
            "reasons": it["reasons"],
        }
        for it in items[:12]
    ]
    prompt = (
        "다음 보유 종목 점검 결과에 대해 symbol별 AI 분석을 작성하세요. "
        "반드시 제공된 수치와 reasons만 근거로 사용하세요. "
        "JSON 형식: {\"items\":[{\"symbol\":\"...\",\"ai_summary\":\"1문장 요약\","
        "\"ai_rationale\":\"판단 근거 1문장\",\"ai_risk\":\"핵심 리스크 1문장\","
        "\"ai_action\":\"대응 시나리오 1문장\",\"ai_watchpoints\":[\"관찰1\",\"관찰2\",\"관찰3\"]}]}\n"
        + json.dumps(sample, ensure_ascii=False)
    )
    try:
        raw, used_provider = get_llm_router().chat(
            messages=[{"role": "system", "content": HOLDING_EXIT_SYSTEM}, {"role": "user", "content": prompt}],
            provider=provider or None,
            temperature=0.2,
            num_predict=min(settings.max_new_tokens, 1600),
        )
        start = raw.find("{")
        end = raw.rfind("}")
        parsed = json.loads(raw[start : end + 1]) if start >= 0 and end >= start else {}
        by_symbol = {str(x.get("symbol") or "").upper(): x for x in parsed.get("items", []) if isinstance(x, dict)}
        for it in items:
            ai = by_symbol.get(str(it.get("symbol") or "").upper()) or {}
            _fill_ai_fields(it, ai)
        return items, True, used_provider
    except Exception as exc:
        log.warning("holding exit LLM context failed: %s", exc)
        for it in items:
            _fill_ai_fields(it, {})
        return items, False, "none"


def _rule_summary(item: dict[str, Any]) -> str:
    reason = (item.get("reasons") or [""])[0]
    return f"{item.get('decision_label', '보유')} 판단: {reason}".strip()


def _fill_ai_fields(item: dict[str, Any], ai: dict[str, Any]) -> None:
    reasons = item.get("reasons") or []
    highlights = item.get("highlights") or []
    first_reason = str(reasons[0] if reasons else "기술지표와 손익 기준을 종합했습니다.")
    first_highlight = str(highlights[0] if highlights else item.get("decision_label") or "보유")
    watchpoints = ai.get("ai_watchpoints") if isinstance(ai, dict) else []
    if not isinstance(watchpoints, list):
        watchpoints = []
    item["ai_summary"] = str(ai.get("ai_summary") or _rule_summary(item)).strip()
    item["ai_rationale"] = str(ai.get("ai_rationale") or f"{first_reason}이 핵심 판단 근거입니다.").strip()
    item["ai_risk"] = str(
        ai.get("ai_risk")
        or f"현재 판단은 {item.get('decision_label', '보유')}이며, 리스크 점수는 {float(item.get('risk_score') or 0):.0f}입니다."
    ).strip()
    item["ai_action"] = str(
        ai.get("ai_action")
        or f"{first_highlight} 관점에서 가격·추세·손익률 변화를 재점검하세요."
    ).strip()
    item["ai_watchpoints"] = [str(x).strip() for x in watchpoints if str(x).strip()][:3] or [
        "평가손익률",
        "RSI/MACD",
        "이동평균 이탈 여부",
    ]


def build_holdings_exit(provider: str = "", force: bool = False, days: int = 160) -> dict[str, Any]:
    as_of = today_stamp()
    if not force:
        cached = _load_cache(as_of)
        if cached:
            out = dict(cached)
            out["cached"] = True
            return out

    # force는 점검/AI 캐시만 무시한다. 계좌 강제조회까지 전달하면 KIS 유량 제한을 쉽게 유발한다.
    account = broker.get_account_overview(force=False)
    holdings = [_with_currency_amounts(h, account) for h in list(account.get("holdings") or [])]
    total_stock_value = (
        _safe_float(account.get("current_amount"))
        or sum(_safe_float(h.get("amount_krw")) for h in holdings)
    )
    if not holdings:
        result = {
            "as_of": as_of,
            "items": [],
            "summary": {"sell": 0, "trim": 0, "hold": 0, "total": 0},
            "used_llm": False,
            "provider": "none",
            "cached": False,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "disclaimer": "보유 종목 매도 점검은 기술지표와 계좌 손익 기반 참고 정보이며 투자 권유가 아닙니다.",
        }
        _save_cache(result)
        return result

    items = [_analyze_one(h, i, total_stock_value, days) for i, h in enumerate(holdings, start=1)]
    order = {"sell": 0, "trim": 1, "hold": 2}
    items.sort(key=lambda x: (order.get(str(x.get("decision")), 9), -_safe_float(x.get("risk_score"))))
    for i, item in enumerate(items, start=1):
        item["rank"] = i
    items, used_llm, used_provider = _attach_ai_context(items, provider=provider)
    summary = {"sell": 0, "trim": 0, "hold": 0, "total": len(items)}
    for item in items:
        key = str(item.get("decision") or "hold")
        if key in summary:
            summary[key] += 1
    result = {
        "as_of": as_of,
        "items": items,
        "summary": summary,
        "used_llm": used_llm,
        "provider": used_provider,
        "cached": False,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "disclaimer": "보유 종목 매도 점검은 기술지표와 계좌 손익 기반 참고 정보이며 투자 권유가 아닙니다.",
    }
    _save_cache(result)
    return result
