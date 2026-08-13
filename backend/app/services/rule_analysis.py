"""지표 기반 룰 분석 (LLM 없이 즉시 제공, AI 프롬프트 보조 입력)."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from app.services.indicators import calculate_indicators, latest_summary
from app.services.market_data import fetch_ohlcv, resolve_stock_name


def _f(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        v = float(row.get(key, default))
        if v != v:  # NaN
            return default
        return v
    except (TypeError, ValueError):
        return default


def _s(row: pd.Series, key: str, default: str = "") -> str:
    v = row.get(key, default)
    return str(v) if v is not None else default


def build_rule_analysis_from_df(
    ind: pd.DataFrame,
    *,
    symbol: str,
    market: str,
    stock_name: str = "",
) -> Dict[str, Any]:
    if ind is None or ind.empty:
        return {
            "symbol": symbol,
            "market": market.upper(),
            "stock_name": stock_name or symbol,
            "as_of": "",
            "price": 0.0,
            "score": 0.0,
            "stance": "watch",
            "bias": "neutral",
            "summary_text": "시세/지표 데이터가 없어 룰 분석을 수행할 수 없습니다.",
            "signals": [],
            "metrics": {},
            "rules": [],
            "horizon": {
                "medium_trend": "down",
                "medium_trend_label": "중기추세 확인 불가",
                "short_momentum": "weakening",
                "short_momentum_label": "단기모멘텀 확인 불가",
                "narrative": "데이터가 없어 시간축별 해석을 제공할 수 없습니다.",
                "macd_evidence": "",
                "rsi_zone": "중립",
            },
        }

    row = ind.iloc[-1]
    prev = ind.iloc[-2] if len(ind) > 1 else row
    summary = latest_summary(ind)
    as_of = (
        ind.index[-1].strftime("%Y-%m-%d")
        if hasattr(ind.index[-1], "strftime")
        else str(ind.index[-1])[:10]
    )

    rsi = _f(row, "RSI_14", 50)
    macd = _f(row, "MACD")
    signal = _f(row, "Signal")
    hist = _f(row, "MACD_Histogram")
    prev_hist = _f(prev, "MACD_Histogram")
    bb_pos = _f(row, "BB_Position", 0.5)
    stoch_k = _f(row, "%K", 50)
    stoch_d = _f(row, "%D", 50)
    vol_ratio = _f(row, "Volume_Ratio", 1)
    change = _f(row, "Price_Change")
    sma20 = _f(row, "SMA_20")
    sma60 = _f(row, "SMA_60")
    sma120 = _f(row, "SMA_120")
    price = _f(row, "close")
    trend20 = _s(row, "Trend_20", "DOWN")
    trend_long = _s(row, "trend_long", "DOWN")
    macd_cross = bool(row.get("MACD_Cross", False))

    score = 50.0
    rules: List[Dict[str, Any]] = []
    signals: List[str] = []

    def add(rule_id: str, title: str, detail: str, direction: str, weight: float) -> None:
        nonlocal score
        score += weight
        rules.append(
            {
                "id": rule_id,
                "title": title,
                "detail": detail,
                "direction": direction,  # bullish | bearish | neutral
                "weight": weight,
            }
        )
        if direction != "neutral":
            signals.append(f"{title}: {detail}")

    # RSI
    if rsi < 30:
        add("rsi_oversold", "RSI 과매도", f"RSI {rsi:.1f} (<30)", "bullish", 8)
    elif rsi < 40:
        add("rsi_low", "RSI 낮은 구간", f"RSI {rsi:.1f} (30-40, 반등 관찰)", "bullish", 5)
    elif rsi > 70:
        add("rsi_overbought", "RSI 과매수", f"RSI {rsi:.1f} (>70)", "bearish", -10)
    elif rsi > 60:
        add("rsi_high", "RSI 높은 구간", f"RSI {rsi:.1f} (60-70)", "bearish", -3)
    else:
        add("rsi_neutral", "RSI 중립", f"RSI {rsi:.1f}", "neutral", 2)

    # MACD
    if macd > signal:
        w = 10 if hist > prev_hist else 6
        add(
            "macd_bull",
            "MACD 상승 우위",
            f"MACD {macd:.4f} > Signal {signal:.4f}, hist {hist:.4f}",
            "bullish",
            w,
        )
    else:
        w = -8 if hist < prev_hist else -4
        add(
            "macd_bear",
            "MACD 하락 우위",
            f"MACD {macd:.4f} < Signal {signal:.4f}, hist {hist:.4f}",
            "bearish",
            w,
        )
    if macd_cross and macd > signal:
        add("macd_golden", "MACD 골든크로스", "MACD가 Signal을 상향 돌파", "bullish", 8)
    elif macd_cross and macd < signal:
        add("macd_dead", "MACD 데드크로스", "MACD가 Signal을 하향 이탈", "bearish", -8)

    # Trend / MA
    if price > sma20 > sma60:
        add("ma_stack_bull", "이평 정배열 단기", "종가 > SMA20 > SMA60", "bullish", 8)
    elif price < sma20 < sma60:
        add("ma_stack_bear", "이평 역배열 단기", "종가 < SMA20 < SMA60", "bearish", -8)
    elif trend20 == "UP":
        add("trend20_up", "단기 상승 추세", "종가 > SMA20", "bullish", 4)
    else:
        add("trend20_down", "단기 하락 추세", "종가 < SMA20", "bearish", -4)

    if trend_long == "UP":
        add("trend_long_up", "중장기 상승", "SMA20 > SMA120", "bullish", 10)
    else:
        add("trend_long_down", "중장기 하락", "SMA20 < SMA120", "bearish", -8)

    if sma20 and abs(price - sma20) / sma20 < 0.01:
        add("near_sma20", "SMA20 근접", "지지/저항 테스트 구간", "neutral", 0)

    # Bollinger
    if bb_pos <= 0.15:
        add("bb_lower", "볼린저 하단 근접", f"BB position {bb_pos:.2f}", "bullish", 7)
    elif bb_pos >= 0.85:
        add("bb_upper", "볼린저 상단 근접", f"BB position {bb_pos:.2f}", "bearish", -7)
    elif 0.35 <= bb_pos <= 0.65:
        add("bb_mid", "볼린저 중단 밴드", f"BB position {bb_pos:.2f}", "neutral", 1)

    # Stochastic
    if stoch_k < 20 and stoch_k > stoch_d:
        add("stoch_turn_up", "스토캐스틱 과매도 반등", f"%K {stoch_k:.1f} > %D {stoch_d:.1f}", "bullish", 6)
    elif stoch_k > 80 and stoch_k < stoch_d:
        add("stoch_turn_down", "스토캐스틱 과매수 꺾임", f"%K {stoch_k:.1f} < %D {stoch_d:.1f}", "bearish", -6)
    elif stoch_k > 80:
        add("stoch_ob", "스토캐스틱 과매수", f"%K {stoch_k:.1f}", "bearish", -3)
    elif stoch_k < 20:
        add("stoch_os", "스토캐스틱 과매도", f"%K {stoch_k:.1f}", "bullish", 3)

    # Volume / price change
    if vol_ratio >= 2.0:
        dir_ = "bullish" if change >= 0 else "bearish"
        add("vol_surge", "거래량 급증", f"{vol_ratio:.1f}x vs 20일 평균", dir_, 6 if change >= 0 else -4)
    elif vol_ratio <= 0.5:
        add("vol_dry", "거래량 위축", f"{vol_ratio:.1f}x", "neutral", -1)

    if change >= 5:
        add("spike_up", "급등 주의", f"일간 {change:+.2f}%", "bearish", -3)
    elif change <= -5:
        add("spike_down", "급락", f"일간 {change:+.2f}%", "bullish", 2)
    elif 0 < change < 3:
        add("mild_up", "완만한 상승", f"일간 {change:+.2f}%", "bullish", 3)

    score = max(0.0, min(100.0, score))
    if score >= 65:
        stance, bias = "buy", "bullish"
    elif score >= 55:
        stance, bias = "watch", "slightly_bullish"
    elif score <= 35:
        stance, bias = "avoid", "bearish"
    elif score <= 45:
        stance, bias = "watch", "slightly_bearish"
    else:
        stance, bias = "watch", "neutral"

    bull = sum(1 for r in rules if r["direction"] == "bullish")
    bear = sum(1 for r in rules if r["direction"] == "bearish")

    # 시간축 분리: 중기 추세(이평) vs 단기 모멘텀(MACD/RSI)
    medium_up = trend_long == "UP"
    short_improving = macd > signal
    medium_label = "중기추세 상승" if medium_up else "중기추세 하락"
    short_label = "단기모멘텀 개선" if short_improving else "단기모멘텀 둔화"
    if not medium_up and short_improving:
        narrative = (
            "하락 추세 안에서 단기 반등 신호가 발생한 상태입니다. "
            "중장기 이평 아래에 있으나 MACD가 상승 우위로 전환된 조합으로, "
            "추세 전환 확인 전까지는 반등 구간의 리스크 관리가 필요합니다."
        )
    elif medium_up and not short_improving:
        narrative = (
            "상승 추세 안에서 단기 모멘텀이 둔화된 상태입니다. "
            "중기 방향은 우상향이나 MACD 약화로 조정·횡보 가능성을 염두에 두세요."
        )
    elif medium_up and short_improving:
        narrative = "중기 추세와 단기 모멘텀이 함께 우호적인 상태입니다."
    else:
        narrative = "중기 추세와 단기 모멘텀이 함께 비우호적인 상태입니다."

    macd_evidence = (
        f"MACD {macd:.4f} {'>' if macd > signal else '<'} Signal {signal:.4f}, "
        f"Histogram {hist:+.4f}"
    )

    summary_text = (
        f"{stock_name or symbol} 지표 룰 분석 점수 {score:.0f}/100. "
        f"{medium_label} / {short_label}. "
        f"스탠스={stance}. 강세 룰 {bull}개 / 약세 룰 {bear}개. "
        f"{narrative}"
    )

    horizon = {
        "medium_trend": "up" if medium_up else "down",
        "medium_trend_label": medium_label,
        "short_momentum": "improving" if short_improving else "weakening",
        "short_momentum_label": short_label,
        "narrative": narrative,
        "macd_evidence": macd_evidence,
        "rsi_zone": (
            "과매수" if rsi > 70 else "과매도" if rsi < 30 else "중립"
        ),
    }

    metrics = {
        "price": price,
        "price_change": change,
        "rsi": round(rsi, 2),
        "macd": round(macd, 4),
        "macd_signal_line": round(signal, 4),
        "macd_histogram": round(hist, 4),
        "macd_bias": summary.get("macd_signal", "bearish"),
        "macd_evidence": macd_evidence,
        "bb_position": round(bb_pos, 3),
        "stoch_k": round(stoch_k, 2),
        "stoch_d": round(stoch_d, 2),
        "volume_ratio": round(vol_ratio, 2),
        "sma_20": round(sma20, 4),
        "sma_60": round(sma60, 4),
        "sma_120": round(sma120, 4),
        "trend_20": trend20,
        "trend_long": trend_long,
        "medium_trend_label": medium_label,
        "short_momentum_label": short_label,
    }

    return {
        "symbol": symbol,
        "market": market.upper(),
        "stock_name": stock_name or symbol,
        "as_of": as_of,
        "price": price,
        "score": round(score, 1),
        "stance": stance,
        "bias": bias,
        "summary_text": summary_text,
        "signals": signals[:8],
        "metrics": metrics,
        "rules": rules,
        "horizon": horizon,
    }


def build_rule_analysis(
    symbol: str,
    market: str = "KRX",
    stock_name: str = "",
    days: int = 120,
) -> Dict[str, Any]:
    name = stock_name or resolve_stock_name(symbol, market)
    df = fetch_ohlcv(symbol, market=market, timeframe="day", days=days)
    if df.empty:
        return build_rule_analysis_from_df(
            pd.DataFrame(), symbol=symbol, market=market, stock_name=name
        )
    ind = calculate_indicators(df)
    return build_rule_analysis_from_df(ind, symbol=symbol, market=market, stock_name=name)


def format_rule_analysis_for_prompt(rule: Dict[str, Any]) -> str:
    """LLM에 넣을 구조화 텍스트."""
    if not rule:
        return "Rule analysis unavailable."
    horizon = rule.get("horizon") or {}
    lines = [
        f"Rule-based technical score: {rule.get('score', 0)}/100",
        f"Stance: {rule.get('stance')} | Bias: {rule.get('bias')}",
        f"Medium-term trend: {horizon.get('medium_trend_label', 'N/A')}",
        f"Short-term momentum: {horizon.get('short_momentum_label', 'N/A')}",
        f"Horizon narrative: {horizon.get('narrative', '')}",
        f"MACD evidence: {horizon.get('macd_evidence') or (rule.get('metrics') or {}).get('macd_evidence', '')}",
        f"Summary: {rule.get('summary_text', '')}",
        "Key metrics:",
    ]
    metrics = rule.get("metrics") or {}
    for k in (
        "price",
        "price_change",
        "rsi",
        "macd",
        "macd_signal_line",
        "macd_histogram",
        "macd_bias",
        "bb_position",
        "stoch_k",
        "volume_ratio",
        "trend_20",
        "trend_long",
        "medium_trend_label",
        "short_momentum_label",
    ):
        if k in metrics:
            lines.append(f"- {k}: {metrics[k]}")
    lines.append("Triggered rules:")
    for r in rule.get("rules") or []:
        lines.append(
            f"- [{r.get('direction')}] {r.get('title')} ({r.get('weight'):+g}): {r.get('detail')}"
        )
    return "\n".join(lines)
