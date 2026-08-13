"""시세/차트 데이터 수집 (공용 소스 우선, KIS는 보조)."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd

from app.services import kis_client
from app.services.indicators import calculate_indicators, ensure_ohlcv, latest_summary, series_to_list

log = logging.getLogger(__name__)

# 분/시간봉 KIS 응답 캐시 (유량 보호)
_OHLCV_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
_OHLCV_TTL_INTRADAY = 25.0
_OHLCV_TTL_DAILY = 120.0

POPULAR_KR = {
    "005930": "Samsung Electronics",
    "000660": "SK hynix",
    "034220": "LG Display",
    "035420": "NAVER",
    "035720": "Kakao",
    "207940": "Samsung Biologics",
    "051910": "LG Chem",
    "006400": "Samsung SDI",
    "005380": "Hyundai Motor",
    "000270": "Kia",
    "005490": "POSCO Holdings",
    "105560": "KB Financial",
    "055550": "Shinhan Financial",
    "066570": "LG Electronics",
    "068270": "Celltrion",
}

POPULAR_US = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "META": "Meta",
    "NVDA": "NVIDIA",
    "AMD": "AMD",
}


def _from_kis(symbol: str, timeframe: str, minute_interval: int, hour_interval: int) -> pd.DataFrame:
    if kis_client.is_rate_limited():
        log.debug("KIS chart skipped (rate-limit cooldown): %s", symbol)
        return pd.DataFrame()
    kis = kis_client.get_kis()
    if not kis:
        return pd.DataFrame()
    try:
        stock = kis.stock(symbol)
        if timeframe in {"realtime", "minute"}:
            bars = stock.chart(period=max(1, int(minute_interval)))
        elif timeframe == "hour":
            bars = stock.chart(period=60 * max(1, int(hour_interval)))
        elif timeframe == "week":
            bars = stock.chart(period="week")
        elif timeframe == "month":
            bars = stock.chart(period="month")
        else:
            bars = stock.chart()

        rows = []
        for b in bars:
            dt = getattr(b, "dt", None) or getattr(b, "time", None)
            rows.append({
                "datetime": pd.to_datetime(dt),
                "open": float(getattr(b, "open", 0) or 0),
                "high": float(getattr(b, "high", 0) or 0),
                "low": float(getattr(b, "low", 0) or 0),
                "close": float(getattr(b, "close", 0) or 0),
                "volume": float(getattr(b, "volume", 0) or 0),
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("datetime").sort_index()
        return ensure_ohlcv(df)
    except Exception as e:
        msg = str(e)
        if "호출 횟수" in msg or "EGW00201" in msg or "EGW00204" in msg:
            kis_client.mark_rate_limited()
        log.warning("KIS chart failed for %s: %s", symbol, e)
        return pd.DataFrame()


def _from_pykrx(symbol: str, days: int = 180) -> pd.DataFrame:
    try:
        from pykrx import stock
        end = datetime.now()
        start = end - timedelta(days=max(days * 2, 90))
        df = stock.get_market_ohlcv_by_date(
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            symbol,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume",
        })
        df.index = pd.to_datetime(df.index)
        return ensure_ohlcv(df.tail(days))
    except Exception as e:
        log.warning("pykrx failed for %s: %s", symbol, e)
        return pd.DataFrame()


def _yf_ticker_candidates(symbol: str, market: str, yf_symbol: str = "") -> list[str]:
    """Yahoo 티커 후보. yf_symbol이 있으면 그걸 최우선 (KS/KQ 오탐 로그 방지)."""
    preferred = (yf_symbol or "").strip()
    if preferred:
        out = [preferred]
        # 혹시 실패하면 기존 후보도 시도
        for c in _yf_ticker_candidates(symbol, market, yf_symbol=""):
            if c not in out:
                out.append(c)
        return out
    if market.upper() == "KRX" or (symbol or "").isdigit():
        # KQ를 먼저: 코스닥을 KS로 치면 delisted 에러 로그가 남음
        bare = symbol
        return [f"{bare}.KQ", f"{bare}.KS", bare]
    return [symbol]


def _from_yfinance(
    symbol: str,
    days: int = 180,
    market: str = "US",
    yf_symbol: str = "",
) -> pd.DataFrame:
    try:
        import logging as _logging

        import yfinance as yf

        period = f"{max(days, 30)}d"
        # yfinance 내부 ERROR 로그 소음 완화 (후보 순회 중 실패는 정상)
        yf_logger = _logging.getLogger("yfinance")
        prev_level = yf_logger.level
        yf_logger.setLevel(_logging.CRITICAL)
        try:
            for ticker_id in _yf_ticker_candidates(symbol, market, yf_symbol=yf_symbol):
                ticker = yf.Ticker(ticker_id)
                df = ticker.history(period=period)
                if df is None or df.empty:
                    continue
                df = df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
                })
                return ensure_ohlcv(df)
        finally:
            yf_logger.setLevel(prev_level)
        return pd.DataFrame()
    except Exception as e:
        log.warning("yfinance failed for %s: %s", symbol, e)
        return pd.DataFrame()


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = (
        df.resample(rule, label="right", closed="right")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna(subset=["close"])
    )
    return ensure_ohlcv(out)


def _intraday_yf_plan(timeframe: str, minute_interval: int, hour_interval: int) -> tuple[str, str, Optional[str]]:
    """
    Returns:
        (yf_interval, yf_period, resample_rule|None)
    yfinance 제한: 1m/5m ≈ 7일, 15m/30m/60m ≈ 60일
    """
    tf = (timeframe or "day").lower()
    if tf in {"realtime", "minute"}:
        mins = max(1, int(minute_interval))
        if mins <= 1:
            return "1m", "7d", None
        if mins == 2:
            return "2m", "7d", None
        if mins == 3:
            return "1m", "7d", "3min"
        if mins == 5:
            return "5m", "7d", None
        if mins == 10:
            return "5m", "7d", "10min"
        if mins == 15:
            return "15m", "60d", None
        if mins == 30:
            return "30m", "60d", None
        if mins < 60:
            return "5m", "7d", f"{mins}min"
        # 60분 이상은 hour 경로로 처리
        return "60m", "60d", None

    hours = max(1, int(hour_interval))
    if hours == 1:
        return "60m", "60d", None
    if hours == 2:
        return "60m", "60d", "2h"
    if hours == 4:
        return "60m", "60d", "4h"
    return "60m", "60d", f"{hours}h"


def _from_yfinance_intraday(
    symbol: str,
    market: str,
    timeframe: str,
    minute_interval: int,
    hour_interval: int,
) -> pd.DataFrame:
    """다중 거래일 분/시간봉 (yfinance). KIS 당일분봉 한계를 보완."""
    try:
        import yfinance as yf

        interval, period, resample = _intraday_yf_plan(timeframe, minute_interval, hour_interval)
        for ticker_id in _yf_ticker_candidates(symbol, market):
            ticker = yf.Ticker(ticker_id)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
            if df is None or df.empty:
                continue
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
            })
            df = ensure_ohlcv(df)
            if resample:
                df = _resample_ohlcv(df, resample)
            if not df.empty:
                log.debug(
                    "yfinance intraday %s interval=%s period=%s bars=%s",
                    ticker_id, interval, period, len(df),
                )
                return df
        return pd.DataFrame()
    except Exception as e:
        log.warning("yfinance intraday failed for %s: %s", symbol, e)
        return pd.DataFrame()


def _merge_ohlcv(base: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    """base(과거) + overlay(최신/당일) 병합. 동일 시각은 overlay 우선."""
    if base is None or base.empty:
        return ensure_ohlcv(overlay) if overlay is not None else pd.DataFrame()
    if overlay is None or overlay.empty:
        return ensure_ohlcv(base)

    a = ensure_ohlcv(base.copy())
    b = ensure_ohlcv(overlay.copy())
    # tz 통일: 둘 다 naive 또는 둘 다 같은 tz로
    if getattr(a.index, "tz", None) is not None and getattr(b.index, "tz", None) is None:
        b.index = pd.to_datetime(b.index).tz_localize(a.index.tz)
    elif getattr(b.index, "tz", None) is not None and getattr(a.index, "tz", None) is None:
        a.index = pd.to_datetime(a.index).tz_localize(b.index.tz)
    elif getattr(a.index, "tz", None) is not None and getattr(b.index, "tz", None) is not None:
        b.index = b.index.tz_convert(a.index.tz)

    combined = pd.concat([a, b])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return ensure_ohlcv(combined)


def _resample_week(df: pd.DataFrame) -> pd.DataFrame:
    return _resample_ohlcv(df, "W-FRI")


def _fetch_public_daily(symbol: str, market: str, days: int) -> pd.DataFrame:
    if market.upper() == "KRX" or symbol.isdigit():
        df = _from_pykrx(symbol, days=days)
        if df.empty:
            df = _from_yfinance(symbol, days=days, market="KRX")
        return df
    return _from_yfinance(symbol, days=days, market="US")


def fetch_ohlcv(
    symbol: str,
    market: str = "KRX",
    timeframe: str = "day",
    minute_interval: int = 1,
    hour_interval: int = 1,
    days: int = 180,
    *,
    allow_kis: bool = True,
    prefer_yfinance: bool = False,
    yf_symbol: str = "",
) -> pd.DataFrame:
    """
    일/주봉: 공용 데이터 우선.
    분/시간봉: yfinance 다일치 + KIS 당일 병합 (KIS는 당일만 제공).

    allow_kis=False 이면 공용 소스만 사용 (추천 대량 스캔용).
    prefer_yfinance=True 이면 일/주봉도 yfinance 우선 (추천 경로 통일).
    yf_symbol: Yahoo 심볼(예: 125490.KQ) — KS/KQ 오탐 방지.
    """
    tf = (timeframe or "day").lower()
    is_intraday = tf in {"realtime", "minute", "hour"}
    cache_key = (
        f"{symbol}|{market}|{tf}|{minute_interval}|{hour_interval}|{days}"
        f"|kis={int(allow_kis)}|yf={int(prefer_yfinance)}|{yf_symbol or '-'}"
    )
    ttl = _OHLCV_TTL_INTRADAY if is_intraday else _OHLCV_TTL_DAILY
    cached = _OHLCV_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < ttl and not cached[1].empty:
        return cached[1].copy()

    df = pd.DataFrame()

    if is_intraday:
        hist = _from_yfinance_intraday(
            symbol, market, timeframe, minute_interval, hour_interval,
        )
        today = pd.DataFrame()
        need_kis = (
            allow_kis
            and hist.empty
            and not kis_client.is_rate_limited()
        )
        if need_kis:
            today = _from_kis(symbol, timeframe, minute_interval, hour_interval)

        if not hist.empty and not today.empty:
            hist_n = _to_market_naive(hist, market, symbol)
            today_n = _to_market_naive(today, market, symbol)
            df = _merge_ohlcv(hist_n, today_n)
        elif not hist.empty:
            df = hist
        else:
            df = today
    elif tf == "week":
        if prefer_yfinance:
            daily = _from_yfinance(
                symbol, days=max(days * 2, 365), market=market, yf_symbol=yf_symbol,
            )
        else:
            daily = _fetch_public_daily(symbol, market, days=max(days * 2, 365))
        if not daily.empty:
            df = _resample_week(daily)
        if df.empty and allow_kis and not kis_client.is_rate_limited():
            df = _from_kis(symbol, "week", minute_interval, hour_interval)
    else:
        if prefer_yfinance:
            df = _from_yfinance(symbol, days=days, market=market, yf_symbol=yf_symbol)
        else:
            df = _fetch_public_daily(symbol, market, days=days)
        if df.empty and allow_kis and not kis_client.is_rate_limited():
            df = _from_kis(symbol, timeframe, minute_interval, hour_interval)

    if not df.empty:
        _OHLCV_CACHE[cache_key] = (time.time(), df.copy())
    return df


def _market_tz(market: str, symbol: str = "") -> str:
    if (market or "").upper() == "KRX" or (symbol or "").isdigit():
        return "Asia/Seoul"
    return "America/New_York"


def _to_market_naive(df: pd.DataFrame, market: str, symbol: str = "") -> pd.DataFrame:
    """타임존을 거래소 로컬 시각(naive)으로 통일. 차트 축 표시용."""
    if df is None or df.empty:
        return df
    out = df.copy()
    tz_name = _market_tz(market, symbol)
    idx = out.index
    try:
        if getattr(idx, "tz", None) is not None:
            out.index = idx.tz_convert(tz_name).tz_localize(None)
        else:
            # tz-naive면 이미 로컬 시각으로 간주
            out.index = pd.to_datetime(idx)
    except Exception:
        out.index = pd.to_datetime(idx).tz_localize(None)
    return out


def _format_bar_time(idx, timeframe: str) -> str:
    """분/시간봉은 거래소 로컬 wall-clock 문자열로 내려보냄."""
    tf = (timeframe or "day").lower()
    ts = pd.Timestamp(idx)
    if getattr(ts, "tz", None) is not None:
        # 안전장치: 아직 tz가 있으면 제거 전 로컬 시각 유지
        ts = ts.tz_convert("Asia/Seoul").tz_localize(None)
    if tf in {"realtime", "minute", "hour"}:
        return ts.strftime("%Y-%m-%dT%H:%M:%S")
    return ts.strftime("%Y-%m-%d")


def build_chart_payload(
    symbol: str,
    market: str = "KRX",
    timeframe: str = "day",
    minute_interval: int = 1,
    hour_interval: int = 1,
    days: int = 180,
) -> dict:
    raw = fetch_ohlcv(symbol, market, timeframe, minute_interval, hour_interval, days)
    if raw.empty:
        return {
            "symbol": symbol,
            "market": market,
            "timeframe": timeframe,
            "bars": [],
            "indicators": {},
            "summary": {},
        }

    # KIS 분봉은 Asia/Seoul tz-aware → 차트 표시를 위해 로컬 naive로 변환
    raw = _to_market_naive(raw, market, symbol)
    ind = calculate_indicators(raw)
    bars = []
    for idx, row in ind.iterrows():
        bars.append({
            "time": _format_bar_time(idx, timeframe),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })

    indicator_cols = [
        "sma_20", "sma_60", "sma_120", "ema_20",
        "rsi", "macd", "macd_signal", "macd_histogram",
        "stoch_k", "stoch_d",
        "bb20_upper", "bb20_middle", "bb20_lower",
        "volume",
    ]
    return {
        "symbol": symbol,
        "market": market,
        "timeframe": timeframe,
        "bars": bars,
        "indicators": series_to_list(ind, indicator_cols),
        "summary": latest_summary(ind),
    }


def resolve_stock_name(symbol: str, market: str = "KRX") -> str:
    from app.services.symbol_utils import (
        is_plausible_symbol,
        looks_like_kr_ticker,
        normalize_symbol,
    )

    key = normalize_symbol(symbol, market) or (symbol or "").strip()
    if not key:
        return symbol or ""

    if key in POPULAR_KR:
        return POPULAR_KR[key]
    if key in POPULAR_US:
        return POPULAR_US[key]

    if not is_plausible_symbol(key, market):
        # 한글 종목명 등은 그대로 표시용으로만 반환 (외부 API 호출 금지)
        return symbol.strip() if symbol else key

    # KIS 추가 호출 금지: 이미 캐시된 시세 이름만 사용
    cached_quote = kis_client.peek_cached_quote(key)
    if cached_quote.get("name"):
        return str(cached_quote["name"])

    # 해외: yfinance (실패해도 심볼 반환)
    if market.upper() != "KRX" and not looks_like_kr_ticker(key):
        try:
            import yfinance as yf

            info = yf.Ticker(key).info or {}
            return info.get("shortName") or info.get("longName") or key
        except Exception:
            return key

    # 국내: 6자리일 때만 pykrx (잘못된 코드로 NoneType 로그 폭주 방지)
    if looks_like_kr_ticker(key):
        try:
            from pykrx import stock

            name = stock.get_market_ticker_name(key)
            return name or key
        except Exception:
            return key
    return key
