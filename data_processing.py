from typing import List, Optional
import numpy as np
import pandas as pd
import streamlit as st

# Optional TA-Lib (fallback to simple numpy where needed)
try:
    import talib as ta  # type: ignore
    HAS_TALIB = True
except Exception:
    HAS_TALIB = False

# Local imports
from chart_manager import ChartManager


def to_datetime_index(df: pd.DataFrame, column_candidates: List[str]) -> pd.DataFrame:
    """Set one of candidate columns (or index) to DatetimeIndex and sort ascending."""
    for col in column_candidates:
        if col in df.columns:
            out = df.copy()
            out[col] = pd.to_datetime(out[col], errors="coerce")
            out = out.set_index(col).sort_index()
            return out
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    # last resort: try to parse index
    out = df.copy()
    try:
        out.index = pd.to_datetime(out.index, errors="coerce")
        return out.sort_index()
    except Exception:
        return df


def ensure_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and enforce numeric types for open/high/low/close/volume."""
    rename_map_candidates = [
        {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"},
        {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"},
        {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"},
    ]
    out = df.copy()
    for mp in rename_map_candidates:
        inter = set(mp.keys()).intersection(out.columns)
        if inter:
            out = out.rename(columns=mp)

    # fill missing columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            if col == "volume":
                out[col] = 0
            else:
                out[col] = out["close"] if "close" in out.columns else 0

    # numeric casting
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["close"]).sort_index()
    return out


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample to given frequency if DatetimeIndex is present."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    freq = freq.replace("H", "h").replace("T", "min")
    rs = df.resample(freq).agg(agg).dropna(how="any")
    return rs


def dmi_adx_trendline(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Compute a simple DMI-ADX derived single 'trendline' with buy/sell flags."""
    # Ensure series
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)
    close = pd.Series(close).astype(float)
    index = close.index

    if HAS_TALIB:
        plus_di = pd.Series(ta.PLUS_DI(high.values, low.values, close.values, timeperiod=period), index=index)
        minus_di = pd.Series(ta.MINUS_DI(high.values, low.values, close.values, timeperiod=period), index=index)
        adx = pd.Series(ta.ADX(high.values, low.values, close.values, timeperiod=period), index=index)
    else:
        # Lightweight fallback (not identical to TA-Lib)
        tr_up = high.diff()
        tr_down = -low.diff()
        plus_dm = np.where((tr_up > tr_down) & (tr_up > 0), tr_up, 0.0)
        minus_dm = np.where((tr_down > tr_up) & (tr_down > 0), tr_down, 0.0)
        atr = (high - low).rolling(period).mean().replace(0, np.nan)
        plus_di = pd.Series(100 * (pd.Series(plus_dm).rolling(period).sum() / atr), index=index)
        minus_di = pd.Series(100 * (pd.Series(minus_dm).rolling(period).sum() / atr), index=index)
        adx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).rolling(period).mean()

    raw = (plus_di - minus_di) * adx
    max_abs = raw.abs().max()
    trendline = pd.Series(50.0, index=index) if (pd.isna(max_abs) or max_abs == 0) else (50 + 50 * (raw / max_abs))

    buy_signal = (plus_di > minus_di) & (trendline > trendline.shift(1))
    sell_signal = (minus_di > plus_di) & (trendline < trendline.shift(1))

    return pd.DataFrame({"TrendLine": trendline, "BuySignal": buy_signal, "SellSignal": sell_signal}, index=index)


def calculate_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA/BB/RSI/MACD/Stochastic/Volume/DMI-derived trendline + standardized aliases."""
    if df.empty:
        return df
    out = df.copy()
    out = ChartManager.add_moving_averages(out, [20, 120])
    out = ChartManager.add_bollinger_bands(out, standard_col="close", window=20, std_dev=2)
    out = ChartManager.add_bollinger_bands(out, standard_col="open", window=4, std_dev=4)  # 4-4 BB
    out = ChartManager.add_rsi(out, window=14)
    out = ChartManager.add_macd(out)
    out = ChartManager.add_stochastic(out, k_window=14, d_window=3)
    out = ChartManager.add_volume_indicators(out)

    # DMI-ADX Trendline
    try:
        tl = dmi_adx_trendline(out["high"], out["low"], out["close"])
        out["TrendLine"] = tl["TrendLine"].reindex(out.index)
        out["TrendBuy"] = tl["BuySignal"].reindex(out.index)
        out["TrendSell"] = tl["SellSignal"].reindex(out.index)
    except Exception:
        out["TrendLine"] = np.nan
        out["TrendBuy"] = False
        out["TrendSell"] = False

    # Standardized aliases used by chart layer
    out["sma_20"] = out.get("SMA_20", out["close"].rolling(20).mean())
    out["sma_120"] = out.get("SMA_120", out["close"].rolling(120).mean())
    out["rsi"] = out.get("RSI_14", np.nan)
    out["macd"] = out.get("MACD", np.nan)
    out["macd_signal"] = out.get("Signal", np.nan)
    out["macd_histogram"] = out.get("MACD_Histogram", np.nan)
    out["stoch_k"] = out.get("Stoch_K_14", np.nan)
    out["stoch_d"] = out.get("Stoch_D_3", np.nan)

    out["bb20_upper"] = out.get("BB_Upper_20_close_2", np.nan)
    out["bb20_lower"] = out.get("BB_Lower_20_close_2", np.nan)

    out["bb4_upper"] = out.get("BB_Upper_4_open_4", np.nan)
    out["bb4_lower"] = out.get("BB_Lower_4_open_4", np.nan)

    # Trend label
    if {"sma_20", "sma_120"}.issubset(out.columns):
        out["trend_long"] = np.where(out["sma_20"] > out["sma_120"], "상승", "하락")

    out["price_change"] = out["close"].pct_change() * 100
    out["trendline"] = out.get("TrendLine", np.nan)

    return out


# --------------------
# Cached data fetcher
# --------------------
@st.cache_data(show_spinner=False, ttl=60)
def get_data_cached(symbol: str,
                    timeframe: str = "일",
                    minute_interval: int = 1,
                    hour_interval: int = 1,
                    market: str = "KRX",
                    realtime_unit: str = "분") -> pd.DataFrame:
    """
    Fetch recent OHLCV data using KIS (if available in st.session_state.kis_instance),
    normalize to OHLCV indexed by datetime, and return.
    """
    kis = st.session_state.get("kis_instance")  # set in main.py
    if kis is None:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    try:
        # Heuristic mapping from UI selections to pykis API
        stock = kis.stock(symbol)

        # Choose granularity
        if timeframe in {"실시간", "분"}:
            interval = max(1, int(minute_interval))
            # 예: 1분봉, 3분봉, 5분봉
            bars = stock.chart(period=interval)
        elif timeframe == "시간":
            interval = max(1, int(hour_interval))
            # 예: 1시간봉, 2시간봉
            bars = stock.chart(period=60*interval)
        elif timeframe == "주":
            bars = stock.chart(period="week")   # 주봉
        elif timeframe == "월":
            bars = stock.chart(period="month")  # 월봉
        else:  # 기본: 일봉
            bars = stock.chart()


        # Convert to DataFrame
        rows = []
        for b in bars:
            dt = getattr(b, "dt", None) or getattr(b, "time", None)
            rows.append({
                "datetime": pd.to_datetime(dt),
                "open": float(getattr(b, "open", 0)),
                "high": float(getattr(b, "high", 0)),
                "low": float(getattr(b, "low", 0)),
                "close": float(getattr(b, "close", 0)),
                "volume": float(getattr(b, "volume", 0)),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = to_datetime_index(df, ["datetime", "date", "time"])
        df = ensure_ohlcv_columns(df)
        return df
    except Exception as e:
        st.warning(f"데이터 로드 실패: {e}")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])