"""기술적 지표 계산 (Streamlit 의존성 없음)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    rename_candidates = [
        {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"},
        {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"},
    ]
    out = df.copy()
    for mp in rename_candidates:
        if set(mp.keys()) & set(out.columns):
            out = out.rename(columns=mp)
            break
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = out["close"] if col != "volume" and "close" in out.columns else 0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["close"]).sort_index()
    return out


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = ensure_ohlcv(df)

    for w in (5, 10, 20, 60, 120):
        out[f"SMA_{w}"] = out["close"].rolling(w, min_periods=1).mean()
        out[f"EMA_{w}"] = out["close"].ewm(span=w, adjust=False).mean()

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Histogram"] = out["MACD"] - out["Signal"]
    out["MACD_Signal"] = np.where(out["MACD"] > out["Signal"], "BUY", "SELL")
    out["MACD_Cross"] = (out["MACD"] > out["Signal"]) != (
        out["MACD"].shift() > out["Signal"].shift()
    )

    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    # Wilder RSI: 평균손실=0이면 연속 상승 → 100 (NaN을 50으로 채우면 안 됨)
    rsi = pd.Series(np.nan, index=out.index, dtype=float)
    has_loss = loss > 0
    rsi.loc[has_loss] = 100 - (100 / (1 + gain.loc[has_loss] / loss.loc[has_loss]))
    rsi.loc[(loss == 0) & (gain > 0)] = 100.0
    rsi.loc[(loss == 0) & (gain == 0)] = 50.0
    out["RSI_14"] = rsi.fillna(50)
    out["RSI_Signal"] = np.where(
        out["RSI_14"] > 70, "OVERBOUGHT",
        np.where(out["RSI_14"] < 30, "OVERSOLD", "NEUTRAL"),
    )

    mid = out["close"].rolling(20, min_periods=1).mean()
    std = out["close"].rolling(20, min_periods=1).std().fillna(0)
    out["BB_Middle"] = mid
    out["BB_Upper"] = mid + 2 * std
    out["BB_Lower"] = mid - 2 * std
    band = (out["BB_Upper"] - out["BB_Lower"]).replace(0, np.nan)
    out["BB_Position"] = ((out["close"] - out["BB_Lower"]) / band).fillna(0.5)
    out["BB_Signal"] = np.where(
        out["close"] > out["BB_Upper"], "UPPER_BREAK",
        np.where(out["close"] < out["BB_Lower"], "LOWER_BREAK", "INSIDE"),
    )

    low_min = out["low"].rolling(14, min_periods=1).min()
    high_max = out["high"].rolling(14, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    out["%K"] = (100 * (out["close"] - low_min) / denom).fillna(50)
    out["%D"] = out["%K"].rolling(3, min_periods=1).mean()
    out["Stoch_K_14"] = out["%K"]
    out["Stoch_D_3"] = out["%D"]

    out["Volume_SMA_20"] = out["volume"].rolling(20, min_periods=1).mean()
    out["Volume_Ratio"] = (out["volume"] / out["Volume_SMA_20"].replace(0, np.nan)).fillna(1)
    out["Volume_Signal"] = np.where(
        out["Volume_Ratio"] >= 2, "HIGH_VOLUME",
        np.where(out["Volume_Ratio"] <= 0.5, "LOW_VOLUME", "NORMAL"),
    )

    out["Price_Change"] = out["close"].pct_change() * 100
    out["Volatility"] = out["Price_Change"].rolling(20, min_periods=1).std().fillna(0)
    out["High_Low_Ratio"] = ((out["high"] - out["low"]) / out["close"].replace(0, np.nan) * 100).fillna(0)
    out["Trend_5"] = np.where(out["close"] > out["SMA_5"], "UP", "DOWN")
    out["Trend_20"] = np.where(out["close"] > out["SMA_20"], "UP", "DOWN")
    out["trend_long"] = np.where(out["SMA_20"] > out["SMA_120"], "UP", "DOWN")

    # aliases for chart frontend
    out["sma_20"] = out["SMA_20"]
    out["sma_60"] = out["SMA_60"]
    out["sma_120"] = out["SMA_120"]
    out["ema_20"] = out["EMA_20"]
    out["rsi"] = out["RSI_14"]
    out["macd"] = out["MACD"]
    out["macd_signal"] = out["Signal"]
    out["macd_histogram"] = out["MACD_Histogram"]
    out["stoch_k"] = out["Stoch_K_14"]
    out["stoch_d"] = out["Stoch_D_3"]
    out["bb20_upper"] = out["BB_Upper"]
    out["bb20_middle"] = out["BB_Middle"]
    out["bb20_lower"] = out["BB_Lower"]
    out["volume"] = out["volume"]
    out["price_change"] = out["Price_Change"]

    # Korean column aliases for RAG documents
    out["시가"] = out["open"]
    out["고가"] = out["high"]
    out["저가"] = out["low"]
    out["종가"] = out["close"]
    out["거래량"] = out["volume"]

    return out.ffill().fillna(0)


def latest_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    row = df.iloc[-1]
    rsi = float(row.get("RSI_14", 50))
    macd = float(row.get("MACD", 0))
    signal = float(row.get("Signal", 0))
    k = float(row.get("%K", 50))
    return {
        "price": float(row.get("close", 0)),
        "rsi": rsi,
        "rsi_label": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
        "macd": macd,
        "macd_signal": "bullish" if macd > signal else "bearish",
        "stoch_k": k,
        "stoch_label": "overbought" if k > 80 else "oversold" if k < 20 else "neutral",
        "trend_long": str(row.get("trend_long", "SIDEWAYS")),
        "volume_ratio": float(row.get("Volume_Ratio", 1)),
        "price_change": float(row.get("Price_Change", 0)),
    }


def series_to_list(df: pd.DataFrame, columns: list[str]) -> Dict[str, list]:
    result: Dict[str, list] = {}
    for col in columns:
        if col not in df.columns:
            continue
        values = []
        for idx, val in df[col].items():
            try:
                t = pd.Timestamp(idx)
                if getattr(t, "tz", None) is not None:
                    t = t.tz_convert("Asia/Seoul").tz_localize(None)
                if t.hour or t.minute or t.second:
                    ts = t.strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    ts = t.strftime("%Y-%m-%d")
            except Exception:
                ts = str(idx)
            try:
                num = float(val)
                if np.isnan(num):
                    continue
                values.append({"time": ts, "value": num})
            except (TypeError, ValueError):
                continue
        result[col] = values
    return result
