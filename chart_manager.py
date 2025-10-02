import numpy as np
import pandas as pd


class ChartManager:
    """Technical indicator helpers that return a DataFrame (non-destructive)."""

    @staticmethod
    def add_moving_averages(df: pd.DataFrame, windows=[5, 20, 60, 120]) -> pd.DataFrame:
        out = df.copy()
        for w in windows:
            col = f"SMA_{w}"
            if col not in out.columns:
                out[col] = out["close"].rolling(window=w).mean()
        return out

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, standard_col: str = "close", window: int = 20, std_dev: int = 2) -> pd.DataFrame:
        out = df.copy()
        base = out[standard_col]
        sma = base.rolling(window=window).mean()
        std = base.rolling(window=window).std()
        out[f"BB_Upper_{window}_{standard_col}_{std_dev}"] = sma + (std * std_dev)
        out[f"BB_Lower_{window}_{standard_col}_{std_dev}"] = sma - (std * std_dev)
        # legacy aliases
        out["BB_Upper"] = out[f"BB_Upper_{window}_{standard_col}_{std_dev}"]
        out["BB_Lower"] = out[f"BB_Lower_{window}_{standard_col}_{std_dev}"]
        return out

    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        out = df.copy()
        if f"RSI_{14}" in out.columns and window == 14:
            return out
        delta = out["close"].diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.nan)
        out[f"RSI_{window}"] = 100 - (100 / (1 + rs))
        return out

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        out = df.copy()
        if "MACD" in out.columns:
            return out
        ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
        out["MACD"] = ema_fast - ema_slow
        out["Signal"] = out["MACD"].ewm(span=signal, adjust=False).mean()
        out["MACD_Histogram"] = out["MACD"] - out["Signal"]
        return out

    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        out = df.copy()
        if f"Stoch_K_{k_window}" in out.columns:
            return out
        low_min = out["low"].rolling(window=k_window).min()
        high_max = out["high"].rolling(window=k_window).max()
        out[f"Stoch_K_{k_window}"] = 100 * (out["close"] - low_min) / (high_max - low_min)
        out[f"Stoch_D_{d_window}"] = out[f"Stoch_K_{k_window}"].rolling(window=d_window).mean()
        return out

    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Volume_SMA_20"] = out["volume"].rolling(window=20).mean()
        out["Volume_Ratio"] = out["volume"] / out["Volume_SMA_20"]
        return out