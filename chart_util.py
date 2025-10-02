from __future__ import annotations
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

from data_processing import calculate_enhanced_indicators

# intraday 구간 식별
INTRADAY_KEYS = {"실시간", "분", "시간", "Realtime", "Minute", "Hour"}


def format_timestamp(timestamp, timeframe: str) -> str | int:
    """분/시간봉 → timestamp, 일/주/월봉 → YYYY-MM-DD"""
    if hasattr(timestamp, "tz") and timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return int(timestamp.timestamp()) if timeframe in INTRADAY_KEYS else timestamp.strftime("%Y-%m-%d")


def _compute_y_range(candles: list[dict]) -> tuple[float, float]:
    """캔들스틱 데이터에서 y축 범위(min/max) 계산"""
    if not candles:
        return (0.0, 1.0)
    lows = [c.get("low", c.get("value", float("inf"))) for c in candles]
    highs = [c.get("high", c.get("value", float("-inf"))) for c in candles]
    y_min, y_max = float(min(lows)), float(max(highs))
    if not (y_max > y_min):
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.02
    return (y_min - pad, y_max + pad)


def _line_series(name: str, data: List[dict], color: str, width: int = 2) -> dict:
    """기본 라인 시리즈 생성"""
    return {
        "type": "Line",
        "data": data,
        "options": {"color": color, "lineWidth": width, "title": name},
    }


def render_lightweight_chart(df: pd.DataFrame, title: str, indicators: Dict, timeframe: str = "일") -> None:
    """캔들스틱 + 지표 차트 전체 렌더링"""
    if df.empty:
        st.warning("차트 데이터가 없습니다.")
        return

    df_chart = calculate_enhanced_indicators(df)

    # --- 캔들스틱 데이터 준비 ---
    candles = []
    for ts, row in df_chart.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        h = max(h, o, c)
        l = min(l, o, c)
        candles.append({
            "time": format_timestamp(ts, timeframe),
            "open": o, "high": h, "low": l, "close": c
        })
    candles.sort(key=lambda x: x["time"])
    y_min, y_max = _compute_y_range(candles)

    # --- 공통 옵션 ---
    dark = {
        "background": "#1e1e1e",
        "text": "#d4d4d4",
        "grid": "#2d2d30",
        "border": "#3e3e42",
    }
    common_time_scale = {
        "borderColor": dark["border"],
        "timeVisible": timeframe in INTRADAY_KEYS,
        "secondsVisible": timeframe in {"실시간", "Realtime"},
        "rightOffset": 10,
        "barSpacing": 6,
        "fixLeftEdge": True,
        "lockVisibleTimeRangeOnResize": False,
    }

    # --- 메인 캔들 차트 ---
    main_chart_options = {
        "layout": {"textColor": dark["text"], "background": {"type": "solid", "color": dark["background"]}},
        "grid": {"vertLines": {"color": dark["grid"]}, "horzLines": {"color": dark["grid"]}},
        "timeScale": common_time_scale,
        "rightPriceScale": {
            "borderColor": dark["border"],
            "minValue": y_min,
            "maxValue": y_max,
            "scaleMargins": {"top": 0.2, "bottom": 0.1},
        },
        "height": 400,
    }

    main_series = [{
        "type": "Candlestick",
        "data": candles,
        "options": {
            "upColor": "#26a69a", "downColor": "#ef5350",
            "borderUpColor": "#26a69a", "borderDownColor": "#ef5350",
            "wickUpColor": "#26a69a", "wickDownColor": "#ef5350",
        },
    }]
    charts_config = [{"chart": main_chart_options, "series": main_series}]

    # --- 볼륨 차트 ---
    volume = []
    for ts, row in df_chart.iterrows():
        t = format_timestamp(ts, timeframe)
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        volume.append({"time": t, "value": float(row["volume"]), "color": color})
    if volume:
        charts_config.append({
            "chart": {
                "layout": {"textColor": dark["text"], "background": {"type": "solid", "color": dark["background"]}},
                "grid": {"vertLines": {"color": dark["grid"]}, "horzLines": {"color": dark["grid"]}},
                "timeScale": common_time_scale,
                "rightPriceScale": {"borderColor": dark["border"], "scaleMargins": {"top": 0.15, "bottom": 0.15}},
                "height": 120,
            },
            "series": [{"type": "Histogram", "data": volume, "options": {"priceFormat": {"type": "volume"}}}],
        })
        
    # --- SMA(20, 120) 오버레이 ---
    if indicators.get("sma", True):
        if "sma_20" in df_chart.columns:
            sma20 = [{"time": format_timestamp(ts, timeframe), "value": float(v) if pd.notna(v) else None}
                     for ts, v in df_chart["sma_20"].items()]
            main_series.append(_line_series("SMA20", sma20, "#22c55e", 2))  # 초록색

        if "sma_120" in df_chart.columns:
            sma120 = [{"time": format_timestamp(ts, timeframe), "value": float(v) if pd.notna(v) else None}
                      for ts, v in df_chart["sma_120"].items()]
            main_series.append(_line_series("SMA120", sma120, "#e11d48", 2))  # 빨간색
            
    # --- 볼린저 밴드 (20,2) 오버레이 ---
    if indicators.get("bb20", True) and {"bb20_upper", "bb20_lower"}.issubset(df_chart.columns):
        bb20_upper = [{"time": format_timestamp(ts, timeframe), "value": float(v) if pd.notna(v) else None}
                      for ts, v in df_chart["bb20_upper"].items()]
        bb20_lower = [{"time": format_timestamp(ts, timeframe), "value": float(v) if pd.notna(v) else None}
                      for ts, v in df_chart["bb20_lower"].items()]

        main_series += [
            _line_series("BB20 Upper", bb20_upper, "#fbbf24", 1),
            _line_series("BB20 Lower", bb20_lower, "#fbbf24", 1),
        ]

    # --- 볼린저 밴드 (4,4) 오버레이 ---
    if indicators.get("bb4", True) and {"bb4_upper", "bb4_lower"}.issubset(df_chart.columns):
        bb4_upper = [{"time": format_timestamp(ts, timeframe), "value": float(v) if pd.notna(v) else None}
                     for ts, v in df_chart["bb4_upper"].items()]
        bb4_lower = [{"time": format_timestamp(ts, timeframe), "value": float(v) if pd.notna(v) else None}
                     for ts, v in df_chart["bb4_lower"].items()]

        main_series += [
            _line_series("BB4 Upper", bb4_upper, "#a78bfa", 1),
            _line_series("BB4 Lower", bb4_lower, "#a78bfa", 1),
        ]
        
    # --- RSI 차트 ---
    if indicators.get("rsi", True) and "rsi" in df_chart.columns:
        rsi_data = [{"time": format_timestamp(ts, timeframe), "value": float(v) if pd.notna(v) else None}
                    for ts, v in df_chart["rsi"].items()]
        rsi_series = [_line_series("RSI", rsi_data, "#fbbf24", 2)]
        for level, color in [(70, "#ef4444"), (50, "#6b7280"), (30, "#3b82f6")]:
            ref = [{"time": d["time"], "value": level} for d in rsi_data]
            rsi_series.append(_line_series(f"RSI {level}", ref, color, 1) | {
                "options": {"lineStyle": 2, "priceLineVisible": False, "lastValueVisible": False}
            })
        charts_config.append({
            "chart": {
                "layout": {"textColor": dark["text"], "background": {"type": "solid", "color": dark["background"]}},
                "grid": {"vertLines": {"color": dark["grid"]}, "horzLines": {"color": dark["grid"]}},
                "timeScale": common_time_scale,
                "rightPriceScale": {"borderColor": dark["border"], "scaleMargins": {"top": 0.1, "bottom": 0.1}},
                "height": 120,
            },
            "series": rsi_series,
        })

    # --- Stochastic 차트 ---
    if indicators.get("stochastic", True) and {"stoch_k", "stoch_d"}.issubset(df_chart.columns):
        stoch_k = [{"time": format_timestamp(ts, timeframe), "value": float(v) if pd.notna(v) else None}
                   for ts, v in df_chart["stoch_k"].items()]
        stoch_d = [{"time": format_timestamp(ts, timeframe), "value": float(v) if pd.notna(v) else None}
                   for ts, v in df_chart["stoch_d"].items()]
        stoch_series = [
            _line_series("%K", stoch_k, "#10b981", 2),
            _line_series("%D", stoch_d, "#0ea5e9", 2),
        ]
        charts_config.append({
            "chart": {
                "layout": {"textColor": dark["text"], "background": {"type": "solid", "color": dark["background"]}},
                "grid": {"vertLines": {"color": dark["grid"]}, "horzLines": {"color": dark["grid"]}},
                "timeScale": common_time_scale,
                "rightPriceScale": {"borderColor": dark["border"], "scaleMargins": {"top": 0.1, "bottom": 0.1}},
                "height": 120,
            },
            "series": stoch_series,
        })

    # --- MACD 차트 ---
    if indicators.get("macd", True) and {"macd", "macd_signal", "macd_histogram"}.issubset(df_chart.columns):
        macd, macd_signal, macd_hist = [], [], []
        for ts, row in df_chart.iterrows():
            t = format_timestamp(ts, timeframe)
            macd.append({"time": t, "value": float(row["macd"]) if pd.notna(row["macd"]) else None})
            macd_signal.append({"time": t, "value": float(row["macd_signal"]) if pd.notna(row["macd_signal"]) else None})
            macd_hist.append({
                "time": t,
                "value": float(row["macd_histogram"]) if pd.notna(row["macd_histogram"]) else None,
                "color": "#26a69a" if (pd.notna(row["macd_histogram"]) and row["macd_histogram"] >= 0) else "#ef5350"
            })
        macd_series = [
            _line_series("MACD", macd, "#14b8a6", 2),
            _line_series("Signal", macd_signal, "#8b5cf6", 2),
            {"type": "Histogram", "data": macd_hist},
        ]
        charts_config.append({
            "chart": {
                "layout": {"textColor": dark["text"], "background": {"type": "solid", "color": dark["background"]}},
                "grid": {"vertLines": {"color": dark["grid"]}, "horzLines": {"color": dark["grid"]}},
                "timeScale": common_time_scale,
                "rightPriceScale": {"borderColor": dark["border"], "scaleMargins": {"top": 0.1, "bottom": 0.1}},
                "height": 120,
            },
            "series": macd_series,
        })

    # --- TrendLine 차트 ---
    if indicators.get("trendline", True) and "TrendLine" in df_chart.columns:
        trendline_data = [{"time": format_timestamp(ts, timeframe), "value": float(v) if pd.notna(v) else None}
                          for ts, v in df_chart["TrendLine"].items()]
        buy_markers = [{"time": format_timestamp(ts, timeframe), "position": "belowBar",
                        "color": "#26a69a", "shape": "arrowUp", "text": "Buy"}
                       for ts, flag in df_chart["TrendBuy"].items() if flag]
        sell_markers = [{"time": format_timestamp(ts, timeframe), "position": "aboveBar",
                         "color": "#ef5350", "shape": "arrowDown", "text": "Sell"}
                        for ts, flag in df_chart["TrendSell"].items() if flag]

        trendline_series = [_line_series("TrendLine", trendline_data, "#3b82f6", 2)]
        charts_config.append({
            "chart": {
                "layout": {"textColor": dark["text"], "background": {"type": "solid", "color": dark["background"]}},
                "grid": {"vertLines": {"color": dark["grid"]}, "horzLines": {"color": dark["grid"]}},
                "timeScale": common_time_scale,
                "rightPriceScale": {"borderColor": dark["border"], "scaleMargins": {"top": 0.1, "bottom": 0.1}},
                "height": 120,
            },
            "series": trendline_series,
            "markers": buy_markers + sell_markers,
        })

    # --- 최종 렌더링 ---
    renderLightweightCharts(charts_config)
