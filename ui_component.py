import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from kis_api import get_account_balance, get_account_holdings, get_stock_quote, place_buy_order, place_sell_order, get_pending_orders, cancel_order
from data_processing import calculate_enhanced_indicators, get_data_cached
from chart_util import render_lightweight_chart


def render_welcome_screen():
    st.title("📈 통합 주식 거래 및 분석 시스템")
    st.info("좌측 사이드바에서 KIS API 연결 상태를 확인하시고, 종목을 선택하여 차트를 확인하세요.")
    st.caption("secret.json 설정 후 새로고침이 필요할 수 있습니다.")


def _refresh_cached_balance(kis_instance):
    st.session_state.cached_balance = get_account_balance(kis_instance)
    st.session_state.last_balance_update = time.time()


def _refresh_cached_holdings(kis_instance):
    st.session_state.cached_holdings = get_account_holdings(kis_instance)


def _render_indicator_settings() -> Dict[str, bool]:
    st.subheader("🔢 기술적 지표")
    return {
        "sma": st.checkbox("📈 이동평균선 (20, 120)", value=True),
        "bollinger": st.checkbox("📊 볼린저밴드 (20,2 & 4,4)", value=True),
        "rsi": st.checkbox("⚡ RSI (14)", value=True),
        "stochastic": st.checkbox("🎯 스토캐스틱", value=True),
        "macd": st.checkbox("📉 MACD", value=True),
        "trendline": st.checkbox("🔄 DMI Trendline", value=True),
    }


def render_sidebar(kis_instance):
    """Render the integrated sidebar and return settings dict."""
    with st.sidebar:
        st.title("📈 통합 거래 시스템")

        # --- Account ---
        if kis_instance:
            st.header("💰 계좌 잔고")
            if "last_balance_update" not in st.session_state:
                st.session_state.last_balance_update = 0
                st.session_state.cached_balance = None

            if time.time() - st.session_state.last_balance_update > 300:
                _refresh_cached_balance(kis_instance)

            balance = st.session_state.cached_balance or {"total_amount": 0, "deposit": 0, "stocks_value": 0, "profit_loss": 0, "profit_loss_rate": 0.0, "market_value": 0}
            st.metric("총 평가금액", f"{balance['total_amount']:,} 원")
            cols = st.columns(2)
            with cols[0]:
                st.metric("보유주식", f"{balance['stocks_value']:,} 원")
                st.metric("예수금", f"{balance['deposit']:,} 원")
            with cols[1]:
                pl = balance.get("profit_loss", 0)
                pr = balance.get("profit_loss_rate", 0.0)
                st.metric("평가손익", f"{pl:+,} 원", f"{pr:+.2f}%")

            if st.button("🔄 잔고 새로고침", width='content'):
                _refresh_cached_balance(kis_instance)
                _refresh_cached_holdings(kis_instance)
                st.rerun()

            # --- Holdings ---
            st.subheader("📊 보유종목")
            if "cached_holdings" not in st.session_state:
                _refresh_cached_holdings(kis_instance)
            holdings = st.session_state.get("cached_holdings", [])
            if holdings:
                st.caption("클릭하여 차트 보기")
                for i, h in enumerate(holdings):
                    if h.get("qty", 0) <= 0:
                        continue
                    name, sym, prate = h.get("name", ""), h.get("symbol", ""), h.get("profit_rate", 0.0)
                    emoji = "🔴" if prate > 0 else "🔵" if prate < 0 else "⚪"
                    label = f"{emoji} {name} ({sym})  |  {h.get('qty', 0):,}주  |  {prate:+.2f}%"
                    if st.button(label, key=f"holding_{i}_{sym}", width='content'):
                        st.session_state.selected_holding = {"symbol": sym, "name": name, "market": "KRX" if sym.isdigit() else "NASDAQ"}
                        st.success(f"✅ {name}({sym}) 차트로 이동합니다!")
                        st.rerun()
            else:
                st.info("보유종목이 없습니다.")
            st.divider()
        else:
            st.warning("⚠️ KIS API 연결이 필요합니다. secret.json 파일을 확인해주세요.")
            st.divider()

        # --- Symbol settings ---
        st.header("📊 종목 설정")
        default_symbol, default_market = "005930", "KRX"
        if "selected_holding" in st.session_state:
            sel = st.session_state.selected_holding
            default_symbol, default_market = sel.get("symbol", default_symbol), sel.get("market", default_market)
            st.success(f"📊 선택된 종목: {sel.get('name', '')} ({default_symbol})")
            if st.button("❌ 종목 선택 해제", width='content'):
                del st.session_state.selected_holding
                st.rerun()

        symbol = st.text_input("종목코드", value=default_symbol, help="예: 005930 (삼성전자)")
        market = st.selectbox("시장", ["KRX", "NASDAQ", "NYSE", "AMEX"], index=["KRX", "NASDAQ", "NYSE", "AMEX"].index(default_market))

        use_kis_data = kis_instance is not None

        # Timeframe
        if use_kis_data:
            st.subheader("📈 차트 설정")
            timeframe = st.radio("시간 단위", options=["실시간", "분", "시간", "일", "주", "월"], index=3)
            minute_interval = 1
            hour_interval = 1
            realtime_unit = "분"
            if timeframe == "실시간":
                realtime_unit = st.selectbox("실시간 차트 단위", ["분", "시간", "일", "월"], index=0)
                if realtime_unit == "분":
                    minute_interval = st.selectbox("분 간격", [1, 3, 5, 10, 15, 30, 60], index=0)
                elif realtime_unit == "시간":
                    hour_interval = st.selectbox("시간 간격", [1, 2, 4, 6, 12], index=0)
            elif timeframe == "분":
                minute_interval = st.selectbox("분 간격", [1, 3, 5, 10, 15, 30, 60], index=2)
            elif timeframe == "시간":
                hour_interval = st.selectbox("시간 간격", [1, 2, 4, 6, 12], index=0)

            indicators = _render_indicator_settings()

            st.subheader("🔄 새로고침")
            enable_refresh = st.checkbox("자동 새로고침", value=True)
            if enable_refresh:
                refresh_sec = st.slider("주기 (초)", min_value=10, max_value=300, value=60, step=10)
                st.caption(f"🔄 매 {refresh_sec}초마다 새로고침")
                try:
                    from streamlit_autorefresh import st_autorefresh  # optional
                    st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh")
                except Exception:
                    pass
            else:
                if st.button("🔄 수동 새로고침"):
                    st.rerun()
        else:
            timeframe, minute_interval, hour_interval, realtime_unit = "일", 1, 1, "분"
            indicators = {"sma": True, "bollinger": True, "rsi": True, "stochastic": True, "macd": True, "trendline": True}

    return {
        "symbol": symbol,
        "market": market,
        "use_kis_data": use_kis_data,
        "timeframe": timeframe,
        "minute_interval": int(minute_interval),
        "hour_interval": int(hour_interval),
        "realtime_unit": realtime_unit,
        "indicators": indicators,
    }


def render_stock_info_header(df: pd.DataFrame, quote_info: Dict[str, Any], symbol: str, market: str):
    if df.empty:
        return
    curr = float(df["close"].iloc[-1])
    currency_symbol = "원" if market == "KRX" else "$"
    fmt = "{:,.0f}" if market == "KRX" else "{:,.2f}"
    cols = st.columns(4)
    with cols[0]:
        st.metric("현재가", fmt.format(curr) + currency_symbol)
    with cols[1]:
        if quote_info and quote_info.get("change") is not None:
            chg = quote_info.get("change", 0.0)
            rate = quote_info.get("rate", 0.0)
            chg_str = (f"{chg:+,.0f}" if market == "KRX" else f"{chg:+,.2f}") + currency_symbol
            st.metric("전일대비", chg_str, f"{rate:+.2f}%")
        else:
            st.metric("전일대비", "-", "-")
    with cols[2]:
        st.metric("거래량", f"{df['volume'].iloc[-1]:,.0f}주")
    with cols[3]:
        st.metric("데이터 수", f"{len(df):,}개")


def render_chart_tab(df: pd.DataFrame, symbol: str, market: str, settings: Dict[str, Any]):
    st.subheader("📈 통합 차트 분석")
    if df.empty:
        st.warning("차트 데이터가 없습니다.")
        return
    title_suffix = f"{settings['timeframe']}"
    plot_title = f"{symbol} ({market}) - {title_suffix} [KIS API]" if settings["use_kis_data"] else f"{symbol} ({market}) - {title_suffix}"
    render_lightweight_chart(df, plot_title, settings["indicators"], settings["timeframe"])

    st.subheader("📊 기술적 분석 요약")
    ind = calculate_enhanced_indicators(df)
    latest = ind.iloc[-1]
    cols = st.columns(4)
    with cols[0]:
        rsi = float(latest.get("rsi", np.nan))
        st.metric("RSI(14)", f"{rsi:.1f}")
        st.caption("신호: " + ("과매수" if rsi > 70 else "과매도" if rsi < 30 else "중립"))
    with cols[1]:
        macd, sig = float(latest.get("macd", np.nan)), float(latest.get("macd_signal", np.nan))
        st.metric("MACD", f"{macd:.3f}")
        st.caption("추세: " + ("상승" if macd > sig else "하락"))
    with cols[2]:
        k = float(latest.get("stoch_k", np.nan))
        st.metric("스토캐스틱 %K", f"{k:.1f}")
        st.caption("신호: " + ("과매수" if k > 80 else "과매도" if k < 20 else "중립"))
    with cols[3]:
        disp = float(latest.get("displacement_20", np.nan))
        st.metric("이격도", f"{disp:.1f}%")
        st.caption("위치: " + ("고점권" if disp > 105 else "저점권" if disp < 95 else "중립"))

    with st.expander("📋 상세 데이터 미리보기"):
        display_cols = ["open", "high", "low", "close", "volume", "sma_20", "sma_120", "rsi", "macd", "stoch_k"]
        st.dataframe(ind[display_cols].round(3), width='content')


def _render_order_tab(kis_instance, symbol: str, quote_info: Dict[str, Any]):
    st.warning("⚠️ 실제 주문이 체결될 수 있습니다. 주문 전 반드시 확인하세요!")
    balance = st.session_state.get("cached_balance", {})
    holdings = st.session_state.get("cached_holdings", [])
    current_holding = next((h for h in holdings if h.get("symbol") == symbol), None)

    cols = st.columns(3)
    with cols[0]:
        deposit = int(balance.get("deposit", 0))
        st.metric("💰 예수금", f"{deposit:,}원")
    with cols[1]:
        qty = int(current_holding.get("qty", 0)) if current_holding else 0
        st.metric("📊 보유주식", f"{qty:,}주")
    with cols[2]:
        current_price = float(quote_info.get("price", 0) if quote_info else 0)
        st.metric("🔢 매수가능", f"{(deposit // current_price) if current_price > 0 else 0:,}주")

    st.markdown("---")
    side = st.radio("주문 유형", ["매수", "매도"], horizontal=True)
    price_type = st.radio("가격", ["시장가", "지정가"], horizontal=True)
    qty = st.number_input("수량", min_value=1, value=1, step=1)
    price = None
    if price_type == "지정가":
        price = st.number_input("지정가", min_value=1, value=int(quote_info.get("price", 0) or 0), step=1)

    if st.button("주문 접수", type="primary", width='content'):
        if side == "매수":
            res = place_buy_order(kis_instance, symbol, qty, price=None if price_type == "시장가" else int(price))
        else:
            res = place_sell_order(kis_instance, symbol, qty, price=None if price_type == "시장가" else int(price))
        if res.get("success"):
            st.success(res.get("message", "주문 접수"))
        else:
            st.error(res.get("message", "주문 실패"))


def _render_pending_orders_tab(kis_instance):
    orders = get_pending_orders(kis_instance)
    if not orders:
        st.info("미체결 주문이 없습니다.")
        return
    st.dataframe(pd.DataFrame(orders), width='content')


def _render_holdings_tab(kis_instance):
    holdings = get_account_holdings(kis_instance)
    if not holdings:
        st.info("보유종목이 없습니다.")
        return
    df = pd.DataFrame(holdings)
    st.dataframe(df, width='content')


def render_trading_tab(kis_instance, symbol: str, quote_info: Dict[str, Any]):
    st.subheader("🏦 실시간 주식 거래")
    if not kis_instance:
        st.error("KIS API 연결이 필요합니다. secret.json 파일을 확인해주세요.")
        return
    subtab1, subtab2, subtab3 = st.tabs(["📈 매수/매도", "📋 미체결 주문", "📊 보유종목"])
    with subtab1:
        _render_order_tab(kis_instance, symbol, quote_info)
    with subtab2:
        _render_pending_orders_tab(kis_instance)
    with subtab3:
        _render_holdings_tab(kis_instance)