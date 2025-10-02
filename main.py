import time
import warnings

import streamlit as st

from kis_api import init_pykis, get_stock_quote
from data_processing import get_data_cached
from ui_component import render_sidebar, render_stock_info_header, render_chart_tab, render_trading_tab, render_welcome_screen

warnings.filterwarnings("ignore")


def _init_session():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    st.session_state.setdefault("current_chat", None)
    st.session_state.setdefault("drawing_lines", [])
    st.session_state.setdefault("initialized", True)


def render_main_content(kis_instance, settings):
    st.title("📈 통합 주식 거래 및 분석 시스템")
    st.markdown("**KIS API 실시간 데이터 + AI 분석 + 고급 기술적 지표**" if settings["use_kis_data"] else "**AI와 기술적 분석을 결합한 지능형 주식 분석 플랫폼**")

    # Quote cache (30s)
    quote_info = {}
    if kis_instance and settings["symbol"] and settings["use_kis_data"]:
        if "last_quote_update" not in st.session_state:
            st.session_state.last_quote_update = 0
            st.session_state.cached_quote = None
        if time.time() - st.session_state.last_quote_update > 30:
            st.session_state.cached_quote = get_stock_quote(kis_instance, settings["symbol"])
            st.session_state.last_quote_update = time.time()
        quote_info = st.session_state.cached_quote or {}

    # Load data via cached fetcher
    df = get_data_cached(
        settings["symbol"],
        settings["timeframe"],
        int(settings["minute_interval"]),
        int(settings["hour_interval"]),
        settings["market"],
        settings["realtime_unit"],
    ) if settings["use_kis_data"] and kis_instance else None

    if df is not None and not df.empty:
        render_stock_info_header(df, quote_info, settings["symbol"], settings["market"])
        tab1, tab2, tab3 = st.tabs(["📈 차트 분석", "🏦 실시간 거래", "🤖 AI 분석"])
        with tab1:
            render_chart_tab(df, settings["symbol"], settings["market"], settings)
        with tab2:
            render_trading_tab(kis_instance, settings["symbol"], quote_info)
        with tab3:
            st.info("AI 분석 기능은 추후 모듈 연동 예정입니다.")
    else:
        if not kis_instance:
            render_welcome_screen()
        else:
            st.warning("⚠️ KIS API에서 데이터를 불러올 수 없습니다. 종목코드와 시장을 확인해주세요.")


def main():
    st.set_page_config(page_title="통합 주식 거래 및 분석 시스템", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    _init_session()

    kis_instance = init_pykis()
    st.session_state.kis_instance = kis_instance

    settings = render_sidebar(kis_instance)
    render_main_content(kis_instance, settings)


if __name__ == "__main__":
    main()