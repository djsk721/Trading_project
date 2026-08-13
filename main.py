import time
import warnings
import streamlit as st

from kis_api import init_pykis, get_stock_quote
from data_processing import get_data_cached
from ui_component import (
    render_sidebar, render_stock_info_header,
    render_chart_tab, render_trading_tab, render_welcome_screen
)

warnings.filterwarnings("ignore")


def _init_session():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    st.session_state.setdefault("current_chat", None)
    st.session_state.setdefault("drawing_lines", [])
    st.session_state.setdefault("initialized", True)


def main():
    st.set_page_config(
        page_title="TradingView 스타일 주식 분석 시스템",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    _init_session()

    # --- KIS API 연결 ---
    kis_instance = init_pykis()
    st.session_state.kis_instance = kis_instance

    # --- 사이드바 (심볼, 지표, 주기 설정) ---
    settings = render_sidebar(kis_instance)

    # --- 본문 레이아웃 ---
    st.title("📊 TradingView 스타일 주식 차트")

    col_chart, col_trade = st.columns([3, 1])   # 3:1 비율 (TradingView와 유사)

    with col_chart:
        # 데이터 가져오기
        df = get_data_cached(
            settings["symbol"], settings["timeframe"],
            settings["minute_interval"], settings["hour_interval"],
            settings["market"], settings["realtime_unit"]
        ) if settings["use_kis_data"] and kis_instance else None

        if df is not None and not df.empty:
            # 실시간 시세
            if "last_quote_update" not in st.session_state:
                st.session_state.last_quote_update = 0
                st.session_state.cached_quote = None
            if time.time() - st.session_state.last_quote_update > 30:
                st.session_state.cached_quote = get_stock_quote(kis_instance, settings["symbol"])
                st.session_state.last_quote_update = time.time()
            quote_info = st.session_state.cached_quote or {}

            # 상단 가격 박스 + 차트
            render_stock_info_header(df, quote_info, settings["symbol"], settings["market"])
            render_chart_tab(df, settings["symbol"], settings["market"], settings)

        else:
            if not kis_instance:
                render_welcome_screen()
            else:
                st.warning("⚠️ 데이터를 불러올 수 없습니다. 종목코드와 시장을 확인해주세요.")

    with col_trade:
        st.subheader("🏦 거래 패널")
        if kis_instance and settings["symbol"]:
            render_trading_tab(kis_instance, settings["symbol"], get_stock_quote(kis_instance, settings["symbol"]))
        else:
            st.info("⚠️ KIS API 연결 필요")


if __name__ == "__main__":
    main()
