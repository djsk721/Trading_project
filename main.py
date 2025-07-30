"""
RAG 기반 주식 분석 시스템 - 메인 애플리케이션

리팩토링된 모듈형 구조로 개선된 주식 분석 시스템
- 설정 기반 관리
- 모듈별 책임 분리
- 확장 가능한 구조
"""

import streamlit as st
import warnings
from datetime import datetime

# 모듈 임포트
from config.settings import AppConfig
from models.stock_data import StockDataManager
from models.documents import DocumentGenerator
from services.chat_manager import ChatManager
from services.query_processor import QueryProcessor
from ui.chart_manager import ChartManager

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(**AppConfig.PAGE_CONFIG)


def initialize_app():
    """애플리케이션 초기화"""
    # 세션 상태 초기화
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
    
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = None
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True


def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.title("📈 RAG 주식 분석")
        
        # 채팅 관리 섹션
        st.subheader("채팅 관리")
        
        # 새 채팅 버튼
        if st.button("🆕 새 분석 시작", use_container_width=True):
            new_chat = ChatManager.add_new_chat()
            st.session_state.current_chat = new_chat
            st.rerun()
        
        # 채팅 목록
        chat_list = ChatManager.get_chat_list()
        if chat_list:
            current_chat = st.session_state.get('current_chat')
            if current_chat not in chat_list and chat_list:
                st.session_state.current_chat = chat_list[0]
                current_chat = chat_list[0]
            
            selected_chat = st.selectbox(
                "기존 분석 선택",
                options=chat_list,
                index=chat_list.index(current_chat) if current_chat in chat_list else 0
            )
            
            if selected_chat != current_chat:
                st.session_state.current_chat = selected_chat
                st.rerun()
            
            # 채팅 삭제 버튼
            if st.button("🗑️ 선택된 분석 삭제", use_container_width=True):
                if ChatManager.delete_chat(selected_chat):
                    st.success("분석이 삭제되었습니다.")
                    st.rerun()
        
        st.divider()
        
        # 주식 설정 섹션
        st.subheader("주식 설정")
        
        # 종목 선택 방식
        input_method = st.radio(
            "종목 선택 방식",
            ["📋 인기 종목", "🔍 전체 종목", "⌨️ 직접 입력"],
            horizontal=True
        )
        
        ticker = None
        stock_name_display = None
        
        if input_method == "📋 인기 종목":
            popular_stocks = StockDataManager.get_popular_stocks()
            selected_stock = st.selectbox(
                "인기 종목 선택",
                options=list(popular_stocks.keys()),
                index=list(popular_stocks.keys()).index("LG디스플레이 (034220)") 
                if "LG디스플레이 (034220)" in popular_stocks else 0
            )
            ticker = popular_stocks[selected_stock]
            stock_name_display = selected_stock
            
        elif input_method == "🔍 전체 종목":
            with st.spinner("전체 종목 리스트를 로드하는 중..."):
                all_stocks = StockDataManager.get_all_stocks()
            
            search_term = st.text_input("종목명 검색", placeholder="예: 삼성, LG, 네이버 등")
            
            if search_term:
                filtered_stocks = StockDataManager.search_stocks(search_term)
                if filtered_stocks:
                    selected_stock = st.selectbox(
                        f"검색 결과 ({len(filtered_stocks)}개)",
                        options=list(filtered_stocks.keys())
                    )
                    ticker = filtered_stocks[selected_stock]
                    stock_name_display = selected_stock
                else:
                    st.warning("검색 결과가 없습니다.")
            else:
                selected_stock = st.selectbox(
                    f"전체 종목 ({len(all_stocks)}개)",
                    options=list(all_stocks.keys()),
                    index=0
                )
                ticker = all_stocks[selected_stock]
                stock_name_display = selected_stock
        
        else:  # 직접 입력
            ticker = st.text_input(
                "종목 코드 입력", 
                value=AppConfig.DATA_CONFIG["DEFAULT_TICKER"],
                help="6자리 종목 코드를 입력하세요"
            )
            if ticker and StockDataManager.validate_ticker(ticker):
                stock_name_display = f"{StockDataManager.get_stock_name(ticker)} ({ticker})"
            elif ticker:
                st.error("유효하지 않은 종목 코드입니다.")
        
        # 선택된 종목 정보
        if ticker and stock_name_display:
            st.info(f"**선택된 종목**: {stock_name_display}")
        
        # 데이터 기간 설정
        config = AppConfig.DATA_CONFIG
        days = st.slider(
            "데이터 기간 (영업일)", 
            min_value=config["MIN_DAYS"], 
            max_value=config["MAX_DAYS"], 
            value=config["DEFAULT_DAYS"]
        )
        
        # 데이터 로드 버튼
        if st.button("📊 데이터 로드", use_container_width=True, type="primary"):
            if ticker:
                with st.spinner("데이터를 로드하는 중..."):
                    stock_data = StockDataManager.get_stock_data(ticker, days)
                    stock_name = StockDataManager.get_stock_name(ticker)
                    
                    if stock_data is not None:
                        st.session_state.stock_data = stock_data
                        st.session_state.stock_name = stock_name
                        st.session_state.ticker = ticker
                        st.success(f"{stock_name} 데이터 로드 완료!")
                        st.rerun()
                    else:
                        st.error("데이터 로드에 실패했습니다.")
            else:
                st.warning("종목을 선택해주세요.")
        
        # 현재 로드된 주식 정보
        if 'stock_data' in st.session_state:
            st.success(f"**현재 종목**: {st.session_state.stock_name}")
            st.info(f"**데이터 기간**: {len(st.session_state.stock_data)}일")


def render_main_content():
    """메인 콘텐츠 렌더링"""
    # 헤더
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📈 RAG 기반 주식 분석 시스템")
        st.markdown("**AI와 기술적 분석을 결합한 지능형 주식 분석 플랫폼**")
    
    with col2:
        if 'stock_data' in st.session_state:
            st.metric(
                "🎯 분석 중인 종목",
                st.session_state.stock_name,
                delta=f"({st.session_state.ticker})"
            )
    
    # 주식 데이터가 없는 경우
    if 'stock_data' not in st.session_state:
        render_welcome_screen()
        return
    
    # 주식 데이터가 있는 경우
    stock_data = st.session_state.stock_data
    stock_name = st.session_state.stock_name
    
    # 현재 주식 정보 표시
    render_stock_summary(stock_data, stock_name)
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📊 차트 분석", "🤖 AI 분석", "📈 기술적 지표"])
    
    with tab1:
        render_chart_tab(stock_data, stock_name)
    
    with tab2:
        render_ai_analysis_tab(stock_data, stock_name)
    
    with tab3:
        render_technical_tab(stock_data, stock_name)


def render_welcome_screen():
    """환영 화면 렌더링"""
    st.markdown("---")
    
    # 시스템 소개
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 AI 분석
        - **RAG 기술** 활용
        - **한국어 특화** LLM
        - **실시간 분석**
        """)
    
    with col2:
        st.markdown("""
        ### 📊 기술적 지표
        - **20+ 고급 지표**
        - **실시간 차트**
        - **신호 분석**
        """)
    
    with col3:
        st.markdown("""
        ### 💼 투자 전략
        - **20일 예측**
        - **리스크 관리**
        - **포트폴리오 제안**
        """)
    
    st.markdown("---")
    st.info("👈 **시작하기**: 사이드바에서 종목을 선택하고 데이터를 로드해주세요.")
    
    # 인기 종목 빠른 로드
    st.subheader("⚡ 빠른 시작 - 인기 종목")
    popular_stocks = list(StockDataManager.get_popular_stocks().items())[:6]
    
    cols = st.columns(3)
    for i, (display_name, ticker) in enumerate(popular_stocks):
        col = cols[i % 3]
        name = display_name.split(' (')[0]
        
        if col.button(f"📈 {name}", key=f"quick_{ticker}", use_container_width=True):
            with st.spinner(f"{name} 데이터 로드 중..."):
                stock_data = StockDataManager.get_stock_data(ticker)
                if stock_data is not None:
                    st.session_state.stock_data = stock_data
                    st.session_state.stock_name = name
                    st.session_state.ticker = ticker
                    st.rerun()


def render_stock_summary(stock_data, stock_name):
    """주식 요약 정보 렌더링"""
    current_price = stock_data['종가'].iloc[-1]
    price_change = stock_data['Price_Change'].iloc[-1]
    current_date = stock_data.index[-1].strftime('%Y-%m-%d')
    
    st.markdown("---")
    st.markdown(f"### 📊 {stock_name} 현황 ({current_date})")
    
    # 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        col1.metric(
            "💰 현재가",
            f"{current_price:,.0f}원",
            f"{price_change:+.2f}%",
            delta_color="inverse" if price_change < 0 else "normal"
        )
    
    with col2:
        latest = stock_data.iloc[-1]
        col2.metric(
            "📈 거래량",
            f"{latest['거래량']:,.0f}주",
            f"{latest['Volume_Ratio']:.1f}배"
        )
    
    with col3:
        col3.metric(
            "📊 RSI(14)",
            f"{latest['RSI_14']:.1f}",
            latest['RSI_Signal']
        )
    
    with col4:
        col4.metric(
            "🎯 추세",
            latest['Trend_20'],
            f"단기: {latest['Trend_5']}"
        )


def render_chart_tab(stock_data, stock_name):
    """차트 탭 렌더링"""
    st.subheader("📊 기술적 분석 차트")
    
    # 차트 옵션
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "차트 유형",
            AppConfig.UI_CONFIG["CHART_TYPES"]
        )
    
    with col2:
        chart_period = st.selectbox(
            "표시 기간",
            AppConfig.UI_CONFIG["CHART_PERIODS"]
        )
    
    # 기간별 데이터 필터링
    period_map = {
        "최근 3개월": 63,
        "최근 1개월": 21,
        "최근 2주": 10
    }
    
    if chart_period in period_map:
        display_data = stock_data.tail(period_map[chart_period])
    else:
        display_data = stock_data
    
    # 차트 생성
    if chart_type == "기술적 지표 대시보드":
        fig = ChartManager.create_technical_dashboard(display_data, stock_name)
    else:
        fig = ChartManager.create_candlestick_chart(display_data, stock_name)
    
    st.plotly_chart(fig, use_container_width=True)


def render_ai_analysis_tab(stock_data, stock_name):
    """AI 분석 탭 렌더링"""
    st.subheader("🤖 AI 기반 주식 분석")
    
    # RAG 시스템 초기화
    if 'query_processor' not in st.session_state:
        with st.spinner("AI 시스템을 초기화하는 중..."):
            # 문서 생성
            doc_generator = DocumentGenerator(stock_data, stock_name)
            documents = doc_generator.generate_documents()
            
            # 쿼리 프로세서 초기화
            query_processor = QueryProcessor()
            llm_loaded = query_processor.initialize_llm()
            
            if llm_loaded:
                retriever_type = query_processor.setup_rag(documents)
                st.session_state.query_processor = query_processor
                st.success(f"AI 시스템 초기화 완료! (검색: {retriever_type})")
            else:
                st.error("LLM 모델 로드에 실패했습니다.")
                return
    
    # 분석 유형 선택
    analysis_type = st.selectbox(
        "분석 유형",
        AppConfig.UI_CONFIG["ANALYSIS_TYPES"]
    )
    
    # 샘플 질문
    sample_questions = AppConfig.get_sample_questions(analysis_type)
    
    with st.expander("💡 예시 질문"):
        for i, question in enumerate(sample_questions[:4]):  # 처음 4개만
            if st.button(question, key=f"sample_{i}"):
                st.session_state.current_question = question
    
    # 질문 입력
    user_question = st.text_input(
        "질문을 입력하세요:",
        value=st.session_state.get('current_question', ''),
        placeholder=f"{analysis_type}에 관한 질문을 입력해주세요..."
    )
    
    # 분석 실행
    if st.button("🚀 AI 분석 실행", type="primary") and user_question:
        with st.spinner(f"{analysis_type} 분석 중..."):
            query_processor = st.session_state.query_processor
            current_date = stock_data.index[-1].strftime('%Y-%m-%d')
            current_price = stock_data['종가'].iloc[-1]
            
            result, retrieved_docs = query_processor.process_query(
                user_question, analysis_type, stock_name, 
                current_date, current_price, stock_data
            )
            
            st.markdown("### 🎯 분석 결과")
            st.markdown(result)
            
            if retrieved_docs:
                with st.expander(f"📚 참고 문서 ({len(retrieved_docs)}개)"):
                    for i, doc in enumerate(retrieved_docs[:3]):
                        st.write(f"**{i+1}.** {doc.metadata.get('date', 'Unknown')} - {doc.metadata.get('type', 'Unknown')}")


def render_technical_tab(stock_data, stock_name):
    """기술적 지표 탭 렌더링"""
    st.subheader("📈 기술적 지표 상세")
    
    # 최신 데이터
    latest = stock_data.iloc[-1]
    
    # 지표 카테고리별 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 추세 지표**")
        st.metric("SMA(5)", f"{latest['SMA_5']:,.0f}원")
        st.metric("SMA(20)", f"{latest['SMA_20']:,.0f}원")
        st.write(f"**단기 추세**: {latest['Trend_5']}")
        st.write(f"**장기 추세**: {latest['Trend_20']}")
    
    with col2:
        st.markdown("**⚡ 모멘텀 지표**")
        st.metric("RSI(14)", f"{latest['RSI_14']:.1f}", latest['RSI_Signal'])
        st.metric("MACD", f"{latest['MACD']:.2f}", latest['MACD_Signal'])
        st.metric("스토캐스틱 %K", f"{latest['%K']:.1f}")
    
    # 종합 신호
    from models.technical_indicators import TechnicalIndicators
    signal_info = TechnicalIndicators.calculate_signal_strength(stock_data)
    
    st.markdown("---")
    st.markdown("### 🎯 종합 기술적 신호")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("강세 신호", f"{signal_info['bullish_signals']}개")
    with col2:
        st.metric("약세 신호", f"{signal_info['bearish_signals']}개")
    with col3:
        st.metric("종합 판단", signal_info['overall_signal'])


def main():
    """메인 함수"""
    # 애플리케이션 초기화
    initialize_app()
    
    # UI 렌더링
    render_sidebar()
    render_main_content()
    
    # 오래된 채팅 정리 (백그라운드)
    if st.session_state.get('initialized'):
        ChatManager.cleanup_old_chats(max_chats=5)


if __name__ == "__main__":
    main() 