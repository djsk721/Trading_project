"""
애플리케이션 설정 파일
"""
from typing import List
import json 


config = json.load(open('env.json', 'r'))


class AppConfig:
    """애플리케이션 설정 클래스"""
    
    # Streamlit 페이지 설정
    PAGE_CONFIG = {
        "page_title": "RAG 주식 분석 시스템",
        "page_icon": "📈",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # 캐시 설정 (초 단위)
    CACHE_TTL = {
        "ALL_STOCKS": 3600,      # 1시간
        "POPULAR_STOCKS": 1800,   # 30분
    }
    
    # 데이터 설정
    DATA_CONFIG = {
        "DEFAULT_DAYS": 500,
        "MAX_DAYS": 500,
        "MIN_DAYS": 30,
        "DEFAULT_TICKER": "034220",  # LG디스플레이
        "MAX_STOCKS_DISPLAY": 730,  
    }
    
    # 기술적 지표 설정
    TECHNICAL_INDICATORS = {
        "SMA_PERIODS": [5, 10, 20],
        "EMA_PERIODS": [12, 26],
        "RSI_PERIOD": 14,
        "MACD_FAST": 12,
        "MACD_SLOW": 26,
        "MACD_SIGNAL": 9,
        "BB_PERIOD": 20,
        "BB_STD": 2,
        "STOCH_PERIOD": 14,
        "STOCH_SMOOTH": 3,
        "VOLUME_PERIOD": 20,
        "VOLATILITY_PERIOD": 20,
    }
    
    # 모델 설정
    MODEL_CONFIG = {
        "EMBEDDING_MODEL": "jhgan/ko-sroberta-nli",
        "FALLBACK_EMBEDDING": "sentence-transformers/all-MiniLM-L6-v2",
        "LLM_MODEL": "MLP-KTLim/llama-3-Korean-Bllossom-8B",
        "MAX_NEW_TOKENS": 2048,
        "REPETITION_PENALTY": 1.1,
    }
    
    # RAG 설정
    RAG_CONFIG = {
        "BM25_K": 5,
        "FAISS_K": 5,
        "BM25_WEIGHT": 0.7,
        "FAISS_WEIGHT": 0.3,
        "MAX_DOCS": 8,
        "MAX_CONTEXT_LENGTH": 2500,
        "MAX_DOC_TYPES_PER_TYPE": 3,
    }
    
    # UI 설정
    UI_CONFIG = {
        "ANALYSIS_TYPES": ["기본 분석", "20일 예측", "투자전략"],
        "CHART_TYPES": ["기술적 지표 대시보드", "캔들스틱 + 볼린저밴드"],
        "CHART_PERIODS": ["전체", "최근 3개월", "최근 1개월", "최근 2주"],
        "INDICATOR_CATEGORIES": ["전체 보기", "추세 지표", "모멘텀 지표", "변동성 지표", "거래량 지표"],
        "MAX_HISTORY_DISPLAY": 5,
    }
    
    # 인기 종목 리스트
    POPULAR_STOCKS = {
        "삼성전자 (005930)": "005930",
        "SK하이닉스 (000660)": "000660", 
        "LG디스플레이 (034220)": "034220",
        "NAVER (035420)": "035420",
        "카카오 (035720)": "035720",
        "삼성바이오로직스 (207940)": "207940",
        "LG화학 (051910)": "051910",
        "삼성SDI (006400)": "006400",
        "현대차 (005380)": "005380",
        "기아 (000270)": "000270",
        "포스코홀딩스 (005490)": "005490",
        "KB금융 (105560)": "105560",
        "신한지주 (055550)": "055550",
        "LG전자 (066570)": "066570",
        "하나금융지주 (086790)": "086790",
        "삼성물산 (028260)": "028260",
        "POSCO-International (022100)": "022100",
        "KT&G (033780)": "033780",
        "한국전력 (015760)": "015760",
        "셀트리온 (068270)": "068270"
    }
    
    # 인기 해외 주식 리스트
    POPULAR_INTERNATIONAL_STOCKS = {
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "Google (GOOGL)": "GOOGL",
        "Amazon (AMZN)": "AMZN",
        "Tesla (TSLA)": "TSLA",
        "Meta (META)": "META",
        "NVIDIA (NVDA)": "NVDA",
        "Netflix (NFLX)": "NFLX",
        "Intel (INTC)": "INTC",
        "AMD (AMD)": "AMD",
        "Coca-Cola (KO)": "KO",
        "Johnson & Johnson (JNJ)": "JNJ",
        "Visa (V)": "V",
        "Mastercard (MA)": "MA",
        "JPMorgan Chase (JPM)": "JPM",
        "Berkshire Hathaway (BRK-B)": "BRK-B",
        "UnitedHealth (UNH)": "UNH",
        "Procter & Gamble (PG)": "PG",
        "Home Depot (HD)": "HD",
        "Bank of America (BAC)": "BAC"
    }
    
    # 데이터 소스 설정
    DATA_SOURCE_CONFIG = {
        "KOREAN_SOURCE": "pykrx",      # 한국 주식: pykrx
        "INTERNATIONAL_SOURCE": "yfinance",  # 해외 주식: yfinance
        "KOREAN_PRICE_COLUMN": "종가",    # 한국 주식 가격 컬럼
        "INTERNATIONAL_PRICE_COLUMN": "Close",  # 해외 주식 가격 컬럼
        "SUPPORT_INTERNATIONAL": True,   # 해외 주식 지원 여부
    }
    
    # 환율 설정
    CURRENCY_CONFIG = {
        "DEFAULT_EXCHANGE_RATE": 1300.0,  # 기본 USD/KRW 환율
        "SUPPORTED_CURRENCIES": ["KRW", "USD"],  # 지원 통화
        "CURRENCY_SYMBOLS": {
            "KRW": "원",
            "USD": "$"
        },
        "CURRENCY_NAMES": {
            "KRW": "원화",
            "USD": "달러"
        }
    }
    
    # 특별 이벤트 감지 임계값
    EVENT_THRESHOLDS = {
        "PRICE_CHANGE": 5.0,     # 5% 이상 변동
        "VOLUME_RATIO": 2.0,     # 거래량 2배 이상
        "RSI_OVERBOUGHT": 70,
        "RSI_OVERSOLD": 30,
        "STOCH_OVERBOUGHT": 80,
        "STOCH_OVERSOLD": 20,
        "HIGH_VOLATILITY": 3.0,
        "MEDIUM_VOLATILITY": 1.5,
    }
    
    # 예측 설정
    PREDICTION_CONFIG = {
        "FORECAST_DAYS": 20,
        "CONFIDENCE_LEVELS": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    
    # 프롬프트 템플릿
    PROMPT_TEMPLATES = config
    
    @classmethod
    def get_sample_questions(cls, analysis_type: str) -> List[str]:
        """분석 유형별 샘플 질문 반환"""
        questions = {
            "기본 분석": [
                "지금 매수하기 좋은 타이밍인가요?",
                "최근 거래량 급증의 원인은 무엇인가요?",
                "RSI가 과매수 구간에 있는데 어떻게 해석해야 하나요?",
                "볼린저 밴드 기준으로 현재 주가 위치는 어떤가요?",
                "MACD 크로스오버 신호가 있었나요?",
                "최근 변동성이 높은 이유는 무엇인가요?",
                "기술적 분석 기준으로 지지선과 저항선은 어디인가요?",
                "현재 추세의 지속 가능성은 어떤가요?"
            ],
            "20일 예측": [
                "향후 20일간 주가를 예측해주세요",
                "다음 달 주가 전망과 목표가를 알려주세요",
                "기술적 지표 기반으로 20일 예측해주세요"
            ],
            "투자전략": [
                "단기 투자 전략을 세워주세요",
                "분할매수 전략을 추천해주세요",
                "손절매와 익절매 전략을 알려주세요",
                "포트폴리오 비중을 어떻게 가져가야 할까요?",
                "리스크 관리 방안을 제시해주세요",
                "현재 상황에서 최적의 투자전략은 무엇인가요?"
            ]
        }
        return questions.get(analysis_type, []) 