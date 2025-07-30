"""
RAG용 문서 생성 모듈
"""
import pandas as pd
from typing import List
from langchain.schema import Document

from config.settings import AppConfig


class DocumentGenerator:
    """RAG용 문서 생성 클래스"""
    
    def __init__(self, stock_data: pd.DataFrame, stock_name: str):
        """
        초기화
        
        Args:
            stock_data: 주식 데이터프레임
            stock_name: 종목명
        """
        self.stock_data = stock_data
        self.stock_name = stock_name
        self.documents = []
    
    def generate_documents(self) -> List[Document]:
        """
        다양한 유형의 문서 생성
        
        Returns:
            List[Document]: 생성된 문서 리스트
        """
        self.documents = []
        
        for idx, (date, row) in enumerate(self.stock_data.iterrows()):
            date_str = date.strftime('%Y-%m-%d')
            
            # 1. 일일 가격 동향 문서
            price_doc = self._create_price_document(date_str, row)
            self.documents.append(price_doc)
            
            # 2. 기술적 지표 분석 문서
            technical_doc = self._create_technical_document(date_str, row)
            self.documents.append(technical_doc)
            
            # 3. 거래량 분석 문서
            volume_doc = self._create_volume_document(date_str, row)
            self.documents.append(volume_doc)
            
            # 4. 특별 이벤트 문서 (조건부 생성)
            if self._is_special_event(row):
                event_doc = self._create_event_document(date_str, row)
                self.documents.append(event_doc)
        
        return self.documents
    
    def _create_price_document(self, date: str, row: pd.Series) -> Document:
        """
        가격 동향 문서 생성
        
        Args:
            date: 날짜 문자열
            row: 해당 날짜 데이터
        
        Returns:
            Document: 가격 동향 문서
        """
        sma_5_status = "상승" if row['종가'] > row['SMA_5'] else "하락"
        sma_20_status = "상승" if row['종가'] > row['SMA_20'] else "하락"
        
        content = f"""
        {date} {self.stock_name} 가격 동향 분석:
        - 종가: {row['종가']:,.0f}원
        - 일일 변동률: {row['Price_Change']:.2f}%
        - 5일 이동평균 대비: {sma_5_status} (현재가: {row['종가']:,.0f}원, 5일평균: {row['SMA_5']:,.0f}원)
        - 20일 이동평균 대비: {sma_20_status} (현재가: {row['종가']:,.0f}원, 20일평균: {row['SMA_20']:,.0f}원)
        - 가격 추세: {row['Trend_20']} (장기), {row['Trend_5']} (단기)
        - 고가: {row['고가']:,.0f}원, 저가: {row['저가']:,.0f}원
        - 거래대금: {(row['종가'] * row['거래량']):,.0f}원
        """.strip()
        
        return Document(
            page_content=content,
            metadata={
                "date": date,
                "type": "price_analysis",
                "stock": self.stock_name,
                "price": float(row['종가']),
                "change": float(row['Price_Change']),
                "trend_short": str(row['Trend_5']),
                "trend_long": str(row['Trend_20'])
            }
        )
    
    def _create_technical_document(self, date: str, row: pd.Series) -> Document:
        """
        기술적 지표 문서 생성
        
        Args:
            date: 날짜 문자열
            row: 해당 날짜 데이터
        
        Returns:
            Document: 기술적 지표 문서
        """
        # RSI 해석
        rsi_interpretation = self._get_rsi_interpretation(row['RSI_14'])
        
        # MACD 해석
        macd_interpretation = self._get_macd_interpretation(row['MACD'], row['Signal'])
        
        # 볼린저밴드 해석
        bb_interpretation = self._get_bb_interpretation(row['BB_Position'], row['BB_Signal'])
        
        content = f"""
        {date} {self.stock_name} 기술적 지표 분석:
        - RSI(14): {row['RSI_14']:.1f} - {rsi_interpretation} ({row['RSI_Signal']})
        - MACD: {row['MACD']:.2f}, Signal: {row['Signal']:.2f} - {macd_interpretation} ({row['MACD_Signal']})
        - 스토캐스틱 %K: {row['%K']:.1f}, %D: {row['%D']:.1f} - {self._get_stoch_interpretation(row['%K'], row['%D'])}
        - 볼린저밴드 위치: {row['BB_Position']:.2f} - {bb_interpretation} ({row['BB_Signal']})
        - 변동성: {row['Volatility']:.2f}% - {self._get_volatility_interpretation(row['Volatility'])}
        - 고저가 비율: {row['High_Low_Ratio']:.2f}%
        """.strip()
        
        return Document(
            page_content=content,
            metadata={
                "date": date,
                "type": "technical_analysis",
                "stock": self.stock_name,
                "rsi": float(row['RSI_14']),
                "rsi_signal": str(row['RSI_Signal']),
                "macd": float(row['MACD']),
                "macd_signal": str(row['MACD_Signal']),
                "bb_position": float(row['BB_Position']),
                "bb_signal": str(row['BB_Signal']),
                "volatility": float(row['Volatility'])
            }
        )
    
    def _create_volume_document(self, date: str, row: pd.Series) -> Document:
        """
        거래량 분석 문서 생성
        
        Args:
            date: 날짜 문자열
            row: 해당 날짜 데이터
        
        Returns:
            Document: 거래량 분석 문서
        """
        volume_interpretation = self._get_volume_interpretation(row['Volume_Ratio'], row['Volume_Signal'])
        price_volume_relation = "긍정적" if row['Price_Change'] * row['Volume_Ratio'] > 0 else "부정적"
        
        content = f"""
        {date} {self.stock_name} 거래량 분석:
        - 거래량: {row['거래량']:,.0f}주
        - 20일 평균 대비: {row['Volume_Ratio']:.2f}배 - {volume_interpretation} ({row['Volume_Signal']})
        - 거래량 신호: {row['Volume_Signal']}
        - 가격-거래량 관계: {price_volume_relation}
        - 거래대금: {(row['종가'] * row['거래량']):,.0f}원
        - 평균 거래량 대비 상태: {self._get_volume_status(row['Volume_Ratio'])}
        """.strip()
        
        return Document(
            page_content=content,
            metadata={
                "date": date,
                "type": "volume_analysis",
                "stock": self.stock_name,
                "volume": float(row['거래량']),
                "volume_ratio": float(row['Volume_Ratio']),
                "volume_signal": str(row['Volume_Signal']),
                "price_volume_relation": price_volume_relation
            }
        )
    
    def _is_special_event(self, row: pd.Series) -> bool:
        """
        특별 이벤트 조건 확인
        
        Args:
            row: 해당 날짜 데이터
        
        Returns:
            bool: 특별 이벤트 여부
        """
        thresholds = AppConfig.EVENT_THRESHOLDS
        
        return (
            abs(row['Price_Change']) > thresholds['PRICE_CHANGE'] or  # 5% 이상 변동
            row['Volume_Ratio'] > thresholds['VOLUME_RATIO'] or       # 거래량 2배 이상
            row['MACD_Cross'] or                                      # MACD 크로스오버
            row['RSI_Signal'] in ['OVERBOUGHT', 'OVERSOLD']           # RSI 극값
        )
    
    def _create_event_document(self, date: str, row: pd.Series) -> Document:
        """
        특별 이벤트 문서 생성
        
        Args:
            date: 날짜 문자열
            row: 해당 날짜 데이터
        
        Returns:
            Document: 특별 이벤트 문서
        """
        events = []
        thresholds = AppConfig.EVENT_THRESHOLDS
        
        if abs(row['Price_Change']) > thresholds['PRICE_CHANGE']:
            direction = "상승" if row['Price_Change'] > 0 else "하락"
            events.append(f"급격한 가격 {direction} ({row['Price_Change']:.2f}%)")
        
        if row['Volume_Ratio'] > thresholds['VOLUME_RATIO']:
            events.append(f"거래량 급증 ({row['Volume_Ratio']:.1f}배)")
        
        if row['MACD_Cross']:
            signal_direction = "상승" if row['MACD'] > row['Signal'] else "하락"
            events.append(f"MACD {signal_direction} 크로스오버 발생")
        
        if row['RSI_Signal'] in ['OVERBOUGHT', 'OVERSOLD']:
            signal_kr = "과매수" if row['RSI_Signal'] == 'OVERBOUGHT' else "과매도"
            events.append(f"RSI {signal_kr} 신호 (RSI: {row['RSI_14']:.1f})")
        
        # 이벤트 심각도 계산
        severity = self._calculate_event_severity(row)
        
        content = f"""
        {date} {self.stock_name} 특별 이벤트:
        - 발생 이벤트: {', '.join(events)}
        - 이벤트 심각도: {severity}
        - 종가: {row['종가']:,.0f}원 (변동률: {row['Price_Change']:.2f}%)
        - 거래량: {row['거래량']:,.0f}주 (평균 대비: {row['Volume_Ratio']:.1f}배)
        - 주요 지표: RSI {row['RSI_14']:.1f}, MACD {row['MACD']:.2f}, Signal {row['Signal']:.2f}
        - 투자자 관심도: {self._get_investor_attention(row['Volume_Ratio'], abs(row['Price_Change']))}
        """.strip()
        
        return Document(
            page_content=content,
            metadata={
                "date": date,
                "type": "special_event",
                "stock": self.stock_name,
                "events": events,
                "severity": severity,
                "price_change": float(row['Price_Change']),
                "volume_ratio": float(row['Volume_Ratio'])
            }
        )
    
    def _get_rsi_interpretation(self, rsi_value: float) -> str:
        """RSI 값 해석"""
        if rsi_value > 70:
            return "과매수 구간으로 조정 가능성"
        elif rsi_value < 30:
            return "과매도 구간으로 반등 가능성"
        elif rsi_value > 50:
            return "상승 모멘텀 유지"
        else:
            return "하락 모멘텀 또는 횡보"
    
    def _get_macd_interpretation(self, macd: float, signal: float) -> str:
        """MACD 값 해석"""
        if macd > signal:
            return "상승 신호, 매수 타이밍"
        else:
            return "하락 신호, 매도 고려"
    
    def _get_bb_interpretation(self, position: float, signal: str) -> str:
        """볼린저밴드 위치 해석"""
        if signal == 'UPPER_BREAK':
            return "상단 돌파, 강한 상승 신호"
        elif signal == 'LOWER_BREAK':
            return "하단 돌파, 강한 하락 신호"
        elif position > 0.8:
            return "상단 근접, 과매수 우려"
        elif position < 0.2:
            return "하단 근접, 과매도 상태"
        else:
            return "중간 구간, 횡보 또는 추세 전환 대기"
    
    def _get_stoch_interpretation(self, k_value: float, d_value: float) -> str:
        """스토캐스틱 해석"""
        if k_value > 80:
            return "과매수 구간"
        elif k_value < 20:
            return "과매도 구간"
        elif k_value > d_value:
            return "상승 모멘텀"
        else:
            return "하락 모멘텀"
    
    def _get_volatility_interpretation(self, volatility: float) -> str:
        """변동성 해석"""
        thresholds = AppConfig.EVENT_THRESHOLDS
        
        if volatility > thresholds['HIGH_VOLATILITY']:
            return "높은 변동성, 위험 증가"
        elif volatility > thresholds['MEDIUM_VOLATILITY']:
            return "보통 변동성"
        else:
            return "낮은 변동성, 안정적"
    
    def _get_volume_interpretation(self, ratio: float, signal: str) -> str:
        """거래량 해석"""
        if signal == 'HIGH_VOLUME':
            return "거래량 급증, 관심 집중"
        elif signal == 'LOW_VOLUME':
            return "거래량 저조, 관심 부족"
        else:
            return "정상 거래량 수준"
    
    def _get_volume_status(self, ratio: float) -> str:
        """거래량 상태 해석"""
        if ratio > 3:
            return "매우 높음"
        elif ratio > 2:
            return "높음"
        elif ratio > 1.5:
            return "다소 높음"
        elif ratio > 0.5:
            return "보통"
        else:
            return "낮음"
    
    def _calculate_event_severity(self, row: pd.Series) -> str:
        """이벤트 심각도 계산"""
        severity_score = 0
        
        # 가격 변동률 기여도
        if abs(row['Price_Change']) > 10:
            severity_score += 3
        elif abs(row['Price_Change']) > 5:
            severity_score += 2
        elif abs(row['Price_Change']) > 3:
            severity_score += 1
        
        # 거래량 기여도
        if row['Volume_Ratio'] > 5:
            severity_score += 3
        elif row['Volume_Ratio'] > 3:
            severity_score += 2
        elif row['Volume_Ratio'] > 2:
            severity_score += 1
        
        # 기술적 지표 기여도
        if row['RSI_Signal'] in ['OVERBOUGHT', 'OVERSOLD']:
            severity_score += 1
        
        if row['MACD_Cross']:
            severity_score += 1
        
        if severity_score >= 5:
            return "매우 높음"
        elif severity_score >= 3:
            return "높음"
        elif severity_score >= 2:
            return "보통"
        else:
            return "낮음"
    
    def _get_investor_attention(self, volume_ratio: float, price_change: float) -> str:
        """투자자 관심도 평가"""
        attention_score = volume_ratio + abs(price_change) / 2
        
        if attention_score > 5:
            return "매우 높음"
        elif attention_score > 3:
            return "높음"
        elif attention_score > 2:
            return "보통"
        else:
            return "낮음" 