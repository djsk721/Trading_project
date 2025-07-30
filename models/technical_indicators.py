"""
기술적 지표 계산 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from config.settings import AppConfig


class TechnicalIndicators:
    """기술적 지표 계산 클래스"""
    
    @staticmethod
    def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        고급 기술적 지표 계산 함수
        
        Args:
            df: OHLCV 데이터프레임
        
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 데이터프레임
        """
        df_result = df.copy()
        config = AppConfig.TECHNICAL_INDICATORS
        
        # 이동평균선 계산
        df_result = TechnicalIndicators._calculate_moving_averages(df_result, config)
        
        # MACD 지표 계산
        df_result = TechnicalIndicators._calculate_macd(df_result, config)
        
        # RSI 지표 계산
        df_result = TechnicalIndicators._calculate_rsi(df_result, config)
        
        # 볼린저 밴드 계산
        df_result = TechnicalIndicators._calculate_bollinger_bands(df_result, config)
        
        # 스토캐스틱 계산
        df_result = TechnicalIndicators._calculate_stochastic(df_result, config)
        
        # 거래량 분석
        df_result = TechnicalIndicators._calculate_volume_indicators(df_result, config)
        
        # 추세 분석
        df_result = TechnicalIndicators._calculate_trend_indicators(df_result)
        
        # 변동성 지표
        df_result = TechnicalIndicators._calculate_volatility_indicators(df_result, config)
        
        return df_result.fillna(method='ffill').fillna(0)
    
    @staticmethod
    def _calculate_moving_averages(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """이동평균선 계산"""
        # 단순 이동평균선
        for period in config['SMA_PERIODS']:
            df[f'SMA_{period}'] = df['종가'].rolling(window=period, min_periods=1).mean()
        
        # 지수 이동평균선
        for period in config['EMA_PERIODS']:
            df[f'EMA_{period}'] = df['종가'].ewm(span=period, adjust=False).mean()
        
        return df
    
    @staticmethod
    def _calculate_macd(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """MACD 지표 계산"""
        fast_ema = df['종가'].ewm(span=config['MACD_FAST'], adjust=False).mean()
        slow_ema = df['종가'].ewm(span=config['MACD_SLOW'], adjust=False).mean()
        
        df['MACD'] = fast_ema - slow_ema
        df['Signal'] = df['MACD'].ewm(span=config['MACD_SIGNAL'], adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal']
        
        # MACD 신호 생성
        df['MACD_Signal'] = np.where(df['MACD'] > df['Signal'], 'BUY', 'SELL')
        df['MACD_Cross'] = (df['MACD'] > df['Signal']) != (df['MACD'].shift() > df['Signal'].shift())
        
        return df
    
    @staticmethod
    def _calculate_rsi(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """RSI 지표 계산"""
        period = config['RSI_PERIOD']
        delta = df['종가'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        df[f'RSI_{period}'] = df[f'RSI_{period}'].fillna(50)
        
        # RSI 신호 생성
        overbought = AppConfig.EVENT_THRESHOLDS['RSI_OVERBOUGHT']
        oversold = AppConfig.EVENT_THRESHOLDS['RSI_OVERSOLD']
        
        df['RSI_Signal'] = np.where(
            df[f'RSI_{period}'] > overbought, 'OVERBOUGHT',
            np.where(df[f'RSI_{period}'] < oversold, 'OVERSOLD', 'NEUTRAL')
        )
        
        return df
    
    @staticmethod
    def _calculate_bollinger_bands(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """볼린저 밴드 계산"""
        period = config['BB_PERIOD']
        std_dev = config['BB_STD']
        
        df['BB_Middle'] = df['종가'].rolling(window=period, min_periods=1).mean()
        df['BB_Std'] = df['종가'].rolling(window=period, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + std_dev * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - std_dev * df['BB_Std']
        
        # 볼린저 밴드 위치 및 신호
        df['BB_Position'] = (df['종가'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['BB_Signal'] = np.where(
            df['종가'] > df['BB_Upper'], 'UPPER_BREAK',
            np.where(df['종가'] < df['BB_Lower'], 'LOWER_BREAK', 'MIDDLE')
        )
        
        return df
    
    @staticmethod
    def _calculate_stochastic(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """스토캐스틱 오실레이터 계산"""
        period = config['STOCH_PERIOD']
        smooth_period = config['STOCH_SMOOTH']
        
        low_period = df['저가'].rolling(window=period, min_periods=1).min()
        high_period = df['고가'].rolling(window=period, min_periods=1).max()
        denominator = high_period - low_period
        
        df['%K'] = np.where(
            denominator != 0,
            (df['종가'] - low_period) / denominator * 100,
            50
        )
        df['%D'] = df['%K'].rolling(window=smooth_period, min_periods=1).mean()
        
        return df
    
    @staticmethod
    def _calculate_volume_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """거래량 분석 지표 계산"""
        period = config['VOLUME_PERIOD']
        
        df[f'Volume_SMA_{period}'] = df['거래량'].rolling(window=period, min_periods=1).mean()
        df['Volume_Ratio'] = df['거래량'] / df[f'Volume_SMA_{period}']
        
        # 거래량 신호 생성
        df['Volume_Signal'] = np.where(
            df['Volume_Ratio'] > 1.5, 'HIGH_VOLUME',
            np.where(df['Volume_Ratio'] < 0.5, 'LOW_VOLUME', 'NORMAL')
        )
        
        return df
    
    @staticmethod
    def _calculate_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """추세 분석 지표 계산"""
        # 가격 변화율
        df['Price_Change'] = df['종가'].pct_change() * 100
        
        # 추세 방향
        df['Trend_5'] = np.where(df['종가'] > df['SMA_5'], 'UP', 'DOWN')
        df['Trend_20'] = np.where(df['종가'] > df['SMA_20'], 'UP', 'DOWN')
        
        return df
    
    @staticmethod
    def _calculate_volatility_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """변동성 지표 계산"""
        period = config['VOLATILITY_PERIOD']
        
        # 변동성 (표준편차)
        df['Volatility'] = df['Price_Change'].rolling(window=period).std()
        
        # 고저가 비율
        df['High_Low_Ratio'] = (df['고가'] - df['저가']) / df['종가'] * 100
        
        return df
    
    @staticmethod
    def get_current_indicators(df: pd.DataFrame) -> Dict[str, float]:
        """
        현재 기술적 지표 값들 추출
        
        Args:
            df: 기술적 지표가 계산된 데이터프레임
        
        Returns:
            Dict: 현재 지표 값들
        """
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        return {
            "current_rsi": float(latest['RSI_14']),
            "current_macd": float(latest['MACD']),
            "current_stoch_k": float(latest['%K']),
            "current_volume": float(latest['거래량']),
            "support_level": float(df['저가'].rolling(20).min().iloc[-1]),
            "resistance_level": float(df['고가'].rolling(20).max().iloc[-1]),
            "current_price": float(latest['종가']),
            "price_change": float(latest['Price_Change']),
            "volatility": float(latest['Volatility']),
            "bb_position": float(latest['BB_Position']),
            "volume_ratio": float(latest['Volume_Ratio'])
        }
    
    @staticmethod
    def calculate_signal_strength(df: pd.DataFrame) -> Dict[str, Any]:
        """
        종합 기술적 신호 강도 계산
        
        Args:
            df: 기술적 지표가 계산된 데이터프레임
        
        Returns:
            Dict: 신호 강도 정보
        """
        if df.empty:
            return {"bullish_signals": 0, "bearish_signals": 0, "overall_signal": "중립"}
        
        latest = df.iloc[-1]
        
        bullish_signals = 0
        bearish_signals = 0
        
        # 추세 신호
        if latest['Trend_5'] == 'UP':
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if latest['Trend_20'] == 'UP':
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # RSI 신호
        if latest['RSI_Signal'] == 'OVERSOLD':
            bullish_signals += 1
        elif latest['RSI_Signal'] == 'OVERBOUGHT':
            bearish_signals += 1
        
        # MACD 신호
        if latest['MACD_Signal'] == 'BUY':
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # 볼린저밴드 신호
        if latest['BB_Signal'] == 'LOWER_BREAK':
            bullish_signals += 1
        elif latest['BB_Signal'] == 'UPPER_BREAK':
            bearish_signals += 1
        
        total_signals = bullish_signals + bearish_signals
        bullish_pct = (bullish_signals / total_signals * 100) if total_signals > 0 else 50
        
        overall_signal = "강세" if bullish_signals > bearish_signals else "약세" if bearish_signals > bullish_signals else "중립"
        
        return {
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "bullish_percentage": bullish_pct,
            "overall_signal": overall_signal
        } 