"""
ResNet 기반 주가 예측 서비스
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import torch
import logging
from pathlib import Path
from datetime import datetime, timedelta

from models.resnet import PricePredictor, ResNetConfig
from utils.data_preprocessing import DataPreprocessor
from utils.feature_engineering import FeatureEngineer
from models.stock_data import StockData


class PredictionService:
    """주가 예측 서비스 클래스"""
    
    def __init__(self, model_path: str, config_path: str):
        self.logger = logging.getLogger(__name__)
        
        # 설정 로드
        self.config = ResNetConfig.from_yaml(config_path)
        
        # 모델 초기화 및 로드
        self.predictor = PricePredictor(self.config)
        self.predictor.load_model(model_path)
        
        # 전처리기 및 특성 엔지니어링
        self.preprocessor = DataPreprocessor(
            sequence_length=self.config.sequence_length
        )
        self.feature_engineer = FeatureEngineer()
        
        # 주식 데이터 로더
        self.stock_data = StockData()
        
        self.logger.info("예측 서비스 초기화 완료")
    
    def predict_single_stock(self, symbol: str, 
                           prediction_days: int = 1,
                           use_latest_data: bool = True) -> Dict[str, Any]:
        """
        단일 종목 주가 예측
        
        Args:
            symbol: 종목 코드 또는 심볼
            prediction_days: 예측할 일수
            use_latest_data: 최신 데이터 사용 여부
            
        Returns:
            예측 결과 딕셔너리
        """
        try:
            # 주식 데이터 수집
            if use_latest_data:
                # 최신 데이터 다운로드
                data = self._fetch_latest_data(symbol)
            else:
                # 저장된 데이터 사용
                data = self._load_cached_data(symbol)
            
            if data is None or len(data) < self.config.sequence_length:
                raise ValueError(f"충분한 데이터가 없습니다. 필요: {self.config.sequence_length}, 보유: {len(data) if data is not None else 0}")
            
            # 특성 엔지니어링
            features_data = self.feature_engineer.create_technical_features(data)
            
            # 시퀀스 데이터 준비
            feature_columns = self._get_feature_columns(features_data)
            sequences, _ = self.preprocessor.create_sequences(
                features_data[feature_columns], 
                target_column='close'
            )
            
            if len(sequences) == 0:
                raise ValueError("생성된 시퀀스가 없습니다.")
            
            # 최신 시퀀스 사용
            latest_sequence = sequences[-1:]  # 마지막 시퀀스
            
            # 예측 수행
            predictions = []
            current_sequence = latest_sequence.copy()
            
            for day in range(prediction_days):
                # 예측
                pred = self.predictor.predict(current_sequence)
                predictions.append(pred[0])
                
                # 다음 날 예측을 위해 시퀀스 업데이트 (간단한 방법)
                if day < prediction_days - 1:
                    # 마지막 값을 예측값으로 업데이트하고 시퀀스를 한 칸씩 이동
                    new_row = current_sequence[0, -1, :].copy()
                    new_row[0] = pred[0]  # close 가격 업데이트 (첫 번째 특성이라 가정)
                    
                    # 시퀀스 업데이트
                    current_sequence = np.roll(current_sequence, -1, axis=1)
                    current_sequence[0, -1, :] = new_row
            
            # 결과 구성
            current_price = data['close'].iloc[-1]
            predicted_prices = predictions
            
            result = {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_prices': [float(p) for p in predicted_prices],
                'prediction_dates': self._generate_prediction_dates(prediction_days),
                'price_changes': [float(p - current_price) for p in predicted_prices],
                'percentage_changes': [float((p - current_price) / current_price * 100) for p in predicted_prices],
                'confidence_score': self._calculate_confidence_score(latest_sequence),
                'market_trend': self._analyze_trend(predicted_prices),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"{symbol} 예측 완료: {len(predictions)}일")
            return result
            
        except Exception as e:
            self.logger.error(f"{symbol} 예측 중 오류: {e}")
            raise
    
    def predict_multiple_stocks(self, symbols: List[str], 
                              prediction_days: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        여러 종목 동시 예측
        
        Args:
            symbols: 종목 코드/심볼 리스트
            prediction_days: 예측할 일수
            
        Returns:
            종목별 예측 결과 딕셔너리
        """
        results = {}
        
        for symbol in symbols:
            try:
                result = self.predict_single_stock(symbol, prediction_days)
                results[symbol] = result
            except Exception as e:
                self.logger.error(f"{symbol} 예측 실패: {e}")
                results[symbol] = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return results
    
    def get_prediction_analysis(self, symbol: str, 
                              prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        예측 결과 상세 분석
        
        Args:
            symbol: 종목 코드
            prediction_result: 예측 결과
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            current_price = prediction_result['current_price']
            predicted_prices = prediction_result['predicted_prices']
            
            analysis = {
                'symbol': symbol,
                'summary': {
                    'direction': 'up' if predicted_prices[-1] > current_price else 'down',
                    'magnitude': abs(predicted_prices[-1] - current_price),
                    'percentage': abs((predicted_prices[-1] - current_price) / current_price * 100),
                    'volatility': np.std(predicted_prices) if len(predicted_prices) > 1 else 0
                },
                'risk_assessment': {
                    'risk_level': self._assess_risk_level(predicted_prices, current_price),
                    'volatility_score': self._calculate_volatility_score(predicted_prices),
                    'trend_consistency': self._calculate_trend_consistency(predicted_prices)
                },
                'recommendations': self._generate_recommendations(predicted_prices, current_price),
                'technical_signals': self._analyze_technical_signals(symbol),
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"{symbol} 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def _fetch_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """최신 주식 데이터 수집"""
        try:
            # 충분한 기간의 데이터 수집 (기술적 지표 계산을 위해)
            days_needed = self.config.sequence_length + 100  # 여유분 포함
            
            if symbol.isdigit() and len(symbol) == 6:
                # 한국 주식
                data = self.stock_data.get_korean_stock_data(
                    symbol, days=days_needed
                )
            else:
                # 해외 주식
                data = self.stock_data.get_us_stock_data(
                    symbol, days=days_needed
                )
            
            return data
            
        except Exception as e:
            self.logger.error(f"{symbol} 데이터 수집 실패: {e}")
            return None
    
    def _load_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """캐시된 데이터 로드"""
        # 구현 필요: 저장된 데이터에서 로드
        return None
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """예측에 사용할 특성 컬럼 선택"""
        # 기본 OHLCV
        base_columns = ['open', 'high', 'low', 'close']
        if 'volume' in data.columns:
            base_columns.append('volume')
        
        # 기술적 지표 추가
        tech_columns = []
        for col in data.columns:
            if any(indicator in col for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'atr']):
                tech_columns.append(col)
        
        # 수익률 관련
        return_columns = [col for col in data.columns if 'return_' in col]
        
        # 모든 특성 결합
        feature_columns = base_columns + tech_columns + return_columns
        
        # 실제 존재하는 컬럼만 선택
        available_columns = [col for col in feature_columns if col in data.columns]
        
        # NaN이 너무 많은 컬럼 제외
        clean_columns = []
        for col in available_columns:
            if data[col].notna().sum() > len(data) * 0.7:  # 70% 이상 유효한 데이터
                clean_columns.append(col)
        
        return clean_columns[:self.config.input_channels]  # 설정된 채널 수만큼
    
    def _generate_prediction_dates(self, prediction_days: int) -> List[str]:
        """예측 날짜 생성"""
        dates = []
        current_date = datetime.now()
        
        for i in range(1, prediction_days + 1):
            # 주말 건너뛰기
            pred_date = current_date + timedelta(days=i)
            while pred_date.weekday() >= 5:  # 토요일(5), 일요일(6)
                pred_date += timedelta(days=1)
            dates.append(pred_date.strftime('%Y-%m-%d'))
        
        return dates
    
    def _calculate_confidence_score(self, sequence: np.ndarray) -> float:
        """예측 신뢰도 점수 계산"""
        # 간단한 신뢰도 계산 (실제로는 더 복잡한 방법 사용)
        # 데이터의 변동성과 최신성을 고려
        volatility = np.std(sequence)
        trend_consistency = self._calculate_trend_consistency(sequence.flatten())
        
        # 0-1 사이의 점수로 정규화
        confidence = max(0.1, min(0.9, 0.8 - volatility * 0.1 + trend_consistency * 0.2))
        return float(confidence)
    
    def _analyze_trend(self, prices: List[float]) -> str:
        """가격 추세 분석"""
        if len(prices) < 2:
            return 'neutral'
        
        if prices[-1] > prices[0]:
            return 'bullish'
        elif prices[-1] < prices[0]:
            return 'bearish'
        else:
            return 'neutral'
    
    def _assess_risk_level(self, predicted_prices: List[float], current_price: float) -> str:
        """리스크 레벨 평가"""
        max_change = max(abs(p - current_price) / current_price for p in predicted_prices)
        
        if max_change < 0.02:  # 2% 미만
            return 'low'
        elif max_change < 0.05:  # 5% 미만
            return 'medium'
        else:
            return 'high'
    
    def _calculate_volatility_score(self, prices: List[float]) -> float:
        """변동성 점수 계산"""
        if len(prices) < 2:
            return 0.0
        return float(np.std(prices) / np.mean(prices))
    
    def _calculate_trend_consistency(self, values: np.ndarray) -> float:
        """추세 일관성 계산"""
        if len(values) < 2:
            return 0.0
        
        # 연속된 값들의 방향성 일치도
        directions = np.sign(np.diff(values))
        consistency = np.abs(np.mean(directions))
        return float(consistency)
    
    def _generate_recommendations(self, predicted_prices: List[float], current_price: float) -> List[str]:
        """투자 추천 생성"""
        recommendations = []
        
        final_price = predicted_prices[-1]
        change_pct = (final_price - current_price) / current_price * 100
        
        if change_pct > 3:
            recommendations.append("강한 매수 신호")
        elif change_pct > 1:
            recommendations.append("매수 고려")
        elif change_pct < -3:
            recommendations.append("매도 고려")
        elif change_pct < -1:
            recommendations.append("주의 관찰")
        else:
            recommendations.append("중립적 전망")
        
        # 변동성 기반 추천
        volatility = np.std(predicted_prices)
        if volatility > current_price * 0.02:
            recommendations.append("높은 변동성 - 리스크 관리 필요")
        
        return recommendations
    
    def _analyze_technical_signals(self, symbol: str) -> Dict[str, str]:
        """기술적 신호 분석"""
        # 간단한 기술적 신호 (실제로는 더 상세한 분석 필요)
        return {
            'rsi_signal': 'neutral',
            'macd_signal': 'neutral',
            'bollinger_signal': 'neutral',
            'volume_signal': 'neutral'
        }