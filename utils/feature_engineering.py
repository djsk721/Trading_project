"""
주식 데이터 특성 엔지니어링 유틸리티
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

from models.technical_indicators import TechnicalIndicators


class FeatureEngineer:
    """주식 데이터 특성 엔지니어링 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.technical_indicators = TechnicalIndicators()
        self.scalers = {}
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 기반 특성 생성
        
        Args:
            data: OHLCV 데이터
            
        Returns:
            기술적 특성이 추가된 DataFrame
        """
        df = data.copy()
        df = TechnicalIndicators.calculate_advanced_indicators(df)

        return df
    
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', 
                      fit_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        특성 스케일링
        
        Args:
            df: 스케일링할 데이터
            method: 스케일링 방법 ('standard', 'minmax')
            fit_columns: 스케일러를 피팅할 컬럼들 (None이면 모든 수치형 컬럼)
            
        Returns:
            스케일링된 DataFrame
        """
        df_result = df.copy()
        
        if fit_columns is None:
            fit_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"지원하지 않는 스케일링 방법: {method}")
        
        # 스케일러 피팅 및 변환
        for col in fit_columns:
            if col in df.columns:
                # NaN 값 처리
                valid_mask = ~df[col].isna()
                if valid_mask.sum() > 0:
                    df_result.loc[valid_mask, col] = scaler.fit_transform(
                        df.loc[valid_mask, col].values.reshape(-1, 1)
                    ).flatten()
                    
                    # 스케일러 저장
                    self.scalers[col] = scaler
        
        return df_result
    
    def create_target_features(self, df: pd.DataFrame, target_column: str = '종가', 
                             periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        타겟 변수 생성 (미래 가격 예측용)
        
        Args:
            df: 원본 데이터
            target_column: 타겟으로 사용할 컬럼
            periods: 예측 기간들
            
        Returns:
            타겟 특성이 추가된 DataFrame
        """
        df_result = df.copy()
        
        for period in periods:
            # 미래 가격
            df_result[f'target_{period}d'] = df[target_column].shift(-period)
            
            # 미래 수익률
            df_result[f'target_return_{period}d'] = (
                df_result[f'target_{period}d'] / df[target_column] - 1
            )
            
            # 미래 방향 (상승/하락)
            df_result[f'target_direction_{period}d'] = (
                df_result[f'target_return_{period}d'] > 0
            ).astype(int)
        
        return df_result
    
    def remove_highly_correlated_features(self, df: pd.DataFrame, 
                                        threshold: float = 0.95) -> pd.DataFrame:
        """
        고도로 상관된 특성 제거
        
        Args:
            df: 원본 데이터
            threshold: 상관관계 임계값
            
        Returns:
            상관성 높은 특성이 제거된 DataFrame
        """
        # 수치형 데이터만 선택
        numeric_df = df.select_dtypes(include=[np.number])
        
        # 상관관계 행렬 계산
        corr_matrix = numeric_df.corr().abs()
        
        # 상삼각 행렬 생성 (중복 제거)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 높은 상관관계를 가진 특성 찾기
        high_corr_features = [
            column for column in upper_tri.columns 
            if any(upper_tri[column] > threshold)
        ]
        
        self.logger.info(f"제거될 고상관 특성들: {high_corr_features}")
        
        # 비수치형 데이터와 결합
        non_numeric_df = df.select_dtypes(exclude=[np.number])
        low_corr_numeric = numeric_df.drop(columns=high_corr_features)
        
        result = pd.concat([non_numeric_df, low_corr_numeric], axis=1)
        
        return result
    
    def get_feature_importance_names(self) -> List[str]:
        """특성 중요도 분석을 위한 특성명 리스트 반환"""
        feature_groups = {
            'price': ['return_', 'hl_spread', 'oc_spread', 'price_position'],
            'moving_average': ['sma_', 'ema_', '_ratio'],
            'momentum': ['rsi', 'stoch', 'williams', 'macd', 'roc'],
            'volatility': ['bb_', 'atr', 'volatility_'],
            'volume': ['volume_', 'obv', 'vwap'],
            'trend': ['adx', 'sar', 'trend_'],
            'lag': ['_lag_'],
            'rolling': ['_mean_', '_std_', '_max_', '_min_'],
            'interaction': ['_x_', '_div_', '_minus_']
        }
        
        return feature_groups