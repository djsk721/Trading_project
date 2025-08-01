"""
시계열 주식 데이터 전처리 유틸리티
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from models.technical_indicators import TechnicalIndicators


class DataPreprocessor:
    """시계열 주식 데이터 전처리 클래스"""
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.logger = logging.getLogger(__name__)
        self.technical_indicators = TechnicalIndicators()
        self.scalers = {}
    
    def create_sequences(self, data: pd.DataFrame, feature_columns: List[str], 
                        target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 데이터를 시퀀스로 변환
        
        Args:
            data: 주식 데이터 DataFrame
            feature_columns: 특성으로 사용할 컬럼들
            target_column: 예측 대상 컬럼명
            
        Returns:
            (sequences, targets) 튜플 - sequences: (num_samples, sequence_length, num_features)
        """
        # 결측값 처리
        data = data[feature_columns + [target_column]].fillna(method='ffill').fillna(method='bfill')
        
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            # 입력 시퀀스 - 특성들만 선택
            seq = data[feature_columns].iloc[i:i + self.sequence_length].values
            sequences.append(seq)
            
            # 타겟 (다음날 종가)
            target = data[target_column].iloc[i + self.sequence_length]
            targets.append(target)
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        self.logger.info(f"시퀀스 생성 완료: {sequences.shape}, 타겟: {targets.shape}")
        
        return sequences, targets
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'minmax') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        데이터 정규화
        
        Args:
            data: 정규화할 데이터
            method: 정규화 방법 ('minmax', 'zscore', 'robust')
            
        Returns:
            (정규화된 데이터, 정규화 파라미터)
        """
        normalized_data = data.copy()
        normalization_params = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            if method == 'minmax':
                min_val = data[column].min()
                max_val = data[column].max()
                normalized_data[column] = (data[column] - min_val) / (max_val - min_val)
                normalization_params[column] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
                
            elif method == 'zscore':
                mean_val = data[column].mean()
                std_val = data[column].std()
                normalized_data[column] = (data[column] - mean_val) / std_val
                normalization_params[column] = {'mean': mean_val, 'std': std_val, 'method': 'zscore'}
                
            elif method == 'robust':
                median_val = data[column].median()
                mad_val = np.median(np.abs(data[column] - median_val))
                normalized_data[column] = (data[column] - median_val) / mad_val
                normalization_params[column] = {'median': median_val, 'mad': mad_val, 'method': 'robust'}
        
        return normalized_data, normalization_params
    
    def denormalize_data(self, data: np.ndarray, column: str, params: Dict[str, Any]) -> np.ndarray:
        """데이터 역정규화"""
        if column not in params:
            return data
        
        param = params[column]
        
        if param['method'] == 'minmax':
            return data * (param['max'] - param['min']) + param['min']
        elif param['method'] == 'zscore':
            return data * param['std'] + param['mean']
        elif param['method'] == 'robust':
            return data * param['mad'] + param['median']
        
        return data
    
    def prepare_features_for_resnet(self, data: pd.DataFrame, 
                                   feature_columns: List[str]) -> np.ndarray:
        """
        ResNet 모델을 위한 특성 준비 (정규화 포함)
        
        Args:
            data: 특성 데이터
            feature_columns: 사용할 특성 컬럼들
            
        Returns:
            정규화된 특성 배열 (num_samples, num_features, sequence_length)
        """
        # 특성만 선택
        features = data[feature_columns].fillna(method='ffill').fillna(method='bfill')
        
        # 각 특성별로 정규화
        normalized_features = features.copy()
        for col in feature_columns:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                normalized_features[col] = self.scalers[col].fit_transform(
                    features[col].values.reshape(-1, 1)
                ).flatten()
            else:
                normalized_features[col] = self.scalers[col].transform(
                    features[col].values.reshape(-1, 1)
                ).flatten()
        
        return normalized_features
    
    def create_resnet_sequences(self, data: pd.DataFrame, feature_columns: List[str],
                               target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        ResNet 모델을 위한 시퀀스 생성
        
        Args:
            data: 원본 데이터
            feature_columns: 특성 컬럼들
            target_column: 타겟 컬럼
            
        Returns:
            (sequences, targets) - sequences: (num_samples, num_features, sequence_length)
        """
        # 특성 준비
        normalized_features = self.prepare_features_for_resnet(data, feature_columns)
        
        sequences = []
        targets = []
        
        for i in range(len(normalized_features) - self.sequence_length):
            # 시퀀스 생성: (sequence_length, num_features) -> (num_features, sequence_length)
            seq = normalized_features.iloc[i:i + self.sequence_length].values.T
            sequences.append(seq)
            
            # 타겟
            target = data[target_column].iloc[i + self.sequence_length]
            targets.append(target)
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        self.logger.info(f"ResNet 시퀀스 생성 완료: {sequences.shape}, 타겟: {targets.shape}")
        
        return sequences, targets
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 추가
        
        Args:
            data: OHLCV 데이터
            
        Returns:
            기술적 지표가 추가된 데이터
        """
        df = data.copy()
        
        if len(df) < 20:
            self.logger.warning("데이터가 부족하여 일부 기술적 지표를 계산할 수 없습니다.")
            return df
        
        try:
            # 이동평균
            for period in [5, 10, 20]:
                if len(df) >= period:
                    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI
            if len(df) >= 14:
                df['rsi'] = self.technical_indicators.calculate_rsi(df['close'])
            
            # MACD
            if len(df) >= 26:
                macd, signal, histogram = self.technical_indicators.calculate_macd(df['close'])
                df['macd'] = macd
                df['macd_signal'] = signal
                df['macd_histogram'] = histogram
            
            # 볼린저 밴드
            if len(df) >= 20:
                bb_upper, bb_lower = self.technical_indicators.calculate_bollinger_bands(df['close'])
                df['bb_upper'] = bb_upper
                df['bb_lower'] = bb_lower
                df['bb_width'] = (bb_upper - bb_lower) / df['close']
            
            # 수익률 특성
            df['return_1d'] = df['close'].pct_change()
            df['return_5d'] = df['close'].pct_change(5)
            df['return_10d'] = df['close'].pct_change(10)
            
            # 변동성
            df['volatility_10d'] = df['return_1d'].rolling(window=10).std()
            df['volatility_20d'] = df['return_1d'].rolling(window=20).std()
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 오류: {e}")
        
        return df
    
    def save_processed_data(self, sequences: np.ndarray, targets: np.ndarray, 
                          output_dir: str, prefix: str = "processed"):
        """전처리된 시계열 데이터 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        sequences_file = output_path / f"{prefix}_sequences.npy"
        targets_file = output_path / f"{prefix}_targets.npy"
        
        np.save(sequences_file, sequences)
        np.save(targets_file, targets)
        
        self.logger.info(f"전처리된 데이터 저장 완료:")
        self.logger.info(f"  시퀀스: {sequences_file} (shape: {sequences.shape})")
        self.logger.info(f"  타겟: {targets_file} (shape: {targets.shape})")
    
    def load_processed_data(self, data_dir: str, prefix: str = "processed") -> Tuple[np.ndarray, np.ndarray]:
        """저장된 전처리 데이터 로드"""
        data_path = Path(data_dir)
        
        sequences_file = data_path / f"{prefix}_sequences.npy"
        targets_file = data_path / f"{prefix}_targets.npy"
        
        if not sequences_file.exists() or not targets_file.exists():
            raise FileNotFoundError(f"전처리된 데이터 파일을 찾을 수 없습니다: {data_path}")
        
        sequences = np.load(sequences_file)
        targets = np.load(targets_file)
        
        self.logger.info(f"전처리된 데이터 로드 완료:")
        self.logger.info(f"  시퀀스: {sequences.shape}")
        self.logger.info(f"  타겟: {targets.shape}")
        
        return sequences, targets
    
    def get_feature_columns(self, data: pd.DataFrame, max_features: int = 20) -> List[str]:
        """
        사용할 특성 컬럼 선택
        
        Args:
            data: 데이터프레임
            max_features: 최대 특성 수
            
        Returns:
            선택된 특성 컬럼 리스트
        """
        # 기본 OHLCV
        base_columns = []
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                base_columns.append(col)
        
        # 기술적 지표
        tech_columns = []
        for col in data.columns:
            if any(indicator in col for indicator in ['sma_', 'ema_', 'rsi', 'macd', 'bb_', 'return_', 'volatility_']):
                tech_columns.append(col)
        
        # 결합
        all_columns = base_columns + tech_columns
        
        # NaN이 적은 컬럼들만 선택
        clean_columns = []
        for col in all_columns:
            if col in data.columns:
                nan_ratio = data[col].isna().sum() / len(data)
                if nan_ratio < 0.3:  # 30% 미만 결측값
                    clean_columns.append(col)
        
        # 최대 특성 수만큼 선택
        return clean_columns[:max_features]