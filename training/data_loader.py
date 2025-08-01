"""
시계열 주식 데이터 로더 클래스
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Optional, List
import logging

from models.resnet.model_config import ResNetConfig


class TimeSeriesDataset(Dataset):
    """시계열 주식 데이터셋"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, transform=None):
        self.sequences = sequences
        self.targets = targets
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"데이터셋 생성: sequences {sequences.shape}, targets {targets.shape}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        # numpy array를 torch tensor로 변환
        sequence = torch.FloatTensor(sequence)
        target = torch.FloatTensor([target])
        
        # 변환 적용
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, target


class StockDataLoader:
    """주식 시계열 데이터 로더 관리 클래스"""
    
    def __init__(self, data_dir: str, config: ResNetConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 데이터 변환 설정
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
    
    def _get_train_transform(self):
        """훈련용 데이터 변환"""
        def transform(tensor):
            # 정규화
            if hasattr(self.config, 'normalization_mean') and hasattr(self.config, 'normalization_std'):
                # 시계열 데이터용 정규화 (각 특성별로)
                if len(tensor.shape) == 2:  # (sequence_length, features)
                    mean = torch.tensor(self.config.normalization_mean[:tensor.shape[1]])
                    std = torch.tensor(self.config.normalization_std[:tensor.shape[1]])
                    tensor = (tensor - mean) / std
            
            return tensor
        
        return transform
    
    def _get_val_transform(self):
        """검증용 데이터 변환"""
        return self._get_train_transform()  # 동일한 정규화 적용
    
    def load_from_files(self, train_ratio: float = 0.7, val_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """파일에서 데이터 로드 후 데이터 로더 생성"""
        
        # 데이터 파일 확인
        sequences_file = self.data_dir / "sequences.npy"
        targets_file = self.data_dir / "targets.npy"
        
        if not sequences_file.exists() or not targets_file.exists():
            raise FileNotFoundError(
                f"데이터 파일을 찾을 수 없습니다: {sequences_file}, {targets_file}"
            )
        
        # 데이터 로드
        sequences = np.load(sequences_file)
        targets = np.load(targets_file)
        
        self.logger.info(f"데이터 로드 완료: sequences {sequences.shape}, targets {targets.shape}")
        
        return self.create_data_loaders(sequences, targets, train_ratio, val_ratio)
    
    def create_data_loaders(self, sequences: np.ndarray, targets: np.ndarray,
                          train_ratio: float = 0.7, val_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """시퀀스 데이터에서 데이터 로더 생성"""
        
        # 전체 데이터셋 생성
        full_dataset = TimeSeriesDataset(sequences, targets)
        
        # 데이터셋 분할
        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 각 데이터셋에 변환 적용을 위해 새로운 데이터셋 생성
        train_sequences = sequences[train_dataset.indices]
        train_targets = targets[train_dataset.indices]
        train_dataset = TimeSeriesDataset(train_sequences, train_targets, self.train_transform)
        
        val_sequences = sequences[val_dataset.indices]
        val_targets = targets[val_dataset.indices]
        val_dataset = TimeSeriesDataset(val_sequences, val_targets, self.val_transform)
        
        test_sequences = sequences[test_dataset.indices]
        test_targets = targets[test_dataset.indices]
        test_dataset = TimeSeriesDataset(test_sequences, test_targets, self.val_transform)
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        self.logger.info(f"데이터 로더 생성 완료")
        self.logger.info(f"훈련: {len(train_dataset)}, 검증: {len(val_dataset)}, 테스트: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def create_single_dataloader(self, sequences: np.ndarray, targets: np.ndarray,
                               batch_size: Optional[int] = None, 
                               shuffle: bool = False) -> DataLoader:
        """단일 데이터 로더 생성"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        dataset = TimeSeriesDataset(sequences, targets, self.val_transform)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
    
    def prepare_data_from_dataframe(self, df: pd.DataFrame, 
                                  feature_columns: List[str],
                                  target_column: str = 'close',
                                  sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        DataFrame에서 시퀀스 데이터 준비
        
        Args:
            df: 원본 데이터프레임
            feature_columns: 특성으로 사용할 컬럼들
            target_column: 타겟 컬럼
            sequence_length: 시퀀스 길이
            
        Returns:
            (sequences, targets) 튜플
        """
        if sequence_length is None:
            sequence_length = self.config.sequence_length
        
        # 필요한 컬럼 확인
        missing_cols = set(feature_columns + [target_column]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"누락된 컬럼들: {missing_cols}")
        
        sequences = []
        targets = []
        
        for i in range(len(df) - sequence_length):
            # 특성 시퀀스
            feature_seq = df[feature_columns].iloc[i:i + sequence_length].values
            sequences.append(feature_seq)
            
            # 타겟 (다음 시점의 값)
            target_val = df[target_column].iloc[i + sequence_length]
            targets.append(target_val)
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        self.logger.info(f"시퀀스 데이터 생성 완료: sequences {sequences.shape}, targets {targets.shape}")
        
        return sequences, targets
    
    def save_processed_data(self, sequences: np.ndarray, targets: np.ndarray, 
                          save_dir: str, prefix: str = "processed"):
        """전처리된 데이터 저장"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        sequences_file = save_path / f"{prefix}_sequences.npy"
        targets_file = save_path / f"{prefix}_targets.npy"
        
        np.save(sequences_file, sequences)
        np.save(targets_file, targets)
        
        self.logger.info(f"데이터 저장 완료:")
        self.logger.info(f"  시퀀스: {sequences_file}")
        self.logger.info(f"  타겟: {targets_file}")
    
    def get_data_info(self, sequences: np.ndarray, targets: np.ndarray) -> dict:
        """데이터 정보 반환"""
        return {
            'total_samples': len(sequences),
            'sequence_length': sequences.shape[1],
            'num_features': sequences.shape[2],
            'target_shape': targets.shape,
            'sequences_dtype': sequences.dtype,
            'targets_dtype': targets.dtype,
            'sequences_memory_mb': sequences.nbytes / (1024 * 1024),
            'targets_memory_mb': targets.nbytes / (1024 * 1024)
        }