"""
ResNet 모델 설정 클래스
"""

from dataclasses import dataclass
from typing import Optional, List
import yaml


@dataclass
class ResNetConfig:
    """ResNet 모델 설정"""
    
    # 모델 아키텍처 설정
    model_type: str = 'resnet18'  # 'resnet18', 'resnet34'
    num_classes: int = 1  # 회귀이므로 1
    input_channels: int = 3  # RGB 채널 또는 특성 채널 수
    dropout_rate: float = 0.5
    
    # 입력 데이터 설정
    image_size: int = 224  # 입력 이미지 크기
    sequence_length: int = 60  # 시계열 길이 (60일)
    
    # 훈련 설정
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-4
    
    # 학습률 스케줄링
    scheduler_type: str = 'step'  # 'step', 'cosine', 'plateau'
    step_size: int = 30
    gamma: float = 0.1
    
    # 얼리 스타핑
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    
    # 정규화 설정
    normalization_mean: List[float] = None
    normalization_std: List[float] = None
    
    # 모델 저장 설정
    save_best_only: bool = True
    monitor_metric: str = 'val_loss'  # 'val_loss', 'val_mae', 'val_mse'
    
    def __post_init__(self):
        """초기화 후 기본값 설정"""
        if self.normalization_mean is None:
            self.normalization_mean = [0.485, 0.456, 0.406]  # ImageNet 기본값
        if self.normalization_std is None:
            self.normalization_std = [0.229, 0.224, 0.225]   # ImageNet 기본값
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ResNetConfig':
        """YAML 파일에서 설정 로드"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """설정을 YAML 파일로 저장"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, allow_unicode=True)
    
    def update(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"알 수 없는 설정 키: {key}")
    
    def validate(self):
        """설정 값 검증"""
        if self.model_type not in ['resnet18', 'resnet34']:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        if self.num_classes < 1:
            raise ValueError("num_classes는 1 이상이어야 합니다.")
        
        if self.input_channels < 1:
            raise ValueError("input_channels는 1 이상이어야 합니다.")
        
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate는 0과 1 사이여야 합니다.")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate는 양수여야 합니다.")
        
        if self.batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다.")
        
        if self.num_epochs < 1:
            raise ValueError("num_epochs는 1 이상이어야 합니다.")