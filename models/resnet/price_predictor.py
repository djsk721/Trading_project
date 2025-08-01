"""
주가 예측을 위한 ResNet 래퍼 클래스
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path

from .resnet_model import create_resnet18, create_resnet34
from .model_config import ResNetConfig


class PricePredictor:
    """ResNet 기반 주가 예측 클래스"""
    
    def __init__(self, config: ResNetConfig):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # 모델 초기화
        self._initialize_model()
    
    def _initialize_model(self):
        """모델 초기화"""
        if self.config.model_type == 'resnet18':
            self.model = create_resnet18(
                num_classes=self.config.num_classes,
                input_channels=self.config.input_channels,
                dropout_rate=self.config.dropout_rate
            )
        elif self.config.model_type == 'resnet34':
            self.model = create_resnet34(
                num_classes=self.config.num_classes,
                input_channels=self.config.input_channels,
                dropout_rate=self.config.dropout_rate
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.config.model_type}")
        
        self.model.to(self.device)
        self.logger.info(f"{self.config.model_type} 모델이 {self.device}에 로드되었습니다.")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        주가 예측 수행
        
        Args:
            data: 입력 데이터 (batch_size, channels, height, width)
            
        Returns:
            예측된 주가 값들
        """
        if self.model is None:
            raise RuntimeError("모델이 초기화되지 않았습니다.")
        
        self.model.eval()
        
        # numpy array를 torch tensor로 변환
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)
        
        # 배치 차원 추가 (단일 샘플인 경우)
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        
        data = data.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(data)
            predictions = predictions.cpu().numpy()
        
        return predictions.flatten() if predictions.shape[1] == 1 else predictions
    
    def predict_batch(self, data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        배치 단위로 예측 수행
        
        Args:
            data: 입력 데이터
            batch_size: 배치 크기
            
        Returns:
            예측 결과
        """
        predictions = []
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_pred = self.predict(batch_data)
            predictions.extend(batch_pred)
        
        return np.array(predictions)
    
    def load_model(self, model_path: str):
        """저장된 모델 로드"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.logger.info(f"모델이 {model_path}에서 로드되었습니다.")
    
    def save_model(self, model_path: str, include_config: bool = True):
        """모델 저장"""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
        }
        
        if include_config:
            save_dict['config'] = self.config.__dict__
        
        torch.save(save_dict, model_path)
        self.logger.info(f"모델이 {model_path}에 저장되었습니다.")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': self.config.model_type,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': f"({self.config.input_channels}, H, W)",
            'output_classes': self.config.num_classes
        }
    
    def set_training_mode(self, training: bool = True):
        """훈련/평가 모드 설정"""
        if self.model is not None:
            self.model.train(training)
    
    def get_device(self) -> torch.device:
        """사용 중인 디바이스 반환"""
        return self.device