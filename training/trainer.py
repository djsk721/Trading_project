"""
ResNet 모델 훈련 클래스
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
import time

from models.resnet.model_config import ResNetConfig
from training.metrics import PredictionMetrics


class EarlyStopping:
    """얼리 스타핑 클래스"""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score - self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class ResNetTrainer:
    """ResNet 모델 훈련 클래스"""
    
    def __init__(self, model: nn.Module, config: ResNetConfig, 
                 device: torch.device, output_dir: str):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "best_models").mkdir(exist_ok=True)
        
        # 손실 함수 및 옵티마이저 설정
        self.criterion = nn.MSELoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 메트릭 계산기
        self.metrics = PredictionMetrics()
        
        # 얼리 스타핑
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        )
        
        # 훈련 기록
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_mse': [],
            'val_rmse': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """옵티마이저 생성"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _create_scheduler(self):
        """학습률 스케줄러 생성"""
        if self.config.scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        elif self.config.scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=10,
                factor=0.5
            )
        else:
            return None
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 순전파
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'Avg Loss': f"{total_loss / (batch_idx + 1):.6f}"
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(dataloader, desc="Validation"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # 예측값과 실제값 저장
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 메트릭 계산
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        
        metrics = self.metrics.calculate_metrics(predictions, targets)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              start_epoch: int = 0):
        """전체 훈련 프로세스"""
        self.logger.info(f"훈련 시작: {self.config.num_epochs} 에포크")
        
        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            
            # 훈련
            train_loss = self.train_epoch(train_loader)
            
            # 검증
            val_metrics = self.validate_epoch(val_loader)
            val_loss = val_metrics['loss']
            
            # 학습률 업데이트
            if self.scheduler:
                if self.config.scheduler_type == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 현재 학습률
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 훈련 기록 업데이트
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_mae'].append(val_metrics['mae'])
            self.train_history['val_mse'].append(val_metrics['mse'])
            self.train_history['val_rmse'].append(val_metrics['rmse'])
            self.train_history['learning_rate'].append(current_lr)
            
            # 최고 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_best_model()
            
            # 에포크 결과 로그
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Val MAE: {val_metrics['mae']:.6f}, "
                f"Val RMSE: {val_metrics['rmse']:.6f}, "
                f"LR: {current_lr:.8f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # 체크포인트 저장
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, train_loss, val_loss)
            
            # 얼리 스타핑 확인
            if self.early_stopping(val_loss):
                self.logger.info(f"얼리 스타핑: {epoch+1} 에포크에서 훈련 중단")
                break
        
        self.logger.info(f"훈련 완료. 최고 성능: Epoch {self.best_epoch+1}, Val Loss: {self.best_val_loss:.6f}")
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """모델 평가"""
        return self.validate_epoch(dataloader)
    
    def _save_best_model(self):
        """최고 성능 모델 저장"""
        model_path = self.output_dir / "best_models" / f"best_{self.config.model_type}.pth"
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }, model_path)
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """체크포인트 저장"""
        checkpoint_path = self.output_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_history': self.train_history,
            'config': self.config.__dict__
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        return checkpoint['epoch'] + 1