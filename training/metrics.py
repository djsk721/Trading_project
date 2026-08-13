"""
모델 예측 성능 평가 메트릭
"""

import numpy as np
from typing import Dict, Tuple
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class PredictionMetrics:
    """주가 예측 모델 평가 메트릭 클래스"""
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        예측 성능 메트릭 계산
        
        Args:
            predictions: 예측값 배열
            targets: 실제값 배열
            
        Returns:
            메트릭 딕셔너리
        """
        # 기본 회귀 메트릭
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # 추가 메트릭
        mape = self._calculate_mape(targets, predictions)
        directional_accuracy = self._calculate_directional_accuracy(targets, predictions)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    def _calculate_mape(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        MAPE (Mean Absolute Percentage Error) 계산
        """
        # 0으로 나누기 방지
        mask = targets != 0
        if not np.any(mask):
            return float('inf')
        
        return np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    
    def _calculate_directional_accuracy(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        방향성 정확도 계산 (상승/하락 방향 예측 정확도)
        """
        if len(targets) < 2:
            return 0.0
        
        # 실제 방향 (1: 상승, 0: 하락)
        actual_direction = np.diff(targets) > 0
        
        # 예측 방향
        predicted_direction = np.diff(predictions) > 0
        
        # 방향 일치 비율
        return np.mean(actual_direction == predicted_direction) * 100
    
    def calculate_trading_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                                prices: np.ndarray = None) -> Dict[str, float]:
        """
        트레이딩 관련 메트릭 계산
        
        Args:
            predictions: 예측값
            targets: 실제값  
            prices: 실제 주가 (수익률 계산용)
            
        Returns:
            트레이딩 메트릭 딕셔너리
        """
        metrics = {}
        
        if prices is not None:
            # 예측 기반 트레이딩 시뮬레이션
            returns = self._simulate_trading(predictions, prices)
            metrics['total_return'] = np.sum(returns)
            metrics['annualized_return'] = np.mean(returns) * 252  # 연간화
            metrics['volatility'] = np.std(returns) * np.sqrt(252)
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
            metrics['max_drawdown'] = self._calculate_max_drawdown(np.cumsum(returns))
        
        return metrics
    
    def _simulate_trading(self, predictions: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """
        간단한 트레이딩 시뮬레이션
        상승 예측 시 매수, 하락 예측 시 매도
        """
        if len(predictions) < 2 or len(prices) < 2:
            return np.array([])
        
        # 예측 방향
        predicted_direction = np.diff(predictions) > 0
        
        # 실제 수익률
        actual_returns = np.diff(prices) / prices[:-1]
        
        # 예측에 따른 포지션 (1: 롱, -1: 숏)
        positions = np.where(predicted_direction, 1, -1)
        
        # 트레이딩 수익률
        trading_returns = positions * actual_returns
        
        return trading_returns
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """최대 낙폭 계산"""
        if len(cumulative_returns) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown) * 100  # 백분율로 변환
    
    def plot_predictions(self, predictions: np.ndarray, targets: np.ndarray, 
                        save_path: str = None, title: str = "Prediction vs Actual"):
        """
        예측값 vs 실제값 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # 1. 시계열 플롯
        axes[0, 0].plot(targets, label='Actual', alpha=0.8)
        axes[0, 0].plot(predictions, label='Predicted', alpha=0.8)
        axes[0, 0].set_title('Time Series Comparison')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 산점도
        axes[0, 1].scatter(targets, predictions, alpha=0.6)
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 1].set_title('Actual vs Predicted')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 오차 분포
        errors = predictions - targets
        axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(errors), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(errors):.4f}')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 오차 시계열
        axes[1, 1].plot(errors, alpha=0.8)
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.8)
        axes[1, 1].set_title('Error Over Time')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Prediction Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"플롯이 저장되었습니다: {save_path}")
        
        plt.show()
    
    def print_metrics_report(self, metrics: Dict[str, float]):
        """메트릭 리포트 출력"""
        print("=" * 50)
        print("모델 성능 평가 리포트")
        print("=" * 50)
        
        print(f"MAE (Mean Absolute Error):     {metrics.get('mae', 0):.6f}")
        print(f"MSE (Mean Squared Error):      {metrics.get('mse', 0):.6f}")
        print(f"RMSE (Root Mean Squared Error): {metrics.get('rmse', 0):.6f}")
        print(f"R² Score:                      {metrics.get('r2', 0):.6f}")
        print(f"MAPE (Mean Absolute % Error):  {metrics.get('mape', 0):.2f}%")
        print(f"Directional Accuracy:          {metrics.get('directional_accuracy', 0):.2f}%")
        
        # 트레이딩 메트릭이 있는 경우
        if 'total_return' in metrics:
            print("\n" + "=" * 30)
            print("트레이딩 성능")
            print("=" * 30)
            print(f"Total Return:                  {metrics.get('total_return', 0):.4f}")
            print(f"Annualized Return:             {metrics.get('annualized_return', 0):.4f}")
            print(f"Volatility:                    {metrics.get('volatility', 0):.4f}")
            print(f"Sharpe Ratio:                  {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"Max Drawdown:                  {metrics.get('max_drawdown', 0):.2f}%")
        
        print("=" * 50)