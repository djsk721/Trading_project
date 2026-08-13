"""
시계열 데이터 및 모델 결과 시각화 유틸리티
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import logging


class TimeSeriesVisualizer:
    """시계열 데이터 시각화 클래스"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.logger = logging.getLogger(__name__)
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False 
        sns.set_palette("husl")
    
    def plot_time_series(self, data: pd.DataFrame, columns: List[str], 
                        title: str = "Time Series Data", save_path: Optional[str] = None):
        """
        시계열 데이터 플롯
        
        Args:
            data: 시계열 데이터
            columns: 플롯할 컬럼들
            title: 그래프 제목
            save_path: 저장 경로
        """
        fig, axes = plt.subplots(len(columns), 1, figsize=self.figsize, sharex=True)
        if len(columns) == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            if col in data.columns:
                axes[i].plot(data.index, data[col], linewidth=1.5)
                axes[i].set_title(f'{col}')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylabel(col)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"그래프가 저장되었습니다: {save_path}")
        
        plt.show()
    
    def plot_predictions_comparison(self, actual: np.ndarray, predicted: np.ndarray,
                                  dates: Optional[pd.DatetimeIndex] = None,
                                  title: str = "Predictions vs Actual",
                                  save_path: Optional[str] = None):
        """
        예측값과 실제값 비교 플롯
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 인덱스 설정
        if dates is not None:
            x_axis = dates[-len(actual):]
        else:
            x_axis = range(len(actual))
        
        # 1. 시계열 비교
        axes[0, 0].plot(x_axis, actual, label='Actual', alpha=0.8, linewidth=2)
        axes[0, 0].plot(x_axis, predicted, label='Predicted', alpha=0.8, linewidth=2)
        axes[0, 0].set_title('Time Series Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 산점도
        axes[0, 1].scatter(actual, predicted, alpha=0.6)
        min_val = min(np.min(actual), np.min(predicted))
        max_val = max(np.max(actual), np.max(predicted))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].set_title('Actual vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 오차 분포
        errors = predicted - actual
        axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(errors), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(errors):.4f}')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 오차 시계열
        axes[1, 1].plot(x_axis, errors, alpha=0.8)
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.8)
        axes[1, 1].set_title('Error Over Time')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Prediction Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"예측 비교 그래프가 저장되었습니다: {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: Optional[str] = None):
        """
        훈련 히스토리 플롯
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. Loss
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. MAE
        if 'val_mae' in history:
            axes[0, 1].plot(history['val_mae'], label='Validation MAE', linewidth=2, color='orange')
            axes[0, 1].set_title('Mean Absolute Error')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. RMSE
        if 'val_rmse' in history:
            axes[1, 0].plot(history['val_rmse'], label='Validation RMSE', linewidth=2, color='green')
            axes[1, 0].set_title('Root Mean Squared Error')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Learning Rate
        if 'learning_rate' in history:
            axes[1, 1].plot(history['learning_rate'], label='Learning Rate', linewidth=2, color='red')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Training History', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"훈련 히스토리가 저장되었습니다: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                              top_n: int = 20, title: str = "Feature Importance",
                              save_path: Optional[str] = None):
        """
        특성 중요도 플롯
        """
        # 중요도 순으로 정렬
        sorted_idx = np.argsort(importance_scores)[::-1][:top_n]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        plt.figure(figsize=self.figsize)
        plt.barh(range(len(sorted_features)), sorted_scores)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"특성 중요도 그래프가 저장되었습니다: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = "Correlation Matrix",
                               save_path: Optional[str] = None):
        """
        상관관계 행렬 히트맵
        """
        # 수치형 데이터만 선택
        numeric_data = data.select_dtypes(include=[np.number])
        
        plt.figure(figsize=self.figsize)

        correlation_matrix = numeric_data.corr()
        
        # 마스크 생성 (상삼각 부분 숨기기)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5)
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"상관관계 행렬이 저장되었습니다: {save_path}")
        
        plt.show()
    
    def create_interactive_plot(self, data: pd.DataFrame, 
                              y_columns: List[str],
                              title: str = "Interactive Time Series",
                              save_path: Optional[str] = None):
        """
        인터랙티브 시계열 플롯 (Plotly 사용)
        """
        fig = make_subplots(
            rows=len(y_columns), cols=1,
            subplot_titles=y_columns,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        for i, col in enumerate(y_columns, 1):
            if col in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col],
                        mode='lines',
                        name=col,
                        line=dict(width=2)
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title=title,
            height=300 * len(y_columns),
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"인터랙티브 플롯이 저장되었습니다: {save_path}")
        
        fig.show()
    
    def plot_residuals_analysis(self, actual: np.ndarray, predicted: np.ndarray,
                               features: Optional[pd.DataFrame] = None,
                               save_path: Optional[str] = None):
        """
        잔차 분석 플롯
        """
        residuals = actual - predicted
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. 잔차 vs 예측값
        axes[0, 0].scatter(predicted, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 잔차 히스토그램
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q 플롯
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 잔차의 시계열 플롯
        axes[1, 0].plot(residuals, alpha=0.8)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 잔차의 자기상관
        from statsmodels.tsa.stattools import acf
        lags = min(40, len(residuals) // 4)
        autocorr = acf(residuals, nlags=lags)
        axes[1, 1].stem(range(len(autocorr)), autocorr)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].set_title('Residuals Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 잔차의 절댓값 vs 예측값
        axes[1, 2].scatter(predicted, np.abs(residuals), alpha=0.6)
        axes[1, 2].set_xlabel('Predicted Values')
        axes[1, 2].set_ylabel('|Residuals|')
        axes[1, 2].set_title('|Residuals| vs Predicted')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Residuals Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"잔차 분석이 저장되었습니다: {save_path}")
        
        plt.show()
    
    def save_all_plots(self, output_dir: str):
        """모든 플롯을 지정된 디렉토리에 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"모든 플롯이 {output_dir}에 저장됩니다.")