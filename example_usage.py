#!/usr/bin/env python3
"""
ResNet 기반 주가 예측 모델 사용 예제
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 프로젝트 모듈들
from models.resnet import ResNetConfig, PricePredictor
from utils.data_preprocessing import DataPreprocessor
from utils.feature_engineering import FeatureEngineer
from training.data_loader import StockDataLoader
from training.trainer import ResNetTrainer
from utils.visualization import TimeSeriesVisualizer
from models.stock_data import StockDataManager


def example_data_preparation():
    """데이터 준비 예제"""
    logger.info("=== 데이터 준비 예제 ===")
    
    # 삼성전자 데이터 수집 (예시)
    symbol = "005930"  # 삼성전자
    data = StockDataManager.get_stock_data(symbol, days=200, target_currency='KRW')
    
    logger.info(f"원본 데이터 크기: {data.shape}")
    logger.info(f"데이터 컬럼: {list(data.columns)}")
    
    # 특성 엔지니어링
    feature_engineer = FeatureEngineer()
    enhanced_data = feature_engineer.create_technical_features(data)
    
    logger.info(f"특성 엔지니어링 후 크기: {enhanced_data.shape}")
    logger.info(f"추가된 특성 수: {enhanced_data.shape[1] - data.shape[1]}")
    
    # 데이터 전처리
    preprocessor = DataPreprocessor(sequence_length=60)
    enhanced_data = preprocessor.add_technical_indicators(enhanced_data)
    
    # 사용할 특성 선택
    feature_columns = preprocessor.get_feature_columns(enhanced_data, max_features=20)
    logger.info(f"선택된 특성들: {feature_columns}")
    
    # ResNet용 시퀀스 생성
    sequences, targets = preprocessor.create_resnet_sequences(
        enhanced_data, 
        feature_columns, 
        target_column='종가'
    )
    
    logger.info(f"생성된 시퀀스 shape: {sequences.shape}")
    logger.info(f"타겟 shape: {targets.shape}")
    
    # 데이터 저장
    preprocessor.save_processed_data(sequences, targets, "data/processed")
    
    return sequences, targets, feature_columns


def example_model_training():
    """모델 훈련 예제"""
    logger.info("=== 모델 훈련 예제 ===")
    
    # 설정 로드
    config = ResNetConfig.from_yaml("config/resnet_config.yaml")
    logger.info(f"모델 설정: {config.model_type}, 입력 채널: {config.input_channels}")
    
    # 데이터 로더 생성
    data_loader = StockDataLoader("data/processed", config)
    train_loader, val_loader, test_loader = data_loader.load_from_files()
    
    # 모델 생성
    predictor = PricePredictor(config)
    logger.info(f"모델 정보: {predictor.get_model_info()}")
    
    # 트레이너 생성
    trainer = ResNetTrainer(
        model=predictor.model,
        config=config,
        device=predictor.device,
        output_dir="data/models"
    )
    
    # 훈련 실행 (짧은 예제)
    config.num_epochs = 5  # 예제용으로 짧게
    logger.info("훈련 시작...")
    trainer.train(train_loader, val_loader)
    
    # 최고 모델 저장
    predictor.save_model("data/models/best_models/example_model.pth")
    
    return predictor, trainer


def example_prediction():
    """예측 예제"""
    logger.info("=== 예측 예제 ===")
    
    # 저장된 모델 로드
    config = ResNetConfig.from_yaml("config/resnet_config.yaml")
    predictor = PricePredictor(config)
    
    try:
        predictor.load_model("data/models/best_models/example_model.pth")
        logger.info("모델 로드 완료")
    except FileNotFoundError:
        logger.warning("저장된 모델이 없습니다. 먼저 훈련을 실행하세요.")
        return
    
    # 테스트 데이터로 예측
    data_loader = StockDataLoader("data/processed", config)
    _, _, test_loader = data_loader.load_from_files()
    
    # 예측 수행
    predictor.set_training_mode(False)
    predictions = []
    actuals = []
    
    for batch_data, batch_targets in test_loader:
        batch_predictions = predictor.predict(batch_data.numpy())
        predictions.extend(batch_predictions)
        actuals.extend(batch_targets.numpy().flatten())
        
        if len(predictions) >= 100:  # 예제용으로 100개만
            break
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    logger.info(f"예측 완료: {len(predictions)}개 샘플")
    
    # 성능 평가
    from training.metrics import PredictionMetrics
    metrics_calculator = PredictionMetrics()
    metrics = metrics_calculator.calculate_metrics(predictions, actuals)
    
    logger.info("예측 성능:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.6f}")
    
    return predictions, actuals


def example_visualization():
    """시각화 예제"""
    logger.info("=== 시각화 예제 ===")
    
    try:    
        data = StockDataManager.get_stock_data("005930", days=200, target_currency='KRW')
        
        # 시각화
        visualizer = TimeSeriesVisualizer()
        
        # 1. 시계열 데이터 플롯
        visualizer.plot_time_series(
            data, 
            columns=['종가', '거래량'], 
            title="삼성전자 주가 및 거래량",
            save_path="data/plots/timeseries_example.png"
        )
        
        # 2. 상관관계 행렬
        feature_engineer = FeatureEngineer()
        enhanced_data = feature_engineer.create_technical_features(data)
        
        visualizer.plot_correlation_matrix(
            enhanced_data,
            title="기술적 지표 상관관계",
            save_path="data/plots/correlation_example.png"
        )
        
        logger.info("시각화 완료")
        
    except Exception as e:
        logger.error(f"시각화 오류: {e}")


def main():
    """메인 실행 함수"""
    logger.info("ResNet 기반 주가 예측 시스템 예제 실행")
    
    try:
        # 1. 데이터 준비
        sequences, targets, feature_columns = example_data_preparation()
        
        # 2. 모델 훈련 (선택사항)
        # predictor, trainer = example_model_training()
        
        # 3. 예측 (훈련된 모델이 있을 때)
        # predictions, actuals = example_prediction()
        
        # 4. 시각화
        example_visualization()
        
        logger.info("모든 예제 실행 완료!")
        
    except Exception as e:
        logger.error(f"예제 실행 중 오류: {e}")
        raise


if __name__ == "__main__":
    main()