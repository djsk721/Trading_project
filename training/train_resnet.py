#!/usr/bin/env python3
"""
ResNet 모델 훈련 메인 스크립트
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from models.resnet import PricePredictor, ResNetConfig
from training.trainer import ResNetTrainer
from training.data_loader import StockDataLoader
from utils.data_preprocessing import DataPreprocessor


def setup_logging(log_dir: str):
    """로깅 설정"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='ResNet 주가 예측 모델 훈련')
    parser.add_argument('--config', type=str, default='config/resnet_config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='전처리된 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default='data/models',
                       help='모델 저장 디렉토리')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='로그 저장 디렉토리')
    parser.add_argument('--resume', type=str, default=None,
                       help='훈련 재개할 체크포인트 경로')
    parser.add_argument('--gpu', type=int, default=None,
                       help='사용할 GPU ID (None이면 자동 선택)')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging(args.log_dir)
    
    # GPU 설정
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 디바이스: {device}")
    
    try:
        # 설정 로드
        config = ResNetConfig.from_yaml(args.config)
        config.validate()
        logger.info(f"설정 로드 완료: {args.config}")
        
        # 데이터 로더 생성
        data_loader = StockDataLoader(
            data_dir=args.data_dir,
            config=config
        )
        
        train_loader, val_loader, test_loader = data_loader.get_data_loaders()
        logger.info(f"데이터 로더 생성 완료")
        logger.info(f"훈련 샘플: {len(train_loader.dataset)}")
        logger.info(f"검증 샘플: {len(val_loader.dataset)}")
        logger.info(f"테스트 샘플: {len(test_loader.dataset)}")
        
        # 모델 생성
        predictor = PricePredictor(config)
        logger.info(f"모델 생성 완료: {config.model_type}")
        logger.info(f"모델 정보: {predictor.get_model_info()}")
        
        # 트레이너 생성
        trainer = ResNetTrainer(
            model=predictor.model,
            config=config,
            device=device,
            output_dir=args.output_dir
        )
        
        # 체크포인트에서 재개
        start_epoch = 0
        if args.resume:
            start_epoch = trainer.load_checkpoint(args.resume)
            logger.info(f"체크포인트에서 훈련 재개: epoch {start_epoch}")
        
        # 훈련 시작
        logger.info("훈련 시작")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            start_epoch=start_epoch
        )
        
        # 최종 평가
        logger.info("최종 평가 시작")
        test_metrics = trainer.evaluate(test_loader)
        logger.info(f"테스트 결과: {test_metrics}")
        
        # 최고 모델 저장
        best_model_path = Path(args.output_dir) / "best_models" / f"best_{config.model_type}.pth"
        predictor.save_model(str(best_model_path))
        logger.info(f"최고 모델 저장 완료: {best_model_path}")
        
    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()