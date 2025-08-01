# ResNet 기반 주가 예측 모델

## 개요

기존 RAG 기반 주식 분석 시스템에 ResNet(Residual Network) 기반의 딥러닝 주가 예측 모델을 추가했습니다. 이 모델은 시계열 주가 데이터를 1D ConvNet으로 처리하여 다음날 주가를 예측합니다.

## 폴더 구조

```
Trading_project/
├── models/
│   ├── resnet/                      # ResNet 모델 관련
│   │   ├── __init__.py
│   │   ├── resnet_model.py         # 1D ResNet 아키텍처
│   │   ├── price_predictor.py      # 예측 모델 래퍼 클래스
│   │   └── model_config.py         # 모델 설정
│   └── ...                         # 기존 모델들
│
├── training/                       # 모델 훈련 관련
│   ├── __init__.py
│   ├── train_resnet.py            # 훈련 메인 스크립트
│   ├── data_loader.py             # 시계열 데이터 로더
│   ├── trainer.py                 # 훈련 클래스
│   └── metrics.py                 # 평가 메트릭
│
├── utils/                         # 유틸리티 (확장)
│   ├── data_preprocessing.py      # 시계열 데이터 전처리
│   ├── feature_engineering.py     # 특성 엔지니어링
│   └── visualization.py           # 시각화 도구
│
├── services/
│   ├── prediction_service.py      # ResNet 예측 서비스
│   └── ...                        # 기존 서비스들
│
├── data/                          # 데이터 저장소
│   ├── raw/                       # 원시 데이터
│   ├── processed/                 # 전처리된 데이터
│   ├── features/                  # 특성 데이터
│   ├── models/                    # 훈련된 모델
│   │   ├── checkpoints/           # 체크포인트
│   │   ├── best_models/           # 최적 모델
│   │   └── backups/               # 모델 백업
│   └── plots/                     # 시각화 결과
│
├── config/
│   └── resnet_config.yaml         # ResNet 설정 파일
│
└── example_usage.py               # 사용 예제
```

## 주요 특징

### 1. 1D ResNet 아키텍처
- **시계열 데이터 특화**: 2D 이미지가 아닌 1D 시계열 데이터 처리
- **입력 형태**: `(batch_size, features, sequence_length)`
- **특성 수**: 최대 20개 (OHLCV + 기술적 지표들)
- **시퀀스 길이**: 60일 기본값

### 2. 포괄적인 특성 엔지니어링
- **기본 가격 특성**: OHLCV, 수익률, 변동성
- **이동평균**: SMA, EMA (5, 10, 20일)
- **모멘텀 지표**: RSI, MACD, Stochastic
- **변동성 지표**: Bollinger Bands, ATR
- **거래량 지표**: OBV, VWAP

### 3. 강력한 훈련 파이프라인
- **자동 체크포인트 저장**
- **얼리 스타핑**: 과적합 방지
- **학습률 스케줄링**: Step, Cosine, Plateau
- **다양한 평가 메트릭**: MAE, MSE, RMSE, 방향성 정확도

### 4. 시각화 및 분석
- **훈련 히스토리 시각화**
- **예측 vs 실제값 비교**
- **잔차 분석**
- **특성 중요도 분석**

## 사용 방법

### 1. 기본 설정

```python
from models.resnet import ResNetConfig, PricePredictor

# 설정 로드
config = ResNetConfig.from_yaml("config/resnet_config.yaml")

# 모델 생성
predictor = PricePredictor(config)
```

### 2. 데이터 준비

```python
from utils.data_preprocessing import DataPreprocessor
from utils.feature_engineering import FeatureEngineer
from models.stock_data import StockData

# 주식 데이터 수집
stock_data = StockData()
data = stock_data.get_korean_stock_data("005930", days=200)  # 삼성전자

# 특성 엔지니어링
feature_engineer = FeatureEngineer()
enhanced_data = feature_engineer.create_technical_features(data)

# 전처리 및 시퀀스 생성
preprocessor = DataPreprocessor(sequence_length=60)
feature_columns = preprocessor.get_feature_columns(enhanced_data, max_features=20)
sequences, targets = preprocessor.create_resnet_sequences(
    enhanced_data, feature_columns, target_column='close'
)
```

### 3. 모델 훈련

```python
from training.trainer import ResNetTrainer
from training.data_loader import StockDataLoader

# 데이터 로더 생성
data_loader = StockDataLoader("data/processed", config)
train_loader, val_loader, test_loader = data_loader.create_data_loaders(
    sequences, targets
)

# 훈련
trainer = ResNetTrainer(predictor.model, config, device, "data/models")
trainer.train(train_loader, val_loader)
```

### 4. 예측 및 평가

```python
# 예측 수행
predictions = predictor.predict(test_sequences)

# 성능 평가
from training.metrics import PredictionMetrics
metrics = PredictionMetrics()
results = metrics.calculate_metrics(predictions, test_targets)
metrics.print_metrics_report(results)
```

### 5. 커맨드라인 훈련

```bash
# 기본 훈련
python training/train_resnet.py --config config/resnet_config.yaml

# GPU 지정
python training/train_resnet.py --config config/resnet_config.yaml --gpu 0

# 체크포인트에서 재개
python training/train_resnet.py --resume data/models/checkpoints/checkpoint_epoch_50.pth
```

## 설정 파일 (config/resnet_config.yaml)

```yaml
# 모델 아키텍처
model_type: "resnet18"
num_classes: 1
input_channels: 20
dropout_rate: 0.3

# 시계열 데이터
sequence_length: 60
feature_size: 20

# 훈련 설정
learning_rate: 0.001
batch_size: 32
num_epochs: 100
early_stopping_patience: 5

# 스케줄링
scheduler_type: "step"
step_size: 30
gamma: 0.1
```

## 성능 지표

모델은 다음과 같은 지표로 평가됩니다:

- **MAE (Mean Absolute Error)**: 절대 오차 평균
- **RMSE (Root Mean Squared Error)**: 제곱근 평균 제곱 오차
- **방향성 정확도**: 상승/하락 방향 예측 정확도
- **R² Score**: 결정계수
- **MAPE**: 평균 절대 백분율 오차

## 예제 실행

```bash
# 전체 예제 실행
python example_usage.py

# 특정 종목 예측 서비스 사용
from services.prediction_service import PredictionService

service = PredictionService(
    model_path="data/models/best_models/resnet18.pth",
    config_path="config/resnet_config.yaml"
)

# 단일 종목 예측
result = service.predict_single_stock("005930", prediction_days=5)
```

## 주의사항

1. **충분한 데이터**: 최소 200일 이상의 데이터 필요
2. **메모리 사용량**: 시퀀스 길이와 배치 크기에 따라 메모리 사용량 증가
3. **GPU 사용**: CUDA 사용 시 훈련 속도 대폭 향상
4. **과적합 방지**: 얼리 스타핑과 드롭아웃 적극 활용

## 향후 개선 사항

- [ ] Transformer 기반 모델 추가
- [ ] 앙상블 예측 기능
- [ ] 실시간 예측 API
- [ ] 멀티 태스크 학습 (가격 + 방향성 동시 예측)
- [ ] 강화학습 기반 포트폴리오 최적화

---

이 ResNet 모델은 기존 RAG 시스템과 완전히 독립적으로 작동하며, 두 시스템을 조합하여 더욱 강력한 주식 분석 플랫폼을 구축할 수 있습니다.