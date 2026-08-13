# Trading Desk

FastAPI + React 기반 주식 분석/거래 데스크입니다.
LLM은 **Ollama `gemma4:31b`**, 검색은 **BM25 + Ollama 임베딩(`embeddinggemma`)** 앙상블 RAG를 사용합니다.

## Features

- 시세/차트 (KIS 우선, pykrx/yfinance 폴백)
- 기술적 지표 요약 + Lightweight Charts
- Gemma4 RAG 분석 (기본 / 20일 예측 / 투자전략)
- 종목 최신 뉴스 검색 (Google News RSS + yfinance 폴백)
- 일일 종목 추천 (기술적 스코어링)
- KIS 계좌 잔고/보유/주문 API

## Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI, pandas, pykis, pykrx, yfinance, httpx |
| Frontend | React + Vite + TypeScript + lightweight-charts |
| LLM/RAG | Ollama (`gemma4:31b`, `embeddinggemma`) |

## Setup

### 1) Ollama

```bash
ollama pull gemma4:31b
ollama pull embeddinggemma
ollama serve
```

### 2) Environment

```bash
cp .env.example .env
# fill KIS keys if you use trading APIs
```

### 3) Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8010
```

API docs: http://localhost:8010/docs

### 4) Frontend

```bash
cd frontend
npm install
npm run dev
```

UI: http://localhost:5173

## Main API

- `GET /api/health`
- `GET /api/market/chart`
- `GET /api/market/quote/{symbol}`
- `GET /api/news?symbol=005930`
- `GET /api/recommend/daily?market=KRX`
- `POST /api/analysis/ask`
- `GET/POST /api/trading/*`

### Analysis request example

```json
{
  "symbol": "005930",
  "market": "KRX",
  "query": "지금 매수 타이밍인가요?",
  "analysis_type": "basic"
}
```

`analysis_type`: `basic` | `forecast_20d` | `strategy`

## Project layout

```text
backend/app/          # FastAPI + services (KIS, RAG, news, recommend)
frontend/src/         # React UI
.env.example          # environment template
```

## Notes

- Streamlit legacy 코드는 제거되었습니다.
- 주문 API는 실제 체결될 수 있으니 모의투자(`KIS_VIRTUAL=true`)를 권장합니다.
- 본 시스템은 투자 참고용이며 투자 권유가 아닙니다.
