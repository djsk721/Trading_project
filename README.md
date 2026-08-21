# Trading Desk

FastAPI + React 기반 주식 분석·거래 데스크입니다.  
국내(KRX)·해외(US) 시세, AI 일일 추천, 계좌/주문, 뉴스·매크로, SEC 13F를 한 화면에서 다룹니다.

> **투자 참고용**이며 투자 권유가 아닙니다. 주문 API는 실제 체결될 수 있으니 모의투자(`KIS_VIRTUAL=true`)를 권장합니다.

---

## Features

### 종목 데스크
- 시세·차트 (일봉/분봉), Lightweight Charts
- 기술적 지표 요약 (RSI, MACD, 추세 등)
- 호가창 MVP: 호가 클릭 → 지정가 매수/매도 가격 반영
- AI 분석 (기본 / 20일 예측 / 투자전략) + 룰 기반 체크리스트
- 종목 뉴스 (Google News RSS, yfinance 폴백) 및 AI 요약

### 일일 추천
- KRX + US 통합 스캔 (`market=ALL`) 또는 시장별 스캔
- 기술 스코어링 → 상위 shortlist → AI 브리핑
- 카드 UI: **상태 배지 · KPI(등락/RSI/점수) · 근거 2줄 · 상세 접기 · 차트 CTA**
- 권장 매수/매도가, 당일 캐시(force로 갱신)

### 내 계좌
- 브로커: **한국투자(KIS/pykis)**, **토스증권** Open API
- 잔고·보유·미체결·주문/취소
- 원화/달러 보유 현금 표시
- UI에서 증권사 선택 후 백엔드 활성 브로커 동기화

### 시장·리서치
- 매크로 보드 / Market Pulse
- 시장 뉴스 브리핑
- SEC 13F 대시보드 (매니저·종목 조회, 데이터 갱신)

### AI / RAG
- 기본 엔진: **NVIDIA NIM** (`LLM_PROVIDER=nvidia`)
- 로컬: Ollama (`OLLAMA_ENABLED=true` 시, `gemma4:31b` + `embeddinggemma`)
- BM25 + 임베딩 앙상블 검색 (설정으로 가중치·k 조절)

---

## Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI, pandas, python-kis, pykrx, yfinance, httpx |
| Frontend | React + Vite + TypeScript + lightweight-charts |
| LLM | NVIDIA NIM (기본) / Ollama (선택) |
| Brokers | 한국투자증권(KIS), 토스증권 Open API |

---

## Setup

### 1) Environment

```bash
cp .env.example .env
# KIS / Toss / NVIDIA_API_KEY 등 채우기
```

주요 변수:

| 변수 | 설명 |
|------|------|
| `KIS_*`, `KIS_VIRTUAL`, `KIS_AUTH_PATH` | 한투 연동 (실전: `KIS_VIRTUAL=false`) |
| `TOSS_CLIENT_ID`, `TOSS_CLIENT_SECRET`, `TOSS_API_BASE_URL` | 토스 Open API |
| `LLM_PROVIDER`, `OLLAMA_ENABLED` | `nvidia` 기본, Ollama는 `OLLAMA_ENABLED=true` |
| `NVIDIA_API_KEY` | 클라우드 AI |

### 2) Backend

```bash
cd backend
# conda 환경 권장 예: conda activate test
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8010
```

- API docs: http://localhost:8010/docs  
- Vite 프록시 `/api` → `127.0.0.1:8010` — **프론트만 켜면 `ECONNREFUSED`가 납니다. 백엔드를 반드시 함께 실행하세요.**

### 3) Frontend

```bash
cd frontend
npm install
npm run dev
```

UI: http://localhost:5173

### 4) Ollama (선택)

```bash
# .env: OLLAMA_ENABLED=true, LLM_PROVIDER=ollama 또는 auto
ollama pull gemma4:31b
ollama pull embeddinggemma
ollama serve
```

---

## Workspaces (UI)

| 화면 | 내용 |
|------|------|
| 데스크 | 종목·차트·호가·주문·AI 분석·뉴스 |
| 추천 | 일일 추천 / 스캔 보드 (상위 20 카드) |
| 계좌 | 잔고·보유·현금·주문 |
| 13F | SEC 13F 매니저·종목 분석 |

---

## Main API

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/health` | 헬스·브로커 힌트 |
| GET | `/api/market/chart` | 차트 |
| GET | `/api/market/quote/{symbol}` | 시세 |
| GET | `/api/market/orderbook/{symbol}` | 호가 |
| GET | `/api/market/macros` | 매크로 |
| GET | `/api/market/popular` | 인기 종목 |
| GET | `/api/news` | 종목 뉴스 |
| GET | `/api/news/market` | 시장 뉴스 |
| POST | `/api/news/summarize` | 뉴스 AI 요약 |
| GET | `/api/recommend/daily?market=ALL\|KRX\|US` | 일일 추천 (`force=true` 갱신) |
| GET | `/api/analysis/rules` | 룰 분석 |
| POST | `/api/analysis/ask` | AI 분석 |
| GET/POST | `/api/trading/*` | 계좌·주문 |
| GET/POST | `/api/settings/broker` | 활성 브로커 |
| GET/POST | `/api/13f/*` | SEC 13F |

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

---

## Project layout

```text
backend/app/
  api/           # market, trading, analysis, news, recommend, 13f, settings
  services/      # KIS, Toss, market data, recommend, RAG, news, 13f
  schemas/       # Pydantic models
frontend/src/
  components/    # Chart, OrderBook, RecBriefCard, Account, 13F, Macro...
  App.tsx        # workspace shell
.env.example
```

---

## TODO

우선순위는 **보유 리스크 → 이벤트 캘린더 → UX/데이터 품질** 순으로 잡았습니다.

### P0 — 보유 종목 매도 추천
현재 일일 추천은 **매수/신규 진입** 중심입니다. 보유 포트폴리오에 대한 **매도·관망·분할 매도** 가이드가 없습니다.

| 항목 | 설계 초안 |
|------|-----------|
| 입력 | `/api/trading/holdings` + 종목별 시세·지표·목표가(매수가 대비) |
| 출력 | 종목별 `sell` / `hold` / `trim` + 근거 2줄 + 상세 접기 (기존 RecBrief 카드 패턴 재사용) |
| 규칙 | 손절/익절 임계, RSI 과열, 추세 이탈, 브로커 목표가 대비 괴리 |
| AI | 보유 맥락(평단·평가손익·비중)을 프롬프트에 넣어 서술 |
| UI | 계좌 또는 추천 워크스페이스에 **「보유 매도 점검」** 섹션 |
| API | `GET /api/recommend/holdings-exit` (당일 캐시 + force) |

### P1 — 배당·배당락 캘린더
최신 배당·배당락일을 한눈에 보기 어렵습니다. 스캔·보유와 연결되는 **이벤트 캘린더**가 필요합니다.

| 항목 | 설계 초안 |
|------|-----------|
| 데이터 | 배당락일, 지급일, 배당금/수익률 (yfinance 등 → 국내 소스 보강) |
| 범위 | 관심/보유/추천 shortlist 우선, 이후 유니버스 확장 |
| UI | 월간 캘린더 + D-day 리스트, 데스크에서 종목 클릭 연동 |
| API | `GET /api/calendar/dividends?from=&to=&scope=holdings\|universe` |
| 알림(후속) | 배당락 D-3 배지·토스트 (푸시는 선택) |

### P2 — 추천·데스크 고도화
- [ ] 추천 카드에 목표가 달성률·손익비 요약 칩
- [ ] 스캔 전체 테이블에 highlights / status 컬럼
- [ ] 일일 추천 실행 진행률(universe → shortlist → AI) UX
- [ ] Ollama 재활성 시 UI에서 엔진 전환 복구

### P3 — 계좌·주문·운영
- [ ] 주문 전 리스크 확인 모달 (수량·예상 금액·모의/실전 배지)
- [ ] 미체결 일괄 관리, 체결 이력 요약
- [ ] 브로커 연결 진단 화면 (`broker_hint` 상세화)
- [ ] 백엔드/프론트 동시 기동 스크립트 (`scripts/dev.sh`)

### P4 — 데이터·품질
- [ ] 국내 배당·공시 데이터 소스 표준화
- [ ] 추천/차트 캐시 무효화 정책 문서화
- [ ] E2E 스모크 (health → quote → recommend → account)

---

## Notes

- `python-kis` 미설치 시 한투 계좌 조회가 실패합니다 (`pip install -r requirements.txt`).
- 토스: WTS에서 IP 허용, `TOSS_API_BASE_URL=https://openapi.tossinvest.com` 사용 (`TOSS_API_URL` 아님).
- 프론트 `:5173`만 실행한 상태에서 API 호출 시 `connect ECONNREFUSED 127.0.0.1:8010` → 백엔드 `:8010` 기동 필요.
