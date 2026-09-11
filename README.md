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
- 보유 종목별 **매도·비중 축소·보유 점검**: 손익률, 포트폴리오 비중, RSI, MACD, 추세, 이동평균, Bollinger Band, AI 보강 근거
- 국내/해외 보유 평가금액은 원통화와 원화 환산 금액을 분리 표시
- UI에서 증권사 선택 후 백엔드 활성 브로커 동기화

### 시장·리서치
- 매크로 보드 / Market Pulse
- 시장 뉴스 브리핑 (한국·미국·세계·리스크·암호화폐)
- SEC 13F 대시보드 (매니저·종목 조회, 데이터 갱신)

### AI / RAG
- 기본 엔진: **Ollama** (`LLM_PROVIDER=ollama`, `OLLAMA_ENABLED=true`)
- 클라우드 대체: NVIDIA NIM (`LLM_PROVIDER=nvidia`)
- 로컬 모델: `gpt-oss:120b` / `gemma4:31b` + `embeddinggemma` (`.env` 기준)
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
| `LLM_PROVIDER`, `OLLAMA_ENABLED` | 기본 `ollama` / `true`. 클라우드는 `nvidia` |
| `NVIDIA_API_KEY` | 클라우드 AI |
| `TRADING_ENABLED`, `TEST_MODE` | 실제 주문 안전 플래그. 기본값은 실주문 차단 |
| `DIVIDEND_CACHE_TTL_SECONDS` | 배당 캘린더 캐시 TTL |

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
| 배당 | 보유·추천 종목 배당락 캘린더 |
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
| GET | `/api/recommend/daily/progress` | 추천 생성 진행 단계 |
| POST/GET | `/api/recommend/daily/jobs` | 추천 생성 비동기 Job 생성/조회 |
| GET | `/api/recommend/holdings-exit?force=false&days=160` | 보유 종목 매도·축소·보유 점검 |
| GET | `/api/calendar/dividends?from=&to=&scope=holdings\|recommend\|all` | 배당·배당락 캘린더 |
| GET | `/api/analysis/rules` | 룰 분석 |
| POST | `/api/analysis/ask` | AI 분석 |
| GET/POST | `/api/trading/*` | 계좌·주문 |
| GET | `/api/trading/capabilities` | 브로커별 지원 기능·주문 안전 상태 |
| GET | `/api/trading/diagnostics` | 브로커 연결 진단 |
| GET/POST | `/api/trading/orders/*` | 주문 가능 수량, 미체결, 이력, 취소, 정정 미지원 응답 |
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

### P0 — 보유 종목 매도 점검 (구현됨)
계좌 화면에서 보유 포트폴리오의 **매도·비중 축소·보유** 판단을 제공합니다.

| 항목 | 구현 내용 |
|------|-----------|
| 입력 | 활성 브로커 계좌 보유 종목 + 종목별 일봉 지표 |
| 출력 | 종목별 `sell` / `trim` / `hold`, 리스크 점수, 핵심 근거, AI 요약 |
| 규칙 | 손절/익절 구간, RSI 과열·약세, MACD, 추세, 20/60/120일 이동평균, Bollinger Band, 포트폴리오 집중도 |
| AI | 평단, 평가손익, 비중, 기술지표 근거만 사용해 보유 맥락을 보강 |
| UI | 계좌 워크스페이스의 **보유 매도 점검** 섹션 |
| API | `GET /api/recommend/holdings-exit?force=false&days=160` |
| 캐시 | 당일 결과 캐시. `force=true`는 분석 캐시만 갱신하고 계좌 강제조회는 하지 않아 KIS rate limit을 줄임 |
| 통화 | 해외 보유 종목은 USD 원통화 금액과 KRW 환산 금액을 분리. 비중 계산은 KRW 환산 기준 |

### P1 — 배당·배당락 캘린더 (MVP 구현됨)
보유·추천 종목과 연결되는 **배당·배당락 캘린더**를 제공합니다.

| 항목 | 구현 내용 |
|------|-----------|
| 데이터 | 미국 종목 yfinance calendar/dividends 우선. 국내는 확인된 데이터만 표시하고 추정하지 않음 |
| 범위 | 보유종목, 추천종목, 보유+추천, 직접입력 |
| UI | 배당 워크스페이스, 월별 카드, D-day 배지, 종목 클릭 시 데스크 연동 |
| API | `GET /api/calendar/dividends?from=&to=&scope=holdings\|recommend\|all&symbols=` |
| 캐시 | `DIVIDEND_CACHE_TTL_SECONDS` 기준 캐시, `force=true` 갱신 |

### P2 — 추천·데스크 고도화
- [x] 추천 카드에 상태·핵심 수치·근거 요약 표시
- [x] 스캔 전체 테이블에 `status_label`, `highlights` 데이터 제공
- [x] 일일 추천 실행 진행률(`universe → scoring → shortlist → news → AI`) UX/API
- [x] 추천 Job API 기반 비동기 실행/조회
- [ ] SSE/WebSocket 기반 실시간 진행률 전달 고도화

### P3 — 계좌·주문·운영
- [x] 주문 전 최종 확인 단계 (종목·매수/매도·수량·가격·예상 금액)
- [x] 주문 안전 플래그: `TRADING_ENABLED=false` 또는 `TEST_MODE=true`이면 실제 브로커 주문/취소 차단
- [x] 브로커 capability API 및 진단 화면
- [x] 주문 가능 금액/수량, 주문/체결 이력, 정정 API 기본 응답
- [ ] 미체결 일괄 관리, 체결 이력 요약
- [x] 백엔드/프론트 동시 기동 스크립트 (`scripts/dev.sh`)

### P4 — 데이터·품질
- [x] 주문 안전 테스트 골격 추가: 실제 KIS/Toss 주문 함수 호출 여부를 mock으로 검증
- [x] 추천/보유점검/배당 캐시 정책 문서화
- [ ] 국내 배당·공시 데이터 소스 표준화
- [ ] E2E 스모크 (health → quote → recommend → account)

---

## Notes

- `python-kis` 미설치 시 한투 계좌 조회가 실패합니다 (`pip install -r requirements.txt`).
- 토스: WTS에서 IP 허용, `TOSS_API_BASE_URL=https://openapi.tossinvest.com` 사용 (`TOSS_API_URL` 아님).
- 프론트 `:5173`만 실행한 상태에서 API 호출 시 `connect ECONNREFUSED 127.0.0.1:8010` → 백엔드 `:8010` 기동 필요.
- 자동화 테스트에서는 실제 주문을 보내지 않습니다. 기본 `.env.example`은 `TRADING_ENABLED=false`이며, `TEST_MODE=true`이면 매수/매도/취소가 브로커로 전달되지 않습니다.
