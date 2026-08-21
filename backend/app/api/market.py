from fastapi import APIRouter, Query
from urllib.parse import unquote

from app.schemas.market import ChartResponse, OrderBookResponse, QuoteResponse
from app.services import broker
from app.services.macro_snapshots import fetch_macro_snapshots
from app.services.market_data import POPULAR_KR, POPULAR_US, build_chart_payload, resolve_stock_name
from app.services.symbol_utils import is_plausible_symbol, normalize_symbol

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/macros")
def macros(force: bool = Query(False, description="캐시 무시하고 재조회")):
    """주요 매크로 스냅샷 (KOSPI, S&P, 환율, WTI, VIX, 10Y 등)."""
    return fetch_macro_snapshots(force=force)


@router.get("/popular")
def popular_stocks(market: str = Query("KRX")):
    data = POPULAR_KR if market.upper() == "KRX" else POPULAR_US
    return {
        "market": market.upper(),
        "items": [{"symbol": s, "name": n} for s, n in data.items()],
    }


@router.get("/quote/{symbol:path}", response_model=QuoteResponse)
def quote(symbol: str, market: str = Query("")):
    raw = unquote(symbol or "").strip()
    key = normalize_symbol(raw, market)
    if not key or not is_plausible_symbol(key, market):
        # 잘못된 입력은 증권사 API를 치지 않음
        return QuoteResponse(symbol=raw[:80], name=raw[:80] if raw else "")
    data = broker.get_stock_quote(key, market=market)
    if not data:
        name = resolve_stock_name(key, market=market or ("KRX" if key.isdigit() else "US"))
        return QuoteResponse(symbol=key, name=name)
    return QuoteResponse(**data)


@router.get("/orderbook/{symbol:path}", response_model=OrderBookResponse)
def orderbook(symbol: str, market: str = Query("")):
    """활성 증권사 호가창 (국내 10호가 / 해외 10호가). 패널 폴링용 짧은 캐시."""
    raw = unquote(symbol or "").strip()
    key = normalize_symbol(raw, market)
    mkt = (market or ("KRX" if (key or "").isdigit() else "US")).upper()
    if not key or not is_plausible_symbol(key, market):
        return OrderBookResponse(ok=False, symbol=raw[:80], market=mkt, message="Invalid symbol")
    data = broker.get_stock_orderbook(key, market=mkt)
    return OrderBookResponse(**data)


@router.get("/chart", response_model=ChartResponse)
def chart(
    symbol: str = Query("005930"),
    market: str = Query("KRX"),
    timeframe: str = Query("day"),
    minute_interval: int = Query(1, ge=1, le=60),
    hour_interval: int = Query(1, ge=1, le=24),
    # 분봉(yfinance)은 7일, 일봉은 최대 500일까지 허용
    days: int = Query(180, ge=1, le=500),
):
    payload = build_chart_payload(
        symbol=symbol,
        market=market,
        timeframe=timeframe,
        minute_interval=minute_interval,
        hour_interval=hour_interval,
        days=days,
    )
    return ChartResponse(**payload)
