from fastapi import APIRouter, Query

from app.schemas.news import (
    MarketNewsResponse,
    NewsResponse,
    NewsSummarizeRequest,
    NewsSummaryResponse,
)
from app.services.market_news_service import (
    market_news_with_prepare,
    summarize_news_item,
)
from app.services.news_service import fetch_news_with_prepare

router = APIRouter(prefix="/news", tags=["news"])


@router.get("", response_model=NewsResponse)
def news(
    symbol: str = Query("005930"),
    market: str = Query("KRX"),
    stock_name: str | None = Query(None),
    prepare: bool = Query(True, description="미요약 기사 백그라운드 요약·번역"),
    provider: str = Query("", description="auto|ollama|nvidia"),
    sort: str = Query("importance", description="importance|date"),
):
    """종목별 뉴스 (한글 제목·중요도 캐시 반영, 기본 중요도 정렬)."""
    payload = fetch_news_with_prepare(
        symbol=symbol,
        market=market,
        stock_name=stock_name,
        prepare=prepare,
        provider=provider,
        sort=sort,
    )
    return NewsResponse(**payload)


@router.get("/market", response_model=MarketNewsResponse)
def market_news(
    prepare: bool = Query(True, description="미요약 기사 AI 요약 백그라운드 시작"),
    force: bool = Query(False, description="목록 캐시 무시하고 재수집"),
    provider: str = Query("", description="auto|ollama|nvidia"),
    sort: str = Query("importance", description="importance|date"),
):
    """시황 뉴스 + 탭별 금일 시황 정리 (한글 제목·중요도·정렬)."""
    payload = market_news_with_prepare(
        prepare=prepare,
        provider=provider,
        force=force,
        sort=sort,
    )
    return MarketNewsResponse(**payload)


@router.post("/summarize", response_model=NewsSummaryResponse)
def summarize(body: NewsSummarizeRequest):
    """뉴스 AI 한글 요약·제목 번역·중요도 (캐시 14일)."""
    result = summarize_news_item(
        url=body.url,
        title=body.title,
        snippet=body.snippet,
        source=body.source,
        provider=body.provider,
        force=body.force,
    )
    return NewsSummaryResponse(**result)
