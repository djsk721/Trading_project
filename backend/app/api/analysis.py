from fastapi import APIRouter, Query

from app.schemas.analysis import AnalysisRequest, AnalysisResponse, RuleAnalysisResponse
from app.services.rag.query_processor import QueryProcessor
from app.services.rule_analysis import build_rule_analysis

router = APIRouter(prefix="/analysis", tags=["analysis"])
_processor = QueryProcessor()


@router.get("/rules", response_model=RuleAnalysisResponse)
def rules(
    symbol: str = Query("005930"),
    market: str = Query("KRX"),
    stock_name: str = Query(""),
    days: int = Query(120, ge=30, le=400),
):
    """선택한 종목의 지표 기반 룰 분석 (LLM 없음)."""
    result = build_rule_analysis(
        symbol=symbol,
        market=market,
        stock_name=stock_name,
        days=days,
    )
    return RuleAnalysisResponse(**result)


@router.post("/ask", response_model=AnalysisResponse)
def ask(body: AnalysisRequest):
    result = _processor.analyze(
        symbol=body.symbol,
        query=body.query,
        analysis_type=body.analysis_type,
        market=body.market,
        stock_name=body.stock_name,
        days=body.days,
        provider=body.provider,
    )
    return AnalysisResponse(**result)
