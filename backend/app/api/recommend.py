from fastapi import APIRouter, Query

from app.schemas.recommend import RecommendResponse
from app.services.recommendation_service import build_daily_recommendations

router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.get("/daily", response_model=RecommendResponse)
def daily(
    market: str = Query("ALL", description="ALL | KRX | US"),
    top_n: int = Query(10, ge=1, le=30),
    provider: str = Query("", description="auto | ollama | nvidia (empty = server default)"),
    force: bool = Query(False, description="당일 캐시를 무시하고 추천을 다시 생성"),
    force_universe: bool = Query(False, description="유니버스 캐시 무시하고 재구축"),
):
    return RecommendResponse(
        **build_daily_recommendations(
            market=market,
            top_n=top_n,
            provider=provider,
            force=force,
            force_universe=force_universe,
        )
    )
