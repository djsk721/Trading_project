from datetime import datetime
import threading
import uuid
from typing import Any

from fastapi import APIRouter, Query

from app.schemas.recommend import HoldingExitResponse, RecommendResponse
from app.services.holding_exit_service import build_holdings_exit
from app.services.recommendation_service import build_daily_recommendations, recommend_progress

router = APIRouter(prefix="/recommend", tags=["recommend"])
_JOBS: dict[str, dict[str, Any]] = {}


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


@router.get("/daily/progress")
def daily_progress():
    return recommend_progress()


@router.post("/daily/jobs")
def start_daily_job(
    market: str = Query("ALL", description="ALL | KRX | US"),
    top_n: int = Query(10, ge=1, le=30),
    provider: str = Query("", description="auto | ollama | nvidia (empty = server default)"),
    force: bool = Query(True, description="당일 캐시를 무시하고 추천을 다시 생성"),
    force_universe: bool = Query(False, description="유니버스 캐시 무시하고 재구축"),
):
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": None,
        "result": None,
        "error": "",
    }

    def _run() -> None:
        _JOBS[job_id].update({"status": "running", "updated_at": datetime.now().isoformat(timespec="seconds")})
        try:
            result = build_daily_recommendations(
                market=market,
                top_n=top_n,
                provider=provider,
                force=force,
                force_universe=force_universe,
            )
            _JOBS[job_id].update({
                "status": "done",
                "result": result,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
        except Exception as exc:
            _JOBS[job_id].update({
                "status": "failed",
                "error": str(exc),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })

    threading.Thread(target=_run, daemon=True).start()
    return _JOBS[job_id]


@router.get("/daily/jobs/{job_id}")
def get_daily_job(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        return {"job_id": job_id, "status": "not_found", "error": "job not found", "result": None}
    return {**job, "progress": recommend_progress()}


@router.get("/holdings-exit", response_model=HoldingExitResponse)
def holdings_exit(
    provider: str = Query("", description="auto | ollama | nvidia (empty = server default)"),
    force: bool = Query(False, description="당일 캐시를 무시하고 보유 점검을 다시 생성"),
    days: int = Query(160, ge=60, le=400),
):
    return HoldingExitResponse(**build_holdings_exit(provider=provider, force=force, days=days))
