"""API routes for SEC Form 13F analytics."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query

from app.services import sec13f_service

router = APIRouter(prefix="/13f", tags=["sec13f"])


@router.post("/update")
async def update_13f(force: bool = False):
    try:
        return await asyncio.to_thread(sec13f_service.ensure_cache, force)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/dashboard")
def dashboard():
    return sec13f_service.dashboard()


@router.get("/managers")
def managers(q: str = "", limit: int = Query(50, ge=1, le=200)):
    return {"items": sec13f_service.search_managers(q, limit)}


@router.get("/managers/analysis")
def managers_analysis(q: str = "", limit: int = Query(30, ge=1, le=100)):
    return sec13f_service.managers_portfolio_analysis(q, limit)


@router.get("/managers/{cik}")
def manager_detail(cik: str, holding_q: str = "", limit: int = Query(150, ge=1, le=500)):
    try:
        return sec13f_service.manager_detail(cik, holding_q, limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/stocks/{ticker}")
def stock_detail(ticker: str):
    return sec13f_service.stock_detail(ticker)
