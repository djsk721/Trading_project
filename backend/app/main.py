"""FastAPI entrypoint."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analysis, market, news, recommend, sec13f, settings as settings_api, trading
from app.core.config import get_settings
from app.schemas.analysis import HealthResponse
from app.services import broker, kis_client
from app.services.rag.llm_router import get_llm_router

logging.basicConfig(level=logging.INFO)
settings = get_settings()

app = FastAPI(title=settings.app_name, version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market.router, prefix="/api")
app.include_router(trading.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(news.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")
app.include_router(sec13f.router, prefix="/api")
app.include_router(settings_api.router, prefix="/api")


@app.on_event("startup")
async def startup_sec13f_cache_check():
    import asyncio

    from app.services import sec13f_service

    async def _check():
        try:
            await asyncio.to_thread(sec13f_service.ensure_cache, False)
        except Exception:
            logging.exception("SEC 13F startup cache check failed")

    asyncio.create_task(_check())


@app.get("/api/health", response_model=HealthResponse)
def health():
    from app.services.kis_client import _kis_setup_hint

    status = get_llm_router().status()
    broker_ok = broker.is_connected()
    hint = ""
    if not broker_ok:
        if broker.active_broker() == "kis":
            hint = _kis_setup_hint() if not kis_client.is_kis_connected() else ""
        elif not broker.is_connected():
            hint = "토스증권 API 키 또는 IP 허용 설정을 확인하세요."
    # UI에는 상세 모델명을 노출하지 않고 연결 상태만 전달
    return HealthResponse(
        status="ok",
        kis_connected=kis_client.is_kis_connected(),
        broker_connected=broker_ok,
        active_broker=broker.active_broker(),
        broker_hint=hint,
        ollama_connected=bool(status["ollama_connected"]),
        nvidia_connected=bool(status["nvidia_connected"]),
        ai_connected=bool(status["ai_connected"]),
        llm_provider=str(status["llm_provider"]),
        llm_model="configured",
        embed_model="configured",
    )


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": "/api/health",
    }
