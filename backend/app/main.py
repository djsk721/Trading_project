"""FastAPI entrypoint."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analysis, market, news, recommend, trading
from app.core.config import get_settings
from app.schemas.analysis import HealthResponse
from app.services import kis_client
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


@app.get("/api/health", response_model=HealthResponse)
def health():
    status = get_llm_router().status()
    # UI에는 상세 모델명을 노출하지 않고 연결 상태만 전달
    return HealthResponse(
        status="ok",
        kis_connected=kis_client.is_kis_connected(),
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
