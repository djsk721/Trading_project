"""애플리케이션 설정 (.env 기반)."""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            str(PROJECT_ROOT / ".env"),
            str(BACKEND_ROOT / ".env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Trading Analysis API"
    app_env: str = "development"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # KIS
    kis_hts_id: str = ""
    kis_app_key: str = ""
    kis_app_secret: str = ""
    kis_account: str = ""
    kis_virtual: bool = True
    kis_auth_path: str = "secret.json"

    # LLM provider: auto | ollama | nvidia
    # auto = round-robin between available providers with failover
    llm_provider: str = "auto"
    # Embedding provider: ollama | nvidia (keep one for vector dim consistency)
    embed_provider: str = "nvidia"
    llm_failover: bool = True

    # Ollama / local
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "gemma4:31b"
    ollama_embed_model: str = "embeddinggemma"
    ollama_timeout_sec: int = 180
    max_new_tokens: int = 2048
    llm_temperature: float = 0.3

    # NVIDIA NIM / API Catalog (OpenAI-compatible)
    # LLM: openai/gpt-oss-120b
    # Embed: nvidia/llama-nemotron-embed-1b-v2 (RAG retrieval companion)
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_llm_model: str = "openai/gpt-oss-120b"
    nvidia_embed_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    nvidia_timeout_sec: int = 180

    # RAG
    bm25_k: int = 5
    faiss_k: int = 5
    bm25_weight: float = 0.7
    faiss_weight: float = 0.3
    max_docs: int = 8
    max_context_length: int = 3500
    max_doc_types_per_type: int = 3

    # Market defaults
    default_ticker: str = "005930"
    default_exchange_rate: float = 1300.0
    cache_ttl_seconds: int = 3600

    # News
    news_max_items: int = 20
    news_language: str = "ko"

    # Recommendations
    recommend_top_n: int = 10
    recommend_lookback_days: int = 60
    recommend_universe_size: int = 100
    recommend_universe_head: int = 80  # 시총/대금 각각 상위 N
    recommend_shortlist_size: int = 20
    recommend_universe_ttl_seconds: int = 86400
    recommend_scan_workers: int = 8

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def resolved_kis_auth_path(self) -> Path:
        path = Path(self.kis_auth_path)
        if path.is_absolute():
            return path
        for base in (PROJECT_ROOT, BACKEND_ROOT, Path.cwd()):
            candidate = base / path
            if candidate.exists():
                return candidate
        return PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
