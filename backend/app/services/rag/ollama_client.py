"""Ollama HTTP 클라이언트 (gemma4:31b / embeddings)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import get_settings

log = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")
        self.llm_model = self.settings.ollama_llm_model
        self.embed_model = self.settings.ollama_embed_model
        self.timeout = self.settings.ollama_timeout_sec

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                return [m.get("name", "") for m in models]
        except Exception as e:
            log.warning("list models failed: %s", e)
            return []

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.settings.llm_temperature,
                "num_predict": num_predict or self.settings.max_new_tokens,
            },
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return (data.get("message") or {}).get("content", "").strip()

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for text in texts:
                resp = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                )
                resp.raise_for_status()
                vectors.append(resp.json().get("embedding", []))
        return vectors

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


_client: Optional[OllamaClient] = None


def get_ollama() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
