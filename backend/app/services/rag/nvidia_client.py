"""NVIDIA NIM / API Catalog 클라이언트 (OpenAI 호환)."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

from app.core.config import get_settings

log = logging.getLogger(__name__)


class NvidiaClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = (self.settings.nvidia_base_url or "").rstrip("/")
        self.api_key = (self.settings.nvidia_api_key or "").strip()
        self.llm_model = self.settings.nvidia_llm_model
        self.embed_model = self.settings.nvidia_embed_model
        self.timeout = self.settings.nvidia_timeout_sec

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.llm_model)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def health(self) -> bool:
        if not self.configured:
            return False
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(f"{self.base_url}/models", headers=self._headers())
                return resp.status_code < 500
        except Exception:
            return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        if not self.configured:
            raise RuntimeError("NVIDIA API is not configured (NVIDIA_API_KEY).")

        payload = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.llm_temperature,
            "max_tokens": num_predict or self.settings.max_new_tokens,
            "stream": False,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message") or {}
            return str(message.get("content") or "").strip()

    def _needs_input_type(self) -> bool:
        model = (self.embed_model or "").lower()
        # NeMo Retriever / Nemotron embed 계열은 query|passage 구분 필요
        markers = ("embedqa", "nemotron-embed", "llama-nemotron-embed", "nv-embed")
        return any(m in model for m in markers)

    def embed(
        self,
        texts: List[str],
        *,
        input_type: str = "passage",
    ) -> List[List[float]]:
        if not self.api_key or not self.embed_model:
            raise RuntimeError("NVIDIA embedding model is not configured.")
        vectors: List[List[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for text in texts:
                payload: Dict[str, object] = {
                    "model": self.embed_model,
                    "input": text,
                    "encoding_format": "float",
                }
                if self._needs_input_type():
                    payload["input_type"] = input_type if input_type in {"query", "passage"} else "passage"
                resp = client.post(
                    f"{self.base_url}/embeddings",
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                items = data.get("data") or []
                vectors.append((items[0].get("embedding") if items else []) or [])
        return vectors

    def embed_one(self, text: str, *, input_type: str = "query") -> List[float]:
        return self.embed([text], input_type=input_type)[0]


_client: Optional[NvidiaClient] = None


def get_nvidia() -> NvidiaClient:
    global _client
    if _client is None:
        _client = NvidiaClient()
    return _client
