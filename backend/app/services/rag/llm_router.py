"""Ollama / NVIDIA LLM 라우터 (선택·번갈아 사용·실패 시 폴백)."""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

from app.core.config import get_settings
from app.services.rag.nvidia_client import get_nvidia
from app.services.rag.ollama_client import get_ollama

log = logging.getLogger(__name__)

_rr_lock = threading.Lock()
_rr_counter = 0


class LLMRouter:
    """chat은 프로바이더 전환, embed는 차원 일관성 위해 기본 Ollama 고정."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def status(self) -> Dict[str, bool | str]:
        ollama = get_ollama()
        nvidia = get_nvidia()
        ollama_ok = ollama.health()
        nvidia_ok = nvidia.health() if nvidia.configured else False
        return {
            "ollama_connected": ollama_ok,
            "nvidia_connected": nvidia_ok,
            "ai_connected": bool(ollama_ok or nvidia_ok),
            "llm_provider": (self.settings.llm_provider or "auto").lower(),
            "embed_provider": (self.settings.embed_provider or "ollama").lower(),
        }

    def _normalize(self, provider: Optional[str]) -> str:
        mode = (provider or self.settings.llm_provider or "auto").strip().lower()
        if mode in {"local", "gemma", "ollama"}:
            return "ollama"
        if mode in {"nim", "nvidia_api", "nvidia"}:
            return "nvidia"
        return "auto"

    def _chat_order(self, provider: Optional[str] = None) -> List[str]:
        global _rr_counter
        mode = self._normalize(provider)
        nvidia = get_nvidia()
        has_nvidia = nvidia.configured
        if mode == "ollama":
            order = ["ollama"]
            if has_nvidia and self.settings.llm_failover:
                order.append("nvidia")
            return order
        if mode == "nvidia":
            order = ["nvidia"] if has_nvidia else []
            if self.settings.llm_failover or not order:
                order.append("ollama")
            return order or ["ollama"]

        # auto: 가용 프로바이더를 라운드로빈으로 시작점 변경
        pool = ["ollama"]
        if has_nvidia:
            pool.append("nvidia")
        if len(pool) == 1:
            return pool
        with _rr_lock:
            start = _rr_counter % len(pool)
            _rr_counter += 1
        return pool[start:] + pool[:start]

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Returns:
            (answer, provider_name)
        """
        errors: List[str] = []
        for name in self._chat_order(provider):
            try:
                if name == "nvidia":
                    text = get_nvidia().chat(
                        messages,
                        temperature=temperature,
                        num_predict=num_predict,
                    )
                else:
                    text = get_ollama().chat(
                        messages,
                        temperature=temperature,
                        num_predict=num_predict,
                    )
                if text:
                    log.info("LLM chat via %s", name)
                    return text, name
                errors.append(f"{name}: empty response")
            except Exception as e:
                log.warning("LLM chat failed via %s: %s", name, e)
                errors.append(f"{name}: {e}")
        raise RuntimeError("All LLM providers failed: " + " | ".join(errors))

    def embed(self, texts: List[str], *, input_type: str = "passage") -> List[List[float]]:
        mode = (self.settings.embed_provider or "ollama").strip().lower()
        nvidia = get_nvidia()
        if mode == "nvidia" and nvidia.api_key and nvidia.embed_model:
            return nvidia.embed(texts, input_type=input_type)
        return get_ollama().embed(texts)

    def embed_one(self, text: str, *, input_type: str = "query") -> List[float]:
        mode = (self.settings.embed_provider or "ollama").strip().lower()
        nvidia = get_nvidia()
        if mode == "nvidia" and nvidia.api_key and nvidia.embed_model:
            return nvidia.embed_one(text, input_type=input_type)
        return get_ollama().embed_one(text)


_router: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
