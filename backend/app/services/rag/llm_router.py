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
    """chat은 프로바이더 전환, embed는 차원 일관성 위해 기본 설정을 따름."""

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def ollama_enabled(self) -> bool:
        return bool(getattr(self.settings, "ollama_enabled", False))

    def status(self) -> Dict[str, bool | str]:
        nvidia = get_nvidia()
        nvidia_ok = nvidia.health() if nvidia.configured else False
        ollama_ok = False
        if self.ollama_enabled:
            ollama_ok = get_ollama().health()
        provider = (self.settings.llm_provider or "nvidia").lower()
        if not self.ollama_enabled and provider in {"auto", "ollama", "local", "gemma"}:
            provider = "nvidia"
        return {
            "ollama_connected": ollama_ok,
            "ollama_enabled": self.ollama_enabled,
            "nvidia_connected": nvidia_ok,
            "ai_connected": bool(ollama_ok or nvidia_ok),
            "llm_provider": provider,
            "embed_provider": (self.settings.embed_provider or "nvidia").lower(),
        }

    def _normalize(self, provider: Optional[str]) -> str:
        mode = (provider or self.settings.llm_provider or "nvidia").strip().lower()
        if mode in {"local", "gemma", "ollama"}:
            mode = "ollama"
        elif mode in {"nim", "nvidia_api", "nvidia"}:
            mode = "nvidia"
        else:
            mode = "auto"
        # 테스트용: Ollama 잠금 시 무조건 NVIDIA
        if not self.ollama_enabled:
            if mode == "ollama":
                log.info("Ollama disabled; forcing NVIDIA provider")
            return "nvidia"
        return mode

    def _chat_order(self, provider: Optional[str] = None) -> List[str]:
        global _rr_counter
        mode = self._normalize(provider)
        nvidia = get_nvidia()
        has_nvidia = nvidia.configured
        allow_ollama = self.ollama_enabled

        if mode == "ollama":
            order = ["ollama"] if allow_ollama else []
            if has_nvidia and self.settings.llm_failover:
                order.append("nvidia")
            return order or (["nvidia"] if has_nvidia else [])

        if mode == "nvidia":
            order = ["nvidia"] if has_nvidia else []
            if allow_ollama and (self.settings.llm_failover or not order):
                order.append("ollama")
            return order

        # auto
        pool: List[str] = []
        if allow_ollama:
            pool.append("ollama")
        if has_nvidia:
            pool.append("nvidia")
        if not pool:
            return []
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
        order = self._chat_order(provider)
        if not order:
            raise RuntimeError("No LLM provider available (Ollama disabled, NVIDIA not configured)")
        for name in order:
            try:
                if name == "nvidia":
                    text = get_nvidia().chat(
                        messages,
                        temperature=temperature,
                        num_predict=num_predict,
                    )
                else:
                    if not self.ollama_enabled:
                        continue
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
        mode = (self.settings.embed_provider or "nvidia").strip().lower()
        nvidia = get_nvidia()
        if mode == "nvidia" and nvidia.api_key and nvidia.embed_model:
            return nvidia.embed(texts, input_type=input_type)
        if not self.ollama_enabled:
            raise RuntimeError("Embedding provider unavailable (Ollama disabled)")
        return get_ollama().embed(texts)

    def embed_one(self, text: str, *, input_type: str = "query") -> List[float]:
        mode = (self.settings.embed_provider or "nvidia").strip().lower()
        nvidia = get_nvidia()
        if mode == "nvidia" and nvidia.api_key and nvidia.embed_model:
            return nvidia.embed_one(text, input_type=input_type)
        if not self.ollama_enabled:
            raise RuntimeError("Embedding provider unavailable (Ollama disabled)")
        return get_ollama().embed_one(text)


_router: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
