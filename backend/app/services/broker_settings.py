"""사용자 증권사 API 키 (메모리 오버레이). .env보다 우선합니다."""
from __future__ import annotations

import threading
from typing import Any, Optional

from app.core.config import get_settings

_lock = threading.Lock()
_kis: dict[str, Any] = {}
_toss: dict[str, Any] = {}
_active: str = "kis"  # kis | toss


def _mask(value: str, keep: int = 4) -> str:
    s = (value or "").strip()
    if not s:
        return ""
    if len(s) <= keep:
        return "·" * len(s)
    return s[:keep] + "·" * min(8, len(s) - keep)


def _normalize_broker(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if raw in {"kis", "korea", "한국투자", "한투", "koreainvestment"}:
        return "kis"
    if raw in {"toss", "tossinvest", "토스", "토스증권"}:
        return "toss"
    return None


def get_active_broker() -> str:
    with _lock:
        return _active


def set_active_broker(broker: str) -> str:
    global _active
    normalized = _normalize_broker(broker) or "kis"
    with _lock:
        _active = normalized
    return normalized


def set_user_keys(payload: dict) -> dict:
    global _kis, _toss
    kis = payload.get("kis") or {}
    toss = payload.get("toss") or {}
    keys_changed = False
    with _lock:
        if kis.get("clear"):
            _kis = {}
            keys_changed = True
        elif any(kis.get(k) for k in ("app_key", "app_secret", "account", "hts_id")):
            merged = dict(_kis)
            for key in ("hts_id", "app_key", "app_secret", "account"):
                val = str(kis.get(key) or "").strip()
                if val:
                    merged[key] = val
            if "virtual" in kis:
                merged["virtual"] = bool(kis.get("virtual"))
            elif "virtual" not in merged:
                merged["virtual"] = True
            _kis = merged
            keys_changed = True
        if toss.get("clear"):
            _toss = {}
            keys_changed = True
        elif any(toss.get(k) for k in ("client_id", "client_secret", "account")):
            merged = dict(_toss)
            for key in ("client_id", "client_secret", "account"):
                val = str(toss.get(key) or "").strip()
                if val:
                    merged[key] = val
            _toss = merged
            keys_changed = True
    active = _normalize_broker(payload.get("active"))
    if active:
        set_active_broker(active)
    if keys_changed:
        from app.services import kis_client
        from app.services import toss_client

        kis_client.reset_kis_instance()
        toss_client.reset_client()
    return status()


def get_kis_override() -> Optional[dict]:
    with _lock:
        if _kis.get("app_key") and _kis.get("app_secret") and _kis.get("account"):
            return dict(_kis)
        return None


def get_toss_override() -> Optional[dict]:
    """사용자 키 > .env. 둘 다 없으면 None."""
    with _lock:
        if _toss.get("client_id") and _toss.get("client_secret"):
            return dict(_toss)
    settings = get_settings()
    if settings.toss_client_id and settings.toss_client_secret:
        return {
            "client_id": settings.toss_client_id,
            "client_secret": settings.toss_client_secret,
            "account": settings.toss_account or "",
        }
    return None


def status() -> dict:
    settings = get_settings()
    override = get_kis_override()
    env_kis = bool(settings.kis_app_key and settings.kis_app_secret and settings.kis_account)
    env_toss = bool(settings.toss_client_id and settings.toss_client_secret)
    toss = get_toss_override()
    with _lock:
        kis_view = dict(_kis)
        toss_view = dict(_toss)
        active = _active
    user_toss = bool(toss_view.get("client_id") and toss_view.get("client_secret"))
    return {
        "active_broker": active,
        "kis_source": "user" if override else ("env" if env_kis else "none"),
        "kis_configured": bool(override or env_kis),
        "kis_virtual": bool(override["virtual"] if override else settings.kis_virtual),
        "kis_hts_id": (override or {}).get("hts_id") or settings.kis_hts_id or "",
        "kis_account_masked": _mask((override or {}).get("account") or settings.kis_account or ""),
        "kis_app_key_masked": _mask((override or {}).get("app_key") or settings.kis_app_key or ""),
        "toss_source": "user" if user_toss else ("env" if env_toss else "none"),
        "toss_configured": bool(toss),
        "toss_ready": bool(toss),
        "toss_client_id_masked": _mask(
            toss_view.get("client_id") or settings.toss_client_id or ""
        ),
        "toss_account_masked": _mask(
            toss_view.get("account") or settings.toss_account or ""
        ),
        "has_user_kis": bool(kis_view.get("app_key")),
        "has_user_toss": bool(toss_view.get("client_id")),
    }
