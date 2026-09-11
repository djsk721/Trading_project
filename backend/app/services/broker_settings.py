"""사용자 증권사 API 키 (메모리 오버레이). .env보다 우선합니다."""
from __future__ import annotations

import threading
from typing import Any, Optional

from app.core.config import get_settings

_lock = threading.Lock()
_kis: dict[str, Any] = {}
_toss: dict[str, Any] = {}
_nvidia: dict[str, Any] = {}
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


def _is_kis_account(value: str) -> bool:
    digits = (value or "").replace("-", "").replace(" ", "")
    return digits.isdigit() and len(digits) in (8, 10)


def normalize_kis_login(hts_id: str, account: str) -> tuple[str, str]:
    """HTS ID와 계좌번호가 뒤바뀐 입력을 바로잡습니다."""
    hts = (hts_id or "").strip()
    acct = (account or "").replace(" ", "")
    if acct.startswith("@"):
        if not hts:
            hts = acct
        acct = ""
    elif acct and not _is_kis_account(acct):
        if not hts:
            hts = acct
        acct = ""
    if hts and _is_kis_account(hts.lstrip("@")) and not _is_kis_account(acct):
        acct = hts
        hts = ""
    return hts.lstrip("@").strip(), acct


def get_active_broker() -> str:
    with _lock:
        return _active


def set_active_broker(broker: str) -> str:
    global _active
    normalized = _normalize_broker(broker) or "kis"
    with _lock:
        _active = normalized
    return normalized


def active_kis_account() -> str:
    with _lock:
        acct = str(_kis.get("account") or "")
    return acct or get_settings().kis_account or ""


def active_kis_virtual() -> bool:
    # get_kis_override()가 같은 Lock을 쓰므로 여기서 잠그면 데드락이 납니다.
    override = get_kis_override()
    if override is not None:
        return bool(override["virtual"])
    return get_settings().kis_virtual


def set_user_keys(payload: dict) -> dict:
    global _kis, _toss, _nvidia
    kis = payload.get("kis") or {}
    toss = payload.get("toss") or {}
    nvidia = payload.get("nvidia") or {}
    keys_changed = False
    nvidia_changed = False
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
                    if key in ("app_key", "app_secret"):
                        merged[key] = "".join(val.replace("\ufeff", "").strip().strip('"').strip("'").split())
            if "virtual" in kis:
                merged["virtual"] = bool(kis.get("virtual"))
            elif "virtual" not in merged:
                merged["virtual"] = True
            hts, acct = normalize_kis_login(
                str(merged.get("hts_id") or ""),
                str(merged.get("account") or ""),
            )
            if hts:
                merged["hts_id"] = hts
            if acct:
                merged["account"] = acct
            elif not _is_kis_account(str(merged.get("account") or "")):
                merged.pop("account", None)
            if merged != _kis:
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
            if merged != _toss:
                _toss = merged
                keys_changed = True
        if nvidia.get("clear"):
            if _nvidia:
                _nvidia = {}
                nvidia_changed = True
        else:
            api_key = str(nvidia.get("api_key") or "").strip()
            if api_key and _nvidia.get("api_key") != api_key:
                _nvidia = {"api_key": api_key}
                nvidia_changed = True
    active = _normalize_broker(payload.get("active"))
    if active:
        set_active_broker(active)
    if keys_changed:
        from app.services import kis_client
        from app.services import toss_client

        kis_client.reset_kis_instance()
        toss_client.reset_client()
    if nvidia_changed:
        from app.services.rag import nvidia_client

        nvidia_client.reset_nvidia_instance()
    return status()


def get_kis_override() -> Optional[dict]:
    """로그인/UI 키를 .env 빈 칸과 합쳐 계좌 조회에 사용합니다."""
    settings = get_settings()
    with _lock:
        user = dict(_kis)
    app_key = "".join(str(user.get("app_key") or "").split()) or settings.kis_app_key
    app_secret = "".join(str(user.get("app_secret") or "").split()) or settings.kis_app_secret
    if not (user.get("app_key") or user.get("app_secret")):
        return None
    if not (app_key and app_secret):
        return None
    hts_id, account = normalize_kis_login(
        str(user.get("hts_id") or ""),
        str(user.get("account") or ""),
    )
    hts_id = hts_id or (settings.kis_hts_id or "").lstrip("@").strip()
    account = account or settings.kis_account
    virtual = user["virtual"] if "virtual" in user else settings.kis_virtual
    return {
        "hts_id": hts_id or "user",
        "app_key": app_key,
        "app_secret": app_secret,
        "account": account,
        "virtual": bool(virtual),
    }


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


def get_nvidia_override() -> Optional[str]:
    """사용자가 입력한 NVIDIA 키. 없으면 None."""
    with _lock:
        api_key = _nvidia.get("api_key")
        return str(api_key).strip() if api_key else None


def status() -> dict:
    settings = get_settings()
    override = get_kis_override()
    env_kis = bool(settings.kis_app_key and settings.kis_app_secret and settings.kis_account)
    env_toss = bool(settings.toss_client_id and settings.toss_client_secret)
    toss = get_toss_override()
    nvidia_override = get_nvidia_override()
    env_nvidia = bool(settings.nvidia_api_key)
    with _lock:
        kis_view = dict(_kis)
        toss_view = dict(_toss)
        nvidia_view = dict(_nvidia)
        active = _active
    user_toss = bool(toss_view.get("client_id") and toss_view.get("client_secret"))
    user_nvidia = bool(nvidia_view.get("api_key"))
    out = {
        "active_broker": active,
        "kis_source": "user" if kis_view.get("app_key") else ("env" if env_kis else "none"),
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
        "nvidia_source": "user" if user_nvidia else ("env" if env_nvidia else "none"),
        "nvidia_configured": bool(user_nvidia or env_nvidia),
        "nvidia_api_key_masked": _mask(
            nvidia_view.get("api_key") or settings.nvidia_api_key or ""
        ),
        "has_user_nvidia": user_nvidia,
    }
    return out