from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.broker_settings import set_user_keys, status

router = APIRouter(prefix="/settings", tags=["settings"])


class KisKeysIn(BaseModel):
    hts_id: str = ""
    app_key: str = ""
    app_secret: str = ""
    account: str = ""
    virtual: bool = True
    clear: bool = False


class TossKeysIn(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    account: str = ""
    clear: bool = False


class NvidiaKeysIn(BaseModel):
    api_key: str = ""
    clear: bool = False


class BrokerKeysIn(BaseModel):
    kis: KisKeysIn = Field(default_factory=KisKeysIn)
    toss: TossKeysIn = Field(default_factory=TossKeysIn)
    nvidia: NvidiaKeysIn = Field(default_factory=NvidiaKeysIn)
    active: Optional[str] = None


@router.get("/broker")
def broker_status():
    return status()


@router.post("/broker")
def save_broker_keys(body: BrokerKeysIn):
    # 미전송 필드는 기본 빈 값으로 덮어쓰지 않음 (로그인 키 유지)
    return set_user_keys(body.model_dump(exclude_unset=True))
