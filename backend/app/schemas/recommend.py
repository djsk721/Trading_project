from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendItem(BaseModel):
    rank: int
    symbol: str
    name: str
    market: str = "KRX"
    score: float
    price: float
    change_pct: float
    reasons: List[str] = Field(default_factory=list)
    rsi: float = 50.0
    macd_signal: str = "NEUTRAL"
    trend: str = "SIDEWAYS"
    stance: str = "watch"
    buy_price: float = 0
    sell_price: float = 0


class ScanItem(BaseModel):
    rank: int
    symbol: str
    name: str
    market: str = "KRX"
    score: float
    price: float
    change_pct: float
    rsi: float = 50.0
    macd_signal: str = "NEUTRAL"
    trend: str = "SIDEWAYS"
    reasons: List[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    as_of: str
    market: str = "ALL"
    items: List[RecommendItem] = Field(default_factory=list)
    scan_items: List[ScanItem] = Field(default_factory=list)
    universe_size: int = 0
    universe_source: str = ""
    shortlist_size: int = 0
    scanned_count: int = 0
    market_commentary: str = ""
    used_llm: bool = False
    provider: str = "none"
    model: str = "ai"
    cached: bool = False
    updated_at: Optional[str] = None
    disclaimer: str = (
        "AI·기술지표·뉴스 기반 일일 참고 추천입니다. "
        "권장 매수/매도가는 참고용이며 투자 조언이 아닙니다."
    )
