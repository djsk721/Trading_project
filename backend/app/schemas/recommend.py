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
    sector: str = ""
    status_label: str = ""
    highlights: List[str] = Field(default_factory=list)
    metric_note: str = ""
    detail_summary: str = ""
    ai_summary: str = ""


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
    sector: str = ""
    status_label: str = ""
    highlights: List[str] = Field(default_factory=list)
    metric_note: str = ""
    detail_summary: str = ""
    ai_summary: str = ""


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


class HoldingExitItem(BaseModel):
    rank: int
    symbol: str
    name: str
    market: str = "KRX"
    decision: str = "hold"
    decision_label: str = "보유"
    risk_score: float = 0
    qty: float = 0
    price: float = 0
    avg_cost: float = 0
    cost_amount: float = 0
    cost_amount_krw: float = 0
    amount: float = 0
    amount_krw: float = 0
    profit: float = 0
    profit_krw: float = 0
    profit_rate: float = 0
    currency: str = "KRW"
    exchange_rate: float = 1
    portfolio_weight: float = 0
    rsi: float = 50
    macd_signal: str = "NEUTRAL"
    trend: str = "SIDEWAYS"
    sma20: float = 0
    sma60: float = 0
    sma120: float = 0
    bb_position: float = 0.5
    reasons: List[str] = Field(default_factory=list)
    highlights: List[str] = Field(default_factory=list)
    ai_summary: str = ""
    ai_rationale: str = ""
    ai_risk: str = ""
    ai_action: str = ""
    ai_watchpoints: List[str] = Field(default_factory=list)


class HoldingExitResponse(BaseModel):
    as_of: str
    items: List[HoldingExitItem] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)
    used_llm: bool = False
    provider: str = "none"
    cached: bool = False
    updated_at: Optional[str] = None
    disclaimer: str = "보유 종목 매도 점검은 참고 정보이며 투자 권유가 아닙니다."
