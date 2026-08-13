from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QuoteResponse(BaseModel):
    symbol: str
    name: str = ""
    market: str = ""
    price: float = 0
    change: float = 0
    rate: float = 0
    volume: float = 0
    amount: float = 0
    open: float = 0
    high: float = 0
    low: float = 0
    prev_price: float = 0


class OhlcvBar(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class ChartResponse(BaseModel):
    symbol: str
    market: str
    timeframe: str
    bars: List[OhlcvBar]
    indicators: Dict[str, Any] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)


class ChartRequest(BaseModel):
    symbol: str = "005930"
    market: str = "KRX"
    timeframe: str = "day"
    minute_interval: int = 1
    hour_interval: int = 1


class OrderBookLevel(BaseModel):
    price: float = 0
    volume: int = 0


class OrderBookResponse(BaseModel):
    ok: bool = False
    symbol: str = ""
    market: str = ""
    name: str = ""
    asks: List[OrderBookLevel] = Field(default_factory=list)
    bids: List[OrderBookLevel] = Field(default_factory=list)
    ask_volume: int = 0
    bid_volume: int = 0
    decimal_places: int = 0
    cached: bool = False
    rate_limited: bool = False
    message: str = ""
    as_of: float = 0
