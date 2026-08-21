from typing import List, Optional

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    title: str
    title_original: str = ""
    summary: str = ""
    url: str = ""
    source: str = ""
    published_at: Optional[str] = None
    symbol: str = ""
    stock_name: str = ""
    importance: Optional[int] = None
    importance_reason: str = ""
    sentiment: str = ""
    sentiment_reason: str = ""
    has_ai_summary: bool = False


class NewsResponse(BaseModel):
    symbol: str
    stock_name: str = ""
    market: str = "KRX"
    items: List[NewsItem] = Field(default_factory=list)
    count: int = 0
    sort: str = "importance"
    preparing: bool = False


class MarketNewsItem(BaseModel):
    id: str = ""
    title: str
    title_original: str = ""
    summary: str = ""
    url: str = ""
    source: str = ""
    published_at: Optional[str] = None
    category: str = ""
    category_label: str = ""
    importance: Optional[int] = None
    importance_reason: str = ""
    sentiment: str = ""
    sentiment_reason: str = ""


class TabDigest(BaseModel):
    category: str = ""
    category_label: str = ""
    day: str = ""
    text: str = ""
    provider: str = ""
    source_count: int = 0
    updated_at: str = ""
    cached: bool = False
    ready: bool = False


class MarketNewsResponse(BaseModel):
    fetched_at: str = ""
    items: List[MarketNewsItem] = Field(default_factory=list)
    count: int = 0
    categories: List[dict] = Field(default_factory=list)
    preparing: bool = False
    digests: dict = Field(default_factory=dict)
    digest_preparing: bool = False
    digest_day: str = ""
    macros: dict = Field(default_factory=dict)
    sort: str = "importance"


class NewsSummarizeRequest(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    source: str = ""
    provider: str = ""
    force: bool = False


class NewsSummaryResponse(BaseModel):
    id: str = ""
    url: str = ""
    title: str = ""
    title_original: str = ""
    title_ko: str = ""
    source: str = ""
    summary_ko: str = ""
    importance: int = 0
    importance_reason: str = ""
    sentiment: str = ""
    sentiment_reason: str = ""
    provider: str = ""
    updated_at: str = ""
    cached: bool = False
    ttl_days: int = 14
