from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    symbol: str = "005930"
    stock_name: str = ""
    market: str = "KRX"
    query: str
    analysis_type: str = Field(
        "basic",
        description="basic | forecast_20d | strategy",
    )
    days: int = 120
    provider: str = Field(
        "",
        description="auto | ollama | nvidia (empty = use server default)",
    )


class RetrievedDoc(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RuleItem(BaseModel):
    id: str
    title: str
    detail: str
    direction: str
    weight: float = 0


class RuleAnalysisResponse(BaseModel):
    symbol: str
    market: str
    stock_name: str = ""
    as_of: str = ""
    price: float = 0
    score: float = 0
    stance: str = "watch"
    bias: str = "neutral"
    summary_text: str = ""
    signals: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    rules: List[RuleItem] = Field(default_factory=list)
    horizon: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    answer: str
    analysis_type: str
    symbol: str
    stock_name: str
    current_price: float = 0
    current_date: str = ""
    sources: List[RetrievedDoc] = Field(default_factory=list)
    model: str = "ai"
    provider: str = ""
    rule_analysis: Optional[RuleAnalysisResponse] = None


class HealthResponse(BaseModel):
    status: str
    kis_connected: bool = False
    ollama_connected: bool = False
    nvidia_connected: bool = False
    ai_connected: bool = False
    llm_provider: str = "auto"
    llm_model: str = ""
    embed_model: str = ""
