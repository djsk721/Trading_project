from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BalanceResponse(BaseModel):
    total_amount: int = 0
    deposit: int = 0
    stocks_value: int = 0
    profit_loss: int = 0
    profit_loss_rate: float = 0.0
    market_value: int = 0


class HoldingItem(BaseModel):
    name: str = ""
    symbol: str = ""
    market: str = ""
    qty: float = 0
    price: float = 0
    amount: float = 0
    profit: float = 0
    profit_rate: float = 0.0
    currency: str = "KRW"
    scope: str = "domestic"


class DepositItem(BaseModel):
    currency: str
    amount: float = 0
    exchange_rate: float = 1
    amount_krw: float = 0
    scope: str = "domestic"


class MarketBook(BaseModel):
    deposit_krw: float = 0
    deposit_usd: float = 0
    stocks_value: float = 0
    stocks_value_krw: float = 0
    exchange_rate: float = 0
    holdings: List[HoldingItem] = Field(default_factory=list)


class AccountOverviewResponse(BaseModel):
    connected: bool = False
    account: str = ""
    virtual: bool = False
    total_eval_krw: float = 0
    purchase_amount: float = 0
    current_amount: float = 0
    profit_loss: float = 0
    profit_loss_rate: float = 0
    deposits: List[DepositItem] = Field(default_factory=list)
    domestic: MarketBook = Field(default_factory=MarketBook)
    overseas: MarketBook = Field(default_factory=MarketBook)
    holdings: List[HoldingItem] = Field(default_factory=list)
    error: Optional[str] = None


class OrderRequest(BaseModel):
    symbol: str
    side: str = Field(..., description="buy or sell")
    qty: float = 1
    price: Optional[float] = None
    order_type: str = Field("market", description="market or limit")


class OrderResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)


class PendingOrder(BaseModel):
    name: str = ""
    symbol: str = ""
    side: str = ""
    qty: float = 0
    price: float = 0
    pending_qty: int = 0
    order_id: str = ""


class CancelOrderRequest(BaseModel):
    symbol: str
    order_id: Optional[str] = None
