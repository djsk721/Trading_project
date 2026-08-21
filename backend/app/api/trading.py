from fastapi import APIRouter

from app.schemas.trading import (
    AccountOverviewResponse,
    BalanceResponse,
    CancelOrderRequest,
    HoldingItem,
    OrderRequest,
    OrderResponse,
    PendingOrder,
)
from app.services import broker

router = APIRouter(prefix="/trading", tags=["trading"])


@router.get("/account", response_model=AccountOverviewResponse)
def account_overview():
    """국내/해외 예수금 및 보유종목 통합 조회."""
    return AccountOverviewResponse(**broker.get_account_overview())


@router.get("/balance", response_model=BalanceResponse)
def balance():
    return BalanceResponse(**broker.get_account_balance())


@router.get("/holdings", response_model=list[HoldingItem])
def holdings():
    return [HoldingItem(**h) for h in broker.get_account_holdings()]


@router.get("/orders/pending", response_model=list[PendingOrder])
def pending_orders():
    return [PendingOrder(**o) for o in broker.get_pending_orders()]


@router.post("/orders", response_model=OrderResponse)
def place_order(body: OrderRequest):
    price = None if body.order_type == "market" else body.price
    qty = int(body.qty) if float(body.qty).is_integer() else body.qty
    if body.side.lower() == "buy":
        res = broker.place_buy_order(body.symbol, qty, price=price)  # type: ignore[arg-type]
    else:
        res = broker.place_sell_order(body.symbol, qty, price=price)  # type: ignore[arg-type]
    return OrderResponse(**res)


@router.post("/orders/cancel", response_model=OrderResponse)
def cancel(body: CancelOrderRequest):
    return OrderResponse(**broker.cancel_order(body.symbol, body.order_id))
