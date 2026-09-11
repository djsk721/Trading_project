from fastapi import APIRouter, Query
import logging

from app.schemas.trading import (
    AccountOverviewResponse,
    AvailableOrderResponse,
    BalanceResponse,
    CancelOrderRequest,
    HoldingItem,
    ModifyOrderRequest,
    OrderRequest,
    OrderResponse,
    PendingOrder,
)
from app.services import broker

router = APIRouter(prefix="/trading", tags=["trading"])


@router.get("/account", response_model=AccountOverviewResponse)
def account_overview(force: bool = Query(False)):
    """국내/해외 예수금 및 보유종목 통합 조회."""
    logging.getLogger(__name__).info("account overview requested force=%s", force)
    return AccountOverviewResponse(**broker.get_account_overview(force=force))


@router.get("/balance", response_model=BalanceResponse)
def balance():
    return BalanceResponse(**broker.get_account_balance())


@router.get("/holdings", response_model=list[HoldingItem])
def holdings():
    return [HoldingItem(**h) for h in broker.get_account_holdings()]


@router.get("/orders/pending", response_model=list[PendingOrder])
def pending_orders():
    return [PendingOrder(**o) for o in broker.get_pending_orders()]


@router.get("/orders/history")
def order_history(symbol: str = "", days: int = 30):
    return {"items": [], "symbol": symbol, "days": days, "message": "주문 이력 연동은 브로커별 API 매핑 후 활성화됩니다."}


@router.get("/fills/history")
def fills_history(symbol: str = "", days: int = 30):
    return {"items": [], "symbol": symbol, "days": days, "message": "체결 이력 연동은 브로커별 API 매핑 후 활성화됩니다."}


@router.get("/orders/available", response_model=AvailableOrderResponse)
def order_available(symbol: str = "", side: str = "buy", price: float = 0):
    account = broker.get_account_overview(force=False)
    currency = "KRW" if symbol.isdigit() else "USD"
    if side.lower() == "sell":
        holding = next((h for h in account.get("holdings") or [] if str(h.get("symbol")) == symbol), {})
        qty_available = float(holding.get("qty") or 0)
        return AvailableOrderResponse(
            symbol=symbol,
            side="sell",
            qty_available=qty_available,
            currency=str(holding.get("currency") or currency),
            message="보유 수량 기준 주문 가능 수량입니다.",
        )
    deposit = 0.0
    for dep in account.get("deposits") or []:
        if str(dep.get("currency") or "").upper() == currency:
            deposit = float(dep.get("amount") or 0)
            break
    qty_available = int(deposit / price) if price > 0 else 0
    return AvailableOrderResponse(
        symbol=symbol,
        side="buy",
        cash_available=deposit,
        qty_available=qty_available,
        currency=currency,
        message="예수금과 입력 가격 기준 단순 산출입니다.",
    )


@router.get("/capabilities")
def capabilities():
    return broker.capabilities()


@router.get("/diagnostics")
def diagnostics():
    return broker.diagnostics()


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


@router.post("/orders/modify", response_model=OrderResponse)
def modify(body: ModifyOrderRequest):
    return OrderResponse(
        success=False,
        message="주문 정정은 아직 해당 브로커 공통 Schema에 연결되지 않았습니다.",
        data={"unsupported": True, "symbol": body.symbol, "order_id": body.order_id},
    )
