from fastapi import APIRouter, Query

from app.services.dividend_calendar_service import build_dividend_calendar

router = APIRouter(prefix="/calendar", tags=["calendar"])


@router.get("/dividends")
def dividends(
    start: str | None = Query(None, alias="from"),
    end: str | None = Query(None, alias="to"),
    scope: str = Query("holdings", description="holdings | recommend | all | manual"),
    symbols: str = Query("", description="comma separated symbols"),
    force: bool = Query(False),
):
    return build_dividend_calendar(start=start, end=end, scope=scope, symbols=symbols, force=force)
