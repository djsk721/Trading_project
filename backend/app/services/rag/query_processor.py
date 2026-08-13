"""RAG 질의 처리 (Ollama gemma4:31b)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd

from app.core.config import get_settings
from app.services.indicators import calculate_indicators, latest_summary
from app.services.market_data import fetch_ohlcv, resolve_stock_name
from app.services.news_service import fetch_news
from app.services.rag.documents import DocumentGenerator, news_to_documents
from app.services.rag.llm_router import get_llm_router
from app.services.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from app.services.rag.retriever import RAGRetriever
from app.services.rule_analysis import (
    build_rule_analysis_from_df,
    format_rule_analysis_for_prompt,
)

log = logging.getLogger(__name__)


def _business_days(start: str, n: int = 20) -> List[str]:
    dt = datetime.strptime(start, "%Y-%m-%d")
    days = []
    cur = dt + timedelta(days=1)
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days


def _normalize_type(analysis_type: str) -> str:
    mapping = {
        "기본 분석": "basic",
        "basic": "basic",
        "20일 예측": "forecast_20d",
        "forecast_20d": "forecast_20d",
        "투자전략": "strategy",
        "strategy": "strategy",
    }
    return mapping.get(analysis_type, "basic")


class QueryProcessor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = RAGRetriever()

    def analyze(
        self,
        *,
        symbol: str,
        query: str,
        analysis_type: str = "basic",
        market: str = "KRX",
        stock_name: str = "",
        days: int = 120,
        provider: str = "",
    ) -> Dict:
        analysis_type = _normalize_type(analysis_type)
        name = stock_name or resolve_stock_name(symbol, market)

        df = fetch_ohlcv(symbol, market=market, timeframe="day", days=days)
        if df.empty:
            empty_rules = build_rule_analysis_from_df(
                pd.DataFrame(), symbol=symbol, market=market, stock_name=name
            )
            return {
                "answer": "시세 데이터를 불러오지 못해 분석을 수행할 수 없습니다.",
                "analysis_type": analysis_type,
                "symbol": symbol,
                "stock_name": name,
                "current_price": 0,
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "sources": [],
                "model": "ai",
                "provider": "none",
                "rule_analysis": empty_rules,
            }

        ind = calculate_indicators(df)
        summary = latest_summary(ind)
        current_price = float(summary.get("price", 0))
        current_date = ind.index[-1].strftime("%Y-%m-%d") if hasattr(ind.index[-1], "strftime") else str(ind.index[-1])[:10]

        # 지표 룰 분석 (AI 평가용 체크리스트)
        rule_analysis = build_rule_analysis_from_df(
            ind, symbol=symbol, market=market, stock_name=name
        )
        rule_block = format_rule_analysis_for_prompt(rule_analysis)

        # 문서: 기술적 분석 + 최신 뉴스
        docs = DocumentGenerator(ind, name).generate(max_rows=40)
        news_payload = fetch_news(symbol, market=market, stock_name=name)
        docs.extend(news_to_documents(news_payload.get("items", []), name))

        retriever_mode = self.retriever.build(docs)
        log.info("RAG built with %s docs (%s)", len(docs), retriever_mode)

        context, used = self.retriever.get_context(query)
        # 룰 요약을 검색 컨텍스트 앞에 고정 배치해 근거를 강화
        context = (
            f"[Rule-based checklist]\n{rule_block}\n\n"
            f"[Retrieved context]\n{context or 'No retrieved context.'}"
        )
        extra: Dict[str, str] = {}
        if analysis_type == "forecast_20d":
            extra = {
                "RSI_14": f"{summary.get('rsi', 0):.2f}",
                "MACD": f"{summary.get('macd', 0):.4f}",
                "BB_position": f"{float(ind.iloc[-1].get('BB_Position', 0.5)):.2f}",
                "date_table": "\n".join(f"- {d}" for d in _business_days(current_date, 20)),
            }

        user_prompt = build_user_prompt(
            analysis_type,
            stock_name=name,
            query=query,
            context=context,
            current_date=current_date,
            current_price=current_price,
            extra=extra,
            rule_block=rule_block,
        )

        used_provider = ""
        try:
            answer, used_provider = get_llm_router().chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                provider=provider or None,
            )
        except Exception as e:
            log.exception("LLM chat failed")
            answer = (
                f"AI 분석 엔진 호출에 실패했습니다. "
                f"AI 서비스 연결 상태를 확인한 뒤 다시 시도해주세요. ({e})"
            )

        sources = [
            {"content": d.page_content[:500], "metadata": d.metadata}
            for d in used
        ]
        return {
            "answer": answer,
            "analysis_type": analysis_type,
            "symbol": symbol,
            "stock_name": name,
            "current_price": current_price,
            "current_date": current_date,
            "sources": sources,
            "model": "ai",
            "provider": used_provider or "none",
            "rule_analysis": rule_analysis,
        }
