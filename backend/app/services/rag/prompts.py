"""Gemma4 친화적 RAG 프롬프트 템플릿."""
from __future__ import annotations

from typing import Dict


SYSTEM_PROMPT = (
    "You are a professional equity research assistant. "
    "Use the provided context (price/technical/volume/news) and the rule-based "
    "technical checklist as primary evidence. "
    "Treat rule analysis as an objective indicator scorecard: agree, disagree, or "
    "refine it with clear reasons. Do not ignore conflicting rules. "
    "If evidence is insufficient, say so clearly. "
    "Respond in Korean. "
    "Write clean GitHub-flavored Markdown: use ## headings, bullet lists, and **bold** for key terms. "
    "Do not wrap the whole answer in a code fence. Do not use raw HTML. "
    "Keep sections short and scannable. Avoid guaranteed-profit claims. "
    "Always include risk caveats."
)


def build_user_prompt(
    analysis_type: str,
    *,
    stock_name: str,
    query: str,
    context: str,
    current_date: str,
    current_price: float,
    extra: Dict[str, str] | None = None,
    rule_block: str = "",
) -> str:
    extra = extra or {}
    price_txt = f"{current_price:,.2f}"
    rules_section = rule_block.strip() or "Rule analysis unavailable."

    if analysis_type in {"forecast_20d", "20일 예측"}:
        return f"""Analyze the next 20 trading days outlook for {stock_name}.

Date: {current_date}
Current price: {price_txt}
RSI_14: {extra.get('RSI_14', 'N/A')}
MACD: {extra.get('MACD', 'N/A')}
BB_position: {extra.get('BB_position', 'N/A')}
Target dates:
{extra.get('date_table', 'N/A')}

Rule-based technical checklist (must evaluate explicitly):
{rules_section}

Context:
{context}

Write Korean Markdown with these ## headings:
## 룰 체크리스트 평가
## 기본 시나리오
## 상승 / 하락 시나리오
## 무효화 레벨
## 신뢰도

Use short bullets. Do not wrap the answer in a code fence."""

    if analysis_type in {"strategy", "투자전략"}:
        return f"""Create an actionable investment strategy for {stock_name}.

Date: {current_date}
Question: {query}
Current price: {price_txt}

Rule-based technical checklist (must evaluate explicitly):
{rules_section}

Context:
{context}

Write Korean Markdown with these ## headings:
## 룰 체크리스트 평가
## 포지션 아이디어
## 진입 / 추가 / 축소
## 손절 / 익절
## 리스크 점검

Use short bullets. Do not wrap the answer in a code fence."""

    # basic
    return f"""Answer the user question about {stock_name} using the rule checklist and context.

Date: {current_date}
Question: {query}
Current price: {price_txt}

Rule-based technical checklist (must evaluate explicitly):
{rules_section}

Context:
{context}

Write Korean Markdown with these ## headings:
## 결론
## 룰 체크리스트 평가
## 근거
## 리스크

Use short bullets. Do not wrap the answer in a code fence."""
