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
    "Respond in Korean. Be concise, structured, and avoid guaranteed-profit claims. "
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

Write:
1) Assessment of the rule checklist (agree / partial / disagree) with reasons
2) Base scenario
3) Bull / bear cases with price zones
4) Key invalidation levels
5) Confidence (1-10) with rationale
"""

    if analysis_type in {"strategy", "투자전략"}:
        return f"""Create an actionable investment strategy for {stock_name}.

Date: {current_date}
Question: {query}
Current price: {price_txt}

Rule-based technical checklist (must evaluate explicitly):
{rules_section}

Context:
{context}

Cover:
1) Assessment of the rule checklist (agree / partial / disagree)
2) Positioning / allocation idea
3) Entry / add / reduce plan
4) Stop-loss and take-profit zones
5) Risk management checklist
"""

    # basic
    return f"""Answer the user question about {stock_name} using the rule checklist and context.

Date: {current_date}
Question: {query}
Current price: {price_txt}

Rule-based technical checklist (must evaluate explicitly):
{rules_section}

Context:
{context}

Structure:
1) Direct answer
2) Rule checklist assessment (which signals matter most now)
3) Supporting evidence from context
4) Risks / uncertainties
"""
