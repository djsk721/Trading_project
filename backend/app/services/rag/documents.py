"""RAG용 분석 문서 생성."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class AnalysisDocument:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentGenerator:
    def __init__(self, stock_data: pd.DataFrame, stock_name: str):
        self.stock_data = stock_data
        self.stock_name = stock_name

    def generate(self, max_rows: int = 40) -> List[AnalysisDocument]:
        docs: List[AnalysisDocument] = []
        df = self.stock_data.tail(max_rows)
        for date, row in df.iterrows():
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)[:10]
            docs.append(self._price_doc(date_str, row))
            docs.append(self._technical_doc(date_str, row))
            docs.append(self._volume_doc(date_str, row))
            if self._is_special(row):
                docs.append(self._event_doc(date_str, row))
        return docs

    def _price_doc(self, date: str, row: pd.Series) -> AnalysisDocument:
        close = float(row.get("종가", row.get("close", 0)))
        sma5 = float(row.get("SMA_5", close))
        sma20 = float(row.get("SMA_20", close))
        change = float(row.get("Price_Change", 0))
        content = (
            f"{date} {self.stock_name} price analysis:\n"
            f"- Close: {close:,.2f}\n"
            f"- Daily change: {change:.2f}%\n"
            f"- vs SMA5: {'above' if close > sma5 else 'below'} ({sma5:,.2f})\n"
            f"- vs SMA20: {'above' if close > sma20 else 'below'} ({sma20:,.2f})\n"
            f"- Short trend: {row.get('Trend_5', 'N/A')}, Long trend: {row.get('Trend_20', 'N/A')}"
        )
        return AnalysisDocument(
            page_content=content,
            metadata={
                "date": date,
                "type": "price_analysis",
                "stock": self.stock_name,
                "price": close,
                "change": change,
            },
        )

    def _technical_doc(self, date: str, row: pd.Series) -> AnalysisDocument:
        rsi = float(row.get("RSI_14", 50))
        macd = float(row.get("MACD", 0))
        signal = float(row.get("Signal", 0))
        bb_pos = float(row.get("BB_Position", 0.5))
        content = (
            f"{date} {self.stock_name} technical indicators:\n"
            f"- RSI(14): {rsi:.1f} ({row.get('RSI_Signal', 'NEUTRAL')})\n"
            f"- MACD: {macd:.3f}, Signal: {signal:.3f} ({row.get('MACD_Signal', 'N/A')})\n"
            f"- Stochastic %K/%D: {float(row.get('%K', 50)):.1f}/{float(row.get('%D', 50)):.1f}\n"
            f"- Bollinger position: {bb_pos:.2f} ({row.get('BB_Signal', 'INSIDE')})\n"
            f"- Volatility: {float(row.get('Volatility', 0)):.2f}%"
        )
        return AnalysisDocument(
            page_content=content,
            metadata={
                "date": date,
                "type": "technical_analysis",
                "stock": self.stock_name,
                "rsi": rsi,
                "macd": macd,
                "bb_position": bb_pos,
            },
        )

    def _volume_doc(self, date: str, row: pd.Series) -> AnalysisDocument:
        vol = float(row.get("거래량", row.get("volume", 0)))
        ratio = float(row.get("Volume_Ratio", 1))
        content = (
            f"{date} {self.stock_name} volume analysis:\n"
            f"- Volume: {vol:,.0f}\n"
            f"- vs 20d avg: {ratio:.2f}x ({row.get('Volume_Signal', 'NORMAL')})\n"
            f"- Price-volume relation: "
            f"{'aligned' if float(row.get('Price_Change', 0)) * ratio > 0 else 'diverging'}"
        )
        return AnalysisDocument(
            page_content=content,
            metadata={
                "date": date,
                "type": "volume_analysis",
                "stock": self.stock_name,
                "volume": vol,
                "volume_ratio": ratio,
            },
        )

    def _is_special(self, row: pd.Series) -> bool:
        return bool(
            abs(float(row.get("Price_Change", 0))) > 5
            or float(row.get("Volume_Ratio", 1)) > 2
            or bool(row.get("MACD_Cross", False))
            or str(row.get("RSI_Signal", "")) in {"OVERBOUGHT", "OVERSOLD"}
        )

    def _event_doc(self, date: str, row: pd.Series) -> AnalysisDocument:
        events = []
        change = float(row.get("Price_Change", 0))
        if abs(change) > 5:
            events.append(f"sharp move {change:.2f}%")
        if float(row.get("Volume_Ratio", 1)) > 2:
            events.append(f"volume surge {float(row.get('Volume_Ratio', 1)):.1f}x")
        if bool(row.get("MACD_Cross", False)):
            events.append("MACD crossover")
        if str(row.get("RSI_Signal", "")) in {"OVERBOUGHT", "OVERSOLD"}:
            events.append(f"RSI {row.get('RSI_Signal')}")
        content = (
            f"{date} {self.stock_name} special events:\n"
            f"- Events: {', '.join(events) if events else 'n/a'}\n"
            f"- Close: {float(row.get('종가', row.get('close', 0))):,.2f}\n"
            f"- RSI: {float(row.get('RSI_14', 50)):.1f}, MACD: {float(row.get('MACD', 0)):.3f}"
        )
        return AnalysisDocument(
            page_content=content,
            metadata={
                "date": date,
                "type": "special_event",
                "stock": self.stock_name,
                "events": events,
            },
        )


def news_to_documents(news_items: List[dict], stock_name: str) -> List[AnalysisDocument]:
    docs: List[AnalysisDocument] = []
    for item in news_items[:12]:
        content = (
            f"News about {stock_name}:\n"
            f"- Title: {item.get('title', '')}\n"
            f"- Source: {item.get('source', '')}\n"
            f"- Published: {item.get('published_at', 'N/A')}\n"
            f"- Summary: {item.get('summary', '')}"
        )
        docs.append(
            AnalysisDocument(
                page_content=content,
                metadata={
                    "date": (item.get("published_at") or "")[:10],
                    "type": "news",
                    "stock": stock_name,
                    "url": item.get("url", ""),
                    "source": item.get("source", ""),
                },
            )
        )
    return docs
