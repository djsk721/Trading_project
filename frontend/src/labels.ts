/** 화면 표시용 한글 라벨. 내부 값(BUY, UP 등)은 그대로 두고 보여줄 때만 바꿉니다. */

export function trendKo(value: string | number | undefined | null): string {
  const s = String(value ?? "").toUpperCase();
  if (s === "UP") return "상승";
  if (s === "DOWN") return "하락";
  if (s === "SIDEWAYS" || s === "NEUTRAL") return "횡보";
  return s ? String(value) : "-";
}

export function macdKo(value: string | number | undefined | null): string {
  const s = String(value ?? "").toUpperCase();
  if (s === "BUY" || s === "BULLISH") return "매수";
  if (s === "SELL" || s === "BEARISH") return "매도";
  if (s === "NEUTRAL") return "중립";
  return s ? String(value) : "-";
}

export function rsiZoneKo(value: string | undefined | null): string {
  const s = String(value ?? "").toUpperCase();
  if (s === "OVERBOUGHT") return "과매수";
  if (s === "OVERSOLD") return "과매도";
  if (s === "NEUTRAL") return "중립";
  return value || "-";
}

export function newsToneKo(value: string | undefined | null): string {
  const s = String(value ?? "").toLowerCase();
  if (s === "bullish" || s === "positive") return "호재";
  if (s === "bearish" || s === "negative") return "악재";
  if (s === "mixed") return "혼합";
  if (s === "neutral") return "중립";
  return "";
}

export function newsToneClass(value: string | undefined | null): string {
  const s = String(value ?? "").toLowerCase();
  if (s === "bullish" || s === "positive") return "tone-bullish";
  if (s === "bearish" || s === "negative") return "tone-bearish";
  if (s === "mixed") return "tone-mixed";
  if (s === "neutral") return "tone-neutral";
  return "";
}

export function llmEngineKo(provider: string | undefined | null): string {
  const s = String(provider ?? "").toLowerCase();
  if (s === "nvidia") return "클라우드 AI";
  if (s === "ollama") return "로컬 AI";
  if (s === "auto") return "자동 전환";
  return provider || "";
}
