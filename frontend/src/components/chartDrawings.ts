/** 차트 드로잉(추세선·피보나치) 타입 및 종목별 localStorage 저장 */

export type ChartPoint = {
  time: string;
  price: number;
};

type DrawingBase = {
  id: string;
  a: ChartPoint;
  b: ChartPoint;
  /** 왼쪽으로 무한 연장 */
  extendLeft: boolean;
  /** 오른쪽으로 무한 연장 */
  extendRight: boolean;
};

type TrendlineDrawing = DrawingBase & {
  kind: "trendline";
};

type FibDrawing = DrawingBase & {
  kind: "fib";
};

export type ChartDrawing = TrendlineDrawing | FibDrawing;

export type DrawMode = "none" | "trendline" | "fib";

const FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1] as const;

const STORAGE_PREFIX = "trading_chart_drawings_v1";

function isPoint(v: unknown): v is ChartPoint {
  if (!v || typeof v !== "object") return false;
  const p = v as ChartPoint;
  return typeof p.time === "string" && typeof p.price === "number" && Number.isFinite(p.price);
}

function normalizeDrawing(v: unknown): ChartDrawing | null {
  if (!v || typeof v !== "object") return null;
  const raw = v as Partial<ChartDrawing> & { kind?: string };
  if (typeof raw.id !== "string") return null;
  if (raw.kind !== "trendline" && raw.kind !== "fib") return null;
  if (!isPoint(raw.a) || !isPoint(raw.b)) return null;
  return {
    id: raw.id,
    kind: raw.kind,
    a: raw.a,
    b: raw.b,
    extendLeft: Boolean(raw.extendLeft),
    extendRight: Boolean(raw.extendRight),
  };
}

function drawingStorageKey(market: string, symbol: string): string {
  return `${STORAGE_PREFIX}:${market.trim().toUpperCase()}:${symbol.trim().toUpperCase()}`;
}

export function loadDrawings(market: string, symbol: string): ChartDrawing[] {
  if (!market || !symbol) return [];
  try {
    const raw = localStorage.getItem(drawingStorageKey(market, symbol));
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.map(normalizeDrawing).filter((d): d is ChartDrawing => d != null);
  } catch {
    return [];
  }
}

export function saveDrawings(market: string, symbol: string, drawings: ChartDrawing[]): void {
  if (!market || !symbol) return;
  try {
    localStorage.setItem(drawingStorageKey(market, symbol), JSON.stringify(drawings));
  } catch {
    // quota 등 무시
  }
}

function newDrawingId(): string {
  return `d_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export function fibPrices(a: ChartPoint, b: ChartPoint): { ratio: number; price: number }[] {
  const high = Math.max(a.price, b.price);
  const low = Math.min(a.price, b.price);
  const range = high - low;
  return FIB_LEVELS.map((ratio) => ({
    ratio,
    price: high - range * ratio,
  }));
}

export function drawingLabel(d: ChartDrawing, index: number): string {
  if (d.kind === "trendline") return `추세선 ${index + 1}`;
  return `피보나치 ${index + 1}`;
}

export function createDrawing(
  kind: "trendline" | "fib",
  a: ChartPoint,
  b: ChartPoint
): ChartDrawing {
  return {
    id: newDrawingId(),
    kind,
    a,
    b,
    extendLeft: false,
    extendRight: kind === "fib",
  };
}
