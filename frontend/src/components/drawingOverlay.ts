/** 차트 드로잉 캔버스: 좌표 변환·히트테스트·렌더 */

import type { IChartApi, ISeriesApi } from "lightweight-charts";
import {
  ChartDrawing,
  ChartPoint,
  fibPrices,
} from "./chartDrawings";

type Theme = "light" | "dark";

export type HitTarget =
  | { drawingId: string; part: "a" | "b" | "body" }
  | null;

export type DragState =
  | {
      drawingId: string;
      part: "a" | "b" | "body";
      startX: number;
      startY: number;
      origin: ChartDrawing;
    }
  | null;

const HANDLE_R = 6;
const HIT_LINE = 8;

function fibLevelColor(ratio: number, theme: Theme): string {
  if (ratio === 0 || ratio === 1) return theme === "dark" ? "#94a3b8" : "#64748b";
  if (ratio === 0.618) return theme === "dark" ? "#fbbf24" : "#d97706";
  if (ratio === 0.5) return theme === "dark" ? "#f59e0b" : "#b45309";
  return theme === "dark" ? "#fcd34d" : "#ca8a04";
}

function trendColor(theme: Theme, selected: boolean): string {
  if (selected) return "#f472b6";
  return theme === "dark" ? "#38bdf8" : "#0284c7";
}

type PointXY = { x: number; y: number };

function distPointToSeg(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  if (len2 <= 1e-9) return Math.hypot(px - ax, py - ay);
  let t = ((px - ax) * dx + (py - ay) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  const qx = ax + t * dx;
  const qy = ay + t * dy;
  return Math.hypot(px - qx, py - qy);
}

/** 두 점을 지나는 직선을 캔버스 경계까지 연장한 세그먼트 */
function lineSegment(
  a: PointXY,
  b: PointXY,
  width: number,
  height: number,
  extendLeft: boolean,
  extendRight: boolean
): { x1: number; y1: number; x2: number; y2: number } {
  if (!extendLeft && !extendRight) {
    return { x1: a.x, y1: a.y, x2: b.x, y2: b.y };
  }

  const dx = b.x - a.x;
  const dy = b.y - a.y;

  if (Math.abs(dx) < 1e-6) {
    const x = a.x;
    const yTop = extendLeft || extendRight ? 0 : Math.min(a.y, b.y);
    const yBot = extendLeft || extendRight ? height : Math.max(a.y, b.y);
    return { x1: x, y1: yTop, x2: x, y2: yBot };
  }

  const m = dy / dx;
  const yAt = (x: number) => a.y + m * (x - a.x);

  const leftAnchor = a.x <= b.x ? a : b;
  const rightAnchor = a.x <= b.x ? b : a;

  let x1 = extendLeft ? 0 : leftAnchor.x;
  let x2 = extendRight ? width : rightAnchor.x;
  let y1 = yAt(x1);
  let y2 = yAt(x2);

  // 화면 밖 y 클리핑(선택적 안정화)
  const clip = (x: number, y: number, otherX: number, otherY: number) => {
    if (y >= 0 && y <= height) return { x, y };
    const targetY = y < 0 ? 0 : height;
    if (Math.abs(m) < 1e-9) return { x, y: targetY };
    const nx = a.x + (targetY - a.y) / m;
    if ((nx - x) * (otherX - x) < 0) return { x, y: targetY };
    return { x: nx, y: targetY };
  };

  const c1 = clip(x1, y1, x2, y2);
  const c2 = clip(x2, y2, x1, y1);
  return { x1: c1.x, y1: c1.y, x2: c2.x, y2: c2.y };
}

function pointToXY(
  chart: IChartApi,
  series: ISeriesApi<"Candlestick">,
  p: ChartPoint,
  toChartTime: (raw: string) => unknown
): PointXY | null {
  const x = chart.timeScale().timeToCoordinate(toChartTime(p.time) as any);
  const y = series.priceToCoordinate(p.price);
  if (x == null || y == null) return null;
  return { x: Number(x), y: Number(y) };
}

export function hitTestDrawings(
  mx: number,
  my: number,
  drawings: ChartDrawing[],
  chart: IChartApi,
  series: ISeriesApi<"Candlestick">,
  width: number,
  height: number,
  toChartTime: (raw: string) => unknown
): HitTarget {
  // 선택 핸들 우선 (역순: 최근 그린 것 우선)
  for (let i = drawings.length - 1; i >= 0; i--) {
    const d = drawings[i];
    const pa = pointToXY(chart, series, d.a, toChartTime);
    const pb = pointToXY(chart, series, d.b, toChartTime);
    if (!pa || !pb) continue;

    if (Math.hypot(mx - pa.x, my - pa.y) <= HANDLE_R + 4) {
      return { drawingId: d.id, part: "a" };
    }
    if (Math.hypot(mx - pb.x, my - pb.y) <= HANDLE_R + 4) {
      return { drawingId: d.id, part: "b" };
    }
  }

  for (let i = drawings.length - 1; i >= 0; i--) {
    const d = drawings[i];
    const pa = pointToXY(chart, series, d.a, toChartTime);
    const pb = pointToXY(chart, series, d.b, toChartTime);
    if (!pa || !pb) continue;

    if (d.kind === "trendline") {
      const seg = lineSegment(pa, pb, width, height, d.extendLeft, d.extendRight);
      if (distPointToSeg(mx, my, seg.x1, seg.y1, seg.x2, seg.y2) <= HIT_LINE) {
        return { drawingId: d.id, part: "body" };
      }
    } else {
      const xLeft = d.extendLeft ? 0 : Math.min(pa.x, pb.x);
      const xRight = d.extendRight ? width : Math.max(pa.x, pb.x);
      for (const { price } of fibPrices(d.a, d.b)) {
        const y = series.priceToCoordinate(price);
        if (y == null) continue;
        if (distPointToSeg(mx, my, xLeft, Number(y), xRight, Number(y)) <= HIT_LINE) {
          return { drawingId: d.id, part: "body" };
        }
      }
      // 앵커 세로 가이드
      if (distPointToSeg(mx, my, pa.x, pa.y, pb.x, pb.y) <= HIT_LINE) {
        return { drawingId: d.id, part: "body" };
      }
    }
  }
  return null;
}

function drawHandle(ctx: CanvasRenderingContext2D, p: PointXY, selected: boolean) {
  ctx.beginPath();
  ctx.arc(p.x, p.y, HANDLE_R, 0, Math.PI * 2);
  ctx.fillStyle = selected ? "#f472b6" : "#ffffff";
  ctx.fill();
  ctx.strokeStyle = selected ? "#db2777" : "#0284c7";
  ctx.lineWidth = 2;
  ctx.stroke();
}

export function paintDrawings(
  ctx: CanvasRenderingContext2D,
  drawings: ChartDrawing[],
  selectedId: string | null,
  pending: ChartPoint | null,
  hoverXY: PointXY | null,
  chart: IChartApi,
  series: ISeriesApi<"Candlestick">,
  width: number,
  height: number,
  theme: Theme,
  toChartTime: (raw: string) => unknown
) {
  ctx.clearRect(0, 0, width, height);

  for (const d of drawings) {
    const selected = d.id === selectedId;
    const pa = pointToXY(chart, series, d.a, toChartTime);
    const pb = pointToXY(chart, series, d.b, toChartTime);
    if (!pa || !pb) continue;

    if (d.kind === "trendline") {
      const seg = lineSegment(pa, pb, width, height, d.extendLeft, d.extendRight);
      ctx.beginPath();
      ctx.moveTo(seg.x1, seg.y1);
      ctx.lineTo(seg.x2, seg.y2);
      ctx.strokeStyle = trendColor(theme, selected);
      ctx.lineWidth = selected ? 2.5 : 2;
      ctx.setLineDash([]);
      ctx.stroke();
    } else {
      const xLeft = d.extendLeft ? 0 : Math.min(pa.x, pb.x);
      const xRight = d.extendRight ? width : Math.max(pa.x, pb.x);
      for (const { ratio, price } of fibPrices(d.a, d.b)) {
        const y = series.priceToCoordinate(price);
        if (y == null) continue;
        ctx.beginPath();
        ctx.moveTo(xLeft, Number(y));
        ctx.lineTo(xRight, Number(y));
        ctx.strokeStyle = selected ? "#f472b6" : fibLevelColor(ratio, theme);
        ctx.lineWidth = selected || ratio === 0.618 || ratio === 0.5 ? 2 : 1;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = selected ? "#f472b6" : fibLevelColor(ratio, theme);
        ctx.font = "11px sans-serif";
        ctx.fillText(`${(ratio * 100).toFixed(1)}%`, xLeft + 4, Number(y) - 4);
      }
      // 앵커 연결선
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.strokeStyle = selected ? "rgba(244,114,182,0.55)" : "rgba(217,119,6,0.45)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (selected) {
      drawHandle(ctx, pa, true);
      drawHandle(ctx, pb, true);
    }
  }

  // 그리는 중 미리보기
  if (pending) {
    const pa = pointToXY(chart, series, pending, toChartTime);
    if (pa) {
      drawHandle(ctx, pa, false);
      if (hoverXY) {
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(hoverXY.x, hoverXY.y);
        ctx.strokeStyle = theme === "dark" ? "#94a3b8" : "#64748b";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }
}

function findBarIndex(time: string, bars: { time: string }[]): number {
  const exact = bars.findIndex((b) => b.time === time);
  if (exact >= 0) return exact;
  const day = time.slice(0, 10);
  return bars.findIndex((b) => b.time === day || b.time.startsWith(day));
}

export function applyDrag(
  origin: ChartDrawing,
  part: "a" | "b" | "body",
  startX: number,
  startY: number,
  curX: number,
  curY: number,
  chart: IChartApi,
  series: ISeriesApi<"Candlestick">,
  bars: { time: string }[],
  resolveTime: (chartTime: unknown) => string | null
): ChartDrawing {
  const priceAt = (y: number) => {
    const p = series.coordinateToPrice(y);
    return p == null ? null : Number(p);
  };
  const timeAt = (x: number) => {
    const t = chart.timeScale().coordinateToTime(x);
    if (t === undefined || t === null) return null;
    return resolveTime(t);
  };

  if (part === "a" || part === "b") {
    const time = timeAt(curX);
    const price = priceAt(curY);
    if (time == null || price == null || !Number.isFinite(price)) return origin;
    return { ...origin, [part]: { time, price } };
  }

  // body: 가격 델타 + 봉 인덱스 시프트
  const startPrice = priceAt(startY);
  const curPrice = priceAt(curY);
  if (startPrice == null || curPrice == null) return origin;
  const dPrice = curPrice - startPrice;

  const startTime = timeAt(startX);
  const curTime = timeAt(curX);
  let dIndex = 0;
  if (startTime && curTime && bars.length) {
    const i0 = findBarIndex(startTime, bars);
    const i1 = findBarIndex(curTime, bars);
    if (i0 >= 0 && i1 >= 0) dIndex = i1 - i0;
  }

  const shift = (p: ChartPoint): ChartPoint => {
    let idx = findBarIndex(p.time, bars);
    if (idx < 0) idx = 0;
    const nextIdx = Math.max(0, Math.min(bars.length - 1, idx + dIndex));
    return {
      time: bars.length ? bars[nextIdx].time : p.time,
      price: p.price + dPrice,
    };
  };

  return { ...origin, a: shift(origin.a), b: shift(origin.b) };
}
