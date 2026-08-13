import { useEffect, useMemo, useRef, useState } from "react";
import {
  CandlestickSeries,
  createChart,
  HistogramSeries,
  IChartApi,
  ISeriesApi,
  LineSeries,
  LineStyle,
  MouseEventParams,
} from "lightweight-charts";
import type { ChartPayload } from "../api";
import {
  ChartDrawing,
  ChartPoint,
  DrawMode,
  createDrawing,
  drawingLabel,
  loadDrawings,
  saveDrawings,
} from "./chartDrawings";
import {
  DragState,
  HitTarget,
  applyDrag,
  hitTestDrawings,
  paintDrawings,
} from "./drawingOverlay";

export type ChartTimeframe =
  | "1m"
  | "3m"
  | "5m"
  | "10m"
  | "15m"
  | "30m"
  | "1h"
  | "2h"
  | "4h"
  | "day"
  | "week";

type Theme = "light" | "dark";

type Props = {
  data: ChartPayload | null;
  theme: Theme;
  timeframe: ChartTimeframe;
  onTimeframeChange: (tf: ChartTimeframe) => void;
  loading?: boolean;
  symbol: string;
  market: string;
};

type BarRow = ChartPayload["bars"][number];

const MINUTE_OPTIONS: { id: ChartTimeframe; label: string }[] = [
  { id: "1m", label: "1분" },
  { id: "3m", label: "3분" },
  { id: "5m", label: "5분" },
  { id: "10m", label: "10분" },
  { id: "15m", label: "15분" },
  { id: "30m", label: "30분" },
];

const HOUR_OPTIONS: { id: ChartTimeframe; label: string }[] = [
  { id: "1h", label: "1시간" },
  { id: "2h", label: "2시간" },
  { id: "4h", label: "4시간" },
];

const QUICK_TFS: { id: ChartTimeframe; label: string }[] = [
  { id: "5m", label: "5분" },
  { id: "15m", label: "15분" },
  { id: "1h", label: "1시간" },
  { id: "day", label: "일" },
  { id: "week", label: "주" },
];

function isMinuteTf(tf: ChartTimeframe) {
  return MINUTE_OPTIONS.some((o) => o.id === tf);
}

function isHourTf(tf: ChartTimeframe) {
  return HOUR_OPTIONS.some((o) => o.id === tf);
}

function isIntradayTf(tf: ChartTimeframe) {
  return isMinuteTf(tf) || isHourTf(tf);
}

function chartTimeKey(t: unknown): string {
  if (typeof t === "string") return t;
  if (typeof t === "number") return String(t);
  if (t && typeof t === "object" && "year" in (t as object)) {
    const bd = t as { year: number; month: number; day: number };
    return `${bd.year}-${String(bd.month).padStart(2, "0")}-${String(bd.day).padStart(2, "0")}`;
  }
  return "";
}

function buildBarLookup(bars: BarRow[]) {
  const map = new Map<string, { bar: BarRow; idx: number }>();
  bars.forEach((bar, idx) => {
    const ct = toChartTime(bar.time);
    map.set(bar.time, { bar, idx });
    if (bar.time.length >= 10) map.set(bar.time.slice(0, 10), { bar, idx });
    if (typeof ct === "string") map.set(ct, { bar, idx });
    if (typeof ct === "number") map.set(String(ct), { bar, idx });
  });
  return map;
}

function ohlcFromBar(bar: BarRow, prev?: BarRow | null): OhlcView {
  return {
    time: bar.time,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    volume: bar.volume,
    change: prev ? bar.close - prev.close : 0,
  };
}

type IndicatorKey =
  | "sma20"
  | "sma60"
  | "sma120"
  | "ema20"
  | "bb"
  | "volume"
  | "rsi"
  | "macd"
  | "stoch";

type OhlcView = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  change: number;
};

const INDICATOR_DEFS: { key: IndicatorKey; label: string; defaultOn: boolean }[] = [
  { key: "sma20", label: "SMA20", defaultOn: true },
  { key: "sma60", label: "SMA60", defaultOn: true },
  { key: "sma120", label: "SMA120", defaultOn: true },
  { key: "ema20", label: "EMA20", defaultOn: false },
  { key: "bb", label: "BB(20,2)", defaultOn: true },
  { key: "volume", label: "거래량", defaultOn: true },
  { key: "rsi", label: "RSI(14)", defaultOn: true },
  { key: "macd", label: "MACD", defaultOn: true },
  { key: "stoch", label: "Stoch", defaultOn: false },
];

type IndLegend = {
  sma20?: number;
  sma60?: number;
  sma120?: number;
  ema20?: number;
  bbUpper?: number;
  bbMiddle?: number;
  bbLower?: number;
  rsi?: number;
  macd?: number;
  macdSignal?: number;
  macdHist?: number;
  stochK?: number;
  stochD?: number;
  volume?: number;
};

type PaneId = "main" | "volume" | "rsi" | "macd" | "stoch";

function themeColors(theme: Theme) {
  if (theme === "light") {
    return {
      bg: "#f8fafc",
      text: "#5f7388",
      grid: "#e6edf5",
      border: "#d5dee8",
      up: "#d64545",
      down: "#2563eb",
      sma: "#b7791f",
      sma60: "#d97706",
      sma120: "#9a3412",
      ema: "#0f766e",
      bb: "#64748b",
      rsi: "#7c3aed",
      macd: "#2563eb",
      signal: "#ea580c",
      stochK: "#0891b2",
      stochD: "#c026d3",
    };
  }
  return {
    bg: "#0c1116",
    text: "#93a4b5",
    grid: "#1b2631",
    border: "#2c3a48",
    up: "#ef6b6b",
    down: "#4aa3ff",
    sma: "#f0b429",
    sma60: "#fb923c",
    sma120: "#f97316",
    ema: "#2dd4bf",
    bb: "#94a3b8",
    rsi: "#c4b5fd",
    macd: "#60a5fa",
    signal: "#fb923c",
    stochK: "#22d3ee",
    stochD: "#e879f9",
  };
}

function toChartTime(raw: string) {
  // 일봉: BusinessDay 문자열
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw as any;

  // 분/시간봉: 거래소 로컬 wall-clock을 UTC 타임스탬프로 인코딩
  // (lightweight-charts가 UTCTimestamp를 UTC로 표시하므로, 축에 KST 시각이 그대로 보이게 함)
  const m = raw.match(
    /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$/
  );
  if (m) {
    const y = Number(m[1]);
    const mo = Number(m[2]) - 1;
    const d = Number(m[3]);
    const h = Number(m[4]);
    const mi = Number(m[5]);
    const s = m[6] ? Number(m[6]) : 0;
    return Math.floor(Date.UTC(y, mo, d, h, mi, s) / 1000) as any;
  }

  const ms = Date.parse(raw);
  if (Number.isNaN(ms)) return raw as any;
  return Math.floor(ms / 1000) as any;
}

function formatLegendTime(raw: string) {
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  const m = raw.match(/^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::\d{2})?/);
  if (m) return `${m[1]} ${m[2]}`;
  return raw.slice(0, 16);
}

function isKrMarket(market: string, symbol = "") {
  return (market || "").toUpperCase() === "KRX" || /^\d{6}$/.test(symbol || "");
}

/** 차트 Y축·호가 표시용 가격 스케일 */
function resolvePriceFormat(market: string, samplePrice: number, symbol = "") {
  if (isKrMarket(market, symbol)) {
    return { type: "price" as const, precision: 0, minMove: 1 };
  }
  const p = Math.abs(samplePrice) || 1;
  if (p >= 1000) return { type: "price" as const, precision: 2, minMove: 0.01 };
  if (p >= 1) return { type: "price" as const, precision: 2, minMove: 0.01 };
  if (p >= 0.1) return { type: "price" as const, precision: 3, minMove: 0.001 };
  if (p >= 0.01) return { type: "price" as const, precision: 4, minMove: 0.0001 };
  return { type: "price" as const, precision: 6, minMove: 0.000001 };
}

function formatPrice(n: number, market = "", symbol = "") {
  if (!Number.isFinite(n)) return "-";
  if (isKrMarket(market, symbol)) {
    return Math.round(n).toLocaleString("ko-KR");
  }
  const { precision } = resolvePriceFormat(market, n, symbol);
  return n.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: precision,
  });
}

function resolveMacdFormat(sample: number, market: string, symbol = "") {
  if (isKrMarket(market, symbol)) {
    const a = Math.abs(sample);
    if (a >= 100) return { type: "price" as const, precision: 0, minMove: 1 };
    if (a >= 10) return { type: "price" as const, precision: 1, minMove: 0.1 };
    return { type: "price" as const, precision: 2, minMove: 0.01 };
  }
  const a = Math.abs(sample) || 0.01;
  if (a >= 1) return { type: "price" as const, precision: 2, minMove: 0.01 };
  if (a >= 0.1) return { type: "price" as const, precision: 3, minMove: 0.001 };
  return { type: "price" as const, precision: 4, minMove: 0.0001 };
}

function formatMacdValue(n: number, market = "", symbol = "") {
  if (!Number.isFinite(n)) return "-";
  const { precision } = resolveMacdFormat(n, market, symbol);
  return n.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: precision,
  });
}

function formatVol(n: number) {
  if (!Number.isFinite(n)) return "-";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

function resolveBarTime(
  chartTime: unknown,
  lookup: Map<string, { bar: BarRow; idx: number }>,
  bars: BarRow[]
): string | null {
  const key = chartTimeKey(chartTime);
  const hit = lookup.get(key);
  if (hit) return hit.bar.time;
  if (typeof chartTime === "string") {
    const byDay = lookup.get(chartTime.slice(0, 10));
    if (byDay) return byDay.bar.time;
  }
  if (typeof chartTime === "number" && bars.length) {
    let best: BarRow | null = null;
    let bestDiff = Infinity;
    for (const bar of bars) {
      const ct = toChartTime(bar.time);
      if (typeof ct !== "number") continue;
      const diff = Math.abs(ct - chartTime);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = bar;
      }
    }
    if (best) return best.time;
  }
  return null;
}

function defaultToggles(): Record<IndicatorKey, boolean> {
  return INDICATOR_DEFS.reduce((acc, d) => {
    acc[d.key] = d.defaultOn;
    return acc;
  }, {} as Record<IndicatorKey, boolean>);
}

/** v5 멀티 pane 높이(px). 가격 패널 비중을 크게 유지 */
function paneHeightPlan(toggles: Record<IndicatorKey, boolean>, totalHeight: number) {
  const panes: { id: PaneId; weight: number }[] = [{ id: "main", weight: 3.8 }];
  if (toggles.volume) panes.push({ id: "volume", weight: 0.95 });
  if (toggles.rsi) panes.push({ id: "rsi", weight: 1.15 });
  if (toggles.macd) panes.push({ id: "macd", weight: 1.15 });
  if (toggles.stoch) panes.push({ id: "stoch", weight: 1.0 });

  const totalW = panes.reduce((a, p) => a + p.weight, 0);
  const plotH = Math.max(200, totalHeight - 30); // 시간축 여유
  return panes.map((p) => ({
    id: p.id,
    height: Math.max(48, Math.round((plotH * p.weight) / totalW)),
  }));
}

function latestIndicatorValue(
  series: { time: string; value: number }[] | undefined,
  barTime: string
): number | undefined {
  if (!series?.length) return undefined;
  const exact = series.find((p) => p.time === barTime || p.time.slice(0, 10) === barTime.slice(0, 10));
  if (exact) return exact.value;
  return series[series.length - 1]?.value;
}

function isBarInProgress(timeframe: ChartTimeframe, lastBarTime: string): boolean {
  // 분/시간봉: 마지막 봉은 항상 진행 중으로 간주
  if (isIntradayTf(timeframe)) return true;
  if (timeframe === "week") return false;

  // 일봉: KST 기준 오늘 날짜와 같으면 미확정(장중)
  const now = new Date();
  const kst = new Date(now.getTime() + 9 * 60 * 60 * 1000);
  const today = kst.toISOString().slice(0, 10);
  const barDay = lastBarTime.slice(0, 10);
  return barDay === today;
}

export default function CandleChart({
  data,
  theme,
  timeframe,
  onTimeframeChange,
  loading,
  symbol,
  market,
}: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<Record<string, ISeriesApi<any> | null>>({});
  const barsRef = useRef<BarRow[]>([]);
  const barLookupRef = useRef<Map<string, { bar: BarRow; idx: number }>>(new Map());
  const dataRef = useRef<ChartPayload | null>(null);
  const drawModeRef = useRef<DrawMode>("none");
  const pendingPointRef = useRef<ChartPoint | null>(null);
  const drawingsRef = useRef<ChartDrawing[]>([]);
  const selectedIdRef = useRef<string | null>(null);
  const dragRef = useRef<DragState>(null);
  const hoverXYRef = useRef<{ x: number; y: number } | null>(null);
  const symbolRef = useRef(symbol);
  const marketRef = useRef(market);
  const [toggles, setToggles] = useState<Record<IndicatorKey, boolean>>(defaultToggles);
  const [indMenuOpen, setIndMenuOpen] = useState(false);
  const indMenuRef = useRef<HTMLDivElement | null>(null);
  const [ohlc, setOhlc] = useState<OhlcView | null>(null);
  const [indLegend, setIndLegend] = useState<IndLegend>({});
  const [drawMode, setDrawMode] = useState<DrawMode>("none");
  const [drawings, setDrawings] = useState<ChartDrawing[]>([]);
  const [pendingPoint, setPendingPoint] = useState<ChartPoint | null>(null);
  const [selectedDrawingId, setSelectedDrawingId] = useState<string | null>(null);
  const [overlayCursor, setOverlayCursor] = useState("default");

  symbolRef.current = symbol;
  marketRef.current = market;
  drawModeRef.current = drawMode;
  pendingPointRef.current = pendingPoint;
  // 드래그 중에는 ref를 상태와 동기화하지 않음 (중간 좌표 유지)
  if (!dragRef.current) {
    drawingsRef.current = drawings;
  }
  selectedIdRef.current = selectedDrawingId;

  const activeIndicatorCount = useMemo(
    () => INDICATOR_DEFS.filter((d) => toggles[d.key]).length,
    [toggles]
  );

  const minuteSelectValue = isMinuteTf(timeframe) ? timeframe : "5m";
  const hourSelectValue = isHourTf(timeframe) ? timeframe : "1h";

  useEffect(() => {
    if (!indMenuOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (!indMenuRef.current?.contains(e.target as Node)) {
        setIndMenuOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIndMenuOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [indMenuOpen]);

  // 종목별 드로잉 로드 (저장은 persistDrawings에서만)
  useEffect(() => {
    setDrawings(loadDrawings(market, symbol));
    setPendingPoint(null);
    setDrawMode("none");
    setSelectedDrawingId(null);
  }, [market, symbol]);

  const persistDrawings = (
    updater: ChartDrawing[] | ((prev: ChartDrawing[]) => ChartDrawing[])
  ) => {
    setDrawings((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      saveDrawings(marketRef.current, symbolRef.current, next);
      return next;
    });
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setPendingPoint(null);
        setDrawMode("none");
        setSelectedDrawingId(null);
        return;
      }
      if ((e.key === "Delete" || e.key === "Backspace") && selectedDrawingId) {
        const tag = (e.target as HTMLElement | null)?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        e.preventDefault();
        persistDrawings((prev) => prev.filter((d) => d.id !== selectedDrawingId));
        setSelectedDrawingId(null);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [selectedDrawingId]);

  const colors = useMemo(() => themeColors(theme), [theme]);
  const chartHeight = useMemo(() => {
    let h = 440;
    if (toggles.volume) h += 80;
    if (toggles.rsi) h += 100;
    if (toggles.macd) h += 100;
    if (toggles.stoch) h += 90;
    return h;
  }, [toggles]);
  const heightPlan = useMemo(
    () => paneHeightPlan(toggles, chartHeight),
    [toggles, chartHeight]
  );
  const heightPlanRef = useRef(heightPlan);
  heightPlanRef.current = heightPlan;

  dataRef.current = data;

  const buildLegendForBar = (barTime: string, volume?: number): IndLegend => {
    const ind = dataRef.current?.indicators || {};
    return {
      sma20: latestIndicatorValue(ind.sma_20, barTime),
      sma60: latestIndicatorValue(ind.sma_60, barTime),
      sma120: latestIndicatorValue(ind.sma_120, barTime),
      ema20: latestIndicatorValue(ind.ema_20, barTime),
      bbUpper: latestIndicatorValue(ind.bb20_upper, barTime),
      bbMiddle: latestIndicatorValue(ind.bb20_middle, barTime),
      bbLower: latestIndicatorValue(ind.bb20_lower, barTime),
      rsi: latestIndicatorValue(ind.rsi, barTime),
      macd: latestIndicatorValue(ind.macd, barTime),
      macdSignal: latestIndicatorValue(ind.macd_signal, barTime),
      macdHist: latestIndicatorValue(ind.macd_histogram, barTime),
      stochK: latestIndicatorValue(ind.stoch_k, barTime),
      stochD: latestIndicatorValue(ind.stoch_d, barTime),
      volume,
    };
  };

  const buildLegendRef = useRef(buildLegendForBar);
  buildLegendRef.current = buildLegendForBar;

  useEffect(() => {
    const bars = data?.bars || [];
    barsRef.current = bars;
    barLookupRef.current = buildBarLookup(bars);
    if (!bars.length) {
      setOhlc(null);
      return;
    }
    const last = bars[bars.length - 1];
    const prev = bars.length > 1 ? bars[bars.length - 2] : null;
    setOhlc(ohlcFromBar(last, prev));
    setIndLegend(buildLegendForBar(last.time, last.volume));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  // 차트 인스턴스 생성/재구성 (v5 멀티 pane: 패널마다 독립 Y축)
  useEffect(() => {
    if (!ref.current) return;
    const chart = createChart(ref.current, {
      layout: {
        background: { color: colors.bg },
        textColor: colors.text,
        panes: {
          separatorColor: colors.border,
          separatorHoverColor:
            theme === "dark" ? "rgba(148,163,184,0.35)" : "rgba(100,116,139,0.35)",
        },
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid },
      },
      rightPriceScale: {
        borderColor: colors.border,
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
      timeScale: {
        borderColor: colors.border,
        timeVisible: true,
        secondsVisible: false,
      },
      localization: {
        locale: "ko-KR",
        timeFormatter: (time: any) => {
          if (typeof time === "string") return time;
          if (typeof time === "object" && time && "year" in time) {
            const bd = time as { year: number; month: number; day: number };
            return `${bd.year}-${String(bd.month).padStart(2, "0")}-${String(bd.day).padStart(2, "0")}`;
          }
          const sec = typeof time === "number" ? time : 0;
          const d = new Date(sec * 1000);
          const yyyy = d.getUTCFullYear();
          const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
          const dd = String(d.getUTCDate()).padStart(2, "0");
          const hh = String(d.getUTCHours()).padStart(2, "0");
          const mi = String(d.getUTCMinutes()).padStart(2, "0");
          return `${yyyy}-${mm}-${dd} ${hh}:${mi}`;
        },
      },
      crosshair: { mode: 0 },
      width: ref.current.clientWidth,
      height: chartHeight,
    });

    const paneIndex: Partial<Record<PaneId, number>> = { main: 0 };
    let nextPane = 1;
    if (toggles.volume) paneIndex.volume = nextPane++;
    if (toggles.rsi) paneIndex.rsi = nextPane++;
    if (toggles.macd) paneIndex.macd = nextPane++;
    if (toggles.stoch) paneIndex.stoch = nextPane++;

    const candle = chart.addSeries(
      CandlestickSeries,
      {
        upColor: colors.up,
        downColor: colors.down,
        borderVisible: false,
        wickUpColor: colors.up,
        wickDownColor: colors.down,
        priceFormat: resolvePriceFormat(marketRef.current, 100, symbolRef.current),
      },
      0
    );
    candle.priceScale().applyOptions({
      scaleMargins: { top: 0.06, bottom: 0.08 },
      borderVisible: true,
    });

    const lineOnPrice = (color: string, width: 1 | 2, style?: LineStyle) =>
      chart.addSeries(
        LineSeries,
        {
          color,
          lineWidth: width,
          lineStyle: style,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        },
        0
      );

    const sma = lineOnPrice(colors.sma, 2);
    const sma60 = lineOnPrice(colors.sma60, 2);
    const sma120 = lineOnPrice(colors.sma120, 2);
    const ema = lineOnPrice(colors.ema, 2);
    const bbUpper = lineOnPrice(colors.bb, 1, LineStyle.Dashed);
    const bbMiddle = lineOnPrice(colors.bb, 1);
    const bbLower = lineOnPrice(colors.bb, 1, LineStyle.Dashed);

    const guide = theme === "dark" ? "rgba(148,163,184,0.55)" : "rgba(100,116,139,0.55)";

    let volume: ISeriesApi<"Histogram"> | null = null;
    if (paneIndex.volume != null) {
      volume = chart.addSeries(
        HistogramSeries,
        {
          priceFormat: { type: "volume" },
          lastValueVisible: true,
          priceLineVisible: false,
          autoscaleInfoProvider: (
            original: () => { priceRange: { minValue: number; maxValue: number } | null } | null
          ) => {
            const base = original();
            if (base?.priceRange) {
              return {
                priceRange: {
                  minValue: 0,
                  maxValue: Math.max(base.priceRange.maxValue * 1.08, 1),
                },
              };
            }
            return { priceRange: { minValue: 0, maxValue: 1 } };
          },
        },
        paneIndex.volume
      );
      volume.priceScale().applyOptions({
        scaleMargins: { top: 0.12, bottom: 0 },
        borderVisible: true,
      });
    }

    let rsi: ISeriesApi<"Line"> | null = null;
    if (paneIndex.rsi != null) {
      rsi = chart.addSeries(
        LineSeries,
        {
          color: colors.rsi,
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: true,
          title: "RSI",
          priceFormat: { type: "price", precision: 1, minMove: 0.1 },
          autoscaleInfoProvider: () => ({
            priceRange: { minValue: 0, maxValue: 100 },
          }),
        },
        paneIndex.rsi
      );
      rsi.priceScale().applyOptions({
        scaleMargins: { top: 0.08, bottom: 0.08 },
        borderVisible: true,
      });
      rsi.createPriceLine({
        price: 70,
        color: theme === "dark" ? "rgba(239,68,68,0.7)" : "rgba(220,38,38,0.55)",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "70",
      });
      rsi.createPriceLine({
        price: 50,
        color: guide,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "50",
      });
      rsi.createPriceLine({
        price: 30,
        color: theme === "dark" ? "rgba(59,130,246,0.7)" : "rgba(37,99,235,0.55)",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "30",
      });
    }

    let macdHist: ISeriesApi<"Histogram"> | null = null;
    let macdLine: ISeriesApi<"Line"> | null = null;
    let macdSignal: ISeriesApi<"Line"> | null = null;
    if (paneIndex.macd != null) {
      macdHist = chart.addSeries(
        HistogramSeries,
        {
          priceFormat: { type: "price", precision: 2, minMove: 0.01 },
          lastValueVisible: false,
          priceLineVisible: false,
        },
        paneIndex.macd
      );
      macdLine = chart.addSeries(
        LineSeries,
        {
          color: colors.macd,
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: true,
          title: "MACD",
          priceFormat: { type: "price", precision: 2, minMove: 0.01 },
        },
        paneIndex.macd
      );
      macdSignal = chart.addSeries(
        LineSeries,
        {
          color: colors.signal,
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: true,
          title: "Signal",
          priceFormat: { type: "price", precision: 2, minMove: 0.01 },
        },
        paneIndex.macd
      );
      macdLine.priceScale().applyOptions({
        scaleMargins: { top: 0.1, bottom: 0.1 },
        borderVisible: true,
      });
      macdLine.createPriceLine({
        price: 0,
        color: guide,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "0",
      });
    }

    let stochK: ISeriesApi<"Line"> | null = null;
    let stochD: ISeriesApi<"Line"> | null = null;
    if (paneIndex.stoch != null) {
      stochK = chart.addSeries(
        LineSeries,
        {
          color: colors.stochK,
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: true,
          title: "%K",
          priceFormat: { type: "price", precision: 1, minMove: 0.1 },
          autoscaleInfoProvider: () => ({
            priceRange: { minValue: 0, maxValue: 100 },
          }),
        },
        paneIndex.stoch
      );
      stochD = chart.addSeries(
        LineSeries,
        {
          color: colors.stochD,
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
          priceFormat: { type: "price", precision: 1, minMove: 0.1 },
          autoscaleInfoProvider: () => ({
            priceRange: { minValue: 0, maxValue: 100 },
          }),
        },
        paneIndex.stoch
      );
      stochK.priceScale().applyOptions({
        scaleMargins: { top: 0.08, bottom: 0.08 },
        borderVisible: true,
      });
      stochK.createPriceLine({
        price: 80,
        color: theme === "dark" ? "rgba(239,68,68,0.55)" : "rgba(220,38,38,0.45)",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "80",
      });
      stochK.createPriceLine({
        price: 20,
        color: theme === "dark" ? "rgba(59,130,246,0.55)" : "rgba(37,99,235,0.45)",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "20",
      });
    }

    // pane 높이 비율 적용
    const panes = chart.panes();
    heightPlan.forEach((plan, i) => {
      if (panes[i]) panes[i].setHeight(plan.height);
    });

    seriesRef.current = {
      candle,
      sma,
      sma60,
      sma120,
      ema,
      bbUpper,
      bbMiddle,
      bbLower,
      volume,
      rsi,
      macdHist,
      macdLine,
      macdSignal,
      stochK,
      stochD,
    };
    chartRef.current = chart;

    const onCrosshair = (param: MouseEventParams) => {
      const bars = barsRef.current;
      // 차트 밖이면 마지막 봉으로 복원
      if (!param.point || param.time === undefined || !bars.length) {
        if (bars.length) {
          const last = bars[bars.length - 1];
          const prev = bars.length > 1 ? bars[bars.length - 2] : null;
          setOhlc(ohlcFromBar(last, prev));
          setIndLegend(buildLegendRef.current(last.time, last.volume));
        }
        return;
      }

      const key = chartTimeKey(param.time);
      let hit = barLookupRef.current.get(key);
      if (!hit) {
        // 초 단위/일자 키 보조 매칭
        for (const [k, v] of barLookupRef.current) {
          if (k === key || (key.length >= 10 && k.startsWith(key.slice(0, 10)))) {
            hit = v;
            break;
          }
        }
      }

      const series = seriesRef.current.candle;
      const candleVal = series
        ? (param.seriesData.get(series) as
            | { open: number; high: number; low: number; close: number }
            | undefined)
        : undefined;

      if (candleVal && typeof candleVal.open === "number") {
        const idx = hit?.idx ?? -1;
        const prev = idx > 0 ? bars[idx - 1] : null;
        const barTime = hit?.bar.time || key;
        setOhlc({
          time: barTime,
          open: candleVal.open,
          high: candleVal.high,
          low: candleVal.low,
          close: candleVal.close,
          volume: hit?.bar.volume ?? 0,
          change: prev ? candleVal.close - prev.close : 0,
        });
        setIndLegend(buildLegendRef.current(barTime, hit?.bar.volume));
        return;
      }

      if (hit) {
        const prev = hit.idx > 0 ? bars[hit.idx - 1] : null;
        setOhlc(ohlcFromBar(hit.bar, prev));
        setIndLegend(buildLegendRef.current(hit.bar.time, hit.bar.volume));
      }
    };
    chart.subscribeCrosshairMove(onCrosshair);

    // 오버레이가 비활성일 때 클릭으로 드로잉 선택
    const onClick = (param: MouseEventParams) => {
      if (drawModeRef.current !== "none") return;
      if (selectedIdRef.current) return;
      if (!param.point) return;
      const candle = seriesRef.current.candle as ISeriesApi<"Candlestick"> | null;
      const canvas = overlayRef.current;
      if (!candle || !canvas) return;
      const hit = hitTestDrawings(
        param.point.x,
        param.point.y,
        drawingsRef.current,
        chart,
        candle,
        canvas.clientWidth || ref.current?.clientWidth || 0,
        canvas.clientHeight || ref.current?.clientHeight || 0,
        toChartTime
      );
      if (hit) setSelectedDrawingId(hit.drawingId);
    };
    chart.subscribeClick(onClick);

    const onRange = () => {
      overlayRef.current?.dispatchEvent(new Event("chart-view-change"));
    };
    chart.timeScale().subscribeVisibleLogicalRangeChange(onRange);

    const onResize = () => {
      if (ref.current) chart.applyOptions({ width: ref.current.clientWidth, height: chartHeight });
      overlayRef.current?.dispatchEvent(new Event("chart-view-change"));
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      chart.unsubscribeCrosshairMove(onCrosshair);
      chart.unsubscribeClick(onClick);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(onRange);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = {};
    };
    // data는 별도 effect에서 setData
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme, toggles, chartHeight, colors, heightPlan]);

  // 데이터 주입
  useEffect(() => {
    const s = seriesRef.current;
    const chart = chartRef.current;
    if (!s.candle || !chart) return;

    if (!data?.bars?.length) {
      Object.values(s).forEach((ser) => ser?.setData([]));
      return;
    }

    const candles = data.bars.map((b) => ({
      time: toChartTime(b.time),
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }));
    s.candle.setData(candles);

    // 시장·가격대에 맞는 Y축 소수점
    const sampleClose =
      data.bars[data.bars.length - 1]?.close ??
      (Number(data.summary?.price || 0) || 1);
    const priceFmt = resolvePriceFormat(marketRef.current, sampleClose, symbolRef.current);
    s.candle.applyOptions({ priceFormat: priceFmt });
    s.sma?.applyOptions({ priceFormat: priceFmt });
    s.sma60?.applyOptions({ priceFormat: priceFmt });
    s.sma120?.applyOptions({ priceFormat: priceFmt });
    s.ema?.applyOptions({ priceFormat: priceFmt });
    s.bbUpper?.applyOptions({ priceFormat: priceFmt });
    s.bbMiddle?.applyOptions({ priceFormat: priceFmt });
    s.bbLower?.applyOptions({ priceFormat: priceFmt });

    const macdSample =
      (data.indicators.macd || []).slice(-1)[0]?.value ??
      (data.indicators.macd_signal || []).slice(-1)[0]?.value ??
      0;
    const macdFmt = resolveMacdFormat(macdSample, marketRef.current, symbolRef.current);
    s.macdLine?.applyOptions({ priceFormat: macdFmt });
    s.macdSignal?.applyOptions({ priceFormat: macdFmt });
    s.macdHist?.applyOptions({ priceFormat: macdFmt });

    const mapLine = (key: string) =>
      (data.indicators[key] || []).map((p) => ({
        time: toChartTime(p.time),
        value: p.value,
      }));

    s.sma?.setData(toggles.sma20 ? mapLine("sma_20") : []);
    s.sma60?.setData(toggles.sma60 ? mapLine("sma_60") : []);
    s.sma120?.setData(toggles.sma120 ? mapLine("sma_120") : []);
    s.ema?.setData(toggles.ema20 ? mapLine("ema_20") : []);

    if (toggles.bb) {
      s.bbUpper?.setData(mapLine("bb20_upper"));
      s.bbMiddle?.setData(mapLine("bb20_middle"));
      s.bbLower?.setData(mapLine("bb20_lower"));
    } else {
      s.bbUpper?.setData([]);
      s.bbMiddle?.setData([]);
      s.bbLower?.setData([]);
    }

    if (toggles.volume) {
      const vol = data.bars.map((b, i) => {
        const prev = i > 0 ? data.bars[i - 1].close : b.open;
        return {
          time: toChartTime(b.time),
          value: b.volume,
          color: b.close >= prev ? colors.up + "99" : colors.down + "99",
        };
      });
      s.volume?.setData(vol);
    } else {
      s.volume?.setData([]);
    }

    s.rsi?.setData(toggles.rsi ? mapLine("rsi") : []);

    if (toggles.macd) {
      const hist = (data.indicators.macd_histogram || []).map((p) => ({
        time: toChartTime(p.time),
        value: p.value,
        color: p.value >= 0 ? colors.up + "88" : colors.down + "88",
      }));
      s.macdHist?.setData(hist);
      s.macdLine?.setData(mapLine("macd"));
      s.macdSignal?.setData(mapLine("macd_signal"));
    } else {
      s.macdHist?.setData([]);
      s.macdLine?.setData([]);
      s.macdSignal?.setData([]);
    }

    if (toggles.stoch) {
      s.stochK?.setData(mapLine("stoch_k"));
      s.stochD?.setData(mapLine("stoch_d"));
    } else {
      s.stochK?.setData([]);
      s.stochD?.setData([]);
    }

    chart.timeScale().fitContent();
    overlayRef.current?.dispatchEvent(new Event("chart-view-change"));
  }, [data, toggles, colors]);

  // 드로잉 캔버스 페인트
  const repaintOverlay = () => {
    const canvas = overlayRef.current;
    const chart = chartRef.current;
    const candle = seriesRef.current.candle as ISeriesApi<"Candlestick"> | null;
    const stage = canvas?.parentElement;
    if (!canvas || !chart || !candle || !stage) return;

    const width = stage.clientWidth;
    const height = stage.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    paintDrawings(
      ctx,
      drawingsRef.current,
      selectedIdRef.current,
      pendingPointRef.current,
      hoverXYRef.current,
      chart,
      candle,
      width,
      height,
      theme,
      toChartTime
    );
  };

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const onView = () => repaintOverlay();
    canvas.addEventListener("chart-view-change", onView);
    repaintOverlay();
    return () => canvas.removeEventListener("chart-view-change", onView);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme, chartHeight, toggles, data]);

  useEffect(() => {
    repaintOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawings, selectedDrawingId, pendingPoint, theme]);

  const clientToLocal = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = overlayRef.current!;
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const resolveTimeFromCoord = (chartTime: unknown) =>
    resolveBarTime(chartTime, barLookupRef.current, barsRef.current);

  const placePointFromXY = (x: number, y: number): ChartPoint | null => {
    const chart = chartRef.current;
    const candle = seriesRef.current.candle as ISeriesApi<"Candlestick"> | null;
    if (!chart || !candle) return null;
    const price = candle.coordinateToPrice(y);
    const time = chart.timeScale().coordinateToTime(x);
    if (price == null || time == null) return null;
    const timeStr = resolveTimeFromCoord(time);
    if (!timeStr || !Number.isFinite(Number(price))) return null;
    return { time: timeStr, price: Number(price) };
  };

  const onOverlayPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const chart = chartRef.current;
    const candle = seriesRef.current.candle as ISeriesApi<"Candlestick"> | null;
    const canvas = overlayRef.current;
    if (!chart || !candle || !canvas) return;

    const { x, y } = clientToLocal(e);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const mode = drawModeRef.current;
    const pricePaneH =
      heightPlanRef.current.find((p) => p.id === "main")?.height ?? height;

    // 가격 패널 밖 클릭은 그리기/이동 대상이 아님 (지표 pane 독립축 보호)
    if (y > pricePaneH) {
      if (mode === "none") setSelectedDrawingId(null);
      return;
    }

    if (mode !== "none") {
      const point = placePointFromXY(x, y);
      if (!point) return;
      e.preventDefault();
      canvas.setPointerCapture(e.pointerId);
      const pending = pendingPointRef.current;
      if (!pending) {
        pendingPointRef.current = point;
        setPendingPoint(point);
        return;
      }
      const drawing = createDrawing(mode, pending, point);
      pendingPointRef.current = null;
      setPendingPoint(null);
      setDrawings((prev) => {
        const next = [...prev, drawing];
        saveDrawings(marketRef.current, symbolRef.current, next);
        return next;
      });
      setSelectedDrawingId(drawing.id);
      setDrawMode("none");
      return;
    }

    const hit: HitTarget = hitTestDrawings(
      x,
      y,
      drawingsRef.current,
      chart,
      candle,
      width,
      height,
      toChartTime
    );
    if (!hit) {
      setSelectedDrawingId(null);
      return;
    }

    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    const origin = drawingsRef.current.find((d) => d.id === hit.drawingId);
    if (!origin) return;
    setSelectedDrawingId(hit.drawingId);
    dragRef.current = {
      drawingId: hit.drawingId,
      part: hit.part,
      startX: x,
      startY: y,
      origin: { ...origin, a: { ...origin.a }, b: { ...origin.b } },
    };
  };

  const onOverlayPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const chart = chartRef.current;
    const candle = seriesRef.current.candle as ISeriesApi<"Candlestick"> | null;
    const canvas = overlayRef.current;
    if (!chart || !candle || !canvas) return;

    const { x, y } = clientToLocal(e);
    hoverXYRef.current = { x, y };

    const drag = dragRef.current;
    if (drag) {
      const next = applyDrag(
        drag.origin,
        drag.part,
        drag.startX,
        drag.startY,
        x,
        y,
        chart,
        candle,
        barsRef.current,
        resolveTimeFromCoord
      );
      drawingsRef.current = drawingsRef.current.map((d) =>
        d.id === drag.drawingId ? next : d
      );
      setOverlayCursor(drag.part === "body" ? "grabbing" : "nwse-resize");
      repaintOverlay();
      return;
    }

    if (drawModeRef.current !== "none") {
      setOverlayCursor("crosshair");
      if (pendingPointRef.current) repaintOverlay();
      return;
    }

    const hit = hitTestDrawings(
      x,
      y,
      drawingsRef.current,
      chart,
      candle,
      canvas.clientWidth,
      canvas.clientHeight,
      toChartTime
    );
    if (!hit) {
      setOverlayCursor("default");
      return;
    }
    setOverlayCursor(hit.part === "body" ? "grab" : "pointer");
  };

  const onOverlayPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (drag) {
      dragRef.current = null;
      const next = drawingsRef.current;
      setDrawings(next);
      saveDrawings(marketRef.current, symbolRef.current, next);
      setOverlayCursor("grab");
    }
    try {
      overlayRef.current?.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    repaintOverlay();
  };

  const toggleDrawMode = (mode: DrawMode) => {
    if (drawMode === mode) {
      setDrawMode("none");
      setPendingPoint(null);
      return;
    }
    setDrawMode(mode);
    setPendingPoint(null);
    setSelectedDrawingId(null);
  };

  const deleteDrawing = (id: string) => {
    persistDrawings((prev) => prev.filter((d) => d.id !== id));
    if (selectedDrawingId === id) setSelectedDrawingId(null);
  };

  const clearAllDrawings = () => {
    persistDrawings([]);
    setSelectedDrawingId(null);
    setPendingPoint(null);
  };

  const updateDrawingFlags = (
    id: string,
    patch: Partial<Pick<ChartDrawing, "extendLeft" | "extendRight">>
  ) => {
    persistDrawings((prev) =>
      prev.map((d) => (d.id === id ? { ...d, ...patch } : d))
    );
  };

  const empty = data && (!data.bars || data.bars.length === 0);
  const overlayActive = drawMode !== "none" || selectedDrawingId != null;
  const selectedDrawing = drawings.find((d) => d.id === selectedDrawingId) || null;
  const lastBarTime = data?.bars?.length ? data.bars[data.bars.length - 1].time : "";
  const barInProgress = !empty && lastBarTime ? isBarInProgress(timeframe, lastBarTime) : false;
  const drawHint =
    drawMode === "none"
      ? selectedDrawing
        ? "핸들·선을 드래그해 이동하세요. 연장 버튼으로 좌/우 연장을 켤 수 있습니다."
        : null
      : pendingPoint
        ? "두 번째 점을 클릭하세요 (Esc 취소)"
        : drawMode === "trendline"
          ? "추세선: 시작점을 클릭하세요"
          : "피보나치: 고점/저점을 클릭하세요";

  return (
    <div className="chart-panel">
      <div className="chart-toolbar">
        <div className="tf-controls" aria-label="차트 시간대">
          <div className="tf-tabs" role="tablist">
            {QUICK_TFS.map((tf) => (
              <button
                key={tf.id}
                type="button"
                className={`tf-tab ${timeframe === tf.id ? "active" : ""}`}
                onClick={() => onTimeframeChange(tf.id)}
                disabled={loading}
              >
                {tf.label}
              </button>
            ))}
          </div>
          <div className={`tf-select-wrap ${isMinuteTf(timeframe) ? "active" : ""}`}>
            <span className="tf-select-label">분봉</span>
            <select
              className="tf-select"
              value={minuteSelectValue}
              disabled={loading}
              onChange={(e) => onTimeframeChange(e.target.value as ChartTimeframe)}
              aria-label="분봉 간격"
            >
              {MINUTE_OPTIONS.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
          <div className={`tf-select-wrap ${isHourTf(timeframe) ? "active" : ""}`}>
            <span className="tf-select-label">시간</span>
            <select
              className="tf-select"
              value={hourSelectValue}
              disabled={loading}
              onChange={(e) => onTimeframeChange(e.target.value as ChartTimeframe)}
              aria-label="시간봉 간격"
            >
              {HOUR_OPTIONS.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="chart-tools-right">
          <div className="draw-tools" role="group" aria-label="차트 그리기 도구">
            <button
              type="button"
              className={`draw-tool-btn ${drawMode === "trendline" ? "active" : ""}`}
              onClick={() => toggleDrawMode("trendline")}
              disabled={loading || !!empty}
              title="추세선 (두 점 클릭)"
            >
              추세선
            </button>
            <button
              type="button"
              className={`draw-tool-btn ${drawMode === "fib" ? "active" : ""}`}
              onClick={() => toggleDrawMode("fib")}
              disabled={loading || !!empty}
              title="피보나치 되돌림 (두 점 클릭)"
            >
              피보나치
            </button>
          </div>
          <div className="indicator-dropdown" ref={indMenuRef}>
            <button
              type="button"
              className={`ind-dropdown-btn ${indMenuOpen ? "open" : ""}`}
              onClick={() => setIndMenuOpen((v) => !v)}
              aria-haspopup="listbox"
              aria-expanded={indMenuOpen}
            >
              지표 <span className="ind-count">{activeIndicatorCount}</span>
              <span className="ind-caret" aria-hidden>
                ▾
              </span>
            </button>
            {indMenuOpen && (
              <div className="ind-dropdown-menu" role="listbox" aria-label="지표 선택">
                <div className="ind-dropdown-head">표시할 지표</div>
                {INDICATOR_DEFS.map((d) => (
                  <label key={d.key} className="ind-dropdown-item">
                    <input
                      type="checkbox"
                      checked={toggles[d.key]}
                      onChange={() =>
                        setToggles((prev) => ({ ...prev, [d.key]: !prev[d.key] }))
                      }
                    />
                    <span>{d.label}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {drawHint && <div className="draw-hint">{drawHint}</div>}

      {drawings.length > 0 && (
        <div className="draw-list" aria-label="저장된 차트 그리기">
          {drawings.map((d, i) => (
            <div
              key={d.id}
              className={`draw-list-item ${selectedDrawingId === d.id ? "selected" : ""}`}
            >
              <button
                type="button"
                className="draw-list-select"
                onClick={() => setSelectedDrawingId(d.id)}
              >
                <span className={`draw-kind ${d.kind}`}>
                  {d.kind === "trendline" ? "TL" : "Fib"}
                </span>
                {drawingLabel(d, i)}
              </button>
              <button
                type="button"
                className={`draw-ext-btn ${d.extendLeft ? "active" : ""}`}
                onClick={() => updateDrawingFlags(d.id, { extendLeft: !d.extendLeft })}
                title="왼쪽 연장"
              >
                ←
              </button>
              <button
                type="button"
                className={`draw-ext-btn ${d.extendRight ? "active" : ""}`}
                onClick={() => updateDrawingFlags(d.id, { extendRight: !d.extendRight })}
                title="오른쪽 연장"
              >
                →
              </button>
              <button
                type="button"
                className="draw-list-delete"
                onClick={() => deleteDrawing(d.id)}
                aria-label={`${drawingLabel(d, i)} 삭제`}
                title="삭제"
              >
                ×
              </button>
            </div>
          ))}
          <button
            type="button"
            className="draw-clear-btn"
            onClick={clearAllDrawings}
            title="이 종목의 그리기 전체 삭제"
          >
            전체 삭제
          </button>
        </div>
      )}

      <div className="chart-stage" style={{ height: chartHeight }}>
        {ohlc && !empty && (
          <div className={`ohlc-legend ${ohlc.change >= 0 ? "up" : "down"}`}>
            <span className="ohlc-time">{formatLegendTime(ohlc.time)}</span>
            {barInProgress ? (
              <span className="chip bar-status live" title="마지막 봉은 장중/미확정일 수 있습니다">
                미확정
              </span>
            ) : (
              <span className="chip bar-status" title="확정 봉 기준">
                확정
              </span>
            )}
            <span>O {formatPrice(ohlc.open, market, symbol)}</span>
            <span>H {formatPrice(ohlc.high, market, symbol)}</span>
            <span>L {formatPrice(ohlc.low, market, symbol)}</span>
            <span>C {formatPrice(ohlc.close, market, symbol)}</span>
            <span>V {formatVol(ohlc.volume)}</span>
          </div>
        )}
        {!empty && (
          <div className="ind-legend" aria-label="지표 범례">
            {toggles.sma20 && (
              <span style={{ color: colors.sma }}>
                SMA20 {indLegend.sma20 != null ? formatPrice(indLegend.sma20, market, symbol) : "-"}
              </span>
            )}
            {toggles.sma60 && (
              <span style={{ color: colors.sma60 }}>
                SMA60 {indLegend.sma60 != null ? formatPrice(indLegend.sma60, market, symbol) : "-"}
              </span>
            )}
            {toggles.sma120 && (
              <span style={{ color: colors.sma120 }}>
                SMA120 {indLegend.sma120 != null ? formatPrice(indLegend.sma120, market, symbol) : "-"}
              </span>
            )}
            {toggles.ema20 && (
              <span style={{ color: colors.ema }}>
                EMA20 {indLegend.ema20 != null ? formatPrice(indLegend.ema20, market, symbol) : "-"}
              </span>
            )}
            {toggles.bb && (
              <span style={{ color: colors.bb }}>
                BB(20,2){" "}
                {indLegend.bbMiddle != null ? formatPrice(indLegend.bbMiddle, market, symbol) : "-"}
              </span>
            )}
            {toggles.rsi && (
              <span style={{ color: colors.rsi }}>
                RSI(14) {indLegend.rsi != null ? indLegend.rsi.toFixed(1) : "-"}
              </span>
            )}
            {toggles.macd && (
              <span style={{ color: colors.macd }}>
                MACD{" "}
                {indLegend.macd != null ? formatMacdValue(indLegend.macd, market, symbol) : "-"}
                {" / "}
                <span style={{ color: colors.signal }}>
                  Sig{" "}
                  {indLegend.macdSignal != null
                    ? formatMacdValue(indLegend.macdSignal, market, symbol)
                    : "-"}
                </span>
                {indLegend.macdHist != null && (
                  <>
                    {" · "}Hist {indLegend.macdHist >= 0 ? "+" : ""}
                    {formatMacdValue(indLegend.macdHist, market, symbol)}
                  </>
                )}
              </span>
            )}
          </div>
        )}
        <div
          className={`chart-wrap ${empty ? "chart-empty" : ""} ${drawMode !== "none" ? "drawing" : ""}`}
          ref={ref}
        >
          {empty && (
            <p className="muted chart-empty-msg">
              {isIntradayTf(timeframe)
                ? "분/시간봉 데이터를 불러오지 못했습니다. 종목코드·시장을 확인하거나 잠시 후 다시 시도해 주세요."
                : "차트 데이터가 없습니다. 종목코드와 시장을 확인한 뒤 새로고침해 주세요."}
            </p>
          )}
        </div>
        <canvas
          ref={overlayRef}
          className={`drawing-canvas ${overlayActive ? "active" : ""}`}
          style={{ cursor: overlayActive ? overlayCursor : "default" }}
          onPointerDown={onOverlayPointerDown}
          onPointerMove={onOverlayPointerMove}
          onPointerUp={onOverlayPointerUp}
          onPointerCancel={onOverlayPointerUp}
        />
        {loading && <div className="chart-loading">차트 로딩 중...</div>}
      </div>
    </div>
  );
}
