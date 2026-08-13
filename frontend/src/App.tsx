import { useEffect, useMemo, useRef, useState } from "react";
import {
  AccountOverview,
  api,
  ChartPayload,
  Health,
  HoldingItem,
  NewsItem,
  NewsSummary,
  Quote,
  RecommendItem,
  RuleAnalysis,
  ScanItem,
} from "./api";
import CandleChart, { ChartTimeframe } from "./components/CandleChart";
import MacroBoard from "./components/MacroBoard";
import MarketPulsePanel from "./components/MarketPulsePanel";
import NewsBriefingModal from "./components/NewsBriefingModal";
import OrderBookPanel from "./components/OrderBookPanel";
import { formatNewsTime } from "./formatTime";

type Workspace = "desk" | "recommend" | "account";
type RecMarket = "KRX" | "US" | "ALL";
type Tab = "ai" | "trade";
type Theme = "light" | "dark";

const MINUTE_TF: Record<string, number> = {
  "1m": 1,
  "3m": 3,
  "5m": 5,
  "10m": 10,
  "15m": 15,
  "30m": 30,
};
const HOUR_TF: Record<string, number> = {
  "1h": 1,
  "2h": 2,
  "4h": 4,
};

/** 종목명/뉴스 제목 등에서 티커만 추출 (잘못된 quote 호출 방지) */
function normalizeSymbolInput(raw: string, market: string): string {
  const s = (raw || "").trim();
  if (!s) return "";
  const kr = s.match(/(?<!\d)(\d{6})(?!\d)/);
  if (kr) return kr[1];
  if (/^[A-Za-z]{1,5}$/.test(s)) return s.toUpperCase();
  if (/^\d{6}$/.test(s)) return s;
  const noise = new Set(["HTTP", "HTTPS", "WWW", "API", "ETF", "ADR", "USD", "KRW", "KRX"]);
  const found = (s.match(/\b([A-Za-z]{1,5})\b/g) || [])
    .map((t) => t.toUpperCase())
    .filter((t) => !noise.has(t));
  if (found.length) {
    found.sort((a, b) => b.length - a.length || a.localeCompare(b));
    return found[0];
  }
  if (/^[A-Za-z0-9.\-]{1,12}$/.test(s) && !/[가-힣]/.test(s)) return s.toUpperCase();
  return "";
}

function isPlausibleSymbol(symbol: string): boolean {
  return /^\d{6}$/.test(symbol) || /^[A-Z]{1,5}$/.test(symbol) || /^[A-Z]{1,4}\.[A-Z]$/.test(symbol);
}

function chartQueryParams(symbol: string, market: string, timeframe: ChartTimeframe) {
  const params = new URLSearchParams({ symbol, market });
  if (timeframe in MINUTE_TF) {
    params.set("timeframe", "minute");
    params.set("minute_interval", String(MINUTE_TF[timeframe]));
    // yfinance: 1~5분 ≈ 7일, 15분+ ≈ 60일
    const mins = MINUTE_TF[timeframe];
    params.set("days", mins <= 10 ? "7" : "60");
  } else if (timeframe in HOUR_TF) {
    params.set("timeframe", "hour");
    params.set("hour_interval", String(HOUR_TF[timeframe]));
    params.set("days", "60");
  } else if (timeframe === "week") {
    params.set("timeframe", "week");
    params.set("days", "500");
  } else {
    params.set("timeframe", "day");
    params.set("days", "180");
  }
  return params;
}

const QUICK_QUESTIONS: Record<string, string[]> = {
  basic: ["지금 매수 타이밍인가요?", "단기 추세는 어떤가요?", "리스크 요인은?"],
  forecast_20d: ["향후 20일 전망은?", "목표가와 손절가를 알려주세요"],
  strategy: ["분할매수 전략 제안", "리스크 관리 방안은?"],
};

const STANCE_LABEL: Record<string, string> = {
  watch: "관망",
  buy: "매수 관심",
  accumulate: "분할매수 관심",
  avoid: "비중 축소/회피",
};

const BIAS_LABEL: Record<string, string> = {
  bullish: "강세",
  slightly_bullish: "약한 강세",
  neutral: "중립",
  slightly_bearish: "약한 약세",
  bearish: "약세",
};

const DIR_LABEL: Record<string, string> = {
  bullish: "강세",
  bearish: "약세",
  neutral: "중립",
};

function loadTheme(): Theme {
  const saved = localStorage.getItem("td_theme");
  if (saved === "dark" || saved === "light") return saved;
  return "light";
}

const REC_STORE_KEY = "td_daily_recommend";
const REC_MARKET_OPTIONS: { id: RecMarket; label: string }[] = [
  { id: "KRX", label: "국내" },
  { id: "US", label: "해외" },
  { id: "ALL", label: "국내/해외 통합" },
];

type RecSnapshot = {
  asOf: string;
  items: RecommendItem[];
  scanItems: ScanItem[];
  meta: { universe: number; scanned: number; shortlist: number; source: string };
  commentary: string;
  mode: string;
  cached: boolean;
};

type RecStore = {
  day: string;
  selected: RecMarket;
  byMarket: Partial<Record<RecMarket, RecSnapshot>>;
};

function todayStamp() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate()
  ).padStart(2, "0")}`;
}

function isRecMarket(value: string): value is RecMarket {
  return value === "KRX" || value === "US" || value === "ALL";
}

function recMarketLabel(market: RecMarket) {
  return REC_MARKET_OPTIONS.find((o) => o.id === market)?.label || "국내/해외 통합";
}

function recMarketHint(market: RecMarket) {
  if (market === "KRX") return "국내 시장 스캔 · AI 일일 추천";
  if (market === "US") return "해외 시장 스캔 · AI 일일 추천";
  return "국내·해외 통합 스캔 · AI 일일 추천";
}

function loadRecStore(): RecStore {
  const empty: RecStore = { day: todayStamp(), selected: "ALL", byMarket: {} };
  try {
    const raw = localStorage.getItem(REC_STORE_KEY);
    if (!raw) return empty;
    const parsed = JSON.parse(raw) as RecStore;
    const selected = isRecMarket(parsed?.selected) ? parsed.selected : "ALL";
    if (parsed?.day !== todayStamp()) {
      return { day: todayStamp(), selected, byMarket: {} };
    }
    return { day: parsed.day, selected, byMarket: parsed.byMarket || {} };
  } catch {
    return empty;
  }
}

function persistRecStore(patch: {
  selected?: RecMarket;
  snapshot?: { market: RecMarket; data: RecSnapshot };
}) {
  const cur = loadRecStore();
  const next: RecStore = {
    day: todayStamp(),
    selected: patch.selected || cur.selected,
    byMarket: { ...cur.byMarket },
  };
  if (patch.snapshot) {
    next.byMarket[patch.snapshot.market] = patch.snapshot.data;
  }
  localStorage.setItem(REC_STORE_KEY, JSON.stringify(next));
}

function recModeFromApi(r: { used_llm?: boolean; provider?: string }, hasItems: boolean) {
  if (!hasItems) return "";
  const engine =
    r.provider && r.provider !== "none"
      ? r.provider === "nvidia"
        ? "NVIDIA"
        : r.provider === "ollama"
          ? "로컬"
          : r.provider
      : "";
  if (r.used_llm) return engine ? `AI 추천 (${engine})` : "AI 추천";
  return "기술지표 추천";
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(loadTheme);
  const [health, setHealth] = useState<Health | null>(null);
  const [market, setMarket] = useState("KRX");
  const [symbol, setSymbol] = useState("005930");
  const [popular, setPopular] = useState<{ symbol: string; name: string }[]>([]);
  const [quote, setQuote] = useState<Quote | null>(null);
  const [chart, setChart] = useState<ChartPayload | null>(null);
  const [tab, setTab] = useState<Tab>("ai");
  const [workspace, setWorkspace] = useState<Workspace>("desk");
  const [news, setNews] = useState<NewsItem[]>([]);
  const [newsSort, setNewsSort] = useState<"importance" | "date">("importance");
  const [stockNewsOpen, setStockNewsOpen] = useState(false);
  const [stockNewsLoading, setStockNewsLoading] = useState(false);
  const [stockNewsSummary, setStockNewsSummary] = useState<NewsSummary | null>(null);
  const [stockNewsActive, setStockNewsActive] = useState<NewsItem | null>(null);
  const [stockName, setStockName] = useState("");
  const [recMarket, setRecMarket] = useState<RecMarket>(() => loadRecStore().selected);
  const recMarketRef = useRef<RecMarket>(recMarket);
  const [recs, setRecs] = useState<RecommendItem[]>(() => {
    const store = loadRecStore();
    return store.byMarket[store.selected]?.items || [];
  });
  const [scanItems, setScanItems] = useState<ScanItem[]>(() => {
    const store = loadRecStore();
    return store.byMarket[store.selected]?.scanItems || [];
  });
  const [showScanAll, setShowScanAll] = useState(false);
  const [recMeta, setRecMeta] = useState(() => {
    const store = loadRecStore();
    return store.byMarket[store.selected]?.meta || { universe: 0, scanned: 0, shortlist: 0, source: "" };
  });
  const [recAsOf, setRecAsOf] = useState(() => {
    const store = loadRecStore();
    return store.byMarket[store.selected]?.asOf || "";
  });
  const [recCommentary, setRecCommentary] = useState(() => {
    const store = loadRecStore();
    return store.byMarket[store.selected]?.commentary || "";
  });
  const [recMode, setRecMode] = useState(() => {
    const store = loadRecStore();
    return store.byMarket[store.selected]?.mode || "";
  });
  const [recCached, setRecCached] = useState(() => {
    const store = loadRecStore();
    return !!store.byMarket[store.selected]?.cached;
  });
  const [analysisType, setAnalysisType] = useState("basic");
  const [llmProvider, setLlmProvider] = useState("nvidia");
  const [query, setQuery] = useState("지금 매수 타이밍인가요?");
  const [answer, setAnswer] = useState("");
  const [answerProvider, setAnswerProvider] = useState("");
  const [ruleAnalysis, setRuleAnalysis] = useState<RuleAnalysis | null>(null);
  const [ruleLoading, setRuleLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [qty, setQty] = useState(1);
  const [orderType, setOrderType] = useState<"market" | "limit">("market");
  const [limitPrice, setLimitPrice] = useState<number | "">("");
  const [orderMsg, setOrderMsg] = useState("");
  const [account, setAccount] = useState<AccountOverview | null>(null);
  const [accountLoading, setAccountLoading] = useState(false);
  const [chartTf, setChartTf] = useState<ChartTimeframe>("day");
  const [marketPulseOpen, setMarketPulseOpen] = useState(false);

  const priceColor = useMemo(() => {
    const rate = quote?.rate ?? Number(chart?.summary?.price_change ?? 0);
    return rate >= 0 ? "up" : "down";
  }, [quote, chart]);

  useEffect(() => {
    recMarketRef.current = recMarket;
  }, [recMarket]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("td_theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!stockNewsOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setStockNewsOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [stockNewsOpen]);

  async function loadRuleAnalysis(nextSymbol = symbol, nextMarket = market, name = stockName) {
    setRuleLoading(true);
    try {
      const r = await api.ruleAnalysis(nextSymbol, nextMarket, name);
      setRuleAnalysis(r);
      if (r.stock_name) setStockName((prev) => prev || r.stock_name);
    } catch {
      setRuleAnalysis(null);
    } finally {
      setRuleLoading(false);
    }
  }

  async function refreshCore(nextTf: ChartTimeframe = chartTf) {
    setError("");
    setRefreshing(true);
    setRuleLoading(true);
    setAnswer("");
    setAnswerProvider("");
    const sym = normalizeSymbolInput(symbol, market) || symbol.trim();
    if (sym && sym !== symbol) {
      setSymbol(sym);
    }
    if (!isPlausibleSymbol(sym)) {
      setError("종목코드/티커를 확인해 주세요. 예: 005930, AAPL, BMNR");
      setRefreshing(false);
      setRuleLoading(false);
      return;
    }
    try {
      const params = chartQueryParams(sym, market, nextTf);
      // 차트/뉴스/룰분석: 종목 선택 시 지표 기반 룰 분석을 먼저 확보
      const [c, n, rules] = await Promise.all([
        api.chart(params),
        api.news(sym, market, {
          prepare: true,
          provider: llmProvider === "auto" ? "" : llmProvider,
          sort: newsSort,
        }),
        api.ruleAnalysis(sym, market, stockName).catch(() => null),
      ]);
      setChart(c);
      setNews(n.items || []);
      setStockName(n.stock_name || sym);
      setRuleAnalysis(rules);

      const summaryPrice = Number(c?.summary?.price || 0);
      const summaryRate = Number(c?.summary?.price_change || 0);
      if (summaryPrice > 0) {
        setQuote({
          symbol: sym,
          name: n.stock_name || sym,
          market,
          price: summaryPrice,
          change: 0,
          rate: summaryRate,
          volume: 0,
        });
      }

      // 시세 API는 후순위(캐시됨). 실패해도 차트 요약으로 표시 유지
      try {
        const q = await api.quote(sym, market);
        if (q?.price) {
          setQuote(q);
          setStockName(n.stock_name || q.name || sym);
        }
      } catch {
        /* ignore quote errors to protect account rate limit */
      }
    } catch (e: any) {
      setError(e.message || "시세/뉴스를 불러오지 못했습니다.");
    } finally {
      setRefreshing(false);
      setRuleLoading(false);
    }
  }

  function handleTimeframeChange(tf: ChartTimeframe) {
    setChartTf(tf);
    refreshCore(tf);
  }

  async function loadAccount() {
    setAccountLoading(true);
    try {
      const a = await api.account();
      setAccount(a);
    } catch {
      setAccount(null);
    } finally {
      setAccountLoading(false);
    }
  }

  // StrictMode 이중 호출 대비: 마운트 시 순차 로딩 (계좌 → 시세)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const h = await api.health();
        if (!cancelled) setHealth(h);
      } catch {
        if (!cancelled) setHealth(null);
      }
      if (cancelled) return;
      await loadAccount();
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    api.popular(market).then((r) => setPopular(r.items || [])).catch(() => setPopular([]));
  }, [market]);

  function openHolding(h: HoldingItem) {
    setSymbol(h.symbol);
    const isKr = (h.scope || "").toLowerCase() === "domestic" || /^\d+$/.test(h.symbol);
    setMarket(isKr ? "KRX" : "US");
    setWorkspace("desk");
    setTab("ai");
  }

  function openSymbolOnDesk(nextSymbol: string, nextMarket?: string) {
    const m =
      nextMarket ||
      (/^\d{6}$/.test(nextSymbol) ? "KRX" : "US");
    setMarket(m);
    setSymbol(nextSymbol);
    setWorkspace("desk");
    setTab("ai");
  }

  function goWorkspace(next: Workspace) {
    setWorkspace(next);
    if (next === "account") loadAccount();
    if (next === "recommend") {
      const store = loadRecStore();
      const snap = store.byMarket[recMarket];
      if (snap && (snap.items.length || snap.scanItems.length)) {
        applyRecSnapshot(snap);
      } else {
        void loadRecommend({ market: recMarket, force: false, stay: true });
      }
    }
  }

  function formatMoney(value: number, currency: "KRW" | "USD" = "KRW") {
    if (currency === "USD") {
      const abs = Math.abs(value);
      const digits = abs >= 1 ? 2 : abs >= 0.01 ? 4 : 6;
      return `$${value.toLocaleString(undefined, {
        minimumFractionDigits: 0,
        maximumFractionDigits: digits,
      })}`;
    }
    return `${Math.round(value).toLocaleString("ko-KR")}원`;
  }

  function formatQuotePrice(value: number) {
    if (!Number.isFinite(value)) return "-";
    if (market === "KRX") return Math.round(value).toLocaleString("ko-KR");
    const abs = Math.abs(value);
    const digits = abs >= 1 ? 2 : abs >= 0.1 ? 3 : abs >= 0.01 ? 4 : 6;
    return value.toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits: digits,
    });
  }

  useEffect(() => {
    const t = window.setTimeout(() => {
      refreshCore(chartTf);
    }, 120);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, market]);

  useEffect(() => {
    setLimitPrice("");
    setOrderType("market");
    setOrderMsg("");
  }, [symbol, market]);

  async function openStockNewsBriefing(item: NewsItem) {
    setStockNewsActive(item);
    setStockNewsOpen(true);
    setStockNewsLoading(true);
    setStockNewsSummary(null);
    try {
      const r = await api.summarizeNews({
        url: item.url,
        title: item.title_original || item.title,
        snippet: item.summary,
        source: item.source,
        provider: llmProvider === "auto" ? "" : llmProvider,
      });
      setStockNewsSummary(r);
      const titleKo = r.title_ko || r.title || item.title;
      setStockNewsActive((prev) => (prev ? { ...prev, title: titleKo } : prev));
      setNews((prev) =>
        prev.map((x) =>
          x.url === item.url
            ? {
                ...x,
                has_ai_summary: true,
                title: titleKo,
                title_original: x.title_original || item.title_original || item.title,
                importance: r.importance ?? x.importance,
                importance_reason: r.importance_reason || x.importance_reason,
              }
            : x
        )
      );
    } catch (e: any) {
      setStockNewsSummary({
        id: "",
        url: item.url,
        title: item.title,
        summary_ko: e.message || "요약을 불러오지 못했습니다.",
        provider: "none",
        cached: false,
      });
    } finally {
      setStockNewsLoading(false);
    }
  }

  function applyRecSnapshot(snap: RecSnapshot | null) {
    if (!snap) {
      setRecs([]);
      setScanItems([]);
      setRecMeta({ universe: 0, scanned: 0, shortlist: 0, source: "" });
      setRecAsOf("");
      setRecCommentary("");
      setRecMode("");
      setRecCached(false);
      return;
    }
    setRecs(snap.items);
    setScanItems(snap.scanItems);
    setRecMeta(snap.meta);
    setRecAsOf(snap.asOf);
    setRecCommentary(snap.commentary);
    setRecMode(snap.mode);
    setRecCached(snap.cached);
  }

  function selectRecMarket(next: RecMarket) {
    setRecMarket(next);
    persistRecStore({ selected: next });
    const store = loadRecStore();
    const snap = store.byMarket[next];
    if (snap && snap.asOf === todayStamp() && (snap.items.length || snap.scanItems.length)) {
      applyRecSnapshot(snap);
      return;
    }
    applyRecSnapshot(null);
    void loadRecommend({ market: next, force: false, stay: true });
  }

  async function loadRecommend(opts?: { market?: RecMarket; force?: boolean; stay?: boolean }) {
    const mkt = opts?.market || recMarketRef.current;
    const force = !!opts?.force;
    if (force) {
      setLoading(true);
      setError("");
    }
    try {
      const r = await api.recommend(mkt, {
        provider: llmProvider === "auto" ? "" : llmProvider,
        force,
      });
      if (recMarketRef.current !== mkt) return;
      const items = r.items || [];
      const scans = r.scan_items || [];
      const snap: RecSnapshot = {
        asOf: r.as_of,
        items,
        scanItems: scans,
        meta: {
          universe: r.universe_size || 0,
          scanned: r.scanned_count || scans.length,
          shortlist: r.shortlist_size || 0,
          source: r.universe_source || "",
        },
        commentary: r.market_commentary || "",
        mode: recModeFromApi(r, items.length > 0 || scans.length > 0),
        cached: !!r.cached,
      };
      applyRecSnapshot(snap);
      if (items.length || scans.length) {
        persistRecStore({ selected: mkt, snapshot: { market: mkt, data: snap } });
      } else {
        persistRecStore({ selected: mkt });
      }
      if (!opts?.stay) setWorkspace("recommend");
    } catch (e: any) {
      if (recMarketRef.current !== mkt) return;
      if (force) setError(e.message || "일일 추천 생성에 실패했습니다.");
    } finally {
      if (force) setLoading(false);
    }
  }

  useEffect(() => {
    const store = loadRecStore();
    const snap = store.byMarket[store.selected];
    if (snap && (snap.items.length || snap.scanItems.length)) return;
    void loadRecommend({ market: store.selected, force: false, stay: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runAnalysis() {
    setLoading(true);
    setError("");
    setAnswer("");
    try {
      // 최신 룰 분석을 먼저 확보한 뒤 AI가 체크리스트를 평가하도록 요청
      if (!ruleAnalysis || ruleAnalysis.symbol !== symbol) {
        await loadRuleAnalysis(symbol, market, stockName);
      }
      const r = await api.analyze({
        symbol,
        market,
        stock_name: stockName,
        query,
        analysis_type: analysisType,
        days: 120,
        provider: llmProvider === "auto" ? "" : llmProvider,
      });
      setAnswer(r.answer);
      setAnswerProvider(r.provider || "");
      if (r.rule_analysis) setRuleAnalysis(r.rule_analysis);
      setTab("ai");
      setWorkspace("desk");
    } catch (e: any) {
      setError(e.message || "AI 분석에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  }

  async function submitOrder() {
    setOrderMsg("");
    try {
      if (orderType === "limit") {
        const p = typeof limitPrice === "number" ? limitPrice : Number(limitPrice);
        if (!Number.isFinite(p) || p <= 0) {
          setOrderMsg("지정가 주문은 호가를 클릭하거나 가격을 입력하세요.");
          return;
        }
      }
      const payload: {
        symbol: string;
        side: "buy" | "sell";
        qty: number;
        order_type: "market" | "limit";
        price?: number;
      } = {
        symbol,
        side,
        qty,
        order_type: orderType,
      };
      if (orderType === "limit") {
        const p = typeof limitPrice === "number" ? limitPrice : Number(limitPrice);
        payload.price = market === "KRX" ? Math.round(p) : p;
      }
      const r: any = await api.order(payload);
      setOrderMsg(r.message || (r.success ? "주문이 접수되었습니다." : "주문 실패"));
    } catch (e: any) {
      setOrderMsg(e.message || "주문 요청에 실패했습니다.");
    }
  }

  function applyOrderBookPrice(nextPrice: number, nextSide: "buy" | "sell") {
    setSide(nextSide);
    setOrderType("limit");
    setLimitPrice(market === "KRX" ? Math.round(nextPrice) : nextPrice);
    setOrderMsg(
      nextSide === "buy"
        ? `지정가 매수 ${formatQuotePrice(nextPrice)} 선택`
        : `지정가 매도 ${formatQuotePrice(nextPrice)} 선택`
    );
  }

  const price = quote?.price || Number(chart?.summary?.price || 0);
  const rate = quote?.rate ?? Number(chart?.summary?.price_change ?? 0);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-left">
          <h1 className="brand">
            Trading<span>Desk</span>
          </h1>
          <nav className="workspace-nav" aria-label="주요 화면">
            {(
              [
                ["desk", "종목 데스크"],
                ["recommend", "일일 추천"],
                ["account", "내 계좌"],
              ] as [Workspace, string][]
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={`workspace-tab ${workspace === id ? "active" : ""}`}
                onClick={() => goWorkspace(id)}
              >
                {label}
              </button>
            ))}
          </nav>
          <div className="top-status">
            <span className="status-pill">
              <span className={`status-dot ${health?.kis_connected ? "on" : "off"}`} />
              거래연동 {health?.kis_connected ? "연결됨" : "미연결"}
            </span>
            <span className="status-pill">
              <span
                className={`status-dot ${
                  health?.ai_connected || health?.ollama_connected || health?.nvidia_connected
                    ? "on"
                    : "off"
                }`}
              />
              AI{" "}
              {health?.ai_connected || health?.ollama_connected || health?.nvidia_connected
                ? "준비됨"
                : "오프라인"}
              {health?.ollama_connected || health?.nvidia_connected
                ? ` · 로컬${health?.ollama_connected ? "✓" : "–"}/NVIDIA${
                    health?.nvidia_connected ? "✓" : "–"
                  }`
                : ""}
            </span>
          </div>
        </div>
        <div className="topbar-actions">
          {(loading || refreshing) && (
            <span className="loading-inline">
              <span className="spinner" />
              {loading ? "처리 중..." : "새로고침 중..."}
            </span>
          )}
          <button
            className="btn secondary"
            onClick={() => setMarketPulseOpen(true)}
            title="한국·미국·세계 시황 뉴스"
          >
            시황 뉴스
          </button>
          <button
            className="btn ghost"
            onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
            title="테마 전환"
          >
            {theme === "light" ? "다크 모드" : "라이트 모드"}
          </button>
          <button
            className="btn icon-btn"
            onClick={() => {
              if (workspace === "account") {
                loadAccount();
              } else {
                refreshCore();
              }
            }}
            disabled={refreshing || loading || accountLoading}
            title="새로고침"
            aria-label="새로고침"
          >
            <span className={`refresh-icon ${refreshing || accountLoading ? "spin" : ""}`} aria-hidden>
              ↻
            </span>
          </button>
        </div>
      </header>

      <MacroBoard />

      <div className={`app app-workspace-${workspace}`}>
        <aside className="panel">
          {workspace === "desk" && (
            <>
              <p className="sub">종목 선택 · 차트 · AI 분석 · 뉴스</p>

              <div className="section-title">종목</div>
              <div className="field">
                <label>시장</label>
                <select
                  value={market}
                  onChange={(e) => {
                    const next = e.target.value;
                    setMarket(next);
                    if (next === "US") {
                      setSymbol("AAPL");
                      setStockName("Apple");
                    } else {
                      setSymbol("005930");
                      setStockName("Samsung Electronics");
                    }
                  }}
                >
                  <option value="KRX">국내 (KRX)</option>
                  <option value="US">해외 (US)</option>
                </select>
              </div>
              <div className="field">
                <label>종목코드 / 티커</label>
                <input
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  onBlur={() => {
                    const n = normalizeSymbolInput(symbol, market);
                    if (n) setSymbol(n);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      const n = normalizeSymbolInput(symbol, market);
                      if (n) setSymbol(n);
                      refreshCore();
                    }
                  }}
                  placeholder="예: 005930, AAPL, BMNR"
                />
              </div>
              <div className="field">
                <label>인기 종목</label>
                <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
                  {popular.map((p) => (
                    <option key={p.symbol} value={p.symbol}>
                      {p.name} ({p.symbol})
                    </option>
                  ))}
                </select>
              </div>
              <button
                className="btn secondary btn-block"
                onClick={() => goWorkspace("account")}
                disabled={accountLoading}
              >
                {accountLoading ? "계좌 조회 중..." : "내 계좌 보기"}
              </button>

              <div className="section-title">AI 분석</div>
              <div className="field">
                <label>AI 엔진</label>
                <select value={llmProvider} onChange={(e) => setLlmProvider(e.target.value)}>
                  <option value="nvidia">NVIDIA API</option>
                  <option value="ollama">로컬 (Ollama)</option>
                  <option value="auto">자동 전환 (번갈아 사용)</option>
                </select>
              </div>
              <div className="field">
                <label>분석 유형</label>
                <select value={analysisType} onChange={(e) => setAnalysisType(e.target.value)}>
                  <option value="basic">기본 분석</option>
                  <option value="forecast_20d">20일 전망</option>
                  <option value="strategy">투자 전략</option>
                </select>
              </div>
              <div className="quick-asks">
                {(QUICK_QUESTIONS[analysisType] || []).map((q) => (
                  <button key={q} className="quick-ask" onClick={() => setQuery(q)} type="button">
                    {q}
                  </button>
                ))}
              </div>
              <div className="field">
                <label>질문</label>
                <textarea
                  rows={3}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runAnalysis();
                  }}
                  placeholder="궁금한 점을 입력하세요 (Ctrl/Cmd + Enter)"
                />
              </div>
              <button
                className="btn btn-block"
                onClick={runAnalysis}
                disabled={loading || !query.trim()}
              >
                {loading ? "분석 중..." : "AI 분석 실행"}
              </button>
              {error && <div className="error-box">{error}</div>}
            </>
          )}

          {workspace === "recommend" && (
            <>
              <p className="sub">{recMarketHint(recMarket)}</p>
              <div className="section-title">추천 범위</div>
              <div className="rec-market-tabs" role="group" aria-label="추천 시장">
                {REC_MARKET_OPTIONS.map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    className={`rec-market-tab ${recMarket === opt.id ? "active" : ""}`}
                    onClick={() => selectRecMarket(opt.id)}
                    disabled={loading}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <div className="section-title">추천 설정</div>
              <div className="field">
                <label>AI 엔진</label>
                <select value={llmProvider} onChange={(e) => setLlmProvider(e.target.value)}>
                  <option value="nvidia">NVIDIA API</option>
                  <option value="ollama">로컬 (Ollama)</option>
                  <option value="auto">자동 전환 (번갈아 사용)</option>
                </select>
              </div>
              <button
                className="btn btn-block"
                onClick={() => void loadRecommend({ market: recMarket, force: true })}
                disabled={loading}
              >
                {loading
                  ? "추천 생성 중..."
                  : recs.length || scanItems.length
                    ? "오늘 추천 업데이트"
                    : "일일 추천 실행"}
              </button>
              {recAsOf ? (
                <p className="muted" style={{ marginTop: 8 }}>
                  {recCached
                    ? `당일 저장분 유지 중 (${recAsOf})`
                    : `오늘 생성됨 (${recAsOf})`}
                </p>
              ) : (
                <p className="muted" style={{ marginTop: 8 }}>
                  당일 추천은 업데이트하기 전까지 유지됩니다.
                </p>
              )}
              {error && <div className="error-box">{error}</div>}
            </>
          )}

          {workspace === "account" && (
            <>
              <p className="sub">통합 잔고 · 국내/해외 보유</p>
              <div className="section-title">계좌</div>
              <button className="btn btn-block" onClick={loadAccount} disabled={accountLoading}>
                {accountLoading ? "조회 중..." : "계좌 새로고침"}
              </button>
              <button
                className="btn secondary btn-block"
                style={{ marginTop: 8 }}
                onClick={() => goWorkspace("desk")}
              >
                종목 데스크로
              </button>
            </>
          )}
        </aside>

        <main className="panel">
          {workspace === "desk" && (
            <>
              <div className="main-head">
                <div className="price-box">
                  <h2>
                    {stockName || symbol}{" "}
                    <span className="muted">({symbol})</span>
                  </h2>
                  <div className={`meta ${priceColor}`}>
                    {formatQuotePrice(price)} {market === "KRX" ? "원" : "USD"} ·{" "}
                    {rate >= 0 ? "+" : ""}
                    {rate.toFixed(2)}%
                  </div>
                </div>
              </div>

              <div className="metrics">
                <div className="metric">
                  <div className="label">RSI(14)</div>
                  <div className="value">{Number(chart?.summary?.rsi ?? 0).toFixed(1)}</div>
                </div>
                <div className="metric">
                  <div className="label">MACD</div>
                  <div className="value">{String(chart?.summary?.macd_signal ?? "-")}</div>
                </div>
                <div className="metric">
                  <div className="label">추세</div>
                  <div className="value">{String(chart?.summary?.trend_long ?? "-")}</div>
                </div>
                <div className="metric">
                  <div className="label">거래량 비율</div>
                  <div className="value">
                    {Number(chart?.summary?.volume_ratio ?? 0).toFixed(2)}x
                  </div>
                </div>
              </div>

              <CandleChart
                data={chart}
                theme={theme}
                timeframe={chartTf}
                onTimeframeChange={handleTimeframeChange}
                loading={refreshing}
                symbol={symbol}
                market={market}
              />

              <div className="tabs">
                {(
                  [
                    ["ai", "AI 분석"],
                    ["trade", "주문"],
                  ] as [Tab, string][]
                ).map(([id, label]) => (
                  <button
                    key={id}
                    className={`tab ${tab === id ? "active" : ""}`}
                    onClick={() => setTab(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {tab === "ai" && (
                <div className="list">
                  <article className="card-item">
                    <h4>
                      지표 룰 분석
                      {ruleAnalysis?.as_of ? (
                        <span className="chip" style={{ marginLeft: 8 }}>
                          {ruleAnalysis.as_of}
                        </span>
                      ) : null}
                    </h4>
                    {ruleLoading && !ruleAnalysis && (
                      <p className="muted">지표 룰 분석 불러오는 중...</p>
                    )}
                    {!ruleLoading && !ruleAnalysis && (
                      <p className="muted">종목을 선택하면 지표 기반 룰 분석이 표시됩니다.</p>
                    )}
                    {ruleAnalysis && (
                      <>
                        <div className="row" style={{ gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
                          <span className="chip accent">점수 {ruleAnalysis.score}/100</span>
                          <span className="chip">
                            {STANCE_LABEL[ruleAnalysis.stance] || ruleAnalysis.stance}
                          </span>
                          <span className="chip">
                            {BIAS_LABEL[ruleAnalysis.bias] || ruleAnalysis.bias}
                          </span>
                          <span className="chip">
                            {formatQuotePrice(Number(ruleAnalysis.price || 0))}
                          </span>
                        </div>

                        <div className="horizon-box">
                          <div className="horizon-row">
                            <span className="label">중기추세</span>
                            <strong>
                              {ruleAnalysis.horizon?.medium_trend_label ||
                                (String(ruleAnalysis.metrics.trend_long) === "UP"
                                  ? "중기추세 상승"
                                  : "중기추세 하락")}
                            </strong>
                          </div>
                          <div className="horizon-row">
                            <span className="label">단기모멘텀</span>
                            <strong>
                              {ruleAnalysis.horizon?.short_momentum_label ||
                                (String(ruleAnalysis.metrics.macd_bias) === "bullish"
                                  ? "단기모멘텀 개선"
                                  : "단기모멘텀 둔화")}
                            </strong>
                          </div>
                          <p className="horizon-narrative">
                            {ruleAnalysis.horizon?.narrative || ruleAnalysis.summary_text}
                          </p>
                          <p className="macd-evidence">
                            {ruleAnalysis.horizon?.macd_evidence ||
                              String(ruleAnalysis.metrics.macd_evidence || "")}
                            {ruleAnalysis.horizon?.rsi_zone
                              ? ` · RSI 구간: ${ruleAnalysis.horizon.rsi_zone}`
                              : ""}
                          </p>
                        </div>

                        <div className="rule-metrics">
                          <span>RSI {String(ruleAnalysis.metrics.rsi ?? "-")}</span>
                          <span>
                            MACD {String(ruleAnalysis.metrics.macd ?? "-")} / Sig{" "}
                            {String(ruleAnalysis.metrics.macd_signal_line ?? "-")}
                          </span>
                          <span>Hist {String(ruleAnalysis.metrics.macd_histogram ?? "-")}</span>
                          <span>BB {String(ruleAnalysis.metrics.bb_position ?? "-")}</span>
                          <span>Vol {String(ruleAnalysis.metrics.volume_ratio ?? "-")}x</span>
                        </div>
                        <ul className="rule-list">
                          {ruleAnalysis.rules.map((r) => (
                            <li key={r.id} className={`rule-item ${r.direction}`}>
                              <strong>
                                [{DIR_LABEL[r.direction] || r.direction}] {r.title}
                              </strong>
                              <span>
                                {r.detail}
                                {typeof r.weight === "number"
                                  ? ` (${r.weight >= 0 ? "+" : ""}${r.weight})`
                                  : ""}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </>
                    )}
                  </article>

                  <article className="card-item">
                    <h4>AI 분석 결과</h4>
                    {answerProvider && (
                      <p className="muted" style={{ marginBottom: 8 }}>
                        사용 엔진: {answerProvider === "nvidia" ? "NVIDIA API" : "로컬"}
                        {" · "}룰 체크리스트를 반영해 평가합니다
                      </p>
                    )}
                    <p className="answer">
                      {answer ||
                        "위에서 지표 룰 분석을 확인한 뒤, 왼쪽에서 질문을 입력하고 AI 분석을 실행하세요."}
                    </p>
                  </article>
                </div>
              )}

              {tab === "trade" && (
                <div className="trade-layout">
                  <OrderBookPanel
                    symbol={symbol}
                    market={market}
                    active={workspace === "desk" && tab === "trade"}
                    pollMs={2000}
                    selectedPrice={typeof limitPrice === "number" ? limitPrice : null}
                    onPriceClick={applyOrderBookPrice}
                  />
                  <div className="card-item trade-form">
                    <h4>주문</h4>
                    <p className="muted" style={{ marginBottom: 10 }}>
                      호가를 클릭하면 지정가가 채워집니다. 실제 주문이 체결될 수 있으니 모의투자를
                      권장합니다.
                    </p>
                    <div className="row">
                      <button
                        className={`btn ${side === "buy" ? "" : "secondary"}`}
                        onClick={() => setSide("buy")}
                      >
                        매수
                      </button>
                      <button
                        className={`btn ${side === "sell" ? "danger" : "secondary"}`}
                        onClick={() => setSide("sell")}
                      >
                        매도
                      </button>
                    </div>
                    <div className="row" style={{ marginTop: 10 }}>
                      <button
                        className={`btn ${orderType === "market" ? "" : "secondary"}`}
                        onClick={() => setOrderType("market")}
                      >
                        시장가
                      </button>
                      <button
                        className={`btn ${orderType === "limit" ? "" : "secondary"}`}
                        onClick={() => setOrderType("limit")}
                      >
                        지정가
                      </button>
                    </div>
                    {orderType === "limit" && (
                      <div className="field" style={{ marginTop: 10 }}>
                        <label>지정가 ({market === "KRX" ? "원" : "USD"})</label>
                        <input
                          type="number"
                          min={0}
                          step={market === "KRX" ? 1 : 0.01}
                          value={limitPrice}
                          placeholder="호가 클릭 또는 직접 입력"
                          onChange={(e) => {
                            const v = e.target.value;
                            if (v === "") {
                              setLimitPrice("");
                              return;
                            }
                            const n = Number(v);
                            setLimitPrice(Number.isFinite(n) ? n : "");
                          }}
                        />
                      </div>
                    )}
                    <div className="field" style={{ marginTop: 10 }}>
                      <label>수량</label>
                      <input
                        type="number"
                        min={1}
                        value={qty}
                        onChange={(e) => setQty(Number(e.target.value) || 1)}
                      />
                    </div>
                    <button
                      className={`btn ${side === "sell" ? "danger" : ""}`}
                      onClick={submitOrder}
                    >
                      {orderType === "limit" ? "지정가" : "시장가"}{" "}
                      {side === "buy" ? "매수" : "매도"} 주문
                      {orderType === "limit" && typeof limitPrice === "number"
                        ? ` @ ${formatQuotePrice(limitPrice)}`
                        : ""}
                    </button>
                    {orderMsg && <p className="muted" style={{ marginTop: 10 }}>{orderMsg}</p>}
                  </div>
                </div>
              )}
            </>
          )}

          {workspace === "recommend" && (
            <div className="list workspace-page">
              <div className="main-head">
                <div className="price-box">
                  <h2>일일 추천</h2>
                  <div className="meta muted">
                    {recMarketLabel(recMarket)} · 기준일 {recAsOf || "-"}
                    {recMode ? ` · ${recMode}` : ""}
                    {recCached && recAsOf ? " · 당일 저장분" : ""}
                  </div>
                </div>
              </div>
              <div className="rec-market-tabs rec-market-tabs-inline" role="group" aria-label="추천 시장">
                {REC_MARKET_OPTIONS.map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    className={`rec-market-tab ${recMarket === opt.id ? "active" : ""}`}
                    onClick={() => selectRecMarket(opt.id)}
                    disabled={loading}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <p className="muted">
                {recMeta.universe > 0
                  ? `유니버스 ${recMeta.universe} → 스캔 ${recMeta.scanned} → shortlist ${recMeta.shortlist}`
                  : "왼쪽에서 범위를 고른 뒤 일일 추천을 실행하세요."}
              </p>
              {(recs.length > 0 || scanItems.length > 0) && (
                <div className="rec-toolbar">
                  <label className="rec-toggle">
                    <input
                      type="checkbox"
                      checked={showScanAll}
                      onChange={(e) => setShowScanAll(e.target.checked)}
                    />
                    스캔 결과 전체
                  </label>
                  {recMeta.source ? (
                    <span className="muted">universe: {recMeta.source}</span>
                  ) : null}
                </div>
              )}
              {recCommentary && !showScanAll && (
                <article className="card-item">
                  <h4>시장 코멘트</h4>
                  <p className="answer">{recCommentary}</p>
                </article>
              )}
              {!showScanAll && recs.length === 0 && (
                <p className="muted">일일 추천을 실행하면 AI Picks가 여기에 표시됩니다.</p>
              )}
              {!showScanAll &&
                recs.map((r) => (
                  <article className="card-item" key={r.symbol}>
                    <h4>
                      #{r.rank} {r.name} ({r.symbol}) · {r.score}점
                      <span className="chip" style={{ marginLeft: 8 }}>
                        {r.market === "KRX" ? "국내" : "해외"}
                      </span>
                      {r.stance ? (
                        <span className="chip accent" style={{ marginLeft: 8 }}>
                          {STANCE_LABEL[r.stance] || r.stance}
                        </span>
                      ) : null}
                    </h4>
                    <p>
                      {r.price.toLocaleString()} · {r.change_pct >= 0 ? "+" : ""}
                      {r.change_pct}% · RSI {r.rsi} · {r.macd_signal} · {r.trend}
                    </p>
                    <div className="rec-targets">
                      <span>
                        권장 매수{" "}
                        <strong>
                          {r.buy_price
                            ? r.market === "KRX"
                              ? Math.round(r.buy_price).toLocaleString("ko-KR")
                              : r.buy_price.toLocaleString()
                            : "-"}
                        </strong>
                      </span>
                      <span>
                        권장 매도{" "}
                        <strong>
                          {r.sell_price
                            ? r.market === "KRX"
                              ? Math.round(r.sell_price).toLocaleString("ko-KR")
                              : r.sell_price.toLocaleString()
                            : "-"}
                        </strong>
                      </span>
                    </div>
                    <p style={{ marginTop: 6 }}>{r.reasons.join(" · ")}</p>
                    <button
                      className="btn secondary"
                      style={{ marginTop: 8 }}
                      onClick={() => openSymbolOnDesk(r.symbol, r.market)}
                    >
                      차트/분석 보기
                    </button>
                  </article>
                ))}
              {showScanAll && (
                <div className="scan-board">
                  {scanItems.length === 0 ? (
                    <p className="muted">스캔 결과가 없습니다. 일일 추천을 다시 실행해 주세요.</p>
                  ) : (
                    <div className="scan-table-wrap">
                      <table className="scan-table">
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>시장</th>
                            <th>종목</th>
                            <th>점수</th>
                            <th>등락</th>
                            <th>RSI</th>
                            <th>MACD</th>
                            <th>추세</th>
                            <th></th>
                          </tr>
                        </thead>
                        <tbody>
                          {scanItems.map((row) => (
                            <tr key={`${row.market}-${row.symbol}`}>
                              <td>{row.rank}</td>
                              <td>{row.market === "KRX" ? "국내" : "해외"}</td>
                              <td>
                                <strong>{row.name}</strong>
                                <div className="muted">{row.symbol}</div>
                              </td>
                              <td>{row.score}</td>
                              <td className={row.change_pct >= 0 ? "up" : "down"}>
                                {row.change_pct >= 0 ? "+" : ""}
                                {row.change_pct}%
                              </td>
                              <td>{row.rsi}</td>
                              <td>{row.macd_signal}</td>
                              <td>{row.trend}</td>
                              <td>
                                <button
                                  className="btn secondary"
                                  onClick={() => openSymbolOnDesk(row.symbol, row.market)}
                                >
                                  보기
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {workspace === "account" && (
            <div className="list workspace-page">
              <div className="main-head">
                <div className="price-box">
                  <h2>내 계좌</h2>
                  <div className="meta muted">통합 잔고 · 국내/해외 보유</div>
                </div>
              </div>
              {!account?.connected && (
                <p className="muted">
                  거래 연동이 되지 않았습니다. `.env`의 KIS 설정 또는 `secret.json`을 확인한 뒤
                  API를 재시작해 주세요.
                </p>
              )}
              {account?.error && <div className="error-box">{account.error}</div>}
              {account?.connected && (
                <>
                  <article className="card-item">
                    <h4>
                      통합 계좌 {account.virtual ? "(모의투자)" : "(실전)"}
                    </h4>
                    <p>계좌번호 {account.account || "-"}</p>
                    <div className="cash-strip">
                      <div className="cash-item">
                        <div className="label">보유 현금 (원화)</div>
                        <div className="value">{formatMoney(account.domestic.deposit_krw)}</div>
                      </div>
                      <div className="cash-item">
                        <div className="label">보유 현금 (달러)</div>
                        <div className="value">
                          {formatMoney(account.overseas.deposit_usd, "USD")}
                        </div>
                        {account.overseas.deposit_krw ? (
                          <div className="muted" style={{ fontSize: "0.78rem", marginTop: 2 }}>
                            약 {formatMoney(account.overseas.deposit_krw)}
                          </div>
                        ) : null}
                      </div>
                    </div>
                    <div className="metrics" style={{ marginTop: 10, marginBottom: 0 }}>
                      <div className="metric">
                        <div className="label">총평가(원화)</div>
                        <div className="value">{formatMoney(account.total_eval_krw)}</div>
                      </div>
                      <div className="metric">
                        <div className="label">보유주식 평가</div>
                        <div className="value">{formatMoney(account.current_amount)}</div>
                      </div>
                      <div className="metric">
                        <div className="label">매입금액</div>
                        <div className="value">{formatMoney(account.purchase_amount)}</div>
                      </div>
                      <div className="metric">
                        <div className="label">평가손익</div>
                        <div className={`value ${account.profit_loss >= 0 ? "up" : "down"}`}>
                          {account.profit_loss >= 0 ? "+" : ""}
                          {formatMoney(account.profit_loss)} ({account.profit_loss_rate.toFixed(2)}%)
                        </div>
                      </div>
                    </div>
                    <button
                      className="btn secondary"
                      style={{ marginTop: 10 }}
                      onClick={loadAccount}
                      disabled={accountLoading}
                    >
                      계좌 새로고침
                    </button>
                  </article>

                  <article className="card-item">
                    <h4>국내 계좌</h4>
                    <p>
                      예수금 {formatMoney(account.domestic.deposit_krw)} · 주식평가{" "}
                      {formatMoney(account.domestic.stocks_value)}
                    </p>
                    {(account.domestic.holdings || []).length === 0 && (
                      <p className="muted" style={{ marginTop: 8 }}>보유 종목이 없습니다.</p>
                    )}
                    {(account.domestic.holdings || []).map((h) => (
                      <div key={`kr-${h.symbol}`} style={{ marginTop: 10 }}>
                        <strong>
                          {h.name || h.symbol} ({h.symbol})
                        </strong>
                        <p>
                          {h.qty.toLocaleString()}주 · {formatMoney(h.price)} · 평가{" "}
                          {formatMoney(h.amount)}
                        </p>
                        <p className={h.profit >= 0 ? "up" : "down"}>
                          손익 {h.profit >= 0 ? "+" : ""}
                          {formatMoney(h.profit)} ({h.profit_rate.toFixed(2)}%)
                        </p>
                        <button
                          className="btn secondary"
                          style={{ marginTop: 6 }}
                          onClick={() => openHolding(h)}
                        >
                          차트 보기
                        </button>
                      </div>
                    ))}
                  </article>

                  <article className="card-item">
                    <h4>해외 계좌</h4>
                    <p>
                      예수금 {formatMoney(account.overseas.deposit_usd, "USD")}
                      {account.overseas.deposit_krw
                        ? ` (약 ${formatMoney(account.overseas.deposit_krw)})`
                        : ""}
                      {" · "}주식평가 {formatMoney(account.overseas.stocks_value, "USD")}
                      {account.overseas.stocks_value_krw
                        ? ` / ${formatMoney(account.overseas.stocks_value_krw)}`
                        : ""}
                    </p>
                    {account.overseas.exchange_rate ? (
                      <p className="muted" style={{ marginTop: 4 }}>
                        적용환율 {account.overseas.exchange_rate.toLocaleString()}원/USD
                      </p>
                    ) : null}
                    {(account.overseas.holdings || []).length === 0 && (
                      <p className="muted" style={{ marginTop: 8 }}>보유 종목이 없습니다.</p>
                    )}
                    {(account.overseas.holdings || []).map((h) => (
                      <div key={`us-${h.symbol}`} style={{ marginTop: 10 }}>
                        <strong>
                          {h.name || h.symbol} ({h.symbol}/{h.market || "US"})
                        </strong>
                        <p>
                          {h.qty.toLocaleString()}주 · {formatMoney(h.price, "USD")} · 평가{" "}
                          {formatMoney(h.amount, "USD")}
                        </p>
                        <p className={h.profit >= 0 ? "up" : "down"}>
                          손익 {h.profit >= 0 ? "+" : ""}
                          {formatMoney(h.profit, "USD")} ({h.profit_rate.toFixed(2)}%)
                        </p>
                        <button
                          className="btn secondary"
                          style={{ marginTop: 6 }}
                          onClick={() => openHolding(h)}
                        >
                          차트 보기
                        </button>
                      </div>
                    ))}
                  </article>
                </>
              )}
            </div>
          )}
        </main>

        {workspace === "desk" && (
        <aside className="panel side-stock-news">
          <div className="section-title">종목 뉴스</div>
          <p className="muted side-news-hint">
            선택 종목({stockName || symbol}) · 클릭 시 AI 브리핑
          </p>
          <div className="row" style={{ gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
            <button
              type="button"
              className={`market-cat ${newsSort === "importance" ? "active" : ""}`}
              onClick={() => setNewsSort("importance")}
            >
              중요도순
            </button>
            <button
              type="button"
              className={`market-cat ${newsSort === "date" ? "active" : ""}`}
              onClick={() => setNewsSort("date")}
            >
              최신순
            </button>
          </div>
          <div className="list">
            {[...news]
              .sort((a, b) => {
                const ia = a.importance || 0;
                const ib = b.importance || 0;
                const da = a.published_at || "";
                const db = b.published_at || "";
                if (newsSort === "date") return db.localeCompare(da) || ib - ia;
                return ib - ia || db.localeCompare(da);
              })
              .slice(0, 10)
              .map((n, i) => (
              <article
                className="card-item side-news-item"
                key={`side-${n.url || i}`}
                onClick={() => openStockNewsBriefing(n)}
                title={n.importance_reason || "AI 브리핑 보기"}
              >
                <div className="row" style={{ gap: 6, marginBottom: 4, flexWrap: "wrap" }}>
                  {typeof n.importance === "number" && n.importance > 0 && (
                    <span className={`chip imp imp-${n.importance}`}>중요도 {n.importance}</span>
                  )}
                  <span className="chip" title={n.published_at || ""}>
                    {formatNewsTime(n.published_at)}
                  </span>
                  {n.has_ai_summary ? (
                    <span className="chip">브리핑 준비됨</span>
                  ) : (
                    <span className="chip">정리 중</span>
                  )}
                </div>
                <h4 className="news-link">{n.title}</h4>
                <p>{n.source || "언론사"}</p>
                {n.url && (
                  <a
                    className="source-link"
                    href={n.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                  >
                    원문 보기 →
                  </a>
                )}
              </article>
            ))}
            {!news.length && <p className="muted">종목 뉴스가 여기에 표시됩니다.</p>}
          </div>
        </aside>
        )}
      </div>

      <NewsBriefingModal
        open={stockNewsOpen}
        onClose={() => setStockNewsOpen(false)}
        loading={stockNewsLoading}
        summary={stockNewsSummary}
        active={stockNewsActive}
        kicker="종목 뉴스 브리핑"
      />

      <MarketPulsePanel
        open={marketPulseOpen}
        onClose={() => setMarketPulseOpen(false)}
        provider={llmProvider}
      />
    </div>
  );
}
