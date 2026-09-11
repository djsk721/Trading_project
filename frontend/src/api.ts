const BASE = "";

async function request<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const timeoutMs = init?.timeoutMs;
  const { timeoutMs: _omit, ...rest } = init || {};
  const ctrl = timeoutMs ? new AbortController() : null;
  const timer = timeoutMs
    ? window.setTimeout(() => ctrl?.abort(), timeoutMs)
    : 0;
  try {
    const res = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json", ...(rest.headers || {}) },
      ...rest,
      signal: ctrl?.signal || rest.signal,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    return res.json();
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error("요청 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.");
    }
    throw err;
  } finally {
    if (timer) window.clearTimeout(timer);
  }
}

export type Health = {
  status: string;
  kis_connected: boolean;
  broker_connected?: boolean;
  active_broker?: "kis" | "toss" | string;
  broker_hint?: string;
  ollama_connected: boolean;
  nvidia_connected?: boolean;
  ai_connected?: boolean;
  llm_provider?: string;
  llm_model: string;
  embed_model: string;
};

export type Quote = {
  symbol: string;
  name: string;
  market: string;
  price: number;
  change: number;
  rate: number;
  volume: number;
};

export type OrderBookLevel = {
  price: number;
  volume: number;
};

export type OrderBookPayload = {
  ok: boolean;
  symbol: string;
  market: string;
  name?: string;
  asks: OrderBookLevel[];
  bids: OrderBookLevel[];
  ask_volume: number;
  bid_volume: number;
  decimal_places?: number;
  cached?: boolean;
  rate_limited?: boolean;
  message?: string;
  as_of?: number;
};

export type ChartPayload = {
  symbol: string;
  market: string;
  timeframe: string;
  bars: { time: string; open: number; high: number; low: number; close: number; volume: number }[];
  indicators: Record<string, { time: string; value: number }[]>;
  summary: Record<string, number | string>;
};

export type RuleItem = {
  id: string;
  title: string;
  detail: string;
  direction: string;
  weight: number;
};

export type RuleAnalysis = {
  symbol: string;
  market: string;
  stock_name: string;
  as_of: string;
  price: number;
  score: number;
  stance: string;
  bias: string;
  summary_text: string;
  signals: string[];
  metrics: Record<string, number | string>;
  rules: RuleItem[];
  horizon?: {
    medium_trend?: string;
    medium_trend_label?: string;
    short_momentum?: string;
    short_momentum_label?: string;
    narrative?: string;
    macd_evidence?: string;
    rsi_zone?: string;
  };
};

export type NewsItem = {
  title: string;
  title_original?: string;
  summary: string;
  url: string;
  source: string;
  published_at?: string | null;
  importance?: number | null;
  importance_reason?: string;
  sentiment?: string;
  sentiment_reason?: string;
  has_ai_summary?: boolean;
};

export type MarketNewsItem = {
  id: string;
  title: string;
  title_original?: string;
  summary: string;
  url: string;
  source: string;
  published_at?: string | null;
  category: string;
  category_label: string;
  has_ai_summary: boolean;
  importance?: number | null;
  importance_reason?: string;
  sentiment?: string;
  sentiment_reason?: string;
};

export type NewsSummary = {
  id: string;
  url: string;
  title: string;
  title_original?: string;
  title_ko?: string;
  source?: string;
  summary_ko: string;
  importance?: number;
  importance_reason?: string;
  sentiment?: string;
  sentiment_reason?: string;
  provider: string;
  updated_at?: string;
  cached: boolean;
};

export type TabDigest = {
  category: string;
  category_label: string;
  day: string;
  text: string;
  provider: string;
  source_count: number;
  updated_at?: string;
  cached: boolean;
  ready: boolean;
};

export type MacroSnapshot = {
  as_of: string;
  source: string;
  ttl_sec: number;
  items: {
    id: string;
    ticker: string;
    label: string;
    unit: string;
    price: number | null;
    prev: number | null;
    change: number | null;
    change_pct: number | null;
    price_text: string;
    change_text: string;
    ok: boolean;
  }[];
  ok_count: number;
  errors?: string[];
  cached?: boolean;
};

export type RecommendItem = {
  rank: number;
  symbol: string;
  name: string;
  market: string;
  score: number;
  price: number;
  change_pct: number;
  reasons: string[];
  rsi: number;
  macd_signal: string;
  trend: string;
  stance?: string;
  buy_price?: number;
  sell_price?: number;
  sector?: string;
  status_label?: string;
  highlights?: string[];
  metric_note?: string;
  detail_summary?: string;
  ai_summary?: string;
};

export type HoldingExitItem = {
  rank: number;
  symbol: string;
  name: string;
  market: string;
  decision: "sell" | "trim" | "hold" | string;
  decision_label: string;
  risk_score: number;
  qty: number;
  price: number;
  avg_cost: number;
  cost_amount: number;
  cost_amount_krw?: number;
  amount: number;
  amount_krw?: number;
  profit: number;
  profit_krw?: number;
  profit_rate: number;
  currency?: string;
  exchange_rate?: number;
  portfolio_weight: number;
  rsi: number;
  macd_signal: string;
  trend: string;
  sma20?: number;
  sma60?: number;
  sma120?: number;
  bb_position?: number;
  reasons: string[];
  highlights: string[];
  ai_summary: string;
  ai_rationale?: string;
  ai_risk?: string;
  ai_action?: string;
  ai_watchpoints?: string[];
};

export type HoldingExitResponse = {
  as_of: string;
  items: HoldingExitItem[];
  summary: Record<string, number>;
  used_llm: boolean;
  provider: string;
  cached: boolean;
  updated_at?: string | null;
  disclaimer: string;
};

export type ScanItem = {
  rank: number;
  symbol: string;
  name: string;
  market: string;
  score: number;
  price: number;
  change_pct: number;
  rsi: number;
  macd_signal: string;
  trend: string;
  reasons?: string[];
  sector?: string;
  status_label?: string;
  highlights?: string[];
  metric_note?: string;
  detail_summary?: string;
  ai_summary?: string;
};

export type HoldingItem = {
  name: string;
  symbol: string;
  market: string;
  qty: number;
  price: number;
  amount: number;
  profit: number;
  profit_rate: number;
  currency?: string;
  scope?: string;
};

export type AccountOverview = {
  connected: boolean;
  account: string;
  virtual: boolean;
  total_eval_krw: number;
  purchase_amount: number;
  current_amount: number;
  profit_loss: number;
  profit_loss_rate: number;
  deposits: {
    currency: string;
    amount: number;
    exchange_rate: number;
    amount_krw: number;
    scope: string;
  }[];
  domestic: {
    deposit_krw: number;
    stocks_value: number;
    holdings: HoldingItem[];
  };
  overseas: {
    deposit_usd: number;
    deposit_krw: number;
    stocks_value: number;
    stocks_value_krw?: number;
    exchange_rate?: number;
    holdings: HoldingItem[];
  };
  holdings: HoldingItem[];
  error?: string | null;
};

export type BrokerCapabilities = {
  active_broker: string;
  connected: boolean;
  trading_enabled: boolean;
  test_mode: boolean;
  rate_limited: boolean;
  supports: Record<string, boolean>;
  disabled_reason?: string;
  checks?: { name: string; ok: boolean; detail: string }[];
};

export type DividendEvent = {
  symbol: string;
  name: string;
  market: string;
  ex_date: string;
  pay_date?: string | null;
  amount: number;
  currency: string;
  yield_pct: number;
  d_day: number;
  source: string;
  scope: string;
};

export type DividendCalendar = {
  as_of: string;
  from: string;
  to: string;
  scope: string;
  target_count: number;
  items: DividendEvent[];
  recent_items?: DividendEvent[];
  cached: boolean;
  source_note: string;
};

export type Sec13FManager = {
  cik: string;
  manager_name: string;
  filing_date?: string;
  report_period?: string;
  portfolio_value: number;
  holdings_count: number;
};

export type Sec13FHolding = {
  cik: string;
  manager_name: string;
  report_period: string;
  issuer: string;
  ticker: string;
  cusip: string;
  security_class?: string;
  value: number;
  shares: number;
  previous_shares?: number;
  current_shares?: number;
  previous_value?: number;
  current_value?: number;
  put_call?: string;
  change_type?: "NEW" | "INCREASED" | "DECREASED" | "SOLD" | "UNCHANGED" | string;
  share_change?: number;
  value_change?: number;
  portfolio_weight?: number;
};

export type Sec13FDashboard = {
  metadata: Record<string, any>;
  manager_count: number;
  issuer_count: number;
  top_managers: Sec13FManager[];
  recent_new_holdings: Sec13FHolding[];
  shared_buys: { ticker: string; issuer: string; manager_count: number; total_value: number }[];
};

export type Sec13FManagerAnalysis = {
  manager_count: number;
  holding_count: number;
  total_value: number;
  top_holdings: (Sec13FHolding & { manager_count?: number })[];
  change_summary: Record<string, number>;
};

export const api = {
  health: () => request<Health>("/api/health"),
  popular: (market: string) =>
    request<{ items: { symbol: string; name: string }[] }>(`/api/market/popular?market=${market}`),
  quote: (symbol: string, market = "") => {
    const params = market ? `?market=${encodeURIComponent(market)}` : "";
    return request<Quote>(`/api/market/quote/${encodeURIComponent(symbol)}${params}`);
  },
  orderbook: (symbol: string, market = "") => {
    const params = market ? `?market=${encodeURIComponent(market)}` : "";
    return request<OrderBookPayload>(
      `/api/market/orderbook/${encodeURIComponent(symbol)}${params}`
    );
  },
  chart: (params: URLSearchParams) =>
    request<ChartPayload>(`/api/market/chart?${params.toString()}`),
  account: (force = false) =>
    request<AccountOverview>(`/api/trading/account${force ? "?force=true" : ""}`, {
      timeoutMs: 20000,
    }),
  order: (body: object) =>
    request("/api/trading/orders", { method: "POST", body: JSON.stringify(body) }),
  news: (
    symbol: string,
    market: string,
    opts?: { prepare?: boolean; provider?: string; sort?: "importance" | "date" }
  ) => {
    const params = new URLSearchParams({
      symbol,
      market,
      prepare: String(opts?.prepare ?? true),
      sort: opts?.sort || "importance",
    });
    if (opts?.provider) params.set("provider", opts.provider);
    return request<{
      items: NewsItem[];
      stock_name: string;
      count: number;
      sort?: string;
      preparing?: boolean;
    }>(`/api/news?${params.toString()}`);
  },
  macros: (force = false) =>
    request<MacroSnapshot>(`/api/market/macros?force=${force}`),
  marketNews: (opts?: {
    prepare?: boolean;
    force?: boolean;
    provider?: string;
    sort?: "importance" | "date";
  }) => {
    const params = new URLSearchParams({
      prepare: String(opts?.prepare ?? true),
      force: String(opts?.force ?? false),
      sort: opts?.sort || "importance",
    });
    if (opts?.provider) params.set("provider", opts.provider);
    return request<{
      fetched_at: string;
      items: MarketNewsItem[];
      count: number;
      categories: { id: string; label: string }[];
      preparing: boolean;
      digests?: Record<string, TabDigest>;
      digest_preparing?: boolean;
      digest_day?: string;
      macros?: MacroSnapshot;
      sort?: string;
    }>(`/api/news/market?${params.toString()}`);
  },
  summarizeNews: (body: {
    url: string;
    title?: string;
    snippet?: string;
    source?: string;
    provider?: string;
    force?: boolean;
  }) =>
    request<NewsSummary>("/api/news/summarize", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  recommend: (market: string, opts?: { provider?: string; top_n?: number; force?: boolean; force_universe?: boolean }) => {
    const params = new URLSearchParams({ market });
    if (opts?.provider) params.set("provider", opts.provider);
    if (opts?.top_n != null) params.set("top_n", String(opts.top_n));
    if (opts?.force) params.set("force", "true");
    if (opts?.force_universe) params.set("force_universe", "true");
    return request<{
      as_of: string;
      market?: string;
      items: RecommendItem[];
      scan_items?: ScanItem[];
      universe_size?: number;
      universe_source?: string;
      shortlist_size?: number;
      scanned_count?: number;
      disclaimer: string;
      market_commentary?: string;
      used_llm?: boolean;
      provider?: string;
      model?: string;
      cached?: boolean;
      updated_at?: string | null;
    }>(`/api/recommend/daily?${params.toString()}`);
  },
  holdingsExit: (opts?: { provider?: string; force?: boolean; days?: number }) => {
    const params = new URLSearchParams();
    if (opts?.provider) params.set("provider", opts.provider);
    if (opts?.force) params.set("force", "true");
    if (opts?.days) params.set("days", String(opts.days));
    const qs = params.toString();
    return request<HoldingExitResponse>(`/api/recommend/holdings-exit${qs ? `?${qs}` : ""}`);
  },
  recommendProgress: () =>
    request<{ stage: string; stage_label: string; progress: number; message: string; updated_at?: string | null }>(
      "/api/recommend/daily/progress"
    ),
  startRecommendJob: (market: string, opts?: { provider?: string; top_n?: number; force?: boolean; force_universe?: boolean }) => {
    const params = new URLSearchParams({ market });
    if (opts?.provider) params.set("provider", opts.provider);
    if (opts?.top_n != null) params.set("top_n", String(opts.top_n));
    if (opts?.force != null) params.set("force", String(opts.force));
    if (opts?.force_universe) params.set("force_universe", "true");
    return request<Record<string, unknown>>(`/api/recommend/daily/jobs?${params.toString()}`, { method: "POST" });
  },
  recommendJob: (jobId: string) =>
    request<Record<string, unknown>>(`/api/recommend/daily/jobs/${encodeURIComponent(jobId)}`),
  dividends: (opts?: { from?: string; to?: string; scope?: string; symbols?: string; force?: boolean }) => {
    const params = new URLSearchParams();
    if (opts?.from) params.set("from", opts.from);
    if (opts?.to) params.set("to", opts.to);
    if (opts?.scope) params.set("scope", opts.scope);
    if (opts?.symbols) params.set("symbols", opts.symbols);
    if (opts?.force) params.set("force", "true");
    const qs = params.toString();
    return request<DividendCalendar>(`/api/calendar/dividends${qs ? `?${qs}` : ""}`);
  },
  brokerCapabilities: () => request<BrokerCapabilities>("/api/trading/capabilities"),
  brokerDiagnostics: () => request<BrokerCapabilities>("/api/trading/diagnostics"),
  brokerStatus: () => request<Record<string, unknown>>("/api/settings/broker"),
  saveBrokerKeys: (body: object) =>
    request<Record<string, unknown>>("/api/settings/broker", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  ruleAnalysis: (symbol: string, market: string, stockName = "") => {
    const params = new URLSearchParams({
      symbol,
      market,
      stock_name: stockName,
      days: "120",
    });
    return request<RuleAnalysis>(`/api/analysis/rules?${params.toString()}`);
  },
  analyze: (body: object) =>
    request<{
      answer: string;
      sources: { content: string; metadata: Record<string, unknown> }[];
      model: string;
      provider?: string;
      current_price: number;
      current_date: string;
      stock_name: string;
      rule_analysis?: RuleAnalysis;
    }>("/api/analysis/ask", { method: "POST", body: JSON.stringify(body) }),
  sec13fUpdate: (force = false) =>
    request<Record<string, any>>(`/api/13f/update?force=${force}`, { method: "POST" }),
  sec13fDashboard: () => request<Sec13FDashboard>("/api/13f/dashboard"),
  sec13fManagers: (q = "", limit = 50) => {
    const params = new URLSearchParams({ q, limit: String(limit) });
    return request<{ items: Sec13FManager[] }>(`/api/13f/managers?${params.toString()}`);
  },
  sec13fManagersAnalysis: (q = "", limit = 30) => {
    const params = new URLSearchParams({ q, limit: String(limit) });
    return request<Sec13FManagerAnalysis>(`/api/13f/managers/analysis?${params.toString()}`);
  },
  sec13fManager: (cik: string, opts?: { holding_q?: string; limit?: number }) => {
    const params = new URLSearchParams();
    if (opts?.holding_q) params.set("holding_q", opts.holding_q);
    if (opts?.limit) params.set("limit", String(opts.limit));
    const qs = params.toString();
    return request<{ manager: Sec13FManager; holdings: Sec13FHolding[] }>(
      `/api/13f/managers/${encodeURIComponent(cik)}${qs ? `?${qs}` : ""}`
    );
  },
  sec13fStock: (ticker: string) =>
    request<{
      ticker: string;
      issuer: string;
      holder_count: number;
      holders: Sec13FHolding[];
      new_holders: Sec13FHolding[];
      sold_holders: Sec13FHolding[];
    }>(`/api/13f/stocks/${encodeURIComponent(ticker)}`),
};
