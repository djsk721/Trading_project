const BASE = "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

export type Health = {
  status: string;
  kis_connected: boolean;
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
  account: () => request<AccountOverview>("/api/trading/account"),
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
};
