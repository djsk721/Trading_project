import { useEffect, useMemo, useRef, useState } from "react";
import { api, OrderBookPayload } from "../api";

type Props = {
  symbol: string;
  market: string;
  /** 주문 탭이 보일 때만 폴링 */
  active: boolean;
  pollMs?: number;
  selectedPrice?: number | null;
  /** 매도호가 클릭 → buy, 매수호가 클릭 → sell */
  onPriceClick?: (price: number, side: "buy" | "sell") => void;
};

function formatPrice(value: number, market: string) {
  if (!Number.isFinite(value) || value <= 0) return "-";
  if ((market || "").toUpperCase() === "KRX") {
    return Math.round(value).toLocaleString("ko-KR");
  }
  const abs = Math.abs(value);
  const digits = abs >= 1 ? 2 : abs >= 0.1 ? 3 : abs >= 0.01 ? 4 : 6;
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

function formatVolume(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "-";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 10_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toLocaleString();
}

function pricesClose(a: number, b: number) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
  const scale = Math.max(Math.abs(a), Math.abs(b), 1);
  return Math.abs(a - b) / scale < 1e-9 || Math.abs(a - b) < 1e-6;
}

export default function OrderBookPanel({
  symbol,
  market,
  active,
  pollMs = 2000,
  selectedPrice = null,
  onPriceClick,
}: Props) {
  const [book, setBook] = useState<OrderBookPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inflight = useRef(false);
  const hasBook = useRef(false);

  useEffect(() => {
    hasBook.current = false;
    setBook(null);
    setError("");
  }, [symbol, market]);

  useEffect(() => {
    if (!active || !symbol) return;
    let cancelled = false;

    async function load() {
      if (inflight.current) return;
      inflight.current = true;
      if (!hasBook.current) setLoading(true);
      try {
        const r = await api.orderbook(symbol, market);
        if (cancelled) return;
        setBook(r);
        hasBook.current = Boolean(r.asks?.length || r.bids?.length || r.ok);
        setError(
          r.ok || (r.asks?.length || 0) + (r.bids?.length || 0) > 0
            ? r.rate_limited
              ? "KIS 호출 한도 — 캐시 표시"
              : ""
            : r.rate_limited
              ? "KIS 호출 한도 — 잠시 후 재시도"
              : r.message || "호가 없음"
        );
      } catch (e: unknown) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "호가 조회 실패");
      } finally {
        inflight.current = false;
        if (!cancelled) setLoading(false);
      }
    }

    load();
    const id = window.setInterval(load, Math.max(1000, pollMs));
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [active, symbol, market, pollMs]);

  const maxVol = useMemo(() => {
    const vols = [...(book?.asks || []), ...(book?.bids || [])].map((x) => x.volume || 0);
    return Math.max(1, ...vols);
  }, [book]);

  const askRows = useMemo(() => {
    return [...(book?.asks || [])].filter((a) => a.price > 0).reverse();
  }, [book]);

  const bidRows = useMemo(() => {
    return (book?.bids || []).filter((b) => b.price > 0);
  }, [book]);

  const spread = useMemo(() => {
    const bestAsk = book?.asks?.[0]?.price || 0;
    const bestBid = book?.bids?.[0]?.price || 0;
    if (bestAsk <= 0 || bestBid <= 0) return null;
    return bestAsk - bestBid;
  }, [book]);

  return (
    <aside className="orderbook-panel" aria-label="호가창">
      <div className="orderbook-head">
        <div className="section-title" style={{ margin: 0 }}>
          호가창
        </div>
        <span className="muted orderbook-meta">
          {loading && !book ? "조회 중…" : book?.cached ? "캐시" : "LIVE"}
          {book?.rate_limited ? " · 한도" : ""}
        </span>
      </div>

      <p className="muted orderbook-hint-top">
        매도호가 클릭 → 지정가 매수 · 매수호가 클릭 → 지정가 매도
      </p>

      <div className="orderbook-cols">
        <span>잔량</span>
        <span>호가</span>
      </div>

      <div className="orderbook-body">
        {askRows.length === 0 && bidRows.length === 0 && (
          <p className="muted orderbook-empty">
            {error || (loading ? "호가 불러오는 중…" : "호가 데이터가 없습니다.")}
          </p>
        )}

        {askRows.map((row, i) => {
          const selected = selectedPrice != null && pricesClose(selectedPrice, row.price);
          return (
            <button
              type="button"
              className={`ob-row ask clickable${selected ? " selected" : ""}`}
              key={`ask-${row.price}-${i}`}
              title={`지정가 매수 ${formatPrice(row.price, market)}`}
              onClick={() => onPriceClick?.(row.price, "buy")}
            >
              <span
                className="ob-bar ask-bar"
                style={{ width: `${Math.min(100, (row.volume / maxVol) * 100)}%` }}
              />
              <span className="ob-vol">{formatVolume(row.volume)}</span>
              <span className="ob-price down">{formatPrice(row.price, market)}</span>
            </button>
          );
        })}

        {(askRows.length > 0 || bidRows.length > 0) && (
          <div className="ob-spread">
            <span className="muted">스프레드</span>
            <strong>{spread != null ? formatPrice(spread, market) : "-"}</strong>
          </div>
        )}

        {bidRows.map((row, i) => {
          const selected = selectedPrice != null && pricesClose(selectedPrice, row.price);
          return (
            <button
              type="button"
              className={`ob-row bid clickable${selected ? " selected" : ""}`}
              key={`bid-${row.price}-${i}`}
              title={`지정가 매도 ${formatPrice(row.price, market)}`}
              onClick={() => onPriceClick?.(row.price, "sell")}
            >
              <span
                className="ob-bar bid-bar"
                style={{ width: `${Math.min(100, (row.volume / maxVol) * 100)}%` }}
              />
              <span className="ob-vol">{formatVolume(row.volume)}</span>
              <span className="ob-price up">{formatPrice(row.price, market)}</span>
            </button>
          );
        })}
      </div>

      <div className="orderbook-foot">
        <span>매도합 {formatVolume(book?.ask_volume || 0)}</span>
        <span>매수합 {formatVolume(book?.bid_volume || 0)}</span>
      </div>
      {error && (askRows.length > 0 || bidRows.length > 0) && (
        <p className="muted orderbook-hint">{error}</p>
      )}
    </aside>
  );
}
