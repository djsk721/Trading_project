import { FormEvent, useEffect, useMemo, useState } from "react";
import { api, DividendCalendar, DividendEvent } from "../api";

type Props = {
  onOpenSymbol: (symbol: string, market?: string) => void;
  recommendCount?: number;
  recommendedSymbols?: string[];
};

function addDays(days: number) {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

function money(value: number, currency: string) {
  if (!Number.isFinite(value) || value <= 0) return "-";
  if (currency === "KRW") return `${Math.round(value).toLocaleString("ko-KR")}원`;
  return `$${value.toLocaleString(undefined, { maximumFractionDigits: 4 })}`;
}

function ddayLabel(d: number) {
  if (d === 0) return "D-day";
  if (d > 0) return `D-${d}`;
  return `D+${Math.abs(d)}`;
}

export default function DividendCalendarWorkspace({ onOpenSymbol, recommendCount = 0, recommendedSymbols = [] }: Props) {
  const [scope, setScope] = useState("holdings");
  const [symbols, setSymbols] = useState("");
  const [from, setFrom] = useState(addDays(0));
  const [to, setTo] = useState(addDays(60));
  const [data, setData] = useState<DividendCalendar | null>(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");

  async function load(force = false) {
    setLoading(true);
    setMsg("");
    try {
      const scopedSymbols = symbols.trim() || (scope === "recommend" ? recommendedSymbols.join(",") : "");
      const r = await api.dividends({ from, to, scope, symbols: scopedSymbols, force });
      setData(r);
    } catch (err: unknown) {
      setMsg(err instanceof Error ? err.message : "배당 캘린더 조회에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const byMonth = useMemo(() => {
    const map = new Map<string, DividendEvent[]>();
    for (const item of data?.items || []) {
      const key = item.ex_date.slice(0, 7);
      map.set(key, [...(map.get(key) || []), item]);
    }
    return Array.from(map.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [data]);

  const recentItems = data?.recent_items || [];

  function submit(e: FormEvent) {
    e.preventDefault();
    void load(true);
  }

  return (
    <div className="list workspace-page dividend-page">
      <div className="main-head">
        <div className="price-box">
          <h2>배당·배당락 캘린더</h2>
          <div className="meta muted">
            {data ? `${data.from} ~ ${data.to} · ${data.target_count}` : ""}
          </div>
        </div>
        <button className="btn" onClick={() => load(true)} disabled={loading}>
          {loading ? "조회 중..." : "새로고침"}
        </button>
      </div>

      <form className="dividend-filter" onSubmit={submit}>
        <label>
          범위
          <select value={scope} onChange={(e) => setScope(e.target.value)}>
            <option value="holdings">보유종목</option>
            <option value="recommend">추천종목</option>
            <option value="all">보유+추천</option>
            <option value="manual">직접입력</option>
          </select>
        </label>
        <label>
          시작일
          <input type="date" value={from} onChange={(e) => setFrom(e.target.value)} />
        </label>
        <label>
          종료일
          <input type="date" value={to} onChange={(e) => setTo(e.target.value)} />
        </label>
        <label className="dividend-symbols">
          종목 직접입력
          <input value={symbols} onChange={(e) => setSymbols(e.target.value)} placeholder="AAPL, MSFT, 005930" />
        </label>
        <button className="btn secondary" type="submit" disabled={loading}>조회</button>
      </form>

      {msg ? <div className="error-box">{msg}</div> : null}
      <section className="dividend-overview">
        <div>
          <span>향후 배당락</span>
          <strong>{data?.items.length || 0}</strong>
        </div>
        <div>
          <span>최근 배당락 확인</span>
          <strong>{recentItems.length}</strong>
        </div>
        <div>
          <span>조회 대상</span>
          <strong>{data?.target_count || 0}</strong>
        </div>
      </section>

      {data && data.items.length === 0 ? (
        <article className="dividend-empty compact-empty">
          <h4>향후 0</h4>
        </article>
      ) : null}

      <div className="dividend-month-grid">
        {byMonth.map(([month, items]) => (
          <section key={month} className="dividend-month-card">
            <h3>{month}</h3>
            <div className="dividend-event-list">
              {items.map((item) => (
                <button
                  type="button"
                  className="dividend-event"
                  key={`${item.market}-${item.symbol}-${item.ex_date}`}
                  onClick={() => onOpenSymbol(item.symbol, item.market)}
                >
                  <span className={`dividend-dday ${item.d_day <= 3 && item.d_day >= 0 ? "soon" : ""}`}>{ddayLabel(item.d_day)}</span>
                  <strong>{item.name || item.symbol}</strong>
                  <em>{item.symbol} · 배당락 {item.ex_date}</em>
                  <span>{money(item.amount, item.currency)} · 지급 {item.pay_date || "-"}</span>
                </button>
              ))}
            </div>
          </section>
        ))}
      </div>

      {recentItems.length > 0 ? (
        <section className="dividend-recent-section">
          <div className="holdings-head">
            <h3>최근 배당락</h3>
          </div>
          <div className="dividend-recent-grid">
            {recentItems.map((item) => (
              <button
                type="button"
                className="dividend-recent-card"
                key={`recent-${item.market}-${item.symbol}-${item.ex_date}`}
                onClick={() => onOpenSymbol(item.symbol, item.market)}
              >
                <strong>{item.name || item.symbol}</strong>
                <span>{item.symbol} · {item.market}</span>
                <em>배당락 {item.ex_date}</em>
                <b>{money(item.amount, item.currency)}</b>
              </button>
            ))}
          </div>
        </section>
      ) : data ? (
        <article className="dividend-empty compact-empty">
          <h4>최근 0</h4>
        </article>
      ) : null}
    </div>
  );
}
