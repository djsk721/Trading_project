import { useMemo, useState } from "react";
import { RecommendItem, ScanItem } from "../api";

type BriefLike = Pick<
  ScanItem,
  | "rank"
  | "symbol"
  | "name"
  | "market"
  | "score"
  | "price"
  | "change_pct"
  | "rsi"
  | "trend"
  | "sector"
  | "status_label"
  | "highlights"
  | "metric_note"
  | "detail_summary"
  | "ai_summary"
  | "reasons"
>;

type PickOverlay = Pick<
  RecommendItem,
  "rank" | "score" | "stance" | "buy_price" | "sell_price" | "reasons"
>;

type Props = {
  row: BriefLike;
  pick?: PickOverlay | null;
  onOpenChart: () => void;
};

function statusClass(label: string) {
  if (label.includes("상승")) return "up";
  if (label.includes("하락") || label.includes("과매도")) return "down";
  if (label.includes("과열")) return "warn";
  return "neutral";
}

function formatPrice(value: number, market: string) {
  if (!Number.isFinite(value) || value <= 0) return "-";
  if ((market || "").toUpperCase() === "KRX") {
    return Math.round(value).toLocaleString("ko-KR");
  }
  return value.toLocaleString();
}

export default function RecBriefCard({ row, pick, onOpenChart }: Props) {
  const [open, setOpen] = useState(false);

  const statusLabel =
    row.status_label ||
    (Number(row.rsi) >= 70 ? "과열 주의" : Number(row.rsi) <= 30 ? "과매도 구간" : "관망");

  const highlights = useMemo(() => {
    const fromAi = (row.highlights || []).filter(Boolean).slice(0, 2);
    if (fromAi.length) return fromAi;
    return (row.reasons || []).slice(0, 2);
  }, [row.highlights, row.reasons]);

  const metricNote =
    row.metric_note ||
    (Number.isFinite(row.rsi) ? `RSI ${Number(row.rsi).toFixed(1)}` : "");

  const detail =
    row.detail_summary || row.ai_summary || "";

  const changeClass = row.change_pct >= 0 ? "up" : "down";

  return (
    <article className="rec-brief-card">
      <div className="rec-brief-top">
        <div className="rec-brief-title-row">
          <div className="rec-brief-rank">#{pick?.rank ?? row.rank}</div>
          <div className="rec-brief-names">
            <strong className="rec-brief-name">{row.name}</strong>
            {row.sector ? <span className="rec-brief-sector">{row.sector}</span> : null}
          </div>
        </div>
        <div className="rec-brief-symbol">{row.symbol}</div>
      </div>

      <div className="rec-brief-kpi">
        <span className={`rec-status-badge ${statusClass(statusLabel)}`}>{statusLabel}</span>
        <span className={`rec-kpi ${changeClass}`}>
          {row.change_pct >= 0 ? "+" : ""}
          {row.change_pct}%
        </span>
        <span className="rec-kpi muted-kpi">RSI {Number(row.rsi).toFixed(1)}</span>
        {pick ? <span className="rec-kpi score-kpi">{pick.score}점</span> : null}
        {pick ? <span className="rec-pick-badge">AI Pick</span> : null}
      </div>

      <ul className="rec-highlights">
        {highlights.length > 0 ? (
          highlights.map((line) => <li key={line}>{line}</li>)
        ) : (
          <li className="muted">추천 근거 생성 중</li>
        )}
      </ul>

      {metricNote ? <p className="rec-metric-note">{metricNote}</p> : null}

      {pick && (pick.buy_price || pick.sell_price) ? (
        <div className="rec-targets compact">
          <span>
            매수 <strong>{formatPrice(pick.buy_price || 0, row.market)}</strong>
          </span>
          <span>
            매도 <strong>{formatPrice(pick.sell_price || 0, row.market)}</strong>
          </span>
        </div>
      ) : null}

      {detail ? (
        <div className="rec-detail-block">
          <button
            type="button"
            className="rec-detail-toggle"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
          >
            {open ? "상세 근거 접기" : "상세 근거 보기"}
          </button>
          {open ? <p className="rec-detail-text">{detail}</p> : null}
        </div>
      ) : null}

      <div className="rec-brief-actions">
        <button type="button" className="btn secondary" onClick={onOpenChart}>
          차트/분석 →
        </button>
      </div>
    </article>
  );
}
