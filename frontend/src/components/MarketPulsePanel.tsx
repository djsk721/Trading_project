import { useEffect, useMemo, useState } from "react";
import { api, MarketNewsItem, NewsSummary, TabDigest } from "../api";
import { formatNewsTime } from "../formatTime";
import NewsBriefingModal, { parseBriefing } from "./NewsBriefingModal";

type Props = {
  open: boolean;
  onClose: () => void;
  provider?: string;
};

const CAT_ALL = "all";

export default function MarketPulsePanel({ open, onClose, provider = "nvidia" }: Props) {
  const [items, setItems] = useState<MarketNewsItem[]>([]);
  const [categories, setCategories] = useState<{ id: string; label: string }[]>([]);
  const [cat, setCat] = useState(CAT_ALL);
  const [sortMode, setSortMode] = useState<"importance" | "date">("importance");
  const [loading, setLoading] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [digestPreparing, setDigestPreparing] = useState(false);
  const [digests, setDigests] = useState<Record<string, TabDigest>>({});
  const [digestDay, setDigestDay] = useState("");
  const [error, setError] = useState("");
  const [fetchedAt, setFetchedAt] = useState("");

  const [summaryOpen, setSummaryOpen] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summary, setSummary] = useState<NewsSummary | null>(null);
  const [active, setActive] = useState<MarketNewsItem | null>(null);

  const filtered = useMemo(() => {
    const base = cat === CAT_ALL ? items : items.filter((i) => i.category === cat);
    return [...base].sort((a, b) => {
      const ia = a.importance || 0;
      const ib = b.importance || 0;
      const da = a.published_at || "";
      const db = b.published_at || "";
      if (sortMode === "date") {
        return db.localeCompare(da) || ib - ia;
      }
      return ib - ia || db.localeCompare(da);
    });
  }, [items, cat, sortMode]);

  const tabDigest = digests[cat];
  const tabBriefing = useMemo(
    () => (tabDigest?.text ? parseBriefing(tabDigest.text) : null),
    [tabDigest]
  );

  function applyMarketPayload(r: Awaited<ReturnType<typeof api.marketNews>>) {
    setItems(r.items || []);
    setCategories(r.categories || []);
    setFetchedAt(r.fetched_at || "");
    setPreparing(Boolean(r.preparing));
    setDigests(r.digests || {});
    setDigestPreparing(Boolean(r.digest_preparing));
    setDigestDay(r.digest_day || "");
  }

  async function loadMarket(force = false) {
    setLoading(true);
    setError("");
    try {
      const r = await api.marketNews({
        prepare: true,
        force,
        provider: provider === "auto" ? "" : provider,
        sort: sortMode,
      });
      applyMarketPayload(r);
    } catch (e: any) {
      setError(e.message || "시황 뉴스를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!open) return;
    loadMarket(false);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (summaryOpen) setSummaryOpen(false);
        else onClose();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // 백그라운드 요약·탭 시황 정리 진행 중이면 주기적으로 갱신
  useEffect(() => {
    if (!open || (!preparing && !digestPreparing)) return;
    const t = window.setInterval(() => {
      api.marketNews({ prepare: false, force: false, sort: sortMode }).then((r) => {
        applyMarketPayload(r);
      }).catch(() => undefined);
    }, 8000);
    return () => window.clearInterval(t);
  }, [open, preparing, digestPreparing, sortMode]);

  async function openSummary(item: MarketNewsItem) {
    setActive(item);
    setSummaryOpen(true);
    setSummaryLoading(true);
    setSummary(null);
    try {
      const r = await api.summarizeNews({
        url: item.url,
        title: item.title_original || item.title,
        snippet: item.summary,
        source: item.source,
        provider: provider === "auto" ? "" : provider,
      });
      setSummary(r);
      const titleKo = r.title_ko || r.title || item.title;
      setActive((prev) => (prev ? { ...prev, title: titleKo } : prev));
      setItems((prev) =>
        prev.map((x) =>
          x.id === item.id
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
      setSummary({
        id: item.id,
        url: item.url,
        title: item.title,
        source: item.source,
        summary_ko: e.message || "요약을 불러오지 못했습니다.",
        provider: "none",
        cached: false,
      });
    } finally {
      setSummaryLoading(false);
    }
  }

  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="market-pulse-panel"
        role="dialog"
        aria-modal="true"
        aria-label="시황 뉴스"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="market-pulse-head">
          <div>
            <h2>시황 뉴스</h2>
            <p className="muted">
              한국·미국·세계·리스크 시황
              {digestDay ? ` · ${digestDay}` : ""}
              {fetchedAt ? ` · 수집 ${fetchedAt.slice(0, 16).replace("T", " ")}` : ""}
              {digestPreparing ? " · 시황 정리 중…" : preparing ? " · 브리핑 준비 중…" : ""}
            </p>
          </div>
          <div className="row">
            <button
              type="button"
              className="btn secondary"
              onClick={() => loadMarket(true)}
              disabled={loading}
            >
              {loading ? "불러오는 중…" : "새로고침"}
            </button>
            <button type="button" className="btn ghost" onClick={onClose}>
              닫기
            </button>
          </div>
        </div>

        <div className="market-cat-tabs" role="tablist">
          <button
            type="button"
            className={`market-cat ${cat === CAT_ALL ? "active" : ""}`}
            onClick={() => setCat(CAT_ALL)}
          >
            전체
          </button>
          {categories.map((c) => (
            <button
              key={c.id}
              type="button"
              className={`market-cat ${cat === c.id ? "active" : ""}`}
              onClick={() => setCat(c.id)}
            >
              {c.label}
            </button>
          ))}
          <span className="market-sort-sep" aria-hidden />
          <button
            type="button"
            className={`market-cat ${sortMode === "importance" ? "active" : ""}`}
            onClick={() => setSortMode("importance")}
          >
            중요도순
          </button>
          <button
            type="button"
            className={`market-cat ${sortMode === "date" ? "active" : ""}`}
            onClick={() => setSortMode("date")}
          >
            최신순
          </button>
        </div>

        {error && <div className="error-box">{error}</div>}

        <div className="market-pulse-list">
          <section className="tab-digest" aria-label="금일 시황 정리">
            <div className="tab-digest-head">
              <span className="tab-digest-kicker">금일 시황</span>
              <span className="tab-digest-meta">
                {tabDigest?.category_label || (cat === CAT_ALL ? "전체" : cat)}
                {tabDigest?.source_count
                  ? ` · 기사 ${tabDigest.source_count}건 반영`
                  : ""}
                {tabDigest?.updated_at
                  ? ` · 정리 ${formatNewsTime(tabDigest.updated_at, { withYear: true })}`
                  : ""}
              </span>
            </div>
            {(!tabDigest?.ready || !tabDigest?.text) && (
              <div className="tab-digest-loading">
                <span className="spinner" />
                <p className="muted">오늘 기사 요약을 모아 시황을 정리하는 중…</p>
              </div>
            )}
            {tabDigest?.ready && tabBriefing && (
              <div className="tab-digest-body">
                {tabBriefing.lead && <p className="tab-digest-lead">{tabBriefing.lead}</p>}
                {tabBriefing.body.map((p, i) => (
                  <p key={i}>{p}</p>
                ))}
                {!tabBriefing.lead && !tabBriefing.body.length && (
                  <p>{tabDigest.text}</p>
                )}
                {tabBriefing.note && (
                  <p className="tab-digest-note">{tabBriefing.note}</p>
                )}
              </div>
            )}
          </section>

          {loading && !filtered.length && <p className="muted">시황 뉴스를 불러오는 중…</p>}
          {!loading && !filtered.length && (
            <p className="muted">표시할 시황 뉴스가 없습니다.</p>
          )}
          {filtered.map((n) => (
            <article key={n.id || n.url} className="card-item market-news-item">
              <div className="row" style={{ gap: 6, marginBottom: 6, flexWrap: "wrap" }}>
                <span className="chip accent">{n.category_label || n.category}</span>
                {typeof n.importance === "number" && n.importance > 0 && (
                  <span className={`chip imp imp-${n.importance}`} title={n.importance_reason || ""}>
                    중요도 {n.importance}
                  </span>
                )}
                <span className="chip">{n.source || "언론사"}</span>
                <span className="chip" title={n.published_at || ""}>
                  {formatNewsTime(n.published_at)}
                </span>
                {n.has_ai_summary ? (
                  <span className="chip">브리핑 준비됨</span>
                ) : (
                  <span className="chip">정리 중</span>
                )}
              </div>
              <h4>{n.title}</h4>
              {n.summary && <p>{n.summary}</p>}
              <div className="row" style={{ marginTop: 10, gap: 8 }}>
                <button type="button" className="btn" onClick={() => openSummary(n)}>
                  브리핑 보기
                </button>
                {n.url && (
                  <a
                    className="source-link"
                    href={n.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    원문 보기 →
                  </a>
                )}
              </div>
            </article>
          ))}
        </div>
      </div>

      <NewsBriefingModal
        open={summaryOpen}
        onClose={() => setSummaryOpen(false)}
        loading={summaryLoading}
        summary={summary}
        active={active}
        kicker="시황 브리핑"
      />
    </div>
  );
}
