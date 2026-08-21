import { useEffect, useMemo, useRef, useState } from "react";
import { api, Sec13FDashboard, Sec13FHolding, Sec13FManager, Sec13FManagerAnalysis } from "../api";

const MANAGER_RESULT_LIMIT = 30;

function moneyUsd(value: number) {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  return `$${Math.round(value).toLocaleString()}`;
}

function pct(value?: number) {
  const n = Number(value || 0) * 100;
  if (!Number.isFinite(n)) return "0.00%";
  if (n >= 10) return `${n.toFixed(1)}%`;
  if (n >= 1) return `${n.toFixed(2)}%`;
  return `${n.toFixed(3)}%`;
}

function compactNumber(value?: number) {
  const n = Number(value || 0);
  if (!Number.isFinite(n)) return "-";
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return Math.round(n).toLocaleString();
}

function signedNumber(value?: number) {
  const n = Number(value || 0);
  if (!Number.isFinite(n) || n === 0) return "0";
  return `${n > 0 ? "+" : "-"}${compactNumber(Math.abs(n))}`;
}

function changeLabel(type?: string) {
  const key = String(type || "UNCHANGED").toUpperCase();
  if (key === "NEW") return "신규";
  if (key === "INCREASED") return "매수 증가";
  if (key === "DECREASED") return "매도 감소";
  if (key === "SOLD") return "전량 매도";
  if (key === "UNKNOWN") return "전분기 기준 없음";
  return "변동 없음";
}

function changeClass(type?: string) {
  const key = String(type || "UNCHANGED").toUpperCase();
  if (key === "NEW") return "new";
  if (key === "INCREASED") return "buy";
  if (key === "DECREASED") return "sell";
  if (key === "SOLD") return "sold";
  if (key === "UNKNOWN") return "unknown";
  return "flat";
}

export default function Sec13FWorkspace() {
  const pageRef = useRef<HTMLElement | null>(null);
  const [dashboard, setDashboard] = useState<Sec13FDashboard | null>(null);
  const [managers, setManagers] = useState<Sec13FManager[]>([]);
  const [managerQuery, setManagerQuery] = useState("");
  const [selectedManager, setSelectedManager] = useState<Sec13FManager | null>(null);
  const [managerHoldings, setManagerHoldings] = useState<Sec13FHolding[]>([]);
  const [managerAnalysis, setManagerAnalysis] = useState<Sec13FManagerAnalysis | null>(null);
  const [holdingQuery, setHoldingQuery] = useState("");
  const [detailTab, setDetailTab] = useState<"holdings" | "analysis">("holdings");
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [message, setMessage] = useState("");

  async function loadDashboard() {
    setLoading(true);
    setMessage("");
    try {
      const [d, m, a] = await Promise.all([
        api.sec13fDashboard(),
        api.sec13fManagers("", MANAGER_RESULT_LIMIT),
        api.sec13fManagersAnalysis("", MANAGER_RESULT_LIMIT),
      ]);
      setDashboard(d);
      setManagers(m.items || []);
      setManagerAnalysis(a);
    } catch (e: any) {
      setMessage(e.message || "13F 캐시를 불러오지 못했습니다. 업데이트를 먼저 실행하세요.");
    } finally {
      setLoading(false);
    }
  }

  async function runUpdate(force = false) {
    setLoading(true);
    setMessage("SEC 최신 13F 데이터 확인 중...");
    try {
      const r = await api.sec13fUpdate(force);
      setMessage(
        r.update_status === "current"
          ? `최신 캐시 사용 중: ${r.dataset || "-"}`
          : `업데이트 완료: ${r.dataset || "-"}`
      );
      await loadDashboard();
    } catch (e: any) {
      setMessage(e.message || "SEC 13F 업데이트에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  }

  async function searchManagers() {
    setLoading(true);
    setAnalysisLoading(true);
    try {
      const [r, a] = await Promise.all([
        api.sec13fManagers(managerQuery, MANAGER_RESULT_LIMIT),
        api.sec13fManagersAnalysis(managerQuery, MANAGER_RESULT_LIMIT),
      ]);
      setManagers(r.items || []);
      setManagerAnalysis(a);
    } catch (e: any) {
      setMessage(e.message || "투자자 검색 실패");
    } finally {
      setLoading(false);
      setAnalysisLoading(false);
    }
  }

  async function openManager(cik: string) {
    if (!cik) return;
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
    setDetailLoading(true);
    try {
      const r = await api.sec13fManager(cik, { limit: 500 });
      setSelectedManager(r.manager);
      setManagerHoldings(r.holdings || []);
      setHoldingQuery("");
    } catch (e: any) {
      setMessage(e.message || "투자자 상세 조회 실패");
    } finally {
      setDetailLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard();
    window.requestAnimationFrame(() => {
      pageRef.current?.focus({ preventScroll: true });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const q = managerQuery.trim();
    const timer = window.setTimeout(() => {
      if (q.length >= 2 || q.length === 0) {
        void searchManagers();
      }
    }, 300);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [managerQuery]);

  useEffect(() => {
    if (!selectedManager) return;
    const q = holdingQuery.trim();
    const timer = window.setTimeout(async () => {
      setDetailLoading(true);
      try {
          const r = await api.sec13fManager(selectedManager.cik, {
            holding_q: q,
            limit: 500,
          });
        setSelectedManager(r.manager);
        setManagerHoldings(r.holdings || []);
      } catch (e: any) {
        setMessage(e.message || "보유 기업 검색 실패");
      } finally {
        setDetailLoading(false);
      }
    }, 300);
    return () => window.clearTimeout(timer);
  }, [holdingQuery, selectedManager?.cik]);

  const meta = dashboard?.metadata || {};
  const portfolioRows = useMemo(
    () => [...managerHoldings].sort((a, b) => Number(b.portfolio_weight || 0) - Number(a.portfolio_weight || 0)).slice(0, holdingQuery.trim() ? 500 : 120),
    [managerHoldings, holdingQuery]
  );
  const topWeight = Number(portfolioRows[0]?.portfolio_weight || 0);
  const analysis = useMemo(() => {
    const rows = [...managerHoldings].sort((a, b) => Number(b.portfolio_weight || 0) - Number(a.portfolio_weight || 0));
    const totalWeight = rows.reduce((sum, h) => sum + Number(h.portfolio_weight || 0), 0);
    const top5Weight = rows.slice(0, 5).reduce((sum, h) => sum + Number(h.portfolio_weight || 0), 0);
    const top10Weight = rows.slice(0, 10).reduce((sum, h) => sum + Number(h.portfolio_weight || 0), 0);
    const buckets = rows.reduce<Record<string, number>>((acc, h) => {
      const key = String(h.change_type || "UNCHANGED").toUpperCase();
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});
    const increased = rows
      .filter((h) => Number(h.share_change || 0) > 0)
      .sort((a, b) => Number(b.share_change || 0) - Number(a.share_change || 0))
      .slice(0, 5);
    const decreased = rows
      .filter((h) => Number(h.share_change || 0) < 0)
      .sort((a, b) => Number(a.share_change || 0) - Number(b.share_change || 0))
      .slice(0, 5);
    return { rows, totalWeight, top5Weight, top10Weight, buckets, increased, decreased };
  }, [managerHoldings]);

  return (
    <section ref={pageRef} className="workspace-page sec13f-page" tabIndex={-1}>
      <div className="main-head">
        <div>
          <h2>SEC Form 13F</h2>
          <p className="sub">기관별 13F 보유 종목이 전체 포트폴리오에서 차지하는 비중만 빠르게 확인합니다.</p>
        </div>
        <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
          <button className="btn secondary" onClick={() => runUpdate(false)} disabled={loading}>
            최신 확인
          </button>
          <button className="btn ghost" onClick={() => runUpdate(true)} disabled={loading}>
            강제 갱신
          </button>
        </div>
      </div>

      {message ? <div className="error-box">{message}</div> : null}

      <div className="metrics">
        <div className="metric">
          <div className="label">최근 SEC 분기</div>
          <div className="value">{meta.dataset || "-"}</div>
        </div>
        <div className="metric">
          <div className="label">업데이트</div>
          <div className="value">{meta.updated_at ? String(meta.updated_at).slice(0, 10) : "-"}</div>
        </div>
        <div className="metric">
          <div className="label">Manager</div>
          <div className="value">{(dashboard?.manager_count || 0).toLocaleString()}</div>
        </div>
        <div className="metric">
          <div className="label">종목/CUSIP</div>
          <div className="value">{(dashboard?.issuer_count || 0).toLocaleString()}</div>
        </div>
      </div>

      <div className="sec13f-content-grid">
        <article className="card-item sec13f-manager-card">
          <div>
            <h4>Manager 선택</h4>
            <p className="sub">검색 후 기관을 클릭하세요.</p>
          </div>
          <div className="sec13f-manager-controls">
            <div className="row" style={{ gap: 8 }}>
              <input
                value={managerQuery}
                onChange={(e) => setManagerQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && searchManagers()}
                placeholder="Manager, CIK"
              />
              <button className="btn secondary" onClick={searchManagers} disabled={loading}>검색</button>
            </div>
            <div className="manager-pick-list" aria-label="Manager 선택">
              {managers.map((m) => (
                <button
                  key={m.cik}
                  type="button"
                  className={`manager-pick ${selectedManager?.cik === m.cik ? "active" : ""}`}
                  onClick={() => openManager(m.cik)}
                >
                  <span>{m.manager_name}</span>
                  <strong>{moneyUsd(m.portfolio_value)}</strong>
                </button>
              ))}
              {!managers.length ? <p className="muted">검색 결과가 없습니다.</p> : null}
            </div>
          </div>
        </article>

        {selectedManager ? (
          <article className="card-item sec13f-allocation-card">
          <div className="main-head">
            <div>
              <h4>{selectedManager.manager_name}</h4>
              <p className="sub">
                CIK {selectedManager.cik} · 포트폴리오 {moneyUsd(selectedManager.portfolio_value)} · {selectedManager.holdings_count.toLocaleString()}개 보유
              </p>
            </div>
            <span className="chip accent">{meta.dataset || "-"}</span>
          </div>

          <div className="sec13f-detail-tabs" aria-label="13F 상세 보기">
            <button className={detailTab === "holdings" ? "active" : ""} onClick={() => setDetailTab("holdings")}>
              보유 목록
            </button>
            <button className={detailTab === "analysis" ? "active" : ""} onClick={() => setDetailTab("analysis")}>
              포트폴리오 분석
            </button>
          </div>

          {detailTab === "holdings" ? (
            <>
              <div className="holding-filter">
                <input
                  value={holdingQuery}
                  onChange={(e) => setHoldingQuery(e.target.value)}
                  placeholder="보유 기업명 일부, CUSIP 입력"
                />
              </div>

              <div className="allocation-list">
                {detailLoading ? <p className="muted">비중 데이터를 불러오는 중...</p> : null}
                {portfolioRows.map((h, i) => {
                  const weight = Number(h.portfolio_weight || 0);
                  const width = topWeight > 0 ? Math.max(2, (weight / topWeight) * 100) : 0;
                  return (
                    <div className="allocation-row" key={`${h.cusip}-${i}`}>
                      <div className="allocation-rank">{i + 1}</div>
                      <div className="allocation-main">
                        <div className="allocation-title">
                          <strong>{h.ticker || h.issuer || h.cusip}</strong>
                          <span>{h.ticker ? h.issuer : h.cusip}</span>
                        </div>
                        <div className="allocation-meta">
                          <span className="holding-chip">
                            <em>현재</em>
                            <strong>{compactNumber(h.current_shares ?? h.shares)}주</strong>
                          </span>
                          <span className="holding-chip">
                            <em>전분기</em>
                            <strong>{compactNumber(h.previous_shares)}주</strong>
                          </span>
                          <span className={`holding-chip delta ${Number(h.share_change || 0) > 0 ? "up" : Number(h.share_change || 0) < 0 ? "down" : ""}`}>
                            <em>증감</em>
                            <strong>{signedNumber(h.share_change)}주</strong>
                          </span>
                          <span className={`change-badge ${changeClass(h.change_type)}`}>{changeLabel(h.change_type)}</span>
                        </div>
                        <div className="allocation-track">
                          <div className="allocation-bar" style={{ width: `${width}%` }} />
                        </div>
                      </div>
                      <div className="allocation-pct">{pct(weight)}</div>
                    </div>
                  );
                })}
              </div>
              {!detailLoading && !portfolioRows.length ? (
                <p className="muted">
                  {holdingQuery.trim() ? "해당 기업 검색 결과가 없습니다." : "선택한 Manager의 보유 종목이 없습니다."}
                </p>
              ) : null}
            </>
          ) : (
            <div className="portfolio-analysis">
              <div className="analysis-metrics">
                <div className="analysis-tile">
                  <span>분석 종목</span>
                  <strong>{analysis.rows.length.toLocaleString()}개</strong>
                  <em>현재 로드된 상위 종목 기준</em>
                </div>
                <div className="analysis-tile">
                  <span>Top 5 집중도</span>
                  <strong>{pct(analysis.top5Weight)}</strong>
                  <em>상위 5개 비중 합계</em>
                </div>
                <div className="analysis-tile">
                  <span>Top 10 집중도</span>
                  <strong>{pct(analysis.top10Weight)}</strong>
                  <em>상위 10개 비중 합계</em>
                </div>
                <div className="analysis-tile">
                  <span>로드 비중</span>
                  <strong>{pct(analysis.totalWeight)}</strong>
                  <em>전체 포트폴리오 대비</em>
                </div>
              </div>

              <div className="analysis-section">
                <h5>변화 유형 분포</h5>
                <div className="change-summary">
                  {["NEW", "INCREASED", "DECREASED", "SOLD", "UNCHANGED", "UNKNOWN"].map((key) => (
                    <span key={key} className={`change-badge ${changeClass(key)}`}>
                      {changeLabel(key)} {analysis.buckets[key] || 0}
                    </span>
                  ))}
                </div>
              </div>

              <div className="analysis-columns">
                <div className="analysis-section">
                  <h5>주요 매수 증가</h5>
                  {analysis.increased.length ? analysis.increased.map((h) => (
                    <div className="analysis-move-row" key={`up-${h.cusip}`}>
                      <strong>{h.ticker || h.issuer || h.cusip}</strong>
                      <span className="up">{signedNumber(h.share_change)}주</span>
                      <em>{pct(h.portfolio_weight)}</em>
                    </div>
                  )) : <p className="muted">매수 증가 데이터가 없습니다.</p>}
                </div>
                <div className="analysis-section">
                  <h5>주요 매도 감소</h5>
                  {analysis.decreased.length ? analysis.decreased.map((h) => (
                    <div className="analysis-move-row" key={`down-${h.cusip}`}>
                      <strong>{h.ticker || h.issuer || h.cusip}</strong>
                      <span className="down">{signedNumber(h.share_change)}주</span>
                      <em>{pct(h.portfolio_weight)}</em>
                    </div>
                  )) : <p className="muted">매도 감소 데이터가 없습니다.</p>}
                </div>
              </div>
            </div>
          )}
          </article>
        ) : (
          <article className="card-item sec13f-allocation-card">
            <div className="main-head">
              <div>
                <h4>현재 Manager 목록 포트폴리오 분석</h4>
                <p className="sub">왼쪽 목록에 표시된 기관들의 보유 종목을 합산해 보여줍니다.</p>
              </div>
              <span className="chip accent">{managerAnalysis?.manager_count || 0}개 기관</span>
            </div>
            {analysisLoading ? <p className="muted">목록 분석을 불러오는 중...</p> : null}
            {managerAnalysis && managerAnalysis.top_holdings.length ? (
              <div className="portfolio-analysis">
                <div className="analysis-metrics">
                  <div className="analysis-tile">
                    <span>분석 기관</span>
                    <strong>{managerAnalysis.manager_count.toLocaleString()}개</strong>
                    <em>현재 Manager 검색 결과 기준</em>
                  </div>
                  <div className="analysis-tile">
                    <span>종목/CUSIP</span>
                    <strong>{managerAnalysis.holding_count.toLocaleString()}개</strong>
                    <em>중복 CUSIP 제거</em>
                  </div>
                  <div className="analysis-tile">
                    <span>합산 가치</span>
                    <strong>{moneyUsd(managerAnalysis.total_value)}</strong>
                    <em>목록 기관 보유액 합계</em>
                  </div>
                  <div className="analysis-tile">
                    <span>Top 보유</span>
                    <strong>{managerAnalysis.top_holdings[0]?.ticker || managerAnalysis.top_holdings[0]?.issuer || "-"}</strong>
                    <em>{pct(managerAnalysis.top_holdings[0]?.portfolio_weight)}</em>
                  </div>
                </div>

                <div className="analysis-section">
                  <h5>변화 유형 분포</h5>
                  <div className="change-summary">
                    {["NEW", "INCREASED", "DECREASED", "SOLD", "UNCHANGED", "UNKNOWN"].map((key) => (
                      <span key={key} className={`change-badge ${changeClass(key)}`}>
                        {changeLabel(key)} {managerAnalysis.change_summary[key] || 0}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="analysis-section">
                  <h5>목록 기관들이 많이 보유한 종목</h5>
                  <div className="allocation-list compact-analysis-list">
                    {managerAnalysis.top_holdings.slice(0, 10).map((h, i) => (
                      <div className="allocation-row" key={`group-${h.cusip}-${i}`}>
                        <div className="allocation-rank">{i + 1}</div>
                        <div className="allocation-main">
                          <div className="allocation-title">
                            <strong>{h.ticker || h.issuer || h.cusip}</strong>
                            <span>{h.ticker ? h.issuer : h.cusip}</span>
                          </div>
                          <div className="allocation-meta">
                            <span className="holding-chip">
                              <em>보유기관</em>
                              <strong>{h.manager_count || 0}곳</strong>
                            </span>
                            <span className="holding-chip">
                              <em>합산</em>
                              <strong>{moneyUsd(h.value)}</strong>
                            </span>
                          </div>
                        </div>
                        <div className="allocation-pct">{pct(h.portfolio_weight)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : !analysisLoading ? (
              <div className="sec13f-empty-card">
                <h4>분석 결과가 없습니다.</h4>
                <p className="sub">검색어를 바꾸거나 최신 확인을 실행해 주세요.</p>
              </div>
            ) : null}
          </article>
        )}
      </div>
    </section>
  );
}
