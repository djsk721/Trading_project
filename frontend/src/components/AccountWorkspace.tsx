import { FormEvent, useEffect, useState } from "react";
import { AccountOverview, BrokerCapabilities, HoldingExitItem, HoldingExitResponse, HoldingItem, api } from "../api";
import { brokerKeysPayload, clearBrokerCreds, persistBrokerCreds } from "../brokerSession";

type BrokerStatus = {
  active_broker?: "kis" | "toss" | string;
  kis_source?: string;
  kis_configured?: boolean;
  kis_virtual?: boolean;
  kis_hts_id?: string;
  kis_account_masked?: string;
  kis_app_key_masked?: string;
  toss_source?: string;
  toss_configured?: boolean;
  toss_ready?: boolean;
  toss_client_id_masked?: string;
  toss_account_masked?: string;
};

type Props = {
  account: AccountOverview | null;
  loading: boolean;
  onRefresh: () => void;
  onOpenHolding: (h: HoldingItem) => void;
  formatMoney: (value: number, currency?: "KRW" | "USD") => string;
  activeBroker?: "kis" | "toss";
  onBrokerChange?: (broker: "kis" | "toss") => void;
};

const KEYS_STORE = "td_broker_keys";

function loadLocalKeys() {
  try {
    return JSON.parse(localStorage.getItem(KEYS_STORE) || "{}") as {
      kis?: Record<string, string | boolean>;
      toss?: Record<string, string>;
    };
  } catch {
    return {};
  }
}

export default function AccountWorkspace({
  account,
  loading,
  onRefresh,
  onOpenHolding,
  formatMoney,
  activeBroker,
  onBrokerChange,
}: Props) {
  const [status, setStatus] = useState<BrokerStatus | null>(null);
  const [kis, setKis] = useState({
    hts_id: "",
    app_key: "",
    app_secret: "",
    account: "",
    virtual: true,
  });
  const [toss, setToss] = useState({ client_id: "", client_secret: "", account: "" });
  const [active, setActive] = useState<"kis" | "toss">(activeBroker || "kis");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const [exitCheck, setExitCheck] = useState<HoldingExitResponse | null>(null);
  const [exitLoading, setExitLoading] = useState(false);
  const [exitMsg, setExitMsg] = useState("");
  const [diagnostics, setDiagnostics] = useState<BrokerCapabilities | null>(null);

  useEffect(() => {
    const local = loadLocalKeys();
    if (local.kis) {
      setKis((prev) => ({
        ...prev,
        hts_id: String(local.kis?.hts_id || ""),
        account: String(local.kis?.account || ""),
        virtual: local.kis?.virtual !== false,
      }));
    }
    if (local.toss) {
      setToss((prev) => ({
        ...prev,
        account: String(local.toss?.account || ""),
      }));
    }
    api
      .brokerStatus()
      .then((r) => {
        const next = r as BrokerStatus;
        setStatus(next);
        if (next.active_broker === "toss" || next.active_broker === "kis") {
          setActive(next.active_broker);
        }
      })
      .catch(() => setStatus(null));
    api.brokerDiagnostics().then(setDiagnostics).catch(() => setDiagnostics(null));
  }, []);

  useEffect(() => {
    if (activeBroker === "kis" || activeBroker === "toss") {
      setActive(activeBroker);
    }
  }, [activeBroker]);

  async function saveKeys(e: FormEvent) {
    e.preventDefault();
    setSaving(true);
    setMsg("");
    try {
      localStorage.setItem(
        KEYS_STORE,
        JSON.stringify({
          kis: { hts_id: kis.hts_id, account: kis.account, virtual: kis.virtual },
          toss: { account: toss.account },
        })
      );
      const payload = {
        active,
        kis: {
          hts_id: kis.hts_id,
          app_key: kis.app_key,
          app_secret: kis.app_secret,
          account: kis.account,
          virtual: kis.virtual,
        },
        toss: {
          client_id: toss.client_id,
          client_secret: toss.client_secret,
          account: toss.account,
        },
      };
      persistBrokerCreds(payload);
      const r = await api.saveBrokerKeys(payload);
      setStatus(r as BrokerStatus);
      if ((r as BrokerStatus).active_broker === "toss" || (r as BrokerStatus).active_broker === "kis") {
        setActive((r as BrokerStatus).active_broker as "kis" | "toss");
      }
      setKis((prev) => ({ ...prev, app_key: "", app_secret: "" }));
      setToss((prev) => ({ ...prev, client_id: "", client_secret: "" }));
      setMsg("연동 정보를 저장했습니다. 선택한 증권사로 시세·주문이 연결됩니다.");
      onBrokerChange?.(active);
      onRefresh();
      api.brokerDiagnostics().then(setDiagnostics).catch(() => setDiagnostics(null));
    } catch (err: unknown) {
      setMsg(err instanceof Error ? err.message : "저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  }

  async function clearUserKeys() {
    setSaving(true);
    try {
      const r = await api.saveBrokerKeys({
        kis: { clear: true },
        toss: { clear: true },
      });
      clearBrokerCreds();
      setStatus(r as BrokerStatus);
      setMsg("개인 키를 지웠습니다. 서버 기본 연동(.env)을 사용합니다.");
      onRefresh();
      api.brokerDiagnostics().then(setDiagnostics).catch(() => setDiagnostics(null));
    } catch (err: unknown) {
      setMsg(err instanceof Error ? err.message : "삭제에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  }

  async function switchActive(next: "kis" | "toss") {
    setActive(next);
    setSaving(true);
    try {
      const r = await api.saveBrokerKeys(brokerKeysPayload(next));
      setStatus(r as BrokerStatus);
      setMsg(next === "toss" ? "토스증권으로 시세·주문을 전환했습니다." : "한국투자증권으로 시세·주문을 전환했습니다.");
      onBrokerChange?.(next);
      onRefresh();
      api.brokerDiagnostics().then(setDiagnostics).catch(() => setDiagnostics(null));
    } catch (err: unknown) {
      setMsg(err instanceof Error ? err.message : "증권사 전환에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  }

  async function loadExitCheck(force = false) {
    setExitLoading(true);
    setExitMsg("");
    try {
      const r = await api.holdingsExit({ force, days: 160 });
      setExitCheck(r);
    } catch (err: unknown) {
      setExitMsg(err instanceof Error ? err.message : "보유 매도 점검에 실패했습니다.");
    } finally {
      setExitLoading(false);
    }
  }

  const kisSource =
    status?.kis_source === "user"
      ? "내 API 키"
      : status?.kis_source === "env"
        ? "서버 기본값"
        : "미연결";

  return (
    <div className="account-page">
      <div className="account-hero">
        <div>
          <p className="account-kicker">내 계좌</p>
          <h2>
            {account?.connected
              ? account.virtual
                ? "모의투자 통합 잔고"
                : "실전 통합 잔고"
              : "계좌 연동 필요"}
          </h2>
          <p className="muted">
            {account?.account
              ? `계좌 ${account.account}`
              : "로그인에서 고른 증권사로 시세·주문이 연결됩니다. API 키는 아래에서 저장하세요."}
          </p>
        </div>
        <button className="btn" onClick={onRefresh} disabled={loading}>
          {loading ? "조회 중..." : "잔고 새로고침"}
        </button>
      </div>

      {!account?.connected && !loading && (
        <p className="muted">
          아직 거래 연동이 없습니다. 로그인에서 넣은 키로 연결되며, 계좌번호는 12345678-01 형식입니다.
          HTS ID(@로 시작)를 계좌 칸에 넣지 마세요.
        </p>
      )}
      {loading && (
        <p className="muted">로그인 키로 잔고를 조회하고 있습니다...</p>
      )}
      {account?.error && !loading && <div className="error-box">{account.error}</div>}

      {account?.connected && (
        <>
          <div className="account-metrics">
            <div className="account-metric">
              <div className="label">총평가</div>
              <div className="value">{formatMoney(account.total_eval_krw)}</div>
            </div>
            <div className="account-metric">
              <div className="label">원화 현금</div>
              <div className="value">{formatMoney(account.domestic.deposit_krw)}</div>
            </div>
            <div className="account-metric">
              <div className="label">달러 현금</div>
              <div className="value">{formatMoney(account.overseas.deposit_usd, "USD")}</div>
            </div>
            <div className="account-metric">
              <div className="label">평가손익</div>
              <div className={`value ${account.profit_loss >= 0 ? "up" : "down"}`}>
                {account.profit_loss >= 0 ? "+" : ""}
                {formatMoney(account.profit_loss)} ({account.profit_loss_rate.toFixed(2)}%)
              </div>
            </div>
          </div>

          <section className="holdings-exit-panel">
            <div className="holdings-head">
              <div>
                <h3>보유 매도 점검</h3>
                <p className="muted">
                  손익, 비중, RSI, MACD, 추세, 이동평균, Bollinger Band로 sell / trim / hold를 점검합니다.
                </p>
              </div>
              <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                <button className="btn secondary" onClick={() => loadExitCheck(false)} disabled={exitLoading}>
                  {exitLoading ? "점검 중..." : "점검 보기"}
                </button>
                <button className="btn ghost" onClick={() => loadExitCheck(true)} disabled={exitLoading}>
                  점검 재생성
                </button>
              </div>
            </div>
            {exitMsg ? <div className="error-box">{exitMsg}</div> : null}
            {exitCheck ? (
              <>
                <div className="exit-summary-row">
                  <span className="exit-pill sell">매도 {exitCheck.summary.sell || 0}</span>
                  <span className="exit-pill trim">축소 {exitCheck.summary.trim || 0}</span>
                  <span className="exit-pill hold">보유 {exitCheck.summary.hold || 0}</span>
                  <span className="muted">
                    {exitCheck.cached ? "캐시" : "갱신"} · {exitCheck.as_of} · AI {exitCheck.used_llm ? "사용" : "미사용"}
                  </span>
                </div>
                <div className="exit-card-grid">
                  {exitCheck.items.map((item) => (
                    <HoldingExitCard
                      key={`${item.market}-${item.symbol}`}
                      item={item}
                      formatMoney={formatMoney}
                      onOpen={() => onOpenHolding({
                        symbol: item.symbol,
                        name: item.name,
                        market: item.market,
                        qty: item.qty,
                        price: item.price,
                        amount: item.amount,
                        profit: item.profit,
                        profit_rate: item.profit_rate,
                        currency: item.currency,
                        scope: item.currency === "USD" ? "overseas" : "domestic",
                      })}
                    />
                  ))}
                </div>
                <p className="muted">{exitCheck.disclaimer}</p>
              </>
            ) : (
              <p className="muted">점검 보기를 누르면 당일 캐시를 표시하고, 점검 재생성은 계좌 캐시를 재사용해 분석만 다시 만듭니다.</p>
            )}
          </section>

          <HoldingsTable
            title="국내 보유"
            subtitle={`예수금 ${formatMoney(account.domestic.deposit_krw)} · 주식 ${formatMoney(
              account.domestic.stocks_value
            )}`}
            rows={account.domestic.holdings || []}
            currency="KRW"
            formatMoney={formatMoney}
            onOpen={onOpenHolding}
          />
          <HoldingsTable
            title="해외 보유"
            subtitle={`예수금 ${formatMoney(account.overseas.deposit_usd, "USD")} · 주식 ${formatMoney(
              account.overseas.stocks_value,
              "USD"
            )}${
              account.overseas.exchange_rate
                ? ` · 환율 ${account.overseas.exchange_rate.toLocaleString()}원`
                : ""
            }`}
            rows={account.overseas.holdings || []}
            currency="USD"
            formatMoney={formatMoney}
            onOpen={onOpenHolding}
          />
        </>
      )}

      <section className="account-settings">
        <h3>증권사 연동</h3>
        <p className="muted">
          현재 사용: {active === "toss" ? "토스증권" : "한국투자증권"}
          {" · "}
          한투: {kisSource}
          {status?.kis_account_masked ? ` · 계좌 ${status.kis_account_masked}` : ""}
          {" · "}
          토스: {status?.toss_configured ? (status?.toss_ready ? "키 저장됨" : "설정됨") : "미설정"}
        </p>
        <div className="broker-active-row" role="radiogroup" aria-label="사용할 증권사">
          <label className="rec-toggle">
            <input
              type="radio"
              name="active-broker"
              checked={active === "kis"}
              onChange={() => switchActive("kis")}
              disabled={saving}
            />
            한국투자증권 사용
          </label>
          <label className="rec-toggle">
            <input
              type="radio"
              name="active-broker"
              checked={active === "toss"}
              onChange={() => switchActive("toss")}
              disabled={saving}
            />
            토스증권 사용
          </label>
        </div>
        <form className="broker-grid" onSubmit={saveKeys}>
          <fieldset>
            <legend>한국투자증권</legend>
            <div className="field">
              <label>HTS 아이디</label>
              <input
                value={kis.hts_id}
                onChange={(e) => setKis({ ...kis, hts_id: e.target.value })}
                autoComplete="off"
              />
            </div>
            <div className="field">
              <label>앱 키</label>
              <input
                value={kis.app_key}
                onChange={(e) => setKis({ ...kis, app_key: e.target.value })}
                placeholder={status?.kis_app_key_masked || "앱 키"}
                autoComplete="off"
              />
            </div>
            <div className="field">
              <label>앱 시크릿</label>
              <input
                type="password"
                value={kis.app_secret}
                onChange={(e) => setKis({ ...kis, app_secret: e.target.value })}
                placeholder="변경할 때만 입력"
                autoComplete="off"
              />
            </div>
            <div className="field">
              <label>계좌번호</label>
              <input
                value={kis.account}
                onChange={(e) => setKis({ ...kis, account: e.target.value })}
                placeholder="예: 12345678-01"
                autoComplete="off"
              />
            </div>
            <label className="rec-toggle">
              <input
                type="checkbox"
                checked={kis.virtual}
                onChange={(e) => setKis({ ...kis, virtual: e.target.checked })}
              />
              모의투자
            </label>
          </fieldset>
          <fieldset>
            <legend>토스증권</legend>
            <div className="field">
              <label>클라이언트 ID</label>
              <input
                value={toss.client_id}
                onChange={(e) => setToss({ ...toss, client_id: e.target.value })}
                placeholder={status?.toss_client_id_masked || "클라이언트 ID"}
                autoComplete="off"
              />
            </div>
            <div className="field">
              <label>클라이언트 시크릿</label>
              <input
                type="password"
                value={toss.client_secret}
                onChange={(e) => setToss({ ...toss, client_secret: e.target.value })}
                placeholder="변경할 때만 입력"
                autoComplete="off"
              />
            </div>
            <div className="field">
              <label>계좌번호</label>
              <input
                value={toss.account}
                onChange={(e) => setToss({ ...toss, account: e.target.value })}
                autoComplete="off"
              />
            </div>
            <p className="muted">
              계좌번호를 비우면 토스 종합매매 계좌를 자동으로 사용합니다. 토스는 모의투자가 없습니다.
            </p>
          </fieldset>
          <div className="broker-actions">
            <button className="btn" type="submit" disabled={saving}>
              {saving ? "저장 중..." : "연동 정보 저장"}
            </button>
            <button className="btn secondary" type="button" onClick={clearUserKeys} disabled={saving}>
              개인 키 지우기
            </button>
          </div>
        </form>
        {msg && <p className="muted" style={{ marginTop: 10 }}>{msg}</p>}
      </section>

      <section className="account-settings broker-diagnostics">
        <div className="holdings-head">
          <h3>브로커 진단</h3>
          <button
            type="button"
            className="btn secondary"
            onClick={() => api.brokerDiagnostics().then(setDiagnostics).catch(() => setDiagnostics(null))}
          >
            진단 새로고침
          </button>
        </div>
        {diagnostics ? (
          <>
            <div className="exit-summary-row">
              <span className={`exit-pill ${diagnostics.connected ? "hold" : "sell"}`}>
                API {diagnostics.connected ? "연결" : "미연결"}
              </span>
              <span className={`exit-pill ${diagnostics.trading_enabled ? "hold" : "trim"}`}>
                주문 {diagnostics.trading_enabled ? "활성" : "차단"}
              </span>
              <span className={`exit-pill ${diagnostics.rate_limited ? "sell" : "hold"}`}>
                Rate limit {diagnostics.rate_limited ? "cooldown" : "정상"}
              </span>
            </div>
            {diagnostics.disabled_reason ? <p className="muted">{diagnostics.disabled_reason}</p> : null}
            <div className="broker-check-grid">
              {(diagnostics.checks || []).map((c) => (
                <div key={c.name} className={`broker-check ${c.ok ? "ok" : "warn"}`}>
                  <strong>{c.name}</strong>
                  <span>{c.detail}</span>
                </div>
              ))}
            </div>
            <div className="broker-cap-list">
              {Object.entries(diagnostics.supports || {}).map(([key, ok]) => (
                <span key={key} className={`mini-cap ${ok ? "ok" : "off"}`}>
                  {key.split("_").join(" ")} {ok ? "ON" : "OFF"}
                </span>
              ))}
            </div>
          </>
        ) : (
          <p className="muted">진단 정보를 불러오지 못했습니다.</p>
        )}
      </section>
    </div>
  );
}

function HoldingsTable({
  title,
  subtitle,
  rows,
  currency,
  formatMoney,
  onOpen,
}: {
  title: string;
  subtitle: string;
  rows: HoldingItem[];
  currency: "KRW" | "USD";
  formatMoney: (value: number, currency?: "KRW" | "USD") => string;
  onOpen: (h: HoldingItem) => void;
}) {
  return (
    <section className="holdings-block">
      <div className="holdings-head">
        <h3>{title}</h3>
        <p className="muted">{subtitle}</p>
      </div>
      {rows.length === 0 ? (
        <p className="muted">보유 종목이 없습니다.</p>
      ) : (
        <div className="scan-table-wrap">
          <table className="scan-table holdings-table">
            <thead>
              <tr>
                <th>종목</th>
                <th>수량</th>
                <th>현재가</th>
                <th>평가</th>
                <th>손익</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((h) => (
                <tr key={`${h.market}-${h.symbol}`}>
                  <td>
                    <strong>{h.name || h.symbol}</strong>
                    <div className="muted">{h.symbol}</div>
                  </td>
                  <td>{h.qty.toLocaleString()}주</td>
                  <td>{formatMoney(h.price, currency)}</td>
                  <td>{formatMoney(h.amount, currency)}</td>
                  <td className={h.profit >= 0 ? "up" : "down"}>
                    {h.profit >= 0 ? "+" : ""}
                    {formatMoney(h.profit, currency)} ({h.profit_rate.toFixed(2)}%)
                  </td>
                  <td>
                    <button className="btn secondary" onClick={() => onOpen(h)}>
                      차트
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function decisionClass(decision: string) {
  if (decision === "sell") return "sell";
  if (decision === "trim") return "trim";
  return "hold";
}

function trendLabel(trend: string) {
  const t = String(trend || "").toUpperCase();
  if (t === "UP") return "상승";
  if (t === "DOWN") return "하락";
  return "중립";
}

function HoldingExitCard({
  item,
  formatMoney,
  onOpen,
}: {
  item: HoldingExitItem;
  formatMoney: (value: number, currency?: "KRW" | "USD") => string;
  onOpen: () => void;
}) {
  const currency = String(item.currency || (item.market === "US" ? "USD" : "KRW")).toUpperCase() === "USD" ? "USD" : "KRW";
  const showKrw = currency !== "KRW" && Number(item.amount_krw || 0) > 0;
  const cls = decisionClass(item.decision);
  return (
    <article className={`rec-brief-card exit-card ${cls}`}>
      <div className="rec-brief-top">
        <div className="rec-brief-title-row">
          <div className="rec-brief-rank">#{item.rank}</div>
          <div className="rec-brief-names">
            <strong className="rec-brief-name">{item.name || item.symbol}</strong>
            <span className="rec-brief-sector">{item.symbol}</span>
          </div>
        </div>
        <span className={`exit-decision-badge ${cls}`}>{item.decision_label}</span>
      </div>

      <div className="rec-brief-kpi">
        <span className={`rec-kpi ${item.profit >= 0 ? "up" : "down"}`}>
          {item.profit_rate >= 0 ? "+" : ""}
          {item.profit_rate.toFixed(2)}%
        </span>
        <span className="rec-kpi muted-kpi">리스크 {item.risk_score.toFixed(0)}</span>
        <span className="rec-kpi muted-kpi">비중 {(item.portfolio_weight * 100).toFixed(1)}%</span>
      </div>

      <div className="exit-metrics">
        <span>평균 <strong>{formatMoney(item.avg_cost, currency)}</strong></span>
        <span>현재 <strong>{formatMoney(item.price, currency)}</strong></span>
        <span>평가 <strong>{formatMoney(item.amount, currency)}</strong></span>
        {showKrw ? <span>원화환산 <strong>{formatMoney(item.amount_krw || 0, "KRW")}</strong></span> : null}
        <span>RSI <strong>{item.rsi.toFixed(1)}</strong></span>
        <span>MACD <strong>{item.macd_signal}</strong></span>
        <span>추세 <strong>{trendLabel(item.trend)}</strong></span>
        {currency !== "KRW" ? <span>환율 <strong>{Number(item.exchange_rate || 0).toLocaleString()}원</strong></span> : null}
      </div>

      <ul className="rec-highlights">
        {(item.highlights.length ? item.highlights : item.reasons.slice(0, 2)).map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>

      {item.ai_summary ? <p className="rec-metric-note">{item.ai_summary}</p> : null}

      <div className="exit-ai-panel">
        <div>
          <span>AI 근거</span>
          <p>{item.ai_rationale || item.ai_summary || "룰 기반 근거를 사용했습니다."}</p>
        </div>
        <div>
          <span>리스크 해석</span>
          <p>{item.ai_risk || "손익, 추세, 모멘텀 변화를 함께 확인해야 합니다."}</p>
        </div>
        <div>
          <span>대응 시나리오</span>
          <p>{item.ai_action || "다음 점검 시 가격·추세·손익률 변화를 재확인하세요."}</p>
        </div>
        <div className="exit-watchpoints">
          {(item.ai_watchpoints || []).slice(0, 3).map((point) => (
            <em key={point}>{point}</em>
          ))}
        </div>
      </div>

      <div className="rec-detail-block">
        <details>
          <summary className="rec-detail-toggle">룰 근거 보기</summary>
          <ul className="rec-highlights">
            {item.reasons.map((line) => <li key={line}>{line}</li>)}
          </ul>
        </details>
      </div>

      <div className="rec-brief-actions">
        <button type="button" className="btn secondary" onClick={onOpen}>
          차트/분석 →
        </button>
      </div>
    </article>
  );
}
