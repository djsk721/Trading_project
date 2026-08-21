import { FormEvent, useEffect, useState } from "react";
import { AccountOverview, HoldingItem, api } from "../api";

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
      const r = await api.saveBrokerKeys({
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
      });
      setStatus(r as BrokerStatus);
      if ((r as BrokerStatus).active_broker === "toss" || (r as BrokerStatus).active_broker === "kis") {
        setActive((r as BrokerStatus).active_broker as "kis" | "toss");
      }
      setKis((prev) => ({ ...prev, app_key: "", app_secret: "" }));
      setToss((prev) => ({ ...prev, client_id: "", client_secret: "" }));
      setMsg("연동 정보를 저장했습니다. 선택한 증권사로 시세·주문이 연결됩니다.");
      onBrokerChange?.(active);
      onRefresh();
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
      setStatus(r as BrokerStatus);
      setMsg("개인 키를 지웠습니다. 서버 기본 연동(.env)을 사용합니다.");
      onRefresh();
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
      const r = await api.saveBrokerKeys({ active: next });
      setStatus(r as BrokerStatus);
      setMsg(next === "toss" ? "토스증권으로 시세·주문을 전환했습니다." : "한국투자증권으로 시세·주문을 전환했습니다.");
      onBrokerChange?.(next);
      onRefresh();
    } catch (err: unknown) {
      setMsg(err instanceof Error ? err.message : "증권사 전환에 실패했습니다.");
    } finally {
      setSaving(false);
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

      {!account?.connected && (
        <p className="muted">
          아직 거래 연동이 없습니다. 사용할 증권사를 고른 뒤 API 키를 저장하거나, 서버 `.env`를 확인해 주세요.
        </p>
      )}
      {account?.error && <div className="error-box">{account.error}</div>}

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
                placeholder={status?.kis_account_masked || ""}
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
