import { FormEvent, useEffect, useState } from "react";
import { api } from "../api";
import { persistBrokerCreds, brokerKeysPayload } from "../brokerSession";

export type BrokerId = "kis" | "toss";

type BrokerStatus = {
  active_broker?: string;
  kis_source?: string;
  kis_configured?: boolean;
  kis_virtual?: boolean;
  toss_source?: string;
  toss_configured?: boolean;
  nvidia_source?: string;
  nvidia_configured?: boolean;
};

type Props = {
  onLogin: (payload: { broker: BrokerId }) => void;
};

const SOURCE_LABEL: Record<string, string> = {
  user: "입력됨",
  env: ".env",
  none: "미설정",
};

function SecretInput({
  id,
  label,
  value,
  onChange,
  placeholder,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  const [show, setShow] = useState(false);
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <div className="secret-wrap">
        <input
          id={id}
          type={show ? "text" : "password"}
          autoComplete="off"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
        />
        <button
          type="button"
          className="btn ghost secret-toggle"
          aria-label={show ? "숨기기" : "표시"}
          aria-pressed={show}
          onClick={() => setShow((v) => !v)}
        >
          {show ? "숨기기" : "보기"}
        </button>
      </div>
    </div>
  );
}

export default function LoginScreen({ onLogin }: Props) {
  const [broker, setBroker] = useState<BrokerId>("kis");
  const [status, setStatus] = useState<BrokerStatus | null>(null);

  const [appKey, setAppKey] = useState("");
  const [appSecret, setAppSecret] = useState("");
  const [account, setAccount] = useState("");
  const [htsId, setHtsId] = useState("");
  const [virtual, setVirtual] = useState(false);

  const [clientId, setClientId] = useState("");
  const [clientSecret, setClientSecret] = useState("");
  const [tossAccount, setTossAccount] = useState("");

  const [nvidiaKey, setNvidiaKey] = useState("");

  const [error, setError] = useState("");
  const [canForce, setCanForce] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    api
      .brokerStatus()
      .then((r) => {
        if (!alive) return;
        setStatus(r as BrokerStatus);
        if (typeof (r as BrokerStatus).kis_virtual === "boolean") {
          setVirtual(Boolean((r as BrokerStatus).kis_virtual));
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  const savedReady = !!(status?.kis_configured || status?.toss_configured);

  function buildBody(): Record<string, unknown> | null {
    const body: Record<string, unknown> = { active: broker };
    let touched = false;

    if (broker === "kis") {
      const kis: Record<string, unknown> = { virtual };
      const cleanKey = appKey.trim().replace(/\s+/g, "");
      const cleanSecret = appSecret.trim().replace(/\s+/g, "");
      if (cleanKey) kis.app_key = cleanKey;
      if (cleanSecret) kis.app_secret = cleanSecret;
      if (account.trim()) kis.account = account.trim();
      if (htsId.trim()) kis.hts_id = htsId.trim();
      const acct = String(kis.account || "").replace(/\s+/g, "");
      if (acct.startsWith("@")) {
        setError("계좌번호 칸에 HTS ID가 들어가 있습니다. 계좌는 12345678-01 형식입니다.");
        return null;
      }
      if (acct && !/^\d{8}(-?\d{2})?$/.test(acct)) {
        setError("계좌번호는 12345678 또는 12345678-01 형식이어야 합니다.");
        return null;
      }

      const required = ["app_key", "app_secret", "account"] as const;
      const missing = required.filter((k) => !(k in kis));
      if (missing.length < required.length && missing.length > 0) {
        setError(
          `한국투자증권: App Key·App Secret·계좌번호를 모두 입력하세요. (누락: ${missing.length}개)`
        );
        return null;
      }
      if (cleanKey && cleanKey.length !== 36) {
        setError(
          `한투 App Key는 36자여야 합니다. 지금 ${cleanKey.length}자입니다. App Secret(180자)과 칸이 바뀌지 않았는지 확인하세요.`
        );
        return null;
      }
      if (cleanSecret && cleanSecret.length !== 180) {
        setError(
          `한투 App Secret은 180자여야 합니다. 지금 ${cleanSecret.length}자입니다.`
        );
        return null;
      }
      if (required.some((k) => k in kis)) touched = true;
      body.kis = kis;
    } else {
      const toss: Record<string, unknown> = {};
      if (clientId.trim()) toss.client_id = clientId.trim();
      if (clientSecret.trim()) toss.client_secret = clientSecret.trim();
      if (tossAccount.trim()) toss.account = tossAccount.trim();

      const hasId = "client_id" in toss;
      const hasSecret = "client_secret" in toss;
      if (hasId !== hasSecret) {
        setError("토스증권: Client ID와 Client Secret을 모두 입력하세요.");
        return null;
      }
      if (Object.keys(toss).length > 0) touched = true;
      body.toss = toss;
    }

    if (nvidiaKey.trim()) {
      body.nvidia = { api_key: nvidiaKey.trim() };
      touched = true;
    }

    return touched ? body : {};
  }

  async function connect(body?: Record<string, unknown>) {
    setBusy(true);
    setError("");
    setCanForce(false);
    try {
      if (body && Object.keys(body).length > 0) {
        persistBrokerCreds(body);
      }
      const payload = { ...brokerKeysPayload(broker), ...(body || {}), active: broker };
      await api.saveBrokerKeys(payload);
      const h = await api.health();
      if (h.broker_connected) {
        const next: BrokerId =
          h.active_broker === "toss" ? "toss" : h.active_broker === "kis" ? "kis" : broker;
        onLogin({ broker: next });
        return;
      }
      setError(
        h.broker_hint ||
          "브로커 연결에 실패했습니다. 키·계좌번호·증권사 IP 허용 설정을 확인하세요."
      );
      setCanForce(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "연결 요청에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    const body = buildBody();
    if (body === null) return;
    if (Object.keys(body).length === 0) {
      if (!savedReady) {
        setError("연결에 사용할 API 키를 입력하거나, 저장된 키로 계속하세요.");
        return;
      }
      void connect();
      return;
    }
    void connect(body);
  }

  return (
    <div className="login-shell">
      <div className="login-card">
        <p className="login-kicker">개인 투자 데스크</p>
        <h1 className="login-title">
          트레이딩<span>데스크</span>
        </h1>
        <p className="login-sub">
          증권사·AI API 키를 입력해 연결합니다. 입력값은 이 탭이 열려 있는 동안 계좌 조회에
          사용되며, 로그아웃하거나 탭을 닫으면 지워집니다.
        </p>

        <div className="login-source-row" aria-label="현재 연결 상태">
          <span className={`login-chip ${status?.kis_configured ? "ok" : ""}`}>
            한투 · {SOURCE_LABEL[status?.kis_source ?? "none"]}
          </span>
          <span className={`login-chip ${status?.toss_configured ? "ok" : ""}`}>
            토스 · {SOURCE_LABEL[status?.toss_source ?? "none"]}
          </span>
          <span className={`login-chip ${status?.nvidia_configured ? "ok" : ""}`}>
            AI · {SOURCE_LABEL[status?.nvidia_source ?? "none"]}
          </span>
        </div>

        <div className="login-broker-tabs" role="tablist" aria-label="증권사 선택">
          <button
            type="button"
            role="tab"
            aria-selected={broker === "kis"}
            className={`tab-btn ${broker === "kis" ? "active" : ""}`}
            onClick={() => setBroker("kis")}
          >
            한국투자증권
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={broker === "toss"}
            className={`tab-btn ${broker === "toss" ? "active" : ""}`}
            onClick={() => setBroker("toss")}
          >
            토스증권
          </button>
        </div>

        <form className="login-form" onSubmit={submit}>
          {broker === "kis" ? (
            <>
              <div className="field">
                <label htmlFor="kis-appkey">App Key</label>
                <input
                  id="kis-appkey"
                  autoComplete="off"
                  value={appKey}
                  onChange={(e) => setAppKey(e.target.value)}
                  placeholder="발급받은 App Key"
                />
              </div>
              <SecretInput
                id="kis-appsecret"
                label="App Secret"
                value={appSecret}
                onChange={setAppSecret}
                placeholder="발급받은 App Secret"
              />
              <div className="field">
                <label htmlFor="kis-account">계좌번호</label>
                <input
                  id="kis-account"
                  autoComplete="off"
                  value={account}
                  onChange={(e) => setAccount(e.target.value)}
                  placeholder="예: 12345678-01"
                />
              </div>
              <div className="field">
                <label htmlFor="kis-htsid">HTS ID (선택)</label>
                <input
                  id="kis-htsid"
                  autoComplete="off"
                  value={htsId}
                  onChange={(e) => setHtsId(e.target.value)}
                  placeholder="HTS 로그인 아이디"
                />
              </div>
              <div className="login-switch-row">
                <span className="login-switch-label">투자 모드</span>
                <button
                  type="button"
                  role="switch"
                  aria-checked={virtual}
                  className={`btn ${virtual ? "secondary" : "danger"}`}
                  onClick={() => setVirtual((v) => !v)}
                >
                  {virtual ? "모의투자" : "실전투자"}
                </button>
              </div>
              {!virtual && (
                <p className="login-warn">실전 모드입니다. 실제 주문이 체결될 수 있습니다.</p>
              )}
            </>
          ) : (
            <>
              <div className="field">
                <label htmlFor="toss-clientid">Client ID</label>
                <input
                  id="toss-clientid"
                  autoComplete="off"
                  value={clientId}
                  onChange={(e) => setClientId(e.target.value)}
                  placeholder="토스 Open API Client ID"
                />
              </div>
              <SecretInput
                id="toss-clientsecret"
                label="Client Secret"
                value={clientSecret}
                onChange={setClientSecret}
                placeholder="토스 Open API Client Secret"
              />
              <div className="field">
                <label htmlFor="toss-account">계좌번호 (선택)</label>
                <input
                  id="toss-account"
                  autoComplete="off"
                  value={tossAccount}
                  onChange={(e) => setTossAccount(e.target.value)}
                  placeholder="계좌번호"
                />
              </div>
            </>
          )}

          <details className="login-ai">
            <summary>AI 연결 (선택) — NVIDIA API Key</summary>
            <SecretInput
              id="nvidia-key"
              label="NVIDIA API Key"
              value={nvidiaKey}
              onChange={setNvidiaKey}
              placeholder="nvapi-..."
            />
          </details>

          {error && <div className="error-box">{error}</div>}

          <button className="btn btn-block" type="submit" disabled={busy}>
            {busy ? "연결 중..." : "연결하기"}
          </button>
          {savedReady && (
            <button
              className="btn secondary btn-block"
              type="button"
              disabled={busy}
              onClick={() => void connect()}
            >
              저장된 키로 계속
            </button>
          )}
          {canForce && (
            <button
              className="btn ghost btn-block"
              type="button"
              disabled={busy}
              onClick={() => onLogin({ broker })}
            >
              그래도 계속 (연결 없이 진입)
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
