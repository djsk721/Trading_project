/** 로그인에서 입력한 증권사 키. 탭을 닫으면 지워지고, 새로고침·백엔드 재시작 후에는 다시 서버로 보냅니다. */

export type KisCreds = {
  hts_id?: string;
  app_key?: string;
  app_secret?: string;
  account?: string;
  virtual?: boolean;
};

export type TossCreds = {
  client_id?: string;
  client_secret?: string;
  account?: string;
};

export type StoredBrokerCreds = {
  kis?: KisCreds;
  toss?: TossCreds;
  nvidia?: { api_key?: string };
};

const CREDS_KEY = "td_broker_creds";

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}

function pickFilled(src: Record<string, unknown>, keys: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const key of keys) {
    const val = String(src[key] ?? "").trim();
    if (val) out[key] = val;
  }
  return out;
}

export function loadBrokerCreds(): StoredBrokerCreds | null {
  try {
    const raw = sessionStorage.getItem(CREDS_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed as StoredBrokerCreds;
  } catch {
    return null;
  }
}

export function persistBrokerCreds(body: Record<string, unknown>): StoredBrokerCreds {
  const prev = loadBrokerCreds() || {};
  const next: StoredBrokerCreds = { ...prev };

  const kis = asRecord(body.kis);
  if (kis?.clear) {
    delete next.kis;
  } else if (kis) {
    const filled = pickFilled(kis, ["hts_id", "app_key", "app_secret", "account"]);
    const merged: KisCreds = { ...(prev.kis || {}), ...filled };
    if (typeof kis.virtual === "boolean") merged.virtual = kis.virtual;
    if (Object.keys(merged).length) next.kis = merged;
  }

  const toss = asRecord(body.toss);
  if (toss?.clear) {
    delete next.toss;
  } else if (toss) {
    const filled = pickFilled(toss, ["client_id", "client_secret", "account"]);
    const merged: TossCreds = { ...(prev.toss || {}), ...filled };
    if (Object.keys(merged).length) next.toss = merged;
  }

  const nvidia = asRecord(body.nvidia);
  if (nvidia?.clear) {
    delete next.nvidia;
  } else if (nvidia) {
    const apiKey = String(nvidia.api_key ?? "").trim();
    if (apiKey) next.nvidia = { api_key: apiKey };
  }

  sessionStorage.setItem(CREDS_KEY, JSON.stringify(next));
  return next;
}

export function clearBrokerCreds(): void {
  sessionStorage.removeItem(CREDS_KEY);
}

/** 서버 메모리 키를 로그인 입력값으로 다시 채울 때 사용합니다. */
export function brokerKeysPayload(active?: string): Record<string, unknown> {
  const creds = loadBrokerCreds();
  const body: Record<string, unknown> = {};
  if (active) body.active = active;
  if (creds?.kis && (creds.kis.app_key || creds.kis.app_secret || creds.kis.account)) {
    body.kis = creds.kis;
  }
  if (creds?.toss && (creds.toss.client_id || creds.toss.client_secret)) {
    body.toss = creds.toss;
  }
  if (creds?.nvidia?.api_key) {
    body.nvidia = creds.nvidia;
  }
  return body;
}
