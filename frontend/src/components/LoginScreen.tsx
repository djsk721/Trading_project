import { FormEvent, useState } from "react";

export type BrokerId = "kis" | "toss";

type Props = {
  onLogin: (payload: { user: string; broker: BrokerId }) => void;
};

const DEMO_ID = "test";
const DEMO_PW = "test";

export default function LoginScreen({ onLogin }: Props) {
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [broker, setBroker] = useState<BrokerId>("kis");
  const [error, setError] = useState("");

  function submit(e: FormEvent) {
    e.preventDefault();
    if (userId.trim() === DEMO_ID && password === DEMO_PW) {
      onLogin({ user: DEMO_ID, broker });
      return;
    }
    setError("아이디 또는 비밀번호가 올바르지 않습니다. (데모: test / test)");
  }

  return (
    <div className="login-shell">
      <div className="login-card">
        <p className="login-kicker">개인 투자 데스크</p>
        <h1 className="login-title">
          트레이딩<span>데스크</span>
        </h1>
        <p className="login-sub">증권사 연동 후 차트·추천·계좌를 한 화면에서 봅니다.</p>

        <form className="login-form" onSubmit={submit}>
          <div className="field">
            <label htmlFor="login-id">아이디</label>
            <input
              id="login-id"
              autoComplete="username"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="test"
            />
          </div>
          <div className="field">
            <label htmlFor="login-pw">비밀번호</label>
            <input
              id="login-pw"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="test"
            />
          </div>
          <div className="field">
            <label htmlFor="login-broker">투자 증권사</label>
            <select
              id="login-broker"
              value={broker}
              onChange={(e) => setBroker(e.target.value as BrokerId)}
            >
              <option value="kis">한국투자증권</option>
              <option value="toss">토스증권</option>
            </select>
          </div>
          <p className="muted login-hint">
            선택한 증권사의 시세·호가·계좌·주문이 동일하게 동작합니다. 로그인 후 내 계좌에서 API 키를
            저장하세요.
          </p>
          {error && <div className="error-box">{error}</div>}
          <button className="btn btn-block" type="submit">
            로그인
          </button>
        </form>
      </div>
    </div>
  );
}
