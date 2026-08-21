import { REFERENCE_PORTFOLIOS } from "../referencePortfolios";

type Props = {
  onOpenSymbol: (symbol: string, market: string) => void;
};

export default function ReferencePortfolios({ onOpenSymbol }: Props) {
  return (
    <div className="account-page">
      <div className="account-hero">
        <div>
          <p className="account-kicker">참고 포트폴리오</p>
          <h2>유명 투자자 · 기관 공시</h2>
          <p className="muted">
            따라 사기가 아닌 참고용입니다. 공시 지연이 있고, 비중은 대략값입니다. 투자 권유가 아닙니다.
          </p>
        </div>
      </div>
      <div className="ref-grid">
        {REFERENCE_PORTFOLIOS.map((p) => (
          <article className="card-item" key={p.id}>
            <h4>
              {p.person}
              <span className="chip" style={{ marginLeft: 8 }}>
                {p.vehicle}
              </span>
            </h4>
            <p className="muted">
              기준 {p.asOf} · {p.delayNote}
            </p>
            <div className="scan-table-wrap" style={{ marginTop: 10 }}>
              <table className="scan-table">
                <thead>
                  <tr>
                    <th>종목</th>
                    <th>시장</th>
                    <th>참고 비중</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {p.holdings.map((h) => (
                    <tr key={`${p.id}-${h.symbol}`}>
                      <td>
                        <strong>{h.name}</strong>
                        <div className="muted">{h.symbol}</div>
                      </td>
                      <td>{h.market === "KRX" ? "국내" : "해외"}</td>
                      <td>{h.weight.toFixed(1)}%</td>
                      <td>
                        <button
                          className="btn secondary"
                          onClick={() => onOpenSymbol(h.symbol, h.market)}
                        >
                          차트
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
