import { useEffect, useState } from "react";
import { api, MacroSnapshot } from "../api";

type Props = {
  refreshMs?: number;
};

export default function MacroBoard({ refreshMs = 60_000 }: Props) {
  const [macros, setMacros] = useState<MacroSnapshot | null>(null);

  useEffect(() => {
    let alive = true;
    async function load(force = false) {
      try {
        const m = await api.macros(force);
        if (alive && m?.items?.length) setMacros(m);
      } catch {
        /* ignore */
      }
    }
    load(false);
    const t = window.setInterval(() => load(false), refreshMs);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, [refreshMs]);

  return (
    <section className="macro-board macro-board-main" aria-label="실시간 매크로 지표">
      <div className="macro-board-head">
        <div className="macro-board-title">
          <span className="macro-live-dot" aria-hidden />
          <span className="macro-board-kicker">LIVE 지표</span>
        </div>
        <span className="macro-board-meta">
          {macros?.as_of
            ? `갱신 ${macros.as_of.slice(0, 16).replace("T", " ")} UTC`
            : "불러오는 중…"}
          {macros?.source ? ` · ${macros.source}` : ""}
        </span>
      </div>
      <div className="macro-board-grid">
        {(macros?.items || []).map((m) => {
          const pct = m.change_pct;
          const dir =
            pct == null || !m.ok ? "flat" : pct > 0 ? "up" : pct < 0 ? "down" : "flat";
          return (
            <div key={m.id} className={`macro-cell ${dir}`}>
              <div className="macro-label">{m.label}</div>
              <div className="macro-price">{m.ok ? m.price_text : "—"}</div>
              <span className={`macro-chg ${dir}`}>{m.ok ? m.change_text : "n/a"}</span>
            </div>
          );
        })}
        {!macros?.items?.length && (
          <p className="muted" style={{ margin: 0, gridColumn: "1 / -1" }}>
            지표를 불러오는 중…
          </p>
        )}
      </div>
    </section>
  );
}
