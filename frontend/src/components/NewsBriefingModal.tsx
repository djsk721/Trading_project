import { useMemo } from "react";
import { NewsSummary } from "../api";
import { formatNewsTime } from "../formatTime";

type BriefingSource = {
  title: string;
  title_original?: string;
  url?: string;
  source?: string;
  published_at?: string | null;
  importance?: number | null;
  category_label?: string;
};

type Props = {
  open: boolean;
  onClose: () => void;
  loading: boolean;
  summary: NewsSummary | null;
  active: BriefingSource | null;
  kicker?: string;
};

type Briefing = {
  lead: string;
  body: string[];
  note: string;
};

function cleanBriefText(s: string) {
  return (s || "")
    .replace(/^\s*[-*•]\s+/gm, "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/`+/g, "")
    .trim();
}

/** LEAD/BODY/NOTE 또는 구형 1/2/3 요약을 브리핑 구조로 파싱 */
export function parseBriefing(raw: string): Briefing {
  const text = cleanBriefText(raw);
  if (!text) return { lead: "", body: [], note: "" };

  const labeled: { key: "lead" | "body" | "note"; re: RegExp }[] = [
    { key: "lead", re: /(?:^|\n)\s*(?:LEAD|한\s*줄\s*핵심|1[).]|①)\s*[:：]?\s*/i },
    {
      key: "body",
      re: /(?:^|\n)\s*(?:BODY|배경\s*[/／]?\s*내용|배경|2[).]|②)\s*[:：]?\s*/i,
    },
    {
      key: "note",
      re: /(?:^|\n)\s*(?:NOTE|시장\s*시사점|시사점|3[).]|③)\s*[:：]?\s*/i,
    },
  ];

  const hits: { key: "lead" | "body" | "note"; start: number; contentAt: number }[] = [];
  for (const p of labeled) {
    const m = p.re.exec(text);
    if (m && typeof m.index === "number") {
      hits.push({ key: p.key, start: m.index, contentAt: m.index + m[0].length });
    }
  }

  if (hits.length >= 2) {
    hits.sort((a, b) => a.start - b.start);
    const bag: Record<string, string> = {};
    hits.forEach((h, i) => {
      const end = i + 1 < hits.length ? hits[i + 1].start : text.length;
      bag[h.key] = text.slice(h.contentAt, end).trim();
    });
    const bodyParas = (bag.body || "")
      .split(/\n{2,}|\n/)
      .map((p) => p.trim())
      .filter(Boolean);
    return {
      lead: bag.lead || "",
      body: bodyParas,
      note: bag.note || "",
    };
  }

  const paras = text.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);
  if (paras.length === 1) {
    const sentences = paras[0].split(/(?<=[.!?。])\s+/).filter(Boolean);
    if (sentences.length >= 3) {
      return {
        lead: sentences[0],
        body: [sentences.slice(1, -1).join(" ")],
        note: sentences[sentences.length - 1],
      };
    }
    return { lead: paras[0], body: [], note: "" };
  }
  return {
    lead: paras[0] || "",
    body: paras.slice(1, -1).length ? paras.slice(1, -1) : paras.slice(1, 2),
    note: paras.length > 2 ? paras[paras.length - 1] : "",
  };
}

export default function NewsBriefingModal({
  open,
  onClose,
  loading,
  summary,
  active,
  kicker = "뉴스 브리핑",
}: Props) {
  const briefing = useMemo(
    () => (summary ? parseBriefing(summary.summary_ko) : null),
    [summary]
  );

  if (!open) return null;

  return (
    <div className="modal-backdrop summary-layer" role="presentation" onClick={onClose}>
      <div
        className="summary-modal"
        role="dialog"
        aria-modal="true"
        aria-label={kicker}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="summary-modal-head">
          <div className="summary-byline">
            <span className="summary-kicker">{kicker}</span>
            <span className="summary-byline-sep">·</span>
            <span>
              {[
                active?.category_label,
                active?.source,
                formatNewsTime(active?.published_at, { withYear: true }),
                summary?.importance
                  ? `중요도 ${summary.importance}`
                  : active?.importance
                    ? `중요도 ${active.importance}`
                    : "",
              ]
                .filter(Boolean)
                .join(" · ")}
            </span>
          </div>
          <button type="button" className="btn ghost" onClick={onClose}>
            닫기
          </button>
        </div>

        <h3 className="summary-title">{active?.title || summary?.title}</h3>

        {loading && (
          <div className="summary-loading">
            <span className="spinner" />
            <div>
              <strong>브리핑 정리 중</strong>
              <p className="muted">기사 핵심을 읽고 짧게 다듬고 있습니다.</p>
            </div>
          </div>
        )}

        {!loading && summary && briefing && (
          <article className="briefing-article">
            {briefing.lead && <p className="briefing-lead">{briefing.lead}</p>}
            {briefing.body.length > 0 && (
              <div className="briefing-body">
                {briefing.body.map((p, i) => (
                  <p key={i}>{p}</p>
                ))}
              </div>
            )}
            {!briefing.lead && !briefing.body.length && (
              <div className="briefing-body">
                <p>{summary.summary_ko}</p>
              </div>
            )}
            {briefing.note && (
              <aside className="briefing-note">
                <span className="briefing-note-label">시장에서 볼 포인트</span>
                <p>{briefing.note}</p>
              </aside>
            )}
          </article>
        )}

        <div className="summary-footer">
          {active?.url && (
            <a
              className="btn secondary"
              href={active.url}
              target="_blank"
              rel="noopener noreferrer"
            >
              원문 보기
            </a>
          )}
          <button type="button" className="btn" onClick={onClose}>
            닫기
          </button>
        </div>
      </div>
    </div>
  );
}
