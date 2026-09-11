import { Fragment, ReactNode } from "react";

type Props = {
  text: string;
  className?: string;
  placeholder?: string;
};

function unwrapFence(raw: string): string {
  const t = (raw || "").trim();
  const m = t.match(/^```(?:markdown|md|text)?\s*\n([\s\S]*?)\n```$/i);
  return m ? m[1].trim() : t;
}

function renderInline(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const re =
    /(\*\*([^*]+)\*\*|\*([^*\n]+)\*|`([^`]+)`|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[2]) nodes.push(<strong key={`b${i}`}>{m[2]}</strong>);
    else if (m[3]) nodes.push(<em key={`e${i}`}>{m[3]}</em>);
    else if (m[4]) nodes.push(<code key={`c${i}`}>{m[4]}</code>);
    else if (m[5] && m[6]) {
      nodes.push(
        <a key={`a${i}`} href={m[6]} target="_blank" rel="noopener noreferrer">
          {m[5]}
        </a>
      );
    }
    last = m.index + m[0].length;
    i += 1;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function headingLevel(line: string): number {
  const m = line.match(/^(#{1,3})\s+(.+)$/);
  return m ? m[1].length : 0;
}

export default function MarkdownBody({ text, className = "", placeholder = "" }: Props) {
  const src = unwrapFence(text);
  if (!src.trim()) {
    return <p className={`md-body ${className}`.trim()}>{placeholder}</p>;
  }

  const lines = src.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      blocks.push(<hr key={`hr${key++}`} />);
      i += 1;
      continue;
    }

    const h = headingLevel(trimmed);
    if (h) {
      const title = trimmed.replace(/^#{1,3}\s+/, "").replace(/\*+/g, "").trim();
      const Tag = (h === 1 ? "h3" : h === 2 ? "h4" : "h5") as "h3" | "h4" | "h5";
      blocks.push(<Tag key={`h${key++}`}>{title}</Tag>);
      i += 1;
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ol key={`ol${key++}`}>
          {items.map((it, idx) => (
            <li key={idx}>{renderInline(it)}</li>
          ))}
        </ol>
      );
      continue;
    }

    if (/^[-*+]\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*+]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*+]\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ul key={`ul${key++}`}>
          {items.map((it, idx) => (
            <li key={idx}>{renderInline(it)}</li>
          ))}
        </ul>
      );
      continue;
    }

    if (trimmed.startsWith("> ")) {
      const quote: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        quote.push(lines[i].trim().replace(/^>\s?/, ""));
        i += 1;
      }
      blocks.push(
        <blockquote key={`q${key++}`}>{renderInline(quote.join(" "))}</blockquote>
      );
      continue;
    }

    const para: string[] = [];
    while (i < lines.length) {
      const t = lines[i].trim();
      if (
        !t ||
        headingLevel(t) ||
        /^[-*+]\s+/.test(t) ||
        /^\d+\.\s+/.test(t) ||
        t.startsWith("> ") ||
        /^(-{3,}|\*{3,}|_{3,})$/.test(t)
      ) {
        break;
      }
      para.push(t);
      i += 1;
    }
    blocks.push(
      <p key={`p${key++}`}>
        {para.map((part, idx) => (
          <Fragment key={idx}>
            {idx > 0 ? " " : null}
            {renderInline(part)}
          </Fragment>
        ))}
      </p>
    );
  }

  return <div className={`md-body ${className}`.trim()}>{blocks}</div>;
}
