import { newsToneClass, newsToneKo } from "../labels";

export default function NewsToneChip({ tone }: { tone?: string | null }) {
  const label = newsToneKo(tone);
  const cls = newsToneClass(tone);
  if (!label) return null;
  return (
    <span className={`chip ${cls}`} title="기사 톤 참고 · 투자 조언 아님">
      {label}
    </span>
  );
}
