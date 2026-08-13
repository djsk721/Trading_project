/** ISO 시각을 KST 날짜·시분으로 표시 (예: 08-13 15:42) */
export function formatNewsTime(iso?: string | null, opts?: { withYear?: boolean }): string {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) {
    // 파싱 실패 시 원문에서 날짜·시간만 최대한 추출
    const m = String(iso).match(/(\d{4}-)?(\d{2}-\d{2})[T\s](\d{2}:\d{2})/);
    if (m) return opts?.withYear && m[1] ? `${m[1]}${m[2]} ${m[3]}` : `${m[2]} ${m[3]}`;
    return String(iso).slice(0, 16).replace("T", " ");
  }
  const parts = new Intl.DateTimeFormat("sv-SE", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(d);
  const get = (type: string) => parts.find((p) => p.type === type)?.value || "";
  const y = get("year");
  const mo = get("month");
  const day = get("day");
  const h = get("hour");
  const mi = get("minute");
  if (opts?.withYear) return `${y}-${mo}-${day} ${h}:${mi}`;
  return `${mo}-${day} ${h}:${mi}`;
}
