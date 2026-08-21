export type RefHolding = {
  symbol: string;
  name: string;
  market: "KRX" | "US";
  weight: number;
};

export type RefPortfolio = {
  id: string;
  person: string;
  vehicle: string;
  asOf: string;
  delayNote: string;
  holdings: RefHolding[];
};

/** 공개 공시 기반 참고 구성. 실시간 보유가 아니며 비중은 대략값입니다. */
export const REFERENCE_PORTFOLIOS: RefPortfolio[] = [
  {
    id: "buffett",
    person: "워런 버핏",
    vehicle: "버크셔 해서웨이",
    asOf: "2025-12-31",
    delayNote: "미국 13F 공시 기준. 실제 보유와 최대 45일 이상 차이가 날 수 있습니다.",
    holdings: [
      { symbol: "AAPL", name: "애플", market: "US", weight: 22.1 },
      { symbol: "AXP", name: "아메리칸 익스프레스", market: "US", weight: 18.4 },
      { symbol: "BAC", name: "뱅크오브아메리카", market: "US", weight: 11.2 },
      { symbol: "KO", name: "코카콜라", market: "US", weight: 10.8 },
      { symbol: "CVX", name: "셰브론", market: "US", weight: 6.5 },
      { symbol: "MCO", name: "무디스", market: "US", weight: 4.1 },
    ],
  },
  {
    id: "wood",
    person: "캐시 우드",
    vehicle: "ARK 이노베이션",
    asOf: "2026-06-30",
    delayNote: "펀드 공시·보도 기준 참고 구성입니다. 일간 리밸런싱과 다를 수 있습니다.",
    holdings: [
      { symbol: "TSLA", name: "테슬라", market: "US", weight: 10.5 },
      { symbol: "COIN", name: "코인베이스", market: "US", weight: 8.2 },
      { symbol: "ROKU", name: "로쿠", market: "US", weight: 7.1 },
      { symbol: "CRSP", name: "크리스퍼", market: "US", weight: 5.4 },
      { symbol: "HOOD", name: "로빈후드", market: "US", weight: 4.8 },
    ],
  },
  {
    id: "nps",
    person: "국민연금",
    vehicle: "국내 주식 주요 보유",
    asOf: "2025-12-31",
    delayNote: "대량보유 공시 기준 참고입니다. 연기금 전체 포트폴리오가 아닙니다.",
    holdings: [
      { symbol: "005930", name: "삼성전자", market: "KRX", weight: 19.8 },
      { symbol: "000660", name: "SK하이닉스", market: "KRX", weight: 8.6 },
      { symbol: "005380", name: "현대차", market: "KRX", weight: 3.4 },
      { symbol: "035420", name: "네이버", market: "KRX", weight: 2.8 },
      { symbol: "051910", name: "LG화학", market: "KRX", weight: 2.1 },
      { symbol: "068270", name: "셀트리온", market: "KRX", weight: 1.9 },
    ],
  },
];
