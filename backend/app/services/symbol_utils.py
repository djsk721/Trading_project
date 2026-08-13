"""종목코드/티커 정규화·검증 (잘못된 입력으로 KIS/pykrx 폭주 방지)."""
from __future__ import annotations

import re
from typing import Optional

_NOISE = {
    "HTTP",
    "HTTPS",
    "WWW",
    "URL",
    "API",
    "CEO",
    "ETF",
    "ADR",
    "USD",
    "KRW",
    "KRX",
    "NYSE",
    "NASDAQ",
    "REAL",
    "TIME",
    "LIVE",
}


def normalize_symbol(raw: str, market: str = "") -> str:
    """입력에서 실제 티커만 추출. 불가하면 빈 문자열."""
    s = (raw or "").strip()
    if not s:
        return ""

    # 국내 6자리 (문자열에 포함돼 있으면 최우선)
    m = re.search(r"(?<!\d)(\d{6})(?!\d)", s)
    if m:
        return m.group(1)

    # 순수 티커
    if re.fullmatch(r"[A-Za-z]{1,5}", s):
        return s.upper()
    if re.fullmatch(r"\d{6}", s):
        return s
    if re.fullmatch(r"[A-Za-z]{1,4}\.[A-Za-z]", s):
        return s.upper()

    # 긴 문자열/한글 이름+티커 혼입: ASCII 티커 후보 추출
    candidates = [
        t.upper()
        for t in re.findall(r"\b([A-Za-z]{1,5})\b", s)
        if t.upper() not in _NOISE
    ]
    if candidates:
        # BMNR 같이 의미 있는 길이 우선
        candidates.sort(key=lambda x: (len(x), x), reverse=True)
        return candidates[0]

    # 허용 가능한 짧은 코드만 통과
    compact = re.sub(r"\s+", "", s)
    if re.fullmatch(r"[A-Za-z0-9.\-]{1,12}", compact) and not re.search(r"[가-힣]", compact):
        return compact.upper()
    return ""


def is_plausible_symbol(symbol: str, market: str = "") -> bool:
    s = (symbol or "").strip().upper()
    if not s:
        return False
    if re.fullmatch(r"\d{6}", s):
        return True
    if re.fullmatch(r"[A-Z]{1,5}", s):
        return True
    if re.fullmatch(r"[A-Z]{1,4}\.[A-Z]", s):
        return True
    if market.upper() == "US" and re.fullmatch(r"[A-Z0-9.\-]{1,12}", s):
        return True
    return False


def looks_like_kr_ticker(symbol: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", (symbol or "").strip()))
