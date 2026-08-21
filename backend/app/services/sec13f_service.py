"""SEC Form 13F dataset downloader, parser, and Parquet cache access."""
from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import pandas as pd

log = logging.getLogger(__name__)

SEC_13F_PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
USER_AGENT = "TradingProject/2.0 contact@example.com"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
CACHE_DIR = DATA_ROOT / "cache" / "13f"
TEMP_DIR = DATA_ROOT / "temp" / "13f"
META_FILE = CACHE_DIR / "metadata.json"


@dataclass(frozen=True)
class SecDataset:
    dataset: str
    title: str
    url: str
    filename: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _norm_col(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _first_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {_norm_col(c): c for c in df.columns}
    for alias in aliases:
        found = lookup.get(_norm_col(alias))
        if found:
            return found
    return None


def _series_or_default(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _read_table(path: Path) -> pd.DataFrame:
    for sep in ("\t", ",", "|"):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    raise ValueError(f"Unable to parse SEC table: {path.name}")


def _find_file(extract_dir: Path, keywords: list[str]) -> Path | None:
    candidates = [p for p in extract_dir.rglob("*") if p.is_file()]
    for p in candidates:
        stem = _norm_col(p.stem)
        if all(k in stem for k in keywords):
            return p
    return None


def _dataset_id(title: str, href: str) -> str:
    text = f"{title} {href}"
    q = re.search(r"(20\d{2})\s*Q([1-4])", text, re.I)
    if q:
        return f"{q.group(1)}Q{q.group(2)}"
    year_match = re.search(r"(20\d{2})", text)
    months = re.findall(
        r"january|february|march|april|may|june|july|august|september|october|november|december",
        text,
        re.I,
    )
    if year_match and months:
        year = int(year_match.group(1))
        end_month = months[-1].lower()
        month_to_q = {
            "february": "Q1",
            "may": "Q2",
            "august": "Q3",
            "november": "Q4",
        }
        qtr = month_to_q.get(end_month, "Q4")
        return f"{year}{qtr}"
    return Path(href).stem.upper()


def read_metadata() -> dict[str, Any]:
    if not META_FILE.exists():
        return {"status": "empty"}
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def discover_latest_dataset() -> SecDataset:
    with httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
        res = client.get(SEC_13F_PAGE)
        res.raise_for_status()
    links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', res.text, flags=re.I | re.S)
    for href, label in links:
        clean_label = re.sub(r"<[^>]+>", "", label).strip()
        if "13f" not in clean_label.lower() or ".zip" not in href.lower():
            continue
        url = urljoin(SEC_13F_PAGE, href)
        filename = Path(url.split("?")[0]).name or f"{_dataset_id(clean_label, url)}.zip"
        return SecDataset(_dataset_id(clean_label, url), clean_label, url, filename)
    raise RuntimeError("SEC 13F ZIP link not found")


def _download_zip(dataset: SecDataset, dest: Path) -> None:
    with httpx.stream("GET", dataset.url, timeout=120, headers={"User-Agent": USER_AGENT}) as res:
        res.raise_for_status()
        with dest.open("wb") as f:
            for chunk in res.iter_bytes():
                if chunk:
                    f.write(chunk)
    if dest.stat().st_size < 1024:
        raise RuntimeError("Downloaded SEC ZIP is unexpectedly small")


def _normalize_holdings(
    info: pd.DataFrame,
    submission: pd.DataFrame,
    dataset: str,
    coverpage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    acc_i = _first_col(info, ["ACCESSION_NUMBER", "ACCESSIONNUMBER", "ACCESSIONNO"])
    acc_s = _first_col(submission, ["ACCESSION_NUMBER", "ACCESSIONNUMBER", "ACCESSIONNO"])
    if not acc_i or not acc_s:
        raise ValueError("Required accession number column missing")

    manager_cols = {
        "accession": acc_s,
        "cik": _first_col(submission, ["CIK", "CIK_NUMBER", "FILINGMANAGER_CIK"]),
        "manager_name": _first_col(submission, ["FILINGMANAGER_NAME", "MANAGER_NAME", "NAME"]),
        "filing_date": _first_col(submission, ["FILING_DATE", "FILINGDATE", "SUBMISSIONTYPE"]),
        "report_period": _first_col(submission, ["PERIODOFREPORT", "REPORT_PERIOD", "REPORTCALENDARORQUARTER"]),
    }
    keep = [c for c in manager_cols.values() if c]
    sub = submission[keep].copy()
    sub.columns = [k for k, c in manager_cols.items() if c]
    if coverpage is not None and not coverpage.empty:
        acc_c = _first_col(coverpage, ["ACCESSION_NUMBER", "ACCESSIONNUMBER", "ACCESSIONNO"])
        name_c = _first_col(coverpage, ["FILINGMANAGER_NAME", "MANAGER_NAME", "NAME"])
        period_c = _first_col(coverpage, ["REPORTCALENDARORQUARTER", "PERIODOFREPORT", "REPORT_PERIOD"])
        cover_cols = [c for c in [acc_c, name_c, period_c] if c]
        if acc_c and name_c:
            cover = coverpage[cover_cols].copy()
            rename_cover = {acc_c: "accession", name_c: "manager_name_cover"}
            if period_c:
                rename_cover[period_c] = "report_period_cover"
            cover = cover.rename(columns=rename_cover)
            sub = sub.merge(cover.drop_duplicates("accession"), on="accession", how="left")
            sub["manager_name"] = _series_or_default(sub, "manager_name").replace("", pd.NA)
            sub["manager_name"] = sub["manager_name"].fillna(sub["manager_name_cover"])
            if "report_period_cover" in sub.columns:
                sub["report_period"] = _series_or_default(sub, "report_period").replace("", pd.NA)
                sub["report_period"] = sub["report_period"].fillna(sub["report_period_cover"])
            sub = sub.drop(columns=[c for c in ["manager_name_cover", "report_period_cover"] if c in sub.columns])

    value_col = _first_col(info, ["VALUE", "VALUE_AMT"])
    shares_col = _first_col(info, ["SSHPRNAMT", "SHARES", "SSH_PRNAMT"])
    name_col = _first_col(info, ["NAMEOFISSUER", "ISSUER", "ISSUERNAME"])
    cusip_col = _first_col(info, ["CUSIP"])
    class_col = _first_col(info, ["TITLEOFCLASS", "SECURITY_CLASS", "CLASS"])
    put_call_col = _first_col(info, ["PUTCALL", "PUT_CALL"])
    ticker_col = _first_col(info, ["TICKER", "SYMBOL"])
    required = [value_col, shares_col, name_col, cusip_col]
    if any(c is None for c in required):
        raise ValueError("Required INFOTABLE columns missing")

    out = info[[c for c in [acc_i, name_col, ticker_col, cusip_col, class_col, value_col, shares_col, put_call_col] if c]].copy()
    rename = {
        acc_i: "accession",
        name_col: "issuer",
        cusip_col: "cusip",
        value_col: "value",
        shares_col: "shares",
    }
    if ticker_col:
        rename[ticker_col] = "ticker"
    if class_col:
        rename[class_col] = "security_class"
    if put_call_col:
        rename[put_call_col] = "put_call"
    out = out.rename(columns=rename)
    out = out.merge(sub.drop_duplicates("accession"), on="accession", how="left")
    out["ticker"] = _series_or_default(out, "ticker").fillna("").astype(str).str.upper().str.strip()
    out["issuer"] = out["issuer"].fillna("").astype(str).str.strip()
    out["cusip"] = out["cusip"].fillna("").astype(str).str.strip()
    out["security_class"] = _series_or_default(out, "security_class").fillna("").astype(str).str.strip()
    out["put_call"] = _series_or_default(out, "put_call").fillna("").astype(str).str.strip()
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0) * 1000.0
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce").fillna(0.0)
    out["cik"] = _series_or_default(out, "cik").fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    out["manager_name"] = _series_or_default(out, "manager_name", "Unknown Manager").fillna("Unknown Manager").astype(str).str.strip()
    out["report_period"] = _series_or_default(out, "report_period", dataset).fillna(dataset).astype(str)
    out["filing_date"] = _series_or_default(out, "filing_date").fillna("").astype(str)
    return out[
        [
            "cik",
            "manager_name",
            "filing_date",
            "report_period",
            "issuer",
            "ticker",
            "cusip",
            "security_class",
            "value",
            "shares",
            "put_call",
        ]
    ]


def _build_tables(extract_dir: Path, dataset: str, previous_holdings: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    info_path = _find_file(extract_dir, ["INFOTABLE"])
    sub_path = _find_file(extract_dir, ["SUBMISSION"])
    cover_path = _find_file(extract_dir, ["COVERPAGE"])
    if not info_path or not sub_path:
        raise FileNotFoundError("Expected INFOTABLE and SUBMISSION files were not found")
    cover_df = _read_table(cover_path) if cover_path else None
    holdings = _normalize_holdings(_read_table(info_path), _read_table(sub_path), dataset, cover_df)
    if holdings.empty:
        raise ValueError("SEC holdings table is empty")

    totals = holdings.groupby(["cik", "manager_name", "report_period"], dropna=False).agg(
        portfolio_value=("value", "sum"),
        holdings_count=("cusip", "nunique"),
        filing_date=("filing_date", "max"),
    ).reset_index()
    managers = totals[["cik", "manager_name", "filing_date", "report_period", "portfolio_value", "holdings_count"]]

    current = holdings.groupby(["cik", "manager_name", "report_period", "issuer", "ticker", "cusip"], dropna=False).agg(
        current_shares=("shares", "sum"),
        current_value=("value", "sum"),
    ).reset_index()
    if previous_holdings is not None and not previous_holdings.empty:
        prev = previous_holdings.groupby(["cik", "cusip"], dropna=False).agg(
            previous_shares=("shares", "sum"),
            previous_value=("value", "sum"),
        ).reset_index()
        tx = current.merge(prev, on=["cik", "cusip"], how="outer")
        for col in ["manager_name", "issuer", "ticker", "report_period"]:
            tx[col] = tx[col].fillna("")
        tx[["previous_shares", "previous_value", "current_shares", "current_value"]] = tx[
            ["previous_shares", "previous_value", "current_shares", "current_value"]
        ].fillna(0.0)
    else:
        tx = current.copy()
        tx["previous_shares"] = 0.0
        tx["previous_value"] = 0.0

    tx["share_change"] = tx["current_shares"] - tx["previous_shares"]
    tx["value_change"] = tx["current_value"] - tx["previous_value"]
    tx["change_type"] = "UNKNOWN"
    if previous_holdings is not None and not previous_holdings.empty:
        tx["change_type"] = "UNCHANGED"
        tx.loc[(tx["previous_shares"] <= 0) & (tx["current_shares"] > 0), "change_type"] = "NEW"
        tx.loc[(tx["previous_shares"] > 0) & (tx["current_shares"] <= 0), "change_type"] = "SOLD"
        tx.loc[(tx["previous_shares"] > 0) & (tx["current_shares"] > tx["previous_shares"]), "change_type"] = "INCREASED"
        tx.loc[(tx["previous_shares"] > 0) & (tx["current_shares"] < tx["previous_shares"]) & (tx["current_shares"] > 0), "change_type"] = "DECREASED"
    portfolio = current.groupby("cik")["current_value"].sum().rename("portfolio_total")
    tx = tx.merge(portfolio, on="cik", how="left")
    tx["portfolio_weight"] = (tx["current_value"] / tx["portfolio_total"].replace(0, pd.NA)).fillna(0.0)
    transactions = tx[
        [
            "cik",
            "manager_name",
            "report_period",
            "issuer",
            "ticker",
            "cusip",
            "previous_shares",
            "current_shares",
            "previous_value",
            "current_value",
            "change_type",
            "share_change",
            "value_change",
            "portfolio_weight",
        ]
    ]
    return {"managers": managers, "holdings": holdings, "transactions": transactions}


def _validate_cache(path: Path) -> None:
    required = ["managers.parquet", "holdings.parquet", "transactions.parquet", "metadata.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cache validation failed, missing: {missing}")
    for name in required[:3]:
        if pd.read_parquet(path / name).empty:
            raise ValueError(f"Cache validation failed, empty {name}")


def ensure_cache(force: bool = False) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    local_meta = read_metadata()
    try:
        latest = discover_latest_dataset()
    except Exception as exc:
        log.exception("SEC latest dataset discovery failed")
        if (CACHE_DIR / "holdings.parquet").exists():
            return {**local_meta, "update_status": "failed", "error": str(exc)}
        raise
    if not force and local_meta.get("dataset") == latest.dataset and local_meta.get("status") == "success":
        return {**local_meta, "update_status": "current"}

    work = Path(tempfile.mkdtemp(prefix=f"{latest.dataset.lower()}-", dir=TEMP_DIR))
    zip_path = work / latest.filename
    next_cache = work / "cache"
    previous_holdings = load_holdings() if (CACHE_DIR / "holdings.parquet").exists() else None
    try:
        _download_zip(latest, zip_path)
        extract_dir = work / "extract"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad:
                raise zipfile.BadZipFile(f"Corrupted ZIP member: {bad}")
            zf.extractall(extract_dir)
        tables = _build_tables(extract_dir, latest.dataset, previous_holdings)
        next_cache.mkdir()
        for key, df in tables.items():
            df.to_parquet(next_cache / f"{key}.parquet", index=False)
        meta = {
            "dataset": latest.dataset,
            "title": latest.title,
            "filename": latest.filename,
            "source_url": latest.url,
            "status": "success",
            "updated_at": _now_iso(),
            "record_count": int(len(tables["holdings"])),
            "manager_count": int(tables["managers"]["cik"].nunique()),
            "holding_count": int(tables["holdings"]["cusip"].nunique()),
        }
        (next_cache / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _validate_cache(next_cache)
        backup = CACHE_DIR.with_name("13f_old")
        if backup.exists():
            shutil.rmtree(backup)
        if CACHE_DIR.exists():
            CACHE_DIR.rename(backup)
        next_cache.rename(CACHE_DIR)
        if backup.exists():
            shutil.rmtree(backup)
        _clear_read_cache()
        if zip_path.exists():
            zip_path.unlink()
        return {**meta, "update_status": "updated"}
    except Exception as exc:
        log.exception("SEC 13F update failed")
        err = {**local_meta, "update_status": "failed", "error": str(exc), "checked_dataset": latest.dataset}
        if not (CACHE_DIR / "holdings.parquet").exists():
            raise
        return err
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _cache_key(name: str) -> tuple[str, float]:
    path = CACHE_DIR / f"{name}.parquet"
    return (str(path), path.stat().st_mtime if path.exists() else 0.0)


@lru_cache(maxsize=16)
def _read_cached(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    return pd.read_parquet(path)


def _clear_read_cache() -> None:
    _read_cached.cache_clear()


def _load(name: str) -> pd.DataFrame:
    path, mtime = _cache_key(name)
    if not Path(path).exists():
        return pd.DataFrame()
    return _read_cached(path, mtime).copy()


def load_managers() -> pd.DataFrame:
    return _load("managers")


def load_holdings() -> pd.DataFrame:
    return _load("holdings")


def load_transactions() -> pd.DataFrame:
    return _load("transactions")


def dashboard() -> dict[str, Any]:
    meta = read_metadata()
    managers = _latest_manager_rows(load_managers())
    holdings = load_holdings()
    tx = load_transactions()
    if managers.empty or holdings.empty or tx.empty:
        return {
            "metadata": meta,
            "manager_count": 0,
            "issuer_count": 0,
            "top_managers": [],
            "recent_new_holdings": [],
            "shared_buys": [],
        }
    top = managers.sort_values("portfolio_value", ascending=False).head(10)
    new = tx[tx["change_type"] == "NEW"].sort_values("current_value", ascending=False).head(10)
    shared = (
        tx[tx["change_type"].isin(["NEW", "INCREASED"])]
        .groupby(["ticker", "issuer"], dropna=False)
        .agg(manager_count=("cik", "nunique"), total_value=("current_value", "sum"))
        .reset_index()
        .sort_values(["manager_count", "total_value"], ascending=False)
        .head(10)
    )
    return {
        "metadata": meta,
        "manager_count": int(managers["cik"].nunique()) if not managers.empty else 0,
        "issuer_count": int(holdings["cusip"].nunique()) if not holdings.empty else 0,
        "top_managers": top.to_dict("records"),
        "recent_new_holdings": new.to_dict("records"),
        "shared_buys": shared.to_dict("records"),
    }


def search_managers(query: str = "", limit: int = 50) -> list[dict[str, Any]]:
    df = _latest_manager_rows(load_managers())
    if df.empty:
        return []
    if query:
        tokens = [t for t in re.split(r"\s+", query.lower().strip()) if t]
        name_norm = df["manager_name"].fillna("").astype(str).str.lower()
        cik_norm = df["cik"].astype(str).str.lower()
        compact_name = name_norm.str.replace(r"[^a-z0-9]", "", regex=True)
        compact_query = re.sub(r"[^a-z0-9]", "", query.lower())
        phrase = re.escape(query.lower().strip())
        mask = cik_norm.str.contains(phrase, na=False) | name_norm.str.contains(phrase, na=False)
        if compact_query:
            mask = mask | compact_name.str.contains(re.escape(compact_query), na=False)
        if len(tokens) > 1:
            all_tokens = pd.Series(True, index=df.index)
            for token in tokens:
                all_tokens = all_tokens & name_norm.str.contains(re.escape(token), na=False)
            mask = mask | all_tokens
        elif len(tokens) == 1:
            mask = mask | name_norm.str.contains(re.escape(tokens[0]), na=False)
        df = df[mask]
    return df.sort_values("portfolio_value", ascending=False).head(limit).to_dict("records")


def _latest_manager_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "report_period" not in df.columns:
        return df
    out = df.copy()
    out["_report_period_dt"] = pd.to_datetime(out["report_period"], format="%d-%b-%Y", errors="coerce")
    out = out.sort_values(["_report_period_dt", "portfolio_value"], ascending=[False, False])
    out = out.drop_duplicates(["cik"], keep="first").drop(columns=["_report_period_dt"])
    return out


def managers_portfolio_analysis(query: str = "", limit: int = 30) -> dict[str, Any]:
    selected = pd.DataFrame(search_managers(query, limit))
    if selected.empty:
        return {"manager_count": 0, "holding_count": 0, "total_value": 0.0, "top_holdings": [], "change_summary": {}}
    ciks = selected["cik"].astype(str).tolist()
    latest = selected[["cik", "report_period"]].copy() if "report_period" in selected.columns else selected[["cik"]].copy()
    holdings = load_holdings()
    h = holdings[holdings["cik"].astype(str).isin(ciks)].copy()
    if not h.empty and "report_period" in latest.columns:
        h = h.merge(latest, on=["cik", "report_period"], how="inner")
    if h.empty:
        return {
            "manager_count": int(len(ciks)),
            "holding_count": 0,
            "total_value": 0.0,
            "top_holdings": [],
            "change_summary": {},
        }
    rows = (
        h.groupby(["issuer", "ticker", "cusip"], as_index=False, dropna=False)
        .agg(
            value=("value", "sum"),
            shares=("shares", "sum"),
            manager_count=("cik", "nunique"),
        )
        .sort_values(["value", "manager_count"], ascending=False)
    )
    total = float(rows["value"].sum())
    rows["portfolio_weight"] = (rows["value"] / total) if total else 0.0

    tx = load_transactions()
    tx = tx[tx["cik"].astype(str).isin(ciks)].copy()
    if not tx.empty and "report_period" in latest.columns:
        tx = tx.merge(latest, on=["cik", "report_period"], how="inner")
    if not tx.empty:
        no_prior = (
            "previous_shares" in tx.columns
            and pd.to_numeric(tx["previous_shares"], errors="coerce").fillna(0.0).sum() <= 0
            and set(tx["change_type"].fillna("").astype(str).str.upper().unique()) <= {"NEW", "UNKNOWN"}
        )
        if no_prior:
            tx["change_type"] = "UNKNOWN"
        change_summary = tx["change_type"].fillna("UNKNOWN").astype(str).str.upper().value_counts().to_dict()
    else:
        change_summary = {}

    return {
        "manager_count": int(len(ciks)),
        "holding_count": int(rows["cusip"].nunique()),
        "total_value": total,
        "top_holdings": rows.head(20).to_dict("records"),
        "change_summary": {str(k): int(v) for k, v in change_summary.items()},
    }


def manager_detail(cik: str, holding_query: str = "", limit: int = 150) -> dict[str, Any]:
    raw_managers = load_managers()
    managers = _latest_manager_rows(raw_managers)
    holdings = load_holdings()
    transactions = load_transactions()
    m = managers[managers["cik"].astype(str) == str(cik)]
    if m.empty:
        raise KeyError("Manager not found")
    summary = m.sort_values("portfolio_value", ascending=False).iloc[0].to_dict()
    h = holdings[holdings["cik"].astype(str) == str(cik)]
    if "report_period" in h.columns and summary.get("report_period"):
        h = h[h["report_period"].astype(str) == str(summary["report_period"])]
    rows = h[["issuer", "ticker", "cusip", "value", "shares"]].copy()
    rows = (
        rows.sort_values("value", ascending=False)
        .groupby("cusip", as_index=False, dropna=False)
        .agg(
            issuer=("issuer", "first"),
            ticker=("ticker", "first"),
            value=("value", "max"),
            shares=("shares", "max"),
        )
    )
    total = float(rows["value"].sum())
    summary["portfolio_value"] = total
    summary["holdings_count"] = int(rows["cusip"].nunique())
    tx = transactions[transactions["cik"].astype(str) == str(cik)].copy()
    if "report_period" in tx.columns and summary.get("report_period"):
        tx = tx[tx["report_period"].astype(str) == str(summary["report_period"])]
    tx_value_cols = ["previous_shares", "current_shares", "previous_value", "current_value", "share_change", "value_change"]
    no_prior_baseline = (
        not tx.empty
        and "previous_shares" in tx.columns
        and pd.to_numeric(tx["previous_shares"], errors="coerce").fillna(0.0).sum() <= 0
        and set(tx["change_type"].fillna("").astype(str).str.upper().unique()) <= {"NEW", "UNKNOWN"}
    )
    if not tx.empty:
        join_cols = ["cusip"] if "cusip" in tx.columns else ["issuer", "ticker"]
        tx_summary = (
            tx.groupby(join_cols, as_index=False, dropna=False)
            .agg(
                previous_shares=("previous_shares", "sum"),
                current_shares=("current_shares", "sum"),
                previous_value=("previous_value", "sum"),
                current_value=("current_value", "sum"),
                change_type=("change_type", "first"),
                share_change=("share_change", "sum"),
                value_change=("value_change", "sum"),
            )
        )
        rows = rows.merge(tx_summary, on=join_cols, how="left")
    for col in tx_value_cols:
        rows[col] = pd.to_numeric(rows[col], errors="coerce") if col in rows.columns else pd.NA
    rows["current_shares"] = rows["current_shares"].fillna(rows["shares"])
    rows["current_value"] = rows["current_value"].fillna(rows["value"])
    rows["previous_shares"] = rows["previous_shares"].fillna(0.0)
    rows["previous_value"] = rows["previous_value"].fillna(0.0)
    rows["share_change"] = rows["share_change"].fillna(rows["current_shares"] - rows["previous_shares"])
    rows["value_change"] = rows["value_change"].fillna(rows["current_value"] - rows["previous_value"])
    if "change_type" not in rows.columns:
        rows["change_type"] = "UNCHANGED"
    rows["change_type"] = rows["change_type"].fillna("UNCHANGED")
    rows.loc[(rows["previous_shares"] <= 0) & (rows["current_shares"] > 0), "change_type"] = "NEW"
    rows.loc[(rows["previous_shares"] > 0) & (rows["current_shares"] > rows["previous_shares"]), "change_type"] = "INCREASED"
    rows.loc[(rows["previous_shares"] > 0) & (rows["current_shares"] < rows["previous_shares"]), "change_type"] = "DECREASED"
    if no_prior_baseline:
        rows["change_type"] = "UNKNOWN"
        rows["share_change"] = 0.0
        rows["value_change"] = 0.0
    if holding_query:
        q = holding_query.lower().strip()
        compact = re.sub(r"[^a-z0-9]", "", q)
        text = (
            rows["issuer"].fillna("").astype(str)
            + " "
            + rows["ticker"].fillna("").astype(str)
            + " "
            + rows["cusip"].fillna("").astype(str)
        ).str.lower()
        compact_text = text.str.replace(r"[^a-z0-9]", "", regex=True)
        mask = text.str.contains(re.escape(q), na=False)
        if compact:
            mask = mask | compact_text.str.contains(re.escape(compact), na=False)
        rows = rows[mask]
    rows["portfolio_weight"] = (rows["value"] / total) if total else 0.0
    rows = rows.sort_values("portfolio_weight", ascending=False).head(limit)
    return {"manager": summary, "holdings": rows.to_dict("records")}


def stock_detail(ticker: str) -> dict[str, Any]:
    holdings = load_holdings()
    tx = load_transactions()
    if holdings.empty:
        return {"ticker": ticker.upper(), "issuer": "", "holder_count": 0, "holders": [], "new_holders": [], "sold_holders": []}
    target = ticker.upper().strip()
    h = holdings[
        (holdings["ticker"].str.upper() == target)
        | (holdings["cusip"].str.upper() == target)
        | (holdings["issuer"].str.upper().str.contains(re.escape(target), na=False))
    ]
    t = tx[
        (tx["ticker"].str.upper() == target)
        | (tx["issuer"].str.upper().str.contains(re.escape(target), na=False))
    ]
    issuer = ""
    if not h.empty:
        issuer = str(h.iloc[0].get("issuer", ""))
    holders = t.sort_values("current_value", ascending=False).head(300).to_dict("records")
    return {
        "ticker": target,
        "issuer": issuer,
        "holder_count": int(t["cik"].nunique()) if not t.empty else int(h["cik"].nunique()),
        "holders": holders,
        "new_holders": t[t["change_type"] == "NEW"].sort_values("current_value", ascending=False).head(50).to_dict("records"),
        "sold_holders": t[t["change_type"] == "SOLD"].sort_values("previous_value", ascending=False).head(50).to_dict("records"),
    }
