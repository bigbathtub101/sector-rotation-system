"""
data_feeds.py — Phase 1: Unified Data Ingestion Module
=======================================================
Global Sector Rotation System

Pulls daily OHLCV prices (yfinance), macroeconomic data (FRED),
and SEC EDGAR filings. Stores everything in a structured SQLite
database with full validation, backfill logic, and error logging.

Dependencies: yfinance, pandas, requests, pyyaml, fredapi, sqlite3
"""

import os
import re
import sys
import json
import time
import logging
import sqlite3
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml
import pandas as pd
import numpy as np
import requests

# ---------------------------------------------------------------------------
# ENVIRONMENT VARIABLES CONTRACT
# ---------------------------------------------------------------------------
# The following secrets are read from os.environ (never hardcoded):
#
#   FRED_API_KEY         - Required for macro data (Phase 1). Free at fred.stlouisfed.org
#   SEC_EDGAR_EMAIL      - Optional override for SEC User-Agent header (Phase 1)
#   TELEGRAM_BOT_TOKEN   - Required for alert delivery (Phase 5)
#   TELEGRAM_CHAT_ID     - Required for alert delivery (Phase 5)
#   GMAIL_USERNAME       - Required for email alerts (Phase 5)
#   GMAIL_PASSWORD       - Gmail App Password for SMTP (Phase 5)
#   GOOGLE_SHEETS_CREDENTIALS - Service account JSON for gspread (Phase 5)
#
# In GitHub Actions, these are stored as repository secrets.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent
LOG_FILE = LOG_DIR / "data_errors.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("data_feeds")

# ---------------------------------------------------------------------------
# CONFIG LOADER
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load master configuration from YAML."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", path)
    return cfg


# ---------------------------------------------------------------------------
# DATABASE INITIALIZATION
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "rotation_system.db"


def init_database(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Create / connect to the SQLite database and ensure all required
    tables exist.  Tables: prices, macro_data, filings, signals, allocations.
    """
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            date        TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            adj_close   REAL,
            volume      INTEGER,
            stale_price INTEGER DEFAULT 0,
            fetched_at  TEXT,
            PRIMARY KEY (date, ticker)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS macro_data (
            date        TEXT NOT NULL,
            series_id   TEXT NOT NULL,
            value       REAL,
            fetched_at  TEXT,
            PRIMARY KEY (date, series_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS filings (
            cik             TEXT NOT NULL,
            ticker          TEXT,
            company_name    TEXT,
            filing_type     TEXT NOT NULL,
            filing_date     TEXT,
            accession_number TEXT NOT NULL,
            primary_document TEXT,
            filing_url      TEXT,
            raw_text        TEXT,
            fetched_at      TEXT,
            PRIMARY KEY (cik, accession_number)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            date            TEXT NOT NULL,
            signal_type     TEXT NOT NULL,
            signal_data     TEXT,
            created_at      TEXT,
            PRIMARY KEY (date, signal_type)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS allocations (
            date            TEXT NOT NULL,
            regime          TEXT,
            allocation_json TEXT,
            dollar_taxable  TEXT,
            dollar_roth     TEXT,
            created_at      TEXT,
            PRIMARY KEY (date)
        )
    """)

    # Index for faster lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_data(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker)")

    conn.commit()
    logger.info("Database initialized at %s", db_path)
    return conn


# ===========================================================================
# 1. PRICE DATA — yfinance
# ===========================================================================
def _get_all_tickers(cfg: dict) -> List[str]:
    """Flatten all ticker lists from config into a single deduplicated list."""
    tickers = []
    for key in ["sector_etfs", "geographic_etfs", "industry_etfs", "thematic_etfs",
                 "factor_etfs", "volatility", "benchmarks",
                 "watchlist_biotech", "watchlist_ai_software",
                 "watchlist_defense", "watchlist_green_materials",
                 "watchlist_semiconductors", "watchlist_energy_transition",
                 "watchlist_fintech"]:
        tickers.extend(cfg["tickers"].get(key, []))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in tickers:
        if isinstance(t, str) and t not in seen:
            seen.add(t)
            unique.append(t)
        elif not isinstance(t, str):
            logger.warning(
                "Skipping non-string ticker value: %r (check config.yaml for unquoted YAML booleans like ON/OFF/YES/NO)",
                t,
            )
    return unique


def fetch_prices(
    cfg: dict,
    start_date: str = None,
    end_date: str = None,
    tickers: List[str] = None,
) -> pd.DataFrame:
    """
    Pull daily OHLCV data for all configured tickers via yfinance.

    Parameters
    ----------
    cfg : dict          Master config.
    start_date : str    ISO date string (default: 2 years ago).
    end_date : str      ISO date string (default: today).
    tickers : list      Override ticker list (default: all from config).

    Returns
    -------
    pd.DataFrame with columns:
        date, ticker, open, high, low, close, adj_close, volume
    """
    import yfinance as yf

    if tickers is None:
        tickers = _get_all_tickers(cfg)
    if end_date is None:
        end_date = dt.date.today().isoformat()
    if start_date is None:
        start_date = (dt.date.today() - dt.timedelta(days=730)).isoformat()

    logger.info(
        "Fetching prices for %d tickers from %s to %s",
        len(tickers), start_date, end_date,
    )

    all_frames = []
    failed_tickers = []

    # yfinance supports batch download — use it for speed
    try:
        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )
    except Exception as e:
        logger.error("Batch yfinance download failed: %s", e)
        raw = pd.DataFrame()

    if raw.empty:
        logger.warning("yfinance batch returned empty — falling back to per-ticker download")
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
                if df.empty:
                    failed_tickers.append(ticker)
                    continue
                df = df.reset_index()
                df["ticker"] = ticker
                all_frames.append(df)
            except Exception as e:
                logger.error("Failed to fetch %s: %s", ticker, e)
                failed_tickers.append(ticker)
    else:
        # Parse the multi-level column DataFrame
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df_t = raw.copy()
                else:
                    df_t = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else pd.DataFrame()

                if df_t.empty or df_t.dropna(how="all").empty:
                    failed_tickers.append(ticker)
                    continue

                df_t = df_t.reset_index()
                df_t["ticker"] = ticker
                all_frames.append(df_t)
            except Exception as e:
                logger.error("Error parsing %s from batch: %s", ticker, e)
                failed_tickers.append(ticker)

    if failed_tickers:
        logger.warning("Failed tickers (%d): %s", len(failed_tickers), failed_tickers)

    if not all_frames:
        logger.error("No price data fetched for any ticker.")
        return pd.DataFrame()

    prices = pd.concat(all_frames, ignore_index=True)

    # Normalize column names — handle yfinance version differences
    # yfinance may return 'Adj Close', 'adj close', 'Adj_Close', etc.
    col_map = {}
    for c in prices.columns:
        cl = str(c).lower().strip().replace(" ", "_")
        if cl in ("date", "datetime", "index"):
            col_map[c] = "date"
        elif cl in ("adj_close", "adj close", "adjclose", "adjusted_close"):
            col_map[c] = "adj_close"
        elif cl == "price" and "adj_close" not in [col_map.get(x) for x in col_map]:
            # Some yfinance versions use 'Price' as a column name
            col_map[c] = "adj_close"
        else:
            col_map[c] = cl
    prices.rename(columns=col_map, inplace=True)

    # Ensure date column is string
    if "date" in prices.columns:
        prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")

    # Keep only expected columns
    expected = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    for col in expected:
        if col not in prices.columns:
            prices[col] = None
    prices = prices[expected]

    logger.info("Fetched %d price rows for %d tickers", len(prices), prices["ticker"].nunique())
    return prices


# ===========================================================================
# 2. MACROECONOMIC DATA — FRED
# ===========================================================================
def fetch_macro_data(
    cfg: dict,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Pull macroeconomic series from FRED API.

    Requires environment variable FRED_API_KEY.
    Register free at https://fred.stlouisfed.org/docs/api/api_key.html

    Returns
    -------
    pd.DataFrame with columns: date, series_id, value
    """
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        logger.warning(
            "FRED_API_KEY not set — skipping macro data fetch. "
            "Register free at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return pd.DataFrame(columns=["date", "series_id", "value"])

    if end_date is None:
        end_date = dt.date.today().isoformat()
    if start_date is None:
        start_date = (dt.date.today() - dt.timedelta(days=730)).isoformat()

    fred_series = cfg["fred"]["series"]
    all_rows = []

    for name, series_id in fred_series.items():
        logger.info("Fetching FRED series: %s (%s)", name, series_id)
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}"
                f"&api_key={api_key}"
                f"&file_type=json"
                f"&observation_start={start_date}"
                f"&observation_end={end_date}"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for obs in data.get("observations", []):
                val = obs.get("value", ".")
                if val == ".":
                    continue  # FRED uses "." for missing
                all_rows.append({
                    "date": obs["date"],
                    "series_id": series_id,
                    "value": float(val),
                })
        except Exception as e:
            logger.error("Failed to fetch FRED series %s: %s", series_id, e)

    if not all_rows:
        logger.warning("No macro data fetched from FRED.")
        return pd.DataFrame(columns=["date", "series_id", "value"])

    df = pd.DataFrame(all_rows)
    logger.info("Fetched %d macro observations across %d series",
                len(df), df["series_id"].nunique())
    return df


# ===========================================================================
# 2B. ETF TOP HOLDINGS LOOKUP
# ===========================================================================
def get_etf_top_holdings(
    etf_ticker: str,
    n: int = 20,
) -> List[str]:
    """
    Retrieve the top N holdings of an ETF by weight using yfinance.

    Falls back to an empty list if the ETF has no holdings data
    (e.g., delisted, non-US, or yfinance data gap).

    Parameters
    ----------
    etf_ticker : str   ETF ticker symbol (e.g., 'XLK').
    n : int            Number of top holdings to return (default: 20).

    Returns
    -------
    List of ticker strings for the top holdings, ordered by weight.
    """
    import yfinance as yf

    try:
        etf = yf.Ticker(etf_ticker)

        # yfinance exposes holdings via .funds_data.top_holdings or
        # the older .info['holdings'] path. Try the modern API first.
        try:
            holdings_df = etf.funds_data.top_holdings
            if holdings_df is not None and not holdings_df.empty:
                # holdings_df index is ticker symbols, sorted by weight desc
                tickers = holdings_df.index.tolist()[:n]
                logger.info(
                    "ETF %s: retrieved %d/%d top holdings via funds_data",
                    etf_ticker, len(tickers), n,
                )
                return [str(t).upper() for t in tickers]
        except (AttributeError, Exception):
            pass

        # Fallback: some yfinance versions expose holdings in .info
        info = etf.info or {}
        holdings = info.get("holdings", [])
        if holdings:
            tickers = [h.get("symbol", "") for h in holdings[:n] if h.get("symbol")]
            if tickers:
                logger.info(
                    "ETF %s: retrieved %d/%d top holdings via info",
                    etf_ticker, len(tickers), n,
                )
                return [t.upper() for t in tickers]

        logger.warning("ETF %s: no holdings data available in yfinance", etf_ticker)
        return []

    except Exception as e:
        logger.error("Failed to get holdings for ETF %s: %s", etf_ticker, e)
        return []


def get_sector_etf_top_holdings(
    cfg: dict,
    n_per_etf: int = None,
) -> Dict[str, List[str]]:
    """
    For each sector ETF in config, retrieve the top N holdings.

    Parameters
    ----------
    cfg : dict         Master config.
    n_per_etf : int    Holdings per ETF (default: from config sec_edgar.top_holdings_per_etf).

    Returns
    -------
    Dict mapping ETF ticker → list of holding tickers.
    Example: {'XLK': ['AAPL', 'MSFT', 'NVDA', ...], ...}
    """
    if n_per_etf is None:
        n_per_etf = cfg["sec_edgar"].get("top_holdings_per_etf", 5)

    sector_etfs = cfg["tickers"]["sector_etfs"]
    result = {}

    for etf in sector_etfs:
        holdings = get_etf_top_holdings(etf, n=n_per_etf)
        result[etf] = holdings
        if not holdings:
            logger.warning("No holdings resolved for %s — filings will be skipped", etf)

    total = sum(len(v) for v in result.values())
    logger.info(
        "Resolved %d total holdings across %d sector ETFs",
        total, len(sector_etfs),
    )
    return result


# ===========================================================================
# 3. SEC EDGAR FILINGS
# ===========================================================================

# Mapping of sector ETF top holdings to their CIK numbers.
# This is bootstrapped by the CIK lookup function and cached.
_CIK_CACHE: Dict[str, str] = {}


def _sec_headers(cfg: dict) -> dict:
    """Return compliant SEC User-Agent header."""
    config_ua = cfg["sec_edgar"]["user_agent"].strip()
    ua = config_ua

    # Allow override via environment variable
    email = os.environ.get("SEC_EDGAR_EMAIL", "")
    if email:
        # Strip ALL control characters (anything outside printable ASCII 0x20-0x7E)
        # Colab Secrets can inject \r\n and other invisible characters
        email = re.sub(r"[^\x20-\x7E]", "", email).strip()
        if email:
            candidate_ua = f"QuantSystemBuilder/1.0 ({email})"
            # Final validation — strip control chars from the full string too
            candidate_ua = re.sub(r"[^\x20-\x7E]", "", candidate_ua).strip()
            if "\r" in candidate_ua or "\n" in candidate_ua or not candidate_ua:
                logger.warning(
                    "SEC_EDGAR_EMAIL produced an invalid User-Agent after sanitization "
                    "(repr=%r) — falling back to config.yaml user_agent", candidate_ua
                )
            else:
                ua = candidate_ua
        else:
            logger.warning(
                "SEC_EDGAR_EMAIL was set but contained only control characters — "
                "falling back to config.yaml user_agent"
            )

    logger.debug("SEC User-Agent header value: %r", ua)
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def _sec_sleep(cfg: dict):
    """Rate-limit pause between SEC API calls."""
    time.sleep(cfg["sec_edgar"]["rate_limit_sleep"])


def lookup_cik(ticker: str, cfg: dict) -> Optional[str]:
    """
    Look up a company's CIK number from SEC EDGAR company tickers JSON.
    """
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]

    headers = _sec_headers(cfg)
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=headers, timeout=30)
        _sec_sleep(cfg)
        resp.raise_for_status()
        data = resp.json()

        # Build full cache
        for entry in data.values():
            t = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", ""))
            _CIK_CACHE[t] = cik

        return _CIK_CACHE.get(ticker.upper())
    except Exception as e:
        logger.error("CIK lookup failed for %s: %s", ticker, e)
        return None


def fetch_filings_for_ticker(
    ticker: str,
    cfg: dict,
    filing_types: List[str] = None,
    max_filings: int = 5,
) -> List[dict]:
    """
    Fetch recent filings for a single ticker from SEC EDGAR.

    Returns a list of dicts with filing metadata and raw text.
    """
    if filing_types is None:
        filing_types = cfg["sec_edgar"]["filing_types"]

    cik = lookup_cik(ticker, cfg)
    if not cik:
        logger.warning("No CIK found for %s — skipping filings", ticker)
        return []

    cik_padded = cik.zfill(10)
    headers = _sec_headers(cfg)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    try:
        resp = requests.get(submissions_url, headers=headers, timeout=30)
        _sec_sleep(cfg)
        resp.raise_for_status()
        sub_data = resp.json()
    except Exception as e:
        logger.error("Failed to fetch submissions for %s (CIK %s): %s", ticker, cik, e)
        return []

    company_name = sub_data.get("name", ticker)
    recent = sub_data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    results = []
    count = 0

    for i, form in enumerate(forms):
        if count >= max_filings:
            break
        if form not in filing_types:
            continue

        accession = accessions[i] if i < len(accessions) else ""
        filing_date = dates[i] if i < len(dates) else ""
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        accession_path = accession.replace("-", "")

        filing_url = (
            f"{cfg['sec_edgar']['archives_url']}"
            f"{cik}/{accession_path}/{primary_doc}"
        )

        # Attempt to download filing text with exponential backoff on 503
        raw_text = ""
        for attempt in range(3):
            try:
                doc_resp = requests.get(filing_url, headers=headers, timeout=60)
                _sec_sleep(cfg)
                if doc_resp.status_code == 200:
                    # Take first 100KB to avoid memory issues
                    raw_text = doc_resp.text[:100_000]
                    break
                elif doc_resp.status_code == 503:
                    wait = 2 ** attempt
                    logger.warning(
                        "503 fetching filing %s for %s (attempt %d/3) — retrying in %ds",
                        accession, ticker, attempt + 1, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        "Non-200 status (%d) fetching filing %s for %s",
                        doc_resp.status_code, accession, ticker,
                    )
                    break
            except Exception as e:
                logger.error("Failed to download filing %s for %s: %s", accession, ticker, e)
                break
        else:
            logger.warning(
                "All retries exhausted for filing %s for %s", accession, ticker,
            )

        results.append({
            "cik": cik,
            "ticker": ticker,
            "company_name": company_name,
            "filing_type": form,
            "filing_date": filing_date,
            "accession_number": accession,
            "primary_document": primary_doc,
            "filing_url": filing_url,
            "raw_text": raw_text,
            "fetched_at": dt.datetime.utcnow().isoformat(),
        })
        count += 1

    logger.info("Fetched %d filings for %s (%s)", len(results), ticker, company_name)
    return results


def fetch_filings_via_efts(
    cfg: dict,
    tickers: List[str],
    filing_types: List[str] = None,
    max_filings_per_ticker: int = 5,
) -> List[dict]:
    """
    Fetch filing metadata for a list of tickers using the SEC EFTS full-text
    search API (https://efts.sec.gov/LATEST/search-index).

    This is faster than hitting /submissions/ per-ticker because it can
    discover filings for many tickers in batched queries.

    Returns a list of filing dicts (same schema as fetch_filings_for_ticker),
    with raw_text populated by a subsequent download from the Archives URL.
    """
    if filing_types is None:
        filing_types = cfg["sec_edgar"]["filing_types"]

    efts_url = cfg["sec_edgar"].get(
        "efts_search_url", "https://efts.sec.gov/LATEST/search-index"
    )
    batch_size = cfg["sec_edgar"].get("efts_batch_size", 10)
    headers = _sec_headers(cfg)
    archives_url = cfg["sec_edgar"]["archives_url"]

    all_filings: List[dict] = []

    # Process tickers in batches
    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start:batch_start + batch_size]
        # Build quoted ticker query  e.g. "NVDA" OR "AAPL"
        q_terms = " OR ".join(f'"{t}"' for t in batch)
        forms_param = ",".join(filing_types)

        params = {
            "q": q_terms,
            "forms": forms_param,
            "dateRange": "custom",
            "startdt": (
                dt.date.today() - dt.timedelta(days=365 * 3)
            ).isoformat(),
            "enddt": dt.date.today().isoformat(),
        }

        try:
            resp = requests.get(efts_url, params=params, headers=headers, timeout=30)
            _sec_sleep(cfg)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("EFTS query failed for batch %s: %s", batch, e)
            return []  # signal failure so caller can fall back

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            logger.debug("EFTS returned 0 hits for batch %s", batch)
            continue

        # Count per ticker so we respect max_filings_per_ticker
        ticker_counts: Dict[str, int] = {}

        for hit in hits:
            src = hit.get("_source", {})
            form_type = src.get("form_type", "")
            if form_type not in filing_types:
                continue

            # EFTS returns entity_name + display_names; match back to ticker
            display_names = src.get("display_names", [])
            # display_names is a list of "Name (CIK NNNNNN)" strings
            filing_ticker = ""
            cik = src.get("entity_id", "")
            # Try to match to one of our batch tickers by reverse-CIK lookup
            # (we may not have the CIK yet, so just record what EFTS gives us)
            entity_name = src.get("entity_name", "")

            accession_raw = src.get("file_num", "") or src.get("accession_no", "")
            # EFTS stores accession as "0001234567-24-000001"
            accession = accession_raw.replace("-", "") if accession_raw else ""
            filing_date = src.get("period_of_report", "") or src.get("file_date", "")
            primary_doc = src.get("file_num", "")  # not always present

            # Build the archives URL from CIK + accession
            # CIK in EFTS is zero-padded 10-digit
            cik_str = str(cik).lstrip("0") if cik else ""
            if not cik_str or not accession:
                continue

            filing_url = f"{archives_url}{cik_str}/{accession}/"

            # Match the hit back to one of the tickers we queried
            matched_ticker = ""
            for t in batch:
                if t.upper() in entity_name.upper():
                    matched_ticker = t
                    break
            if not matched_ticker:
                continue

            if ticker_counts.get(matched_ticker, 0) >= max_filings_per_ticker:
                continue

            # Download filing text
            raw_text = ""
            for attempt in range(3):
                try:
                    doc_resp = requests.get(filing_url, headers=headers, timeout=60)
                    _sec_sleep(cfg)
                    if doc_resp.status_code == 200:
                        raw_text = doc_resp.text[:100_000]
                        break
                    elif doc_resp.status_code == 503:
                        wait = 2 ** attempt
                        logger.warning(
                            "503 fetching EFTS filing %s (attempt %d/3) — retrying in %ds",
                            accession_raw, attempt + 1, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.warning(
                            "Non-200 status (%d) fetching EFTS filing %s",
                            doc_resp.status_code, accession_raw,
                        )
                        break
                except Exception as e:
                    logger.error("Failed to download EFTS filing %s: %s", accession_raw, e)
                    break

            all_filings.append({
                "cik": cik_str,
                "ticker": matched_ticker,
                "company_name": entity_name,
                "filing_type": form_type,
                "filing_date": filing_date,
                "accession_number": accession_raw,
                "primary_document": src.get("file_num", ""),
                "filing_url": filing_url,
                "raw_text": raw_text,
                "fetched_at": dt.datetime.utcnow().isoformat(),
            })
            ticker_counts[matched_ticker] = ticker_counts.get(matched_ticker, 0) + 1

    logger.info("EFTS: fetched %d filings for %d tickers", len(all_filings), len(tickers))
    return all_filings


def fetch_all_filings(
    cfg: dict,
    tickers: List[str] = None,
) -> List[dict]:
    """
    Fetch filings for the top holdings of each sector ETF
    plus all watchlist stocks.

    If no tickers are provided, dynamically resolves top holdings
    per sector ETF via yfinance, then adds all watchlist tickers.

    Primary path: EFTS full-text search API (faster, batch queries).
    Fallback: per-ticker /submissions/ endpoint (original approach).
    """
    if tickers is None:
        tickers = []

        # Dynamically resolve top holdings for each sector ETF
        try:
            holdings_map = get_sector_etf_top_holdings(cfg)
            for etf, holding_tickers in holdings_map.items():
                tickers.extend(holding_tickers)
        except Exception as e:
            logger.error("Failed to resolve ETF top holdings: %s", e)

        # Add watchlist tickers
        for key in ["watchlist_biotech", "watchlist_ai_software",
                     "watchlist_defense", "watchlist_green_materials",
                     "watchlist_semiconductors", "watchlist_energy_transition",
                     "watchlist_fintech"]:
            tickers.extend(cfg["tickers"].get(key, []))

    # Deduplicate while preserving order
    tickers = list(dict.fromkeys(tickers))

    use_efts = cfg["sec_edgar"].get("use_efts_primary", True)

    if use_efts:
        logger.info("Using EFTS as primary filing discovery for %d tickers", len(tickers))
        try:
            all_filings = fetch_filings_via_efts(cfg, tickers)
            if all_filings is not None:
                logger.info("EFTS succeeded: %d filings fetched", len(all_filings))
                logger.info("Total filings fetched: %d for %d tickers", len(all_filings), len(tickers))
                return all_filings
        except Exception as e:
            logger.error("EFTS primary path raised an exception: %s — falling back to submissions", e)

    # Fallback: per-ticker submissions approach
    logger.info("Using per-ticker submissions fallback for %d tickers", len(tickers))
    all_filings = []
    for ticker in tickers:
        filings = fetch_filings_for_ticker(ticker, cfg)
        all_filings.extend(filings)

    logger.info("Total filings fetched: %d for %d tickers", len(all_filings), len(tickers))
    return all_filings


# ===========================================================================
# 4. DATA VALIDATION & RESILIENCE LAYER
# ===========================================================================
def validate_prices(prices: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate price data:
    - Flag stale/missing prices with stale_price=True
    - Backfill using last valid close
    - Return warnings list

    Returns
    -------
    (validated_prices_df, warnings_list)
    """
    warnings = []
    if prices.empty:
        warnings.append("DATA_QUALITY_WARNING: No price data available.")
        return prices, warnings

    prices = prices.copy()
    # Drop rows with null dates before any processing to avoid downstream errors
    null_date_count = prices["date"].isna().sum()
    if null_date_count:
        logger.warning("Dropping %d rows with null date in validate_prices()", null_date_count)
        prices = prices.dropna(subset=["date"])
    if prices.empty:
        warnings.append("DATA_QUALITY_WARNING: No price data available after dropping null-date rows.")
        return prices, warnings

    prices["stale_price"] = 0

    # Identify the most recent trading date in the data
    latest_date = prices["date"].max()

    # Check each ticker for staleness on the latest date
    tickers_in_data = prices["ticker"].unique()
    sector_etfs = set(cfg["tickers"]["sector_etfs"])
    stale_sector_count = 0
    stale_tickers = []

    for ticker in tickers_in_data:
        mask = (prices["ticker"] == ticker) & (prices["date"] == latest_date)
        if mask.sum() == 0:
            # No data for latest date — backfill
            ticker_data = prices[prices["ticker"] == ticker].sort_values("date")
            if not ticker_data.empty:
                last_row = ticker_data.iloc[-1].copy()
                last_row["date"] = latest_date
                last_row["stale_price"] = 1
                prices = pd.concat([prices, pd.DataFrame([last_row])], ignore_index=True)
                stale_tickers.append(ticker)
                logger.warning(
                    "BACKFILL: %s missing on %s — used last valid close from %s",
                    ticker, latest_date, ticker_data.iloc[-1]["date"],
                )
                if ticker in sector_etfs:
                    stale_sector_count += 1

        # Check for NaN closes on latest date
        mask2 = (prices["ticker"] == ticker) & (prices["date"] == latest_date)
        row = prices.loc[mask2]
        if not row.empty and pd.isna(row["close"].values[0]):
            ticker_data = prices[
                (prices["ticker"] == ticker) & prices["close"].notna()
            ].sort_values("date")
            if not ticker_data.empty:
                fill_val = ticker_data.iloc[-1]["close"]
                prices.loc[mask2, "close"] = fill_val
                prices.loc[mask2, "adj_close"] = fill_val
                prices.loc[mask2, "stale_price"] = 1
                stale_tickers.append(ticker)
                logger.warning(
                    "BACKFILL: %s had NaN close on %s — filled with %s",
                    ticker, latest_date, fill_val,
                )
                if ticker in sector_etfs:
                    stale_sector_count += 1

    if stale_tickers:
        warnings.append(
            f"STALE_PRICES: {len(stale_tickers)} tickers backfilled on {latest_date}: "
            f"{', '.join(stale_tickers)}"
        )

    stale_limit = cfg["data_quality"]["stale_sector_etf_limit"]
    if stale_sector_count >= stale_limit:
        msg = (
            f"DATA_QUALITY_WARNING: {stale_sector_count} sector ETFs have stale prices "
            f"on {latest_date} (limit: {stale_limit}). "
            f"Regime transition signals SUPPRESSED for this day."
        )
        warnings.append(msg)
        logger.critical(msg)

    logger.info(
        "Price validation complete: %d rows, %d stale tickers, %d warnings",
        len(prices), len(stale_tickers), len(warnings),
    )
    return prices, warnings


def validate_macro(macro: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Basic validation for macro data — check for gaps and log."""
    warnings = []
    if macro.empty:
        warnings.append("MACRO_WARNING: No macro data available.")
        return macro, warnings

    for sid in macro["series_id"].unique():
        series = macro[macro["series_id"] == sid].sort_values("date")
        if len(series) < 2:
            warnings.append(f"MACRO_WARNING: Series {sid} has fewer than 2 observations.")

    return macro, warnings


# ===========================================================================
# 5. DATABASE STORAGE
# ===========================================================================
def store_prices(conn: sqlite3.Connection, prices: pd.DataFrame):
    """Insert or replace price rows into the database."""
    if prices.empty:
        return
    now = dt.datetime.utcnow().isoformat()
    prices = prices.copy()
    # Drop rows with null dates to avoid NOT NULL constraint violations
    null_date_count = prices["date"].isna().sum()
    if null_date_count:
        logger.warning("Dropping %d rows with null date before store_prices()", null_date_count)
        prices = prices.dropna(subset=["date"])
    if prices.empty:
        return
    prices["fetched_at"] = now

    rows = prices[
        ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume",
         "stale_price", "fetched_at"]
    ].values.tolist()

    conn.executemany(
        """
        INSERT OR REPLACE INTO prices
            (date, ticker, open, high, low, close, adj_close, volume, stale_price, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    logger.info("Stored %d price rows in database.", len(rows))


def store_macro(conn: sqlite3.Connection, macro: pd.DataFrame):
    """Insert or replace macro data rows."""
    if macro.empty:
        return
    now = dt.datetime.utcnow().isoformat()
    macro = macro.copy()
    macro["fetched_at"] = now

    rows = macro[["date", "series_id", "value", "fetched_at"]].values.tolist()

    conn.executemany(
        """
        INSERT OR REPLACE INTO macro_data (date, series_id, value, fetched_at)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    logger.info("Stored %d macro rows in database.", len(rows))


def store_filings(conn: sqlite3.Connection, filings: List[dict]):
    """Insert or replace filings into the database."""
    if not filings:
        return
    rows = [
        (
            f["cik"], f["ticker"], f["company_name"], f["filing_type"],
            f["filing_date"], f["accession_number"], f["primary_document"],
            f["filing_url"], f["raw_text"], f["fetched_at"],
        )
        for f in filings
    ]

    conn.executemany(
        """
        INSERT OR REPLACE INTO filings
            (cik, ticker, company_name, filing_type, filing_date,
             accession_number, primary_document, filing_url, raw_text, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    logger.info("Stored %d filings in database.", len(rows))


# ===========================================================================
# 6. QUERY HELPERS
# ===========================================================================
def get_prices(
    conn: sqlite3.Connection,
    tickers: List[str] = None,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """Read prices from the database with optional filters."""
    query = "SELECT * FROM prices WHERE 1=1"
    params = []

    if tickers:
        placeholders = ",".join(["?"] * len(tickers))
        query += f" AND ticker IN ({placeholders})"
        params.extend(tickers)
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY date, ticker"
    return pd.read_sql_query(query, conn, params=params)


def get_macro(
    conn: sqlite3.Connection,
    series_ids: List[str] = None,
    start_date: str = None,
) -> pd.DataFrame:
    """Read macro data from the database."""
    query = "SELECT * FROM macro_data WHERE 1=1"
    params = []

    if series_ids:
        placeholders = ",".join(["?"] * len(series_ids))
        query += f" AND series_id IN ({placeholders})"
        params.extend(series_ids)
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)

    query += " ORDER BY date, series_id"
    return pd.read_sql_query(query, conn, params=params)


def get_filings(
    conn: sqlite3.Connection,
    tickers: List[str] = None,
    filing_type: str = None,
) -> pd.DataFrame:
    """Read filings from the database."""
    query = "SELECT cik, ticker, company_name, filing_type, filing_date, accession_number, filing_url, fetched_at FROM filings WHERE 1=1"
    params = []

    if tickers:
        placeholders = ",".join(["?"] * len(tickers))
        query += f" AND ticker IN ({placeholders})"
        params.extend(tickers)
    if filing_type:
        query += " AND filing_type = ?"
        params.append(filing_type)

    query += " ORDER BY filing_date DESC"
    return pd.read_sql_query(query, conn, params=params)


# ===========================================================================
# MASTER INGESTION FUNCTION
# ===========================================================================
def run_full_ingestion(
    cfg: dict = None,
    start_date: str = None,
    end_date: str = None,
    skip_filings: bool = False,
) -> dict:
    """
    Run the complete data ingestion pipeline:
      1. Fetch prices via yfinance
      2. Validate prices (backfill, staleness)
      3. Fetch macro data from FRED
      4. Validate macro data
      5. Fetch SEC filings (optional — can be slow)
      6. Store everything in SQLite

    Returns a summary dict.
    """
    if cfg is None:
        cfg = load_config()

    conn = init_database()
    summary = {
        "timestamp": dt.datetime.utcnow().isoformat(),
        "prices": {},
        "macro": {},
        "filings": {},
        "warnings": [],
    }

    # --- Prices ---
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching price data")
    logger.info("=" * 60)
    prices = fetch_prices(cfg, start_date=start_date, end_date=end_date)
    prices, price_warnings = validate_prices(prices, cfg)
    summary["warnings"].extend(price_warnings)
    store_prices(conn, prices)
    summary["prices"] = {
        "rows": len(prices),
        "tickers": int(prices["ticker"].nunique()) if not prices.empty else 0,
        "date_range": (
            f"{prices['date'].min()} to {prices['date'].max()}"
            if not prices.empty else "N/A"
        ),
        "stale_count": int(prices["stale_price"].sum()) if not prices.empty else 0,
    }

    # --- Macro ---
    logger.info("=" * 60)
    logger.info("STEP 2: Fetching macro data from FRED")
    logger.info("=" * 60)
    macro = fetch_macro_data(cfg, start_date=start_date, end_date=end_date)
    macro, macro_warnings = validate_macro(macro)
    summary["warnings"].extend(macro_warnings)
    store_macro(conn, macro)
    summary["macro"] = {
        "rows": len(macro),
        "series": int(macro["series_id"].nunique()) if not macro.empty else 0,
    }

    # --- Filings ---
    if not skip_filings:
        logger.info("=" * 60)
        logger.info("STEP 3: Fetching SEC EDGAR filings")
        logger.info("=" * 60)
        filings = fetch_all_filings(cfg)
        store_filings(conn, filings)
        summary["filings"] = {
            "total": len(filings),
            "tickers_covered": len(set(f["ticker"] for f in filings)),
        }
    else:
        logger.info("STEP 3: SEC filings skipped (skip_filings=True)")
        summary["filings"] = {"total": 0, "tickers_covered": 0, "skipped": True}

    conn.close()

    # Log summary
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Summary: %s", json.dumps(summary, indent=2))

    return summary


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sector Rotation Data Feeds")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--skip-filings", action="store_true", help="Skip SEC filing fetch")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config)) if args.config else load_config()
    summary = run_full_ingestion(
        cfg=cfg,
        start_date=args.start,
        end_date=args.end,
        skip_filings=args.skip_filings,
    )
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
