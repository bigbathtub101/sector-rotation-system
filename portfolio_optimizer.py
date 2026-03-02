"""
portfolio_optimizer.py -- Phase 3: Factor Scoring & CVaR Portfolio Optimizer
============================================================================
Global Sector Rotation System

Implements:
  1. Fama-French Five-Factor scoring with McLean-Pontiff decay
  2. Momentum overlay (12-1 month)
  3. CVaR optimization with Ledoit-Wolf shrinkage
  4. Longin-Solnik tail correlation adjustment for EM positions
  5. Regime-conditional allocation bands (Section 8.1)
  6. US Equities sub-sector allocation (DeMiguel equal-weight + valuation filter)
  7. Bivector Beta for Energy/Materials structural diversifiers
  8. Dollar allocation across Taxable / Roth IRA accounts

Dependencies: numpy, pandas, scipy, sklearn, pypfopt, statsmodels, yaml
"""

import json
import logging
import sqlite3
import datetime as dt
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sp_stats

# ETF Auto-Selector: dynamically chooses best-in-class ETFs per exposure slot
try:
    from etf_selector import (
        get_selected_tickers as _get_etf_selections,
        get_ticker_asset_class_map as _get_selector_asset_class_map,
    )
    _ETF_SELECTOR_AVAILABLE = True
except ImportError:
    _ETF_SELECTOR_AVAILABLE = False

# ---------------------------------------------------------------------------
# LOGGING & CONFIG
# ---------------------------------------------------------------------------
logger = logging.getLogger("portfolio_optimizer")
CONFIG_PATH = Path(__file__).parent / "config.yaml"
DB_PATH = Path(__file__).parent / "rotation_system.db"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ===========================================================================
# 1. FAMA-FRENCH FIVE-FACTOR SCORING
# ===========================================================================

def download_ff_factors(start_date: str = None) -> pd.DataFrame:
    """
    Download Fama-French 5 factors from Ken French's data library.
    Returns a DataFrame with DAILY factor returns: Mkt-RF, SMB, HML, RMW, CMA, RF.

    Uses direct CSV download from the Dartmouth server instead of
    pandas_datareader (which is broken on newer pandas/numpy due to
    the deprecated `date_parser` argument).

    Falls back to a simulated dataset if the download fails.
    """
    if start_date is None:
        start_date = (dt.date.today() - dt.timedelta(days=1100)).isoformat()

    try:
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urlopen(req, timeout=30)
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))

        # The zip contains a single CSV file
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        raw = zf.read(csv_name).decode("utf-8")

        # Find the header row (starts with date column -- skip description lines)
        lines = raw.strip().split("\n")
        header_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped[0].isdigit() is False and "Mkt-RF" in stripped:
                header_idx = i
                break

        if header_idx is None:
            # Try finding first numeric line as data start
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and stripped.split(",")[0].strip().isdigit():
                    header_idx = i - 1
                    break

        # Parse from header onward
        csv_text = "\n".join(lines[header_idx:]) if header_idx else raw
        df = pd.read_csv(io.StringIO(csv_text), skipinitialspace=True)

        # First column is the date (unnamed or named oddly)
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})

        # Drop any rows where Date is not a valid YYYYMMDD integer
        df["Date"] = pd.to_numeric(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Date"] = df["Date"].astype(int).astype(str)
        df = df[df["Date"].str.len() == 8]  # keep only YYYYMMDD rows

        df.index = pd.to_datetime(df["Date"], format="%Y%m%d")
        df.index.name = "date"
        df = df.drop(columns=["Date"])

        # Clean column names
        df.columns = [c.strip() for c in df.columns]

        # Convert to numeric and from percent to decimal
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df / 100.0
        df = df.dropna(how="all")

        # Filter to start_date
        df = df[df.index >= start_date]

        if df.empty:
            raise ValueError("FF5 DataFrame empty after filtering")

        logger.info("Downloaded Fama-French 5 factors: %d rows (direct CSV)", len(df))
        return df

    except Exception as e:
        logger.warning("Failed to download FF factors: %s -- using synthetic proxy", e)
        return _generate_synthetic_ff_factors(start_date)


def _generate_synthetic_ff_factors(start_date: str) -> pd.DataFrame:
    """
    Generate synthetic FF factor returns as a fallback.
    Uses realistic statistical properties for smoke testing.
    """
    dates = pd.bdate_range(start=start_date, end=dt.date.today())
    np.random.seed(42)
    n = len(dates)
    df = pd.DataFrame({
        "Mkt-RF": np.random.normal(0.0004, 0.01, n),
        "SMB": np.random.normal(0.0001, 0.005, n),
        "HML": np.random.normal(0.0001, 0.005, n),
        "RMW": np.random.normal(0.0001, 0.004, n),
        "CMA": np.random.normal(0.0001, 0.004, n),
        "RF": np.full(n, 0.0002),
    }, index=dates)
    df.index.name = "date"
    logger.info("Generated synthetic FF factors: %d rows", n)
    return df


# [TRUNCATED]
