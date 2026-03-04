"""
conftest.py — Shared pytest fixtures for the Sector Rotation System test suite.
================================================================================

Provides:
  - cfg: loaded config.yaml dict
  - db_path: temp SQLite DB seeded with synthetic data (auto-cleaned)
  - db_conn: SQLite connection to the seeded temp DB
  - sample_returns: synthetic return DataFrame for optimizer tests
  - sample_regime_probs: sample regime probability dicts
"""

import datetime as dt
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Locate repo root and config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Ensure repo root is importable
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Also add src/ for the new package layout
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cfg():
    """Load config.yaml once for the entire test session."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _make_temp_db():
    """Create a temp directory with an empty DB and return the DB path."""
    tmpdir = tempfile.mkdtemp(prefix="srs_test_")
    db_path = os.path.join(tmpdir, "rotation_system.db")
    return db_path


def _seed_db(db_path: str, cfg: dict, n_days: int = 120):
    """Seed a SQLite DB with synthetic price, macro, filings, signals,
    and allocations data — enough for regime detection and optimization tests."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # ----- Schema -----
    c.execute("""CREATE TABLE IF NOT EXISTS prices (
        date TEXT NOT NULL, ticker TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, adj_close REAL,
        volume INTEGER, stale_price INTEGER DEFAULT 0, fetched_at TEXT,
        PRIMARY KEY (date, ticker))""")
    c.execute("""CREATE TABLE IF NOT EXISTS macro_data (
        date TEXT NOT NULL, series_id TEXT NOT NULL, value REAL, fetched_at TEXT,
        PRIMARY KEY (date, series_id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS filings (
        cik TEXT NOT NULL, ticker TEXT, company_name TEXT,
        filing_type TEXT NOT NULL, filing_date TEXT,
        accession_number TEXT NOT NULL, primary_document TEXT,
        filing_url TEXT, raw_text TEXT, fetched_at TEXT,
        PRIMARY KEY (cik, accession_number))""")
    c.execute("""CREATE TABLE IF NOT EXISTS signals (
        date TEXT NOT NULL, signal_type TEXT NOT NULL,
        signal_data TEXT, created_at TEXT,
        PRIMARY KEY (date, signal_type))""")
    c.execute("""CREATE TABLE IF NOT EXISTS allocations (
        date TEXT NOT NULL, regime TEXT, allocation_json TEXT,
        dollar_taxable TEXT, dollar_roth TEXT, created_at TEXT,
        PRIMARY KEY (date))""")
    c.execute("""CREATE TABLE IF NOT EXISTS nlp_scores (
        date TEXT, ticker TEXT, sentiment_score REAL, confidence REAL,
        raw_json TEXT, PRIMARY KEY (date, ticker))""")
    c.execute("""CREATE TABLE IF NOT EXISTS nlp_sector_signals (
        date TEXT, sector TEXT, avg_sentiment REAL, filing_count INTEGER,
        tickers_json TEXT, PRIMARY KEY (date, sector))""")

    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")

    # ----- Synthetic Prices -----
    np.random.seed(42)
    end_date = dt.date(2026, 2, 27)
    dates = pd.bdate_range(end=end_date, periods=n_days).strftime("%Y-%m-%d").tolist()

    all_tickers = (
        cfg["tickers"]["sector_etfs"]
        + cfg["tickers"]["benchmarks"]
        + cfg["tickers"]["volatility"]
        + cfg["tickers"]["geographic_etfs"]
        + cfg["tickers"]["factor_etfs"]
    )

    for ticker in all_tickers:
        base = 50 + np.random.rand() * 200
        prices_list = [base]
        for _ in range(1, n_days):
            ret = np.random.normal(0.0003, 0.015)
            prices_list.append(prices_list[-1] * (1 + ret))
        for i, d in enumerate(dates):
            p = prices_list[i]
            c.execute(
                "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?,?,?,?)",
                (d, ticker, p * 0.99, p * 1.01, p * 0.98, p, p, int(1e6 + np.random.rand() * 5e6), 0, None),
            )

    # ----- Macro Data -----
    for series_id in ["FEDFUNDS", "T10Y2Y", "CPIAUCSL", "UNRATE", "CFNAI", "INDPRO"]:
        for d in dates[-6:]:
            val = np.random.uniform(-1, 5)
            c.execute("INSERT OR REPLACE INTO macro_data VALUES (?,?,?,?)", (d, series_id, val, None))

    # ----- Filings -----
    c.execute(
        "INSERT OR REPLACE INTO filings VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("12345", "NBIX", "Neurocrine", "10-K", dates[-5], "0001-23-456789", "doc.htm", "http://example.com", "Risk factors...", None),
    )

    # ----- Signals (regime) -----
    regimes = ["offense", "defense", "panic"]
    for i, d in enumerate(dates):
        regime = regimes[i % 3]
        sig_data = json.dumps({
            "regime": regime,
            "wedge_volume_percentile": np.random.uniform(0, 100),
            "regime_probabilities": {"panic": 0.1, "defense": 0.3, "offense": 0.6},
        })
        c.execute(
            "INSERT OR REPLACE INTO signals VALUES (?,?,?,?)",
            (d, "regime_state", sig_data, None),
        )

    # ----- Allocations -----
    alloc = {
        "us_equities": 0.40, "intl_developed": 0.15, "em_equities": 0.10,
        "energy_materials": 0.10, "healthcare": 0.12, "cash_short_duration": 0.13,
    }
    c.execute(
        "INSERT OR REPLACE INTO allocations VALUES (?,?,?,?,?,?)",
        (dates[-1], "offense", json.dumps(alloc), None, None, None),
    )

    # ----- NLP Scores -----
    c.execute(
        "INSERT OR REPLACE INTO nlp_scores VALUES (?,?,?,?,?)",
        (dates[-1], "NBIX", 0.72, 0.88, json.dumps({"label": "positive", "score": 0.72})),
    )

    # ----- NLP Sector Signals -----
    for s in cfg["tickers"]["sector_etfs"]:
        c.execute(
            "INSERT OR REPLACE INTO nlp_sector_signals VALUES (?,?,?,?,?)",
            (dates[-1], s, np.random.uniform(-0.5, 0.8), 2, json.dumps(["NBIX"])),
        )

    conn.commit()
    conn.close()


@pytest.fixture
def db_path(cfg):
    """Create a seeded temp SQLite DB and yield its path; clean up after."""
    path = _make_temp_db()
    _seed_db(path, cfg, n_days=120)
    yield path
    tmpdir = os.path.dirname(path)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def db_conn(db_path):
    """Provide a SQLite connection to the seeded temp DB."""
    conn = sqlite3.connect(db_path)
    yield conn
    conn.close()


@pytest.fixture
def sample_returns():
    """Generate a synthetic return DataFrame for optimizer tests."""
    np.random.seed(42)
    tickers = ["XLK", "XLV", "XLE", "XLF", "XLI", "XLB", "XLU", "XLP", "XLRE", "XLC", "XLY", "SPY"]
    n_days = 504
    dates = pd.bdate_range(end="2026-02-27", periods=n_days)
    data = np.random.normal(0.0004, 0.015, (n_days, len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def sample_regime_probs():
    """Sample regime probability dicts for testing."""
    return {
        "deep_panic": {"panic": 0.85, "defense": 0.12, "offense": 0.03},
        "core_defense": {"panic": 0.05, "defense": 0.75, "offense": 0.20},
        "full_offense": {"panic": 0.01, "defense": 0.04, "offense": 0.95},
        "transition": {"panic": 0.30, "defense": 0.45, "offense": 0.25},
    }
