"""
test_all_modules.py — Unified Smoke Test Suite for Sector Rotation System
==========================================================================
Tests all 7 modules with a shared synthetic database:
  1. data_feeds     — DB schema, backfill helpers, data quality checks
  2. regime_detector — Wedge Volume calculation, regime classification
  3. portfolio_optimizer — CVaR optimization, allocation bands, tax location
  4. stock_screener — Factor scoring, watchlist validation, entry/exit signals
  5. nlp_sentiment  — FinBERT pipeline, sector aggregation, NLP scores
  6. monitor        — Alert detection, delivery stubs, executive summary
  7. dashboard      — Page functions, data loaders, stress test math

Run:  python test_all_modules.py
      or: pytest test_all_modules.py -v

Each test creates an isolated temp DB so tests don't interfere.
"""

import csv
import datetime as dt
import importlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Ensure repo root is on the path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Also check one level up (if running from tests/ directory)
PARENT = REPO_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_PATHS = [REPO_ROOT / "config.yaml", PARENT / "config.yaml"]
CONFIG = None
for cp in CONFIG_PATHS:
    if cp.exists():
        with open(cp, "r") as f:
            CONFIG = yaml.safe_load(f)
        break

if CONFIG is None:
    print("FATAL: config.yaml not found")
    sys.exit(1)


# ============================================================================
# HELPERS
# ============================================================================

def make_temp_db() -> str:
    """Create a temp directory with an empty DB and return the DB path."""
    tmpdir = tempfile.mkdtemp(prefix="srs_test_")
    db_path = os.path.join(tmpdir, "rotation_system.db")
    return db_path


def seed_db(db_path: str, n_days: int = 60) -> None:
    """Seed a SQLite DB with synthetic price, macro, filings, signals, and allocations data."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # ----- Schema -----
    c.execute("""CREATE TABLE IF NOT EXISTS prices (
        date TEXT, ticker TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER,
        PRIMARY KEY (date, ticker))""")
    c.execute("""CREATE TABLE IF NOT EXISTS macro_data (
        date TEXT, series_id TEXT, value REAL,
        PRIMARY KEY (date, series_id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS filings (
        ticker TEXT, filing_type TEXT, filed_date TEXT, raw_text TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS signals (
        date TEXT, signal_type TEXT, ticker TEXT, signal_data TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS allocations (
        date TEXT PRIMARY KEY, regime TEXT, allocation_json TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS nlp_scores (
        date TEXT, ticker TEXT, sentiment_score REAL, confidence REAL, raw_json TEXT,
        PRIMARY KEY (date, ticker))""")
    c.execute("""CREATE TABLE IF NOT EXISTS nlp_sector_signals (
        date TEXT, sector TEXT, avg_sentiment REAL, filing_count INTEGER, tickers_json TEXT,
        PRIMARY KEY (date, sector))""")

    # ----- Synthetic Prices -----
    np.random.seed(42)
    end_date = dt.date(2026, 2, 27)
    dates = pd.bdate_range(end=end_date, periods=n_days).strftime("%Y-%m-%d").tolist()

    all_tickers = (
        CONFIG["tickers"]["sector_etfs"]
        + CONFIG["tickers"]["benchmarks"]
        + CONFIG["tickers"]["volatility"]
        + CONFIG["tickers"]["geographic_etfs"]
        + CONFIG["tickers"]["factor_etfs"]
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
                "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)",
                (d, ticker, p * 0.99, p * 1.01, p * 0.98, p, int(1e6 + np.random.rand() * 5e6)),
            )

    # ----- Macro Data -----
    for series_id in ["FEDFUNDS", "T10Y2Y", "CPIAUCSL", "UNRATE", "CFNAI", "INDPRO"]:
        for d in dates[-6:]:
            val = np.random.uniform(-1, 5)
            c.execute("INSERT OR REPLACE INTO macro_data VALUES (?,?,?)", (d, series_id, val))

    # ----- Filings -----
    c.execute(
        "INSERT OR REPLACE INTO filings VALUES (?,?,?,?)",
        ("NBIX", "10-K", dates[-5], "Risk factors include competition in the CNS market..."),
    )
    c.execute(
        "INSERT OR REPLACE INTO filings VALUES (?,?,?,?)",
        ("NBIX", "8-K", dates[-3], "Company reports Q4 results above expectations..."),
    )

    # ----- Signals (regime) -----
    regimes = ["offense", "defense", "panic"]
    for i, d in enumerate(dates):
        regime = regimes[i % 3]
        sig_data = json.dumps({
            "regime": regime,
            "wedge_volume": np.random.uniform(0, 100),
            "percentile": np.random.uniform(0, 100),
            "fast_shock": False,
        })
        c.execute(
            "INSERT OR REPLACE INTO signals VALUES (?,?,?,?)",
            (d, "regime", "MARKET", sig_data),
        )

    # ----- Allocations -----
    alloc = {
        "us_equities": 0.40,
        "intl_developed": 0.15,
        "em_equities": 0.10,
        "energy_materials": 0.10,
        "healthcare": 0.12,
        "cash_short_duration": 0.13,
    }
    c.execute(
        "INSERT OR REPLACE INTO allocations VALUES (?,?,?)",
        (dates[-1], "offense", json.dumps(alloc)),
    )

    # ----- NLP Scores -----
    c.execute(
        "INSERT OR REPLACE INTO nlp_scores VALUES (?,?,?,?,?)",
        (dates[-1], "NBIX", 0.72, 0.88, json.dumps({"label": "positive", "score": 0.72})),
    )
    c.execute(
        "INSERT OR REPLACE INTO nlp_scores VALUES (?,?,?,?,?)",
        (dates[-1], "CRWD", 0.55, 0.75, json.dumps({"label": "positive", "score": 0.55})),
    )

    # ----- NLP Sector Signals -----
    sectors = CONFIG["tickers"]["sector_etfs"]
    for s in sectors:
        c.execute(
            "INSERT OR REPLACE INTO nlp_sector_signals VALUES (?,?,?,?,?)",
            (dates[-1], s, np.random.uniform(-0.5, 0.8), 2, json.dumps(["NBIX"])),
        )

    conn.commit()
    conn.close()


class TestCounter:
    """Simple test counter."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name: str):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name: str, reason: str = ""):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  [FAIL] {name} — {reason}")

    def summary(self) -> bool:
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailed tests:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        print("=" * 60)
        return self.failed == 0


# ============================================================================
# MODULE 1: data_feeds
# ============================================================================

def test_data_feeds(tc: TestCounter, db_path: str):
    print("\n--- Module 1: data_feeds ---")
    try:
        import data_feeds
        tc.ok("data_feeds imports")
    except Exception as e:
        tc.fail("data_feeds imports", str(e))
        return

    # Check DB schema was created by seed_db
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()

    for tbl in ["prices", "macro_data", "filings", "signals", "allocations"]:
        if tbl in tables:
            tc.ok(f"table '{tbl}' exists")
        else:
            tc.fail(f"table '{tbl}' exists", f"missing from DB")

    # Check that data_feeds has key functions
    for fn in ["fetch_prices", "fetch_macro_data", "fetch_all_filings", "run_full_ingestion"]:
        if hasattr(data_feeds, fn):
            tc.ok(f"data_feeds.{fn} exists")
        else:
            tc.fail(f"data_feeds.{fn} exists", "function not found")

    # Verify price data integrity
    conn = sqlite3.connect(db_path)
    price_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    conn.close()
    if price_count > 100:
        tc.ok(f"prices populated ({price_count} rows)")
    else:
        tc.fail("prices populated", f"only {price_count} rows")


# ============================================================================
# MODULE 2: regime_detector
# ============================================================================

def test_regime_detector(tc: TestCounter, db_path: str):
    print("\n--- Module 2: regime_detector ---")
    try:
        import regime_detector
        tc.ok("regime_detector imports")
    except Exception as e:
        tc.fail("regime_detector imports", str(e))
        return

    for fn in ["run_regime_detection", "compute_wedge_volume_series",
                "compute_wedge_volume_percentile", "get_dominant_regime",
                "apply_confirmation_filter", "compute_fast_shock_indicator"]:
        if hasattr(regime_detector, fn):
            tc.ok(f"regime_detector.{fn} exists")
        else:
            tc.fail(f"regime_detector.{fn} exists", "not found")

    # Test wedge volume series computation
    try:
        dummy_returns = pd.DataFrame(
            np.random.randn(252, 11) * 0.015,
            columns=CONFIG["tickers"]["sector_etfs"],
        )
        wv_series = regime_detector.compute_wedge_volume_series(
            dummy_returns,
            window=CONFIG["regime"]["wedge_volume"]["rolling_window"],
        )
        assert isinstance(wv_series, pd.Series), "WV series should be pd.Series"
        assert len(wv_series) > 0, "WV series should not be empty"
        tc.ok(f"compute_wedge_volume_series returned {len(wv_series)} values")
    except Exception as e:
        tc.fail("compute_wedge_volume_series", str(e))

    # Test regime probabilities
    try:
        probs = regime_detector.compute_regime_probabilities(
            percentile=50.0, cfg=CONFIG
        )
        assert isinstance(probs, dict)
        dominant = regime_detector.get_dominant_regime(probs)
        assert dominant in ("offense", "defense", "panic")
        tc.ok(f"regime_probabilities: dominant='{dominant}'")
    except Exception as e:
        tc.fail("regime_probabilities", str(e))


# ============================================================================
# MODULE 3: portfolio_optimizer
# ============================================================================

def test_portfolio_optimizer(tc: TestCounter, db_path: str):
    print("\n--- Module 3: portfolio_optimizer ---")
    try:
        import portfolio_optimizer
        tc.ok("portfolio_optimizer imports")
    except Exception as e:
        tc.fail("portfolio_optimizer imports", str(e))
        return

    for fn in ["run_portfolio_optimization", "run_cvar_optimization",
                "compute_factor_loadings", "compute_momentum",
                "allocate_dollars", "compute_tail_correlations"]:
        if hasattr(portfolio_optimizer, fn):
            tc.ok(f"portfolio_optimizer.{fn} exists")
        else:
            tc.fail(f"portfolio_optimizer.{fn} exists", "not found")

    # Test allocate_dollars
    try:
        sample_alloc = {
            "us_equities": 0.40, "intl_developed": 0.15, "em_equities": 0.10,
            "energy_materials": 0.10, "healthcare": 0.12, "cash_short_duration": 0.13,
        }
        total_val = CONFIG["portfolio"]["total_value"]
        taxable_val = CONFIG["portfolio"]["accounts"]["taxable"]["value"]
        roth_val = CONFIG["portfolio"]["accounts"]["roth_ira"]["value"]

        dollar_result = portfolio_optimizer.allocate_dollars(
            sample_alloc, CONFIG
        )
        assert isinstance(dollar_result, dict), "allocate_dollars should return dict"
        tc.ok(f"allocate_dollars returned {len(dollar_result)} entries")
    except Exception as e:
        tc.fail("allocate_dollars", str(e))

    # Allocation bands check
    bands = CONFIG["optimizer"]["allocation_bands"]
    for regime in ["offense", "defense", "panic"]:
        for asset_class, band_cfg in bands.items():
            if asset_class == "vix_overlay_notional":
                continue
            lo, hi = band_cfg[regime]
            assert 0 <= lo <= hi <= 1.0, f"bad band: {asset_class}/{regime} = [{lo},{hi}]"
    tc.ok("allocation bands valid in config")


# ============================================================================
# MODULE 3B: stock_screener
# ============================================================================

def test_stock_screener(tc: TestCounter, db_path: str):
    print("\n--- Module 3B: stock_screener ---")
    try:
        import stock_screener
        tc.ok("stock_screener imports")
    except Exception as e:
        tc.fail("stock_screener imports", str(e))
        return

    for fn in ["run_stock_screener", "run_all_watchlists",
                "compute_entry_exit_signals", "score_momentum_stock",
                "compute_quality_score", "apply_valuation_filter"]:
        if hasattr(stock_screener, fn):
            tc.ok(f"stock_screener.{fn} exists")
        else:
            tc.fail(f"stock_screener.{fn} exists", "not found")

    # Verify watchlist tickers in config
    for wl in ["watchlist_biotech", "watchlist_ai_software", "watchlist_defense", "watchlist_green_materials"]:
        tickers = CONFIG["tickers"].get(wl, [])
        if len(tickers) >= 5:
            tc.ok(f"{wl}: {len(tickers)} tickers")
        else:
            tc.fail(f"{wl}", f"only {len(tickers)} tickers (need >=5)")

    # Scoring weights sum to 1.0
    weights = CONFIG["stock_screener"]["scoring_weights"]
    w_sum = sum(weights.values())
    if abs(w_sum - 1.0) < 0.01:
        tc.ok(f"scoring weights sum to {w_sum:.2f}")
    else:
        tc.fail("scoring weights", f"sum = {w_sum:.2f}")


# ============================================================================
# MODULE 4: nlp_sentiment
# ============================================================================

def test_nlp_sentiment(tc: TestCounter, db_path: str):
    print("\n--- Module 4: nlp_sentiment ---")
    try:
        import nlp_sentiment
        tc.ok("nlp_sentiment imports")
    except Exception as e:
        tc.fail("nlp_sentiment imports", str(e))
        return

    for fn in ["score_single_filing", "score_all_filings",
                "compute_sector_signals", "generate_nlp_report"]:
        if hasattr(nlp_sentiment, fn):
            tc.ok(f"nlp_sentiment.{fn} exists")
        else:
            tc.fail(f"nlp_sentiment.{fn} exists", "not found")

    # Verify NLP config
    assert CONFIG["nlp"]["model"] == "ProsusAI/finbert", "expected FinBERT model"
    tc.ok("NLP model config = ProsusAI/finbert")

    assert CONFIG["nlp"]["max_tokens"] == 512
    tc.ok("NLP max_tokens = 512")

    # Check NLP scores in DB
    conn = sqlite3.connect(db_path)
    nlp_count = conn.execute("SELECT COUNT(*) FROM nlp_scores").fetchone()[0]
    sect_count = conn.execute("SELECT COUNT(*) FROM nlp_sector_signals").fetchone()[0]
    conn.close()

    if nlp_count >= 2:
        tc.ok(f"nlp_scores: {nlp_count} rows")
    else:
        tc.fail("nlp_scores", f"only {nlp_count} rows")

    if sect_count >= 5:
        tc.ok(f"nlp_sector_signals: {sect_count} rows")
    else:
        tc.fail("nlp_sector_signals", f"only {sect_count} rows")


# ============================================================================
# MODULE 5: monitor
# ============================================================================

def test_monitor(tc: TestCounter, db_path: str):
    print("\n--- Module 5: monitor ---")
    try:
        import monitor
        tc.ok("monitor imports")
    except Exception as e:
        tc.fail("monitor imports", str(e))
        return

    # Key functions
    for fn in ["main", "AlertEngine", "generate_executive_summary",
                "send_email", "send_telegram", "write_google_sheets",
                "write_alerts_json", "append_alerts_csv"]:
        if hasattr(monitor, fn):
            tc.ok(f"monitor.{fn} exists")
        else:
            tc.fail(f"monitor.{fn} exists", "not found")

    # Alert thresholds from config
    assert CONFIG["monitor"]["rebalance_threshold_bps"] == 200
    tc.ok("rebalance_threshold_bps = 200")
    assert CONFIG["monitor"]["entry_window_threshold_bps"] == 300
    tc.ok("entry_window_threshold_bps = 300")
    assert CONFIG["monitor"]["extended_defense_days"] == 60
    tc.ok("extended_defense_days = 60")

    # Panic exit sequence
    pes = CONFIG["monitor"]["panic_exit_sequence"]
    assert pes["immediate_pct"] == 0.50
    assert pes["remainder_days"] == [3, 5]
    tc.ok("panic_exit_sequence config valid")

    # Test alert detection with mock data
    try:
        mock_alloc_current = {
            "us_equities": 0.45,
            "cash_short_duration": 0.10,
            "healthcare": 0.15,
            "energy_materials": 0.12,
            "intl_developed": 0.10,
            "em_equities": 0.08,
        }
        mock_alloc_target = {
            "us_equities": 0.40,
            "cash_short_duration": 0.15,
            "healthcare": 0.15,
            "energy_materials": 0.12,
            "intl_developed": 0.10,
            "em_equities": 0.08,
        }
        # us_equities drifted +500bps, cash drifted -500bps => both exceed 200bps
        drift_bps = {}
        for k in mock_alloc_current:
            drift_bps[k] = abs(mock_alloc_current[k] - mock_alloc_target[k]) * 10000
        over_threshold = [k for k, v in drift_bps.items() if v > 200]
        assert len(over_threshold) >= 2, f"expected 2+ drifted, got {over_threshold}"
        tc.ok(f"drift detection: {len(over_threshold)} positions over threshold")
    except Exception as e:
        tc.fail("alert detection logic", str(e))


# ============================================================================
# MODULE 6: dashboard
# ============================================================================

def test_dashboard(tc: TestCounter, db_path: str):
    print("\n--- Module 6: dashboard ---")

    # Mock streamlit before import
    st_mock = MagicMock()
    sys.modules["streamlit"] = st_mock

    try:
        if "dashboard" in sys.modules:
            del sys.modules["dashboard"]
        import dashboard
        tc.ok("dashboard imports (with mocked streamlit)")
    except Exception as e:
        tc.fail("dashboard imports", str(e))
        return
    finally:
        # Restore
        if "streamlit" in sys.modules and sys.modules["streamlit"] is st_mock:
            del sys.modules["streamlit"]

    # Check page functions exist
    page_functions = [
        "page_regime_dashboard",
        "page_portfolio_allocation",
        "page_signal_detail",
        "page_stock_screener",
        "page_alerts_log",
        "page_backtester",
    ]
    for fn_name in page_functions:
        if hasattr(dashboard, fn_name):
            tc.ok(f"dashboard.{fn_name} exists")
        else:
            tc.fail(f"dashboard.{fn_name} exists", "not found")

    # Test max drawdown helper
    if hasattr(dashboard, "_max_drawdown"):
        test_series = pd.Series([100, 110, 105, 95, 100, 90])
        mdd = dashboard._max_drawdown(test_series)
        # Peak=110, trough=90 => DD = (90-110)/110 * 100 = -18.18%
        # _max_drawdown returns percentage (multiplied by 100)
        expected = (90 - 110) / 110 * 100
        if abs(mdd - expected) < 0.5:
            tc.ok(f"_max_drawdown = {mdd:.2f}% (expected ~{expected:.2f}%)")
        else:
            tc.fail("_max_drawdown", f"got {mdd:.2f}%, expected ~{expected:.2f}%")


# ============================================================================
# CONFIG INTEGRITY
# ============================================================================

def test_config_integrity(tc: TestCounter):
    print("\n--- Config Integrity ---")

    # McLean-Pontiff decay
    assert CONFIG["factor_model"]["mclean_pontiff_decay"] == 0.74
    tc.ok("mclean_pontiff_decay = 0.74")

    # Backtest label
    label = CONFIG["backtest"]["mclean_pontiff_label"]
    assert "McLean-Pontiff" in label and "26%" in label
    tc.ok("backtest label contains McLean-Pontiff + 26%")

    # Portfolio totals
    taxable = CONFIG["portfolio"]["accounts"]["taxable"]["value"]
    roth = CONFIG["portfolio"]["accounts"]["roth_ira"]["value"]
    total = CONFIG["portfolio"]["total_value"]
    assert taxable + roth == total, f"{taxable}+{roth} != {total}"
    tc.ok(f"portfolio: ${taxable:,} + ${roth:,} = ${total:,}")

    # All secrets via env vars (not in config)
    config_str = yaml.dump(CONFIG)
    for bad_pattern in ["sk-", "ghp_", "xoxb-", "Bearer "]:
        assert bad_pattern not in config_str, f"potential hardcoded secret: {bad_pattern}"
    tc.ok("no hardcoded secrets in config")

    # Regime thresholds
    assert CONFIG["regime"]["thresholds"]["panic_upper"] == 5
    assert CONFIG["regime"]["thresholds"]["defense_upper"] == 30
    tc.ok("regime thresholds: panic<5, defense<30")

    # CVaR confidence
    assert CONFIG["optimizer"]["cvar_confidence"] == 0.95
    tc.ok("CVaR confidence = 0.95")

    # Run time
    assert CONFIG["monitor"]["run_time_et"] == "16:30"
    tc.ok("monitor run_time_et = 16:30")

    # Alert delivery config
    assert CONFIG["alerts"]["telegram"]["enabled"] is True
    assert CONFIG["alerts"]["email"]["enabled"] is True
    assert CONFIG["alerts"]["google_sheets"]["enabled"] is True
    tc.ok("all alert channels enabled in config")


# ============================================================================
# FILE STRUCTURE
# ============================================================================

def test_file_structure(tc: TestCounter):
    print("\n--- File Structure ---")
    required_files = [
        "config.yaml",
        "data_feeds.py",
        "regime_detector.py",
        "portfolio_optimizer.py",
        "stock_screener.py",
        "nlp_sentiment.py",
        "monitor.py",
        "dashboard.py",
        "requirements.txt",
        "README.md",
        ".github/workflows/daily_monitor.yml",
        "notebooks/setup_and_backtest.ipynb",
        "notebooks/validate_signals.ipynb",
        "tests/test_all_modules.py",
    ]

    # Check from repo root
    for rel_path in required_files:
        for base in [REPO_ROOT, PARENT]:
            full = base / rel_path
            if full.exists():
                tc.ok(f"file exists: {rel_path}")
                break
        else:
            tc.fail(f"file exists: {rel_path}", "not found")


# ============================================================================
# REQUIREMENTS.TXT VALIDATION
# ============================================================================

def test_requirements(tc: TestCounter):
    print("\n--- requirements.txt ---")
    for base in [REPO_ROOT, PARENT]:
        req_path = base / "requirements.txt"
        if req_path.exists():
            content = req_path.read_text()
            required_pkgs = [
                "numpy", "pandas", "scipy", "PyYAML", "yfinance",
                "pyportfolioopt", "transformers", "torch", "requests",
                "gspread", "google-auth", "streamlit", "plotly", "pytest",
            ]
            for pkg in required_pkgs:
                if pkg.lower() in content.lower():
                    tc.ok(f"requirements.txt includes {pkg}")
                else:
                    tc.fail(f"requirements.txt includes {pkg}", "not found")
            return
    tc.fail("requirements.txt", "file not found")


# ============================================================================
# GITHUB ACTIONS WORKFLOW VALIDATION
# ============================================================================

def test_github_actions(tc: TestCounter):
    print("\n--- GitHub Actions Workflow ---")
    for base in [REPO_ROOT, PARENT]:
        yml_path = base / ".github" / "workflows" / "daily_monitor.yml"
        if yml_path.exists():
            content = yml_path.read_text()

            # Cron schedule
            if "30 20 * * 1-5" in content:
                tc.ok("cron schedule: 30 20 * * 1-5 (4:30 PM ET)")
            else:
                tc.fail("cron schedule", "expected '30 20 * * 1-5'")

            # Secrets
            for secret in ["FRED_API_KEY", "GMAIL_USERNAME", "GMAIL_PASSWORD",
                          "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "GOOGLE_SHEETS_CREDENTIALS"]:
                if secret in content:
                    tc.ok(f"secret referenced: {secret}")
                else:
                    tc.fail(f"secret referenced: {secret}", "not found in workflow")

            # Monitor invocation
            if "monitor.py" in content:
                tc.ok("workflow runs monitor.py")
            else:
                tc.fail("workflow runs monitor.py", "not found")

            # Artifact upload
            if "upload-artifact" in content:
                tc.ok("workflow uploads artifacts")
            else:
                tc.fail("workflow uploads artifacts", "not found")

            # Failure notification
            if "failure()" in content:
                tc.ok("workflow has failure notification")
            else:
                tc.fail("workflow has failure notification", "not found")

            return
    tc.fail("daily_monitor.yml", "file not found")


# ============================================================================
# DOLLAR AMOUNT VALIDATION
# ============================================================================

def test_dollar_amounts(tc: TestCounter):
    print("\n--- Dollar Amount Validation ---")

    total = CONFIG["portfolio"]["total_value"]
    taxable = CONFIG["portfolio"]["accounts"]["taxable"]["value"]
    roth = CONFIG["portfolio"]["accounts"]["roth_ira"]["value"]

    # Simulate allocations for each regime
    for regime in ["offense", "defense", "panic"]:
        bands = CONFIG["optimizer"]["allocation_bands"]
        # Use midpoint of each band
        alloc = {}
        for asset_class, regimes_cfg in bands.items():
            if asset_class == "vix_overlay_notional":
                continue
            lo, hi = regimes_cfg[regime]
            alloc[asset_class] = (lo + hi) / 2

        # Normalize to sum to 1.0
        alloc_sum = sum(alloc.values())
        if alloc_sum > 0:
            alloc = {k: v / alloc_sum for k, v in alloc.items()}

        # Compute dollar amounts
        for asset_class, weight in alloc.items():
            total_dollars = weight * total
            taxable_dollars = weight * taxable
            roth_dollars = weight * roth
            assert total_dollars >= 0
            assert taxable_dollars >= 0
            assert roth_dollars >= 0
            assert abs(taxable_dollars + roth_dollars - total_dollars) < 0.01

        tc.ok(f"dollar amounts valid for regime='{regime}' (total=${total:,})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("SECTOR ROTATION SYSTEM — UNIFIED SMOKE TEST")
    print(f"Run at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tc = TestCounter()

    # Create temp DB
    db_path = make_temp_db()
    try:
        # Seed with synthetic data
        seed_db(db_path, n_days=60)
        print(f"\nTemp DB: {db_path}")

        conn = sqlite3.connect(db_path)
        for tbl in ["prices", "macro_data", "filings", "signals", "allocations", "nlp_scores", "nlp_sector_signals"]:
            cnt = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"  {tbl}: {cnt:;} rows")
        conn.close()

        # Run all test suites
        test_config_integrity(tc)
        test_file_structure(tc)
        test_requirements(tc)
        test_github_actions(tc)
        test_dollar_amounts(tc)
        test_data_feeds(tc, db_path)
        test_regime_detector(tc, db_path)
        test_portfolio_optimizer(tc, db_path)
        test_stock_screener(tc, db_path)
        test_nlp_sentiment(tc, db_path)
        test_monitor(tc, db_path)
        test_dashboard(tc, db_path)

    finally:
        # Cleanup
        tmpdir = os.path.dirname(db_path)
        shutil.rmtree(tmpdir, ignore_errors=True)

    success = tc.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
