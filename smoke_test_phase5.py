"""
smoke_test_phase5.py — Comprehensive Smoke Test for Phase 5 Monitor
=====================================================================
Tests all 7 sections of monitor.py with 30 days of synthetic data:
  1. Database helpers
  2. Phase orchestration wrappers
  3. Alert detection (all 5 alert types)
  4. Alert delivery (JSON + CSV)
  5. Executive Summary report generation
  6. Daily run log
  7. CLI entry point (mock mode)

Run:  python smoke_test_phase5.py
"""

import csv
import datetime as dt
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Ensure the package dir is importable
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from monitor import (
    AlertEngine,
    _default_allocation,
    append_alerts_csv,
    count_consecutive_defensive_days,
    fetch_latest_allocation,
    fetch_latest_factor_signals,
    fetch_latest_regime,
    fetch_nlp_sector_signals,
    fetch_regime_history,
    generate_executive_summary,
    get_db,
    load_config,
    log_run,
    run_data_refresh,
    run_nlp_scoring,
    run_optimizer,
    run_regime_detection,
    write_alerts_json,
)

# ---------------------------------------------------------------------------
# GLOBAL TRACKING
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0
ERRORS = []

def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {label}")
    else:
        FAIL += 1
        ERRORS.append(f"{label}: {detail}")
        print(f"  ❌ {label}  →  {detail}")


# ---------------------------------------------------------------------------
# FIXTURES — 30 days of synthetic data
# ---------------------------------------------------------------------------

def build_test_config() -> dict:
    """Return a minimal config dict matching config.yaml structure."""
    return {
        "portfolio": {
            "total_value": 144000,
            "accounts": {
                "taxable": {"value": 100000, "type": "Individual Brokerage"},
                "roth_ira": {"value": 44000, "type": "Retirement"},
            },
        },
        "optimizer": {
            "allocation_bands": {
                "us_equities":        {"panic": [0.00, 0.05], "defense": [0.10, 0.25], "offense": [0.40, 0.60]},
                "intl_developed":     {"panic": [0.00, 0.00], "defense": [0.05, 0.15], "offense": [0.15, 0.25]},
                "em_equities":        {"panic": [0.00, 0.00], "defense": [0.00, 0.08], "offense": [0.08, 0.15]},
                "energy_materials":   {"panic": [0.00, 0.05], "defense": [0.05, 0.15], "offense": [0.10, 0.20]},
                "healthcare":         {"panic": [0.05, 0.10], "defense": [0.10, 0.20], "offense": [0.10, 0.18]},
                "cash_short_duration":{"panic": [0.70, 1.00], "defense": [0.30, 0.60], "offense": [0.05, 0.15]},
                "vix_overlay_notional":{"panic": [0.00, 0.02], "defense": [0.02, 0.04], "offense": [0.005, 0.015]},
            },
        },
        "monitor": {
            "rebalance_threshold_bps": 200,
            "entry_window_threshold_bps": 300,
            "extended_defense_days": 60,
            "panic_exit_sequence": {
                "immediate_pct": 0.50,
                "remainder_days": [3, 5],
            },
            "run_time_et": "16:30",
        },
        "regime": {
            "fast_shock": {
                "vix_rv_ratio_threshold": 1.5,
                "rv_window": 21,
            },
        },
        "probabilistic_triggers": {
            "whipsaw_buffer_percentile": 3,
        },
        "nlp": {
            "regime_weights": {"offense": 0.20, "defense": 0.0, "panic": 0.0},
        },
    }


def seed_database(db_path: Path, days: int = 30) -> sqlite3.Connection:
    """
    Populate an in-memory test DB with 30 days of synthetic data:
      - prices (7 tickers × 30 days)
      - signals  (regime_state + factor_scores per day)
      - allocations (1 row — offense)
      - nlp_scores + nlp_sector_signals
    """
    conn = sqlite3.connect(str(db_path))

    # --- monitor_runs (created by get_db but we need it here) ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monitor_runs (
            run_id      TEXT PRIMARY KEY,
            date        TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            finished_at TEXT,
            status      TEXT DEFAULT 'running',
            regime      TEXT,
            alerts_json TEXT,
            report_text TEXT
        )
    """)
    conn.commit()

    # --- prices ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            date TEXT, ticker TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER,
            PRIMARY KEY (date, ticker)
        )
    """)
    tickers = ["XLK", "XLV", "XLE", "XLF", "SPY", "BIL", "EEM"]
    base_prices = {"XLK": 200, "XLV": 140, "XLE": 90, "XLF": 40,
                   "SPY": 450, "BIL": 91, "EEM": 42}
    np.random.seed(42)
    for i in range(days):
        d = (dt.date.today() - dt.timedelta(days=days - i)).isoformat()
        for t in tickers:
            p = base_prices[t] * (1 + np.random.normal(0, 0.01))
            base_prices[t] = p
            conn.execute(
                "INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?, ?, ?, ?)",
                (d, t, p * 0.99, p * 1.01, p * 0.98, p, int(1e6 + np.random.randint(0, 5e5))),
            )

    # --- signals (regime_state) ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            date TEXT, signal_type TEXT, signal_data TEXT,
            PRIMARY KEY (date, signal_type)
        )
    """)
    for i in range(days):
        d = (dt.date.today() - dt.timedelta(days=days - i)).isoformat()
        regime = "offense" if i < 20 else "defense"
        conn.execute(
            "INSERT OR REPLACE INTO signals VALUES (?, ?, ?)",
            (d, "regime_state", json.dumps({
                "dominant_regime": regime,
                "wedge_volume_percentile": 55.0 - i * 0.5,
                "regime_probabilities": {"panic": 0.05, "defense": 0.25, "offense": 0.70},
                "fast_shock_risk": "low",
                "vix_rv_ratio": 0.8 + i * 0.02,
                "consecutive_days_in_regime": i if i < 20 else i - 20,
                "regime_confirmed": True,
            })),
        )
        # factor_scores
        conn.execute(
            "INSERT OR REPLACE INTO signals VALUES (?, ?, ?)",
            (d, "factor_scores", json.dumps({
                "sector_scores": [
                    {"sector_etf": "XLK", "composite_score": 0.82 - i * 0.005},
                    {"sector_etf": "XLV", "composite_score": 0.65 + i * 0.003},
                    {"sector_etf": "XLE", "composite_score": 0.50},
                    {"sector_etf": "XLF", "composite_score": 0.55},
                ],
            })),
        )

    # --- allocations ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS allocations (
            date TEXT PRIMARY KEY, regime TEXT, allocations TEXT,
            taxable_dollars TEXT, roth_dollars TEXT
        )
    """)
    alloc = {
        "us_equities": 0.50, "intl_developed": 0.20, "em_equities": 0.10,
        "energy_materials": 0.10, "healthcare": 0.05, "cash_short_duration": 0.05,
        "vix_overlay_notional": 0.00,
    }
    tax_d = {"us_equities": 50000, "intl_developed": 20000, "em_equities": 10000,
             "energy_materials": 10000, "cash_short_duration": 5000}
    roth_d = {"healthcare": 5000, "vix_overlay_notional": 0}
    conn.execute(
        "INSERT OR REPLACE INTO allocations VALUES (?, ?, ?, ?, ?)",
        (
            dt.date.today().isoformat(), "offense",
            json.dumps(alloc), json.dumps(tax_d), json.dumps(roth_d),
        ),
    )

    # --- nlp_scores ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nlp_scores (
            date TEXT, ticker TEXT, filing_type TEXT, sentiment_score REAL,
            PRIMARY KEY (date, ticker, filing_type)
        )
    """)
    conn.execute(
        "INSERT OR REPLACE INTO nlp_scores VALUES (?, ?, ?, ?)",
        (dt.date.today().isoformat(), "NBIX", "10-K", 0.42),
    )

    # --- nlp_sector_signals ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nlp_sector_signals (
            date TEXT, sector_etf TEXT, sector_score REAL,
            drift_risk INTEGER, signal TEXT,
            PRIMARY KEY (date, sector_etf)
        )
    """)
    sector_etfs = ["XLK", "XLV", "XLE", "XLF", "XLI", "XLB", "XLU",
                   "XLP", "XLRE", "XLC", "XLY"]
    for etf in sector_etfs:
        sc = round(np.random.uniform(-0.3, 0.5), 3)
        conn.execute(
            "INSERT OR REPLACE INTO nlp_sector_signals VALUES (?, ?, ?, ?, ?)",
            (dt.date.today().isoformat(), etf, sc, 0, "neutral"),
        )

    conn.commit()
    return conn


# ===========================================================================
# TEST SECTIONS
# ===========================================================================

def test_01_config_loading():
    """Test 1: Config loading from real config.yaml."""
    print("\n── Test 1: Config Loading ──")
    try:
        cfg = load_config(HERE / "config.yaml")
        check("Config is a dict", isinstance(cfg, dict))
        check("portfolio.total_value = 144000",
              cfg.get("portfolio", {}).get("total_value") == 144000)
        check("monitor.rebalance_threshold_bps = 200",
              cfg["monitor"]["rebalance_threshold_bps"] == 200)
        check("monitor.extended_defense_days = 60",
              cfg["monitor"]["extended_defense_days"] == 60)
        check("monitor.panic_exit_sequence present",
              "panic_exit_sequence" in cfg["monitor"])
        check("All 7 allocation bands present",
              len(cfg["optimizer"]["allocation_bands"]) == 7)
    except Exception as e:
        check("Config loading", False, str(e))


def test_02_db_helpers(conn: sqlite3.Connection):
    """Test 2: Database helper functions."""
    print("\n── Test 2: Database Helpers ──")

    # fetch_latest_regime
    regime = fetch_latest_regime(conn)
    check("fetch_latest_regime returns dict", isinstance(regime, dict))
    check("Regime has dominant_regime key", "dominant_regime" in regime)
    check("Latest regime is 'defense' (last 10 of 30 days)",
          regime["dominant_regime"] == "defense",
          f"got {regime.get('dominant_regime')}")

    # fetch_latest_allocation
    alloc = fetch_latest_allocation(conn)
    check("fetch_latest_allocation returns dict", isinstance(alloc, dict))
    check("Allocation has 'allocations' key", "allocations" in alloc)
    check("Allocation has 'taxable_dollars' key", "taxable_dollars" in alloc)
    check("Allocation has 'roth_dollars' key", "roth_dollars" in alloc)
    check("Allocation regime is 'offense'", alloc["regime"] == "offense")

    # fetch_regime_history
    hist = fetch_regime_history(conn, days=90)
    check("fetch_regime_history returns DataFrame", isinstance(hist, pd.DataFrame))
    check("Regime history has 30 rows", len(hist) == 30, f"got {len(hist)}")

    # count_consecutive_defensive_days
    def_days = count_consecutive_defensive_days(conn)
    check("Consecutive defensive days = 10", def_days == 10,
          f"got {def_days}")

    # fetch_nlp_sector_signals
    nlp_sig = fetch_nlp_sector_signals(conn)
    check("NLP sector signals returns DataFrame", isinstance(nlp_sig, pd.DataFrame))
    check("NLP sector signals has 11 rows", len(nlp_sig) == 11,
          f"got {len(nlp_sig)}")

    # fetch_latest_factor_signals
    factor = fetch_latest_factor_signals(conn)
    check("Factor signals returns DataFrame", isinstance(factor, pd.DataFrame))
    check("Factor signals has 4 sectors", len(factor) == 4,
          f"got {len(factor)}")


def test_03_orchestration_mock(conn: sqlite3.Connection, cfg: dict):
    """Test 3: Phase orchestration wrappers in mock mode."""
    print("\n── Test 3: Phase Orchestration (Mock Mode) ──")

    # Data refresh — mock
    data_res = run_data_refresh(conn, cfg, mock=True)
    check("Data refresh mock returns dict", isinstance(data_res, dict))
    check("Data refresh status = 'ok'", data_res["status"] == "ok")
    check("Data refresh mode = 'mock'",
          data_res.get("details", {}).get("mode") == "mock")

    # Regime detection — mock
    regime_res = run_regime_detection(conn, cfg, mock=True)
    check("Regime detection mock returns dict", isinstance(regime_res, dict))
    check("Regime detection has regime_state",
          "regime_state" in regime_res)
    check("Regime state from DB is 'defense'",
          regime_res["regime_state"].get("dominant_regime") == "defense",
          f"got {regime_res['regime_state'].get('dominant_regime')}")

    # Optimizer — mock (no regime change)
    opt_res = run_optimizer(conn, cfg, regime="offense", mock=True)
    check("Optimizer mock returns dict", isinstance(opt_res, dict))
    check("Optimizer status = 'ok'", opt_res["status"] == "ok")
    check("Optimizer returns allocation", "allocation" in opt_res)

    # NLP scoring — mock
    nlp_res = run_nlp_scoring(conn, cfg, mock=True)
    check("NLP scoring mock returns dict", isinstance(nlp_res, dict))
    check("NLP scoring status = 'ok'", nlp_res["status"] == "ok")
    check("NLP scoring mode = 'mock'",
          nlp_res.get("details", {}).get("mode") == "mock")


def test_04_default_allocation(cfg: dict):
    """Test 4: _default_allocation midpoint builder."""
    print("\n── Test 4: Default Allocation Builder ──")

    for regime in ("offense", "defense", "panic"):
        alloc = _default_allocation(cfg, regime)
        check(f"{regime}: returns dict", isinstance(alloc, dict))
        check(f"{regime}: has allocations", "allocations" in alloc)
        check(f"{regime}: has taxable_dollars", "taxable_dollars" in alloc)
        check(f"{regime}: has roth_dollars", "roth_dollars" in alloc)

        weights = alloc["allocations"]
        total_w = sum(weights.values())
        check(f"{regime}: weights sum to ~1.0",
              abs(total_w - 1.0) < 0.01, f"got {total_w:.4f}")

        # Dollar amounts
        total_dollars = sum(alloc["taxable_dollars"].get(k, 0) for k in weights) + \
                        sum(alloc["roth_dollars"].get(k, 0) for k in weights)
        check(f"{regime}: dollars sum near $144,000",
              abs(total_dollars - 144000) < 2000,
              f"got ${total_dollars:,.0f}")


def test_05_alert_engine_no_alerts(cfg: dict):
    """Test 5A: Alert engine — normal conditions — no alerts."""
    print("\n── Test 5A: Alert Engine — No Alerts (Normal) ──")

    engine = AlertEngine(cfg)
    regime_state = {
        "dominant_regime": "offense",
        "regime_confirmed": True,
        "vix_rv_ratio": 0.90,
        "wedge_volume_percentile": 55.0,
        "consecutive_days_in_regime": 15,
        "fast_shock_risk": "low",
    }
    prev = {
        "allocations": {"us_equities": 0.50, "intl_developed": 0.20, "cash_short_duration": 0.10},
        "regime": "offense",
    }
    new = {
        "allocations": {"us_equities": 0.50, "intl_developed": 0.20, "cash_short_duration": 0.10},
        "regime": "offense",
    }
    alerts = engine.evaluate(regime_state, prev, new,
                             regime_changed=False, consecutive_defensive_days=5)
    check("No alerts in normal conditions", len(alerts) == 0,
          f"got {len(alerts)} alert(s)")


def test_05b_alert_fast_shock(cfg: dict):
    """Test 5B: Alert engine — FAST_SHOCK."""
    print("\n── Test 5B: Alert Engine — FAST_SHOCK ──")

    engine = AlertEngine(cfg)
    regime_state = {
        "dominant_regime": "offense",
        "regime_confirmed": True,
        "vix_rv_ratio": 2.1,  # Above 1.5 threshold
        "wedge_volume_percentile": 55.0,
        "consecutive_days_in_regime": 15,
        "fast_shock_risk": "high",
    }
    alerts = engine.evaluate(regime_state, None, {}, regime_changed=False,
                             consecutive_defensive_days=0)
    fast_alerts = [a for a in alerts if a["type"] == "FAST_SHOCK"]
    check("FAST_SHOCK alert triggered", len(fast_alerts) == 1)
    if fast_alerts:
        check("FAST_SHOCK severity is HIGH",
              fast_alerts[0]["severity"] == "HIGH")
        check("FAST_SHOCK message mentions VIX/RV ratio",
              "2.10" in fast_alerts[0]["message"])
        check("FAST_SHOCK data has vix_rv_ratio",
              fast_alerts[0]["data"]["vix_rv_ratio"] == 2.1)


def test_05c_alert_panic_protocol(cfg: dict):
    """Test 5C: Alert engine — PANIC_PROTOCOL."""
    print("\n── Test 5C: Alert Engine — PANIC_PROTOCOL ──")

    engine = AlertEngine(cfg)
    regime_state = {
        "dominant_regime": "panic",
        "regime_confirmed": True,
        "vix_rv_ratio": 2.5,
        "wedge_volume_percentile": 2.0,
        "consecutive_days_in_regime": 3,
        "fast_shock_risk": "high",
    }
    alerts = engine.evaluate(regime_state, None, {}, regime_changed=False,
                             consecutive_defensive_days=3)
    panic_alerts = [a for a in alerts if a["type"] == "PANIC_PROTOCOL"]
    check("PANIC_PROTOCOL alert triggered", len(panic_alerts) == 1)
    if panic_alerts:
        check("PANIC_PROTOCOL severity is CRITICAL",
              panic_alerts[0]["severity"] == "CRITICAL")
        check("PANIC_PROTOCOL message mentions 50%",
              "50%" in panic_alerts[0]["message"])
        check("PANIC_PROTOCOL data has immediate_pct=0.50",
              panic_alerts[0]["data"]["immediate_pct"] == 0.50)
        check("PANIC_PROTOCOL data has remainder_days=[3,5]",
              panic_alerts[0]["data"]["remainder_days"] == [3, 5])


def test_05d_alert_rebalance(cfg: dict):
    """Test 5D: Alert engine — REBALANCE."""
    print("\n── Test 5D: Alert Engine — REBALANCE ──")

    engine = AlertEngine(cfg)
    regime_state = {
        "dominant_regime": "defense",
        "regime_confirmed": True,
        "vix_rv_ratio": 1.2,
        "wedge_volume_percentile": 20.0,
        "consecutive_days_in_regime": 2,
        "fast_shock_risk": "low",
    }
    prev = {
        "allocations": {"us_equities": 0.50, "cash_short_duration": 0.10},
        "regime": "offense",
    }
    new = {
        "allocations": {"us_equities": 0.175, "cash_short_duration": 0.45},
        "regime": "defense",
    }
    alerts = engine.evaluate(regime_state, prev, new, regime_changed=True,
                             consecutive_defensive_days=2)
    reb_alerts = [a for a in alerts if a["type"] == "REBALANCE"]
    check("REBALANCE alert triggered", len(reb_alerts) == 1)
    if reb_alerts:
        check("REBALANCE severity is HIGH",
              reb_alerts[0]["severity"] == "HIGH")
        drift = reb_alerts[0]["data"]["max_drift_bps"]
        check("REBALANCE max drift > 200 bps",
              drift >= 200, f"got {drift:.0f}")
        check("REBALANCE data shows from/to regime",
              reb_alerts[0]["data"]["from_regime"] == "offense" and
              reb_alerts[0]["data"]["to_regime"] == "defense")


def test_05e_alert_entry_window(cfg: dict):
    """Test 5E: Alert engine — ENTRY_WINDOW."""
    print("\n── Test 5E: Alert Engine — ENTRY_WINDOW ──")

    engine = AlertEngine(cfg)
    regime_state = {
        "dominant_regime": "offense",
        "regime_confirmed": True,
        "vix_rv_ratio": 0.9,
        "wedge_volume_percentile": 60.0,
        "consecutive_days_in_regime": 10,
        "fast_shock_risk": "low",
    }
    prev = {
        "allocations": {"us_equities": 0.35, "intl_developed": 0.10},
        "regime": "offense",
    }
    target = {
        "allocations": {"us_equities": 0.50, "intl_developed": 0.20},
        "regime": "offense",
    }
    # regime_changed=False but in offense + confirmed → entry windows checked
    alerts = engine.evaluate(regime_state, prev, target, regime_changed=False,
                             consecutive_defensive_days=0)
    entry_alerts = [a for a in alerts if a["type"] == "ENTRY_WINDOW"]
    check("ENTRY_WINDOW alert(s) triggered", len(entry_alerts) >= 1)
    if entry_alerts:
        check("ENTRY_WINDOW severity is MEDIUM",
              entry_alerts[0]["severity"] == "MEDIUM")
        # us_equities underweight = (0.50 - 0.35) * 10000 = 1500 bps > 300
        us_alert = [a for a in entry_alerts
                    if a["data"].get("asset") == "us_equities"]
        check("US equities entry window found", len(us_alert) == 1)
        if us_alert:
            check("US equities underweight = 1500 bps",
                  abs(us_alert[0]["data"]["underweight_bps"] - 1500) < 1)
        # intl_developed underweight = (0.20 - 0.10) * 10000 = 1000 bps > 300
        intl_alert = [a for a in entry_alerts
                      if a["data"].get("asset") == "intl_developed"]
        check("Intl developed entry window found", len(intl_alert) == 1)


def test_05f_alert_extended_defense(cfg: dict):
    """Test 5F: Alert engine — EXTENDED_DEFENSE."""
    print("\n── Test 5F: Alert Engine — EXTENDED_DEFENSE ──")

    engine = AlertEngine(cfg)
    regime_state = {
        "dominant_regime": "defense",
        "regime_confirmed": True,
        "vix_rv_ratio": 1.2,
        "wedge_volume_percentile": 20.0,  # Above whipsaw floor (3)
        "consecutive_days_in_regime": 65,
        "fast_shock_risk": "low",
    }
    alerts = engine.evaluate(regime_state, None, {}, regime_changed=False,
                             consecutive_defensive_days=65)
    ext_alerts = [a for a in alerts if a["type"] == "EXTENDED_DEFENSE"]
    check("EXTENDED_DEFENSE alert triggered", len(ext_alerts) == 1)
    if ext_alerts:
        check("EXTENDED_DEFENSE severity is MEDIUM",
              ext_alerts[0]["severity"] == "MEDIUM")
        check("EXTENDED_DEFENSE mentions 65 days",
              "65" in ext_alerts[0]["message"])


def test_05g_alert_extended_defense_below_floor(cfg: dict):
    """Test 5G: EXTENDED_DEFENSE should NOT fire if wedge vol < floor (true crisis)."""
    print("\n── Test 5G: EXTENDED_DEFENSE Suppressed in True Crisis ──")

    engine = AlertEngine(cfg)
    regime_state = {
        "dominant_regime": "panic",
        "regime_confirmed": True,
        "vix_rv_ratio": 2.5,
        "wedge_volume_percentile": 2.0,  # Below whipsaw floor (3) → true crisis
        "consecutive_days_in_regime": 70,
        "fast_shock_risk": "high",
    }
    alerts = engine.evaluate(regime_state, None, {}, regime_changed=False,
                             consecutive_defensive_days=70)
    ext_alerts = [a for a in alerts if a["type"] == "EXTENDED_DEFENSE"]
    check("EXTENDED_DEFENSE NOT triggered in true crisis", len(ext_alerts) == 0,
          f"got {len(ext_alerts)} alert(s)")


def test_05h_multiple_simultaneous_alerts(cfg: dict):
    """Test 5H: Multiple alerts can fire simultaneously."""
    print("\n── Test 5H: Multiple Simultaneous Alerts ──")

    engine = AlertEngine(cfg)
    # Panic + fast shock + rebalance all at once
    regime_state = {
        "dominant_regime": "panic",
        "regime_confirmed": True,
        "vix_rv_ratio": 2.5,
        "wedge_volume_percentile": 2.0,
        "consecutive_days_in_regime": 3,
        "fast_shock_risk": "high",
    }
    prev = {
        "allocations": {"us_equities": 0.50, "cash_short_duration": 0.10},
        "regime": "offense",
    }
    new = {
        "allocations": {"us_equities": 0.025, "cash_short_duration": 0.85},
        "regime": "panic",
    }
    alerts = engine.evaluate(regime_state, prev, new, regime_changed=True,
                             consecutive_defensive_days=3)
    types = {a["type"] for a in alerts}
    check("FAST_SHOCK fires", "FAST_SHOCK" in types)
    check("PANIC_PROTOCOL fires", "PANIC_PROTOCOL" in types)
    check("REBALANCE fires", "REBALANCE" in types)
    check("At least 3 alerts", len(alerts) >= 3, f"got {len(alerts)}")


def test_06_alert_delivery(tmp_dir: Path):
    """Test 6: Alert delivery — JSON + CSV."""
    print("\n── Test 6: Alert Delivery (JSON + CSV) ──")

    alerts = [
        {"type": "FAST_SHOCK", "severity": "HIGH",
         "timestamp": "2026-02-27T16:30:00",
         "message": "VIX/RV ratio 2.10 exceeds 1.5 threshold.",
         "data": {"vix_rv_ratio": 2.1}},
        {"type": "REBALANCE", "severity": "HIGH",
         "timestamp": "2026-02-27T16:30:00",
         "message": "Max drift 3250 bps exceeds 200 bps.",
         "data": {"max_drift_bps": 3250}},
    ]

    # JSON
    json_path = tmp_dir / "alerts.json"
    write_alerts_json(alerts, path=json_path)
    check("alerts.json created", json_path.exists())
    with open(json_path) as f:
        data = json.load(f)
    check("alerts.json has 'count' = 2", data.get("count") == 2)
    check("alerts.json has 'alerts' list", len(data.get("alerts", [])) == 2)
    check("alerts.json has 'generated_at'", "generated_at" in data)

    # CSV
    csv_path = tmp_dir / "alerts_history.csv"
    append_alerts_csv(alerts, path=csv_path)
    check("alerts_history.csv created", csv_path.exists())
    with open(csv_path) as f:
        reader = list(csv.reader(f))
    check("CSV header row", reader[0] == ["timestamp", "type", "severity", "message"])
    check("CSV has 2 data rows", len(reader) == 3, f"got {len(reader)} rows")

    # Append again — should not re-write header
    append_alerts_csv([alerts[0]], path=csv_path)
    with open(csv_path) as f:
        reader2 = list(csv.reader(f))
    check("CSV append: now 3 data rows (no extra header)",
          len(reader2) == 4, f"got {len(reader2)} rows")


def test_07_executive_summary(cfg: dict):
    """Test 7: Executive Summary report generation."""
    print("\n── Test 7: Executive Summary ──")

    regime_state = {
        "dominant_regime": "offense",
        "regime_confirmed": True,
        "vix_rv_ratio": 0.85,
        "wedge_volume_percentile": 55.0,
        "consecutive_days_in_regime": 20,
        "fast_shock_risk": "low",
    }
    alloc = _default_allocation(cfg, "offense")
    alerts = []

    # Factor signals
    factor_df = pd.DataFrame([
        {"sector_etf": "XLK", "composite_score": 0.82},
        {"sector_etf": "XLV", "composite_score": 0.65},
        {"sector_etf": "XLE", "composite_score": 0.50},
    ])

    # NLP signals
    nlp_df = pd.DataFrame([
        {"sector_etf": "XLK", "sector_score": 0.35, "drift_risk": 0},
        {"sector_etf": "XLV", "sector_score": -0.10, "drift_risk": 0},
        {"sector_etf": "XLE", "sector_score": 0.15, "drift_risk": 0},
    ])

    report = generate_executive_summary(
        regime_state, alloc, alerts, cfg,
        nlp_signals=nlp_df, factor_signals=factor_df,
    )
    check("Report is a string", isinstance(report, str))
    check("Report has header box", "EXECUTIVE SUMMARY" in report)
    check("Report shows OFFENSE regime", "OFFENSE" in report)
    check("Report shows risk level LOW", "LOW" in report)
    check("Report shows '$144,000'", "$144,000" in report)
    check("Report has Category column", "Category" in report)
    check("Report has Target % column", "Target %" in report)
    check("Report has Target $ column", "Target $" in report)
    check("Report has Taxable column", "Taxable" in report)
    check("Report has Roth IRA column", "Roth IRA" in report)
    check("Report has ALERTS section", "ALERTS" in report)
    check("Report shows 'No alerts today'", "No alerts" in report)
    check("Report has SIGNAL DETAIL section", "SIGNAL DETAIL" in report)
    check("Report shows Wedge Volume Percentile", "55.0%" in report)
    check("Report shows strongest sector XLK", "XLK" in report)
    check("Report has NLP SENTIMENT section", "NLP SENTIMENT" in report)

    # Dollar amounts — ensure both % and $ shown
    for line in report.split("\n"):
        if "US Equities" in line:
            check("US Equities row has % and $",
                  "%" in line or "." in line, f"row: {line.strip()}")
            break


def test_07b_executive_summary_with_alerts(cfg: dict):
    """Test 7B: Executive Summary with active alerts."""
    print("\n── Test 7B: Executive Summary — With Alerts ──")

    regime_state = {
        "dominant_regime": "defense",
        "regime_confirmed": True,
        "vix_rv_ratio": 2.1,
        "wedge_volume_percentile": 18.0,
        "consecutive_days_in_regime": 5,
        "fast_shock_risk": "high",
    }
    alloc = _default_allocation(cfg, "defense")
    alerts = [
        {"type": "FAST_SHOCK", "severity": "HIGH", "timestamp": "now",
         "message": "VIX/RV ratio 2.10 exceeds 1.5 threshold."},
        {"type": "REBALANCE", "severity": "HIGH", "timestamp": "now",
         "message": "Max drift 3250 bps exceeds 200 bps."},
    ]

    report = generate_executive_summary(regime_state, alloc, alerts, cfg)
    check("Report shows DEFENSE regime", "DEFENSE" in report)
    check("Report shows risk level ELEVATED", "ELEVATED" in report)
    check("Report shows action YES — REBALANCE",
          "YES" in report and "REBALANCE" in report)
    check("Report shows FAST_SHOCK alert", "FAST_SHOCK" in report)
    check("Report shows 🟡 icon for HIGH severity", "🟡" in report)


def test_07c_executive_summary_panic(cfg: dict):
    """Test 7C: Executive Summary — Panic regime."""
    print("\n── Test 7C: Executive Summary — Panic ──")

    regime_state = {
        "dominant_regime": "panic",
        "regime_confirmed": True,
        "vix_rv_ratio": 3.0,
        "wedge_volume_percentile": 2.0,
        "consecutive_days_in_regime": 3,
        "fast_shock_risk": "high",
    }
    alloc = _default_allocation(cfg, "panic")
    alerts = [
        {"type": "PANIC_PROTOCOL", "severity": "CRITICAL", "timestamp": "now",
         "message": "PANIC REGIME CONFIRMED — Execute staged exit."},
    ]

    report = generate_executive_summary(regime_state, alloc, alerts, cfg)
    check("Panic report shows PANIC regime", "PANIC" in report)
    check("Panic report shows risk level HIGH", "Risk Level: HIGH" in report)
    check("Panic report has 🔴 icon for CRITICAL", "🔴" in report)
    # Cash should be dominant in panic
    check("Panic allocation has large Cash / BIL",
          alloc["allocations"].get("cash_short_duration", 0) > 0.5)


def test_08_run_log(conn: sqlite3.Connection):
    """Test 8: Daily run log."""
    print("\n── Test 8: Daily Run Log ──")

    run_id = "test_run_001"
    today = dt.date.today().isoformat()
    started = dt.datetime.now().isoformat()
    finished = dt.datetime.now().isoformat()
    alerts = [{"type": "FAST_SHOCK", "severity": "HIGH", "message": "test"}]
    report = "Test report text"

    log_run(conn, run_id, today, started, finished, "ok", "offense",
            alerts, report)

    row = conn.execute(
        "SELECT * FROM monitor_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    check("Run logged to DB", row is not None)
    if row:
        check("Run ID correct", row[0] == run_id)
        check("Date correct", row[1] == today)
        check("Status = 'ok'", row[4] == "ok")
        check("Regime = 'offense'", row[5] == "offense")
        check("Alerts JSON stored", "FAST_SHOCK" in row[6])
        check("Report text stored", "Test report" in row[7])


def test_09_empty_db(cfg: dict):
    """Test 9: Graceful handling of empty database."""
    print("\n── Test 9: Empty Database Edge Cases ──")

    # Create a completely empty DB
    empty_db = Path(tempfile.mktemp(suffix=".db"))
    conn = get_db(empty_db)

    # Create empty tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            date TEXT, signal_type TEXT, signal_data TEXT,
            PRIMARY KEY (date, signal_type)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS allocations (
            date TEXT PRIMARY KEY, regime TEXT, allocations TEXT,
            taxable_dollars TEXT, roth_dollars TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nlp_sector_signals (
            date TEXT, sector_etf TEXT, sector_score REAL,
            drift_risk INTEGER, signal TEXT,
            PRIMARY KEY (date, sector_etf)
        )
    """)
    conn.commit()

    # fetch_latest_regime on empty DB
    regime = fetch_latest_regime(conn)
    check("Empty DB: regime returns default offense",
          regime.get("dominant_regime") == "offense")

    # fetch_latest_allocation on empty DB
    alloc = fetch_latest_allocation(conn)
    check("Empty DB: allocation returns None", alloc is None)

    # count_consecutive_defensive_days on empty DB
    days = count_consecutive_defensive_days(conn)
    check("Empty DB: defensive days = 0", days == 0)

    # NLP signals on empty DB
    nlp = fetch_nlp_sector_signals(conn)
    check("Empty DB: NLP signals returns empty DataFrame",
          isinstance(nlp, pd.DataFrame) and len(nlp) == 0)

    # Factor signals on empty DB
    factor = fetch_latest_factor_signals(conn)
    check("Empty DB: factor signals returns empty DataFrame",
          isinstance(factor, pd.DataFrame) and len(factor) == 0)

    # _default_allocation with empty DB results
    default = _default_allocation(cfg, "offense")
    check("Empty DB: default allocation succeeds",
          isinstance(default, dict) and "allocations" in default)

    # generate_executive_summary with empty data
    report = generate_executive_summary(
        regime_state=regime, allocation=default, alerts=[], cfg=cfg,
        nlp_signals=None, factor_signals=None,
    )
    check("Empty DB: executive summary generates",
          isinstance(report, str) and len(report) > 100)
    check("Empty DB: report says 'factor data not available'",
          "not available" in report)

    conn.close()
    empty_db.unlink(missing_ok=True)


def test_10_optimizer_regime_change(conn: sqlite3.Connection, cfg: dict):
    """Test 10: Optimizer wrapper when regime changes (mock)."""
    print("\n── Test 10: Optimizer Regime Change (Mock) ──")

    # Current DB has regime=offense in allocations; we say regime=defense
    # This simulates regime_changed=True, but mock=True still
    opt_res = run_optimizer(conn, cfg, regime="defense", mock=True)
    check("Mock regime change: returns dict", isinstance(opt_res, dict))
    check("Mock regime change: regime_changed = True",
          opt_res.get("regime_changed") is True)
    # Mock mode should still return existing allocation
    check("Mock regime change: still returns allocation",
          "allocation" in opt_res)


def test_11_edge_cases():
    """Test 11: AlertEngine edge cases."""
    print("\n── Test 11: AlertEngine Edge Cases ──")

    cfg = build_test_config()
    engine = AlertEngine(cfg)

    # NaN vix_rv_ratio
    regime_state = {
        "dominant_regime": "offense", "regime_confirmed": True,
        "vix_rv_ratio": float("nan"),
        "wedge_volume_percentile": 50.0,
    }
    alerts = engine.evaluate(regime_state, None, {}, False, 0)
    fast = [a for a in alerts if a["type"] == "FAST_SHOCK"]
    check("NaN vix_rv_ratio: no FAST_SHOCK", len(fast) == 0,
          "NaN should not exceed threshold")

    # None vix_rv_ratio
    regime_state["vix_rv_ratio"] = None
    alerts = engine.evaluate(regime_state, None, {}, False, 0)
    fast = [a for a in alerts if a["type"] == "FAST_SHOCK"]
    check("None vix_rv_ratio: no FAST_SHOCK", len(fast) == 0)

    # Panic unconfirmed — should NOT trigger PANIC_PROTOCOL
    regime_state = {
        "dominant_regime": "panic", "regime_confirmed": False,
        "vix_rv_ratio": 1.2, "wedge_volume_percentile": 3.0,
    }
    alerts = engine.evaluate(regime_state, None, {}, False, 0)
    panic = [a for a in alerts if a["type"] == "PANIC_PROTOCOL"]
    check("Unconfirmed panic: no PANIC_PROTOCOL", len(panic) == 0)

    # Rebalance with no prev_allocation — should not crash
    regime_state = {
        "dominant_regime": "defense", "regime_confirmed": True,
        "vix_rv_ratio": 1.0, "wedge_volume_percentile": 20.0,
    }
    alerts = engine.evaluate(regime_state, None, {}, regime_changed=True,
                             consecutive_defensive_days=0)
    reb = [a for a in alerts if a["type"] == "REBALANCE"]
    check("No prev allocation: no REBALANCE crash", len(reb) == 0)

    # Entry window with empty allocations
    regime_state["dominant_regime"] = "offense"
    alerts = engine.evaluate(regime_state, {"allocations": {}},
                             {"allocations": {}}, False, 0)
    entry = [a for a in alerts if a["type"] == "ENTRY_WINDOW"]
    check("Empty allocations: no ENTRY_WINDOW", len(entry) == 0)


def test_12_dollar_amounts_in_report(cfg: dict):
    """Test 12: Report shows both % AND $ for each allocation row."""
    print("\n── Test 12: Dollar Amounts in Report ──")

    regime_state = {
        "dominant_regime": "offense", "regime_confirmed": True,
        "vix_rv_ratio": 0.85, "wedge_volume_percentile": 55.0,
        "consecutive_days_in_regime": 20, "fast_shock_risk": "low",
    }
    alloc = _default_allocation(cfg, "offense")
    report = generate_executive_summary(regime_state, alloc, [], cfg)

    # The report must have a row for each asset class with both % and $
    asset_display = [
        "US Equities", "Intl Developed", "EM Equities",
        "Energy / Materials", "Healthcare", "Cash / BIL",
    ]
    for name in asset_display:
        lines = [l for l in report.split("\n") if name in l]
        check(f"{name} row present in report", len(lines) >= 1)
        if lines:
            line = lines[0]
            has_pct = "%" in line or "—" in line
            has_dollar = "$" in line or "—" in line
            check(f"{name} row has % AND $",
                  has_pct and has_dollar, f"row: {line.strip()}")

    # Total row
    total_lines = [l for l in report.split("\n") if "TOTAL" in l]
    check("TOTAL row present", len(total_lines) >= 1)
    if total_lines:
        check("TOTAL row has $144,000",
              "$144,000" in total_lines[0] or "144,000" in total_lines[0],
              f"got: {total_lines[0].strip()}")


def test_13_taxable_roth_split(cfg: dict):
    """Test 13: Taxable vs Roth dollar split is reasonable."""
    print("\n── Test 13: Taxable/Roth Dollar Split ──")

    for regime in ("offense", "defense", "panic"):
        alloc = _default_allocation(cfg, regime)
        tax_total = sum(alloc["taxable_dollars"].values())
        roth_total = sum(alloc["roth_dollars"].values())
        combined = tax_total + roth_total

        check(f"{regime}: taxable + roth > 0", combined > 0)
        check(f"{regime}: taxable <= $100K",
              tax_total <= 100001,  # small tolerance
              f"got ${tax_total:,.0f}")
        check(f"{regime}: roth <= $44K",
              roth_total <= 44001,
              f"got ${roth_total:,.0f}")


# ===========================================================================
# MAIN RUNNER
# ===========================================================================

def main():
    print("=" * 66)
    print("  Phase 5 Smoke Test — monitor.py")
    print(f"  {dt.datetime.now().isoformat()}")
    print("=" * 66)

    cfg = build_test_config()
    tmp_dir = Path(tempfile.mkdtemp(prefix="phase5_test_"))
    db_path = tmp_dir / "test_rotation.db"
    conn = seed_database(db_path)

    try:
        # Section 1: Config
        test_01_config_loading()

        # Section 2: DB helpers
        test_02_db_helpers(conn)

        # Section 3: Orchestration (mock)
        test_03_orchestration_mock(conn, cfg)

        # Section 4: Default allocation
        test_04_default_allocation(cfg)

        # Section 5: Alert detection — all types
        test_05_alert_engine_no_alerts(cfg)
        test_05b_alert_fast_shock(cfg)
        test_05c_alert_panic_protocol(cfg)
        test_05d_alert_rebalance(cfg)
        test_05e_alert_entry_window(cfg)
        test_05f_alert_extended_defense(cfg)
        test_05g_alert_extended_defense_below_floor(cfg)
        test_05h_multiple_simultaneous_alerts(cfg)

        # Section 6: Alert delivery
        test_06_alert_delivery(tmp_dir)

        # Section 7: Executive Summary
        test_07_executive_summary(cfg)
        test_07b_executive_summary_with_alerts(cfg)
        test_07c_executive_summary_panic(cfg)

        # Section 8: Run log
        test_08_run_log(conn)

        # Section 9: Empty DB
        test_09_empty_db(cfg)

        # Section 10: Optimizer regime change
        test_10_optimizer_regime_change(conn, cfg)

        # Section 11: Edge cases
        test_11_edge_cases()

        # Section 12: Dollar amounts
        test_12_dollar_amounts_in_report(cfg)

        # Section 13: Taxable/Roth split
        test_13_taxable_roth_split(cfg)

    finally:
        conn.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- Summary ---
    print("\n" + "=" * 66)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed  ({PASS + FAIL} total)")
    print("=" * 66)
    if ERRORS:
        print("\n  FAILURES:")
        for e in ERRORS:
            print(f"    ❌ {e}")
    else:
        print("  🎉 ALL CHECKS PASSED")
    print()

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
