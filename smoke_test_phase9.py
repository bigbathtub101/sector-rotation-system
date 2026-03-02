#!/usr/bin/env python3
"""
smoke_test_phase9.py — Phase 9: Holdings Tracker Smoke Test
=============================================================
Tests holdings_tracker.py in isolation with synthetic data,
then verifies integration with monitor.py and dashboard.py
data flows.
"""

import datetime as dt
import json
import os
import sqlite3
import sys
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

# ── Import the module under test ──
import holdings_tracker as ht

# ── Also import data_feeds for DB init ──
import data_feeds

# ── Config ──
CONFIG_PATH = REPO_ROOT / "config.yaml"
TEST_DB = REPO_ROOT / "smoke_test_phase9.db"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def seed_prices(conn, cfg, n_days=30):
    """Seed 30 days of synthetic prices for testing."""
    tickers_cfg = cfg.get("tickers", {})
    all_tickers = []
    for key in ["sector_etfs", "industry_etfs", "thematic_etfs",
                 "geographic_etfs", "factor_etfs"]:
        all_tickers.extend(tickers_cfg.get(key, []))
    for key in ["watchlist_biotech", "watchlist_ai_software",
                 "watchlist_defense", "watchlist_green_materials"]:
        all_tickers.extend(tickers_cfg.get(key, []))
    all_tickers.extend(["BIL", "SPY", "SGOV", "JAAA"])
    all_tickers = list(dict.fromkeys(all_tickers))

    dates = pd.bdate_range(end=dt.date.today(), periods=n_days)
    np.random.seed(42)

    rows = []
    for ticker in all_tickers:
        base = np.random.uniform(20, 300)
        prices = base * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n_days)))
        for d, p in zip(dates, prices):
            rows.append((
                d.strftime("%Y-%m-%d"), ticker,
                round(p * 1.001, 4), round(p * 1.01, 4),
                round(p * 0.99, 4), round(p, 4), round(p, 4),
                int(np.random.uniform(5e5, 1e7)), 0,
                dt.datetime.utcnow().isoformat(),
            ))

    conn.executemany(
        "INSERT OR REPLACE INTO prices "
        "(date, ticker, open, high, low, close, adj_close, volume, stale_price, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(all_tickers), len(rows)


def seed_target_allocation(conn, cfg):
    """Seed a target allocation so drift can be computed."""
    alloc = {
        "us_equities": 0.55,
        "intl_developed": 0.10,
        "em_equities": 0.08,
        "energy_materials": 0.07,
        "healthcare": 0.10,
        "industry_sub": 0.03,
        "thematic": 0.04,
        "cash_short_duration": 0.03,
    }
    today = dt.date.today().isoformat()
    now = dt.datetime.now().isoformat()

    # Compute dollar splits
    total = cfg.get("portfolio", {}).get("total_value", 144000)
    taxable_val = 100000
    roth_val = 44000
    taxable_d = {k: round(v * taxable_val, 2) for k, v in alloc.items()}
    roth_d = {k: round(v * roth_val, 2) for k, v in alloc.items()}

    conn.execute(
        "INSERT OR REPLACE INTO allocations "
        "(date, regime, allocation_json, dollar_taxable, dollar_roth, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (today, "offense", json.dumps(alloc), json.dumps(taxable_d),
         json.dumps(roth_d), now),
    )
    conn.commit()


def run_test(name, func):
    """Run a test and print result."""
    try:
        result = func()
        print(f"  ✅ {name}")
        return True, result
    except Exception as exc:
        print(f"  ❌ {name}: {exc}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    print("=" * 70)
    print("  SMOKE TEST — Phase 9: Holdings Tracker")
    print("=" * 70)

    cfg = load_config()
    results = []

    # Clean up old test DB
    if TEST_DB.exists():
        TEST_DB.unlink()

    # --- Test 1: DB initialization ---
    def test_db_init():
        conn = data_feeds.init_database(TEST_DB)
        ht.init_holdings_tables(TEST_DB)
        # Verify tables exist
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "trades" in tables, f"trades table missing. Found: {tables}"
        assert "holdings" in tables, f"holdings table missing. Found: {tables}"
        conn.close()
        return tables
    passed, _ = run_test("DB initialization (trades + holdings tables)", test_db_init)
    results.append(passed)

    # --- Test 2: Seed test data ---
    def test_seed():
        conn = sqlite3.connect(str(TEST_DB))
        n_tickers, n_rows = seed_prices(conn, cfg)
        seed_target_allocation(conn, cfg)
        conn.close()
        assert n_tickers > 10, f"Only {n_tickers} tickers seeded"
        return f"{n_tickers} tickers, {n_rows} price rows"
    passed, detail = run_test("Seed synthetic prices + target allocation", test_seed)
    results.append(passed)
    if detail:
        print(f"       → {detail}")

    # --- Test 3: Record BUY trades ---
    def test_buy_trades():
        conn = ht.init_holdings_tables(TEST_DB)
        trades = [
            ("XLK", 200, 210.50, "taxable"),
            ("XLV", 150, 145.00, "taxable"),
            ("EEM", 300, 42.50, "taxable"),
            ("XBI", 100, 88.00, "roth_ira"),
            ("CRWD", 30, 380.00, "roth_ira"),
            ("SPY", 50, 520.00, "taxable"),
            ("BIL", 100, 91.50, "taxable"),
        ]
        for ticker, shares, price, account in trades:
            result = ht.record_trade(conn, ticker, "BUY", shares, price, account)
            assert result["status"] == "ok", f"Failed: {result}"

        # Verify trades in DB
        count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        assert count == len(trades), f"Expected {len(trades)} trades, got {count}"
        conn.close()
        return f"{count} trades recorded"
    passed, detail = run_test("Record BUY trades (7 positions across 2 accounts)", test_buy_trades)
    results.append(passed)
    if detail:
        print(f"       → {detail}")

    # --- Test 4: Record a SELL ---
    def test_sell_trade():
        conn = ht.init_holdings_tables(TEST_DB)
        result = ht.record_trade(conn, "BIL", "SELL", 50, 91.80, "taxable")
        assert result["status"] == "ok", f"Sell failed: {result}"

        # Verify can't over-sell
        result = ht.record_trade(conn, "BIL", "SELL", 999, 91.80, "taxable")
        assert result["status"] == "error", "Over-sell should fail"
        conn.close()
        return "SELL ok, over-sell blocked"
    passed, detail = run_test("SELL trade + over-sell validation", test_sell_trade)
    results.append(passed)
    if detail:
        print(f"       → {detail}")

    # --- Test 5: Refresh holdings with mock prices ---
    def test_refresh():
        conn = ht.init_holdings_tables(TEST_DB)
        mock_prices = {
            "XLK": 215.00, "XLV": 148.00, "EEM": 43.00,
            "XBI": 90.00, "CRWD": 395.00, "SPY": 525.00, "BIL": 91.60,
        }
        result = ht.refresh_holdings(conn, cfg, mock_prices=mock_prices)
        assert result["status"] == "ok"
        assert result["positions"] >= 6, f"Expected ≥6 positions, got {result['positions']}"

        # Check portfolio value is reasonable
        val = result["portfolio_market_value"]
        assert 50000 < val < 200000, f"Portfolio value ${val} out of range"

        # Verify holdings table populated
        count = conn.execute("SELECT COUNT(*) FROM holdings WHERE shares > 0").fetchone()[0]
        assert count >= 6
        conn.close()
        return f"{result['positions']} positions, portfolio=${val:,.2f}"
    passed, detail = run_test("Refresh holdings with mock prices", test_refresh)
    results.append(passed)
    if detail:
        print(f"       → {detail}")

    # --- Test 6: Drift calculation ---
    def test_drift():
        conn = ht.init_holdings_tables(TEST_DB)
        drift = ht.compute_drift(conn, cfg)
        assert drift["status"] == "ok", f"Drift status: {drift['status']}"
        assert "drift_bps" in drift
        assert "max_drift_bps" in drift
        assert drift["max_drift_bps"] >= 0
        assert "deployment_pct" in drift
        assert 0 < drift["deployment_pct"] <= 100
        assert "accounts" in drift

        # Print drift summary
        print(f"\n       Drift analysis:")
        for cls, detail in drift.get("drift_detail", {}).items():
            actual = detail.get("actual_pct", 0)
            target = detail.get("target_pct", 0)
            dbps = detail.get("drift_bps", 0)
            if actual != 0 or target != 0:
                print(f"         {cls:<22s}: actual={actual:>5.1f}%  target={target:>5.1f}%  drift={dbps:>+6.0f}bp")

        conn.close()
        return f"max_drift={drift['max_drift_bps']:.0f}bps, deployed={drift['deployment_pct']:.1f}%"
    passed, detail = run_test("Compute drift (actual vs target allocation)", test_drift)
    results.append(passed)
    if detail:
        print(f"       → {detail}")

    # --- Test 7: Human-readable summary ---
    def test_summary():
        conn = ht.init_holdings_tables(TEST_DB)
        summary = ht.get_holdings_summary(conn, cfg)
        assert "HOLDINGS VS TARGET" in summary
        assert "$" in summary
        assert "%" in summary
        print(f"\n{summary}\n")
        conn.close()
        return f"{len(summary)} chars"
    passed, detail = run_test("Human-readable holdings summary", test_summary)
    results.append(passed)
    if detail:
        print(f"       → {detail}")

    # --- Test 8: Monitor integration functions ---
    def test_monitor_integration():
        conn = ht.init_holdings_tables(TEST_DB)

        # Test get_actual_weights
        weights = ht.get_actual_weights(conn, cfg)
        assert weights is not None, "Should have weights"
        assert isinstance(weights, dict)
        total = sum(weights.values())
        # Weights should sum to something reasonable (not necessarily 1.0 since
        # some portfolio may be uninvested cash)
        assert 0.3 < total < 1.1, f"Weights sum {total} out of range"

        # Test get_holdings_for_alerts
        alert_data = ht.get_holdings_for_alerts(conn, cfg)
        assert alert_data["has_holdings"] is True
        assert "actual_weights" in alert_data
        assert "max_drift_bps" in alert_data
        assert "deployment_pct" in alert_data

        conn.close()
        return f"weights_sum={total:.3f}, alert_keys={list(alert_data.keys())}"
    passed, detail = run_test("Monitor integration functions", test_monitor_integration)
    results.append(passed)
    if detail:
        print(f"       → {detail}")

    # --- Test 9: CSV import ---
    def test_csv_import():
        # Create a test CSV
        csv_path = REPO_ROOT / "test_trades_import.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "ticker", "action", "shares", "price", "account", "notes"])
            writer.writerow(["2026-03-01", "ALB", "BUY", "40", "105.00", "taxable", "lithium play"])
            writer.writerow(["2026-03-01", "NBIX", "BUY", "50", "130.00", "roth_ira", "biotech"])

        conn = ht.init_holdings_tables(TEST_DB)
        result = ht.import_trades_csv(conn, str(csv_path))
        assert result["imported"] == 2, f"Expected 2 imports, got {result}"

        # Verify total trades now
        count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        assert count == 10, f"Expected 10 total trades, got {count}"  # 7 buys + 1 sell + 2 imports

        conn.close()
        csv_path.unlink()
        return f"imported={result['imported']}, total_trades={count}"
    # Need csv import
    import csv as csv_module
    # Hacky but needed for test
    import csv
    passed, detail = run_test("CSV trade import", test_csv_import)
    results.append(passed)
    if detail:
        print(f"       → {detail}")

    # --- Test 10: Ticker-to-asset-class mapping ---
    def test_mapping():
        mapping = ht.build_ticker_to_asset_class(cfg)
        assert mapping.get("XLK") == "us_equities", f"XLK → {mapping.get('XLK')}"
        assert mapping.get("EEM") == "em_equities", f"EEM → {mapping.get('EEM')}"
        assert mapping.get("XBI") == "healthcare", f"XBI → {mapping.get('XBI')}"  # XBI is in industry_etfs
        assert mapping.get("SGOV") == "cash_short_duration", f"SGOV → {mapping.get('SGOV')}"
        assert mapping.get("CRWD") == "us_equities", f"CRWD → {mapping.get('CRWD')}"
        assert mapping.get("BOTZ") == "thematic", f"BOTZ → {mapping.get('BOTZ')}"
        assert mapping.get("ALB") == "energy_materials", f"ALB → {mapping.get('ALB')}"
        return f"{len(mapping)} tickers mapped"
    passed, detail = run_test("Ticker-to-asset-class mapping", test_mapping)
    results.append(passed)
    if detail:
        print(f"       → {detail}")

    # --- Summary ---
    print()
    print("=" * 70)
    passed_count = sum(results)
    total = len(results)
    failed = total - passed_count
    print(f"  Phase 9 Smoke Test: {passed_count}/{total} passed"
          f" ({failed} failed) ")
    if failed == 0:
        print("  🎉 ALL TESTS PASSED")
    else:
        print(f"  ⚠️  {failed} TESTS FAILED")
    print("=" * 70)

    # Cleanup
    if TEST_DB.exists():
        TEST_DB.unlink()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
