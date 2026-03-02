"""
deployment_day1.py — Phase 11: First Live Deployment Trade Plan
================================================================
Generates the exact trades needed to move from 100% cash (SGOV/JAAA)
to the system's recommended allocation.

Steps:
  1. Seeds the holdings tracker with current cash positions
  2. Reads the target allocation from current_allocation.json
  3. Fetches live prices for every target ticker via yfinance
  4. Computes exact share counts (rounded down) and residual cash
  5. Outputs a trade-by-trade execution plan with dollar amounts

Usage:
  python deployment_day1.py                  # Full output
  python deployment_day1.py --seed-only      # Just seed cash positions
  python deployment_day1.py --csv            # Also export to deployment_trades.csv
"""

import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys
from math import floor
from pathlib import Path

import pandas as pd
import yaml

# Allow importing sibling modules
sys.path.insert(0, str(Path(__file__).parent))

from holdings_tracker import init_holdings_tables, record_trade, refresh_holdings, load_config

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "rotation_system.db"
CONFIG_PATH = BASE_DIR / "config.yaml"
ALLOC_PATH = BASE_DIR / "current_allocation.json"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("deployment_day1")


def load_allocation() -> dict:
    """Load the current target allocation produced by monitor.py."""
    if not ALLOC_PATH.exists():
        logger.error("current_allocation.json not found. Run 'python monitor.py' first.")
        sys.exit(1)
    with open(ALLOC_PATH) as f:
        return json.load(f)


def fetch_live_prices(tickers: list) -> dict:
    """Fetch current prices for all tickers via yfinance."""
    import yfinance as yf

    logger.info("\nFetching live prices for %d tickers...", len(tickers))
    prices = {}
    failed = []

    # Batch download
    try:
        data = yf.download(tickers, period="1d", progress=False)
        if "Close" in data.columns or len(tickers) == 1:
            close = data["Close"] if len(tickers) > 1 else pd.DataFrame({tickers[0]: data["Close"]})
            for t in tickers:
                if t in close.columns:
                    val = close[t].dropna()
                    if len(val) > 0:
                        prices[t] = round(float(val.iloc[-1]), 2)
                    else:
                        failed.append(t)
                else:
                    failed.append(t)
    except Exception as e:
        logger.warning("Batch download failed: %s — falling back to individual", e)
        failed = tickers

    # Retry failed individually
    for t in failed:
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
            if price:
                prices[t] = round(float(price), 2)
            else:
                logger.warning("  Could not get price for %s", t)
        except Exception as e:
            logger.warning("  Failed to get price for %s: %s", t, e)

    return prices


def seed_cash_positions(conn, cfg):
    """
    Record the starting cash positions: 100% SGOV in taxable, 100% JAAA in Roth.
    These represent the current state before any deployment trades.
    """
    logger.info("\n" + "=" * 70)
    logger.info("SEEDING STARTING CASH POSITIONS")
    logger.info("=" * 70)

    taxable_value = cfg.get("portfolio", {}).get("accounts", {}).get("taxable", {}).get("value", 100000)
    roth_value = cfg.get("portfolio", {}).get("accounts", {}).get("roth_ira", {}).get("value", 44000)

    # Get current SGOV and JAAA prices
    prices = fetch_live_prices(["SGOV", "JAAA"])

    sgov_price = prices.get("SGOV")
    jaaa_price = prices.get("JAAA")

    if not sgov_price or not jaaa_price:
        logger.error("Could not fetch SGOV/JAAA prices. Cannot seed.")
        return False

    # Calculate shares
    sgov_shares = round(taxable_value / sgov_price, 4)
    jaaa_shares = round(roth_value / jaaa_price, 4)

    logger.info(f"  SGOV @ ${sgov_price:.2f} → {sgov_shares:.4f} shares = ${sgov_shares * sgov_price:,.2f} (taxable)")
    logger.info(f"  JAAA @ ${jaaa_price:.2f} → {jaaa_shares:.4f} shares = ${jaaa_shares * jaaa_price:,.2f} (roth_ira)")

    # Record buys
    r1 = record_trade(conn, "SGOV", "BUY", sgov_shares, sgov_price, "taxable",
                       notes="Day 0: Starting cash position")
    r2 = record_trade(conn, "JAAA", "BUY", jaaa_shares, jaaa_price, "roth_ira",
                       notes="Day 0: Starting cash position")

    if r1["status"] != "ok" or r2["status"] != "ok":
        logger.error("Failed to seed: %s / %s", r1.get("message", ""), r2.get("message", ""))
        return False

    refresh_holdings(conn, cfg)
    logger.info("  Cash positions seeded successfully.\n")
    return True


def generate_trade_plan(alloc: dict, prices: dict, cfg: dict) -> list:
    """
    Generate the exact trade plan to move from cash to target allocation.

    Returns list of trade dicts with:
      ticker, action, account, target_dollars, price, shares, actual_dollars, residual
    """
    positions = alloc["positions"]
    trades = []

    taxable_value = cfg.get("portfolio", {}).get("accounts", {}).get("taxable", {}).get("value", 100000)
    roth_value = cfg.get("portfolio", {}).get("accounts", {}).get("roth_ira", {}).get("value", 44000)

    # Step 1: SELL all cash positions
    sgov_price = prices.get("SGOV")
    jaaa_price = prices.get("JAAA")

    if sgov_price:
        sgov_shares = round(taxable_value / sgov_price, 4)
        trades.append({
            "order": 0,
            "ticker": "SGOV",
            "action": "SELL",
            "account": "taxable",
            "target_dollars": taxable_value,
            "price": sgov_price,
            "shares": sgov_shares,
            "actual_dollars": round(sgov_shares * sgov_price, 2),
            "reason": "Liquidate cash position to fund deployment"
        })

    if jaaa_price:
        jaaa_shares = round(roth_value / jaaa_price, 4)
        trades.append({
            "order": 0,
            "ticker": "JAAA",
            "action": "SELL",
            "account": "roth_ira",
            "target_dollars": roth_value,
            "price": jaaa_price,
            "shares": jaaa_shares,
            "actual_dollars": round(jaaa_shares * jaaa_price, 2),
            "reason": "Liquidate cash position to fund deployment"
        })

    # Step 2: BUY target positions
    taxable_spent = 0.0
    roth_spent = 0.0
    order = 1

    for ticker, pos in sorted(positions.items(), key=lambda x: -x[1]["total_dollars"]):
        price = prices.get(ticker)
        if not price:
            logger.warning("  No price for %s — skipping", ticker)
            continue

        # Determine account and target dollars
        if pos["taxable_dollars"] > 0 and pos["roth_dollars"] > 0:
            # Split position — handle both accounts
            for acct, target_d in [("taxable", pos["taxable_dollars"]),
                                    ("roth_ira", pos["roth_dollars"])]:
                if target_d < 1:
                    continue
                shares = floor(target_d / price)
                if shares <= 0:
                    continue
                actual = round(shares * price, 2)
                if acct == "taxable":
                    taxable_spent += actual
                else:
                    roth_spent += actual

                trades.append({
                    "order": order,
                    "ticker": ticker,
                    "action": "BUY",
                    "account": acct,
                    "target_dollars": round(target_d, 2),
                    "price": price,
                    "shares": shares,
                    "actual_dollars": actual,
                    "reason": pos.get("reason", "")
                })
                order += 1
        else:
            acct = "roth_ira" if pos["roth_dollars"] > 0 else "taxable"
            target_d = pos["roth_dollars"] if acct == "roth_ira" else pos["taxable_dollars"]

            shares = floor(target_d / price)
            if shares <= 0:
                # For expensive stocks, allow fractional if target > $50
                if target_d >= 50:
                    shares = round(target_d / price, 4)
                else:
                    continue

            actual = round(shares * price, 2)
            if acct == "taxable":
                taxable_spent += actual
            else:
                roth_spent += actual

            trades.append({
                "order": order,
                "ticker": ticker,
                "action": "BUY",
                "account": acct,
                "target_dollars": round(target_d, 2),
                "price": price,
                "shares": shares if isinstance(shares, int) else round(shares, 4),
                "actual_dollars": actual,
                "reason": pos.get("reason", "")
            })
            order += 1

    # Summary
    taxable_residual = round(taxable_value - taxable_spent, 2)
    roth_residual = round(roth_value - roth_spent, 2)

    return trades, taxable_spent, roth_spent, taxable_residual, roth_residual


def print_trade_plan(trades, taxable_spent, roth_spent, taxable_residual, roth_residual,
                     taxable_value, roth_value):
    """Print the formatted trade plan."""

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "  DAY 1 DEPLOYMENT — TRADE EXECUTION PLAN".center(78) + "║")
    print("║" + f"  {dt.date.today().isoformat()}".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # === SELL ORDERS (liquidate cash) ===
    sells = [t for t in trades if t["action"] == "SELL"]
    buys = [t for t in trades if t["action"] == "BUY"]

    print("\n" + "─" * 80)
    print("  STEP 1: LIQUIDATE CASH POSITIONS")
    print("─" * 80)
    print(f"  {'Ticker':<8} {'Action':<6} {'Account':<10} {'Shares':>10} {'Price':>10} {'Amount':>12}")
    print("  " + "─" * 68)
    for t in sells:
        print(f"  {t['ticker']:<8} {t['action']:<6} {t['account']:<10} "
              f"{t['shares']:>10.2f} {t['price']:>10.2f} ${t['actual_dollars']:>10,.2f}")

    # === BUY ORDERS (by account) ===
    taxable_buys = [t for t in buys if t["account"] == "taxable"]
    roth_buys = [t for t in buys if t["account"] == "roth_ira"]

    print("\n" + "─" * 80)
    print(f"  STEP 2: DEPLOY TAXABLE ACCOUNT (${taxable_value:,.0f})")
    print("─" * 80)
    print(f"  {'Ticker':<8} {'Action':<6} {'Shares':>8} {'Price':>10} {'Target $':>12} {'Actual $':>12} {'Diff':>8}")
    print("  " + "─" * 68)
    for t in sorted(taxable_buys, key=lambda x: -x["actual_dollars"]):
        diff = t["actual_dollars"] - t["target_dollars"]
        print(f"  {t['ticker']:<8} {'BUY':<6} {t['shares']:>8} "
              f"${t['price']:>9.2f} ${t['target_dollars']:>10,.2f} ${t['actual_dollars']:>10,.2f} "
              f"{'$' if diff >= 0 else '-$'}{abs(diff):>5.0f}")
    print("  " + "─" * 68)
    print(f"  {'TOTAL':<8} {'':6} {'':>8} {'':>10} "
          f"${taxable_value:>10,.2f} ${taxable_spent:>10,.2f}")
    print(f"  {'RESIDUAL CASH':46} ${taxable_residual:>10,.2f}")

    print("\n" + "─" * 80)
    print(f"  STEP 3: DEPLOY ROTH IRA (${roth_value:,.0f})")
    print("─" * 80)
    print(f"  {'Ticker':<8} {'Action':<6} {'Shares':>8} {'Price':>10} {'Target $':>12} {'Actual $':>12} {'Diff':>8}")
    print("  " + "─" * 68)
    for t in sorted(roth_buys, key=lambda x: -x["actual_dollars"]):
        diff = t["actual_dollars"] - t["target_dollars"]
        print(f"  {t['ticker']:<8} {'BUY':<6} {t['shares']:>8} "
              f"${t['price']:>9.2f} ${t['target_dollars']:>10,.2f} ${t['actual_dollars']:>10,.2f} "
              f"{'$' if diff >= 0 else '-$'}{abs(diff):>5.0f}")
    print("  " + "─" * 68)
    print(f"  {'TOTAL':<8} {'':6} {'':>8} {'':>10} "
          f"${roth_value:>10,.2f} ${roth_spent:>10,.2f}")
    print(f"  {'RESIDUAL CASH':46} ${roth_residual:>10,.2f}")

    # === GRAND SUMMARY ===
    total_spent = taxable_spent + roth_spent
    total_residual = taxable_residual + roth_residual
    total_value = taxable_value + roth_value
    n_buys = len(buys)

    print("\n" + "═" * 80)
    print("  DEPLOYMENT SUMMARY")
    print("═" * 80)
    print(f"  Total portfolio:       ${total_value:>12,.2f}")
    print(f"  Total deployed:        ${total_spent:>12,.2f}  ({total_spent/total_value*100:.1f}%)")
    print(f"  Residual cash:         ${total_residual:>12,.2f}  ({total_residual/total_value*100:.1f}%)")
    print(f"  Number of buy orders:  {n_buys:>12}")
    print(f"  Number of positions:   {n_buys:>12}")
    print(f"  Regime:                {'OFFENSE':>12}")
    print("═" * 80)

    # Warnings
    if total_residual > 1000:
        print(f"\n  NOTE: ${total_residual:,.2f} residual cash due to whole-share rounding.")
        print("  Consider leaving in SGOV/JAAA or adding to largest positions.")


def export_csv(trades, output_path):
    """Export trade plan to CSV."""
    fieldnames = ["order", "ticker", "action", "account", "shares", "price",
                  "target_dollars", "actual_dollars", "reason"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow({k: t.get(k, "") for k in fieldnames})
    logger.info(f"\nTrade plan exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Day 1 Deployment Trade Plan")
    parser.add_argument("--seed-only", action="store_true",
                        help="Only seed cash positions, don't generate trade plan")
    parser.add_argument("--csv", action="store_true",
                        help="Also export trade plan to deployment_trades.csv")
    parser.add_argument("--no-seed", action="store_true",
                        help="Skip seeding (if already seeded)")
    parser.add_argument("--db", type=str, default=None,
                        help="DB path override")
    parser.add_argument("--config", type=str, default=None,
                        help="Config path override")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DB_PATH
    config_path = Path(args.config) if args.config else CONFIG_PATH

    cfg = load_config(config_path)
    conn = init_holdings_tables(db_path)

    taxable_value = cfg.get("portfolio", {}).get("accounts", {}).get("taxable", {}).get("value", 100000)
    roth_value = cfg.get("portfolio", {}).get("accounts", {}).get("roth_ira", {}).get("value", 44000)

    # Step 1: Seed cash positions
    if not args.no_seed:
        # Check if already seeded
        cur = conn.execute("SELECT COUNT(*) FROM trades")
        count = cur.fetchone()[0]
        if count > 0:
            logger.info("Holdings already seeded (%d trades in DB). Use --no-seed to skip.", count)
        else:
            success = seed_cash_positions(conn, cfg)
            if not success:
                conn.close()
                sys.exit(1)

    if args.seed_only:
        conn.close()
        return

    # Step 2: Load target allocation
    alloc = load_allocation()

    # Step 3: Fetch live prices for all target tickers + cash
    target_tickers = list(alloc["positions"].keys())
    all_tickers = list(set(target_tickers + ["SGOV", "JAAA"]))
    prices = fetch_live_prices(all_tickers)

    # Step 4: Generate trade plan
    trades, taxable_spent, roth_spent, taxable_residual, roth_residual = \
        generate_trade_plan(alloc, prices, cfg)

    # Step 5: Print the plan
    print_trade_plan(trades, taxable_spent, roth_spent, taxable_residual, roth_residual,
                     taxable_value, roth_value)

    # Step 6: Export CSV if requested
    if args.csv:
        csv_path = BASE_DIR / "deployment_trades.csv"
        export_csv(trades, csv_path)

    conn.close()


if __name__ == "__main__":
    main()
