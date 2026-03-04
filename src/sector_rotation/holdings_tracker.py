"""
holdings_tracker.py — Phase 9: Holdings Tracker & Drift Engine
================================================================
Global Sector Rotation System

Tracks ACTUAL holdings (what you own) vs TARGET allocation (what the
system recommends).  Computes per-position and per-asset-class drift,
feeds meaningful alerts to the monitor, and provides CLI for recording
trades.

Database Tables (added to rotation_system.db)
-----------------------------------------------
  trades        — immutable trade log (one row per buy/sell)
  holdings      — current snapshot (materialized from trades + prices)

CLI Usage
---------
  # Record a buy
  python holdings_tracker.py buy XLK 50 --price 210.00 --account taxable

  # Record a sell
  python holdings_tracker.py sell XLK 25 --price 215.00 --account taxable

  # Import multiple trades from CSV
  python holdings_tracker.py import trades.csv

  # Show current holdings vs target
  python holdings_tracker.py status

  # Recompute holdings snapshot (pulls latest prices)
  python holdings_tracker.py refresh

Dependencies: yfinance (for price refresh), pandas, pyyaml, sqlite3
"""

import argparse
import csv
import datetime as dt
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent
LOG_FILE = LOG_DIR / "holdings_tracker.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("holdings_tracker")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"
DB_PATH = Path(__file__).parent / "rotation_system.db"

# ---------------------------------------------------------------------------
# ASSET CLASS MAPPING
# ---------------------------------------------------------------------------
# Maps tickers to asset class categories used by the optimizer.
# This is critical for computing asset-class-level drift against
# the target allocation bands.

def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load master configuration from YAML."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_ticker_to_asset_class(cfg: dict) -> Dict[str, str]:
    """
    Build a mapping from ticker → asset class key.
    Uses the config.yaml ticker universes to classify each ticker.
    Returns dict like {'XLK': 'us_equities', 'EEM': 'em_equities', ...}
    """
    tickers_cfg = cfg.get("tickers", {})
    mapping = {}

    # US equity ETFs (sectors)
    for t in tickers_cfg.get("sector_etfs", []):
        mapping[t] = "us_equities"

    # Industry sub-sector ETFs — most are industry_sub, but healthcare
    # sub-sectors (XBI, IHI, XPH) should map to healthcare
    healthcare_industry_etfs = {"XBI", "IHI", "XPH"}
    for t in tickers_cfg.get("industry_etfs", []):
        if t in healthcare_industry_etfs:
            mapping[t] = "healthcare"
        else:
            mapping[t] = "industry_sub"

    # Thematic ETFs
    for t in tickers_cfg.get("thematic_etfs", []):
        mapping[t] = "thematic"

    # Geographic ETFs — split into developed vs EM
    developed_set = {"VGK", "EWJ", "EWY", "EWT"}
    em_set = {"EEM", "INDA", "EWZ", "FXI", "MCHI", "KWEB", "VWO", "IEMG"}
    for t in tickers_cfg.get("geographic_etfs", []):
        if t in developed_set:
            mapping[t] = "intl_developed"
        elif t in em_set:
            mapping[t] = "em_equities"
        else:
            mapping[t] = "intl_developed"  # default geo → developed

    # Factor ETFs → US equities
    for t in tickers_cfg.get("factor_etfs", []):
        mapping[t] = "us_equities"

    # Healthcare-related watchlist tickers
    for t in tickers_cfg.get("watchlist_biotech", []):
        mapping[t] = "healthcare"

    # Energy/materials watchlist
    for t in tickers_cfg.get("watchlist_green_materials", []):
        mapping[t] = "energy_materials"
    for t in tickers_cfg.get("watchlist_energy_transition", []):
        mapping[t] = "energy_materials"

    # AI / software / defense / fintech / semis → US equities
    for key in ["watchlist_ai_software", "watchlist_defense",
                "watchlist_semiconductors", "watchlist_fintech"]:
        for t in tickers_cfg.get(key, []):
            mapping[t] = "us_equities"

    # Cash proxies
    for t in ["BIL", "SGOV", "JAAA", "SHV", "SHY"]:
        mapping[t] = "cash_short_duration"

    # Benchmarks that are also investable
    mapping["SPY"] = "us_equities"
    mapping["QQQ"] = "us_equities"
    mapping["IWM"] = "us_equities"
    mapping["AGG"] = "cash_short_duration"
    mapping["GLD"] = "energy_materials"
    mapping["TLT"] = "cash_short_duration"

    return mapping


# ===========================================================================
# DATABASE SCHEMA
# ===========================================================================

def init_holdings_tables(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Create holdings-related tables in the existing rotation_system.db.
    Safe to call multiple times (IF NOT EXISTS).
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Immutable trade log — one row per buy/sell
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            action          TEXT NOT NULL CHECK(action IN ('BUY', 'SELL')),
            shares          REAL NOT NULL CHECK(shares > 0),
            price           REAL NOT NULL CHECK(price > 0),
            total_cost      REAL NOT NULL,
            account         TEXT NOT NULL CHECK(account IN ('taxable', 'roth_ira')),
            notes           TEXT DEFAULT '',
            created_at      TEXT NOT NULL
        )
    """)

    # Current holdings snapshot — recomputed from trades + latest prices
    cur.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            ticker          TEXT NOT NULL,
            account         TEXT NOT NULL CHECK(account IN ('taxable', 'roth_ira')),
            shares          REAL NOT NULL DEFAULT 0,
            avg_cost        REAL NOT NULL DEFAULT 0,
            cost_basis      REAL NOT NULL DEFAULT 0,
            current_price   REAL DEFAULT NULL,
            market_value    REAL DEFAULT NULL,
            unrealized_pnl  REAL DEFAULT NULL,
            asset_class     TEXT DEFAULT '',
            weight_pct      REAL DEFAULT 0,
            updated_at      TEXT NOT NULL,
            PRIMARY KEY (ticker, account)
        )
    """)

    # Indexes for fast lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_holdings_asset ON holdings(asset_class)")

    conn.commit()
    logger.info("Holdings tables initialized at %s", db_path)
    return conn


# ===========================================================================
# TRADE RECORDING
# ===========================================================================

def record_trade(conn: sqlite3.Connection,
                 ticker: str,
                 action: str,
                 shares: float,
                 price: float,
                 account: str,
                 date: str = None,
                 notes: str = "") -> Dict[str, Any]:
    """
    Record a single trade (BUY or SELL).

    Parameters
    ----------
    ticker  : e.g. "XLK"
    action  : "BUY" or "SELL"
    shares  : number of shares (positive)
    price   : price per share
    account : "taxable" or "roth_ira"
    date    : trade date (defaults to today)
    notes   : optional description

    Returns
    -------
    dict with trade details and validation status
    """
    action = action.upper()
    ticker = ticker.upper()
    account = account.lower().replace("roth", "roth_ira").replace("ira_ira", "ira")
    if account not in ("taxable", "roth_ira"):
        return {"status": "error", "message": f"Invalid account: {account}. Use 'taxable' or 'roth_ira'."}

    if action not in ("BUY", "SELL"):
        return {"status": "error", "message": f"Invalid action: {action}. Use 'BUY' or 'SELL'."}

    if shares <= 0 or price <= 0:
        return {"status": "error", "message": "Shares and price must be positive."}

    # Validate sell doesn't exceed current holdings
    if action == "SELL":
        current = _get_current_shares(conn, ticker, account)
        if shares > current + 0.001:  # small float tolerance
            return {
                "status": "error",
                "message": (
                    f"Cannot sell {shares:.4f} shares of {ticker} in {account} — "
                    f"only {current:.4f} shares held."
                ),
            }

    trade_date = date or dt.date.today().isoformat()
    total_cost = round(shares * price, 2)
    now = dt.datetime.now().isoformat()

    conn.execute(
        "INSERT INTO trades (date, ticker, action, shares, price, total_cost, "
        "account, notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (trade_date, ticker, action, shares, price, total_cost, account, notes, now),
    )
    conn.commit()

    logger.info("Recorded %s %s %.4f @ $%.2f (%s) = $%.2f",
                action, ticker, shares, price, account, total_cost)

    return {
        "status": "ok",
        "trade": {
            "date": trade_date,
            "ticker": ticker,
            "action": action,
            "shares": shares,
            "price": price,
            "total_cost": total_cost,
            "account": account,
            "notes": notes,
        },
    }


def _get_current_shares(conn: sqlite3.Connection,
                        ticker: str, account: str) -> float:
    """Get current share count from trades ledger (not the snapshot)."""
    rows = conn.execute(
        "SELECT action, shares FROM trades WHERE ticker = ? AND account = ?",
        (ticker, account),
    ).fetchall()
    total = 0.0
    for action, shares in rows:
        if action == "BUY":
            total += shares
        elif action == "SELL":
            total -= shares
    return max(total, 0.0)


def import_trades_csv(conn: sqlite3.Connection,
                      csv_path: str) -> Dict[str, Any]:
    """
    Import trades from a CSV file.

    Expected columns: date, ticker, action, shares, price, account, notes (optional)
    """
    path = Path(csv_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found: {csv_path}"}

    imported = 0
    errors = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):  # row 1 is header
            try:
                result = record_trade(
                    conn,
                    ticker=row["ticker"].strip(),
                    action=row["action"].strip(),
                    shares=float(row["shares"]),
                    price=float(row["price"]),
                    account=row["account"].strip(),
                    date=row.get("date", "").strip() or None,
                    notes=row.get("notes", "").strip(),
                )
                if result["status"] == "ok":
                    imported += 1
                else:
                    errors.append(f"Row {i}: {result['message']}")
            except Exception as exc:
                errors.append(f"Row {i}: {exc}")

    return {
        "status": "ok" if not errors else "partial",
        "imported": imported,
        "errors": errors,
    }


# ===========================================================================
# HOLDINGS SNAPSHOT (from trades ledger + latest prices)
# ===========================================================================

def refresh_holdings(conn: sqlite3.Connection,
                     cfg: dict,
                     mock_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Recompute the holdings table from the trades ledger.

    Steps:
    1. Aggregate trades → net shares + avg cost per (ticker, account)
    2. Fetch latest prices (from DB or yfinance)
    3. Compute market value, P&L, weight
    4. Upsert holdings table

    Parameters
    ----------
    mock_prices : dict of {ticker: price} for testing without yfinance

    Returns
    -------
    dict with portfolio summary
    """
    now = dt.datetime.now().isoformat()
    ticker_map = build_ticker_to_asset_class(cfg)

    # Step 1: Aggregate trades into positions
    rows = conn.execute(
        "SELECT ticker, account, action, shares, price FROM trades ORDER BY date, trade_id"
    ).fetchall()

    # positions[ticker][account] = {'shares': float, 'cost_basis': float}
    positions: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ticker, account, action, shares, price in rows:
        if ticker not in positions:
            positions[ticker] = {}
        if account not in positions[ticker]:
            positions[ticker][account] = {"shares": 0.0, "cost_basis": 0.0}

        pos = positions[ticker][account]
        if action == "BUY":
            pos["cost_basis"] += shares * price
            pos["shares"] += shares
        elif action == "SELL":
            if pos["shares"] > 0:
                avg_cost = pos["cost_basis"] / pos["shares"]
                pos["cost_basis"] -= shares * avg_cost  # reduce cost basis proportionally
                pos["shares"] -= shares
            pos["shares"] = max(pos["shares"], 0.0)
            pos["cost_basis"] = max(pos["cost_basis"], 0.0)

    # Filter out zero positions
    active_positions = {}
    for ticker, accounts in positions.items():
        for account, data in accounts.items():
            if data["shares"] > 0.01:  # tolerance
                if ticker not in active_positions:
                    active_positions[ticker] = {}
                active_positions[ticker][account] = data

    # Step 2: Fetch latest prices
    all_tickers = list(active_positions.keys())
    prices = {}
    if mock_prices:
        prices = mock_prices
    elif all_tickers:
        # Try DB prices first (today or most recent)
        for ticker in all_tickers:
            row = conn.execute(
                "SELECT close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                (ticker,),
            ).fetchone()
            if row and row[0]:
                prices[ticker] = row[0]

        # Fall back to yfinance for missing prices
        missing = [t for t in all_tickers if t not in prices]
        if missing:
            try:
                import yfinance as yf
                data = yf.download(missing, period="1d", progress=False)
                if not data.empty:
                    if "Close" in data.columns:
                        for t in missing:
                            try:
                                if len(missing) == 1:
                                    p = data["Close"].iloc[-1]
                                else:
                                    p = data["Close"][t].iloc[-1]
                                if not np.isnan(p):
                                    prices[t] = float(p)
                            except Exception:
                                pass
            except Exception as exc:
                logger.warning("yfinance price fetch failed: %s", exc)

    # Step 3: Compute market value + weights
    total_val = cfg.get("portfolio", {}).get("total_value", 144000)

    holdings_data = []
    portfolio_market_value = 0.0

    for ticker, accounts in active_positions.items():
        for account, data in accounts.items():
            shares = data["shares"]
            cost_basis = data["cost_basis"]
            avg_cost = cost_basis / shares if shares > 0 else 0
            price = prices.get(ticker)
            mkt_val = shares * price if price else None
            pnl = (mkt_val - cost_basis) if mkt_val else None
            asset_class = ticker_map.get(ticker, "us_equities")

            if mkt_val:
                portfolio_market_value += mkt_val

            holdings_data.append({
                "ticker": ticker,
                "account": account,
                "shares": round(shares, 4),
                "avg_cost": round(avg_cost, 4),
                "cost_basis": round(cost_basis, 2),
                "current_price": round(price, 4) if price else None,
                "market_value": round(mkt_val, 2) if mkt_val else None,
                "unrealized_pnl": round(pnl, 2) if pnl else None,
                "asset_class": asset_class,
                "updated_at": now,
            })

    # Compute weights
    for h in holdings_data:
        if h["market_value"] and portfolio_market_value > 0:
            h["weight_pct"] = round(h["market_value"] / portfolio_market_value * 100, 2)
        else:
            h["weight_pct"] = 0.0

    # Step 4: Upsert holdings table
    conn.execute("DELETE FROM holdings")
    for h in holdings_data:
        conn.execute(
            "INSERT INTO holdings (ticker, account, shares, avg_cost, cost_basis, "
            "current_price, market_value, unrealized_pnl, asset_class, weight_pct, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (h["ticker"], h["account"], h["shares"], h["avg_cost"],
             h["cost_basis"], h["current_price"], h["market_value"],
             h["unrealized_pnl"], h["asset_class"], h["weight_pct"],
             h["updated_at"]),
        )
    conn.commit()

    logger.info("Holdings refreshed: %d positions, portfolio=$%.2f",
                len(holdings_data), portfolio_market_value)

    return {
        "status": "ok",
        "positions": len(holdings_data),
        "portfolio_market_value": round(portfolio_market_value, 2),
        "holdings": holdings_data,
    }


# ===========================================================================
# DRIFT CALCULATION ENGINE
# ===========================================================================

def compute_drift(conn: sqlite3.Connection,
                  cfg: dict) -> Dict[str, Any]:
    """
    Compare ACTUAL holdings vs TARGET allocation.

    Returns drift analysis at two levels:
    1. Asset class level — e.g. "us_equities: actual 45% vs target 65% → -2000 bps"
    2. Position level   — individual ticker weights vs target

    This is the key function that feeds into monitor.py alerts.
    """
    total_val = cfg.get("portfolio", {}).get("total_value", 144000)
    taxable_val = cfg.get("portfolio", {}).get("accounts", {}).get("taxable", {}).get("value", 100000)
    roth_val = cfg.get("portfolio", {}).get("accounts", {}).get("roth_ira", {}).get("value", 44000)

    # --- Fetch actual holdings ---
    holdings = conn.execute(
        "SELECT ticker, account, shares, market_value, asset_class, weight_pct "
        "FROM holdings WHERE shares > 0"
    ).fetchall()

    if not holdings:
        return {
            "status": "no_holdings",
            "message": "No holdings recorded yet. Use 'holdings_tracker.py buy' to record trades.",
            "actual": {},
            "target": {},
            "drift_bps": {},
            "max_drift_bps": 0,
            "total_invested": 0,
            "total_cash": total_val,
            "deployment_pct": 0.0,
        }

    # Build actual asset class weights from holdings
    actual_by_class: Dict[str, float] = {}   # asset_class → market value
    actual_by_ticker: Dict[str, Dict] = {}   # ticker → {mkt_val, weight, account}
    total_market_value = 0.0

    for ticker, account, shares, mkt_val, asset_class, weight in holdings:
        mkt_val = mkt_val or 0
        total_market_value += mkt_val

        if asset_class not in actual_by_class:
            actual_by_class[asset_class] = 0.0
        actual_by_class[asset_class] += mkt_val

        if ticker not in actual_by_ticker:
            actual_by_ticker[ticker] = {"market_value": 0, "account": account}
        actual_by_ticker[ticker]["market_value"] += mkt_val

    # Convert to percentages
    actual_pct: Dict[str, float] = {}
    for cls, val in actual_by_class.items():
        actual_pct[cls] = val / total_val if total_val > 0 else 0

    # Add ticker-level weights
    for ticker, data in actual_by_ticker.items():
        data["weight_pct"] = data["market_value"] / total_val if total_val > 0 else 0

    # --- Fetch target allocation ---
    target_row = conn.execute(
        "SELECT allocation_json FROM allocations ORDER BY date DESC LIMIT 1"
    ).fetchone()

    target_pct: Dict[str, float] = {}
    if target_row and target_row[0]:
        raw = json.loads(target_row[0])
        for key, val in raw.items():
            if isinstance(val, dict):
                target_pct[key] = val.get("pct", 0) / 100.0
            elif isinstance(val, (int, float)):
                target_pct[key] = float(val)

    # --- Compute per-asset-class drift ---
    all_classes = set(list(actual_pct.keys()) + list(target_pct.keys()))
    drift_bps: Dict[str, float] = {}
    drift_detail: Dict[str, Dict] = {}

    for cls in all_classes:
        actual = actual_pct.get(cls, 0.0)
        target = target_pct.get(cls, 0.0)
        drift = (actual - target) * 10000  # basis points
        drift_bps[cls] = round(drift, 1)
        actual_dollars = actual_by_class.get(cls, 0)
        target_dollars = target * total_val
        drift_detail[cls] = {
            "actual_pct": round(actual * 100, 2),
            "target_pct": round(target * 100, 2),
            "drift_bps": round(drift, 1),
            "actual_dollars": round(actual_dollars, 2),
            "target_dollars": round(target_dollars, 2),
            "dollar_gap": round(actual_dollars - target_dollars, 2),
        }

    max_drift = max(abs(d) for d in drift_bps.values()) if drift_bps else 0

    # Implied cash (uninvested portion of portfolio)
    cash_actual = max(total_val - total_market_value, 0)
    cash_target_pct = target_pct.get("cash_short_duration", 0)
    cash_actual_pct = cash_actual / total_val if total_val > 0 else 0
    deployment_pct = total_market_value / total_val * 100 if total_val > 0 else 0

    # Account-level breakdown
    taxable_invested = sum(
        mkt_val or 0 for _, acct, _, mkt_val, _, _ in holdings if acct == "taxable"
    )
    roth_invested = sum(
        mkt_val or 0 for _, acct, _, mkt_val, _, _ in holdings if acct == "roth_ira"
    )

    result = {
        "status": "ok",
        "as_of": dt.datetime.now().isoformat(),
        "actual_by_class": {k: round(v * 100, 2) for k, v in actual_pct.items()},
        "target_by_class": {k: round(v * 100, 2) for k, v in target_pct.items()},
        "drift_bps": drift_bps,
        "drift_detail": drift_detail,
        "max_drift_bps": round(max_drift, 1),
        "actual_by_ticker": actual_by_ticker,
        "total_invested": round(total_market_value, 2),
        "total_cash": round(cash_actual, 2),
        "deployment_pct": round(deployment_pct, 2),
        "accounts": {
            "taxable": {
                "invested": round(taxable_invested, 2),
                "available": round(taxable_val - taxable_invested, 2),
            },
            "roth_ira": {
                "invested": round(roth_invested, 2),
                "available": round(roth_val - roth_invested, 2),
            },
        },
    }

    return result


def get_holdings_summary(conn: sqlite3.Connection,
                         cfg: dict) -> str:
    """
    Generate a human-readable holdings vs target summary.
    Used in monitor executive summary and CLI status command.
    """
    drift = compute_drift(conn, cfg)
    total_val = cfg.get("portfolio", {}).get("total_value", 144000)

    if drift["status"] == "no_holdings":
        return (
            "══ HOLDINGS STATUS ══\n"
            "No holdings recorded. Portfolio is 100% cash.\n"
            f"  Total portfolio: ${total_val:,.0f}\n"
            "  Use 'holdings_tracker.py buy' to record your first trade.\n"
        )

    lines = [
        "══ HOLDINGS VS TARGET ══",
        f"  Portfolio value:  ${total_val:,.0f}",
        f"  Total invested:   ${drift['total_invested']:,.0f} ({drift['deployment_pct']:.1f}% deployed)",
        f"  Cash / uninvested: ${drift['total_cash']:,.0f}",
        f"  Max asset-class drift: {drift['max_drift_bps']:.0f} bps",
        "",
        f"  {'Asset Class':<22s} {'Actual%':>8s} {'Target%':>8s} {'Drift':>8s} {'Actual$':>10s} {'Target$':>10s} {'Gap$':>10s}",
        f"  {'─' * 22} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 10}",
    ]

    DISPLAY_NAMES = {
        "us_equities": "US Equities",
        "intl_developed": "Intl Developed",
        "em_equities": "EM Equities",
        "energy_materials": "Energy/Materials",
        "healthcare": "Healthcare",
        "industry_sub": "Industry Sub",
        "thematic": "Thematic",
        "cash_short_duration": "Cash/Short Dur",
        "vix_overlay_notional": "VIX Overlay",
    }

    for cls in DISPLAY_NAMES:
        detail = drift.get("drift_detail", {}).get(cls, {})
        actual = detail.get("actual_pct", 0)
        target = detail.get("target_pct", 0)
        dbps = detail.get("drift_bps", 0)
        actual_d = detail.get("actual_dollars", 0)
        target_d = detail.get("target_dollars", 0)
        gap_d = detail.get("dollar_gap", 0)

        if actual == 0 and target == 0:
            continue

        flag = ""
        if abs(dbps) >= 200:
            flag = " ⚠️"
        elif abs(dbps) >= 300:
            flag = " 🚨"

        name = DISPLAY_NAMES.get(cls, cls)
        lines.append(
            f"  {name:<22s} {actual:>7.1f}% {target:>7.1f}% {dbps:>+7.0f}bp"
            f" ${actual_d:>9,.0f} ${target_d:>9,.0f} ${gap_d:>+9,.0f}{flag}"
        )

    # Account summary
    accts = drift.get("accounts", {})
    lines.extend([
        "",
        "  Account Breakdown:",
        f"    Taxable:  ${accts.get('taxable', {}).get('invested', 0):,.0f} invested"
        f"  /  ${accts.get('taxable', {}).get('available', 0):,.0f} available",
        f"    Roth IRA: ${accts.get('roth_ira', {}).get('invested', 0):,.0f} invested"
        f"  /  ${accts.get('roth_ira', {}).get('available', 0):,.0f} available",
    ])

    return "\n".join(lines)


# ===========================================================================
# FUNCTIONS FOR MONITOR INTEGRATION
# ===========================================================================

def get_actual_weights(conn: sqlite3.Connection,
                       cfg: dict) -> Optional[Dict[str, float]]:
    """
    Return actual asset-class weights as a dict matching the
    target allocation format: {'us_equities': 0.45, ...}

    Returns None if no holdings are recorded.
    Used by monitor.py AlertEngine for drift calculation.
    """
    drift = compute_drift(conn, cfg)
    if drift["status"] == "no_holdings":
        return None

    # Convert from % to fraction
    return {
        cls: pct / 100.0
        for cls, pct in drift.get("actual_by_class", {}).items()
    }


def get_holdings_for_alerts(conn: sqlite3.Connection,
                            cfg: dict) -> Dict[str, Any]:
    """
    Package holdings data for the alert engine.
    Returns everything the AlertEngine needs to compute meaningful drift.
    """
    drift = compute_drift(conn, cfg)
    return {
        "has_holdings": drift["status"] != "no_holdings",
        "actual_weights": {
            cls: pct / 100.0
            for cls, pct in drift.get("actual_by_class", {}).items()
        } if drift["status"] == "ok" else {},
        "max_drift_bps": drift.get("max_drift_bps", 0),
        "deployment_pct": drift.get("deployment_pct", 0),
        "drift_detail": drift.get("drift_detail", {}),
    }


# ===========================================================================
# CLI INTERFACE
# ===========================================================================

def cli_status(conn: sqlite3.Connection, cfg: dict):
    """Print current holdings vs target summary."""
    # First refresh holdings with latest prices
    refresh_holdings(conn, cfg)
    print()
    print(get_holdings_summary(conn, cfg))
    print()

    # Also print individual positions
    holdings = conn.execute(
        "SELECT ticker, account, shares, avg_cost, current_price, "
        "market_value, unrealized_pnl, weight_pct, asset_class "
        "FROM holdings WHERE shares > 0 ORDER BY market_value DESC"
    ).fetchall()

    if holdings:
        print(f"  {'Ticker':<8s} {'Account':<10s} {'Shares':>8s} {'AvgCost':>8s} "
              f"{'Price':>8s} {'MktVal':>10s} {'P&L':>10s} {'Wt%':>6s} {'Class':<15s}")
        print(f"  {'─' * 8} {'─' * 10} {'─' * 8} {'─' * 8} "
              f"{'─' * 8} {'─' * 10} {'─' * 10} {'─' * 6} {'─' * 15}")
        for row in holdings:
            ticker, acct, shares, avg_cost, price, mkt_val, pnl, wt, cls = row
            pnl_str = f"${pnl:>+9,.2f}" if pnl else "    —    "
            price_str = f"${price:>7,.2f}" if price else "   —   "
            mkt_str = f"${mkt_val:>9,.2f}" if mkt_val else "    —    "
            print(f"  {ticker:<8s} {acct:<10s} {shares:>8.2f} ${avg_cost:>7,.2f} "
                  f"{price_str} {mkt_str} {pnl_str} {wt:>5.1f}% {cls:<15s}")


def cli_trades(conn: sqlite3.Connection):
    """Print trade history."""
    rows = conn.execute(
        "SELECT date, ticker, action, shares, price, total_cost, account, notes "
        "FROM trades ORDER BY date DESC, trade_id DESC LIMIT 50"
    ).fetchall()

    if not rows:
        print("No trades recorded yet.")
        return

    print(f"\n  {'Date':<12s} {'Action':<6s} {'Ticker':<8s} {'Shares':>8s} "
          f"{'Price':>8s} {'Total':>10s} {'Account':<10s} {'Notes'}")
    print(f"  {'─' * 12} {'─' * 6} {'─' * 8} {'─' * 8} "
          f"{'─' * 8} {'─' * 10} {'─' * 10} {'─' * 20}")
    for row in rows:
        date, ticker, action, shares, price, total, account, notes = row
        print(f"  {date:<12s} {action:<6s} {ticker:<8s} {shares:>8.2f} "
              f"${price:>7,.2f} ${total:>9,.2f} {account:<10s} {notes or ''}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 9 — Holdings Tracker & Drift Engine",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # --- buy ---
    buy_p = sub.add_parser("buy", help="Record a BUY trade")
    buy_p.add_argument("ticker", type=str, help="Ticker symbol")
    buy_p.add_argument("shares", type=float, help="Number of shares")
    buy_p.add_argument("--price", type=float, required=True, help="Price per share")
    buy_p.add_argument("--account", type=str, required=True,
                       choices=["taxable", "roth_ira", "roth"],
                       help="Account: taxable or roth_ira")
    buy_p.add_argument("--date", type=str, default=None, help="Trade date (YYYY-MM-DD)")
    buy_p.add_argument("--notes", type=str, default="", help="Optional notes")
    buy_p.add_argument("--db", type=str, default=None, help="DB path override")
    buy_p.add_argument("--config", type=str, default=None, help="Config path override")

    # --- sell ---
    sell_p = sub.add_parser("sell", help="Record a SELL trade")
    sell_p.add_argument("ticker", type=str, help="Ticker symbol")
    sell_p.add_argument("shares", type=float, help="Number of shares")
    sell_p.add_argument("--price", type=float, required=True, help="Price per share")
    sell_p.add_argument("--account", type=str, required=True,
                        choices=["taxable", "roth_ira", "roth"],
                        help="Account: taxable or roth_ira")
    sell_p.add_argument("--date", type=str, default=None, help="Trade date (YYYY-MM-DD)")
    sell_p.add_argument("--notes", type=str, default="", help="Optional notes")
    sell_p.add_argument("--db", type=str, default=None, help="DB path override")
    sell_p.add_argument("--config", type=str, default=None, help="Config path override")

    # --- import ---
    imp_p = sub.add_parser("import", help="Import trades from CSV")
    imp_p.add_argument("csv_file", type=str, help="Path to CSV file")
    imp_p.add_argument("--db", type=str, default=None, help="DB path override")
    imp_p.add_argument("--config", type=str, default=None, help="Config path override")

    # --- status ---
    stat_p = sub.add_parser("status", help="Show holdings vs target")
    stat_p.add_argument("--db", type=str, default=None, help="DB path override")
    stat_p.add_argument("--config", type=str, default=None, help="Config path override")

    # --- trades ---
    trades_p = sub.add_parser("trades", help="Show trade history")
    trades_p.add_argument("--db", type=str, default=None, help="DB path override")

    # --- refresh ---
    ref_p = sub.add_parser("refresh", help="Refresh prices and recompute holdings")
    ref_p.add_argument("--db", type=str, default=None, help="DB path override")
    ref_p.add_argument("--config", type=str, default=None, help="Config path override")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Resolve paths
    db_path = Path(args.db) if hasattr(args, 'db') and args.db else DB_PATH
    config_path = Path(args.config) if hasattr(args, 'config') and args.config else CONFIG_PATH

    conn = init_holdings_tables(db_path)
    cfg = load_config(config_path)

    if args.command == "buy":
        result = record_trade(conn, args.ticker, "BUY", args.shares,
                              args.price, args.account, args.date, args.notes)
        print(json.dumps(result, indent=2))
        if result["status"] == "ok":
            refresh_holdings(conn, cfg)
            print("\nHoldings updated.")

    elif args.command == "sell":
        result = record_trade(conn, args.ticker, "SELL", args.shares,
                              args.price, args.account, args.date, args.notes)
        print(json.dumps(result, indent=2))
        if result["status"] == "ok":
            refresh_holdings(conn, cfg)
            print("\nHoldings updated.")

    elif args.command == "import":
        result = import_trades_csv(conn, args.csv_file)
        print(json.dumps(result, indent=2))
        if result["imported"] > 0:
            refresh_holdings(conn, cfg)
            print(f"\n{result['imported']} trades imported. Holdings updated.")

    elif args.command == "status":
        cli_status(conn, cfg)

    elif args.command == "trades":
        cli_trades(conn)

    elif args.command == "refresh":
        result = refresh_holdings(conn, cfg)
        print(f"Holdings refreshed: {result['positions']} positions, "
              f"portfolio=${result['portfolio_market_value']:,.2f}")

    conn.close()


if __name__ == "__main__":
    main()
