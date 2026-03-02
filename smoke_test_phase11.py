"""
smoke_test_phase11.py — Phase 11: First Live Deployment Validation
===================================================================
Validates:
  1. Full pipeline ran with live data (current_allocation.json is fresh)
  2. Allocation JSON has valid structure with positions, dollars, accounts
  3. Holdings tracker was seeded (trades table has entries)
  4. deployment_day1.py produced deployment_trades.csv
  5. Trade plan accounts balance correctly ($100K taxable + $44K Roth)
  6. All target tickers have valid prices and share counts
"""

import csv
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
PASS = 0
FAIL = 0
TESTS = []


def test(name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    icon = "\u2705" if condition else "\u274c"
    msg = f"  {icon} {name}"
    if detail and not condition:
        msg += f" \u2014 {detail}"
    print(msg)
    TESTS.append({"name": name, "status": status, "detail": detail})


# ===========================================================================
# TEST 1: current_allocation.json freshness and structure
# ===========================================================================
print("=" * 70)
print("TEST 1: current_allocation.json — live data output")
print("=" * 70)

alloc_path = BASE_DIR / "current_allocation.json"
test("current_allocation.json exists", alloc_path.exists())

alloc = {}
if alloc_path.exists():
    with open(alloc_path) as f:
        alloc = json.load(f)

    test("has 'date' field", "date" in alloc)
    test("has 'regime' field", "regime" in alloc)
    test("has 'positions' dict", "positions" in alloc and isinstance(alloc["positions"], dict))
    test("has 'total_portfolio' field", "total_portfolio" in alloc)

    # Check date is recent (within last 3 days to handle weekends)
    if "date" in alloc:
        alloc_date = date.fromisoformat(alloc["date"])
        days_old = (date.today() - alloc_date).days
        test("allocation date is recent (<=3 days old)",
             days_old <= 3,
             f"Date is {alloc['date']} ({days_old} days old)")

    # Regime
    if "regime" in alloc:
        test("regime is valid",
             alloc["regime"] in ("offense", "defense", "panic"),
             f"Got: {alloc['regime']}")

    # Portfolio values
    test("total_portfolio = $144,000",
         alloc.get("total_portfolio") == 144000,
         f"Got: {alloc.get('total_portfolio')}")
    test("taxable_account = $100,000",
         alloc.get("taxable_account") == 100000,
         f"Got: {alloc.get('taxable_account')}")
    test("roth_ira_account = $44,000",
         alloc.get("roth_ira_account") == 44000,
         f"Got: {alloc.get('roth_ira_account')}")


# ===========================================================================
# TEST 2: Positions validation
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 2: Position allocations")
print("=" * 70)

positions = alloc.get("positions", {})
test("at least 10 positions", len(positions) >= 10, f"Got {len(positions)}")

# Check each position has required fields
required_fields = ["pct", "total_dollars", "taxable_dollars", "roth_dollars", "account"]
missing = []
for ticker, pos in positions.items():
    for field in required_fields:
        if field not in pos:
            missing.append(f"{ticker}.{field}")

test("all positions have required fields",
     len(missing) == 0,
     f"Missing: {missing[:5]}")

# Validate dollar totals
taxable_sum = sum(p["taxable_dollars"] for p in positions.values())
roth_sum = sum(p["roth_dollars"] for p in positions.values())
total_sum = sum(p["total_dollars"] for p in positions.values())

test("taxable allocations sum to ~$100K",
     abs(taxable_sum - 100000) < 10,
     f"Got ${taxable_sum:,.2f}")

test("roth allocations sum to ~$44K",
     abs(roth_sum - 44000) < 10,
     f"Got ${roth_sum:,.2f}")

test("total allocations sum to ~$144K",
     abs(total_sum - 144000) < 10,
     f"Got ${total_sum:,.2f}")

# Check weight percentages sum to ~100%
pct_sum = sum(p["pct"] for p in positions.values())
test("weight percentages sum to ~100%",
     abs(pct_sum - 100.0) < 2.0,
     f"Got {pct_sum:.1f}%")


# ===========================================================================
# TEST 3: Holdings tracker seeded
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 3: Holdings tracker seeded with cash positions")
print("=" * 70)

db_path = BASE_DIR / "rotation_system.db"
test("rotation_system.db exists", db_path.exists())

if db_path.exists():
    conn = sqlite3.connect(str(db_path))

    # Check trades table
    cur = conn.execute("SELECT COUNT(*) FROM trades")
    trade_count = cur.fetchone()[0]
    test("trades table has entries",
         trade_count >= 2,
         f"Got {trade_count} trades")

    # Check for SGOV and JAAA seed trades
    cur = conn.execute("SELECT ticker, action, account, shares FROM trades WHERE notes LIKE '%Day 0%'")
    seed_trades = cur.fetchall()
    seed_tickers = {(r[0], r[2]) for r in seed_trades}

    test("SGOV seeded in taxable",
         ("SGOV", "taxable") in seed_tickers,
         f"Found: {seed_tickers}")
    test("JAAA seeded in roth_ira",
         ("JAAA", "roth_ira") in seed_tickers,
         f"Found: {seed_tickers}")

    # Check holdings table
    cur = conn.execute("SELECT COUNT(*) FROM holdings WHERE shares > 0")
    holdings_count = cur.fetchone()[0]
    test("holdings table has active positions",
         holdings_count >= 2,
         f"Got {holdings_count} positions")

    conn.close()


# ===========================================================================
# TEST 4: deployment_trades.csv
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 4: deployment_trades.csv trade plan")
print("=" * 70)

csv_path = BASE_DIR / "deployment_trades.csv"
test("deployment_trades.csv exists", csv_path.exists())

trades = []
if csv_path.exists():
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        trades = list(reader)

    test("has sell orders (SGOV + JAAA)",
         sum(1 for t in trades if t["action"] == "SELL") >= 2,
         f"Sells: {[t['ticker'] for t in trades if t['action'] == 'SELL']}")

    buy_trades = [t for t in trades if t["action"] == "BUY"]
    test("has 10+ buy orders",
         len(buy_trades) >= 10,
         f"Got {len(buy_trades)} buys")

    # All buys have positive shares and price
    invalid_buys = [t for t in buy_trades
                    if float(t["shares"]) <= 0 or float(t["price"]) <= 0]
    test("all buy orders have positive shares and price",
         len(invalid_buys) == 0,
         f"Invalid: {invalid_buys[:3]}")

    # Taxable buys sum to roughly $100K
    taxable_buys = sum(float(t["actual_dollars"]) for t in buy_trades
                       if t["account"] == "taxable")
    test("taxable buys deploy ~$100K",
         abs(taxable_buys - 100000) < 2000,
         f"Got ${taxable_buys:,.2f}")

    # Roth buys sum to roughly $44K
    roth_buys = sum(float(t["actual_dollars"]) for t in buy_trades
                    if t["account"] == "roth_ira")
    test("roth buys deploy ~$44K",
         abs(roth_buys - 44000) < 2000,
         f"Got ${roth_buys:,.2f}")

    # Total deployment is > 95% of portfolio
    total_deployed = taxable_buys + roth_buys
    deployment_pct = total_deployed / 144000 * 100
    test("total deployment > 95% of portfolio",
         deployment_pct > 95,
         f"Got {deployment_pct:.1f}%")

    # Each trade has an account field
    missing_acct = [t for t in trades if t["account"] not in ("taxable", "roth_ira")]
    test("all trades have valid account field",
         len(missing_acct) == 0,
         f"Invalid: {[t['ticker'] for t in missing_acct]}")

    # Prices are reasonable (> $1, < $5000)
    bad_prices = [t for t in buy_trades
                  if float(t["price"]) < 1 or float(t["price"]) > 5000]
    test("all prices are reasonable ($1 - $5000)",
         len(bad_prices) == 0,
         f"Suspicious: {[(t['ticker'], t['price']) for t in bad_prices]}")


# ===========================================================================
# TEST 5: Alerts output
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 5: Alert system output")
print("=" * 70)

alerts_path = BASE_DIR / "alerts.json"
test("alerts.json exists", alerts_path.exists())

if alerts_path.exists():
    with open(alerts_path) as f:
        alerts = json.load(f)
    test("alerts.json is valid JSON",
         isinstance(alerts, (list, dict)))


# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
total = PASS + FAIL
print(f"Phase 11 Smoke Test: {PASS}/{total} passed, {FAIL} failed")
print("=" * 70)

if FAIL > 0:
    print("\nFailed tests:")
    for t in TESTS:
        if t["status"] == "FAIL":
            print(f"  \u274c {t['name']}: {t['detail']}")

sys.exit(0 if FAIL == 0 else 1)
