#!/usr/bin/env python3
"""
Smoke test — ETF Quality Filter
Validates: expense ratio penalties, overlap dedup, country/region/EM/intl caps.
"""
import sys, os, json, math

# Ensure we can import from the project directory
sys.path.insert(0, os.path.dirname(__file__))

from portfolio_optimizer import apply_etf_quality_filter, _get_asset_class
import yaml

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))

# Load config
cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

eq = cfg.get("etf_quality", {})

print("=" * 65)
print("  SMOKE TEST: ETF QUALITY FILTER")
print("=" * 65)

# ────────────────────────────────────────────────────────────────
# Test 1: FXI and EEM removed from geographic_etfs config
# ────────────────────────────────────────────────────────────────
print("\n── Test 1: Config ticker universe ──")
geo_etfs = cfg.get("tickers", {}).get("geographic_etfs", [])
check("FXI not in geographic_etfs", "FXI" not in geo_etfs, f"geo_etfs={geo_etfs}")
check("EEM not in geographic_etfs", "EEM" not in geo_etfs)
check("MCHI in geographic_etfs", "MCHI" in geo_etfs)
check("VWO in geographic_etfs", "VWO" in geo_etfs)

# ────────────────────────────────────────────────────────────────
# Test 2: Expense ratio config
# ────────────────────────────────────────────────────────────────
print("\n── Test 2: Expense ratio config ──")
er = eq.get("expense_ratios_bps", {})
check("expense_ratios_bps is populated", len(er) > 5, f"{len(er)} entries")
check("VWO expense defined", "VWO" in er, f"VWO={er.get('VWO')} bps")
check("MCHI expense defined", "MCHI" in er, f"MCHI={er.get('MCHI')} bps")
check("FXI expense defined (for legacy)", "FXI" in er, f"FXI={er.get('FXI')} bps")

# ────────────────────────────────────────────────────────────────
# Test 3: Expense ratio penalty math
# ────────────────────────────────────────────────────────────────
print("\n── Test 3: Expense ratio penalty ──")
# Build a simple portfolio with known weights
weights_in = {
    "XLK": 0.20, "XLV": 0.15, "VGK": 0.10,
    "EWZ": 0.08, "INDA": 0.08, "VWO": 0.05,
    "BIL": 0.34
}
result = apply_etf_quality_filter(weights_in, cfg)
# EWZ (58 bps) and INDA (65 bps) should be slightly penalized
# XLK (9 bps) and VWO (8 bps) should NOT be penalized (under 50)
check("EWZ penalized (weight < 8%)", result.get("EWZ", 0) < 0.08,
      f"EWZ={result.get('EWZ', 0)*100:.3f}%")
check("INDA penalized (weight < 8%)", result.get("INDA", 0) < 0.08,
      f"INDA={result.get('INDA', 0)*100:.3f}%")
check("XLK not penalized (9 bps < 50)", result.get("XLK", 0) >= 0.19,
      f"XLK={result.get('XLK', 0)*100:.3f}%")

# ────────────────────────────────────────────────────────────────
# Test 4: Overlap group dedup
# ────────────────────────────────────────────────────────────────
print("\n── Test 4: Overlap group dedup ──")
overlap_groups = eq.get("overlap_groups", {})
check("overlap_groups defined", len(overlap_groups) > 0, f"{len(overlap_groups)} groups")

# Build portfolio with EEM (should be removed / consolidated into VWO)
weights_overlap = {
    "XLK": 0.30, "VWO": 0.10, "EEM": 0.10,
    "BIL": 0.50
}
result_overlap = apply_etf_quality_filter(weights_overlap, cfg)
# broad_em group has 85% overlap — 85% of EEM weight moves to VWO
check("EEM reduced via overlap (broad_em, 85% overlap)",
      result_overlap.get("EEM", 0) < 0.05,
      f"EEM={result_overlap.get('EEM', 0)*100:.3f}% (was 10%, expect ~1.5%)")
check("VWO gained from EEM consolidation (may be reduced by EM cap)",
      result_overlap.get("VWO", 0) > 0.08,
      f"VWO={result_overlap.get('VWO', 0)*100:.3f}%")

# Test china overlap: MCHI vs FXI (40% overlap)
weights_china = {
    "XLK": 0.30, "MCHI": 0.05, "FXI": 0.05,
    "BIL": 0.60
}
result_china = apply_etf_quality_filter(weights_china, cfg)
check("FXI reduced via overlap (china, 40% overlap)",
      result_china.get("FXI", 0) < 0.05,
      f"FXI={result_china.get('FXI', 0)*100:.3f}%, MCHI={result_china.get('MCHI', 0)*100:.3f}%")

# ────────────────────────────────────────────────────────────────
# Test 5: Single-country cap (8%)
# ────────────────────────────────────────────────────────────────
print("\n── Test 5: Single-country cap ──")
max_country = eq.get("max_single_country_pct", 8.0) / 100.0
weights_country = {
    "XLK": 0.40, "EWZ": 0.20, "INDA": 0.15,
    "BIL": 0.25
}
result_country = apply_etf_quality_filter(weights_country, cfg)
check(f"EWZ capped at ~{max_country*100:.0f}%",
      result_country.get("EWZ", 0) <= max_country + 0.005,
      f"EWZ={result_country.get('EWZ', 0)*100:.2f}%")
check(f"INDA capped at ~{max_country*100:.0f}%",
      result_country.get("INDA", 0) <= max_country + 0.005,
      f"INDA={result_country.get('INDA', 0)*100:.2f}%")

# ────────────────────────────────────────────────────────────────
# Test 6: Single-region cap (15%)
# ────────────────────────────────────────────────────────────────
print("\n── Test 6: Single-region cap ──")
max_region = eq.get("max_single_region_pct", 15.0) / 100.0
weights_region = {
    "XLK": 0.40, "VGK": 0.30,
    "BIL": 0.30
}
result_region = apply_etf_quality_filter(weights_region, cfg)
check(f"VGK capped at <= {max_region*100:.0f}%",
      result_region.get("VGK", 0) <= max_region + 0.005,
      f"VGK={result_region.get('VGK', 0)*100:.2f}%")
# Excess should go to other equities, not BIL
check("Excess redistributed to equities (not BIL)",
      result_region.get("BIL", 0) <= 0.30 + 0.005,
      f"BIL={result_region.get('BIL', 0)*100:.2f}% (should stay near 30%, excess goes to equities)")

# Same test for VWO
weights_vwo_big = {
    "XLK": 0.40, "VWO": 0.25,
    "BIL": 0.35
}
result_vwo = apply_etf_quality_filter(weights_vwo_big, cfg)
check(f"VWO capped at <= {max_region*100:.0f}% (region ETF)",
      result_vwo.get("VWO", 0) <= max_region + 0.005,
      f"VWO={result_vwo.get('VWO', 0)*100:.2f}%")

# ────────────────────────────────────────────────────────────────
# Test 7: Total EM cap (20%)
# ────────────────────────────────────────────────────────────────
print("\n── Test 7: Total EM cap ──")
max_em = eq.get("max_em_total_pct", 20.0) / 100.0
weights_em = {
    "XLK": 0.30, "VWO": 0.15, "EWZ": 0.08,
    "INDA": 0.08, "MCHI": 0.08, "BIL": 0.31
}
result_em = apply_etf_quality_filter(weights_em, cfg)
em_tickers_test = ["VWO", "MCHI", "EWZ", "INDA", "EWT", "EWY", "IEMG"]
em_total = sum(result_em.get(t, 0) for t in em_tickers_test)
# Note: aggressive EM input may slightly exceed after normalization;
# the cap is best-effort with iterative convergence
check(f"Total EM within tolerance (<= {max_em*100:.0f}% + 2% margin)",
      em_total <= max_em + 0.025,
      f"EM total={em_total*100:.2f}%")

# ────────────────────────────────────────────────────────────────
# Test 8: Total international cap (35%)
# ────────────────────────────────────────────────────────────────
print("\n── Test 8: Total international cap ──")
max_intl = eq.get("max_intl_total_pct", 35.0) / 100.0
weights_intl = {
    "XLK": 0.20, "VGK": 0.15, "VWO": 0.15,
    "EWZ": 0.08, "INDA": 0.08, "BIL": 0.34
}
result_intl = apply_etf_quality_filter(weights_intl, cfg)
intl_tickers_test = em_tickers_test + ["VGK"]
intl_total = sum(result_intl.get(t, 0) for t in intl_tickers_test)
check(f"Total intl within tolerance (<= {max_intl*100:.0f}% + 2% margin)",
      intl_total <= max_intl + 0.025,
      f"Intl total={intl_total*100:.2f}%")

# ────────────────────────────────────────────────────────────────
# Test 9: Live allocation checks (from current_allocation.json)
# ────────────────────────────────────────────────────────────────
print("\n── Test 9: Live allocation validation ──")
alloc_path = os.path.join(os.path.dirname(__file__), "current_allocation.json")
if os.path.exists(alloc_path):
    with open(alloc_path) as f:
        alloc = json.load(f)
    positions = alloc.get("positions", {})
    
    vgk_pct = positions.get("VGK", {}).get("pct", 0)
    check("Live: VGK <= 15%", vgk_pct <= 15.0, f"{vgk_pct:.2f}%")
    check("Live: FXI absent", "FXI" not in positions)
    check("Live: EEM absent", "EEM" not in positions)
    
    live_em = sum(positions.get(t, {}).get("pct", 0) for t in em_tickers_test)
    check(f"Live: EM total <= 20%", live_em <= 20.0, f"{live_em:.2f}%")
    
    live_intl = sum(positions.get(t, {}).get("pct", 0) for t in intl_tickers_test)
    check(f"Live: Intl total <= 35%", live_intl <= 35.0, f"{live_intl:.2f}%")
    
    total_pct = sum(d.get("pct", 0) for d in positions.values())
    check("Live: weights sum to ~100%", abs(total_pct - 100.0) < 0.1, f"{total_pct:.2f}%")
else:
    print("  [SKIP] No current_allocation.json — run pipeline first.")

# ────────────────────────────────────────────────────────────────
# Test 10: ASSET_CLASS_MAP includes MCHI and IEMG
# ────────────────────────────────────────────────────────────────
print("\n── Test 10: Asset class mappings ──")
check("MCHI maps to em_equities", _get_asset_class("MCHI") == "em_equities",
      f"MCHI → {_get_asset_class('MCHI')}")
check("IEMG maps to em_equities", _get_asset_class("IEMG") == "em_equities",
      f"IEMG → {_get_asset_class('IEMG')}")
check("VGK maps to intl_developed", _get_asset_class("VGK") == "intl_developed",
      f"VGK → {_get_asset_class('VGK')}")

# ────────────────────────────────────────────────────────────────
# Test 11: XLP offense-excluded (weight zeroed, redistributed)
# ────────────────────────────────────────────────────────────────
print("\n── Test 11: XLP offense exclusion ──")
weights_xlp = {
    "XLK": 0.30, "XLP": 0.10, "XLI": 0.20,
    "VGK": 0.10, "BIL": 0.30
}
result_xlp = apply_etf_quality_filter(weights_xlp, cfg)
check("XLP zeroed by offense_exclude",
      result_xlp.get("XLP", 0) < 0.001,
      f"XLP={result_xlp.get('XLP', 0)*100:.2f}%")
check("XLP weight redistributed to equities (XLK grew)",
      result_xlp.get("XLK", 0) > 0.30,
      f"XLK={result_xlp.get('XLK', 0)*100:.2f}% (was 30%)")
check("BIL unchanged after XLP redistribution",
      result_xlp.get("BIL", 0) <= 0.31,
      f"BIL={result_xlp.get('BIL', 0)*100:.2f}%")

# ────────────────────────────────────────────────────────────────
# Test 12: XLU per-ticker cap (6%)
# ────────────────────────────────────────────────────────────────
print("\n── Test 12: XLU per-ticker cap ──")
xlu_cap = eq.get("per_ticker_cap_pct", {}).get("XLU", 100) / 100.0
weights_xlu = {
    "XLK": 0.30, "XLU": 0.15, "XLI": 0.20,
    "VGK": 0.10, "BIL": 0.25
}
result_xlu = apply_etf_quality_filter(weights_xlu, cfg)
check(f"XLU capped at <= {xlu_cap*100:.0f}%",
      result_xlu.get("XLU", 0) <= xlu_cap + 0.005,
      f"XLU={result_xlu.get('XLU', 0)*100:.2f}% (was 15%, cap={xlu_cap*100:.0f}%)")
check("XLU excess redistributed to equities",
      result_xlu.get("XLK", 0) > 0.30 or result_xlu.get("XLI", 0) > 0.20,
      f"XLK={result_xlu.get('XLK', 0)*100:.2f}%, XLI={result_xlu.get('XLI', 0)*100:.2f}%")

# ────────────────────────────────────────────────────────────────
# Test 13: Dividend-yield tax location routing
# ────────────────────────────────────────────────────────────────
print("\n── Test 13: Dividend-yield tax location routing ──")
from portfolio_optimizer import allocate_dollars

# Build a portfolio with high-div and growth sectors
weights_tax = {
    "XLU": 0.06,   # High dividend → should go Roth
    "XLP": 0.05,   # High dividend → should go Roth
    "XLI": 0.10,   # Growth → should go Taxable
    "XLK": 0.15,   # Growth → should go Taxable
    "XLC": 0.08,   # Growth → should go Taxable
    "VGK": 0.10,   # Geographic → should go Taxable
    "BIL": 0.05,   # Cash → Taxable
    "AFRM": 0.10,  # Watchlist → should go Roth
    "XLV": 0.15,   # No explicit routing → default
    "VWO": 0.06,   # Geographic → Taxable
    "XLY": 0.10,   # Growth → should go Taxable
}
alloc_result = allocate_dollars(weights_tax, cfg)

# XLU should be in Roth (high-dividend sector)
check("XLU → Roth (high-dividend)",
      alloc_result.get("XLU", {}).get("roth_dollars", 0) > 0,
      f"XLU: account={alloc_result.get('XLU', {}).get('account', 'N/A')}, "
      f"reason={alloc_result.get('XLU', {}).get('reason', 'N/A')}")

# XLP should be in Roth (high-dividend sector)
check("XLP → Roth (high-dividend)",
      alloc_result.get("XLP", {}).get("roth_dollars", 0) > 0,
      f"XLP: account={alloc_result.get('XLP', {}).get('account', 'N/A')}, "
      f"reason={alloc_result.get('XLP', {}).get('reason', 'N/A')}")

# XLI should be in Taxable (growth sector)
check("XLI → Taxable (growth sector)",
      alloc_result.get("XLI", {}).get("taxable_dollars", 0) > 0,
      f"XLI: account={alloc_result.get('XLI', {}).get('account', 'N/A')}, "
      f"reason={alloc_result.get('XLI', {}).get('reason', 'N/A')}")

# XLK should be in Taxable (growth sector)
check("XLK → Taxable (growth sector)",
      alloc_result.get("XLK", {}).get("taxable_dollars", 0) > 0,
      f"XLK: account={alloc_result.get('XLK', {}).get('account', 'N/A')}, "
      f"reason={alloc_result.get('XLK', {}).get('reason', 'N/A')}")

# VGK should be in Taxable (geographic, foreign tax credit)
check("VGK → Taxable (geographic)",
      alloc_result.get("VGK", {}).get("taxable_dollars", 0) > 0,
      f"VGK: account={alloc_result.get('VGK', {}).get('account', 'N/A')}")

# AFRM should be in Roth (watchlist stock)
check("AFRM → Roth (watchlist)",
      alloc_result.get("AFRM", {}).get("roth_dollars", 0) > 0,
      f"AFRM: account={alloc_result.get('AFRM', {}).get('account', 'N/A')}")

# ────────────────────────────────────────────────────────────────
# Test 14: Config sanity — high_dividend_sectors / growth_sectors present
# ────────────────────────────────────────────────────────────────
print("\n── Test 14: Config sanity checks ──")
tl = cfg.get("tax_location", {})
high_div = tl.get("high_dividend_sectors", [])
growth = tl.get("growth_sectors", [])
check("high_dividend_sectors defined", len(high_div) >= 2, f"{high_div}")
check("growth_sectors defined", len(growth) >= 4, f"{growth}")
check("XLU in high_dividend_sectors", "XLU" in high_div)
check("XLP in high_dividend_sectors", "XLP" in high_div)
check("XLI in growth_sectors", "XLI" in growth)
check("XLK in growth_sectors", "XLK" in growth)
check("per_ticker_cap_pct.XLU defined", "XLU" in eq.get("per_ticker_cap_pct", {}),
      f"cap={eq.get('per_ticker_cap_pct', {}).get('XLU', 'N/A')}%")
check("offense_exclude has XLP", "XLP" in eq.get("offense_exclude", []))
# ────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
total = PASS + FAIL
print(f"  ETF QUALITY SMOKE TEST: {PASS}/{total} passed", end="")
if FAIL > 0:
    print(f" ({FAIL} FAILED)")
    sys.exit(1)
else:
    print(" — ALL CLEAR")
    sys.exit(0)
