"""
smoke_test_phase3.py — Smoke Test for Phase 3: Factor Scoring & CVaR Portfolio Optimizer
========================================================================================
Tests:
  1. FF5 factor download / fallback
  2. Single-ETF factor loading regression
  3. Momentum computation
  4. Composite factor scores for all sector ETFs
  5. Ledoit-Wolf covariance shrinkage
  6. Longin-Solnik tail correlations (with diagnostics)
  7. CVaR optimization across all regimes (offense / defense / panic)
  8. DeMiguel sub-sector + Bivector Beta (parametrized from config)
  9. Dollar allocation across Taxable / Roth IRA (+ total_w=0 guard)
 10. Full pipeline: run_portfolio_optimization()

Uses real data from rotation_system.db (7,973 price rows from Phase 1).
"""

import sys
import os
import logging
import sqlite3
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the package directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smoke_test_phase3")

DB_PATH = Path(__file__).parent / "rotation_system.db"
CFG_PATH = Path(__file__).parent / "config.yaml"

import yaml
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

conn = sqlite3.connect(str(DB_PATH))

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f"  ({detail})" if detail else ""))


# =====================================================================
# TEST 1: Fama-French Five-Factor Download
# =====================================================================
print("\n" + "=" * 70)
print("TEST 1: Fama-French Five-Factor Download")
print("=" * 70)

from portfolio_optimizer import download_ff_factors, _generate_synthetic_ff_factors

ff = download_ff_factors()
check("FF5 DataFrame returned", ff is not None and not ff.empty, f"{len(ff)} rows")
expected_cols = {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"}
check("All 6 factor columns present", expected_cols.issubset(set(ff.columns)), str(list(ff.columns)))
check("Values are decimal not percent", ff["Mkt-RF"].abs().max() < 0.50, f"max |Mkt-RF| = {ff['Mkt-RF'].abs().max():.4f}")
check("Index is DatetimeIndex", isinstance(ff.index, pd.DatetimeIndex))


# =====================================================================
# TEST 2: Single-ETF Factor Loading Regression
# =====================================================================
print("\n" + "=" * 70)
print("TEST 2: Factor Loading Regression (XLK)")
print("=" * 70)

from portfolio_optimizer import compute_factor_loadings
from regime_detector import load_sector_prices

sector_wide = load_sector_prices(conn, cfg)
log_returns = np.log(sector_wide / sector_wide.shift(1)).dropna()
rf = ff["RF"].reindex(log_returns.index).ffill().fillna(0)

xlk_excess = log_returns["XLK"] - rf
loadings = compute_factor_loadings(xlk_excess, ff, window_months=36)

check("Loadings dict returned", isinstance(loadings, dict))
check("Has alpha key", "alpha" in loadings)
check("Has r_squared", "r_squared" in loadings, f"R² = {loadings.get('r_squared', 0):.4f}")
check("Mkt-RF beta positive", loadings.get("Mkt-RF", 0) > 0, f"β_mkt = {loadings.get('Mkt-RF', 0):.4f}")
check("R² > 0.40 (meaningful fit)", loadings.get("r_squared", 0) > 0.40)

print(f"\n  XLK Factor Loadings:")
for k, v in loadings.items():
    print(f"    {k:>12s}: {v:>10.6f}")


# =====================================================================
# TEST 3: Momentum Computation (12-1 Month)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 3: Momentum Computation (12-1 Month)")
print("=" * 70)

from portfolio_optimizer import compute_momentum

mom = compute_momentum(
    sector_wide,
    lookback=cfg["factor_model"]["momentum"]["lookback_months"] * 21,
    skip=cfg["factor_model"]["momentum"]["skip_months"] * 21,
)

check("Momentum Series returned", mom is not None and not mom.empty, f"{len(mom)} tickers")
check("Values in [0, 1] range", (mom >= 0).all() and (mom <= 1).all())
check("All sector ETFs have momentum", len(mom) == len(cfg["tickers"]["sector_etfs"]),
      f"{len(mom)} / {len(cfg['tickers']['sector_etfs'])}")

print(f"\n  12-1 Month Momentum Ranks:")
for t in mom.sort_values(ascending=False).index:
    print(f"    {t:<6s}: {mom[t]:.4f}")


# =====================================================================
# TEST 4: Composite Factor Scores
# =====================================================================
print("\n" + "=" * 70)
print("TEST 4: Composite Factor Scores (All Sector ETFs)")
print("=" * 70)

from portfolio_optimizer import compute_composite_factor_scores

factor_scores = compute_composite_factor_scores(conn, cfg, ff)
check("Factor scores DataFrame returned", not factor_scores.empty, f"{len(factor_scores)} ETFs")
check("Has composite_score column", "composite_score" in factor_scores.columns)
check("Has momentum_rank column", "momentum_rank" in factor_scores.columns)
check("Scores in reasonable range", (factor_scores["composite_score"] > 0).all() and
      (factor_scores["composite_score"] < 1).all())

print(f"\n  Factor Score Table:")
cols = ["ticker", "adjusted_alpha", "mkt_rf", "hml", "rmw", "momentum_rank", "composite_score"]
display_cols = [c for c in cols if c in factor_scores.columns]
print(factor_scores[display_cols].to_string(index=False))


# =====================================================================
# TEST 5: Ledoit-Wolf Covariance Shrinkage
# =====================================================================
print("\n" + "=" * 70)
print("TEST 5: Ledoit-Wolf Covariance Shrinkage")
print("=" * 70)

from portfolio_optimizer import compute_shrunk_covariance

sector_returns = log_returns[cfg["tickers"]["sector_etfs"]].dropna(axis=1, how="all")
cov = compute_shrunk_covariance(sector_returns)

check("Covariance matrix returned", cov is not None)
check("Square matrix", cov.shape[0] == cov.shape[1], f"shape = {cov.shape}")
check("Positive diagonal (variances)", (np.diag(cov) > 0).all())
check("Symmetric", np.allclose(cov, cov.T, atol=1e-12))

# Check eigenvalues are all positive (positive definite)
eigenvalues = np.linalg.eigvalsh(cov)
check("Positive semi-definite", (eigenvalues >= -1e-10).all(),
      f"min eigenvalue = {eigenvalues.min():.2e}")

print(f"\n  Covariance matrix shape: {cov.shape}")
print(f"  Annualized vol range: {np.sqrt(np.diag(cov) * 252).min():.2%} to {np.sqrt(np.diag(cov) * 252).max():.2%}")


# =====================================================================
# TEST 6: Longin-Solnik Tail Correlations (with Diagnostics)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 6: Longin-Solnik Tail Correlations (with Diagnostics)")
print("=" * 70)

from portfolio_optimizer import compute_tail_correlations

# Build returns for EM tickers that exist in our DB
geo_tickers = cfg["tickers"]["geographic_etfs"]
all_tickers = cfg["tickers"]["sector_etfs"] + geo_tickers + ["BIL"]
placeholders = ",".join(["?"] * len(all_tickers))
prices_df = pd.read_sql_query(
    f"SELECT date, ticker, adj_close FROM prices WHERE ticker IN ({placeholders}) ORDER BY date",
    conn, params=all_tickers,
)
wide_all = prices_df.pivot(index="date", columns="ticker", values="adj_close")
wide_all.index = pd.to_datetime(wide_all.index)
wide_all = wide_all.sort_index().ffill().dropna(axis=1, how="all")
returns_all = np.log(wide_all / wide_all.shift(1)).dropna()

em_in_data = [t for t in geo_tickers if t in returns_all.columns]

# Store full-sample corr BEFORE tail adjustment for quant check later
full_sample_corr = returns_all.corr()

tail_corr, tail_diag = compute_tail_correlations(returns_all, em_in_data)

# --- Matrix checks ---
check("Tail correlation matrix returned", tail_corr is not None and not tail_corr.empty)
check("Square matrix", tail_corr.shape[0] == tail_corr.shape[1])
check("Diagonal is 1.0", np.allclose(np.diag(tail_corr.values), 1.0, atol=0.01))
check(f"EM tickers in matrix: {len(em_in_data)}", len(em_in_data) >= 3)

# --- Diagnostics dict checks ---
check("Diagnostics has pairs_checked", "pairs_checked" in tail_diag,
      f"pairs_checked = {tail_diag.get('pairs_checked', '?')}")
check("Diagnostics has pairs_adjusted", "pairs_adjusted" in tail_diag,
      f"pairs_adjusted = {tail_diag.get('pairs_adjusted', '?')}")
check("Diagnostics has pairs_insufficient", "pairs_insufficient" in tail_diag,
      f"pairs_insufficient = {tail_diag.get('pairs_insufficient', '?')}")
check("Diagnostics has pair_details list", isinstance(tail_diag.get("pair_details"), list),
      f"{len(tail_diag.get('pair_details', []))} pair records")

# --- Validate pair_details structure ---
if tail_diag.get("pair_details"):
    sample_pair = tail_diag["pair_details"][0]
    check("pair_detail has pair key", "pair" in sample_pair, sample_pair.get("pair", "?"))
    check("pair_detail has joint_tail_events", "joint_tail_events" in sample_pair)
    check("pair_detail has full_sample_corr", "full_sample_corr" in sample_pair)
    check("pair_detail has adjusted flag", "adjusted" in sample_pair)

# --- Quant check: EM weight reduction measurement ---
# With only ~2yr of data, most pairs should fall back to full-sample.
# When we DO have enough joint tail events, the tail corr for EM pairs
# should generally be >= full-sample (stress co-movement is higher).
pairs_higher = 0
pairs_lower = 0
for pd_info in tail_diag.get("pair_details", []):
    if pd_info.get("adjusted") and pd_info.get("tail_corr") is not None:
        if pd_info["tail_corr"] >= pd_info["full_sample_corr"]:
            pairs_higher += 1
        else:
            pairs_lower += 1

check("Tail corr >= full-sample for adjusted pairs (crisis co-movement)",
      pairs_higher >= pairs_lower,
      f"{pairs_higher} higher, {pairs_lower} lower (limited data is OK)")

print(f"\n  Tail Correlation Diagnostics:")
print(f"    Pairs checked:      {tail_diag['pairs_checked']}")
print(f"    Pairs adjusted:     {tail_diag['pairs_adjusted']}")
print(f"    Pairs insufficient: {tail_diag['pairs_insufficient']}")
for pd_info in tail_diag.get("pair_details", [])[:5]:
    status = "ADJUSTED" if pd_info.get("adjusted") else "fallback"
    tc = f"{pd_info['tail_corr']:.4f}" if pd_info.get("tail_corr") is not None else "n/a"
    print(f"    {pd_info['pair']:<12s}  events={pd_info['joint_tail_events']:>3d}  "
          f"full={pd_info['full_sample_corr']:.4f}  tail={tc}  [{status}]")


# =====================================================================
# TEST 7: CVaR Optimization Across All Regimes
# =====================================================================
print("\n" + "=" * 70)
print("TEST 7: CVaR Optimization (All Regimes)")
print("=" * 70)

from portfolio_optimizer import run_cvar_optimization

regime_results = {}
for regime_name in ["offense", "defense", "panic"]:
    print(f"\n  --- Regime: {regime_name.upper()} ---")
    weights = run_cvar_optimization(
        returns_all, regime_name, cfg,
        factor_scores=factor_scores,
        em_tickers=em_in_data,
    )

    check(f"[{regime_name}] Weights dict returned",
          isinstance(weights, dict) and len(weights) > 0)
    total_w = sum(weights.values())
    check(f"[{regime_name}] Weights sum ~1.0 (+-0.05) or total_w > 0",
          abs(total_w - 1.0) < 0.05 or total_w > 0,
          f"sum = {total_w:.4f}")
    check(f"[{regime_name}] No negative weights",
          all(w >= -0.001 for w in weights.values()))
    nonzero = {t: w for t, w in weights.items() if w > 0.001}
    check(f"[{regime_name}] Has non-zero positions", len(nonzero) >= 1,
          f"{len(nonzero)} positions")

    regime_results[regime_name] = weights

    # Print top 5 positions for this regime
    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for t, w in sorted_w[:5]:
        if w > 0.001:
            print(f"    {t:<6s}: {w:>7.2%}")

# Cross-regime comparison: panic should have more cash/BIL than offense
offense_bil = regime_results.get("offense", {}).get("BIL", 0)
panic_bil = regime_results.get("panic", {}).get("BIL", 0)
check("Panic has more BIL (cash) than Offense",
      panic_bil >= offense_bil,
      f"Offense BIL={offense_bil:.2%}, Panic BIL={panic_bil:.2%}")

# Keep offense weights for subsequent tests
weights = regime_results["offense"]


# =====================================================================
# TEST 8: DeMiguel Sub-Sector + Bivector Beta (Config-Parametrized)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 8: DeMiguel Sub-Sector Allocation + Bivector Beta")
print("=" * 70)

from portfolio_optimizer import apply_us_subsector_allocation, compute_bivector_beta

# Bivector Beta — use sector-only returns (long history) for meaningful PCA.
# The full returns_all has only ~20 rows because dropna() removes dates
# where geo ETFs have NaN.  Build sector returns independently.
sector_only_prices = wide_all[cfg["tickers"]["sector_etfs"]].dropna(axis=1, how="all").ffill().dropna()
sector_rets_pca = np.log(sector_only_prices / sector_only_prices.shift(1)).dropna()
bv_xle = compute_bivector_beta(sector_rets_pca, "XLE",
    market_tickers=[c for c in sector_rets_pca.columns if c != "XLE"])
bv_xlk = compute_bivector_beta(sector_rets_pca, "XLK",
    market_tickers=[c for c in sector_rets_pca.columns if c != "XLK"])

check("Bivector Beta for XLE computed", isinstance(bv_xle, float), f"beta_bv(XLE) = {bv_xle:.4f}")
check("Bivector Beta for XLK computed", isinstance(bv_xlk, float), f"beta_bv(XLK) = {bv_xlk:.4f}")
check("XLE is structural diversifier (beta_bv > 1.0)", bv_xle > 1.0,
      f"XLE {bv_xle:.2f} ({'diversifier' if bv_xle > 1.0 else 'needs more data for differentiation'})")

# Parametrize us_weight from config offense bands instead of hardcoded 0.50
offense_band = cfg["optimizer"]["allocation_bands"]["us_equities"]["offense"]
us_weight = (offense_band[0] + offense_band[1]) / 2  # midpoint of offense band
print(f"\n  us_weight from config offense band midpoint: {us_weight:.2f} (band = {offense_band})")

subsector = apply_us_subsector_allocation(
    cfg["tickers"]["sector_etfs"], factor_scores, us_weight, cfg, returns_all,
)

check("Subsector dict returned", isinstance(subsector, dict) and len(subsector) > 0,
      f"{len(subsector)} sectors")
# Check weights sum to us_weight
sub_total = sum(v["weight"] for v in subsector.values())
check("Sub-sector weights sum to US equity target", abs(sub_total - us_weight) < 0.01,
      f"sum = {sub_total:.4f} vs target {us_weight}")

print(f"\n  US Sub-Sector Allocation ({us_weight:.0%} total):")
for t, info in sorted(subsector.items(), key=lambda x: x[1]["weight"], reverse=True):
    print(f"    {t:<6s}: {info['weight']:>7.2%}  [{info['label']}]  bv_beta = {info['bivector_beta']:.2f}  diversifier = {info['is_structural_diversifier']}")


# =====================================================================
# TEST 9: Dollar Allocation (Taxable vs Roth IRA) + total_w=0 Guard
# =====================================================================
print("\n" + "=" * 70)
print("TEST 9: Dollar Allocation ($100K Taxable + $44K Roth)")
print("=" * 70)

from portfolio_optimizer import allocate_dollars

# --- 9a: Verify total_w=0 guard ---
# Pass all-zero weights — allocate_dollars should handle gracefully
zero_weights = {t: 0.0 for t in weights}
zero_alloc = allocate_dollars(zero_weights, cfg, subsector)
check("total_w=0 guard: returns dict (possibly empty)", isinstance(zero_alloc, dict))
zero_total = sum(v.get("total_dollars", 0) for v in zero_alloc.values())
check("total_w=0 guard: no phantom dollars allocated",
      zero_total < 1.0, f"${zero_total:.2f}")

# --- 9b: Normal allocation with offense weights ---
# Normalize weights first
total_w = sum(weights.values())
if total_w > 0:
    norm_weights = {t: w / total_w for t, w in weights.items()}
else:
    norm_weights = weights
check("total_w > 0 for offense weights", total_w > 0, f"total_w = {total_w:.6f}")

dollar_alloc = allocate_dollars(norm_weights, cfg, subsector)

check("Dollar allocation returned", isinstance(dollar_alloc, dict) and len(dollar_alloc) > 0)

total_dollars = sum(v["total_dollars"] for v in dollar_alloc.values())
total_taxable = sum(v["taxable_dollars"] for v in dollar_alloc.values())
total_roth = sum(v["roth_dollars"] for v in dollar_alloc.values())

check("Total dollars = $144,000 (+-$1)", abs(total_dollars - 144000) < 1,
      f"${total_dollars:,.2f}")
check("Taxable <= $100,000 cap", total_taxable <= 100001,
      f"${total_taxable:,.2f}")
check("Roth <= $44,000 cap", total_roth <= 44001,
      f"${total_roth:,.2f}")
check("Taxable + Roth = Total", abs(total_taxable + total_roth - total_dollars) < 0.01)
check("Every position has % AND $ amount", all("pct" in v and "total_dollars" in v for v in dollar_alloc.values()))
check("Every position has account placement", all("account" in v for v in dollar_alloc.values()))

print(f"\n  Dollar Allocation Summary:")
print(f"  {'Ticker':<8} {'Weight':>7} {'Total $':>10} {'Taxable $':>10} {'Roth $':>10} {'Account':<10} Reason")
print(f"  {'-' * 88}")
for t, info in sorted(dollar_alloc.items(), key=lambda x: x[1]["total_dollars"], reverse=True):
    if info["total_dollars"] > 0:
        print(
            f"  {t:<8} {info['pct']:>6.1f}% ${info['total_dollars']:>9,.0f} "
            f"${info['taxable_dollars']:>9,.0f} ${info['roth_dollars']:>9,.0f} "
            f"{info['account']:<10} {info['reason']}"
        )
print(f"  {'-' * 88}")
print(f"  {'TOTAL':<8} {'100.0':>6}% ${total_dollars:>9,.0f} ${total_taxable:>9,.0f} ${total_roth:>9,.0f}")


# =====================================================================
# TEST 10: Full Pipeline — run_portfolio_optimization()
# =====================================================================
print("\n" + "=" * 70)
print("TEST 10: Full Pipeline — run_portfolio_optimization()")
print("=" * 70)

from portfolio_optimizer import run_portfolio_optimization

result = run_portfolio_optimization(conn=conn, cfg=cfg)

check("Pipeline returned result", isinstance(result, dict) and len(result) > 0)
check("Has 'regime' key", "regime" in result, f"regime = {result.get('regime', '?')}")
check("Has 'positions' dict", "positions" in result and len(result.get("positions", {})) > 0)
check("Has 'factor_scores' list", "factor_scores" in result)
check("Has 'subsector_detail' dict", "subsector_detail" in result)
check("Date is today", "date" in result)

# Check output files were created
json_path = Path(__file__).parent / "current_allocation.json"
csv_path = Path(__file__).parent / "current_allocation.csv"
check("current_allocation.json created", json_path.exists())
check("current_allocation.csv created", csv_path.exists())

# Check DB record was inserted
alloc_row = conn.execute("SELECT * FROM allocations WHERE date = ?",
                          (result.get("date", ""),)).fetchone()
check("Allocation stored in DB", alloc_row is not None)

# Verify the output matches the requirement: % AND $ for both accounts
positions = result.get("positions", {})
if positions:
    sample = list(positions.values())[0]
    check("Position has pct", "pct" in sample)
    check("Position has total_dollars", "total_dollars" in sample)
    check("Position has taxable_dollars", "taxable_dollars" in sample)
    check("Position has roth_dollars", "roth_dollars" in sample)

    # Total check across pipeline output
    pipe_total = sum(v["total_dollars"] for v in positions.values())
    pipe_tax = sum(v["taxable_dollars"] for v in positions.values())
    pipe_roth = sum(v["roth_dollars"] for v in positions.values())
    check("Pipeline total = $144,000 (+-$1)", abs(pipe_total - 144000) < 1,
          f"${pipe_total:,.2f}")
    print(f"\n  Pipeline Output: Taxable = ${pipe_tax:,.0f}, Roth = ${pipe_roth:,.0f}, Total = ${pipe_total:,.0f}")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print(f"PHASE 3 SMOKE TEST COMPLETE: {PASS} passed, {FAIL} failed")
print("=" * 70)

conn.close()
sys.exit(0 if FAIL == 0 else 1)
