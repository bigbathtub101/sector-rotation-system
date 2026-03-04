#!/usr/bin/env python3
"""
ETF Auto-Selector — Best-in-class ETF selection per exposure slot.

For each exposure slot (e.g. "US Technology", "India equity"), evaluates
candidate ETFs on expense ratio, AUM (liquidity), and holdings breadth,
then selects the optimal fund.

Runs monthly (or on-demand). Results cached to etf_selections.json.
The portfolio optimizer reads from this cache at runtime.

All candidate pools are defined in config.yaml under etf_selector.exposure_slots.
Zero proprietary dependencies — uses yfinance for live data.

Scoring philosophy: COST IS KING.
  - 75% expense ratio (user goal = minimize fee drag)
  - 15% AUM/liquidity (above $200M = "good enough", only penalize tiny funds)
  - 10% holdings breadth (more diversification = bonus, not a dealbreaker)
"""

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("etf_selector")

# Cache file path
CACHE_FILE = os.path.join(os.path.dirname(__file__), "etf_selections.json")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")

# AUM floor: above this, AUM is "good enough" and barely affects score.
# User explicitly said: prefer cheaper funds even if smaller AUM.
AUM_FLOOR_MILLIONS = 200


def _fetch_etf_metadata(tickers: List[str]) -> Dict[str, dict]:
    """
    Fetch expense ratio, AUM, and holdings count for a list of ETFs
    using yfinance. Returns dict of ticker -> {expense_ratio_pct, expense_ratio_bps, aum_millions, ...}.

    IMPORTANT: yfinance netExpenseRatio returns values like 0.08 meaning 0.08%
    (NOT 8%). This is already a percentage, not a decimal fraction.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — using config fallback expense ratios only")
        return {}

    results = {}
    for ticker in tickers:
        try:
            etf = yf.Ticker(ticker)
            info = etf.info or {}

            # --- Expense Ratio ---
            # yfinance fields and their formats:
            #   netExpenseRatio: returned as percentage already (e.g., 0.08 = 0.08%)
            #   annualReportExpenseRatio: sometimes decimal (0.0008), sometimes None
            #   expenseRatio: same ambiguity
            # Strategy: use netExpenseRatio (most reliable), treat as already-percentage.
            er_pct = None
            net_er = info.get("netExpenseRatio")
            if net_er is not None and net_er > 0:
                # netExpenseRatio is already in percentage form:
                # 0.08 means 0.08%, 0.59 means 0.59%
                er_pct = round(net_er, 4)
            else:
                # Fallback to annualReportExpenseRatio (decimal form: 0.0008 = 0.08%)
                ar_er = info.get("annualReportExpenseRatio") or info.get("expenseRatio")
                if ar_er is not None and ar_er > 0:
                    # These are decimal fractions — need to multiply by 100
                    # But check for sanity: if value > 0.5, it's likely already %
                    if ar_er > 0.5:
                        er_pct = round(ar_er, 4)  # already percentage
                    else:
                        er_pct = round(ar_er * 100, 4)  # decimal → percentage

            er_bps = round(er_pct * 100) if er_pct is not None else None

            # --- AUM ---
            aum = info.get("totalAssets") or 0

            results[ticker] = {
                "expense_ratio_pct": er_pct,
                "expense_ratio_bps": er_bps,
                "aum_millions": round(aum / 1e6, 1) if aum > 0 else None,
                "holdings_count": None,  # yfinance doesn't reliably expose this
                "source": "yfinance",
            }
            logger.info("Fetched %s: ER=%.2f%% (%dbps), AUM=$%.0fM",
                        ticker,
                        er_pct or 0,
                        er_bps or 0,
                        (aum or 0) / 1e6)

            # Rate limit: be polite to Yahoo
            time.sleep(0.3)

        except Exception as e:
            logger.warning("Failed to fetch %s: %s", ticker, e)
            results[ticker] = {
                "expense_ratio_pct": None,
                "expense_ratio_bps": None,
                "aum_millions": None,
                "holdings_count": None,
                "source": "error",
            }

    return results


def _score_candidate(
    meta: dict,
    fallback_er_bps: Optional[int] = None,
    fallback_holdings: Optional[int] = None,
) -> float:
    """
    Score an ETF candidate. Lower is better (like golf).

    Scoring weights (COST IS KING for a cost-minimization investor):
      - Expense ratio: 75% of score (primary driver — directly eats returns)
      - AUM/liquidity:  15% of score (above $200M = good enough)
      - Holdings breadth: 10% of score (more = better diversification, but minor)

    Returns a cost score where LOWER = BETTER.
    """
    # Expense ratio (in bps) — the most important factor
    er_bps = meta.get("expense_ratio_bps")
    if er_bps is None:
        er_bps = fallback_er_bps or 50  # assume moderate if unknown

    # AUM in millions — only penalize truly small/illiquid funds
    aum_m = meta.get("aum_millions") or 100  # assume small if unknown

    # Holdings count — more = better diversification
    holdings = meta.get("holdings_count") or fallback_holdings or 100

    # --- Normalize each component to [0, 1] range ---

    # ER: 0 bps = 0.0 (best), 100 bps = 1.0 (worst) — linear
    er_score = min(er_bps / 100.0, 1.0)

    # AUM: Above $200M floor = near-zero penalty. Below = escalating penalty.
    # This ensures $2.8B FLIN isn't punished vs $9.1B INDA — both are well above floor.
    # Score: 0.0 (best, AUM >> floor) to 1.0 (worst, AUM near zero)
    if aum_m >= AUM_FLOOR_MILLIONS:
        # Above floor: tiny residual score that barely matters
        # $200M → 0.05, $1B → 0.03, $10B → 0.015, $100B → 0.005
        aum_score = max(0.0, 0.05 / math.log10(max(aum_m / AUM_FLOOR_MILLIONS, 1.01)))
        aum_score = min(aum_score, 0.05)  # cap at 0.05 for above-floor funds
    else:
        # Below floor: real penalty — linear from 1.0 (near zero) to 0.10 (at floor)
        aum_score = max(0.10, 1.0 - 0.9 * (aum_m / AUM_FLOOR_MILLIONS))

    # Holdings: 500+ = 0.0 (best), 10 = 1.0 (worst) — log scale
    holdings_score = max(0.0, 1.0 - math.log10(max(holdings, 1)) / math.log10(500))

    # Weighted composite (lower = better)
    composite = 0.75 * er_score + 0.15 * aum_score + 0.10 * holdings_score

    return round(composite, 6)


def select_best_etfs(cfg: dict, force_refresh: bool = False) -> Dict[str, dict]:
    """
    For each exposure slot in config, evaluate candidates and select the best ETF.

    Uses a two-pass strategy:
      1. Fetch live metadata from yfinance
      2. Override with config-defined expense ratios where available
         (config values are hand-verified from prospectuses and are more reliable
          than yfinance for fee waivers, capped ERs, etc.)

    Returns dict of slot_name -> {
        "selected": ticker,
        "candidates": {ticker: {score, expense_ratio_bps, aum_millions}},
        "reason": str,
    }
    """
    slots = cfg.get("etf_selector", {}).get("exposure_slots", {})
    if not slots:
        logger.warning("No exposure_slots defined in config — nothing to select")
        return {}

    # Gather all unique candidate tickers
    all_candidates = set()
    for slot_cfg in slots.values():
        all_candidates.update(slot_cfg.get("candidates", []))

    logger.info("Evaluating %d unique ETF candidates across %d exposure slots",
                len(all_candidates), len(slots))

    # Fetch live metadata from yfinance
    live_meta = _fetch_etf_metadata(list(all_candidates))

    # Get config fallback expense ratios (hand-verified, from prospectuses)
    config_er = cfg.get("etf_quality", {}).get("expense_ratios_bps", {})

    # OVERRIDE yfinance ER with config values where available.
    # Config values are more trustworthy (manually verified from prospectuses).
    # Example: SGOV yfinance shows 9bps but actual net ER is 3bps (fee waiver).
    for ticker in all_candidates:
        if ticker in config_er and ticker in live_meta:
            config_bps = config_er[ticker]
            live_bps = live_meta[ticker].get("expense_ratio_bps")
            if live_bps != config_bps:
                logger.info("ER override %s: yfinance=%sbps → config=%dbps (prospectus verified)",
                            ticker, live_bps, config_bps)
            live_meta[ticker]["expense_ratio_bps"] = config_bps
            live_meta[ticker]["expense_ratio_pct"] = round(config_bps / 100, 4)

    # Score and select per slot
    selections = {}
    for slot_name, slot_cfg in slots.items():
        candidates = slot_cfg.get("candidates", [])
        asset_class = slot_cfg.get("asset_class", "us_equities")
        fallback_holdings = slot_cfg.get("typical_holdings", 100)

        scored = {}
        for ticker in candidates:
            meta = live_meta.get(ticker, {})
            fallback_er = config_er.get(ticker)
            score = _score_candidate(meta, fallback_er, fallback_holdings)
            scored[ticker] = {
                "score": score,
                "expense_ratio_bps": meta.get("expense_ratio_bps") or fallback_er,
                "aum_millions": meta.get("aum_millions"),
            }

        # Sort by score ascending (lower = better)
        ranked = sorted(scored.items(), key=lambda x: x[1]["score"])

        if ranked:
            winner_ticker = ranked[0][0]
            winner_data = ranked[0][1]
            runner_up = ranked[1][0] if len(ranked) > 1 else None

            reason_parts = []
            if winner_data.get("expense_ratio_bps") is not None:
                reason_parts.append(f"ER={winner_data['expense_ratio_bps']}bps")
            if winner_data.get("aum_millions") is not None:
                reason_parts.append(f"AUM=${winner_data['aum_millions']:.0f}M")
            reason = f"Lowest cost score ({', '.join(reason_parts)})"
            if runner_up:
                runner_er = scored[runner_up].get("expense_ratio_bps", "?")
                reason += f" — beat {runner_up} ({runner_er}bps)"

            selections[slot_name] = {
                "selected": winner_ticker,
                "asset_class": asset_class,
                "candidates": scored,
                "reason": reason,
            }
            logger.info("Slot '%s': selected %s (score=%.4f) — %s",
                        slot_name, winner_ticker, winner_data["score"], reason)
        else:
            logger.warning("Slot '%s': no candidates to evaluate", slot_name)

    return selections


def save_selections(selections: Dict[str, dict]) -> str:
    """Save selections to cache file with timestamp."""
    cache = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.1",
        "selections": selections,
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    logger.info("ETF selections saved to %s", CACHE_FILE)
    return CACHE_FILE


def load_selections(max_age_days: int = 31) -> Optional[Dict[str, dict]]:
    """
    Load cached ETF selections. Returns None if cache is missing or stale.

    Args:
        max_age_days: Maximum cache age in days. Default 31 (monthly refresh).
    """
    if not os.path.exists(CACHE_FILE):
        logger.info("No ETF selection cache found — will run fresh selection")
        return None

    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)

        ts = cache.get("timestamp", "")
        if ts:
            cache_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age = datetime.now(cache_time.tzinfo) - cache_time
            if age > timedelta(days=max_age_days):
                logger.info("ETF selection cache is %d days old (max %d) — needs refresh",
                            age.days, max_age_days)
                return None

        selections = cache.get("selections", {})
        logger.info("Loaded %d ETF selections from cache (age: %s)",
                    len(selections), ts)
        return selections

    except Exception as e:
        logger.warning("Failed to load ETF selection cache: %s", e)
        return None


def get_selected_tickers(cfg: dict, max_age_days: int = 31) -> Dict[str, str]:
    """
    High-level API for the portfolio optimizer.

    Returns dict of slot_name -> selected_ticker.
    If cache is fresh, reads from cache. Otherwise, runs selection.
    """
    # Try cache first
    cached = load_selections(max_age_days)
    if cached:
        return {slot: data["selected"] for slot, data in cached.items()}

    # Run fresh selection
    selections = select_best_etfs(cfg)
    if selections:
        save_selections(selections)
    return {slot: data["selected"] for slot, data in selections.items()}


def get_selected_ticker_for_slot(
    cfg: dict,
    slot_name: str,
    fallback: str = None,
) -> str:
    """
    Get the selected ticker for a specific exposure slot.
    Falls back to the provided default if slot not found.
    """
    selections = get_selected_tickers(cfg)
    return selections.get(slot_name, fallback or "")


def get_slot_to_ticker_map(cfg: dict) -> Dict[str, str]:
    """
    Return the full slot → ticker mapping, building a reverse lookup
    so portfolio_optimizer can dynamically substitute tickers.

    Also returns the asset_class for each selected ticker from cache.
    """
    cached = load_selections()
    if not cached:
        # Run fresh selection
        selections = select_best_etfs(cfg)
        if selections:
            save_selections(selections)
            cached = selections

    result = {}
    if cached:
        for slot_name, data in cached.items():
            result[slot_name] = data["selected"]
    return result


def get_ticker_asset_class_map(cfg: dict) -> Dict[str, str]:
    """
    Build ticker → asset_class map from ETF selections cache.
    Used by portfolio_optimizer to dynamically map auto-selected tickers
    to their allocation band categories.
    """
    cached = load_selections()
    if not cached:
        selections = select_best_etfs(cfg)
        if selections:
            save_selections(selections)
            cached = selections

    result = {}
    if cached:
        for slot_name, data in cached.items():
            result[data["selected"]] = data.get("asset_class", "us_equities")
            # Also map all candidates so they're recognized
            for ticker in data.get("candidates", {}):
                if ticker not in result:
                    result[ticker] = data.get("asset_class", "us_equities")
    return result


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ETF Auto-Selector — monthly best-in-class ETF evaluation")
    parser.add_argument("--force", action="store_true",
                        help="Force refresh even if cache is fresh")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate and print results without saving to cache")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load config
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)

    # Check cache freshness
    if not args.force:
        cached = load_selections()
        if cached:
            print("\n" + "=" * 70)
            print("  ETF SELECTIONS (from cache)")
            print("=" * 70)
            for slot, data in cached.items():
                print(f"\n  {slot}:")
                print(f"    Selected: {data['selected']}")
                print(f"    Reason:   {data.get('reason', 'N/A')}")
                candidates = data.get("candidates", {})
                if candidates:
                    ranked = sorted(candidates.items(), key=lambda x: x[1]["score"])
                    for i, (t, d) in enumerate(ranked):
                        marker = " ← SELECTED" if t == data["selected"] else ""
                        er = d.get("expense_ratio_bps", "?")
                        aum = d.get("aum_millions")
                        aum_str = f"${aum:.0f}M" if aum else "N/A"
                        print(f"      {i+1}. {t:6} score={d['score']:.4f}  ER={er}bps  AUM={aum_str}{marker}")
            print("\n" + "=" * 70)
            print("  Cache is fresh — use --force to re-evaluate")
            print("=" * 70)
            sys.exit(0)

    # Run selection
    print("\nFetching live ETF data from yfinance...")
    selections = select_best_etfs(cfg, force_refresh=args.force)

    if not selections:
        print("\nNo exposure slots configured. Add etf_selector.exposure_slots to config.yaml.")
        sys.exit(1)

    # Display results
    print("\n" + "=" * 70)
    print("  ETF AUTO-SELECTION RESULTS")
    print("=" * 70)

    for slot, data in selections.items():
        print(f"\n  {slot}:")
        print(f"    Selected: {data['selected']}")
        print(f"    Reason:   {data.get('reason', 'N/A')}")
        candidates = data.get("candidates", {})
        if candidates:
            ranked = sorted(candidates.items(), key=lambda x: x[1]["score"])
            for i, (t, d) in enumerate(ranked):
                marker = " ← SELECTED" if t == data["selected"] else ""
                er = d.get("expense_ratio_bps", "?")
                aum = d.get("aum_millions")
                aum_str = f"${aum:.0f}M" if aum else "N/A"
                print(f"      {i+1}. {t:6} score={d['score']:.4f}  ER={er}bps  AUM={aum_str}{marker}")

    print("\n" + "=" * 70)

    if not args.dry_run:
        save_selections(selections)
        print(f"  Selections saved to {CACHE_FILE}")
    else:
        print("  (dry-run — not saved)")

    print("=" * 70)
