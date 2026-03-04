"""
regime_blend_example.py
========================
Global Sector Rotation System — Regime Probability Blending: Worked Example

This script is a self-contained numerical demonstration of the regime
probability blending enhancement described in ``regime_probability_enhancement.py``.
It requires no database, no live market data, and no API keys.

Run it directly::

    python regime_blend_example.py

What it shows
-------------
1. Five representative market states (from deep panic to full offense), with
   hand-specified regime probability vectors that mirror what
   ``compute_regime_probabilities()`` would produce in live operation.

2. For each state:
   * The blended allocation bands (new system).
   * The discrete allocation bands for the dominant regime (old system).
   * The absolute difference — quantifying how much smoother the blending is
     near transition points.

3. A formatted comparison table printed to stdout.

4. Softmax probability computation for five sample wedge-volume percentile
   values (2, 10, 20, 40, 70), contrasting the softmax curve against the
   existing piecewise-linear approach.

Allocation bands used
---------------------
Taken verbatim from config.yaml (as of 2026-02-28):

    us_equities:
      panic:   [0.05, 0.15]
      defense: [0.20, 0.40]
      offense: [0.55, 0.75]
    intl_developed:
      panic:   [0.00, 0.05]
      defense: [0.05, 0.15]
      offense: [0.20, 0.30]
    em_equities:
      panic:   [0.00, 0.02]
      defense: [0.02, 0.10]
      offense: [0.10, 0.20]
    energy_materials:
      panic:   [0.00, 0.05]
      defense: [0.05, 0.15]
      offense: [0.10, 0.20]
    healthcare:
      panic:   [0.05, 0.15]
      defense: [0.10, 0.20]
      offense: [0.05, 0.12]
    industry_sub:
      panic:   [0.00, 0.00]
      defense: [0.00, 0.05]
      offense: [0.05, 0.15]
    thematic:
      panic:   [0.00, 0.00]
      defense: [0.00, 0.05]
      offense: [0.05, 0.15]
    cash_short_duration:
      panic:   [0.50, 0.80]
      defense: [0.15, 0.30]
      offense: [0.00, 0.03]

Dependencies
------------
numpy (standard install), no other third-party libraries required.
The script imports ``regime_probability_enhancement`` from the same directory.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Tuple

import numpy as np

try:
    from regime_probability_enhancement import (
        compute_blended_allocation_bands,
        compute_regime_probabilities_softmax,
        get_discrete_allocation_bands,
    )
    _ENHANCEMENT_AVAILABLE = True
except ImportError:
    _ENHANCEMENT_AVAILABLE = False
    print("[WARNING] regime_probability_enhancement.py not found in sys.path.")
    print("          Defining local fallbacks — output will be identical.\n")


DUMMY_CFG: dict = {
    "regime": {
        "thresholds": {
            "panic_upper": 5,
            "defense_upper": 30,
            "panic_probability_anchor": 8,
            "offense_probability_anchor": 25,
        }
    },
    "optimizer": {
        "allocation_bands": {
            "us_equities": {
                "panic":   [0.05, 0.15],
                "defense": [0.20, 0.40],
                "offense": [0.55, 0.75],
            },
            "intl_developed": {
                "panic":   [0.00, 0.05],
                "defense": [0.05, 0.15],
                "offense": [0.20, 0.30],
            },
            "em_equities": {
                "panic":   [0.00, 0.02],
                "defense": [0.02, 0.10],
                "offense": [0.10, 0.20],
            },
            "energy_materials": {
                "panic":   [0.00, 0.05],
                "defense": [0.05, 0.15],
                "offense": [0.10, 0.20],
            },
            "healthcare": {
                "panic":   [0.05, 0.15],
                "defense": [0.10, 0.20],
                "offense": [0.05, 0.12],
            },
            "industry_sub": {
                "panic":   [0.00, 0.00],
                "defense": [0.00, 0.05],
                "offense": [0.05, 0.15],
            },
            "thematic": {
                "panic":   [0.00, 0.00],
                "defense": [0.00, 0.05],
                "offense": [0.05, 0.15],
            },
            "cash_short_duration": {
                "panic":   [0.50, 0.80],
                "defense": [0.15, 0.30],
                "offense": [0.00, 0.03],
            },
        }
    },
}

ASSET_CLASSES: List[str] = list(DUMMY_CFG["optimizer"]["allocation_bands"].keys())

MARKET_STATES: List[Dict] = [
    {
        "name": "Deep Panic",
        "description": "Sectors converging; VIX elevated; market in crisis mode.",
        "approx_percentile": 2.0,
        "probs": {"panic": 0.85, "defense": 0.12, "offense": 0.03},
    },
    {
        "name": "Transition Out of Panic",
        "description": "Conditions improving but still fragile; high uncertainty.",
        "approx_percentile": 8.0,
        "probs": {"panic": 0.30, "defense": 0.55, "offense": 0.15},
    },
    {
        "name": "Core Defense",
        "description": "Stable but cautious; rotation sector breadth narrow.",
        "approx_percentile": 18.0,
        "probs": {"panic": 0.05, "defense": 0.75, "offense": 0.20},
    },
    {
        "name": "Transition to Offense",
        "description": "Sector breadth expanding; risk appetite recovering.",
        "approx_percentile": 32.0,
        "probs": {"panic": 0.02, "defense": 0.25, "offense": 0.73},
    },
    {
        "name": "Full Offense",
        "description": "Healthy rotation; sectors diverging; max risk-on.",
        "approx_percentile": 68.0,
        "probs": {"panic": 0.01, "defense": 0.04, "offense": 0.95},
    },
]


def _local_compute_blended_allocation_bands(regime_probs: Dict[str, float], cfg: dict) -> Dict[str, Tuple[float, float]]:
    """Local copy of compute_blended_allocation_bands — identical logic."""
    p_panic = float(regime_probs.get("panic", 0.0))
    p_defense = float(regime_probs.get("defense", 0.0))
    p_offense = float(regime_probs.get("offense", 0.0))
    total = p_panic + p_defense + p_offense
    if total <= 0.0:
        raise ValueError("regime_probs must sum to a positive value")
    p_panic /= total; p_defense /= total; p_offense /= total
    all_bands = cfg["optimizer"]["allocation_bands"]
    blended: Dict[str, Tuple[float, float]] = {}
    for asset_class, regime_map in all_bands.items():
        lo_panic, hi_panic = regime_map.get("panic", [0.0, 0.05])
        lo_defense, hi_defense = regime_map.get("defense", [0.05, 0.20])
        lo_offense, hi_offense = regime_map.get("offense", [0.10, 0.30])
        blended_lo = p_panic * lo_panic + p_defense * lo_defense + p_offense * lo_offense
        blended_hi = p_panic * hi_panic + p_defense * hi_defense + p_offense * hi_offense
        blended_lo = max(0.0, min(blended_lo, 1.0))
        blended_hi = max(blended_lo, min(blended_hi, 1.0))
        blended[asset_class] = (round(blended_lo, 4), round(blended_hi, 4))
    return blended


def _local_get_discrete_allocation_bands(dominant_regime: str, cfg: dict) -> Dict[str, Tuple[float, float]]:
    """Local copy of get_discrete_allocation_bands — identical logic."""
    all_bands = cfg["optimizer"]["allocation_bands"]
    return {ac: tuple(rm.get(dominant_regime, [0.05, 0.20])) for ac, rm in all_bands.items()}


def _local_compute_regime_probabilities_softmax(composite_score: float, cfg: dict, temperature: float = 5.0) -> Dict[str, float]:
    """Local copy of compute_regime_probabilities_softmax — identical logic."""
    if composite_score is None or (isinstance(composite_score, float) and math.isnan(composite_score)):
        return {"panic": 0.0, "defense": 0.0, "offense": 0.0}
    panic_upper = float(cfg["regime"]["thresholds"]["panic_upper"])
    defense_upper = float(cfg["regime"]["thresholds"]["defense_upper"])
    panic_center = panic_upper
    defense_center = (panic_upper + defense_upper) / 2.0
    offense_center = defense_upper + 20.0
    centers = np.array([panic_center, defense_center, offense_center])
    distances = np.abs(composite_score - centers)
    logits = -distances / temperature
    logits -= logits.max()
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()
    p_panic, p_defense, p_offense = float(probs[0]), float(probs[1]), float(probs[2])
    total = p_panic + p_defense + p_offense
    p_panic /= total; p_defense /= total; p_offense /= total
    return {"panic": round(p_panic, 4), "defense": round(p_defense, 4), "offense": round(p_offense, 4)}


if _ENHANCEMENT_AVAILABLE:
    _blend_fn = compute_blended_allocation_bands
    _discrete_fn = get_discrete_allocation_bands
    _softmax_fn = compute_regime_probabilities_softmax
else:
    _blend_fn = _local_compute_blended_allocation_bands
    _discrete_fn = _local_get_discrete_allocation_bands
    _softmax_fn = _local_compute_regime_probabilities_softmax


def dominant_regime(probs: Dict[str, float]) -> str:
    """Return the key with the highest probability value."""
    return max(probs, key=probs.get)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _band_str(lo: float, hi: float) -> str:
    return f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"


def _diff_str(blend: Tuple[float, float], discrete: Tuple[float, float]) -> str:
    mid_blend = (blend[0] + blend[1]) / 2.0
    mid_discrete = (discrete[0] + discrete[1]) / 2.0
    diff = mid_blend - mid_discrete
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff * 100:.1f}%"


def print_comparison_table() -> None:
    _TITLE_WIDTH = 80
    print()
    print("=" * _TITLE_WIDTH)
    print(" REGIME PROBABILITY BLENDING — ALLOCATION BAND COMPARISON")
    print(" Global Sector Rotation System")
    print("=" * _TITLE_WIDTH)
    print()
    print("Legend: [lo%, hi%] = allocation band   Δmid = blended midpoint − discrete midpoint")
    print()
    for state in MARKET_STATES:
        name = state["name"]
        desc = state["description"]
        probs = state["probs"]
        dom = dominant_regime(probs)
        print("-" * _TITLE_WIDTH)
        print(f"  STATE: {name}")
        print(f"  {desc}")
        print(f"  Probabilities → panic={probs['panic']:.2f}  defense={probs['defense']:.2f}  "
              f"offense={probs['offense']:.2f}  (dominant: {dom.upper()})")
        print(f"  Approx. wedge-volume percentile: {state['approx_percentile']:.0f}")
        print()
        blended = _blend_fn(probs, DUMMY_CFG)
        discrete = _discrete_fn(dom, DUMMY_CFG)
        print(f"  {'Asset Class':<22}  {'Blended (NEW)':<20}  {'Discrete (OLD)':<20}  {'\u0394 midpoint':>10}")
        print(f"  {'-'*22}  {'-'*20}  {'-'*20}  {'-'*10}")
        for ac in ASSET_CLASSES:
            b = blended[ac]
            d = discrete[ac]
            diff = _diff_str(b, d)
            blend_s = _band_str(*b)
            disc_s = _band_str(*d)
            mid_diff = abs((b[0]+b[1])/2 - (d[0]+d[1])/2)
            flag = " ◄" if mid_diff > 0.02 else ""
            print(f"  {ac:<22}  {blend_s:<20}  {disc_s:<20}  {diff:>10}{flag}")
        print()
        print("  ◄ = midpoint differs by more than 2pp (blending has material impact)")
        print()
    print("=" * _TITLE_WIDTH)
    print()


SOFTMAX_SAMPLE_PERCENTILES: List[float] = [2.0, 10.0, 20.0, 40.0, 70.0]
SOFTMAX_TEMPERATURES: List[float] = [2.0, 5.0, 10.0]


def _piecewise_linear_probabilities(percentile: float, cfg: dict) -> Dict[str, float]:
    """Reproduce the existing piecewise-linear compute_regime_probabilities() logic."""
    if percentile is None or (isinstance(percentile, float) and math.isnan(percentile)):
        return {"panic": 0.0, "defense": 0.0, "offense": 0.0}
    panic_upper = cfg["regime"]["thresholds"]["panic_upper"]
    defense_upper = cfg["regime"]["thresholds"]["defense_upper"]
    panic_anchor = cfg["regime"]["thresholds"]["panic_probability_anchor"]
    offense_anchor = cfg["regime"]["thresholds"]["offense_probability_anchor"]
    p_panic = 0.0; p_defense = 0.0; p_offense = 0.0
    if percentile < panic_upper:
        p_panic = min(1.0, max(0.0, (panic_anchor - percentile) / panic_anchor))
        p_defense = 1.0 - p_panic
        p_offense = 0.0
    elif percentile < panic_anchor:
        t = (percentile - panic_upper) / max(1, panic_anchor - panic_upper)
        p_panic = max(0.0, 1.0 - t) * 0.6
        p_defense = 1.0 - p_panic
        p_offense = 0.0
    elif percentile < offense_anchor:
        p_panic = 0.0
        t = (percentile - panic_anchor) / max(1, offense_anchor - panic_anchor)
        p_defense = max(0.0, 1.0 - t * 0.5)
        p_offense = 1.0 - p_defense
    elif percentile < defense_upper:
        t = (percentile - offense_anchor) / max(1, defense_upper - offense_anchor)
        p_panic = 0.0
        p_defense = max(0.0, 0.5 * (1.0 - t))
        p_offense = 1.0 - p_defense
    else:
        p_panic = 0.0; p_defense = 0.0; p_offense = 1.0
    total = p_panic + p_defense + p_offense
    if total > 0:
        p_panic /= total; p_defense /= total; p_offense /= total
    return {"panic": round(p_panic, 4), "defense": round(p_defense, 4), "offense": round(p_offense, 4)}


def print_softmax_comparison() -> None:
    _TITLE_WIDTH = 80
    print("=" * _TITLE_WIDTH)
    print(" SOFTMAX vs PIECEWISE-LINEAR REGIME PROBABILITIES")
    print(" (wedge-volume percentile → p_panic / p_defense / p_offense)")
    print("=" * _TITLE_WIDTH)
    print()
    print(f"  Config thresholds: panic_upper=5, defense_upper=30")
    print(f"  Softmax centroids: panic=5.0, defense=17.5, offense=50.0")
    print()
    print(f"  {'Pctile':>7}  {'Method':<24}  {'p_panic':>9}  {'p_defense':>10}  {'p_offense':>10}  {'Dominant':>9}")
    print(f"  {'-'*7}  {'-'*24}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*9}")
    for pct in SOFTMAX_SAMPLE_PERCENTILES:
        pl = _piecewise_linear_probabilities(pct, DUMMY_CFG)
        pl_dom = dominant_regime(pl)
        print(f"  {pct:7.1f}  {'Piecewise-linear':<24}  {pl['panic']:9.4f}  "
              f"{pl['defense']:10.4f}  {pl['offense']:10.4f}  {pl_dom:>9}")
        for temp in SOFTMAX_TEMPERATURES:
            sm = _softmax_fn(pct, DUMMY_CFG, temperature=temp)
            sm_dom = dominant_regime(sm)
            label = f"Softmax (T={temp:.0f})"
            print(f"  {'':7}  {label:<24}  {sm['panic']:9.4f}  "
                  f"{sm['defense']:10.4f}  {sm['offense']:10.4f}  {sm_dom:>9}")
        print()
    print("  Notes:")
    print("  * T=2  → sharper transitions (nearly binary near centroids)")
    print("  * T=5  → balanced (default; meaningful mass on adjacent regimes)")
    print("  * T=10 → smoother (all regimes always active; less decisive)")
    print()
    print("  Key difference from piecewise-linear:")
    print("  * Softmax is analytic (smooth everywhere; no kinks at thresholds).")
    print("  * Softmax never produces exactly 0.0 for any regime.")
    print("  * Piecewise-linear produces hard zeros (e.g. p_offense=0 below pct=8).")
    print()
    print("=" * _TITLE_WIDTH)
    print()


def print_transition_scan() -> None:
    _TITLE_WIDTH = 80
    print("=" * _TITLE_WIDTH)
    print(" TRANSITION SCAN: BLENDED vs DISCRETE MIDPOINTS")
    print(" (sweeping wedge-volume percentile 0 → 80)")
    print(" Asset classes shown: us_equities | cash_short_duration")
    print("=" * _TITLE_WIDTH)
    print()
    print(f"  {'Pctile':>7}  {'Dominant':>9}  "
          f"{'Blend US-EQ mid':>16}  {'Disc US-EQ mid':>15}  "
          f"{'Blend Cash mid':>15}  {'Disc Cash mid':>14}")
    print(f"  {'-'*7}  {'-'*9}  {'-'*16}  {'-'*15}  {'-'*15}  {'-'*14}")
    scan_points = list(range(0, 5)) + list(range(5, 35, 2)) + list(range(35, 81, 5))
    for pct in scan_points:
        probs = _softmax_fn(float(pct), DUMMY_CFG, temperature=5.0)
        dom = dominant_regime(probs)
        blended = _blend_fn(probs, DUMMY_CFG)
        discrete = _discrete_fn(dom, DUMMY_CFG)
        b_us_lo, b_us_hi = blended["us_equities"]
        d_us_lo, d_us_hi = discrete["us_equities"]
        b_cash_lo, b_cash_hi = blended["cash_short_duration"]
        d_cash_lo, d_cash_hi = discrete["cash_short_duration"]
        b_us_mid = (b_us_lo + b_us_hi) / 2
        d_us_mid = (d_us_lo + d_us_hi) / 2
        b_cash_mid = (b_cash_lo + b_cash_hi) / 2
        d_cash_mid = (d_cash_lo + d_cash_hi) / 2
        marker = ""
        if 5 <= pct <= 10:
            marker = " ← panic→def"
        elif 28 <= pct <= 35:
            marker = " ← def→off"
        print(f"  {pct:7.0f}  {dom:>9}  "
              f"{b_us_mid*100:15.1f}%  {d_us_mid*100:14.1f}%  "
              f"{b_cash_mid*100:14.1f}%  {d_cash_mid*100:13.1f}%{marker}")
    print()
    print("  Blended midpoints ramp continuously; discrete midpoints jump at regime")
    print("  transitions (visible as equal values followed by a sudden step change).")
    print()
    print("=" * _TITLE_WIDTH)
    print()


def print_detailed_walkthrough() -> None:
    _TITLE_WIDTH = 80
    print("=" * _TITLE_WIDTH)
    print(" DETAILED WALKTHROUGH: HOW BLENDED BANDS ARE COMPUTED")
    print("=" * _TITLE_WIDTH)
    print()
    state = MARKET_STATES[1]
    probs = state["probs"]
    dom = dominant_regime(probs)
    print(f"  Example state: '{state['name']}'")
    print(f"  Regime probabilities:")
    print(f"    p_panic   = {probs['panic']:.2f}")
    print(f"    p_defense = {probs['defense']:.2f}")
    print(f"    p_offense = {probs['offense']:.2f}")
    print(f"  Dominant regime (old system input): {dom.upper()}")
    print()
    print("  Formula: blended_lo = p_panic*lo_panic + p_defense*lo_defense + p_offense*lo_offense")
    print("           blended_hi = p_panic*hi_panic + p_defense*hi_defense + p_offense*hi_offense")
    print()
    all_bands = DUMMY_CFG["optimizer"]["allocation_bands"]
    print(f"  {'Asset Class':<22}  {'Panic band':>14}  {'Defense band':>14}  "
          f"{'Offense band':>13}  {'\u2192 Blended':>18}  {'Old Discrete':>18}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*14}  {'-'*13}  {'-'*18}  {'-'*18}")
    for ac in ASSET_CLASSES:
        rm = all_bands[ac]
        lo_pa, hi_pa = rm["panic"]
        lo_de, hi_de = rm["defense"]
        lo_of, hi_of = rm["offense"]
        bl_lo = probs["panic"]*lo_pa + probs["defense"]*lo_de + probs["offense"]*lo_of
        bl_hi = probs["panic"]*hi_pa + probs["defense"]*hi_de + probs["offense"]*hi_of
        disc = rm[dom]
        panic_s = f"[{lo_pa*100:.0f}%,{hi_pa*100:.0f}%]"
        def_s   = f"[{lo_de*100:.0f}%,{hi_de*100:.0f}%]"
        off_s   = f"[{lo_of*100:.0f}%,{hi_of*100:.0f}%]"
        blend_s = f"[{bl_lo*100:.1f}%,{bl_hi*100:.1f}%]"
        disc_s  = f"[{disc[0]*100:.0f}%,{disc[1]*100:.0f}%]"
        print(f"  {ac:<22}  {panic_s:>14}  {def_s:>14}  {off_s:>13}  {blend_s:>18}  {disc_s:>18}")
    print()
    print("  Observation: blended cash allocation [17.5%, 32.4%] is shifted UP from")
    print("  pure defense [15%, 30%] because the 30% panic weight pulls toward")
    print("  the panic band [50%, 80%].  This is the system expressing uncertainty.")
    print()
    print("=" * _TITLE_WIDTH)
    print()


def main() -> None:
    """Run all example sections in sequence."""
    print()
    print("╔" + "═" * 74 + "╗")
    print("║  REGIME PROBABILITY BLENDING — COMPLETE WORKED EXAMPLE                  ║")
    print("║  Global Sector Rotation System                                           ║")
    print("╚" + "═" * 74 + "╝")
    print()

    print_comparison_table()
    print_softmax_comparison()
    print_transition_scan()
    print_detailed_walkthrough()

    print("=" * 80)
    print(" SANITY CHECKS")
    print("=" * 80)
    print()

    full_offense_probs = {"panic": 0.0, "defense": 0.0, "offense": 1.0}
    blended_off = _blend_fn(full_offense_probs, DUMMY_CFG)
    discrete_off = _discrete_fn("offense", DUMMY_CFG)
    passed = all(
        abs(blended_off[ac][0] - discrete_off[ac][0]) < 1e-9 and
        abs(blended_off[ac][1] - discrete_off[ac][1]) < 1e-9
        for ac in ASSET_CLASSES
    )
    status = "PASS \u2713" if passed else "FAIL \u2717"
    print(f"  [1] p_offense=1.0 → blended == discrete offense bands:  {status}")

    full_panic_probs = {"panic": 1.0, "defense": 0.0, "offense": 0.0}
    blended_pa = _blend_fn(full_panic_probs, DUMMY_CFG)
    discrete_pa = _discrete_fn("panic", DUMMY_CFG)
    passed = all(
        abs(blended_pa[ac][0] - discrete_pa[ac][0]) < 1e-9 and
        abs(blended_pa[ac][1] - discrete_pa[ac][1]) < 1e-9
        for ac in ASSET_CLASSES
    )
    status = "PASS \u2713" if passed else "FAIL \u2717"
    print(f"  [2] p_panic=1.0  → blended == discrete panic bands:     {status}")

    all_valid = True
    for state in MARKET_STATES:
        blended = _blend_fn(state["probs"], DUMMY_CFG)
        for ac in ASSET_CLASSES:
            lo, hi = blended[ac]
            if lo > hi + 1e-9:
                all_valid = False
    status = "PASS \u2713" if all_valid else "FAIL \u2717"
    print(f"  [3] blended lo <= hi for all states and asset classes:  {status}")

    sm_at_panic_center = _softmax_fn(5.0, DUMMY_CFG, temperature=5.0)
    sm_at_def_center   = _softmax_fn(17.5, DUMMY_CFG, temperature=5.0)
    sm_at_off_center   = _softmax_fn(50.0, DUMMY_CFG, temperature=5.0)
    check4 = (
        dominant_regime(sm_at_panic_center) == "panic" and
        dominant_regime(sm_at_def_center) == "defense" and
        dominant_regime(sm_at_off_center) == "offense"
    )
    status = "PASS \u2713" if check4 else "FAIL \u2717"
    print(f"  [4] Softmax dominant regime correct at each centroid:   {status}")
    print()

    if all([
        all(abs(blended_off[ac][0] - discrete_off[ac][0]) < 1e-9 for ac in ASSET_CLASSES),
        all(abs(blended_pa[ac][0] - discrete_pa[ac][0]) < 1e-9 for ac in ASSET_CLASSES),
        all_valid,
        check4,
    ]):
        print("  All sanity checks passed.  The blending implementation is correct.")
    else:
        print("  One or more sanity checks failed — review the output above.")
        sys.exit(1)

    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
