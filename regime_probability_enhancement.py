"""
regime_probability_enhancement.py
==================================
Global Sector Rotation System — Regime Probability Blending Enhancement

This module contains two refactored functions intended for integration into
the production codebase:

  A) ``compute_regime_probabilities_softmax``  → regime_detector.py
     An alternative to the existing piecewise-linear
     ``compute_regime_probabilities()``.  Uses softmax over squared distances
     from regime centroids so the probability surface is everywhere smooth and
     differentiable, with a single ``temperature`` hyperparameter controlling
     transition sharpness.

  B) ``compute_blended_allocation_bands``  → portfolio_optimizer.py
     Replaces the discrete allocation-band lookup
     ``cfg["optimizer"]["allocation_bands"][asset_class][dominant_regime]``
     with a probability-weighted blend across *all three* regimes.  The result
     is a continuously varying (lo, hi) corridor that widens smoothly as the
     system transitions between regimes rather than jumping discontinuously.

Integration notes
-----------------
* **regime_detector.py**: drop ``compute_regime_probabilities_softmax`` next to
  the existing ``compute_regime_probabilities()``.  The caller signature is
  identical; swap in ``compute_regime_probabilities_softmax`` wherever
  smooth/differentiable probabilities are preferred.

* **portfolio_optimizer.py**: import ``compute_blended_allocation_bands`` and
  call it inside ``_compute_allocation_bounds()`` after the probability vector
  is available (e.g. from ``build_regime_state_json``).  Replace the line::

      regime_band = band.get(regime, band.get("offense", [0.05, 0.20]))

  with the blended bands produced here.  Pass the blended (lo, hi) values
  directly into the CVaR solver constraints.

Dependencies
------------
numpy (standard installation), no other third-party libraries required.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


_DEFAULT_PANIC_CENTER: float = 5.0
_DEFAULT_DEFENSE_CENTER: float = 17.5
_DEFAULT_OFFENSE_CENTER: float = 50.0


def compute_regime_probabilities_softmax(
    composite_score: float,
    cfg: dict,
    temperature: float = 5.0,
) -> Dict[str, float]:
    """
    Compute regime probabilities using softmax over squared distances from
    regime centroids.

    This is an alternative to the existing piecewise-linear
    ``compute_regime_probabilities()`` in ``regime_detector.py``.

    Algorithm
    ---------
    1. Compute absolute distance from ``composite_score`` to each regime centroid.
    2. Scale by temperature: ``logit_i = -d_i / temperature``.
    3. Apply softmax: ``p_i = exp(logit_i) / sum(exp(logit_j))``.

    Centroid derivation (from config.yaml)
    ---------------------------------------
    * ``panic_center   = panic_upper``                       → 5.0
    * ``defense_center = (panic_upper + defense_upper) / 2`` → 17.5
    * ``offense_center = defense_upper + 20``                → 50.0

    Parameters
    ----------
    composite_score : float
        Current wedge volume percentile, in [0, 100].
    cfg : dict
        Master configuration loaded from ``config.yaml``.
    temperature : float, optional
        Controls smoothness of regime transitions. Default 5.0.

    Returns
    -------
    dict
        Keys ``"panic"``, ``"defense"``, ``"offense"`` summing to 1.0.
    """
    if composite_score is None or (isinstance(composite_score, float) and math.isnan(composite_score)):
        return {"panic": 0.0, "defense": 0.0, "offense": 0.0}

    panic_upper: float = float(cfg["regime"]["thresholds"]["panic_upper"])
    defense_upper: float = float(cfg["regime"]["thresholds"]["defense_upper"])

    panic_center: float = panic_upper
    defense_center: float = (panic_upper + defense_upper) / 2.0
    offense_center: float = defense_upper + 20.0

    centers = np.array([panic_center, defense_center, offense_center], dtype=float)
    distances = np.abs(composite_score - centers)
    logits = -distances / temperature
    logits_shifted = logits - logits.max()
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum()

    p_panic, p_defense, p_offense = float(probs[0]), float(probs[1]), float(probs[2])

    total = p_panic + p_defense + p_offense
    if total > 0.0:
        p_panic /= total
        p_defense /= total
        p_offense /= total

    return {
        "panic": round(p_panic, 4),
        "defense": round(p_defense, 4),
        "offense": round(p_offense, 4),
    }


def compute_blended_allocation_bands(
    regime_probs: Dict[str, float],
    cfg: dict,
) -> Dict[str, Tuple[float, float]]:
    """
    Blend per-asset-class allocation bands as a probability-weighted average
    across all three regimes.

    This replaces the discrete allocation-band lookup in
    ``portfolio_optimizer._compute_allocation_bounds()``.

    Motivation
    ----------
    When the market is in a transition state, the old system snaps entirely to
    the dominant regime's allocation bands and ignores meaningful probability
    mass in other regimes. Blending makes the allocation bands a smooth function
    of the full probability vector.

    Algorithm
    ---------
    For each asset class ``ac`` and bound type (lo / hi)::

        blended_lo[ac] = p_panic  * lo[ac]["panic"]
                       + p_defense * lo[ac]["defense"]
                       + p_offense * lo[ac]["offense"]

    Parameters
    ----------
    regime_probs : dict
        Probability vector with keys ``"panic"``, ``"defense"``, ``"offense"``.
    cfg : dict
        Master configuration loaded from ``config.yaml``.

    Returns
    -------
    dict
        Keys are asset class names; values are tuples ``(blended_lo, blended_hi)``.
    """
    p_panic: float = float(regime_probs.get("panic", 0.0))
    p_defense: float = float(regime_probs.get("defense", 0.0))
    p_offense: float = float(regime_probs.get("offense", 0.0))

    total = p_panic + p_defense + p_offense
    if total <= 0.0:
        raise ValueError(f"regime_probs must sum to a positive value; got {regime_probs!r}")
    p_panic /= total
    p_defense /= total
    p_offense /= total

    all_bands: dict = cfg["optimizer"]["allocation_bands"]
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


def get_discrete_allocation_bands(
    dominant_regime: str,
    cfg: dict,
) -> Dict[str, Tuple[float, float]]:
    """
    Return the discrete per-asset-class allocation bands for a single regime.

    This is the existing behavior of the portfolio optimizer.
    Included here for side-by-side comparison with
    ``compute_blended_allocation_bands()``.

    Parameters
    ----------
    dominant_regime : str
        One of ``"panic"``, ``"defense"``, or ``"offense"``.
    cfg : dict
        Master configuration loaded from ``config.yaml``.

    Returns
    -------
    dict
        Keys are asset class names; values are tuples ``(lo, hi)``.
    """
    all_bands: dict = cfg["optimizer"]["allocation_bands"]
    discrete: Dict[str, Tuple[float, float]] = {}
    for asset_class, regime_map in all_bands.items():
        band = regime_map.get(dominant_regime, [0.05, 0.20])
        discrete[asset_class] = (float(band[0]), float(band[1]))
    return discrete


_INTEGRATION_NOTES = """
Integration Recipe
==================

Step 1 — regime_detector.py
-----------------------------
Add this import at the top of regime_detector.py:

    from regime_probability_enhancement import compute_regime_probabilities_softmax

To switch the live system to softmax probabilities, in ``compute_daily_regime``,
change the call from:

    probs = compute_regime_probabilities(pct_val, cfg)

to:

    probs = compute_regime_probabilities_softmax(pct_val, cfg, temperature=5.0)

Step 2 — portfolio_optimizer.py
---------------------------------
Add this import at the top of portfolio_optimizer.py:

    from regime_probability_enhancement import compute_blended_allocation_bands

Modify ``_compute_allocation_bounds()`` to accept an optional
``regime_probs`` argument and replace the discrete lookup with:

    if regime_probs is not None:
        blended = compute_blended_allocation_bands(regime_probs, cfg)
        group_lo, group_hi = blended.get(ac, (0.0, 0.20))
    else:
        # Backward-compatible discrete fallback
        band = bands.get(ac, {...})
        regime_band = band.get(regime, band.get("offense", [0.05, 0.20]))
        group_lo = regime_band[0]
        group_hi = regime_band[1]

Step 3 — Propagate regime_probs through the call chain
-------------------------------------------------------
In ``run_optimizer()``, pass the probability vector from the regime state:

    regime_state = get_latest_regime_state(conn, cfg)
    regime_probs = regime_state.get("regime_probabilities")  # dict or None

    bounds = _compute_allocation_bounds(
        tickers, regime, cfg, factor_scores, regime_probs=regime_probs
    )
"""
