"""
transaction_costs.py
====================
Transaction cost modelling for a quantitative sector rotation system.

This module estimates the full round-trip cost of entering and exiting a
position (bid-ask spread + square-root market impact) and applies those
costs to the composite alpha scores produced by the stock screener.
Positions whose net alpha (alpha minus annualised transaction costs) is
non-positive are flagged as *not cost-viable* and can be excluded from
the investable universe.

Model
-----
One-way transaction cost:

    cost = spread/2 + k * sqrt(ADV_fraction)

where:
    - ``spread``      : round-trip bid-ask spread in decimal (e.g. 0.001 = 10 bps)
    - ``k``           : market-impact coefficient (default 0.1)
    - ``ADV_fraction``: order size / average daily volume (e.g. 0.01 = 1 % of ADV)

Round-trip cost = 2 × one-way cost.

Annualised cost = round_trip_cost × expected_annual_turnover.

Net alpha = composite_score - annualised_cost.

Usage
-----
>>> from sector_rotation.transaction_costs import TransactionCostModel
>>> model = TransactionCostModel()
>>> cost = model.total_cost("AAPL", adv_fraction=0.01, market_cap=3e12)
>>> df_with_costs = model.apply_to_screener(screener_df)
>>> viable = model.filter_viable(df_with_costs)
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_LARGE_CAP_THRESHOLD: float = 50e9
_MID_CAP_THRESHOLD: float = 2e9

_LARGE_CAP_HALF_SPREAD_RANGE = (0.5e-4, 1.5e-4)
_MID_CAP_HALF_SPREAD_RANGE   = (1.5e-4, 5.0e-4)
_SMALL_CAP_HALF_SPREAD_RANGE = (5.0e-4, 15.0e-4)

_SPREAD_CALIBRATION_C: float = 0.03


def _bps_to_decimal(bps: float) -> float:
    return bps * 1e-4


def _decimal_to_bps(decimal: float) -> float:
    return decimal * 1e4


class TransactionCostModel:
    """
    Estimate and apply transaction costs to a stock screener output.

    Parameters
    ----------
    impact_coefficient : float, optional
        Market-impact scaling factor ``k``.  Default is ``0.1``.
    default_spread_bps : float, optional
        Fallback half-spread in basis points.  Default is ``10.0`` bps.
    """

    def __init__(self, impact_coefficient: float = 0.1, default_spread_bps: float = 10.0) -> None:
        if impact_coefficient < 0:
            raise ValueError("impact_coefficient must be non-negative.")
        if default_spread_bps < 0:
            raise ValueError("default_spread_bps must be non-negative.")
        self.impact_coefficient: float = impact_coefficient
        self.default_spread_bps: float = default_spread_bps
        logger.debug("TransactionCostModel initialised: k=%.4f, default_spread=%.2f bps", self.impact_coefficient, self.default_spread_bps)

    def estimate_spread(self, ticker: str, price: Optional[float] = None, avg_volume: Optional[float] = None, market_cap: Optional[float] = None) -> float:
        """
        Estimate the one-way half-spread for a stock.

        Priority: price+volume model > market-cap tier midpoint > default fallback.
        """
        tier_range: Optional[tuple] = None
        if market_cap is not None:
            if market_cap > _LARGE_CAP_THRESHOLD:
                tier_range = _LARGE_CAP_HALF_SPREAD_RANGE
            elif market_cap > _MID_CAP_THRESHOLD:
                tier_range = _MID_CAP_HALF_SPREAD_RANGE
            else:
                tier_range = _SMALL_CAP_HALF_SPREAD_RANGE

        if price is not None and avg_volume is not None and price > 0 and avg_volume > 0:
            adv_dollar = price * avg_volume
            half_spread = _SPREAD_CALIBRATION_C / math.sqrt(adv_dollar)
            if tier_range is not None:
                lo, hi = tier_range
                half_spread = max(lo, min(hi, half_spread))
            logger.debug("%s: price/volume spread estimate = %.2f bps", ticker, _decimal_to_bps(half_spread))
            return half_spread

        if tier_range is not None:
            lo, hi = tier_range
            half_spread = (lo + hi) / 2.0
            logger.debug("%s: market-cap tier spread estimate = %.2f bps", ticker, _decimal_to_bps(half_spread))
            return half_spread

        half_spread = _bps_to_decimal(self.default_spread_bps)
        logger.debug("%s: using default spread = %.2f bps", ticker, self.default_spread_bps)
        return half_spread

    def estimate_market_impact(self, adv_fraction: float) -> float:
        """
        Estimate one-way market impact using the square-root model.

            market_impact = k * sqrt(ADV_fraction)
        """
        if adv_fraction < 0:
            raise ValueError(f"adv_fraction must be >= 0; got {adv_fraction}")
        if adv_fraction > 1:
            warnings.warn(f"adv_fraction={adv_fraction:.4f} exceeds 1.0 (100 % of ADV). This implies a very large order; impact estimates may be unreliable.", UserWarning, stacklevel=2)
        impact = self.impact_coefficient * math.sqrt(adv_fraction)
        logger.debug("Market impact estimate: k=%.4f, ADV_frac=%.4f → %.2f bps", self.impact_coefficient, adv_fraction, _decimal_to_bps(impact))
        return impact

    def total_cost(self, ticker: str, adv_fraction: float = 0.01, price: Optional[float] = None, avg_volume: Optional[float] = None, market_cap: Optional[float] = None) -> float:
        """
        Compute the total one-way transaction cost in decimal.

            cost = spread/2 + k * sqrt(ADV_fraction)
        """
        half_spread = self.estimate_spread(ticker, price=price, avg_volume=avg_volume, market_cap=market_cap)
        impact = self.estimate_market_impact(adv_fraction)
        cost = half_spread + impact
        logger.debug("%s | one-way cost: half_spread=%.2f bps + impact=%.2f bps = %.2f bps", ticker, _decimal_to_bps(half_spread), _decimal_to_bps(impact), _decimal_to_bps(cost))
        return cost

    def round_trip_cost(self, ticker: str, adv_fraction: float = 0.01, price: Optional[float] = None, avg_volume: Optional[float] = None, market_cap: Optional[float] = None) -> float:
        """
        Compute the total round-trip transaction cost in decimal.

        Round-trip = 2 × one-way cost.
        """
        one_way = self.total_cost(ticker, adv_fraction=adv_fraction, price=price, avg_volume=avg_volume, market_cap=market_cap)
        rt = 2.0 * one_way
        logger.debug("%s | round-trip cost = %.2f bps", ticker, _decimal_to_bps(rt))
        return rt

    def apply_to_screener(self, screener_df: pd.DataFrame, alpha_col: str = "composite_score", annual_turnover: float = 4.0) -> pd.DataFrame:
        """
        Apply transaction costs to a stock screener output DataFrame.

        Adds columns: ``estimated_cost_bps``, ``net_alpha``, ``cost_viable``.
        Returns DataFrame sorted by ``net_alpha`` descending.
        """
        if alpha_col not in screener_df.columns:
            raise KeyError(f"Alpha column '{alpha_col}' not found in DataFrame. Available columns: {list(screener_df.columns)}")
        if annual_turnover < 0:
            raise ValueError(f"annual_turnover must be >= 0; got {annual_turnover}")

        df = screener_df.copy()
        has_ticker       = "ticker"       in df.columns
        has_price        = "price"        in df.columns
        has_avg_volume   = "avg_volume"   in df.columns
        has_market_cap   = "market_cap"   in df.columns
        has_adv_fraction = "adv_fraction" in df.columns

        logger.info("Applying transaction costs to %d positions | turnover=%.1fx/yr", len(df), annual_turnover)
        round_trip_costs: list = []

        for idx, row in df.iterrows():
            ticker     = str(row["ticker"])       if has_ticker       else f"stock_{idx}"
            price      = float(row["price"])      if has_price        else None
            avg_volume = float(row["avg_volume"]) if has_avg_volume   else None
            market_cap = float(row["market_cap"]) if has_market_cap   else None
            adv_frac   = float(row["adv_fraction"]) if has_adv_fraction else 0.01
            rt = self.round_trip_cost(ticker, adv_fraction=adv_frac, price=price, avg_volume=avg_volume, market_cap=market_cap)
            round_trip_costs.append(rt)

        df["estimated_cost_bps"] = [_decimal_to_bps(c) for c in round_trip_costs]
        annualised_costs = [c * annual_turnover for c in round_trip_costs]
        df["net_alpha"]   = df[alpha_col] - annualised_costs
        df["cost_viable"] = df["net_alpha"] >= 0.0

        n_viable = df["cost_viable"].sum()
        n_total  = len(df)
        logger.info("Cost filter: %d/%d positions are cost-viable (%.1f%%)", n_viable, n_total, 100.0 * n_viable / n_total if n_total > 0 else 0.0)
        return df.sort_values("net_alpha", ascending=False).reset_index(drop=True)

    def filter_viable(self, screener_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only the cost-viable rows from an augmented screener DataFrame.

        Requires ``apply_to_screener`` to have been called first.
        """
        if "cost_viable" not in screener_df.columns:
            raise KeyError("'cost_viable' column not found. Run apply_to_screener() before calling filter_viable().")
        viable = screener_df[screener_df["cost_viable"]].copy()
        logger.info("filter_viable: returning %d of %d positions", len(viable), len(screener_df))
        return viable


def _run_example() -> None:
    import textwrap
    sample_data = {
        "ticker": ["AAPL", "MSFT", "NVDA", "JPM", "MID1", "MID2", "MID3", "SML1", "SML2", "SML3"],
        "composite_score": [0.15, 0.10, 0.25, 0.02, 0.12, 0.01, 0.04, 0.20, 0.005, 0.35],
        "market_cap": [3.0e12, 2.8e12, 1.5e12, 550e9, 25e9, 8e9, 5e9, 1.5e9, 800e6, 1.2e9],
        "price": [185.0, 375.0, 650.0, 195.0, 55.0, 40.0, 28.0, 18.0, 9.0, 14.0],
        "avg_volume": [60_000_000, 25_000_000, 45_000_000, 15_000_000, 3_500_000, 2_000_000, 1_500_000, 800_000, 400_000, 600_000],
        "adv_fraction": [0.005, 0.005, 0.005, 0.008, 0.015, 0.020, 0.020, 0.050, 0.050, 0.040],
    }
    screener_df = pd.DataFrame(sample_data)
    model = TransactionCostModel(impact_coefficient=0.1, default_spread_bps=10.0)
    print("=" * 72)
    print(" TRANSACTION COST MODEL — WORKED EXAMPLE")
    print("=" * 72)
    print(f"\nParameters: k={model.impact_coefficient}, default_spread={model.default_spread_bps} bps, annual_turnover=4.0×\n")
    augmented = model.apply_to_screener(screener_df, alpha_col="composite_score", annual_turnover=4.0)
    header = f"{'Ticker':<7} {'MarketCap':>12} {'RawAlpha':>10} {'CostBps':>9} {'AnnCost%':>9} {'NetAlpha':>10} {'Viable':>7}"
    sep = "-" * 72
    print(header)
    print(sep)
    for _, row in augmented.iterrows():
        mcap_str = _fmt_mcap(row["market_cap"])
        raw_alpha_pct = row["composite_score"] * 100
        ann_cost_pct  = row["estimated_cost_bps"] * 4 / 100
        net_alpha_pct = row["net_alpha"] * 100
        viable_str    = "YES" if row["cost_viable"] else "NO ✗"
        print(f"{row['ticker']:<7} {mcap_str:>12} {raw_alpha_pct:>9.2f}% {row['estimated_cost_bps']:>8.1f} {ann_cost_pct:>8.2f}% {net_alpha_pct:>9.2f}% {viable_str:>7}")
    print(sep)
    viable = model.filter_viable(augmented)
    filtered_out = augmented[~augmented["cost_viable"]]
    print(f"\nViable positions ({len(viable)} of {len(augmented)}): " + ", ".join(viable["ticker"].tolist()))
    print(f"Filtered out ({len(filtered_out)}): " + ", ".join(filtered_out["ticker"].tolist()))
    print("\n--- Spot-check: SML2 (small-cap, low alpha) ---")
    hs = model.estimate_spread("SML2", price=9.0, avg_volume=400_000, market_cap=800e6)
    mi = model.estimate_market_impact(adv_fraction=0.05)
    ow = model.total_cost("SML2", adv_fraction=0.05, price=9.0, avg_volume=400_000, market_cap=800e6)
    rt = model.round_trip_cost("SML2", adv_fraction=0.05, price=9.0, avg_volume=400_000, market_cap=800e6)
    ann = rt * 4.0
    print(textwrap.dedent(f"""
    Half-spread           : {_decimal_to_bps(hs):.2f} bps
    Market impact (1-way) : {_decimal_to_bps(mi):.2f} bps
    One-way cost          : {_decimal_to_bps(ow):.2f} bps
    Round-trip cost       : {_decimal_to_bps(rt):.2f} bps
    Annualised cost (4×)  : {_decimal_to_bps(ann):.2f} bps  ({ann*100:.3f}%)
    Raw alpha             : 50 bps  (0.500%)
    Net alpha             : {(0.005 - ann)*100:.3f}%
    """).rstrip())
    print("=" * 72)


def _fmt_mcap(mcap: float) -> str:
    if mcap >= 1e12:
        return f"${mcap / 1e12:.1f}T"
    elif mcap >= 1e9:
        return f"${mcap / 1e9:.0f}B"
    else:
        return f"${mcap / 1e6:.0f}M"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    _run_example()
