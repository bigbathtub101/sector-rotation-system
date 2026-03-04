"""
walk_forward.py — Walk-Forward Validation Module
=================================================
Quantitative Sector Rotation System

Implements a rolling walk-forward backtest engine for the sector rotation
system.  The engine is intentionally self-contained: it defines its own
simplified regime detector and portfolio optimizer so that the full
validation pipeline can be run with nothing but pandas, numpy, and scipy —
no database, no live data, no paid APIs.

Design
------
Training window : 504 trading days  (~2 years)
Test window     : 63 trading days   (~3 months)
Roll step       : 63 trading days   (non-overlapping OOS periods)

For each OOS window the engine:
  1. Trains on the preceding 504-day window.
  2. Detects the dominant market regime from training data.
  3. Computes regime-conditional portfolio weights.
  4. Applies those weights to the 63-day OOS period.
  5. Records Sharpe ratio, maximum drawdown, and alpha vs SPY.

Usage (library)
---------------
    from walk_forward import WalkForwardBacktest, generate_dummy_data

    prices = generate_dummy_data(seed=42)
    wf = WalkForwardBacktest(prices)
    results_df, equity_curve = wf.run()
    wf.summary()

Usage (CLI)
-----------
    python walk_forward.py

Dependencies
------------
    pandas >= 1.3
    numpy  >= 1.21
    scipy  >= 1.7
"""

from __future__ import annotations

import logging
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_WINDOW: int = 504   # ~2 trading years
TEST_WINDOW: int = 63     # ~3 trading months (one OOS period / roll step)
TRADING_DAYS_PER_YEAR: float = 252.0
ANNUALIZATION_FACTOR: float = np.sqrt(TRADING_DAYS_PER_YEAR)

# Sector ticker universe used in the worked example
SECTOR_TICKERS: List[str] = [
    "XLK",   # Technology
    "XLV",   # Health Care
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLE",   # Energy
    "XLB",   # Materials
    "XLU",   # Utilities
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
]

# ============================================================================
# Section 1 — Simplified Regime Detector
# ============================================================================

def detect_regime_simple(
    prices: pd.DataFrame,
    vol_window: int = 21,
    zscore_lookback: int = 252,
) -> str:
    """Detect the dominant market regime from a training price window.

    Uses a rolling-volatility z-score approach as a lightweight proxy for
    the full wedge-volume detector in ``regime_detector.py``.  The logic is
    deliberately simple so the walk-forward engine has no external
    dependencies.

    Algorithm
    ---------
    1. Compute daily log returns for every ticker in ``prices``.
    2. Compute the equal-weighted average cross-sectional return each day.
    3. Compute a 21-day rolling realised volatility of that average.
    4. Z-score the most-recent volatility observation against the trailing
       ``zscore_lookback``-day distribution.
    5. Classify:
       - z >  1.5  → ``"panic"``   (stress / high-vol regime)
       - z < -0.5  → ``"offense"`` (low-vol / risk-on regime)
       - otherwise → ``"defense"`` (transitional regime)

    Parameters
    ----------
    prices:
        Wide-format price DataFrame.  Index must be DatetimeIndex; columns
        are tickers (must include ``'SPY'``).
    vol_window:
        Number of trading days for the rolling realised-vol calculation.
    zscore_lookback:
        Number of trailing observations used to standardise the volatility.

    Returns
    -------
    str
        One of ``"offense"``, ``"defense"``, or ``"panic"``.
    """
    if prices.empty or len(prices) < vol_window + 2:
        logger.warning("Insufficient training data for regime detection; defaulting to 'defense'.")
        return "defense"

    log_returns: pd.DataFrame = np.log(prices / prices.shift(1)).dropna()

    # Equal-weighted average daily return across all tickers
    avg_ret: pd.Series = log_returns.mean(axis=1)

    # Rolling realised volatility (annualised)
    rolling_vol: pd.Series = (
        avg_ret.rolling(window=vol_window, min_periods=max(vol_window // 2, 5))
        .std()
        .dropna()
        * ANNUALIZATION_FACTOR
    )

    if len(rolling_vol) < max(zscore_lookback // 4, 30):
        # Not enough history for a meaningful z-score; use raw level
        current_vol: float = float(rolling_vol.iloc[-1])
        median_vol: float = float(rolling_vol.median())
        if current_vol > median_vol * 1.5:
            return "panic"
        elif current_vol < median_vol * 0.75:
            return "offense"
        return "defense"

    # Use up to zscore_lookback observations for the distribution
    lookback_vol: pd.Series = rolling_vol.iloc[-min(zscore_lookback, len(rolling_vol)):]
    current_vol = float(rolling_vol.iloc[-1])
    mu: float = float(lookback_vol.mean())
    sigma: float = float(lookback_vol.std())

    if sigma < 1e-10:
        return "defense"

    z: float = (current_vol - mu) / sigma

    if z > 1.5:
        return "panic"
    elif z < -0.5:
        return "offense"
    else:
        return "defense"


# ============================================================================
# Section 2 — Simplified Portfolio Optimizer
# ============================================================================

# Regime-conditional allocations (fraction to equities vs. cash/defensive)
_REGIME_EQUITY_ALLOCATION: Dict[str, float] = {
    "offense": 1.00,   # fully invested
    "defense": 0.70,   # 70% equities, 30% cash
    "panic":   0.30,   # 30% equities, 70% cash
}

# Sectors classified as "defensive" (included in defense/panic regimes)
_DEFENSIVE_SECTORS: List[str] = ["XLV", "XLU", "XLP"]

# Sectors classified as "offensive" (over-weighted in offense regime)
_OFFENSIVE_SECTORS: List[str] = ["XLK", "XLC", "XLY", "XLF", "XLI"]


def optimize_weights_simple(
    prices: pd.DataFrame,
    regime: str,
) -> Dict[str, float]:
    """Compute regime-conditional equal-weight portfolio allocations.

    This is a streamlined version of the CVaR optimizer in
    ``portfolio_optimizer.py``.  It applies regime-conditional sector
    tilts with equal-weighting within each bucket, and parks the
    residual allocation in cash (represented as ``SPY`` at zero weight —
    i.e. we simply reduce total equity exposure and let the cash drag
    materialise naturally through the equity allocation fraction).

    Regime logic
    ------------
    - **Offense** : Equal-weight all 11 sector ETFs (100% equity).
    - **Defense** : Equal-weight 3 defensive sectors (``XLV``, ``XLU``,
      ``XLP``) at 70% equity allocation; remainder is cash (zero weight,
      not invested).
    - **Panic**   : Equal-weight the same 3 defensive sectors at 30%
      equity; remainder is cash.

    Parameters
    ----------
    prices:
        Training-window price DataFrame.  Columns are tickers; ``'SPY'``
        column is ignored for weight construction but must be present in
        the broader DataFrame.
    regime:
        One of ``"offense"``, ``"defense"``, ``"panic"``.

    Returns
    -------
    Dict[str, float]
        Mapping of ticker → weight.  Weights sum to ``equity_allocation``
        (< 1.0 in defense / panic, meaning the rest is held in cash).
        ``'SPY'`` is never assigned a non-zero weight by this optimizer.
    """
    equity_alloc: float = _REGIME_EQUITY_ALLOCATION.get(regime, 0.70)

    # Candidate sector tickers (exclude SPY from the investable universe)
    available_sectors: List[str] = [c for c in prices.columns if c != "SPY"]

    if regime == "offense":
        bucket: List[str] = available_sectors
    else:
        # Defense / Panic: tilt towards defensive sectors
        bucket = [t for t in _DEFENSIVE_SECTORS if t in available_sectors]
        if not bucket:
            # Fallback: equal-weight all available sectors
            bucket = available_sectors

    if not bucket:
        logger.warning("No valid tickers in bucket for regime '%s'.", regime)
        return {}

    per_ticker_weight: float = equity_alloc / len(bucket)
    weights: Dict[str, float] = {ticker: per_ticker_weight for ticker in bucket}

    logger.debug(
        "Regime '%s': equity_alloc=%.2f, %d tickers, per_ticker=%.4f",
        regime, equity_alloc, len(bucket), per_ticker_weight,
    )
    return weights


# ============================================================================
# Section 3 — Performance Metrics
# ============================================================================

def compute_sharpe(
    portfolio_returns: pd.Series,
    risk_free_rate_annual: float = 0.04,
) -> float:
    """Compute the annualised Sharpe ratio for a return series.

    Parameters
    ----------
    portfolio_returns:
        Daily arithmetic portfolio returns.
    risk_free_rate_annual:
        Annual risk-free rate (default: 4.0%).

    Returns
    -------
    float
        Annualised Sharpe ratio, or ``np.nan`` if volatility is zero.
    """
    if len(portfolio_returns) < 2:
        return np.nan

    daily_rf: float = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
    excess: pd.Series = portfolio_returns - daily_rf
    mu: float = float(excess.mean())
    sigma: float = float(excess.std(ddof=1))

    if sigma < 1e-12:
        return np.nan

    return (mu / sigma) * ANNUALIZATION_FACTOR


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum drawdown of an equity curve.

    Maximum drawdown is the largest peak-to-trough decline expressed as a
    negative fraction (e.g. ``-0.15`` means a 15% loss from peak).

    Parameters
    ----------
    equity_curve:
        Cumulative equity series (e.g. starting at 1.0).

    Returns
    -------
    float
        Maximum drawdown (≤ 0).
    """
    if equity_curve.empty:
        return 0.0

    running_max: pd.Series = equity_curve.cummax()
    drawdown: pd.Series = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return float(drawdown.min())


def compute_alpha_vs_spy(
    portfolio_returns: pd.Series,
    spy_returns: pd.Series,
    risk_free_rate_annual: float = 0.04,
) -> float:
    """Compute Jensen's alpha of the portfolio relative to SPY.

    Runs an OLS regression of portfolio excess returns on SPY excess returns
    and returns the annualised intercept (alpha).

    Parameters
    ----------
    portfolio_returns:
        Daily portfolio arithmetic returns.
    spy_returns:
        Daily SPY arithmetic returns aligned to the same dates.
    risk_free_rate_annual:
        Annual risk-free rate (default: 4.0%).

    Returns
    -------
    float
        Annualised Jensen's alpha.  Positive values indicate outperformance.
    """
    aligned: pd.DataFrame = pd.concat(
        [portfolio_returns.rename("port"), spy_returns.rename("spy")],
        axis=1,
    ).dropna()

    if len(aligned) < 5:
        return np.nan

    daily_rf: float = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
    port_excess: pd.Series = aligned["port"] - daily_rf
    spy_excess: pd.Series = aligned["spy"] - daily_rf

    slope, intercept, *_ = sp_stats.linregress(spy_excess.values, port_excess.values)
    daily_alpha: float = intercept
    return daily_alpha * TRADING_DAYS_PER_YEAR


# ============================================================================
# Section 4 — WalkForwardBacktest Class
# ============================================================================

class WalkForwardBacktest:
    """Rolling walk-forward backtest engine for sector rotation strategies.

    The engine splits a multi-year price history into alternating training
    and test windows, trains a regime + weight model on each training window,
    and evaluates performance on each out-of-sample test window.

    Parameters
    ----------
    prices:
        Wide-format DataFrame of daily **adjusted close** prices.
        Index must be a ``DatetimeIndex``; columns must include ``'SPY'``
        plus at least 3 sector tickers.
    regime_fn:
        Callable ``(prices: pd.DataFrame) -> str`` that returns one of
        ``"offense"``, ``"defense"``, or ``"panic"`` given a training-window
        price DataFrame.  Defaults to :func:`detect_regime_simple`.
    optimizer_fn:
        Callable ``(prices: pd.DataFrame, regime: str) -> Dict[str, float]``
        that returns a weight dictionary.  Defaults to
        :func:`optimize_weights_simple`.
    train_window:
        Number of trading days in the training window (default: 504).
    test_window:
        Number of trading days in each OOS test window (default: 63).
    risk_free_rate_annual:
        Annual risk-free rate used for Sharpe and alpha calculations
        (default: 0.04).

    Attributes
    ----------
    results_ : pd.DataFrame or None
        Per-window metrics table; populated after :meth:`run` is called.
    equity_curve_ : pd.Series or None
        Cumulative equity curve across all OOS windows; populated after
        :meth:`run` is called.

    Examples
    --------
    >>> prices = generate_dummy_data(seed=42)
    >>> wf = WalkForwardBacktest(prices)
    >>> results_df, equity_curve = wf.run()
    >>> wf.summary()
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        regime_fn: Optional[Callable[[pd.DataFrame], str]] = None,
        optimizer_fn: Optional[Callable[[pd.DataFrame, str], Dict[str, float]]] = None,
        train_window: int = TRAIN_WINDOW,
        test_window: int = TEST_WINDOW,
        risk_free_rate_annual: float = 0.04,
    ) -> None:
        self.prices: pd.DataFrame = prices.copy().sort_index()
        self.regime_fn: Callable = regime_fn if regime_fn is not None else detect_regime_simple
        self.optimizer_fn: Callable = optimizer_fn if optimizer_fn is not None else optimize_weights_simple
        self.train_window: int = train_window
        self.test_window: int = test_window
        self.risk_free_rate_annual: float = risk_free_rate_annual

        # Results populated by run()
        self.results_: Optional[pd.DataFrame] = None
        self.equity_curve_: Optional[pd.Series] = None

        self._validate_inputs()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        """Validate that the price DataFrame meets minimum requirements."""
        if "SPY" not in self.prices.columns:
            raise ValueError(
                "Price DataFrame must include a 'SPY' column for benchmark comparison."
            )
        min_days: int = self.train_window + self.test_window
        if len(self.prices) < min_days:
            raise ValueError(
                f"Price DataFrame has {len(self.prices)} rows, but needs at least "
                f"{min_days} rows (train_window={self.train_window} + "
                f"test_window={self.test_window})."
            )
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError("Price DataFrame index must be a DatetimeIndex.")

        n_sectors: int = len([c for c in self.prices.columns if c != "SPY"])
        if n_sectors < 3:
            raise ValueError(
                f"Need at least 3 sector tickers (plus SPY); found {n_sectors}."
            )

        logger.info(
            "WalkForwardBacktest initialised: %d days, %d tickers, "
            "train=%d, test=%d",
            len(self.prices),
            len(self.prices.columns),
            self.train_window,
            self.test_window,
        )

    # ------------------------------------------------------------------
    # Core single-window evaluation
    # ------------------------------------------------------------------

    def _run_single_window(
        self,
        train_prices: pd.DataFrame,
        test_prices: pd.DataFrame,
    ) -> Tuple[str, Dict[str, float], pd.Series, pd.Series]:
        """Run one train/test iteration and return metrics components.

        Parameters
        ----------
        train_prices:
            Price slice for the training window.
        test_prices:
            Price slice for the OOS test window.

        Returns
        -------
        regime : str
        weights : Dict[str, float]
        port_returns : pd.Series  — daily portfolio returns in test window
        spy_returns  : pd.Series  — daily SPY returns in test window
        """
        # 1. Detect regime from training data
        regime: str = self.regime_fn(train_prices)

        # 2. Compute portfolio weights from training data
        weights: Dict[str, float] = self.optimizer_fn(train_prices, regime)

        # 3. Compute daily returns in the test window
        test_log_ret: pd.DataFrame = np.log(test_prices / test_prices.shift(1)).dropna()

        # Arithmetic returns for performance metrics
        test_arith_ret: pd.DataFrame = test_prices.pct_change().dropna()

        # Portfolio return: weighted sum of arithmetic returns
        port_arith: pd.Series = pd.Series(0.0, index=test_arith_ret.index)
        for ticker, weight in weights.items():
            if ticker in test_arith_ret.columns and weight != 0.0:
                port_arith += weight * test_arith_ret[ticker]

        # SPY benchmark
        spy_arith: pd.Series = (
            test_arith_ret["SPY"] if "SPY" in test_arith_ret.columns
            else pd.Series(0.0, index=test_arith_ret.index)
        )

        return regime, weights, port_arith, spy_arith

    # ------------------------------------------------------------------
    # Main run() method
    # ------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Execute the full walk-forward backtest.

        Iterates over all valid non-overlapping OOS windows, accumulates
        per-window performance metrics, and assembles the cumulative
        equity curve.

        Returns
        -------
        results_df : pd.DataFrame
            One row per OOS window with columns:
            ``window_id``, ``train_start``, ``train_end``,
            ``test_start``, ``test_end``, ``regime``,
            ``sharpe``, ``max_drawdown``, ``alpha_vs_spy``,
            ``ann_return``, ``ann_vol``.
        equity_curve : pd.Series
            Daily cumulative equity values (starting at 1.0) across all
            OOS windows, indexed by date.

        Raises
        ------
        RuntimeError
            If no valid walk-forward windows can be constructed.
        """
        n_rows: int = len(self.prices)
        window_records: List[Dict] = []
        all_port_returns: List[pd.Series] = []

        window_id: int = 0
        # First training window starts at index 0
        train_start_idx: int = 0

        logger.info(
            "Starting walk-forward: %d total days, train=%d, test=%d",
            n_rows, self.train_window, self.test_window,
        )

        while True:
            train_end_idx: int = train_start_idx + self.train_window
            test_end_idx: int = train_end_idx + self.test_window

            if test_end_idx > n_rows:
                logger.info(
                    "Stopping: test window would exceed data bounds "
                    "(train_start=%d, test_end=%d, n_rows=%d).",
                    train_start_idx, test_end_idx, n_rows,
                )
                break

            # Slice the price data
            train_prices: pd.DataFrame = self.prices.iloc[train_start_idx:train_end_idx]
            test_prices: pd.DataFrame = self.prices.iloc[train_end_idx - 1 : test_end_idx]
            # Note: include last training-window day as the price base for return computation

            train_start_date = self.prices.index[train_start_idx]
            train_end_date = self.prices.index[train_end_idx - 1]
            test_start_date = self.prices.index[train_end_idx]
            test_end_date = self.prices.index[min(test_end_idx - 1, n_rows - 1)]

            logger.debug(
                "Window %d: train [%s → %s], test [%s → %s]",
                window_id,
                train_start_date.date(), train_end_date.date(),
                test_start_date.date(), test_end_date.date(),
            )

            try:
                regime, weights, port_ret, spy_ret = self._run_single_window(
                    train_prices, test_prices
                )
            except Exception as exc:
                logger.error("Window %d failed: %s", window_id, exc, exc_info=True)
                # Advance the window and continue
                train_start_idx += self.test_window
                window_id += 1
                continue

            # ---- Performance metrics ----
            sharpe: float = compute_sharpe(port_ret, self.risk_free_rate_annual)
            eq_curve_window: pd.Series = (1.0 + port_ret).cumprod()
            max_dd: float = compute_max_drawdown(eq_curve_window)
            alpha: float = compute_alpha_vs_spy(port_ret, spy_ret, self.risk_free_rate_annual)

            # Annualised return and volatility for the window
            ann_ret: float = (
                float((1.0 + port_ret).prod()) ** (TRADING_DAYS_PER_YEAR / len(port_ret)) - 1.0
                if len(port_ret) > 0 else np.nan
            )
            ann_vol: float = (
                float(port_ret.std(ddof=1)) * ANNUALIZATION_FACTOR
                if len(port_ret) > 1 else np.nan
            )

            record: Dict = {
                "window_id": window_id,
                "train_start": train_start_date,
                "train_end": train_end_date,
                "test_start": test_start_date,
                "test_end": test_end_date,
                "regime": regime,
                "sharpe": round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
                "max_drawdown": round(max_dd, 4) if not np.isnan(max_dd) else np.nan,
                "alpha_vs_spy": round(alpha, 4) if not np.isnan(alpha) else np.nan,
                "ann_return": round(ann_ret, 4) if not np.isnan(ann_ret) else np.nan,
                "ann_vol": round(ann_vol, 4) if not np.isnan(ann_vol) else np.nan,
                "n_positions": sum(1 for w in weights.values() if w > 1e-6),
                "equity_alloc": round(sum(weights.values()), 4),
            }
            window_records.append(record)
            all_port_returns.append(port_ret)

            logger.info(
                "Window %02d | %s → %s | regime=%-8s | "
                "sharpe=%+.3f | max_dd=%.2f%% | alpha=%+.3f%%",
                window_id,
                test_start_date.date(),
                test_end_date.date(),
                regime,
                sharpe if not np.isnan(sharpe) else 0.0,
                max_dd * 100,
                alpha * 100 if not np.isnan(alpha) else 0.0,
            )

            # Roll forward by one test window
            train_start_idx += self.test_window
            window_id += 1

        if not window_records:
            raise RuntimeError(
                "No walk-forward windows could be completed.  "
                "Check that prices has enough rows."
            )

        # ---- Assemble results ----
        results_df: pd.DataFrame = pd.DataFrame(window_records)
        results_df = results_df.set_index("window_id")

        # Cumulative equity curve across all OOS periods
        all_returns: pd.Series = pd.concat(all_port_returns).sort_index()
        equity_curve: pd.Series = (1.0 + all_returns).cumprod()
        equity_curve.name = "portfolio_equity"

        # Drop any duplicate dates (can occur at window boundaries)
        equity_curve = equity_curve[~equity_curve.index.duplicated(keep="first")]

        self.results_ = results_df
        self.equity_curve_ = equity_curve

        logger.info(
            "Walk-forward complete: %d windows, OOS span %s → %s, "
            "terminal equity %.4f",
            len(results_df),
            equity_curve.index[0].date(),
            equity_curve.index[-1].date(),
            equity_curve.iloc[-1],
        )

        return results_df, equity_curve

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, float]:
        """Print and return aggregate performance statistics.

        Must be called after :meth:`run`.

        Returns
        -------
        Dict[str, float]
            Dictionary of aggregate statistics:
            ``n_windows``, ``ann_return``, ``ann_sharpe``,
            ``max_drawdown_full``, ``cumulative_alpha_vs_spy``,
            ``pct_offense``, ``pct_defense``, ``pct_panic``,
            ``terminal_equity``.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called yet.
        """
        if self.results_ is None or self.equity_curve_ is None:
            raise RuntimeError("Call run() before summary().")

        df: pd.DataFrame = self.results_
        curve: pd.Series = self.equity_curve_

        # ---- Full-period annualised return from equity curve ----
        n_oos_days: int = len(curve)
        terminal_equity: float = float(curve.iloc[-1])
        ann_return: float = terminal_equity ** (TRADING_DAYS_PER_YEAR / n_oos_days) - 1.0

        # ---- Annualised Sharpe on the full OOS return series ----
        all_returns: pd.Series = curve.pct_change().dropna()
        ann_sharpe: float = compute_sharpe(all_returns, self.risk_free_rate_annual)

        # ---- Full-period maximum drawdown ----
        max_dd_full: float = compute_max_drawdown(curve)

        # ---- Cumulative alpha vs SPY ----
        # Average of per-window annualised alphas (weighted by window length)
        valid_alphas: pd.Series = df["alpha_vs_spy"].dropna()
        cumulative_alpha: float = float(valid_alphas.mean()) if not valid_alphas.empty else np.nan

        # ---- Regime distribution ----
        regime_counts: pd.Series = df["regime"].value_counts(normalize=True)
        pct_offense: float = float(regime_counts.get("offense", 0.0))
        pct_defense: float = float(regime_counts.get("defense", 0.0))
        pct_panic: float = float(regime_counts.get("panic", 0.0))

        stats: Dict[str, float] = {
            "n_windows": len(df),
            "ann_return": round(ann_return, 4),
            "ann_sharpe": round(ann_sharpe, 4) if not np.isnan(ann_sharpe) else np.nan,
            "max_drawdown_full": round(max_dd_full, 4),
            "cumulative_alpha_vs_spy": round(cumulative_alpha, 4) if not np.isnan(cumulative_alpha) else np.nan,
            "pct_offense": round(pct_offense, 4),
            "pct_defense": round(pct_defense, 4),
            "pct_panic": round(pct_panic, 4),
            "terminal_equity": round(terminal_equity, 4),
        }

        # ---- Pretty-print ----
        divider: str = "=" * 70
        print(f"\n{divider}")
        print("WALK-FORWARD BACKTEST SUMMARY")
        print(divider)
        print(f"  OOS Windows          : {stats['n_windows']}")
        print(f"  OOS Period           : {self.equity_curve_.index[0].date()} "
              f"→ {self.equity_curve_.index[-1].date()}")
        print(f"  Terminal Equity      : {terminal_equity:.4f}  "
              f"(started at 1.0000)")
        print(f"  Annualised Return    : {ann_return * 100:+.2f}%")
        print(f"  Annualised Sharpe    : {ann_sharpe:+.3f}")
        print(f"  Full-Period Max DD   : {max_dd_full * 100:.2f}%")
        print(f"  Avg Alpha vs SPY     : {cumulative_alpha * 100:+.2f}%  (annualised, per window)")
        print(f"  Regime Distribution  : offense={pct_offense:.0%} | "
              f"defense={pct_defense:.0%} | panic={pct_panic:.0%}")
        print(divider)

        return stats


# ============================================================================
# Section 5 — Dummy Data Generator
# ============================================================================

def generate_dummy_data(
    seed: int = 42,
    n_years: int = 5,
    start_date: str = "2019-01-02",
    tickers: Optional[List[str]] = None,
    include_spy: bool = True,
) -> pd.DataFrame:
    """Generate realistic synthetic sector ETF price data.

    Creates a multi-sector price DataFrame using correlated geometric
    Brownian motion with time-varying volatility to simulate distinct
    market regimes (calm, volatile/crisis, recovery).

    Statistical design
    ------------------
    * Five synthetic regimes of roughly equal length are embedded:
      calm growth → elevated vol → crisis (high vol + negative drift) →
      recovery → late-cycle.
    * Sector returns are drawn from a multivariate normal with a realistic
      correlation structure (tech/discretionary highly correlated;
      utilities/staples mildly negatively correlated with cyclicals).
    * SPY return is the equal-weighted average of all sector returns plus
      a small idiosyncratic term.

    Parameters
    ----------
    seed:
        NumPy random seed for full reproducibility (default: 42).
    n_years:
        Number of calendar years of data to generate (default: 5).
    start_date:
        First date in the returned index (default: ``'2019-01-02'``).
    tickers:
        List of sector ticker names (default: :data:`SECTOR_TICKERS`).
    include_spy:
        Whether to include a ``'SPY'`` column (default: True).

    Returns
    -------
    pd.DataFrame
        Daily adjusted-close price DataFrame with a ``DatetimeIndex``.
        All prices start at 100.0 on ``start_date``.
    """
    rng: np.random.Generator = np.random.default_rng(seed)

    if tickers is None:
        tickers = SECTOR_TICKERS

    n_sectors: int = len(tickers)
    trading_days: int = int(n_years * TRADING_DAYS_PER_YEAR)

    date_index: pd.DatetimeIndex = pd.bdate_range(
        start=start_date, periods=trading_days, freq="B"
    )
    actual_days: int = len(date_index)

    # ----------------------------------------------------------------
    # Correlation matrix (realistic sector cross-correlations)
    # ----------------------------------------------------------------
    # Base pairwise correlation for cyclical sectors
    base_corr: float = 0.55

    corr_matrix: np.ndarray = np.full((n_sectors, n_sectors), base_corr)
    np.fill_diagonal(corr_matrix, 1.0)

    # Lookup: if tickers include the standard set, use domain knowledge
    _high_corr_pairs: Dict[str, List[str]] = {
        "XLK": ["XLC", "XLY"],
        "XLF": ["XLI", "XLB"],
        "XLU": ["XLP", "XLRE"],
    }
    _low_corr_pairs: Dict[str, List[str]] = {
        "XLK": ["XLU", "XLP"],
        "XLY": ["XLU", "XLP"],
    }

    ticker_idx: Dict[str, int] = {t: i for i, t in enumerate(tickers)}

    for base, related in _high_corr_pairs.items():
        if base in ticker_idx:
            for r in related:
                if r in ticker_idx:
                    i, j = ticker_idx[base], ticker_idx[r]
                    corr_matrix[i, j] = corr_matrix[j, i] = 0.80

    for base, related in _low_corr_pairs.items():
        if base in ticker_idx:
            for r in related:
                if r in ticker_idx:
                    i, j = ticker_idx[base], ticker_idx[r]
                    corr_matrix[i, j] = corr_matrix[j, i] = 0.20

    # Nearest positive-definite correction via eigenvalue clipping
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = np.clip(eigenvalues, 1e-8, None)
    corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Re-scale to correlation matrix (diagonal = 1)
    diag_sqrt: np.ndarray = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / np.outer(diag_sqrt, diag_sqrt)

    # ----------------------------------------------------------------
    # Time-varying volatility regimes
    # ----------------------------------------------------------------
    # Five roughly-equal regimes across the full horizon
    regime_length: int = actual_days // 5

    # (daily_vol, daily_drift)
    _regime_params: List[Tuple[float, float]] = [
        (0.010, +0.0006),   # calm growth
        (0.015, +0.0002),   # elevated vol
        (0.030, -0.0015),   # crisis
        (0.018, +0.0010),   # recovery
        (0.012, +0.0005),   # late-cycle
    ]

    daily_vols: np.ndarray = np.empty(actual_days)
    daily_drifts: np.ndarray = np.empty(actual_days)

    for seg_idx, (vol, drift) in enumerate(_regime_params):
        seg_start: int = seg_idx * regime_length
        seg_end: int = (seg_idx + 1) * regime_length if seg_idx < 4 else actual_days
        daily_vols[seg_start:seg_end] = vol
        daily_drifts[seg_start:seg_end] = drift

    # ----------------------------------------------------------------
    # Sector-specific drift adjustments (long-run sector differences)
    # ----------------------------------------------------------------
    _sector_drift_adj: Dict[str, float] = {
        "XLK": +0.0003,   # tech outperforms
        "XLC": +0.0001,   # comms slight premium
        "XLV": +0.0001,   # defensive growth
        "XLF": -0.0001,   # financials slight lag
        "XLU": -0.0002,   # utilities drag in bull
        "XLP": -0.0001,   # staples slight lag
        "XLE": +0.0000,   # energy neutral
        "XLB": +0.0000,   # materials neutral
        "XLI": +0.0001,   # industrials slight premium
        "XLRE": -0.0001,  # REIT slight lag
        "XLY": +0.0002,   # discretionary growth premium
    }

    # ----------------------------------------------------------------
    # Generate correlated log returns
    # ----------------------------------------------------------------
    cholesky_L: np.ndarray = np.linalg.cholesky(corr_matrix)

    sector_log_returns: np.ndarray = np.empty((actual_days, n_sectors))

    for day in range(actual_days):
        vol_t: float = daily_vols[day]
        drift_t: float = daily_drifts[day]

        # Draw correlated standard normals
        z: np.ndarray = rng.standard_normal(n_sectors)
        corr_z: np.ndarray = cholesky_L @ z

        for s, ticker in enumerate(tickers):
            sector_adj: float = _sector_drift_adj.get(ticker, 0.0)
            # GBM daily log return: (drift + sector_adj - 0.5*vol^2)*dt + vol*dW
            sector_log_returns[day, s] = (
                (drift_t + sector_adj - 0.5 * vol_t ** 2) + vol_t * corr_z[s]
            )

    # ----------------------------------------------------------------
    # Build price paths (all starting at 100)
    # ----------------------------------------------------------------
    prices_array: np.ndarray = np.empty((actual_days + 1, n_sectors))
    prices_array[0, :] = 100.0
    for day in range(actual_days):
        prices_array[day + 1, :] = prices_array[day, :] * np.exp(sector_log_returns[day, :])

    # Drop the initial seed row; keep only the actual_days rows
    prices_df: pd.DataFrame = pd.DataFrame(
        prices_array[1:],
        index=date_index,
        columns=tickers,
    )

    # ----------------------------------------------------------------
    # Construct SPY as equal-weighted average + small idiosyncratic noise
    # ----------------------------------------------------------------
    if include_spy:
        spy_log_ret: np.ndarray = (
            sector_log_returns.mean(axis=1)
            + rng.normal(0, 0.002, actual_days)
        )
        spy_prices: np.ndarray = np.empty(actual_days + 1)
        spy_prices[0] = 100.0
        for day in range(actual_days):
            spy_prices[day + 1] = spy_prices[day] * np.exp(spy_log_ret[day])

        prices_df["SPY"] = spy_prices[1:]

    # Round to 2 decimal places (realistic tick size)
    prices_df = prices_df.round(2)

    logger.info(
        "Generated dummy price data: %d days × %d tickers, "
        "start=%s, seed=%d",
        len(prices_df),
        len(prices_df.columns),
        start_date,
        seed,
    )

    return prices_df


# ============================================================================
# Section 6 — Utility Functions
# ============================================================================

def print_results_table(results_df: pd.DataFrame) -> None:
    """Pretty-print the per-window results table.

    Parameters
    ----------
    results_df:
        DataFrame returned by :meth:`WalkForwardBacktest.run`.
    """
    header_cols: List[str] = [
        "test_start", "test_end", "regime",
        "sharpe", "max_drawdown", "alpha_vs_spy",
    ]
    # Check which of our desired columns exist
    available: List[str] = [c for c in header_cols if c in results_df.columns]

    print("\n" + "=" * 80)
    print(f"{'WIN':>3}  {'TEST START':>12}  {'TEST END':>12}  "
          f"{'REGIME':>8}  {'SHARPE':>8}  {'MAX DD':>8}  {'ALPHA':>8}")
    print("-" * 80)

    for win_id, row in results_df.iterrows():
        sharpe_str: str = f"{row['sharpe']:+.3f}" if not pd.isna(row.get("sharpe")) else "   NaN"
        dd_str: str = f"{row['max_drawdown'] * 100:.2f}%" if not pd.isna(row.get("max_drawdown")) else "   NaN"
        alpha_str: str = f"{row['alpha_vs_spy'] * 100:+.2f}%" if not pd.isna(row.get("alpha_vs_spy")) else "   NaN"

        print(
            f"{win_id:>3}  "
            f"{str(row['test_start'].date()):>12}  "
            f"{str(row['test_end'].date()):>12}  "
            f"{str(row['regime']):>8}  "
            f"{sharpe_str:>8}  "
            f"{dd_str:>8}  "
            f"{alpha_str:>8}"
        )

    print("=" * 80)


def print_equity_curve_checkpoints(
    equity_curve: pd.Series,
    n_checkpoints: int = 8,
) -> None:
    """Print equity curve values at evenly spaced checkpoints.

    Parameters
    ----------
    equity_curve:
        Cumulative equity Series returned by :meth:`WalkForwardBacktest.run`.
    n_checkpoints:
        Number of equally spaced dates to sample (default: 8).
    """
    idx: np.ndarray = np.linspace(0, len(equity_curve) - 1, n_checkpoints, dtype=int)

    print("\n" + "=" * 50)
    print("EQUITY CURVE CHECKPOINTS")
    print("-" * 50)
    print(f"  {'DATE':>12}  {'EQUITY':>10}  {'DRAWDOWN':>10}")
    print("-" * 50)

    running_max: float = 1.0
    for i in idx:
        date = equity_curve.index[i]
        val: float = float(equity_curve.iloc[i])
        running_max = max(running_max, val)
        dd: float = (val - running_max) / running_max
        print(f"  {str(date.date()):>12}  {val:>10.4f}  {dd * 100:>9.2f}%")

    print("=" * 50)


# ============================================================================
# Section 7 — __main__ Worked Example
# ============================================================================

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Configure logging for the CLI run
    # -----------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        datefmt="%H:%M:%S",
    )

    print("\n" + "#" * 70)
    print("# WALK-FORWARD BACKTEST — WORKED NUMERICAL EXAMPLE")
    print("# Seed: 42 | 5 years daily data | 11 sectors + SPY")
    print("# Train window: 504 days (~2 years)")
    print("# Test window :  63 days (~3 months)")
    print("#" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Generate 5-year dummy price data with seed=42
    # -----------------------------------------------------------------------
    print("\n[1] Generating synthetic price data (seed=42, 5 years)...")
    prices: pd.DataFrame = generate_dummy_data(seed=42, n_years=5, start_date="2019-01-02")

    print(f"    Shape     : {prices.shape[0]} days × {prices.shape[1]} tickers")
    print(f"    Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"    Tickers   : {list(prices.columns)}")
    print("\n    First 3 rows:")
    print(prices.head(3).to_string())
    print("\n    Last 3 rows:")
    print(prices.tail(3).to_string())

    # -----------------------------------------------------------------------
    # Step 2: Initialise and run the walk-forward backtest
    # -----------------------------------------------------------------------
    print("\n[2] Initialising WalkForwardBacktest...")
    wf: WalkForwardBacktest = WalkForwardBacktest(
        prices=prices,
        # Use defaults: detect_regime_simple + optimize_weights_simple
        train_window=504,
        test_window=63,
        risk_free_rate_annual=0.04,
    )

    print("[3] Running walk-forward backtest (this may take a moment)...")
    results_df: pd.DataFrame
    equity_curve: pd.Series
    results_df, equity_curve = wf.run()

    # -----------------------------------------------------------------------
    # Step 3: Per-window results table
    # -----------------------------------------------------------------------
    print("\n[4] Per-window out-of-sample results:")
    print_results_table(results_df)

    # -----------------------------------------------------------------------
    # Step 4: Aggregate summary
    # -----------------------------------------------------------------------
    print("\n[5] Aggregate walk-forward statistics:")
    agg_stats: Dict[str, float] = wf.summary()

    # -----------------------------------------------------------------------
    # Step 5: Equity curve checkpoints
    # -----------------------------------------------------------------------
    print("\n[6] Equity curve at key checkpoints:")
    print_equity_curve_checkpoints(equity_curve, n_checkpoints=8)

    # -----------------------------------------------------------------------
    # Step 6: Regime distribution detail
    # -----------------------------------------------------------------------
    print("\n[7] Regime distribution across OOS windows:")
    regime_dist: pd.Series = results_df["regime"].value_counts()
    for regime_label, count in regime_dist.items():
        pct: float = count / len(results_df) * 100
        bar: str = "█" * int(pct / 5)
        print(f"    {regime_label:>8}: {count:>3} windows  ({pct:5.1f}%)  {bar}")

    # -----------------------------------------------------------------------
    # Step 7: Best and worst OOS windows
    # -----------------------------------------------------------------------
    print("\n[8] Best 3 OOS windows by Sharpe ratio:")
    best: pd.DataFrame = results_df.nlargest(3, "sharpe")[
        ["test_start", "test_end", "regime", "sharpe", "max_drawdown", "alpha_vs_spy"]
    ]
    for _, row in best.iterrows():
        print(
            f"    [{row['test_start'].date()} → {row['test_end'].date()}]  "
            f"regime={row['regime']:>8}  sharpe={row['sharpe']:+.3f}  "
            f"alpha={row['alpha_vs_spy'] * 100:+.2f}%"
        )

    print("\n[9] Worst 3 OOS windows by Sharpe ratio:")
    worst: pd.DataFrame = results_df.nsmallest(3, "sharpe")[
        ["test_start", "test_end", "regime", "sharpe", "max_drawdown", "alpha_vs_spy"]
    ]
    for _, row in worst.iterrows():
        print(
            f"    [{row['test_start'].date()} → {row['test_end'].date()}]  "
            f"regime={row['regime']:>8}  sharpe={row['sharpe']:+.3f}  "
            f"alpha={row['alpha_vs_spy'] * 100:+.2f}%"
        )

    # -----------------------------------------------------------------------
    # Step 8: Annualised return per regime type
    # -----------------------------------------------------------------------
    print("\n[10] Average annualised return by regime:")
    for regime_label in ["offense", "defense", "panic"]:
        regime_rows: pd.DataFrame = results_df[results_df["regime"] == regime_label]
        if not regime_rows.empty:
            avg_ret: float = float(regime_rows["ann_return"].mean())
            avg_sharpe: float = float(regime_rows["sharpe"].mean())
            print(
                f"    {regime_label:>8}: avg_ann_ret={avg_ret * 100:+.2f}%  "
                f"avg_sharpe={avg_sharpe:+.3f}  (n={len(regime_rows)})"
            )
        else:
            print(f"    {regime_label:>8}: no windows")

    # -----------------------------------------------------------------------
    # Step 9: Final equity curve statistics
    # -----------------------------------------------------------------------
    print("\n[11] Equity curve summary:")
    print(f"    Start value    : {equity_curve.iloc[0]:.4f}")
    print(f"    End value      : {equity_curve.iloc[-1]:.4f}")
    print(f"    Min value      : {equity_curve.min():.4f}  "
          f"(on {equity_curve.idxmin().date()})")
    print(f"    Max value      : {equity_curve.max():.4f}  "
          f"(on {equity_curve.idxmax().date()})")

    # Full-period max drawdown
    running_peak: pd.Series = equity_curve.cummax()
    dd_series: pd.Series = (equity_curve - running_peak) / running_peak
    worst_dd_date = dd_series.idxmin()
    print(f"    Worst drawdown : {dd_series.min() * 100:.2f}%  "
          f"(on {worst_dd_date.date()})")

    print("\n" + "#" * 70)
    print("# Example complete.")
    print("#" * 70 + "\n")
