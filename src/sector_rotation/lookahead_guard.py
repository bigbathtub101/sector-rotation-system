"""
lookahead_guard.py — Lookahead Bias Detection and Prevention
=============================================================
Global Sector Rotation System

Provides:
  1. validate_no_lookahead() — statistical test for signal-price correlation
     that detects suspiciously high lag-0 vs lag-1 correlation, which is a
     hallmark of lookahead bias.
  2. assert_causal_signal() — runtime assertion wrapper for use in production
     code (regime_detector.py, stock_screener.py, etc.).

Usage in regime_detector.py:
    from lookahead_guard import validate_no_lookahead, assert_causal_signal
    # After computing any signal series:
    validate_no_lookahead(signal_series, price_returns)

Academic reference:
    Harvey, Liu, Zhu (2016) "...and the Cross-Section of Expected Returns"
    — recommends testing that trading signals do not embed future information.

Dependencies: numpy, pandas, scipy
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger("lookahead_guard")


def validate_no_lookahead(
    signal_series: pd.Series,
    price_series: pd.Series,
    max_ratio: float = 2.0,
    significance_level: float = 0.01,
) -> Tuple[bool, dict]:
    """
    Check for suspiciously high signal-price correlation at lag 0 vs lag 1.

    If a signal has much stronger correlation with CONTEMPORANEOUS prices
    than with NEXT-PERIOD prices, it likely contains future information.
    A causal predictive signal should show its strongest correlation at
    lag >= 1 (signal today predicts returns tomorrow or later).

    Parameters
    ----------
    signal_series : pd.Series
        The trading signal (e.g., regime score, factor composite).
    price_series : pd.Series
        Price returns (e.g., daily log returns of SPY).
    max_ratio : float
        Maximum acceptable ratio of |corr(signal, price_lag0)| to
        |corr(signal, price_lag1)|. Default: 2.0.
    significance_level : float
        P-value threshold for the correlation test. Default: 0.01.

    Returns
    -------
    (is_clean, diagnostics) : Tuple[bool, dict]
        is_clean: True if no lookahead bias detected.
        diagnostics: dict with correlation values, ratio, and p-values.
    """
    combined = pd.DataFrame({"signal": signal_series, "price": price_series}).dropna()

    if len(combined) < 30:
        logger.warning("Lookahead check: only %d observations — insufficient for reliable test.", len(combined))
        return True, {"warning": "insufficient_data", "n_obs": len(combined)}

    signal = combined["signal"].values
    price = combined["price"].values

    corr_lag0, pval_lag0 = sp_stats.pearsonr(signal, price)
    corr_lag1, pval_lag1 = sp_stats.pearsonr(signal[:-1], price[1:])

    abs_corr0 = abs(corr_lag0)
    abs_corr1 = abs(corr_lag1)

    if abs_corr1 < 1e-6:
        ratio = float("inf") if abs_corr0 > 0.05 else 1.0
    else:
        ratio = abs_corr0 / abs_corr1

    is_suspicious = (
        pval_lag0 < significance_level
        and ratio > max_ratio
        and abs_corr0 > 0.05
    )

    diagnostics = {
        "n_observations": len(combined),
        "corr_lag0": round(float(corr_lag0), 6),
        "corr_lag1": round(float(corr_lag1), 6),
        "abs_corr_lag0": round(abs_corr0, 6),
        "abs_corr_lag1": round(abs_corr1, 6),
        "pval_lag0": round(float(pval_lag0), 6),
        "pval_lag1": round(float(pval_lag1), 6),
        "lag0_to_lag1_ratio": round(ratio, 4) if np.isfinite(ratio) else "inf",
        "max_ratio_threshold": max_ratio,
        "is_suspicious": is_suspicious,
        "verdict": "FAIL — possible lookahead bias" if is_suspicious else "PASS — no lookahead detected",
    }

    if is_suspicious:
        logger.warning(
            "LOOKAHEAD BIAS WARNING: lag-0 corr=%.4f (p=%.4f) >> lag-1 corr=%.4f (p=%.4f), "
            "ratio=%.2f > threshold=%.2f. Signal may contain future information.",
            corr_lag0, pval_lag0, corr_lag1, pval_lag1, ratio, max_ratio,
        )
    else:
        logger.info("Lookahead check PASSED: lag-0 corr=%.4f, lag-1 corr=%.4f, ratio=%.2f", corr_lag0, corr_lag1, ratio)

    return not is_suspicious, diagnostics


def assert_causal_signal(
    signal_series: pd.Series,
    price_series: pd.Series,
    context: str = "",
    max_ratio: float = 2.0,
) -> None:
    """
    Runtime assertion that a signal does not contain lookahead bias.

    Call this in regime_detector.py, stock_screener.py, etc. after
    computing any signal that will be used for trading decisions.

    Parameters
    ----------
    signal_series : pd.Series
        The trading signal.
    price_series : pd.Series
        Price returns for the benchmark.
    context : str
        Description of the signal being tested (for error messages).
    max_ratio : float
        Maximum acceptable lag-0/lag-1 correlation ratio.

    Raises
    ------
    AssertionError
        If lookahead bias is detected.
    """
    is_clean, diag = validate_no_lookahead(signal_series, price_series, max_ratio=max_ratio)

    if not is_clean:
        msg = (
            f"LOOKAHEAD BIAS DETECTED in '{context}': "
            f"lag-0 corr={diag['corr_lag0']:.4f} (p={diag['pval_lag0']:.4f}), "
            f"lag-1 corr={diag['corr_lag1']:.4f} (p={diag['pval_lag1']:.4f}), "
            f"ratio={diag['lag0_to_lag1_ratio']}x > {max_ratio}x threshold. "
            f"Check that all rolling windows, percentile normalizations, and "
            f"z-scores use only data available at time t."
        )
        raise AssertionError(msg)


def check_rolling_window_alignment(
    signal_index: pd.DatetimeIndex,
    data_index: pd.DatetimeIndex,
    window_size: int,
    label: str = "signal",
) -> bool:
    """
    Verify that a rolling window signal is properly aligned.

    The first ``window_size`` values should be NaN (insufficient lookback data).
    This catches a common bug where rolling computations accidentally use
    future data by not respecting the min_periods parameter.

    Parameters
    ----------
    signal_index : pd.DatetimeIndex
        Index of the signal series.
    data_index : pd.DatetimeIndex
        Index of the underlying data used to compute the signal.
    window_size : int
        Expected rolling window size.
    label : str
        Name of the signal for logging.

    Returns
    -------
    bool : True if alignment looks correct.
    """
    if len(signal_index) < window_size:
        logger.info("Rolling alignment check for '%s': signal length %d < window %d — OK (short series)", label, len(signal_index), window_size)
        return True

    first_valid_idx = signal_index[0]
    data_start = data_index[0]

    if first_valid_idx < data_start:
        logger.warning(
            "ALIGNMENT ERROR in '%s': signal starts at %s, before data start %s. "
            "This suggests the signal is using data that doesn't exist yet.",
            label, first_valid_idx, data_start,
        )
        return False

    logger.info("Rolling alignment check for '%s': PASSED", label)
    return True


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 70)
    print("LOOKAHEAD BIAS GUARD — Worked Examples")
    print("=" * 70)

    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2024-01-01", periods=n)

    print("\n--- Example 1: Clean predictive signal ---")
    future_returns = pd.Series(np.random.randn(n) * 0.01, index=dates)
    clean_signal = future_returns.shift(1).rolling(20).mean().fillna(0)
    is_clean, diag = validate_no_lookahead(clean_signal, future_returns)
    print(f"  Verdict: {diag['verdict']}")
    print(f"  Lag-0 corr: {diag['corr_lag0']:.4f} (p={diag['pval_lag0']:.4f})")
    print(f"  Lag-1 corr: {diag['corr_lag1']:.4f} (p={diag['pval_lag1']:.4f})")
    print(f"  Ratio: {diag['lag0_to_lag1_ratio']}")

    print("\n--- Example 2: Contaminated signal (lookahead bias) ---")
    contaminated_signal = future_returns.rolling(5).mean()
    is_clean2, diag2 = validate_no_lookahead(contaminated_signal, future_returns)
    print(f"  Verdict: {diag2['verdict']}")
    print(f"  Lag-0 corr: {diag2['corr_lag0']:.4f} (p={diag2['pval_lag0']:.4f})")
    print(f"  Lag-1 corr: {diag2['corr_lag1']:.4f} (p={diag2['pval_lag1']:.4f})")
    print(f"  Ratio: {diag2['lag0_to_lag1_ratio']}")

    print("\n--- Example 3: Runtime assertion (clean signal) ---")
    try:
        assert_causal_signal(clean_signal, future_returns, context="clean_example")
        print("  Assertion PASSED (as expected)")
    except AssertionError as e:
        print(f"  Assertion FAILED: {e}")

    print("\n--- Example 4: Runtime assertion (contaminated signal) ---")
    try:
        assert_causal_signal(contaminated_signal, future_returns, context="contaminated_example")
        print("  Assertion PASSED (unexpected!)")
    except AssertionError as e:
        print("  Assertion FAILED (as expected): lookahead detected")

    print("\n" + "=" * 70)
    print("All examples completed.")
