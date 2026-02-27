"""
portfolio_optimizer.py — Phase 3: Factor Scoring & CVaR Portfolio Optimizer
============================================================================
Global Sector Rotation System

Implements:
  1. Fama-French Five-Factor scoring with McLean-Pontiff decay
  2. Momentum overlay (12-1 month)
  3. CVaR optimization with Ledoit-Wolf shrinkage
  4. Longin-Solnik tail correlation adjustment for EM positions
  5. Regime-conditional allocation bands (Section 8.1)
  6. US Equities sub-sector allocation (DeMiguel equal-weight + valuation filter)
  7. Bivector Beta for Energy/Materials structural diversifiers
  8. Dollar allocation across Taxable / Roth IRA accounts

Dependencies: numpy, pandas, scipy, sklearn, pypfopt, pandas_datareader, statsmodels, yaml
"""

import json
import logging
import sqlite3
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# LOGGING & CONFIG
# ---------------------------------------------------------------------------
logger = logging.getLogger("portfolio_optimizer")
CONFIG_PATH = Path(__file__).parent / "config.yaml"
DB_PATH = Path(__file__).parent / "rotation_system.db"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ===========================================================================
# 1. FAMA-FRENCH FIVE-FACTOR SCORING
# ===========================================================================

def download_ff_factors(start_date: str = None) -> pd.DataFrame:
    """
    Download Fama-French 5 factors from Ken French's data library.
    Returns a DataFrame with DAILY factor returns: Mkt-RF, SMB, HML, RMW, CMA, RF.

    NOTE: We use the daily dataset ("F-F_Research_Data_5_Factors_2x3_daily")
    rather than the monthly version because our factor loading regressions
    run on daily excess returns.  The window_months parameter in
    compute_factor_loadings() is converted to trading days (months * 21)
    to match this daily frequency.

    Falls back to a simulated dataset if pandas_datareader fails
    (some environments block the connection).
    """
    if start_date is None:
        start_date = (dt.date.today() - dt.timedelta(days=1100)).isoformat()

    try:
        import pandas_datareader.data as web

        ff5 = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3_daily",
            "famafrench",
            start=start_date,
        )
        # ff5 is a dict; the first key is the DataFrame
        df = ff5[0] if isinstance(ff5, dict) else ff5
        # Values are in percent — convert to decimal
        df = df / 100.0
        df.index = pd.to_datetime(df.index.astype(str))
        df.index.name = "date"
        logger.info("Downloaded Fama-French 5 factors: %d rows", len(df))
        return df
    except Exception as e:
        logger.warning("Failed to download FF factors: %s — using synthetic proxy", e)
        return _generate_synthetic_ff_factors(start_date)


def _generate_synthetic_ff_factors(start_date: str) -> pd.DataFrame:
    """
    Generate synthetic FF factor returns as a fallback.
    Uses realistic statistical properties for smoke testing.
    """
    dates = pd.bdate_range(start=start_date, end=dt.date.today())
    np.random.seed(42)
    n = len(dates)
    df = pd.DataFrame({
        "Mkt-RF": np.random.normal(0.0004, 0.01, n),
        "SMB": np.random.normal(0.0001, 0.005, n),
        "HML": np.random.normal(0.0001, 0.005, n),
        "RMW": np.random.normal(0.0001, 0.004, n),
        "CMA": np.random.normal(0.0001, 0.004, n),
        "RF": np.full(n, 0.0002),
    }, index=dates)
    df.index.name = "date"
    logger.info("Generated synthetic FF factors: %d rows", n)
    return df


def compute_factor_loadings(
    etf_returns: pd.Series,
    ff_factors: pd.DataFrame,
    window_months: int = 36,
) -> Dict[str, float]:
    """
    Compute rolling factor loadings for a single ETF using OLS regression.

    Uses the most recent `window_months` of data.

    Returns dict with keys: alpha, Mkt-RF, SMB, HML, RMW, CMA, r_squared
    """
    from statsmodels.api import OLS, add_constant

    # Align dates
    combined = pd.DataFrame({"etf_excess": etf_returns}).join(ff_factors, how="inner").dropna()

    # Trim to window
    window_days = window_months * 21  # approx trading days per month
    if len(combined) > window_days:
        combined = combined.iloc[-window_days:]

    if len(combined) < 60:  # minimum for meaningful regression
        return {f: 0.0 for f in ["alpha", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "r_squared"]}

    y = combined["etf_excess"]
    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if c in combined.columns]
    X = add_constant(combined[factor_cols])

    try:
        model = OLS(y, X).fit()
        result = {"alpha": model.params.get("const", 0.0), "r_squared": model.rsquared}
        for fc in factor_cols:
            result[fc] = model.params.get(fc, 0.0)
        return result
    except Exception as e:
        logger.error("OLS regression failed: %s", e)
        return {f: 0.0 for f in ["alpha", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "r_squared"]}


def compute_momentum(prices_wide: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """
    Compute 12-1 month momentum for each ticker.
    Returns cross-sectional rank (0 to 1) for the most recent date.
    """
    if prices_wide.empty or len(prices_wide) < lookback:
        return pd.Series(dtype=float)

    # 12-month return minus most recent 1-month return
    ret_12m = prices_wide.iloc[-1] / prices_wide.iloc[-lookback] - 1
    ret_1m = prices_wide.iloc[-1] / prices_wide.iloc[-skip] - 1
    momentum = ret_12m - ret_1m

    # Cross-sectional rank (0 = worst, 1 = best)
    ranked = momentum.rank(pct=True)
    ranked.name = "momentum_rank"
    return ranked


def compute_composite_factor_scores(
    conn: sqlite3.Connection,
    cfg: dict,
    ff_factors: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute composite factor scores for all sector ETFs.

    Steps:
      1. Load sector ETF prices, compute excess returns
      2. Run FF5 OLS for each ETF
      3. Apply McLean-Pontiff decay to alpha contributions
      4. Compute momentum overlay
      5. Combine into a composite score

    Returns DataFrame with columns: ticker, alpha, factor loadings, momentum, composite_score
    """
    from regime_detector import load_sector_prices

    sector_wide = load_sector_prices(conn, cfg)
    if sector_wide.empty:
        logger.error("No sector prices for factor scoring.")
        return pd.DataFrame()

    if ff_factors is None:
        start = sector_wide.index.min().strftime("%Y-%m-%d")
        ff_factors = download_ff_factors(start_date=start)

    # Compute daily log returns
    log_returns = np.log(sector_wide / sector_wide.shift(1)).dropna()

    # Subtract risk-free rate to get excess returns
    rf = ff_factors["RF"].reindex(log_returns.index).ffill().fillna(0)

    decay = cfg["factor_model"]["mclean_pontiff_decay"]  # 0.74
    window_months = cfg["factor_model"]["rolling_window_months"]  # 36

    # Momentum
    mom_ranks = compute_momentum(
        sector_wide,
        lookback=cfg["factor_model"]["momentum"]["lookback_months"] * 21,
        skip=cfg["factor_model"]["momentum"]["skip_months"] * 21,
    )

    records = []
    for ticker in sector_wide.columns:
        excess_ret = log_returns[ticker] - rf

        loadings = compute_factor_loadings(excess_ret, ff_factors, window_months)

        # Apply McLean-Pontiff decay to alpha
        raw_alpha = loadings.get("alpha", 0.0)
        adjusted_alpha = raw_alpha * decay

        # Composite score: weighted combination of adjusted alpha + factor tilts + momentum
        # Weight scheme: alpha (30%), value HML (15%), quality RMW (20%),
        #                investment CMA (10%), size SMB (5%), momentum (20%)
        mom = mom_ranks.get(ticker, 0.5)
        composite = (
            0.30 * _normalize_signal(adjusted_alpha, scale=0.001) +
            0.15 * _normalize_signal(loadings.get("HML", 0), scale=0.5) +
            0.20 * _normalize_signal(loadings.get("RMW", 0), scale=0.5) +
            0.10 * _normalize_signal(loadings.get("CMA", 0), scale=0.5) +
            0.05 * _normalize_signal(loadings.get("SMB", 0), scale=0.5) +
            0.20 * mom
        )

        records.append({
            "ticker": ticker,
            "raw_alpha": round(raw_alpha * 252, 6),  # annualized
            "adjusted_alpha": round(adjusted_alpha * 252, 6),
            "mkt_rf": round(loadings.get("Mkt-RF", 0), 4),
            "smb": round(loadings.get("SMB", 0), 4),
            "hml": round(loadings.get("HML", 0), 4),
            "rmw": round(loadings.get("RMW", 0), 4),
            "cma": round(loadings.get("CMA", 0), 4),
            "r_squared": round(loadings.get("r_squared", 0), 4),
            "momentum_rank": round(mom, 4),
            "composite_score": round(composite, 4),
        })

    df = pd.DataFrame(records).sort_values("composite_score", ascending=False)
    logger.info("Factor scores computed for %d sector ETFs", len(df))
    return df.reset_index(drop=True)


def _normalize_signal(value: float, scale: float = 1.0) -> float:
    """Map a raw signal to 0-1 range using a sigmoid-like transform."""
    return 1.0 / (1.0 + np.exp(-value / max(scale, 1e-10)))


# ===========================================================================
# 2. CVaR OPTIMIZATION WITH LEDOIT-WOLF SHRINKAGE
# ===========================================================================

def compute_shrunk_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute a Ledoit-Wolf shrunk covariance matrix.
    Never use raw sample covariance — always shrink.
    """
    from sklearn.covariance import LedoitWolf

    clean = returns.dropna()
    if clean.empty or clean.shape[0] < clean.shape[1] + 1:
        logger.warning("Insufficient data for covariance — returning identity.")
        n = returns.shape[1]
        return np.eye(n) * 0.0001

    lw = LedoitWolf().fit(clean.values)
    logger.info("Ledoit-Wolf shrinkage: %.4f", lw.shrinkage_)
    return lw.covariance_


def compute_tail_correlations(
    returns: pd.DataFrame,
    em_tickers: List[str],
    percentile: int = 10,
    min_joint_events: int = 5,
) -> Tuple[pd.DataFrame, dict]:
    """
    Longin-Solnik tail correlation adjustment for EM positions.

    Estimate pairwise correlations conditional on joint returns
    below the `percentile`-th percentile for each EM ETF pair.
    These tail correlations capture crisis-period co-movement
    that full-sample correlations understate.

    Parameters
    ----------
    returns : pd.DataFrame  Wide-format daily returns.
    em_tickers : list       EM/geographic ETF tickers to adjust.
    percentile : int        Joint tail threshold (default 10th).
    min_joint_events : int  Minimum joint tail observations per pair
                            (default 5).  Below this the pair falls
                            back to full-sample correlation.

    Returns
    -------
    Tuple of (correlation_matrix, diagnostics_dict).
    diagnostics_dict has keys: pairs_checked, pairs_adjusted,
    pairs_insufficient, pair_details (list of per-pair info).
    """
    full_corr = returns.corr()
    clean = returns.dropna()

    diag = {
        "pairs_checked": 0,
        "pairs_adjusted": 0,
        "pairs_insufficient": 0,
        "pair_details": [],
    }

    for i, t1 in enumerate(em_tickers):
        for t2 in em_tickers[i + 1:]:
            if t1 not in clean.columns or t2 not in clean.columns:
                continue

            diag["pairs_checked"] += 1
            r1 = clean[t1]
            r2 = clean[t2]

            # Joint threshold: both below their own percentile
            thresh1 = np.percentile(r1, percentile)
            thresh2 = np.percentile(r2, percentile)
            mask = (r1 <= thresh1) & (r2 <= thresh2)
            n_joint = int(mask.sum())

            full_sample_corr = full_corr.loc[t1, t2]
            pair_info = {
                "pair": f"{t1}/{t2}",
                "joint_tail_events": n_joint,
                "full_sample_corr": round(float(full_sample_corr), 4),
                "tail_corr": None,
                "adjusted": False,
            }

            if n_joint < min_joint_events:
                # Not enough tail observations — use full sample as floor
                diag["pairs_insufficient"] += 1
                logger.debug(
                    "Tail corr %s/%s: only %d joint events (need >=%d) — keeping full-sample %.4f",
                    t1, t2, n_joint, min_joint_events, full_sample_corr,
                )
                diag["pair_details"].append(pair_info)
                continue

            tail_corr_val = r1[mask].corr(r2[mask])
            pair_info["tail_corr"] = round(float(tail_corr_val), 4) if np.isfinite(tail_corr_val) else None

            if np.isfinite(tail_corr_val):
                # Tail correlations are typically higher — use them
                full_corr.loc[t1, t2] = tail_corr_val
                full_corr.loc[t2, t1] = tail_corr_val
                diag["pairs_adjusted"] += 1
                pair_info["adjusted"] = True
                logger.info(
                    "Tail correlation %s/%s: %.4f (full-sample: %.4f, %d joint events)",
                    t1, t2, tail_corr_val, full_sample_corr, n_joint,
                )

            diag["pair_details"].append(pair_info)

    logger.info(
        "Tail correlation summary: %d pairs checked, %d adjusted, %d insufficient data (<%d events)",
        diag["pairs_checked"], diag["pairs_adjusted"],
        diag["pairs_insufficient"], min_joint_events,
    )
    return full_corr, diag


def run_cvar_optimization(
    returns: pd.DataFrame,
    regime: str,
    cfg: dict,
    factor_scores: pd.DataFrame = None,
    em_tickers: List[str] = None,
) -> Dict[str, float]:
    """
    Run the CVaR portfolio optimization.

    Steps:
      1. Compute Ledoit-Wolf shrunk covariance
      2. Apply Longin-Solnik tail correlations for EM
      3. Map tickers to asset classes
      4. Get regime-conditional allocation bands
      5. Determine midpoints from factor signal strength
      6. Run PyPortfolioOpt min-CVaR with band constraints

    Returns dict of ticker -> weight.
    """
    from pypfopt import EfficientCVaR
    from pypfopt.expected_returns import mean_historical_return

    if returns.empty:
        return {}

    if em_tickers is None:
        em_tickers = cfg["tickers"]["geographic_etfs"]

    # --- Covariance ---
    cov_matrix = compute_shrunk_covariance(returns)
    cov_df = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)

    # Apply tail correlations for EM pairs
    tail_corr, tail_diag = compute_tail_correlations(
        returns, em_tickers,
        percentile=cfg["optimizer"]["tail_correlation_percentile"],
    )

    # Reconstruct covariance from tail correlations + shrunk variances
    stds = np.sqrt(np.diag(cov_matrix))
    std_df = pd.Series(stds, index=returns.columns)
    for i, t1 in enumerate(returns.columns):
        for j, t2 in enumerate(returns.columns):
            if t1 in em_tickers or t2 in em_tickers:
                if t1 in tail_corr.index and t2 in tail_corr.columns:
                    cov_df.loc[t1, t2] = tail_corr.loc[t1, t2] * std_df[t1] * std_df[t2]

    # --- Expected returns ---
    mu = mean_historical_return(returns, frequency=252)

    # --- Asset class mapping & bounds ---
    bounds = _compute_allocation_bounds(returns.columns.tolist(), regime, cfg, factor_scores)

    # --- Optimize ---
    try:
        ef = EfficientCVaR(
            mu,
            returns,
            beta=cfg["optimizer"]["cvar_confidence"],
        )

        # Apply per-asset bounds
        for ticker in returns.columns:
            lo, hi = bounds.get(ticker, (0.0, 1.0))
            ef.add_constraint(lambda w, t=ticker, lo_=lo: w[returns.columns.get_loc(t)] >= lo_)
            ef.add_constraint(lambda w, t=ticker, hi_=hi: w[returns.columns.get_loc(t)] <= hi_)

        ef.min_cvar()
        weights = ef.clean_weights(cutoff=0.001)
        logger.info("CVaR optimization succeeded: %d non-zero weights", sum(1 for v in weights.values() if v > 0))
        return dict(weights)

    except Exception as e:
        logger.warning("CVaR optimization failed: %s — falling back to band midpoints", e)
        return {t: (bounds[t][0] + bounds[t][1]) / 2 for t in bounds}


# ===========================================================================
# 3. REGIME-CONDITIONAL ALLOCATION BANDS
# ===========================================================================

# Asset class mapping: ticker -> asset_class key in config
ASSET_CLASS_MAP = {
    # US Equities (sector ETFs)
    "XLK": "us_equities", "XLV": "healthcare", "XLE": "energy_materials",
    "XLF": "us_equities", "XLI": "us_equities", "XLB": "energy_materials",
    "XLU": "us_equities", "XLP": "us_equities", "XLRE": "us_equities",
    "XLC": "us_equities", "XLY": "us_equities",
    # International Developed
    "VGK": "intl_developed",
    # Emerging Markets
    "EEM": "em_equities", "INDA": "em_equities", "EWZ": "em_equities",
    "FXI": "em_equities", "EWY": "em_equities", "EWT": "em_equities",
    # Benchmarks / Cash
    "BIL": "cash_short_duration", "AGG": "cash_short_duration",
    "TLT": "cash_short_duration",
    "GLD": "energy_materials",  # commodity bucket
    "SPY": "us_equities", "QQQ": "us_equities", "IWM": "us_equities",
    # Factor ETFs
    "MTUM": "us_equities", "VLUE": "us_equities", "USMV": "us_equities",
    "QUAL": "us_equities", "SIZE": "us_equities",
}


def _get_asset_class(ticker: str) -> str:
    """Map a ticker to its asset class for allocation band purposes."""
    return ASSET_CLASS_MAP.get(ticker, "us_equities")


def _compute_allocation_bounds(
    tickers: List[str],
    regime: str,
    cfg: dict,
    factor_scores: pd.DataFrame = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute per-ticker allocation bounds based on regime bands and factor scores.

    For each ticker, look up its asset class, get the regime band [lo, hi],
    then use factor scores to determine the midpoint within the band.
    """
    bands = cfg["optimizer"]["allocation_bands"]
    bounds = {}

    # Count tickers per asset class for splitting group budget
    class_counts = {}
    for t in tickers:
        ac = _get_asset_class(t)
        class_counts[ac] = class_counts.get(ac, 0) + 1

    for ticker in tickers:
        ac = _get_asset_class(ticker)
        band = bands.get(ac, {"panic": [0, 0.05], "defense": [0.05, 0.15], "offense": [0.1, 0.3]})
        regime_band = band.get(regime, band.get("offense", [0.05, 0.20]))

        group_lo = regime_band[0]
        group_hi = regime_band[1]

        # Split the group budget across tickers in the same class
        n_in_class = max(1, class_counts.get(ac, 1))
        per_ticker_lo = group_lo / n_in_class
        per_ticker_hi = group_hi / n_in_class

        # Adjust midpoint based on factor score if available
        if factor_scores is not None and ticker in factor_scores["ticker"].values:
            score = factor_scores.loc[factor_scores["ticker"] == ticker, "composite_score"].values[0]
            # Score of 0.5 = midpoint, >0.5 = toward upper, <0.5 = toward lower
            per_ticker_hi = per_ticker_lo + (per_ticker_hi - per_ticker_lo) * min(1.0, score + 0.3)

        bounds[ticker] = (round(per_ticker_lo, 6), round(max(per_ticker_lo, per_ticker_hi), 6))

    return bounds


# ===========================================================================
# 4. US EQUITIES SUB-SECTOR ALLOCATION
# ===========================================================================

def compute_bivector_beta(
    returns: pd.DataFrame,
    sector_ticker: str,
    market_tickers: List[str] = None,
    market_proxy: str = "SPY",
    min_obs: int = 30,
) -> float:
    """
    Compute Bivector Beta for a sector — the degree of orthogonality
    to the first principal component of the market factor.

    Bivector Beta > 1.0 means the sector is a structural diversifier
    (its returns are significantly different from the dominant market mode).

    Parameters
    ----------
    returns : pd.DataFrame  Wide-format returns (date × ticker).
    sector_ticker : str     The sector to evaluate.
    market_tickers : list   Tickers for PCA (default: all other columns).
    market_proxy : str      If present in returns, include as a market
                            anchor in PCA (default: "SPY").  The report
                            specifies PC1 of "the market" — SPY ensures
                            PC1 aligns with the broad equity factor.
    min_obs : int           Minimum observations required for PCA.

    Returns
    -------
    float  Bivector Beta value.
    """
    if sector_ticker not in returns.columns:
        return 1.0

    if market_tickers is None:
        market_tickers = [c for c in returns.columns if c != sector_ticker]

    # If SPY (or configured market_proxy) is in the returns but not in
    # market_tickers, add it so PC1 aligns with the broad market factor.
    if market_proxy and market_proxy in returns.columns and market_proxy != sector_ticker:
        if market_proxy not in market_tickers:
            market_tickers = market_tickers + [market_proxy]
            logger.debug("Bivector Beta: added %s as market anchor for PCA", market_proxy)

    # Use pairwise-available data: first select columns, then drop rows
    # where the sector or any market ticker is NaN.  If mixed-length
    # histories cause too many NaN rows, progressively drop the shortest
    # market tickers until we have enough observations.
    candidate_cols = [c for c in market_tickers if c in returns.columns]
    subset = returns[[sector_ticker] + candidate_cols]

    # Sort market tickers by available rows descending, keep those with
    # >= min_obs rows in common with the sector ticker.
    sector_notna = returns[sector_ticker].notna()
    available = {}
    for c in candidate_cols:
        overlap = (sector_notna & returns[c].notna()).sum()
        if overlap >= min_obs:
            available[c] = overlap
    candidate_cols = sorted(available, key=available.get, reverse=True)

    if len(candidate_cols) < 2:
        logger.info("Bivector Beta for %s: insufficient overlapping data, returning 1.0", sector_ticker)
        return 1.0

    clean = returns[[sector_ticker] + candidate_cols].dropna()
    if clean.shape[0] < min_obs:
        logger.info("Bivector Beta for %s: only %d rows after dropna, returning 1.0", sector_ticker, clean.shape[0])
        return 1.0

    from sklearn.decomposition import PCA

    # PCA on market returns (excluding the target sector)
    market_data = clean[candidate_cols].values
    pca = PCA(n_components=1)
    pca.fit(market_data)
    pc1 = pca.transform(market_data).flatten()

    # Correlation of sector with PC1
    sector_rets = clean[sector_ticker].values
    corr = np.corrcoef(sector_rets, pc1)[0, 1]

    # Bivector Beta: inverse of |correlation| — higher = more orthogonal
    bivector_beta = 1.0 / max(abs(corr), 0.01)
    logger.info("Bivector Beta %s: corr=%.4f, β_bv=%.4f (%d obs, %d market tickers)",
                sector_ticker, corr, bivector_beta, clean.shape[0], len(candidate_cols))
    return round(bivector_beta, 4)


def apply_us_subsector_allocation(
    sector_etfs: List[str],
    factor_scores: pd.DataFrame,
    us_equity_weight: float,
    cfg: dict,
    returns: pd.DataFrame = None,
) -> Dict[str, dict]:
    """
    Allocate within US Equities using DeMiguel equal-weight
    with valuation filter and Bivector Beta adjustments.

    Returns dict of ticker -> {weight, label, bivector_beta, ...}
    """
    us_sectors = [t for t in sector_etfs if _get_asset_class(t) == "us_equities"]
    n = len(us_sectors)
    if n == 0:
        return {}

    equal_weight = us_equity_weight / n
    val_cfg = cfg["factor_model"]["valuation"]
    mom_cap = val_cfg["momentum_cap_fraction"]  # 0.50

    result = {}
    energy_materials = ["XLE", "XLB"]

    for ticker in us_sectors:
        row = factor_scores[factor_scores["ticker"] == ticker]
        label = "FUNDAMENTAL_BUY"
        weight = equal_weight

        if not row.empty:
            score = row["composite_score"].values[0]
            mom = row["momentum_rank"].values[0]

            # Valuation filter proxy: if momentum rank is very high but
            # alpha is negative, it's priced for perfection
            alpha = row["adjusted_alpha"].values[0]
            if mom > 0.75 and alpha < 0:
                label = "MOMENTUM_ONLY"
                weight = equal_weight * mom_cap
            elif mom < 0.10 and alpha < -0.05:
                label = "AVOID"
                weight = 0.0
        else:
            score = 0.5

        # Bivector Beta for Energy/Materials
        # Use only tickers that are in the sector_etfs list (long history)
        # to avoid mixed-length NaN drops from geographic/benchmark tickers.
        bv_beta = 1.0
        if ticker in energy_materials and returns is not None:
            sector_only_cols = [c for c in sector_etfs if c in returns.columns]
            if sector_only_cols:
                bv_beta = compute_bivector_beta(
                    returns[sector_only_cols], ticker,
                    market_tickers=[c for c in sector_only_cols if c != ticker],
                )

        result[ticker] = {
            "weight": round(weight, 6),
            "label": label,
            "equal_weight_base": round(equal_weight, 6),
            "bivector_beta": bv_beta,
            "is_structural_diversifier": bv_beta > 1.0 and ticker in energy_materials,
        }

    # Normalize so weights sum to us_equity_weight
    total = sum(v["weight"] for v in result.values())
    if total > 0:
        scale = us_equity_weight / total
        for t in result:
            result[t]["weight"] = round(result[t]["weight"] * scale, 6)

    return result


# ===========================================================================
# 5. DOLLAR ALLOCATION ENGINE
# ===========================================================================

def allocate_dollars(
    weights: Dict[str, float],
    cfg: dict,
    subsector_info: Dict[str, dict] = None,
) -> Dict[str, dict]:
    """
    Convert percentage weights to dollar amounts split across
    Taxable Brokerage and Roth IRA per tax-location rules.

    Rules (from config):
      Roth IRA first: individual thematic stocks, high-turnover,
                      biotech/smallcap, momentum-tilt-only
      Taxable: broad ETFs, geographic, cash, energy/materials, TLH candidates
      Never split a single position across both accounts unless > $10K

    Returns dict of ticker -> {
        pct, total_dollars, taxable_dollars, roth_dollars, account, reason
    }
    """
    total_portfolio = cfg["portfolio"]["total_value"]  # 144000
    taxable_cap = cfg["portfolio"]["accounts"]["taxable"]["value"]  # 100000
    roth_cap = cfg["portfolio"]["accounts"]["roth_ira"]["value"]  # 44000
    split_thresh = cfg["tax_location"]["split_threshold"]  # 10000

    # Classify each position
    roth_tickers = set()
    taxable_tickers = set()

    # Watchlist tickers → Roth
    for key in ["watchlist_biotech", "watchlist_ai_software",
                "watchlist_defense", "watchlist_green_materials"]:
        for t in cfg["tickers"].get(key, []):
            roth_tickers.add(t)

    # Geographic ETFs → Taxable (foreign tax credit)
    for t in cfg["tickers"].get("geographic_etfs", []):
        taxable_tickers.add(t)

    # Cash/bonds → Taxable
    for t in ["BIL", "AGG", "TLT"]:
        taxable_tickers.add(t)

    # Energy/Materials ETFs → Taxable (structural diversifier, low turnover)
    for t in ["XLE", "XLB", "GLD"]:
        taxable_tickers.add(t)

    result = {}
    roth_used = 0.0
    taxable_used = 0.0

    # Sort by dollar amount descending for better bin-packing
    sorted_positions = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    for ticker, weight in sorted_positions:
        if weight <= 0:
            continue

        dollars = round(weight * total_portfolio, 2)
        reason = ""

        # Check subsector info for momentum-only labels
        is_momentum_only = False
        if subsector_info and ticker in subsector_info:
            if subsector_info[ticker].get("label") == "MOMENTUM_ONLY":
                is_momentum_only = True
                roth_tickers.add(ticker)

        # Determine account placement
        if ticker in roth_tickers or is_momentum_only:
            # Roth first
            if roth_used + dollars <= roth_cap:
                account = "roth_ira"
                roth_dollars = dollars
                taxable_dollars = 0.0
                reason = "Roth: thematic/high-turnover/momentum position"
                roth_used += dollars
            elif dollars > split_thresh and roth_used < roth_cap and taxable_used < taxable_cap:
                # Split across both accounts
                roth_avail = round(roth_cap - roth_used, 2)
                taxable_avail = round(taxable_cap - taxable_used, 2)
                roth_dollars = min(roth_avail, dollars)
                taxable_dollars = min(taxable_avail, round(dollars - roth_dollars, 2))
                # If neither account can absorb the full amount, cap it
                if roth_dollars + taxable_dollars < dollars:
                    # Trim to fit within combined capacity
                    roth_dollars = min(roth_avail, dollars)
                    taxable_dollars = min(taxable_avail, round(dollars - roth_dollars, 2))
                account = "split"
                reason = f"Split: ${roth_dollars:.0f} Roth + ${taxable_dollars:.0f} Taxable (exceeds ${split_thresh:,.0f})"
                roth_used += roth_dollars
                taxable_used += taxable_dollars
            elif taxable_used + dollars <= taxable_cap:
                # Roth full — overflow to taxable
                account = "taxable"
                taxable_dollars = dollars
                roth_dollars = 0.0
                reason = "Taxable: Roth capacity full"
                taxable_used += dollars
            else:
                # Both near capacity — place in whichever has room
                roth_avail = roth_cap - roth_used
                taxable_avail = taxable_cap - taxable_used
                if roth_avail >= taxable_avail:
                    roth_dollars = min(dollars, roth_avail)
                    taxable_dollars = min(round(dollars - roth_dollars, 2), taxable_avail)
                else:
                    taxable_dollars = min(dollars, taxable_avail)
                    roth_dollars = min(round(dollars - taxable_dollars, 2), roth_avail)
                if roth_dollars > 0 and taxable_dollars > 0:
                    account = "split"
                    reason = f"Split: capacity constrained (${roth_dollars:.0f} Roth + ${taxable_dollars:.0f} Taxable)"
                elif roth_dollars > 0:
                    account = "roth_ira"
                    reason = "Roth: last available capacity"
                else:
                    account = "taxable"
                    reason = "Taxable: last available capacity"
                roth_used += roth_dollars
                taxable_used += taxable_dollars

        elif ticker in taxable_tickers:
            # Taxable preferred
            if taxable_used + dollars <= taxable_cap:
                account = "taxable"
                taxable_dollars = dollars
                roth_dollars = 0.0
                reason = "Taxable: geographic/cash/energy (foreign tax credit / low turnover)"
                taxable_used += dollars
            elif roth_used + dollars <= roth_cap:
                account = "roth_ira"
                roth_dollars = dollars
                taxable_dollars = 0.0
                reason = "Roth: taxable capacity full"
                roth_used += dollars
            else:
                # Both near capacity — split to fit
                taxable_avail = taxable_cap - taxable_used
                roth_avail = roth_cap - roth_used
                taxable_dollars = min(dollars, taxable_avail)
                roth_dollars = min(round(dollars - taxable_dollars, 2), roth_avail)
                if taxable_dollars > 0 and roth_dollars > 0:
                    account = "split"
                    reason = f"Split: capacity constrained (${taxable_dollars:.0f} Taxable + ${roth_dollars:.0f} Roth)"
                elif taxable_dollars > 0:
                    account = "taxable"
                    reason = "Taxable: last available capacity"
                else:
                    account = "roth_ira"
                    reason = "Roth: last available capacity"
                taxable_used += taxable_dollars
                roth_used += roth_dollars

        else:
            # Default: broad ETF → taxable (long-term hold)
            if taxable_used + dollars <= taxable_cap:
                account = "taxable"
                taxable_dollars = dollars
                roth_dollars = 0.0
                reason = "Taxable: broad ETF, long-term hold"
                taxable_used += dollars
            elif roth_used + dollars <= roth_cap:
                account = "roth_ira"
                roth_dollars = dollars
                taxable_dollars = 0.0
                reason = "Roth: taxable overflow"
                roth_used += dollars
            else:
                # Both near capacity — split to fit
                taxable_avail = taxable_cap - taxable_used
                roth_avail = roth_cap - roth_used
                taxable_dollars = min(dollars, taxable_avail)
                roth_dollars = min(round(dollars - taxable_dollars, 2), roth_avail)
                if taxable_dollars > 0 and roth_dollars > 0:
                    account = "split"
                    reason = f"Split: capacity constrained (${taxable_dollars:.0f} Taxable + ${roth_dollars:.0f} Roth)"
                elif taxable_dollars > 0:
                    account = "taxable"
                    reason = "Taxable: last available capacity"
                else:
                    account = "roth_ira"
                    reason = "Roth: last available capacity"
                taxable_used += taxable_dollars
                roth_used += roth_dollars

        result[ticker] = {
            "pct": round(weight * 100, 2),
            "total_dollars": dollars,
            "taxable_dollars": taxable_dollars,
            "roth_dollars": roth_dollars,
            "account": account,
            "reason": reason,
        }

    logger.info(
        "Dollar allocation: $%.0f taxable ($%.0f cap), $%.0f Roth ($%.0f cap)",
        taxable_used, taxable_cap, roth_used, roth_cap,
    )
    return result


# ===========================================================================
# 6. MASTER ALLOCATION PIPELINE
# ===========================================================================

def run_portfolio_optimization(
    conn: sqlite3.Connection = None,
    cfg: dict = None,
    regime: str = None,
) -> dict:
    """
    Master function: run the full portfolio optimization pipeline.

    1. Load current regime from DB (or accept override)
    2. Compute factor scores for all sector ETFs
    3. Load prices for optimization universe
    4. Run CVaR optimization with regime bands
    5. Apply US sub-sector allocation
    6. Convert to dollar amounts across accounts
    7. Output JSON + CSV

    Returns the complete allocation dict.
    """
    if cfg is None:
        cfg = load_config()

    close_conn = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        close_conn = True

    # --- Step 1: Get current regime ---
    if regime is None:
        from regime_detector import get_latest_regime_state
        state = get_latest_regime_state(conn, cfg)
        regime = state.get("dominant_regime", "offense")
    logger.info("Optimizing for regime: %s", regime)

    # --- Step 2: Factor scores ---
    logger.info("Computing factor scores...")
    ff_factors = download_ff_factors()
    factor_scores = compute_composite_factor_scores(conn, cfg, ff_factors)
    if factor_scores.empty:
        logger.error("Factor scoring failed — cannot optimize.")
        if close_conn:
            conn.close()
        return {}

    logger.info("Top factor scores:\n%s", factor_scores.head(5).to_string())

    # --- Step 3: Load price returns for optimization ---
    sector_tickers = cfg["tickers"]["sector_etfs"]
    geo_tickers = cfg["tickers"]["geographic_etfs"]
    cash_tickers = ["BIL"]
    all_opt_tickers = sector_tickers + geo_tickers + cash_tickers

    placeholders = ",".join(["?"] * len(all_opt_tickers))
    prices = pd.read_sql_query(
        f"SELECT date, ticker, adj_close FROM prices WHERE ticker IN ({placeholders}) ORDER BY date",
        conn, params=all_opt_tickers,
    )

    if prices.empty:
        logger.error("No prices for optimization universe.")
        if close_conn:
            conn.close()
        return {}

    wide = prices.pivot(index="date", columns="ticker", values="adj_close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().ffill().dropna(axis=1, how="all")

    returns = np.log(wide / wide.shift(1)).dropna()

    # --- Step 4: CVaR optimization ---
    logger.info("Running CVaR optimization...")
    weights = run_cvar_optimization(
        returns, regime, cfg,
        factor_scores=factor_scores,
        em_tickers=geo_tickers,
    )

    if not weights:
        logger.error("Optimization returned empty weights.")
        if close_conn:
            conn.close()
        return {}

    # --- Step 5: US sub-sector allocation ---
    us_weight = sum(w for t, w in weights.items() if _get_asset_class(t) == "us_equities")
    subsector = apply_us_subsector_allocation(
        sector_tickers, factor_scores, us_weight, cfg, returns,
    )

    # Update weights with sub-sector detail
    for ticker, info in subsector.items():
        if ticker in weights:
            weights[ticker] = info["weight"]

    # Normalize weights to sum to 1.0
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {t: w / total_w for t, w in weights.items()}
    else:
        logger.error("All weights are zero after sub-sector allocation — cannot allocate dollars.")
        if close_conn:
            conn.close()
        return {}

    # --- Step 6: Dollar allocation ---
    logger.info("Computing dollar allocation...")
    dollar_alloc = allocate_dollars(weights, cfg, subsector)

    # --- Build output ---
    allocation = {
        "date": dt.date.today().isoformat(),
        "regime": regime,
        "total_portfolio": cfg["portfolio"]["total_value"],
        "taxable_account": cfg["portfolio"]["accounts"]["taxable"]["value"],
        "roth_ira_account": cfg["portfolio"]["accounts"]["roth_ira"]["value"],
        "positions": dollar_alloc,
        "factor_scores": factor_scores.to_dict(orient="records"),
        "subsector_detail": {t: v for t, v in subsector.items()},
    }

    # Save JSON
    json_path = Path(__file__).parent / "current_allocation.json"
    with open(json_path, "w") as f:
        json.dump(allocation, f, indent=2, default=str)
    logger.info("Allocation JSON saved to %s", json_path)

    # Save CSV
    csv_path = Path(__file__).parent / "current_allocation.csv"
    rows = []
    for ticker, info in dollar_alloc.items():
        rows.append({
            "ticker": ticker,
            "weight_pct": info["pct"],
            "total_dollars": info["total_dollars"],
            "taxable_dollars": info["taxable_dollars"],
            "roth_dollars": info["roth_dollars"],
            "account": info["account"],
            "reason": info["reason"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Allocation CSV saved to %s", csv_path)

    # Store in DB
    now = dt.datetime.utcnow().isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO allocations (date, regime, allocation_json, dollar_taxable, dollar_roth, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            dt.date.today().isoformat(),
            regime,
            json.dumps(dollar_alloc, default=str),
            json.dumps({t: v["taxable_dollars"] for t, v in dollar_alloc.items()}),
            json.dumps({t: v["roth_dollars"] for t, v in dollar_alloc.items()}),
            now,
        ),
    )
    conn.commit()
    logger.info("Allocation stored in database.")

    if close_conn:
        conn.close()

    return allocation


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: Factor Scoring & CVaR Portfolio Optimizer")
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic data if DB is empty or missing (for initial setup/CI)")
    parser.add_argument("--regime", choices=["offense", "defense", "panic"],
                        default=None, help="Override regime (default: read from DB)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.mock:
        logger.info("--mock mode: generating synthetic price data")
        # Generate synthetic DB if empty
        conn = sqlite3.connect(str(DB_PATH))
        row_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        if row_count == 0:
            logger.info("DB empty — seeding synthetic sector prices for smoke testing")
            cfg = load_config()
            dates = pd.bdate_range(end=dt.date.today(), periods=600)
            np.random.seed(42)
            mock_rows = []
            all_mock_tickers = (cfg["tickers"]["sector_etfs"] +
                                cfg["tickers"]["geographic_etfs"] + ["BIL", "SPY"])
            for ticker in all_mock_tickers:
                base = np.random.uniform(30, 200)
                prices = base * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, len(dates))))
                for d, p in zip(dates, prices):
                    mock_rows.append((d.strftime("%Y-%m-%d"), ticker, p, p, p * 0.99, p, p, int(np.random.uniform(1e6, 5e6))))
            conn.executemany(
                "INSERT OR IGNORE INTO prices (date, ticker, open, high, low, close, adj_close, volume) VALUES (?,?,?,?,?,?,?,?)",
                mock_rows,
            )
            conn.commit()
            logger.info("Seeded %d synthetic price rows", len(mock_rows))
        conn.close()

    result = run_portfolio_optimization(regime=args.regime)
    if result:
        print("\n" + "=" * 70)
        print("PORTFOLIO ALLOCATION — %s REGIME" % result.get("regime", "?").upper())
        print("=" * 70)

        positions = result.get("positions", {})
        print(f"\n{'Ticker':<8} {'Weight':>7} {'Total $':>10} {'Taxable $':>10} {'Roth $':>10} {'Account':<10} Reason")
        print("-" * 90)

        total_t = 0
        total_r = 0
        for ticker, info in sorted(positions.items(), key=lambda x: x[1]["total_dollars"], reverse=True):
            print(
                f"{ticker:<8} {info['pct']:>6.1f}% ${info['total_dollars']:>9,.0f} "
                f"${info['taxable_dollars']:>9,.0f} ${info['roth_dollars']:>9,.0f} "
                f"{info['account']:<10} {info['reason']}"
            )
            total_t += info["taxable_dollars"]
            total_r += info["roth_dollars"]

        print("-" * 90)
        print(f"{'TOTAL':<8} {'100.0':>6}% ${total_t + total_r:>9,.0f} ${total_t:>9,.0f} ${total_r:>9,.0f}")
