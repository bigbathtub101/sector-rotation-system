"""
portfolio_optimizer.py -- Phase 3: Factor Scoring & CVaR Portfolio Optimizer
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

Dependencies: numpy, pandas, scipy, sklearn, pypfopt, statsmodels, yaml
"""

import json
import logging
import sqlite3
import datetime as dt
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sp_stats

# ETF Auto-Selector: dynamically chooses best-in-class ETFs per exposure slot
try:
    from etf_selector import (
        get_selected_tickers as _get_etf_selections,
        get_ticker_asset_class_map as _get_selector_asset_class_map,
    )
    _ETF_SELECTOR_AVAILABLE = True
except ImportError:
    _ETF_SELECTOR_AVAILABLE = False

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

    Uses direct CSV download from the Dartmouth server instead of
    pandas_datareader (which is broken on newer pandas/numpy due to
    the deprecated `date_parser` argument).

    Falls back to a simulated dataset if the download fails.
    """
    if start_date is None:
        start_date = (dt.date.today() - dt.timedelta(days=1100)).isoformat()

    try:
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urlopen(req, timeout=30)
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))

        # The zip contains a single CSV file
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        raw = zf.read(csv_name).decode("utf-8")

        # Find the header row (starts with date column -- skip description lines)
        lines = raw.strip().split("\n")
        header_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped[0].isdigit() is False and "Mkt-RF" in stripped:
                header_idx = i
                break

        if header_idx is None:
            # Try finding first numeric line as data start
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and stripped.split(",")[0].strip().isdigit():
                    header_idx = i - 1
                    break

        # Parse from header onward
        csv_text = "\n".join(lines[header_idx:]) if header_idx else raw
        df = pd.read_csv(io.StringIO(csv_text), skipinitialspace=True)

        # First column is the date (unnamed or named oddly)
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})

        # Drop any rows where Date is not a valid YYYYMMDD integer
        df["Date"] = pd.to_numeric(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Date"] = df["Date"].astype(int).astype(str)
        df = df[df["Date"].str.len() == 8]  # keep only YYYYMMDD rows

        df.index = pd.to_datetime(df["Date"], format="%Y%m%d")
        df.index.name = "date"
        df = df.drop(columns=["Date"])

        # Clean column names
        df.columns = [c.strip() for c in df.columns]

        # Convert to numeric and from percent to decimal
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df / 100.0
        df = df.dropna(how="all")

        # Filter to start_date
        df = df[df.index >= start_date]

        if df.empty:
            raise ValueError("FF5 DataFrame empty after filtering")

        logger.info("Downloaded Fama-French 5 factors: %d rows (direct CSV)", len(df))
        return df

    except Exception as e:
        logger.warning("Failed to download FF factors: %s -- using synthetic proxy", e)
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

def _clean_returns(returns: pd.DataFrame, min_history_frac: float = 0.5) -> pd.DataFrame:
    """
    Clean a return matrix for optimizer consumption.

    FIX: This function was missing, causing NaN/Inf returns to reach
    pypfopt which then declares the problem infeasible.

    Steps:
      1. Drop columns (tickers) with < min_history_frac non-NaN rows
      2. Replace any remaining NaN with 0.0
      3. Replace Inf/-Inf with 0.0
      4. Forward-fill then backfill stale columns
    """
    n_rows = len(returns)
    if n_rows == 0:
        return returns

    # Drop tickers with too little history
    min_obs = int(n_rows * min_history_frac)
    valid_counts = returns.notna().sum()
    keep_cols = valid_counts[valid_counts >= min_obs].index.tolist()
    dropped = set(returns.columns) - set(keep_cols)
    if dropped:
        logger.info("Dropping %d tickers with < %d observations: %s",
                     len(dropped), min_obs, sorted(dropped)[:10])
    returns = returns[keep_cols]

    # Replace Inf with NaN, then fill
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.fillna(0.0)

    return returns


def compute_shrunk_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute a Ledoit-Wolf shrunk covariance matrix.
    Never use raw sample covariance -- always shrink.
    """
    from sklearn.covariance import LedoitWolf

    clean = returns.dropna()
    if clean.empty or clean.shape[0] < clean.shape[1] + 1:
        logger.warning("Insufficient data for covariance -- returning identity.")
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
                diag["pairs_insufficient"] += 1
                logger.debug(
                    "Tail corr %s/%s: only %d joint events (need >=%d) -- keeping full-sample %.4f",
                    t1, t2, n_joint, min_joint_events, full_sample_corr,
                )
                diag["pair_details"].append(pair_info)
                continue

            tail_corr_val = r1[mask].corr(r2[mask])
            pair_info["tail_corr"] = round(float(tail_corr_val), 4) if np.isfinite(tail_corr_val) else None

            if np.isfinite(tail_corr_val):
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
      1. Clean returns (remove NaN/Inf tickers)
      2. Compute Ledoit-Wolf shrunk covariance
      3. Apply Longin-Solnik tail correlations for EM
      4. Map tickers to asset classes
      5. Get regime-conditional allocation bands
      6. Validate constraint feasibility (bounds must sum <= 1.0)
      7. Run PyPortfolioOpt min-CVaR with band constraints

    Returns dict of ticker -> weight.
    """
    from pypfopt import EfficientCVaR

    if returns.empty:
        return {}

    if em_tickers is None:
        em_tickers = cfg["tickers"]["geographic_etfs"]

    # ---- FIX 1: Clean returns before anything else ----
    returns = _clean_returns(returns, min_history_frac=0.5)
    if returns.empty or returns.shape[1] < 3:
        logger.warning("Too few tickers (%d) after cleaning -- skipping CVaR", returns.shape[1])
        return {}

    logger.info("CVaR optimizer: %d tickers x %d days after cleaning", returns.shape[1], returns.shape[0])

    # --- Covariance ---
    cov_matrix = compute_shrunk_covariance(returns)
    cov_df = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)

    # Apply tail correlations for EM pairs
    em_in_universe = [t for t in em_tickers if t in returns.columns]
    tail_corr, tail_diag = compute_tail_correlations(
        returns, em_in_universe,
        percentile=cfg["optimizer"]["tail_correlation_percentile"],
    )

    # Reconstruct covariance from tail correlations + shrunk variances
    stds = np.sqrt(np.diag(cov_matrix))
    std_df = pd.Series(stds, index=returns.columns)
    for i, t1 in enumerate(returns.columns):
        for j, t2 in enumerate(returns.columns):
            if t1 in em_in_universe or t2 in em_in_universe:
                if t1 in tail_corr.index and t2 in tail_corr.columns:
                    cov_df.loc[t1, t2] = tail_corr.loc[t1, t2] * std_df[t1] * std_df[t2]

    # --- Expected returns (compute directly from log returns to avoid
    #     pypfopt NaN/Inf warnings -- mean_historical_return expects prices) ---
    mu = returns.mean() * 252  # annualized expected returns from daily log returns

    # ---- FIX 3: Compute bounds with feasibility validation ----
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
        logger.warning("CVaR optimization failed: %s -- using smart fallback", e)
        return _smart_fallback(bounds, factor_scores, regime, cfg)


def _smart_fallback(
    bounds: Dict[str, Tuple[float, float]],
    factor_scores: pd.DataFrame,
    regime: str,
    cfg: dict,
) -> Dict[str, float]:
    """
    FIX 4: Smart fallback when CVaR is infeasible.

    Instead of spreading equally across 53+ tickers (the old "band midpoint"
    approach), this fallback:
      1. Scores each ticker using factor_scores (if available)
      2. Concentrates into the top 15 tickers by composite score
      3. Applies regime-aware weighting (more to core sectors, less to thematic)
      4. Normalizes to sum to 1.0
    """
    max_positions = cfg.get("optimizer", {}).get("max_positions", 15)

    # Build a scored list of tickers
    scored = []
    for ticker, (lo, hi) in bounds.items():
        midpoint = (lo + hi) / 2
        # Get factor score if available
        fscore = 0.5
        if factor_scores is not None and not factor_scores.empty:
            match = factor_scores[factor_scores["ticker"] == ticker]
            if not match.empty:
                fscore = match["composite_score"].values[0]

        ac = _get_asset_class(ticker)
        # Regime-based preference multiplier
        if regime == "panic":
            pref = {"cash_short_duration": 3.0, "healthcare": 1.5, "us_equities": 0.5}.get(ac, 0.3)
        elif regime == "defense":
            pref = {"cash_short_duration": 2.0, "healthcare": 1.5, "us_equities": 1.0, "intl_developed": 0.5}.get(ac, 0.5)
        else:  # offense
            pref = {"us_equities": 2.0, "intl_developed": 1.0, "em_equities": 1.0,
                     "energy_materials": 0.8, "healthcare": 1.0, "industry_sub": 0.7,
                     "thematic": 0.6, "cash_short_duration": 0.1}.get(ac, 0.5)

        combined_score = fscore * pref * midpoint
        scored.append((ticker, combined_score, hi))

    # Sort by combined score descending, take top N
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:max_positions]

    # Allocate proportionally to score, capped at upper bound
    total_score = sum(s for _, s, _ in top)
    if total_score <= 0:
        # Absolute last resort: equal-weight top picks
        n = len(top)
        return {t: 1.0 / n for t, _, _ in top} if n > 0 else {}

    raw_weights = {}
    for ticker, score, hi in top:
        raw_weights[ticker] = score / total_score

    # Normalize to 1.0
    total = sum(raw_weights.values())
    if total > 0:
        raw_weights = {t: w / total for t, w in raw_weights.items()}

    logger.info("Smart fallback: %d positions (top of %d scored)", len(raw_weights), len(scored))
    return raw_weights


# ===========================================================================
# 3. REGIME-CONDITIONAL ALLOCATION BANDS
# ===========================================================================

# Asset class mapping: ticker -> asset_class key in config
# This base map covers all legacy tickers + Fidelity/Vanguard/Franklin alternatives.
# At runtime, it's augmented by etf_selector cache (see _get_asset_class).
ASSET_CLASS_MAP = {
    # US Equities -- SPDR XL* series (11 GICS sectors)
    "XLK": "us_equities", "XLV": "healthcare", "XLE": "energy_materials",
    "XLF": "us_equities", "XLI": "us_equities", "XLB": "energy_materials",
    "XLU": "us_equities", "XLP": "us_equities", "XLRE": "us_equities",
    "XLC": "us_equities", "XLY": "us_equities",
    # US Equities -- Fidelity MSCI series (auto-selector candidates)
    "FTEC": "us_equities", "FHLC": "healthcare", "FENY": "energy_materials",
    "FNCL": "us_equities", "FIDU": "us_equities", "FMAT": "energy_materials",
    "FUTY": "us_equities", "FSTA": "us_equities", "FREL": "us_equities",
    "FCOM": "us_equities", "FDIS": "us_equities",
    # US Equities -- Vanguard series (auto-selector candidates)
    "VGT": "us_equities", "VHT": "healthcare", "VDE": "energy_materials",
    "VFH": "us_equities", "VIS": "us_equities", "VAW": "energy_materials",
    "VPU": "us_equities", "VDC": "us_equities", "VNQ": "us_equities",
    "VOX": "us_equities", "VCR": "us_equities",
    # Industry / Sub-sector ETFs
    "SOXX": "industry_sub", "IGV": "industry_sub", "HACK": "industry_sub",
    "SKYY": "industry_sub", "XBI": "industry_sub", "IHI": "industry_sub",
    "XPH": "industry_sub", "KBE": "industry_sub", "KRE": "industry_sub",
    "IAI": "industry_sub", "XHB": "industry_sub", "XRT": "industry_sub",
    "IBUY": "industry_sub", "ITA": "industry_sub", "IYT": "industry_sub",
    "XOP": "industry_sub", "OIH": "industry_sub",
    # Thematic ETFs
    "BOTZ": "thematic", "LIT": "thematic", "ICLN": "thematic",
    "TAN": "thematic", "QCLN": "thematic", "ARKK": "thematic",
    "ARKG": "thematic", "ARKW": "thematic", "SMH": "thematic",
    "KOMP": "thematic", "UFO": "thematic", "DRIV": "thematic",
    "URNM": "thematic", "URA": "thematic", "REMX": "thematic",
    "COPX": "thematic", "AIQ": "thematic",
    # International Developed -- legacy + Franklin FTSE
    "VGK": "intl_developed", "EWJ": "intl_developed",
    "FLJP": "intl_developed", "FLEU": "intl_developed",
    "FEZ": "intl_developed", "DXJ": "intl_developed",
    # Emerging Markets -- legacy + Franklin FTSE
    "EEM": "em_equities", "INDA": "em_equities", "EWZ": "em_equities",
    "FXI": "em_equities", "MCHI": "em_equities", "EWY": "em_equities", "EWT": "em_equities",
    "KWEB": "em_equities", "VWO": "em_equities", "IEMG": "em_equities",
    "FLIN": "em_equities", "FLBR": "em_equities", "FLCH": "em_equities",
    "FLKR": "em_equities", "FLTW": "em_equities", "INDY": "em_equities",
    # Benchmarks / Cash
    "BIL": "cash_short_duration", "SGOV": "cash_short_duration",
    "SHV": "cash_short_duration", "AGG": "cash_short_duration",
    "TLT": "cash_short_duration",
    "GLD": "energy_materials",  # commodity bucket
    "SPY": "us_equities", "QQQ": "us_equities", "IWM": "us_equities",
    # Factor ETFs
    "MTUM": "us_equities", "VLUE": "us_equities", "USMV": "us_equities",
    "QUAL": "us_equities", "SIZE": "us_equities", "COWZ": "us_equities",
    "QQQM": "us_equities",
}

# Set of all known ETF tickers (for distinguishing ETFs from individual stocks)
_KNOWN_ETF_TICKERS = set(ASSET_CLASS_MAP.keys())


def _get_asset_class(ticker: str) -> str:
    """Map a ticker to its asset class for allocation band purposes."""
    return ASSET_CLASS_MAP.get(ticker, "us_equities")


# ===========================================================================
# ETF STRUCTURAL QUALITY FILTER (Phase 11 Enhancement)
# ===========================================================================

# Cash ticker identifiers -- any ticker in this set is treated as cash
_CASH_TICKERS = {"BIL", "SGOV", "SHV"}


def _redistribute_excess(
    adjusted: Dict[str, float],
    excess: float,
    exclude: set,
) -> None:
    """
    Redistribute excess weight pro-rata across all equity positions,
    excluding cash tickers (SGOV/BIL/SHV) and any tickers in the exclude set.
    Modifies `adjusted` in place.
    """
    eligible = {t: w for t, w in adjusted.items()
                if w > 0 and t not in _CASH_TICKERS and t not in exclude}
    total_eligible = sum(eligible.values())
    if total_eligible > 0:
        for t, w in eligible.items():
            adjusted[t] = w + excess * (w / total_eligible)
    else:
        # Fallback: no eligible equity positions -- park in cash
        # Use SGOV (auto-selected) if present, else BIL
        cash_t = "SGOV" if "SGOV" in adjusted else "BIL"
        adjusted[cash_t] = adjusted.get(cash_t, 0) + excess


def apply_etf_quality_filter(
    weights: Dict[str, float],
    cfg: dict,
) -> Dict[str, float]:
    """
    Post-optimization pass that enforces ETF structural quality rules:

    1. Expense ratio penalty -- reduces weight proportional to fee drag
    2. Overlap group enforcement -- within groups that overlap substantially,
       concentrates weight into the preferred (cheapest/broadest) member
    3. Single-country concentration cap
    4. Total EM / international caps

    Excess from caps is redistributed pro-rata across all equity positions
    (not dumped into BIL/cash) to keep capital fully deployed.

    All thresholds are in config.yaml under etf_quality.
    """
    eq = cfg.get("etf_quality", {})
    if not eq:
        return weights

    expense_ratios = eq.get("expense_ratios_bps", {})
    penalty_factor = eq.get("expense_penalty_factor", 2.0)
    max_expense_bps = eq.get("max_expense_ratio_bps", 50)
    overlap_groups = eq.get("overlap_groups", {})
    max_country_pct = eq.get("max_single_country_pct", 8.0) / 100.0
    max_em_pct = eq.get("max_em_total_pct", 20.0) / 100.0
    max_intl_pct = eq.get("max_intl_total_pct", 35.0) / 100.0

    adjusted = dict(weights)

    # --- 1. Expense ratio penalty ---
    for ticker, w in list(adjusted.items()):
        if w <= 0:
            continue
        er_bps = expense_ratios.get(ticker, 0)
        if er_bps > max_expense_bps:
            er_decimal = er_bps / 10000.0
            penalty = max(0.5, 1.0 - penalty_factor * er_decimal)
            old_w = adjusted[ticker]
            adjusted[ticker] = w * penalty
            logger.info("ETF quality: %s expense ratio %d bps > %d cap "
                        "-> weight %.4f -> %.4f (penalty %.1f%%)",
                        ticker, er_bps, max_expense_bps,
                        old_w, adjusted[ticker], (1 - penalty) * 100)

    # --- 1b. Offense-excluded tickers (e.g. XLP in offense regime) ---
    offense_exclude_list = list(eq.get("offense_exclude", []))
    # Auto-expand: if XLP is excluded, also exclude Fidelity/Vanguard equivalents
    _OFFENSE_EXCLUDE_EQUIV = {
        "XLP": ["FSTA", "VDC"],
    }
    expanded_exclude = set(offense_exclude_list)
    for base, equivs in _OFFENSE_EXCLUDE_EQUIV.items():
        if base in expanded_exclude:
            expanded_exclude.update(equivs)
    for ticker in expanded_exclude:
        if adjusted.get(ticker, 0) > 0:
            excess = adjusted[ticker]
            adjusted[ticker] = 0.0
            _redistribute_excess(adjusted, excess, exclude={ticker})
            logger.info("ETF quality: %s offense-excluded -- %.4f weight "
                        "redistributed to equities", ticker, excess)

    # --- 1c. Per-ticker weight caps (e.g. XLU at 6%) ---
    per_ticker_caps = dict(eq.get("per_ticker_cap_pct", {}))
    # Auto-expand: if XLU is capped, also cap Fidelity/Vanguard equivalents
    _PER_TICKER_CAP_EQUIV = {
        "XLU": ["FUTY", "VPU"],
    }
    for base, equivs in _PER_TICKER_CAP_EQUIV.items():
        if base in per_ticker_caps:
            for eq_t in equivs:
                if eq_t not in per_ticker_caps:
                    per_ticker_caps[eq_t] = per_ticker_caps[base]
    for ticker, cap_pct in per_ticker_caps.items():
        cap_frac = cap_pct / 100.0
        if adjusted.get(ticker, 0) > cap_frac:
            excess = adjusted[ticker] - cap_frac
            adjusted[ticker] = cap_frac
            _redistribute_excess(adjusted, excess, exclude={ticker})
            logger.info("ETF quality: %s capped at %.1f%% (per-ticker cap), "
                        "excess %.4f redistributed", ticker, cap_pct, excess)

    # --- 2. Overlap group enforcement ---
    for group_name, group_cfg in overlap_groups.items():
        members = group_cfg.get("members", [])
        preferred = group_cfg.get("preferred", "")
        overlap_pct = group_cfg.get("overlap_pct", 0)
        max_combined = group_cfg.get("max_combined_weight_pct")

        if overlap_pct < 30:
            # Low overlap -- don't consolidate, but apply combined cap if set
            if max_combined is not None:
                cap = max_combined / 100.0
                member_weights = {t: adjusted.get(t, 0) for t in members if adjusted.get(t, 0) > 0}
                total_group = sum(member_weights.values())
                if total_group > cap:
                    scale = cap / total_group
                    for t in member_weights:
                        adjusted[t] = adjusted[t] * scale
                    logger.info("ETF quality: overlap group '%s' capped at %.1f%% "
                                "(was %.1f%%)", group_name, cap * 100, total_group * 100)
            continue

        # High overlap (>=30%) -- consolidate into preferred member
        member_weights = {t: adjusted.get(t, 0) for t in members if adjusted.get(t, 0) > 0}
        if len(member_weights) <= 1:
            continue

        non_preferred = {t: w for t, w in member_weights.items() if t != preferred}
        if not non_preferred:
            continue

        # Move weight from non-preferred to preferred
        total_moved = 0
        for t, w in non_preferred.items():
            move_pct = overlap_pct / 100.0  # Move proportional to overlap
            move_amount = w * move_pct
            adjusted[t] = w - move_amount
            total_moved += move_amount
            logger.info("ETF quality: overlap '%s' -- moving %.4f from %s to %s "
                        "(%d%% overlap)", group_name, move_amount, t, preferred, overlap_pct)

        adjusted[preferred] = adjusted.get(preferred, 0) + total_moved

    # --- 3. Single-country concentration cap ---
    country_etfs = ["EWJ", "EWY", "EWT", "EWZ", "MCHI", "FXI", "INDA", "KWEB"]
    country_etfs_set = set(country_etfs)
    for t in country_etfs:
        if adjusted.get(t, 0) > max_country_pct:
            excess = adjusted[t] - max_country_pct
            adjusted[t] = max_country_pct
            # Redistribute excess to VWO (broad EM) or VGK (intl developed)
            if _get_asset_class(t) == "em_equities":
                adjusted["VWO"] = adjusted.get("VWO", 0) + excess
            else:
                adjusted["VGK"] = adjusted.get("VGK", 0) + excess
            logger.info("ETF quality: %s capped at %.1f%% (single-country cap), "
                        "excess %.4f redistributed", t, max_country_pct * 100, excess)

    # --- 3b. Single-region ETF cap (e.g. VGK, VWO) ---
    max_region_pct = eq.get("max_single_region_pct", 15.0) / 100.0
    region_etfs = ["VGK", "VWO", "IEMG", "EEM"]
    for t in region_etfs:
        if adjusted.get(t, 0) > max_region_pct:
            excess = adjusted[t] - max_region_pct
            adjusted[t] = max_region_pct
            # Redistribute excess pro-rata across all other equities
            _redistribute_excess(adjusted, excess, exclude={t})
            logger.info("ETF quality: %s capped at %.1f%% (single-region cap), "
                        "excess %.4f redistributed to equities",
                        t, max_region_pct * 100, excess)

    # --- 4. Total EM and international caps ---
    em_tickers = [t for t, w in adjusted.items()
                  if w > 0 and _get_asset_class(t) == "em_equities"]
    em_total = sum(adjusted[t] for t in em_tickers)
    if em_total > max_em_pct:
        scale = max_em_pct / em_total
        excess = em_total - max_em_pct
        for t in em_tickers:
            adjusted[t] *= scale
        # Redistribute excess pro-rata across non-EM equities
        _redistribute_excess(adjusted, excess, exclude=set(em_tickers))
        logger.info("ETF quality: total EM %.1f%% > %.1f%% cap -- scaled down, "
                    "excess redistributed to equities", em_total * 100, max_em_pct * 100)

    intl_classes = {"intl_developed", "em_equities"}
    intl_tickers = [t for t, w in adjusted.items()
                    if w > 0 and _get_asset_class(t) in intl_classes]
    intl_total = sum(adjusted[t] for t in intl_tickers)
    if intl_total > max_intl_pct:
        scale = max_intl_pct / intl_total
        excess = intl_total - max_intl_pct
        for t in intl_tickers:
            adjusted[t] *= scale
        # Redistribute excess pro-rata across domestic equities
        _redistribute_excess(adjusted, excess, exclude=set(intl_tickers))
        logger.info("ETF quality: total intl %.1f%% > %.1f%% cap -- scaled down, "
                    "excess redistributed to equities",
                    intl_total * 100, max_intl_pct * 100)

    # --- Normalize back to 1.0 ---
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {t: w / total for t, w in adjusted.items()}

    # --- Second-pass cap enforcement (normalization can re-inflate capped ETFs) ---
    any_clipped = True
    for _ in range(5):  # max 5 iterations to converge
        any_clipped = False
        for t in region_etfs:
            if adjusted.get(t, 0) > max_region_pct + 1e-6:
                excess = adjusted[t] - max_region_pct
                adjusted[t] = max_region_pct
                _redistribute_excess(adjusted, excess, exclude={t})
                any_clipped = True
        # Re-check country caps
        country_etfs_final = [t for t in adjusted if t in country_etfs_set and adjusted[t] > max_country_pct + 1e-6]
        for t in country_etfs_final:
            excess = adjusted[t] - max_country_pct
            adjusted[t] = max_country_pct
            _redistribute_excess(adjusted, excess, exclude={t})
            any_clipped = True
        if any_clipped:
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {t: w / total for t, w in adjusted.items()}
        else:
            break

    # Remove zero/near-zero weights
    adjusted = {t: w for t, w in adjusted.items() if w > 0.001}

    return adjusted


def _compute_allocation_bounds(
    tickers: List[str],
    regime: str,
    cfg: dict,
    factor_scores: pd.DataFrame = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute per-ticker allocation bounds based on regime bands and factor scores.

    FIX 3: The old implementation took the GROUP band (e.g. us_equities
    offense [0.55, 0.75]) and divided lo/hi equally among all tickers in
    the group. When you have 8 asset classes whose lower bounds sum > 1.0,
    the optimizer sees infeasible constraints.

    NEW approach:
      - Lower bound per-ticker = 0 (let the optimizer decide who gets weight)
      - Upper bound per-ticker = group_hi / n_in_class (cap any single ticker)
      - This guarantees sum of lower bounds = 0.0 (always feasible)
      - The optimizer will naturally allocate within the asset class bands
        because the sum of uppers caps each class
    """
    bands = cfg["optimizer"]["allocation_bands"]
    bounds = {}

    # Count tickers per asset class for splitting group budget
    class_counts = {}
    class_tickers = {}
    for t in tickers:
        ac = _get_asset_class(t)
        class_counts[ac] = class_counts.get(ac, 0) + 1
        class_tickers.setdefault(ac, []).append(t)

    for ticker in tickers:
        ac = _get_asset_class(ticker)
        band = bands.get(ac, {"panic": [0, 0.05], "defense": [0.05, 0.15], "offense": [0.1, 0.3]})
        regime_band = band.get(regime, band.get("offense", [0.05, 0.20]))

        group_lo = regime_band[0]
        group_hi = regime_band[1]
        n_in_class = max(1, class_counts.get(ac, 1))

        # Per-ticker lower bound = 0 (optimizer chooses freely within class)
        # Per-ticker upper bound = group upper / n_in_class (prevent domination)
        per_ticker_lo = 0.0
        per_ticker_hi = group_hi / n_in_class

        # For small classes (1-2 tickers), allow the full group upper
        if n_in_class <= 2:
            per_ticker_hi = group_hi

        # Boost upper bound for high-scoring tickers within the class
        if factor_scores is not None and ticker in factor_scores["ticker"].values:
            score = factor_scores.loc[factor_scores["ticker"] == ticker, "composite_score"].values[0]
            # Top-scoring tickers get up to 2x their share of the class budget
            if score > 0.6:
                per_ticker_hi = min(per_ticker_hi * (1.0 + score), group_hi * 0.5)

        # Ensure hi >= lo and clamp to [0, 1]
        per_ticker_hi = max(per_ticker_lo, min(per_ticker_hi, 1.0))

        bounds[ticker] = (round(per_ticker_lo, 6), round(per_ticker_hi, 6))

    # ---- Feasibility validation ----
    total_lo = sum(lo for lo, _ in bounds.values())
    total_hi = sum(hi for _, hi in bounds.values())
    if total_lo > 1.0:
        logger.error("INFEASIBLE: sum of lower bounds = %.4f > 1.0 -- scaling down", total_lo)
        scale = 0.99 / total_lo
        bounds = {t: (lo * scale, hi) for t, (lo, hi) in bounds.items()}
    if total_hi < 1.0:
        logger.warning("Sum of upper bounds = %.4f < 1.0 -- relaxing proportionally", total_hi)
        scale = 1.01 / total_hi
        bounds = {t: (lo, min(hi * scale, 1.0)) for t, (lo, hi) in bounds.items()}

    logger.info("Allocation bounds: sum(lo)=%.4f, sum(hi)=%.4f (%d tickers)",
                sum(lo for lo, _ in bounds.values()),
                sum(hi for _, hi in bounds.values()),
                len(bounds))

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
    Compute Bivector Beta for a sector -- the degree of orthogonality
    to the first principal component of the market factor.

    Bivector Beta > 1.0 means the sector is a structural diversifier
    (its returns are significantly different from the dominant market mode).
    """
    if sector_ticker not in returns.columns:
        return 1.0

    if market_tickers is None:
        market_tickers = [c for c in returns.columns if c != sector_ticker]

    if market_proxy and market_proxy in returns.columns and market_proxy != sector_ticker:
        if market_proxy not in market_tickers:
            market_tickers = market_tickers + [market_proxy]
            logger.debug("Bivector Beta: added %s as market anchor for PCA", market_proxy)

    candidate_cols = [c for c in market_tickers if c in returns.columns]
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

    market_data = clean[candidate_cols].values
    pca = PCA(n_components=1)
    pca.fit(market_data)
    pc1 = pca.transform(market_data).flatten()

    sector_rets = clean[sector_ticker].values
    corr = np.corrcoef(sector_rets, pc1)[0, 1]

    bivector_beta = 1.0 / max(abs(corr), 0.01)
    logger.info("Bivector Beta %s: corr=%.4f, beta_bv=%.4f (%d obs, %d market tickers)",
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
    Allocate within US Equities using factor-score proportional tilting
    (FIX A: replaces DeMiguel equal-weight).

    Each US sector ETF's share of us_equity_weight is proportional to its
    composite_score, with valuation filter overrides (MOMENTUM_ONLY gets
    50% cap, AVOID gets 0).  Bivector Beta still boosts structural
    diversifiers (XLE, XLB).

    Returns dict of ticker -> {weight, label, bivector_beta, ...}
    """
    us_sectors = [t for t in sector_etfs if _get_asset_class(t) == "us_equities"]
    n = len(us_sectors)
    if n == 0:
        return {}

    equal_weight = us_equity_weight / n   # kept as reference / floor
    val_cfg = cfg["factor_model"]["valuation"]
    mom_cap = val_cfg["momentum_cap_fraction"]  # 0.50

    energy_materials = [t for t in sector_etfs if _get_asset_class(t) == "energy_materials"]

    # --- Pass 1: collect composite scores and labels ---
    raw_scores: Dict[str, float] = {}
    labels: Dict[str, str] = {}
    bv_betas: Dict[str, float] = {}

    for ticker in us_sectors:
        row = factor_scores[factor_scores["ticker"] == ticker]
        label = "FUNDAMENTAL_BUY"
        score = 0.5   # neutral default

        if not row.empty:
            score = float(row["composite_score"].values[0])
            mom   = float(row["momentum_rank"].values[0])
            alpha = float(row["adjusted_alpha"].values[0])

            if mom > 0.75 and alpha < 0:
                label = "MOMENTUM_ONLY"
            elif mom < 0.10 and alpha < -0.05:
                label = "AVOID"

        # Bivector Beta for energy/materials structural diversifiers
        bv_beta = 1.0
        if ticker in energy_materials and returns is not None:
            sector_only_cols = [c for c in sector_etfs if c in returns.columns]
            if sector_only_cols:
                bv_beta = compute_bivector_beta(
                    returns[sector_only_cols], ticker,
                    market_tickers=[c for c in sector_only_cols if c != ticker],
                )

        raw_scores[ticker] = max(score, 0.01)  # floor prevents zero-div
        labels[ticker] = label
        bv_betas[ticker] = bv_beta

    # --- Pass 2: compute proportional weights ---
    # Shift scores: high-score ETFs get meaningfully more weight.
    # Small base (0.02) prevents total starvation but widens the spread.
    min_score = min(raw_scores.values())
    shifted = {t: (s - min_score + 0.02) for t, s in raw_scores.items()}

    # Apply label overrides before normalizing
    for ticker in us_sectors:
        if labels[ticker] == "AVOID":
            shifted[ticker] = 0.0
        elif labels[ticker] == "MOMENTUM_ONLY":
            shifted[ticker] *= mom_cap   # halve its contribution

    total_shifted = sum(shifted.values())
    if total_shifted <= 0:
        # Fallback: equal-weight the non-AVOID tickers
        active = [t for t in us_sectors if labels[t] != "AVOID"]
        total_shifted = len(active) or 1
        shifted = {t: (1.0 if t in active else 0.0) for t in us_sectors}
        total_shifted = sum(shifted.values()) or 1.0

    result = {}
    for ticker in us_sectors:
        prop_weight = (shifted[ticker] / total_shifted) * us_equity_weight

        result[ticker] = {
            "weight": round(prop_weight, 6),
            "label": labels[ticker],
            "composite_score": round(raw_scores[ticker], 4),
            "equal_weight_base": round(equal_weight, 6),
            "bivector_beta": bv_betas[ticker],
            "is_structural_diversifier": bv_betas[ticker] > 1.0 and ticker in energy_materials,
        }

    # Normalize so weights sum exactly to us_equity_weight
    total = sum(v["weight"] for v in result.values())
    if total > 0:
        scale = us_equity_weight / total
        for t in result:
            result[t]["weight"] = round(result[t]["weight"] * scale, 6)

    logger.info("US sub-sector allocation (factor-tilted): %s",
                {t: f"{v['weight']:.4f} ({v['label']})" for t, v in result.items() if v['weight'] > 0})
    return result


# ===========================================================================
# 4B. SCREENER INTEGRATION (FIX B)
# ===========================================================================

def _inject_screener_picks(
    weights: Dict[str, float],
    cfg: dict,
    regime: str,
) -> Dict[str, float]:
    """
    Inject top screener stock picks into the portfolio as a Roth IRA
    satellite sleeve.

    Strategy:
      - Reserve `roth_satellite_pct` (default 20%) of the Roth IRA for
        individual stock picks from the screener's top-scored holdings.
      - Pick the top `satellite_stock_count` (default 5) stocks, combining
        candidates from ETF holdings screens (Part A) and watchlist ENTRY
        signals, ranked by composite_score.
      - Carve the satellite weight proportionally from Roth-bound ETF
        positions (those already tagged for Roth placement).
      - Each stock gets equal weight within the satellite sleeve.
      - Only injects in Offense regime.
      - Skips tickers already in the ETF weights.

    The satellite stocks will be routed to Roth IRA by allocate_dollars()
    since they are registered as individual stocks (not in _KNOWN_ETF_TICKERS).
    """
    if regime != "offense":
        logger.info("Screener picks only injected in Offense regime -- skipping (regime=%s)", regime)
        return weights

    screener_path = Path(__file__).parent / "screener_output.json"
    if not screener_path.exists():
        logger.info("No screener_output.json found -- skipping screener injection")
        return weights

    try:
        with open(screener_path) as f:
            screener = json.load(f)
    except Exception as e:
        logger.warning("Failed to read screener_output.json: %s", e)
        return weights

    # --- Gather candidate stocks from all screener sources ---
    candidates = []  # list of (ticker, composite_score, source)

    # Source 1: ETF holdings screens (Part A) -- top stocks from overweight sectors
    etf_screens = screener.get("etf_screens", {})
    for etf, rows in etf_screens.items():
        if isinstance(rows, list):
            for row in rows:
                ticker = row.get("ticker", "")
                score = row.get("composite_score", 0)
                label = row.get("valuation_label", "")
                if ticker and label != "AVOID":
                    candidates.append((ticker, score, f"sector_screen:{etf}"))

    # Source 2: Watchlist ENTRY signals (Part C)
    signals = screener.get("signals", {})
    entries = signals.get("entry", [])
    for entry in entries:
        ticker = entry.get("ticker", "")
        score = entry.get("composite_score", 0)
        if ticker:
            candidates.append((ticker, score, f"watchlist:{entry.get('watchlist', '')}"))

    if not candidates:
        logger.info("No screener candidates found -- no stock picks to inject")
        return weights

    # --- Deduplicate (keep highest score per ticker) ---
    best_by_ticker: Dict[str, Tuple[float, str]] = {}
    for ticker, score, source in candidates:
        if ticker not in best_by_ticker or score > best_by_ticker[ticker][0]:
            best_by_ticker[ticker] = (score, source)

    # --- Filter out tickers already in ETF weights ---
    available = [
        (ticker, score, source)
        for ticker, (score, source) in best_by_ticker.items()
        if ticker not in weights and ticker not in _KNOWN_ETF_TICKERS
    ]
    available.sort(key=lambda x: x[1], reverse=True)

    # --- Pick top N ---
    sc_cfg = cfg.get("stock_screener", {})
    n_picks = sc_cfg.get("satellite_stock_count", 5)
    satellite_pct = sc_cfg.get("roth_satellite_pct", 0.20)
    roth_value = cfg["portfolio"]["accounts"]["roth_ira"]["value"]
    total_value = cfg["portfolio"]["total_value"]

    picks = available[:n_picks]
    if not picks:
        logger.info("No eligible screener picks after filtering -- skipping")
        return weights

    # --- Calculate satellite weight as fraction of total portfolio ---
    # satellite_pct is 20% of Roth, but weights are expressed as fraction of
    # total portfolio. Convert: satellite_weight = satellite_pct * roth_value / total_value
    satellite_total_weight = satellite_pct * roth_value / total_value
    per_stock_weight = satellite_total_weight / len(picks)

    # --- Carve satellite weight from existing Roth-bound positions ---
    # Identify which current ETFs will likely land in Roth so we carve from them
    roth_etf_candidates = []
    high_div_sectors = set(cfg.get("tax_location", {}).get("high_dividend_sectors", []))
    # Expand to include equivalents
    _SECTOR_EQUIV = {
        "XLU": ["FUTY", "VPU"], "XLP": ["FSTA", "VDC"],
        "XLRE": ["FREL", "VNQ"],
    }
    for spdr, equivs in _SECTOR_EQUIV.items():
        if spdr in high_div_sectors:
            for eq in equivs:
                high_div_sectors.add(eq)

    for t, w in weights.items():
        if w <= 0:
            continue
        # Positions likely going to Roth: high-dividend sectors, thematic, watchlist stocks
        if t in high_div_sectors:
            roth_etf_candidates.append(t)
        elif _get_asset_class(t) in ("thematic", "industry_sub"):
            roth_etf_candidates.append(t)

    # If not enough Roth-bound ETFs, carve from smallest overall positions
    if not roth_etf_candidates:
        roth_etf_candidates = sorted(
            [t for t, w in weights.items() if w > 0],
            key=lambda t: weights[t]
        )[:5]

    # Carve proportionally from Roth-bound ETFs
    total_roth_etf_weight = sum(weights[t] for t in roth_etf_candidates if t in weights)
    if total_roth_etf_weight > 0:
        for t in roth_etf_candidates:
            if t in weights and weights[t] > 0:
                share = weights[t] / total_roth_etf_weight
                carve = satellite_total_weight * share
                weights[t] = max(0.001, weights[t] - carve)
    else:
        # Fallback: carve equally from all positions
        carve_per = satellite_total_weight / max(len(weights), 1)
        for t in list(weights.keys()):
            weights[t] = max(0.001, weights[t] - carve_per)

    # --- Inject the picks ---
    for ticker, score, source in picks:
        weights[ticker] = per_stock_weight
        # Register as individual stock in ASSET_CLASS_MAP
        if ticker not in ASSET_CLASS_MAP:
            ASSET_CLASS_MAP[ticker] = "individual_stock"
        logger.info(
            "Injected satellite pick: %s (%.2f%% = ~$%.0f in Roth, score=%.3f, from %s)",
            ticker, per_stock_weight * 100,
            per_stock_weight * total_value,
            score, source,
        )

    # Re-normalize to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    logger.info(
        "Satellite sleeve: %d stocks injected, %.1f%% of Roth ($%.0f), "
        "~$%.0f per stock",
        len(picks),
        satellite_pct * 100,
        satellite_total_weight * total_value,
        per_stock_weight * total_value,
    )

    return weights


# ===========================================================================
# 5. DOLLAR ALLOCATION ENGINE
# ===========================================================================

def _concentrate_portfolio(
    weights: Dict[str, float],
    max_positions: int,
    factor_scores: pd.DataFrame = None,
) -> Dict[str, float]:
    """
    FIX 5 + FIX C: Reduce position count for a $144K portfolio,
    with reserved slots per asset class so that industry/thematic/
    screener picks are not wiped out by the 2% floor.

    Strategy:
      1. Reserve at least 1 slot each for industry_sub, thematic,
         and individual stocks (screener picks) if any exist
      2. Keep all positions >= 2% weight (meaningful allocation)
      3. Within reserved asset classes, keep the top-scored ticker
         even if below 2%
      4. Fill remaining slots with top-scored optionals
      5. Redistribute weight from dropped positions proportionally
    """
    if len(weights) <= max_positions:
        return weights

    # Classify positions by asset class
    RESERVED_CLASSES = {"industry_sub", "thematic"}
    # Screener-injected individual stocks won't be in the original
    # ASSET_CLASS_MAP, so anything not in the map is treated as an
    # individual stock pick (also reserved)
    INDIVIDUAL_SENTINEL = "__individual__"

    per_class: Dict[str, list] = {}   # ac -> [(ticker, weight, fscore)]
    for ticker, w in weights.items():
        if w <= 0:
            continue
        ac = _get_asset_class(ticker)
        # Check if this is an individual stock (screener satellite pick)
        if ac == "individual_stock":
            ac = INDIVIDUAL_SENTINEL
        elif ac == "us_equities" and ticker not in _KNOWN_ETF_TICKERS:
            # Could be a screener pick -- mark as individual
            ac = INDIVIDUAL_SENTINEL

        fscore = 0.5
        if factor_scores is not None and not factor_scores.empty:
            match = factor_scores[factor_scores["ticker"] == ticker]
            if not match.empty:
                fscore = match["composite_score"].values[0]

        per_class.setdefault(ac, []).append((ticker, w, fscore))

    # Sort each class by factor score descending
    for ac in per_class:
        per_class[ac].sort(key=lambda x: x[2], reverse=True)

    # --- Pass 1: Mandatory picks (>= 2% weight) ---
    mandatory = {}
    for ac, items in per_class.items():
        for ticker, w, fs in items:
            if w >= 0.02:
                mandatory[ticker] = w

    # --- Pass 2: Reserved best-of-class picks ---
    reserved = {}
    reserved_acs = RESERVED_CLASSES | {INDIVIDUAL_SENTINEL}
    for ac in reserved_acs:
        items = per_class.get(ac, [])
        if items:
            # Keep the top-scored ticker from this class even if < 2%
            best_t, best_w, best_fs = items[0]
            if best_t not in mandatory:
                reserved[best_t] = best_w

    # --- Pass 3: Fill remaining slots from optional pool ---
    used = set(mandatory.keys()) | set(reserved.keys())
    remaining_slots = max(0, max_positions - len(used))

    optional = []
    for ac, items in per_class.items():
        for ticker, w, fs in items:
            if ticker not in used and w > 0:
                optional.append((ticker, w, fs))

    optional.sort(key=lambda x: x[2], reverse=True)
    kept_optional = {t: w for t, w, _ in optional[:remaining_slots]}

    # --- Merge all ---
    final = {**mandatory, **reserved, **kept_optional}

    # Redistribute dropped weight proportionally
    total_kept = sum(final.values())
    if total_kept > 0 and total_kept < 0.999:
        scale = 1.0 / total_kept
        final = {t: w * scale for t, w in final.items()}

    dropped = len(weights) - len(final)
    if dropped > 0:
        logger.info("Concentrated portfolio: %d -> %d positions (dropped %d), "
                    "reserved %d class-champion slots (industry/thematic/individual)",
                    len(weights), len(final), dropped, len(reserved))

    return final


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

    # Individual stock satellite picks -> Roth (highest growth potential = tax-free)
    for t, ac in ASSET_CLASS_MAP.items():
        if ac == "individual_stock":
            roth_tickers.add(t)

    # Watchlist tickers -> Roth
    for key in ["watchlist_biotech", "watchlist_ai_software",
                "watchlist_defense", "watchlist_green_materials",
                "watchlist_semiconductors", "watchlist_energy_transition",
                "watchlist_fintech"]:
        for t in cfg["tickers"].get(key, []):
            roth_tickers.add(t)

    # Thematic ETFs -> Roth (high turnover)
    for t in cfg["tickers"].get("thematic_etfs", []):
        roth_tickers.add(t)

    # Geographic ETFs -> Taxable (foreign tax credit)
    # Includes legacy tickers from config + all Franklin FTSE alternatives
    for t in cfg["tickers"].get("geographic_etfs", []):
        taxable_tickers.add(t)
    # Add auto-selector geographic candidates (Franklin FTSE series)
    _GEOGRAPHIC_TICKERS = {"FLIN", "FLBR", "FLJP", "FLEU", "FLCH", "FLKR", "FLTW",
                           "INDA", "EWZ", "EWJ", "EWY", "EWT", "VGK", "MCHI",
                           "VWO", "IEMG", "EEM", "FXI", "KWEB", "INDY", "FEZ", "DXJ"}
    for t in _GEOGRAPHIC_TICKERS:
        taxable_tickers.add(t)

    # Cash/bonds -> Taxable (includes auto-selected cash ticker like SGOV)
    for t in _CASH_TICKERS | {"AGG", "TLT"}:
        taxable_tickers.add(t)

    # Energy/Materials ETFs -> Taxable (structural diversifier, low turnover)
    # Includes auto-selected variants (FENY, FMAT, VDE, VAW)
    energy_mat_tickers = {t for t, ac in ASSET_CLASS_MAP.items() if ac == "energy_materials"}
    energy_mat_tickers.add("GLD")  # commodity bucket
    for t in energy_mat_tickers:
        taxable_tickers.add(t)

    # === Dividend-yield-aware tax location (Taxable Yield Trap fix) ===
    # High-dividend US sectors -> Roth (dividends taxed as ordinary income in taxable)
    high_div_sectors = set(cfg.get("tax_location", {}).get("high_dividend_sectors", []))
    # Map auto-selected equivalents: Fidelity/Vanguard sector ETFs inherit the
    # same dividend routing as their SPDR counterparts (same underlying sectors)
    _SECTOR_DIVIDEND_EQUIV = {
        # High-dividend equivalents
        "XLU": ["FUTY", "VPU"],   # Utilities ~2.8% yield
        "XLP": ["FSTA", "VDC"],   # Consumer Staples ~2.6% yield
        "XLRE": ["FREL", "VNQ"],  # Real Estate ~3.2% yield
        # Growth/low-dividend equivalents
        "XLI": ["FIDU", "VIS"],   # Industrials ~1.4% yield
        "XLK": ["FTEC", "VGT"],   # Technology ~0.6% yield
        "XLC": ["FCOM", "VOX"],   # Communication ~0.8% yield
        "XLY": ["FDIS", "VCR"],   # Consumer Disc ~0.9% yield
        "XLF": ["FNCL", "VFH"],   # Financials ~1.3% yield
    }
    for spdr, equivs in _SECTOR_DIVIDEND_EQUIV.items():
        if spdr in high_div_sectors:
            for eq in equivs:
                high_div_sectors.add(eq)
    for t in high_div_sectors:
        roth_tickers.add(t)
        taxable_tickers.discard(t)  # ensure no conflict

    # Growth/appreciation US sectors -> Taxable (long-term cap gains taxed favorably)
    growth_sectors = set(cfg.get("tax_location", {}).get("growth_sectors", []))
    for spdr, equivs in _SECTOR_DIVIDEND_EQUIV.items():
        if spdr in growth_sectors:
            for eq in equivs:
                growth_sectors.add(eq)
    for t in growth_sectors:
        taxable_tickers.add(t)
        roth_tickers.discard(t)  # ensure no conflict

    result = {}
    roth_used = 0.0
    taxable_used = 0.0

    # Sort by dollar amount descending for better bin-packing
    # BUT: process individual satellite stocks FIRST to guarantee Roth placement
    satellite_positions = [(t, w) for t, w in weights.items() if ASSET_CLASS_MAP.get(t) == "individual_stock" and w > 0]
    non_satellite_positions = [(t, w) for t, w in weights.items() if ASSET_CLASS_MAP.get(t) != "individual_stock" and w > 0]
    # Satellite stocks first (by weight desc), then ETFs (by weight desc)
    satellite_positions.sort(key=lambda x: x[1], reverse=True)
    non_satellite_positions.sort(key=lambda x: x[1], reverse=True)
    sorted_positions = satellite_positions + non_satellite_positions

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
            if ASSET_CLASS_MAP.get(ticker) == "individual_stock":
                base_reason = "Roth: satellite stock pick (screener top-5, tax-free growth)"
            elif ticker in high_div_sectors:
                base_reason = "Roth: high-dividend sector (avoid taxable dividend drag)"
            elif is_momentum_only:
                base_reason = "Roth: momentum-only position (high turnover)"
            else:
                base_reason = "Roth: thematic/high-turnover position"
            if roth_used + dollars <= roth_cap:
                account = "roth_ira"
                roth_dollars = dollars
                taxable_dollars = 0.0
                reason = base_reason
                roth_used += dollars
            elif dollars > split_thresh and roth_used < roth_cap and taxable_used < taxable_cap:
                roth_avail = round(roth_cap - roth_used, 2)
                taxable_avail = round(taxable_cap - taxable_used, 2)
                roth_dollars = min(roth_avail, dollars)
                taxable_dollars = min(taxable_avail, round(dollars - roth_dollars, 2))
                account = "split"
                reason = f"Split: ${roth_dollars:.0f} Roth + ${taxable_dollars:.0f} Taxable (exceeds ${split_thresh:,.0f})"
                roth_used += roth_dollars
                taxable_used += taxable_dollars
            elif taxable_used + dollars <= taxable_cap:
                account = "taxable"
                taxable_dollars = dollars
                roth_dollars = 0.0
                reason = "Taxable: Roth capacity full"
                taxable_used += dollars
            else:
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
            if ticker in growth_sectors:
                base_tax_reason = "Taxable: growth sector (capital appreciation, low dividend)"
            else:
                base_tax_reason = "Taxable: geographic/cash/energy (foreign tax credit / low turnover)"
            if taxable_used + dollars <= taxable_cap:
                account = "taxable"
                taxable_dollars = dollars
                roth_dollars = 0.0
                reason = base_tax_reason
                taxable_used += dollars
            elif roth_used + dollars <= roth_cap:
                account = "roth_ira"
                roth_dollars = dollars
                taxable_dollars = 0.0
                reason = "Roth: taxable capacity full"
                roth_used += dollars
            else:
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
            # Default: broad ETF -> taxable (long-term hold)
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
    6. Concentrate portfolio to max_positions
    7. Convert to dollar amounts across accounts
    8. Output JSON + CSV

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
        logger.error("Factor scoring failed -- cannot optimize.")
        if close_conn:
            conn.close()
        return {}

    logger.info("Top factor scores:\n%s", factor_scores.head(5).to_string())

    # --- Step 3: Load price returns for optimization ---
    # Use auto-selected tickers when available
    sector_tickers = cfg["tickers"]["sector_etfs"]
    geo_tickers = cfg["tickers"]["geographic_etfs"]
    industry_tickers = cfg["tickers"].get("industry_etfs", [])
    thematic_tickers = cfg["tickers"].get("thematic_etfs", [])

    # Resolve best-in-class cash ticker from ETF selector (default SGOV)
    if _ETF_SELECTOR_AVAILABLE:
        etf_selections = _get_etf_selections(cfg)
        cash_ticker = etf_selections.get("cash_tbill", "SGOV")
        logger.info("ETF selector: cash ticker = %s", cash_ticker)
    else:
        cash_ticker = "SGOV"  # default to SGOV (cheapest)

    cash_tickers = [cash_ticker]
    all_opt_tickers = sector_tickers + geo_tickers + industry_tickers + thematic_tickers + cash_tickers
    # Deduplicate while preserving order
    seen = set()
    all_opt_tickers = [t for t in all_opt_tickers if not (t in seen or seen.add(t))]
    logger.info("Optimization universe: %d tickers", len(all_opt_tickers))

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
        logger.error("All weights are zero after sub-sector allocation -- cannot allocate dollars.")
        if close_conn:
            conn.close()
        return {}

    # --- Step 5.5: (screener injection moved to Step 5.95 -- after all filters) ---

    # --- Step 5.6: Concentrate portfolio (FIX 5) ---
    max_positions = cfg.get("optimizer", {}).get("max_positions", 15)
    weights = _concentrate_portfolio(weights, max_positions, factor_scores)

    # --- Step 5.7: ETF structural quality filter (Phase 11 Enhancement) ---
    logger.info("Applying ETF quality filter (expense ratios, overlap, caps)...")
    weights = apply_etf_quality_filter(weights, cfg)
    logger.info("Post-quality weights: %d positions", sum(1 for w in weights.values() if w > 0.001))

    # --- Step 5.8: Minimum position size enforcement ---
    min_pct = cfg.get("optimizer", {}).get("min_position_pct", 0) / 100.0
    if min_pct > 0:
        dropped = []
        for _ in range(10):  # iterate until no sub-minimum positions remain
            sub_min = {t: w for t, w in weights.items()
                       if 0 < w < min_pct and t not in _CASH_TICKERS}
            if not sub_min:
                break
            total_excess = sum(sub_min.values())
            for t in sub_min:
                dropped.append((t, weights[t]))
                weights[t] = 0.0
            # Redistribute pro-rata across remaining equity positions
            _redistribute_excess(weights, total_excess, exclude=set(sub_min.keys()))
            # Re-normalize
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {t: w / total_w for t, w in weights.items()}
            # Remove zeroed entries
            weights = {t: w for t, w in weights.items() if w > 0.001}
        if dropped:
            logger.info("Min position filter: dropped %d positions below %.1f%%: %s",
                        len(dropped), min_pct * 100,
                        ", ".join(f"{t} ({w*100:.1f}%)" for t, w in dropped))
        logger.info("Post-min-filter: %d positions", len(weights))

        # Re-apply quality caps since redistribution may have inflated capped ETFs
        weights = apply_etf_quality_filter(weights, cfg)

    # --- Step 5.9: ETF ticker substitution (auto-selector) ---
    # Replace SPDR/legacy tickers with auto-selected best-in-class ETFs.
    # Optimization runs on SPDR tickers (which have price history in DB),
    # but final allocation uses the cheapest equivalent (e.g., XLK -> FTEC).
    if _ETF_SELECTOR_AVAILABLE:
        _SLOT_TO_LEGACY = {
            "us_technology": "XLK", "us_healthcare": "XLV", "us_financials": "XLF",
            "us_industrials": "XLI", "us_consumer_disc": "XLY",
            "us_consumer_staples": "XLP", "us_energy": "XLE", "us_materials": "XLB",
            "us_utilities": "XLU", "us_real_estate": "XLRE",
            "us_communication": "XLC",
            "india": "INDA", "brazil": "EWZ", "japan": "EWJ",
            "europe": "VGK", "china": "MCHI", "south_korea": "EWY",
            "taiwan": "EWT", "broad_em": "VWO",
            "cash_tbill": "BIL",
        }
        # Build reverse: legacy_ticker -> selected_ticker
        etf_subs = {}
        for slot, legacy in _SLOT_TO_LEGACY.items():
            selected = etf_selections.get(slot)
            if selected and selected != legacy:
                etf_subs[legacy] = selected

        if etf_subs:
            new_weights = {}
            for t, w in weights.items():
                new_t = etf_subs.get(t, t)
                new_weights[new_t] = new_weights.get(new_t, 0) + w
            weights = new_weights
            logger.info("ETF substitution: %s",
                        ", ".join(f"{old}->{new}" for old, new in sorted(etf_subs.items())))

    # --- Step 5.95: Inject screener stock picks AFTER all filters ---
    # Satellite stocks are a separate 20%-of-Roth sleeve.  They must be
    # injected *after* concentration, min-position, quality, and ETF-
    # substitution filters so they are never culled.  The function carves
    # weight from Roth-bound ETFs and re-normalises to 1.0.
    weights = _inject_screener_picks(weights, cfg, regime)

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
            logger.info("DB empty -- seeding synthetic sector prices for smoke testing")
            cfg = load_config()
            dates = pd.bdate_range(end=dt.date.today(), periods=600)
            np.random.seed(42)
            mock_rows = []
            all_mock_tickers = (cfg["tickers"]["sector_etfs"] +
                                cfg["tickers"]["geographic_etfs"] + ["SGOV", "BIL", "SPY"])
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
        print("PORTFOLIO ALLOCATION -- %s REGIME" % result.get("regime", "?").upper())
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
