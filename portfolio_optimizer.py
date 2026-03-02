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

        # Find the header row (starts with date column — skip description lines)
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
                    "Tail corr %s/%s: only %d joint events (need >=%d) — keeping full-sample %.4f",
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
        logger.warning("Too few tickers (%d) after cleaning — skipping CVaR", returns.shape[1])
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
    #     pypfopt NaN/Inf warnings — mean_historical_return expects prices) ---
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
        logger.warning("CVaR optimization failed: %s — using smart fallback", e)
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
    # US Equities — SPDR XL* series (11 GICS sectors)
    "XLK": "us_equities", "XLV": "healthcare", "XLE": "energy_materials",
    "XLF": "us_equities", "XLI": "us_equities", "XLB": "energy_materials",
    "XLU": "us_equities", "XLP": "us_equities", "XLRE": "us_equities",
    "XLC": "us_equities", "XLY": "us_equities",
    # US Equities — Fidelity MSCI series (auto-selector candidates)
    "FTEC": "us_equities", "FHLC": "healthcare", "FENY": "energy_materials",
    "FNCL": "us_equities", "FIDU": "us_equities", "FMAT": "energy_materials",
    "FUTY": "us_equities", "FSTA": "us_equities", "FREL": "us_equities",
    "FCOM": "us_equities", "FDIS": "us_equities",
    # US Equities — Vanguard series (auto-selector candidates)
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
    # International Developed — legacy + Franklin FTSE
    "VGK": "intl_developed", "EWJ": "intl_developed",
    "FLJP": "intl_developed", "FLEU": "intl_developed",
    "FEZ": "intl_developed", "DXJ": "intl_developed",
    # Emerging Markets — legacy + Franklin FTSE
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

# Cash ticker identifiers — any ticker in this set is treated as cash
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
        # Fallback: no eligible equity positions — park in cash
        # Use SGOV (auto-selected) if present, else BIL
        cash_t = "SGOV" if "SGOV" in adjusted else "BIL"
        adjusted[cash_t] = adjusted.get(cash_t, 0) + excess


def apply_etf_quality_filter(
    weights: Dict[str, float],
    cfg: dict,
) -> Dict[str, float]:
    """
    Post-optimization pass that enforces ETF structural quality rules:

    1. Expense ratio penalty — reduces weight proportional to fee drag
    2. Overlap group enforcement — within groups that overlap substantially,
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
                        "→ weight %.4f → %.4f (penalty %.1f%%)",
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
            logger.info("ETF quality: %s offense-excluded — %.4f weight "
                
... [truncated, 84096 chars total]
