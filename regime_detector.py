"""
regime_detector.py — Phase 2: Geometric Beta Regime Detector
==============================================================
Global Sector Rotation System

Implements the three-tier Panic / Defense / Offense architecture from
Section 2 of the report, driven by the Wedge Volume signal (the scalar
measure of k-dimensional spread across sector return vectors).

Also implements the supplementary Fast-Shock Indicator (Section 9.1)
that catches policy-shock failure modes missed by wedge volume.

Dependencies: numpy, pandas, scipy, pyyaml, sqlite3
"""

import json
import logging
import sqlite3
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logger = logging.getLogger("regime_detector")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"
DB_PATH = Path(__file__).parent / "rotation_system.db"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ===========================================================================
# 1. WEDGE VOLUME COMPUTATION
# ===========================================================================

def compute_log_returns(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a wide-format price DataFrame
    (index = date, columns = tickers, values = adjusted close).
    """
    log_ret = np.log(prices_wide / prices_wide.shift(1))
    return log_ret.dropna(how="all")


def compute_demeaned_returns(log_returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute demeaned log-return vectors using a rolling window.
    For each date, subtract the rolling mean of the past `window` days.
    """
    rolling_mean = log_returns.rolling(window=window, min_periods=max(1, window // 2)).mean()
    demeaned = log_returns - rolling_mean
    return demeaned


def compute_wedge_volume_series(
    log_returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """
    Compute the daily "wedge volume" — the absolute value of the
    determinant of the rolling covariance matrix of demeaned sector
    log-return vectors.

    This is the scalar measure of k-dimensional spread across sectors.
    High wedge volume = sectors diverging (healthy, normal rotation).
    Low wedge volume = sectors converging (crisis, herding behavior).

    Parameters
    ----------
    log_returns : pd.DataFrame
        Wide-format log returns (date index × sector tickers).
    window : int
        Rolling window for covariance estimation (default: 63 trading days).

    Returns
    -------
    pd.Series indexed by date with the raw wedge volume.
    """
    dates = log_returns.index
    n_sectors = log_returns.shape[1]
    wedge_vols = pd.Series(index=dates, dtype=float, name="wedge_volume")

    # We need at least `window` observations and at least n_sectors rows
    min_obs = max(window, n_sectors + 1)

    for i in range(min_obs, len(dates)):
        window_data = log_returns.iloc[max(0, i - window):i]

        # Drop any columns that are all NaN in this window
        valid_cols = window_data.dropna(axis=1, how="all")
        if valid_cols.shape[1] < 2:
            continue

        # Demean within the window
        demeaned = valid_cols - valid_cols.mean()

        # Drop rows with any NaN after demeaning
        demeaned = demeaned.dropna()

        if len(demeaned) < valid_cols.shape[1] + 1:
            # Not enough observations for a well-conditioned covariance
            continue

        try:
            cov_matrix = demeaned.cov()

            # Compute the absolute determinant — this is the wedge volume
            # For numerical stability, use slogdet
            sign, logdet = np.linalg.slogdet(cov_matrix.values)
            if np.isfinite(logdet):
                wedge_vol = np.abs(sign) * np.exp(logdet)
            else:
                wedge_vol = 0.0

            wedge_vols.iloc[i] = wedge_vol
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug("Covariance computation failed at %s: %s", dates[i], e)
            continue

    return wedge_vols


def compute_wedge_volume_percentile(
    wedge_volume: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """
    Normalize the raw wedge volume to a percentile rank using its own
    trailing distribution.

    Parameters
    ----------
    wedge_volume : pd.Series
        Raw wedge volume series.
    lookback : int
        Number of trading days for the percentile normalization window.

    Returns
    -------
    pd.Series of percentile values (0–100).
    """
    percentiles = pd.Series(index=wedge_volume.index, dtype=float,
                            name="wedge_volume_percentile")

    for i in range(lookback, len(wedge_volume)):
        window = wedge_volume.iloc[max(0, i - lookback):i + 1].dropna()
        if len(window) < 10:
            continue
        current_val = wedge_volume.iloc[i]
        if pd.isna(current_val):
            continue
        # Percentile rank: fraction of historical values that are <= current
        pct = (window < current_val).sum() / len(window) * 100.0
        percentiles.iloc[i] = pct

    return percentiles


# ===========================================================================
# 2. THREE-TIER REGIME CLASSIFIER
# ===========================================================================

def compute_regime_probabilities(
    percentile: float,
    cfg: dict,
) -> Dict[str, float]:
    """
    Compute a probabilistic regime vector [P_panic, P_defense, P_offense]
    based on the wedge volume percentile.

    Uses soft boundaries with interpolation, not hard thresholds.

    Parameters
    ----------
    percentile : float
        Current wedge volume percentile (0–100).
    cfg : dict
        Master config (for threshold values).

    Returns
    -------
    dict with keys "panic", "defense", "offense" summing to ~1.0.
    """
    if pd.isna(percentile):
        return {"panic": 0.0, "defense": 0.0, "offense": 0.0}

    panic_upper = cfg["regime"]["thresholds"]["panic_upper"]          # 5
    defense_upper = cfg["regime"]["thresholds"]["defense_upper"]      # 30
    panic_anchor = cfg["regime"]["thresholds"]["panic_probability_anchor"]  # 8
    offense_anchor = cfg["regime"]["thresholds"]["offense_probability_anchor"]  # 25

    p_panic = 0.0
    p_defense = 0.0
    p_offense = 0.0

    if percentile < panic_upper:
        # Deep in Panic territory — probability based on distance below anchor
        p_panic = min(1.0, max(0.0, (panic_anchor - percentile) / panic_anchor))
        p_defense = 1.0 - p_panic
        p_offense = 0.0
    elif percentile < panic_anchor:
        # Transition zone: panic_upper <= pct < panic_anchor
        # Interpolate between Panic and Defense
        t = (percentile - panic_upper) / max(1, panic_anchor - panic_upper)
        p_panic = max(0.0, 1.0 - t) * 0.6  # Still meaningful panic probability
        p_defense = 1.0 - p_panic
        p_offense = 0.0
    elif percentile < offense_anchor:
        # Core Defense territory: panic_anchor <= pct < offense_anchor
        p_panic = 0.0
        # Interpolate between pure defense and beginning of offense
        t = (percentile - panic_anchor) / max(1, offense_anchor - panic_anchor)
        p_defense = max(0.0, 1.0 - t * 0.5)
        p_offense = 1.0 - p_defense
    elif percentile < defense_upper:
        # Transition zone: offense_anchor <= pct < defense_upper
        t = (percentile - offense_anchor) / max(1, defense_upper - offense_anchor)
        p_panic = 0.0
        p_defense = max(0.0, 0.5 * (1.0 - t))
        p_offense = 1.0 - p_defense
    else:
        # pct >= defense_upper → full Offense
        # Probability rises with distance above defense_upper
        p_panic = 0.0
        p_defense = 0.0
        p_offense = 1.0

    # Normalize to sum to 1.0
    total = p_panic + p_defense + p_offense
    if total > 0:
        p_panic /= total
        p_defense /= total
        p_offense /= total

    return {
        "panic": round(p_panic, 4),
        "defense": round(p_defense, 4),
        "offense": round(p_offense, 4),
    }


def get_dominant_regime(probs: Dict[str, float]) -> str:
    """Return the regime with the highest probability."""
    if not probs or all(v == 0.0 for v in probs.values()):
        return "unknown"
    return max(probs, key=probs.get)


def apply_confirmation_filter(
    regime_series: pd.DataFrame,
    consecutive_days: int = 2,
) -> pd.DataFrame:
    """
    Apply the 2-day confirmation filter to reduce whipsaw.

    A regime transition is only recorded after `consecutive_days`
    consecutive days of the new regime being dominant.

    Parameters
    ----------
    regime_series : pd.DataFrame
        Must have columns: date, dominant_regime (raw, unconfirmed).

    Returns
    -------
    DataFrame with added columns:
        - confirmed_regime: the confirmed (filtered) regime
        - consecutive_days_in_regime: running count
        - regime_confirmed: bool
    """
    regime_series = regime_series.copy().sort_values("date").reset_index(drop=True)

    confirmed = []
    consecutive = []
    is_confirmed = []

    current_confirmed = None
    pending_regime = None
    pending_count = 0

    for i, row in regime_series.iterrows():
        raw = row["dominant_regime"]

        if current_confirmed is None:
            # First observation — accept immediately
            current_confirmed = raw
            pending_regime = raw
            pending_count = 1
        elif raw == current_confirmed:
            # Same as confirmed — reset any pending transition
            pending_regime = raw
            pending_count += 1
        elif raw == pending_regime:
            # Continuing a pending transition
            pending_count += 1
            if pending_count >= consecutive_days:
                # Transition confirmed
                current_confirmed = raw
        else:
            # New regime different from both confirmed and pending
            pending_regime = raw
            pending_count = 1

        # Count consecutive days in the confirmed regime
        if raw == current_confirmed:
            conf_count = pending_count
        else:
            conf_count = 0

        confirmed.append(current_confirmed)
        consecutive.append(conf_count if raw == current_confirmed else pending_count)
        is_confirmed.append(raw == current_confirmed)

    regime_series["confirmed_regime"] = confirmed
    regime_series["consecutive_days_in_regime"] = consecutive
    regime_series["regime_confirmed"] = is_confirmed

    return regime_series


# ===========================================================================
# 3. FAST SHOCK INDICATOR (Section 9.1)
# ===========================================================================

def compute_realized_volatility(
    spy_returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    """
    Compute rolling realized volatility (annualized) from SPY log returns.
    """
    rv = spy_returns.rolling(window=window, min_periods=max(1, window // 2)).std() * np.sqrt(252)
    rv.name = "realized_vol"
    return rv


def compute_fast_shock_indicator(
    vix_series: pd.Series,
    realized_vol: pd.Series,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Compute the VIX / Realized Volatility ratio.
    Flag "Fast Shock Risk = HIGH" when ratio exceeds threshold.

    Parameters
    ----------
    vix_series : pd.Series
        VIX index values (index = date).
    realized_vol : pd.Series
        21-day annualized realized vol of SPY (index = date).
    threshold : float
        Ratio above which Fast Shock Risk is HIGH.

    Returns
    -------
    pd.DataFrame with columns: date, vix_rv_ratio, fast_shock_risk
    """
    # Align on common dates
    combined = pd.DataFrame({
        "vix": vix_series,
        "rv": realized_vol,
    }).dropna()

    # VIX is quoted in percentage points, realized vol is a decimal
    # VIX of 20 means 20% expected vol. RV of 0.15 means 15%.
    # Normalize: convert RV to percentage-point basis for comparison
    combined["rv_pct"] = combined["rv"] * 100.0

    # Handle zero RV to avoid division by zero
    combined["vix_rv_ratio"] = np.where(
        combined["rv_pct"] > 0.5,  # Minimum meaningful RV
        combined["vix"] / combined["rv_pct"],
        np.nan,
    )

    combined["fast_shock_risk"] = np.where(
        combined["vix_rv_ratio"] > threshold,
        "high",
        "low",
    )
    combined["fast_shock_risk"] = combined["fast_shock_risk"].fillna("low")

    result = combined[["vix_rv_ratio", "fast_shock_risk"]].copy()
    result.index.name = "date"
    return result.reset_index()


# ===========================================================================
# 4. MASTER REGIME COMPUTATION — TIES EVERYTHING TOGETHER
# ===========================================================================

def load_sector_prices(conn: sqlite3.Connection, cfg: dict) -> pd.DataFrame:
    """
    Load sector ETF prices from the database and pivot to wide format
    (date × ticker with adjusted close values).
    """
    sector_tickers = cfg["tickers"]["sector_etfs"]
    placeholders = ",".join(["?"] * len(sector_tickers))
    query = f"""
        SELECT date, ticker, adj_close
        FROM prices
        WHERE ticker IN ({placeholders})
        ORDER BY date, ticker
    """
    df = pd.read_sql_query(query, conn, params=sector_tickers)
    if df.empty:
        return pd.DataFrame()

    wide = df.pivot(index="date", columns="ticker", values="adj_close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()

    # Forward-fill any gaps (weekends/holidays already excluded by yfinance)
    wide = wide.ffill()

    return wide


def load_vix_series(conn: sqlite3.Connection) -> pd.Series:
    """Load VIX close prices from the database."""
    query = "SELECT date, close FROM prices WHERE ticker = '^VIX' ORDER BY date"
    df = pd.read_sql_query(query, conn)
    if df.empty:
        return pd.Series(dtype=float, name="vix")
    df["date"] = pd.to_datetime(df["date"])
    series = df.set_index("date")["close"]
    series.name = "vix"
    return series


def load_spy_returns(conn: sqlite3.Connection) -> pd.Series:
    """Load SPY adjusted close and compute log returns."""
    query = "SELECT date, adj_close FROM prices WHERE ticker = 'SPY' ORDER BY date"
    df = pd.read_sql_query(query, conn)
    if df.empty:
        return pd.Series(dtype=float, name="spy_log_return")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    log_ret = np.log(df["adj_close"] / df["adj_close"].shift(1)).dropna()
    log_ret.name = "spy_log_return"
    return log_ret


def compute_daily_regime(
    conn: sqlite3.Connection,
    cfg: dict = None,
    start_date: str = None,
) -> pd.DataFrame:
    """
    Master function: compute the full daily regime state.

    1. Load sector prices → compute log returns → compute wedge volume
    2. Percentile-rank the wedge volume
    3. Compute probabilistic regime vector for each day
    4. Apply 2-day confirmation filter
    5. Compute Fast Shock indicator
    6. Merge everything into a single DataFrame

    Parameters
    ----------
    conn : sqlite3.Connection
    cfg : dict (loaded from config.yaml if None)
    start_date : str (ISO date; results are trimmed to this)

    Returns
    -------
    pd.DataFrame with columns matching the daily regime state JSON spec.
    """
    if cfg is None:
        cfg = load_config()

    wv_window = cfg["regime"]["wedge_volume"]["rolling_window"]       # 63
    pct_lookback = cfg["regime"]["wedge_volume"]["percentile_lookback"]  # 252
    consec_days = cfg["regime"]["confirmation"]["consecutive_days"]    # 2
    rv_window = cfg["regime"]["fast_shock"]["rv_window"]               # 21
    vix_rv_thresh = cfg["regime"]["fast_shock"]["vix_rv_ratio_threshold"]  # 1.5

    # -- Step 1: Load sector prices and compute log returns --
    logger.info("Loading sector ETF prices...")
    sector_wide = load_sector_prices(conn, cfg)
    if sector_wide.empty or sector_wide.shape[1] < 5:
        logger.error("Insufficient sector price data for regime computation.")
        return pd.DataFrame()

    logger.info("Sector prices: %d days × %d sectors", *sector_wide.shape)

    log_returns = compute_log_returns(sector_wide)
    logger.info("Log returns: %d days", len(log_returns))

    # -- Step 2: Compute wedge volume --
    logger.info("Computing wedge volume (window=%d)...", wv_window)
    wedge_vol = compute_wedge_volume_series(log_returns, window=wv_window)
    valid_wv = wedge_vol.dropna()
    logger.info("Wedge volume: %d valid observations", len(valid_wv))

    # -- Step 3: Percentile rank --
    logger.info("Computing percentile rank (lookback=%d)...", pct_lookback)
    percentiles = compute_wedge_volume_percentile(wedge_vol, lookback=pct_lookback)

    # -- Step 4: Regime probabilities for each date --
    logger.info("Computing regime probabilities...")
    regime_records = []
    for date_val, pct_val in percentiles.items():
        if pd.isna(pct_val):
            continue
        probs = compute_regime_probabilities(pct_val, cfg)
        dominant = get_dominant_regime(probs)
        wv_raw = wedge_vol.get(date_val, np.nan)
        regime_records.append({
            "date": date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val),
            "wedge_volume_raw": float(wv_raw) if not pd.isna(wv_raw) else None,
            "wedge_volume_percentile": round(float(pct_val), 2),
            "p_panic": probs["panic"],
            "p_defense": probs["defense"],
            "p_offense": probs["offense"],
            "dominant_regime": dominant,
        })

    if not regime_records:
        logger.error("No regime records computed — insufficient data history.")
        return pd.DataFrame()

    regime_df = pd.DataFrame(regime_records)

    # -- Step 5: Apply confirmation filter --
    logger.info("Applying %d-day confirmation filter...", consec_days)
    regime_df = apply_confirmation_filter(regime_df, consecutive_days=consec_days)

    # -- Step 6: Fast Shock indicator --
    logger.info("Computing Fast Shock indicator...")
    vix = load_vix_series(conn)
    spy_ret = load_spy_returns(conn)
    rv = compute_realized_volatility(spy_ret, window=rv_window)

    if not vix.empty and not rv.empty:
        fast_shock_df = compute_fast_shock_indicator(vix, rv, threshold=vix_rv_thresh)
        fast_shock_df["date"] = fast_shock_df["date"].astype(str)
        regime_df = regime_df.merge(fast_shock_df, on="date", how="left")
    else:
        logger.warning("VIX or SPY data insufficient for Fast Shock indicator.")
        regime_df["vix_rv_ratio"] = np.nan
        regime_df["fast_shock_risk"] = "low"

    # Fill NaN fast shock
    regime_df["fast_shock_risk"] = regime_df["fast_shock_risk"].fillna("low")
    regime_df["vix_rv_ratio"] = regime_df["vix_rv_ratio"].round(4)

    # -- Trim to start_date if provided --
    if start_date:
        regime_df = regime_df[regime_df["date"] >= start_date].reset_index(drop=True)

    logger.info("Regime computation complete: %d days", len(regime_df))
    return regime_df


# ===========================================================================
# 5. REGIME STATE JSON BUILDER
# ===========================================================================

def build_regime_state_json(row: pd.Series) -> dict:
    """
    Build the daily regime state JSON object from a single row
    of the regime DataFrame.
    """
    return {
        "date": row.get("date", ""),
        "wedge_volume_percentile": float(row.get("wedge_volume_percentile", 0.0)),
        "regime_probabilities": {
            "panic": float(row.get("p_panic", 0.0)),
            "defense": float(row.get("p_defense", 0.0)),
            "offense": float(row.get("p_offense", 0.0)),
        },
        "dominant_regime": row.get("confirmed_regime", row.get("dominant_regime", "unknown")),
        "fast_shock_risk": row.get("fast_shock_risk", "low"),
        "vix_rv_ratio": float(row.get("vix_rv_ratio", 0.0)) if pd.notna(row.get("vix_rv_ratio")) else 0.0,
        "consecutive_days_in_regime": int(row.get("consecutive_days_in_regime", 0)),
        "regime_confirmed": bool(row.get("regime_confirmed", False)),
    }


def get_latest_regime_state(conn: sqlite3.Connection, cfg: dict = None) -> dict:
    """Convenience: compute regime and return only the latest day's state."""
    regime_df = compute_daily_regime(conn, cfg)
    if regime_df.empty:
        return {}
    latest = regime_df.iloc[-1]
    return build_regime_state_json(latest)


# ===========================================================================
# 6. DATABASE STORAGE
# ===========================================================================

def store_regime_signals(conn: sqlite3.Connection, regime_df: pd.DataFrame):
    """
    Store daily regime states in the signals table.
    signal_type = 'regime_state', signal_data = JSON string.
    """
    if regime_df.empty:
        return

    now = dt.datetime.utcnow().isoformat()
    rows = []
    for _, row in regime_df.iterrows():
        state_json = json.dumps(build_regime_state_json(row))
        rows.append((row["date"], "regime_state", state_json, now))

    conn.executemany(
        """
        INSERT OR REPLACE INTO signals (date, signal_type, signal_data, created_at)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    logger.info("Stored %d regime signals in database.", len(rows))


def store_wedge_volume(conn: sqlite3.Connection, regime_df: pd.DataFrame):
    """
    Store daily wedge volume percentiles in the signals table
    for charting / historical analysis.
    """
    if regime_df.empty:
        return

    now = dt.datetime.utcnow().isoformat()
    rows = []
    for _, row in regime_df.iterrows():
        data = json.dumps({
            "wedge_volume_raw": row.get("wedge_volume_raw"),
            "wedge_volume_percentile": row.get("wedge_volume_percentile"),
        })
        rows.append((row["date"], "wedge_volume", data, now))

    conn.executemany(
        """
        INSERT OR REPLACE INTO signals (date, signal_type, signal_data, created_at)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    logger.info("Stored %d wedge volume signals in database.", len(rows))


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def run_regime_detection(cfg: dict = None) -> dict:
    """
    Run the full regime detection pipeline:
      1. Connect to the database (prices must already be loaded)
      2. Compute daily regime states
      3. Store in the signals table
      4. Return the latest regime state

    Returns the latest regime state dict.
    """
    if cfg is None:
        cfg = load_config()

    conn = sqlite3.connect(str(DB_PATH))

    regime_df = compute_daily_regime(conn, cfg)
    if regime_df.empty:
        logger.error("Regime detection produced no results.")
        conn.close()
        return {}

    # Store results
    store_regime_signals(conn, regime_df)
    store_wedge_volume(conn, regime_df)

    # Get latest state
    latest_state = build_regime_state_json(regime_df.iloc[-1])

    conn.close()

    logger.info("Latest regime state: %s", json.dumps(latest_state, indent=2))
    return latest_state


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    state = run_regime_detection()
    if state:
        print("\n" + "=" * 60)
        print("LATEST REGIME STATE")
        print("=" * 60)
        print(json.dumps(state, indent=2))
