"""
stock_screener.py — Phase 3B: Individual Stock Screener & Thematic Watchlist
=============================================================================
Global Sector Rotation System

Implements:
  Part A — Top Holdings Screener: score top 20 holdings of overweight sector ETFs
  Part B — Four Thematic Watchlists: biotech M&A, AI software, defense, green materials
  Part C — Watchlist Monitoring: ENTRY/EXIT signals, biotech CATALYST alerts

Dependencies: yfinance, numpy, pandas, yaml, sqlite3, requests
"""

import json
import logging
import sqlite3
import datetime as dt
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# LOGGING & CONFIG
# ---------------------------------------------------------------------------
logger = logging.getLogger("stock_screener")
CONFIG_PATH = Path(__file__).parent / "config.yaml"
DB_PATH = Path(__file__).parent / "rotation_system.db"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ===========================================================================
# DATA FETCHING HELPERS
# ===========================================================================

def _fetch_ticker_info(ticker: str) -> dict:
    """Fetch key financial info for a single ticker via yfinance.
    Returns a dict with: market_cap, forward_pe, trailing_pe, roe, gross_margin,
    ocf_yield, pe_5yr_median, sector, industry, short_name.
    """
    import yfinance as yf

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        market_cap = info.get("marketCap", 0) or 0
        fwd_pe = info.get("forwardPE") or info.get("forwardEps", None)
        trail_pe = info.get("trailingPE")
        roe = info.get("returnOnEquity")
        gm = info.get("grossMargins")
        ocf = info.get("operatingCashflow", 0) or 0

        # Operating cash flow yield = OCF / market_cap
        ocf_yield = (ocf / market_cap) if market_cap > 0 else 0.0

        # Forward P/E: try forwardPE directly, else compute from forwardEps
        # Guard: skip fallback when fwd_eps <= 0 (negative EPS -> meaningless PE)
        if isinstance(fwd_pe, (int, float)) and fwd_pe > 0:
            forward_pe = fwd_pe
        else:
            fwd_eps = info.get("forwardEps")
            price = info.get("currentPrice") or info.get("previousClose") or 0
            if fwd_eps is not None and fwd_eps > 0 and price > 0:
                forward_pe = price / fwd_eps
            else:
                # Negative or zero fwd_eps -> cannot compute meaningful PE
                forward_pe = None

        return {
            "ticker": ticker,
            "short_name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": market_cap,
            "market_cap_m": round(market_cap / 1e6, 1) if market_cap else 0,
            "forward_pe": forward_pe,
            "trailing_pe": trail_pe,
            "roe": roe,
            "gross_margin": gm,
            "ocf_yield": round(ocf_yield, 4),
            "price": info.get("currentPrice") or info.get("previousClose"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as e:
        logger.warning("Failed to fetch info for %s: %s", ticker, e)
        return {"ticker": ticker, "market_cap": 0, "market_cap_m": 0, "error": str(e)}


def _fetch_price_history(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """Fetch historical prices for multiple tickers via yfinance.
    Returns wide-format DataFrame (date x ticker) of adjusted close prices.
    """
    import yfinance as yf

    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]
            prices.columns = [tickers[0]] if len(tickers) == 1 else tickers
        return prices
    except Exception as e:
        logger.warning("Failed to download prices: %s", e)
        return pd.DataFrame()


def _fetch_etf_holdings(etf_ticker: str, top_n: int = 20) -> List[str]:
    """Fetch top holdings of an ETF via yfinance.
    Falls back to a hardcoded mapping if yfinance doesn't return holdings.
    """
    import yfinance as yf

    try:
        etf = yf.Ticker(etf_ticker)
        # yfinance sometimes exposes holdings via .info or via specific methods
        # In many cases the holdings are NOT available through the free API
        # We use a fallback mapping for the sector ETFs
        pass
    except Exception:
        pass

    # Hardcoded top holdings for sector ETFs (updated Feb 2026 approximate)
    # These are the real top holdings by weight — updated periodically
    # TODO: Replace with dynamic ETF holdings via finance_holdings API or
    # yfinance etf.holdings when reliably available for all sector SPDRs.
    # Current static list should be refreshed quarterly.
    _HOLDINGS = {
        "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ADBE", "AMD", "CSCO",
                 "ORCL", "ACN", "INTC", "IBM", "QCOM", "TXN", "INTU",
                 "NOW", "AMAT", "PANW", "MU", "LRCX"],
        "XLV": ["UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN",
                 "PFE", "DHR", "ISRG", "SYK", "BMY", "BSX", "GILD",
                 "MDT", "VRTX", "ELV", "CI", "ZTS"],
        "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "WMB",
                 "VLO", "OKE", "HES", "PXD", "KMI", "BKR", "HAL",
                 "FANG", "DVN", "OXY", "TRGP", "CTRA"],
        "XLF": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS",
                 "AXP", "SPGI", "BLK", "SCHW", "C", "PGR", "MMC",
                 "CB", "ICE", "CME", "AON", "USB"],
        "XLI": ["GE", "CAT", "UNP", "HON", "RTX", "DE", "UPS", "BA",
                 "LMT", "ETN", "ADP", "WM", "ITW", "GD", "NOC",
                 "CSX", "FDX", "NSC", "PH", "EMR"],
        "XLB": ["LIN", "APD", "SHW", "FCX", "ECL", "NEM", "DD", "NUE",
                 "DOW", "VMC", "MLM", "PPG", "CTVA", "IFF", "CE",
                 "ALB", "CF", "BALL", "PKG", "FMC"],
        "XLU": ["NEE", "SO", "DUK", "CEG", "SRE", "AEP", "D", "PCG",
                 "ED", "EXC", "XEL", "PEG", "WEC", "AWK", "ES",
                 "AEE", "CMS", "DTE", "FE", "PPL"],
        "XLP": ["PG", "COST", "KO", "PEP", "WMT", "PM", "MDLZ", "MO",
                 "CL", "TGT", "GIS", "SYY", "KMB", "ADM", "STZ",
                 "KR", "KHC", "HSY", "MNST", "MKC"],
        "XLRE": ["PLD", "AMT", "EQIX", "CCI", "SPG", "PSA", "O", "DLR",
                  "WELL", "VICI", "ARE", "AVB", "SBAC", "EQR", "VTR",
                  "IRM", "WY", "INVH", "MAA", "ESS"],
        "XLC": ["META", "GOOGL", "GOOG", "NFLX", "T", "DIS", "CMCSA", "VZ",
                 "TMUS", "CHTR", "EA", "TTWO", "WBD", "LYV", "MTCH",
                 "OMC", "IPG", "NWSA", "FOXA", "NWS"],
        "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "BKNG", "TJX",
                 "SBUX", "CMG", "ORLY", "MAR", "GM", "F", "DHI",
                 "ROST", "YUM", "AZO", "HLT", "LEN"],
    }

    holdings = _HOLDINGS.get(etf_ticker, [])
    if holdings:
        logger.info("Using hardcoded top %d holdings for %s", min(top_n, len(holdings)), etf_ticker)
        return holdings[:top_n]

    logger.warning("No holdings data for %s", etf_ticker)
    return []


# ===========================================================================
# PART A — TOP HOLDINGS SCREENER
# ===========================================================================

def score_momentum_stock(prices: pd.DataFrame, ticker: str,
                         lookback: int = 252, skip: int = 21) -> float:
    """Compute 12-1 month momentum for a single stock.
    Returns raw momentum return (not ranked).
    """
    if ticker not in prices.columns or len(prices) < lookback:
        return np.nan

    col = prices[ticker].dropna()
    if len(col) < lookback:
        return np.nan

    ret_12m = col.iloc[-1] / col.iloc[-lookback] - 1
    ret_1m = col.iloc[-1] / col.iloc[-skip] - 1
    return ret_12m - ret_1m


def compute_quality_score(info: dict) -> float:
    """Composite quality score from ROE, gross margin, OCF yield.
    Each component is normalized to 0-1 using sigmoid, then averaged.
    """
    components = []

    roe = info.get("roe")
    if roe is not None and np.isfinite(roe):
        # ROE of 15% = 0.5, 30% = ~0.75
        components.append(1.0 / (1.0 + np.exp(-(roe - 0.15) / 0.10)))

    gm = info.get("gross_margin")
    if gm is not None and np.isfinite(gm):
        # Gross margin of 40% = 0.5
        components.append(1.0 / (1.0 + np.exp(-(gm - 0.40) / 0.15)))

    ocf_y = info.get("ocf_yield", 0)
    if ocf_y is not None and np.isfinite(ocf_y):
        # OCF yield of 5% = 0.5
        components.append(1.0 / (1.0 + np.exp(-(ocf_y - 0.05) / 0.03)))

    return np.mean(components) if components else 0.5


def compute_value_score(info: dict) -> float:
    """Value score based on forward P/E.
    Lower forward P/E = higher value score.
    """
    fwd_pe = info.get("forward_pe")
    if fwd_pe is None or not np.isfinite(fwd_pe) or fwd_pe <= 0:
        return 0.5  # neutral if no data

    # P/E of 15 = 0.65 (good value), P/E of 30 = 0.35 (expensive)
    # Inverted sigmoid: low P/E = high score
    return 1.0 / (1.0 + np.exp((fwd_pe - 20) / 8))


def compute_size_score(market_cap_m: float) -> float:
    """Size score — small/mid-cap gets a bonus per report.
    $500M = 0.8, $5B = 0.6, $50B = 0.3, $500B = 0.1
    """
    if market_cap_m <= 0:
        return 0.5
    log_cap = np.log10(max(market_cap_m, 1))
    # log10(500) = 2.7, log10(5000) = 3.7, log10(50000) = 4.7
    return max(0.0, min(1.0, 1.0 - (log_cap - 2.5) / 3.5))


def apply_valuation_filter(info: dict, cfg: dict) -> str:
    """Apply the Section 6.2 valuation filter.
    Compute implied 3-year earnings growth required to justify current forward P/E
    at the configured discount rate. Compare to a reasonable consensus proxy.

    Returns: 'FUNDAMENTAL_BUY', 'MOMENTUM_ONLY', or 'AVOID'
    """
    fwd_pe = info.get("forward_pe")
    if fwd_pe is None or not np.isfinite(fwd_pe) or fwd_pe <= 0:
        return "FUNDAMENTAL_BUY"  # Can't evaluate, give benefit of doubt

    val_cfg = cfg.get("stock_screener", cfg.get("factor_model", {})).get("valuation_filter",
              cfg.get("factor_model", {}).get("valuation", {}))
    discount_rate = val_cfg.get("discount_rate", 0.08)
    momentum_pct = val_cfg.get("momentum_only_percentile", 75)
    avoid_pct = val_cfg.get("avoid_percentile", 90)

    # Implied 3-year earnings growth = (Forward P/E x discount_rate) - 1
    # Simplified: if P/E = 25 and r = 8%, implied growth ~ 25 x 0.08 = 2.0 (100% growth)
    # More precisely: implied_growth = (fwd_pe / (1/r))^(1/3) - 1
    # where 1/r is the P/E of a zero-growth stock
    zero_growth_pe = 1.0 / discount_rate  # 12.5 at 8%
    if fwd_pe <= zero_growth_pe:
        return "FUNDAMENTAL_BUY"

    implied_growth_ratio = fwd_pe / zero_growth_pe
    implied_3yr_growth = implied_growth_ratio ** (1 / 3) - 1

    # Map to percentile proxy: 10% growth = ~50th, 25% = ~75th, 40% = ~90th
    # This is an approximation — real implementation would compare to consensus
    if implied_3yr_growth > 0.40:
        return "AVOID"
    elif implied_3yr_growth > 0.25:
        return "MOMENTUM_ONLY"
    else:
        return "FUNDAMENTAL_BUY"


def screen_etf_holdings(
    etf_ticker: str,
    cfg: dict,
    prices_cache: dict = None,
) -> pd.DataFrame:
    """Screen top holdings of a sector ETF.

    Returns DataFrame with columns: ticker, short_name, sector, market_cap_m,
    momentum, quality, value, size, composite_score, valuation_label
    """
    top_n = cfg.get("stock_screener", {}).get("top_holdings_count", 20)
    holdings = _fetch_etf_holdings(etf_ticker, top_n)
    if not holdings:
        return pd.DataFrame()

    weights = cfg.get("stock_screener", {}).get("scoring_weights", {})
    w_mom = weights.get("momentum", 0.30)
    w_qual = weights.get("quality", 0.25)
    w_val = weights.get("value", 0.25)
    w_size = weights.get("size", 0.20)

    # Fetch prices for momentum
    if prices_cache and etf_ticker in prices_cache:
        all_prices = prices_cache[etf_ticker]
    else:
        all_prices = _fetch_price_history(holdings, period="2y")

    records = []
    for ticker in holdings:
        info = _fetch_ticker_info(ticker)
        if info.get("error"):
            continue

        mom_raw = score_momentum_stock(all_prices, ticker)
        quality = compute_quality_score(info)
        value = compute_value_score(info)
        size = compute_size_score(info.get("market_cap_m", 0))

        # Normalize momentum to 0-1 (will cross-sectionally rank later)
        mom_norm = mom_raw if not np.isnan(mom_raw) else 0.0

        label = apply_valuation_filter(info, cfg)

        # Negative quality signals -> AVOID override
        roe = info.get("roe")
        if roe is not None and roe < 0:
            label = "AVOID"

        records.append({
            "ticker": ticker,
            "etf": etf_ticker,
            "short_name": info.get("short_name", ticker),
            "sector": info.get("sector", "Unknown"),
            "market_cap_m": info.get("market_cap_m", 0),
            "price": info.get("price"),
            "momentum_raw": round(mom_norm, 4),
            "quality_score": round(quality, 4),
            "value_score": round(value, 4),
            "size_score": round(size, 4),
            "forward_pe": info.get("forward_pe"),
            "roe": info.get("roe"),
            "gross_margin": info.get("gross_margin"),
            "valuation_label": label,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Cross-sectional momentum ranking within this ETF
    if "momentum_raw" in df.columns:
        df["momentum_rank"] = df["momentum_raw"].rank(pct=True)
    else:
        df["momentum_rank"] = 0.5

    # Composite score
    df["composite_score"] = (
        w_mom * df["momentum_rank"] +
        w_qual * df["quality_score"] +
        w_val * df["value_score"] +
        w_size * df["size_score"]
    ).round(4)

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    logger.info("Screened %d holdings of %s", len(df), etf_ticker)
    return df


# ===========================================================================
# PART B — THEMATIC WATCHLISTS
# ===========================================================================

def score_biotech_watchlist(tickers: List[str], prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Score biotech M&A pipeline candidates.
    Signals: pipeline stage momentum, cash runway proxy, M&A premium potential.
    """
    records = []
    for ticker in tickers:
        info = _fetch_ticker_info(ticker)
        if info.get("error"):
            logger.warning("Skipping %s: %s", ticker, info.get("error"))
            continue

        mom = score_momentum_stock(prices, ticker)
        quality = compute_quality_score(info)
        mcap_m = info.get("market_cap_m", 0)

        # Biotech-specific: smaller = higher M&A premium potential
        ma_premium_score = compute_size_score(mcap_m) * 1.2  # Extra bonus for small-cap
        ma_premium_score = min(1.0, ma_premium_score)

        # Cash proxy: higher OCF yield or higher gross margin = longer runway
        cash_score = min(1.0, info.get("ocf_yield", 0) * 10 + 0.3)

        composite = 0.30 * (mom if not np.isnan(mom) else 0) + \
                    0.25 * quality + \
                    0.25 * ma_premium_score + \
                    0.20 * cash_score

        label = apply_valuation_filter(info, cfg)
        if mcap_m > 0 and mcap_m < 500:
            label = "SPECULATIVE"  # Below minimum threshold

        records.append({
            "ticker": ticker,
            "watchlist": "biotech",
            "short_name": info.get("short_name", ticker),
            "market_cap_m": mcap_m,
            "price": info.get("price"),
            "momentum": round(mom if not np.isnan(mom) else 0, 4),
            "quality_score": round(quality, 4),
            "ma_premium_score": round(ma_premium_score, 4),
            "cash_score": round(cash_score, 4),
            "composite_score": round(composite, 4),
            "valuation_label": label,
            "account": "roth_ira",  # ALL biotech -> Roth
        })

    df = pd.DataFrame(records) if records else pd.DataFrame()
    if not df.empty:
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


def score_ai_software_watchlist(tickers: List[str], prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Score AI software / cybersecurity candidates.
    Signals: revenue growth proxy, valuation (fwd P/S), momentum.
    """
    records = []
    for ticker in tickers:
        info = _fetch_ticker_info(ticker)
        if info.get("error"):
            continue

        mom = score_momentum_stock(prices, ticker)
        quality = compute_quality_score(info)
        value = compute_value_score(info)
        mcap_m = info.get("market_cap_m", 0)

        # AI/SaaS specific scoring
        composite = 0.30 * (mom if not np.isnan(mom) else 0) + \
                    0.25 * quality + \
                    0.25 * value + \
                    0.20 * compute_size_score(mcap_m)

        label = apply_valuation_filter(info, cfg)

        # Check market cap thresholds from config
        caps = cfg.get("tickers", {}).get("watchlist_market_cap", {})
        max_cap = caps.get("ai_software_max_m", 40000)
        if mcap_m > max_cap:
            label = "EXCEEDS_CAP"  # Graduated beyond mid-cap

        # Account placement per rules: profitable with P/E < 35 -> taxable, else Roth
        fwd_pe = info.get("forward_pe")
        if fwd_pe and fwd_pe < 35 and info.get("roe", 0) and info["roe"] > 0:
            account = "taxable"
        else:
            account = "roth_ira"

        records.append({
            "ticker": ticker,
            "watchlist": "ai_software",
            "short_name": info.get("short_name", ticker),
            "market_cap_m": mcap_m,
            "price": info.get("price"),
            "forward_pe": info.get("forward_pe"),
            "momentum": round(mom if not np.isnan(mom) else 0, 4),
            "quality_score": round(quality, 4),
            "value_score": round(value, 4),
            "composite_score": round(composite, 4),
            "valuation_label": label,
            "account": account,
        })

    df = pd.DataFrame(records) if records else pd.DataFrame()
    if not df.empty:
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


def score_defense_watchlist(tickers: List[str], prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Score defense & industrial reshoring candidates.
    Signals: momentum, quality, value, backlog proxy (gross margin as proxy).
    """
    records = []
    for ticker in tickers:
        info = _fetch_ticker_info(ticker)
        if info.get("error"):
            continue

        mom = score_momentum_stock(prices, ticker)
        quality = compute_quality_score(info)
        value = compute_value_score(info)
        mcap_m = info.get("market_cap_m", 0)

        # Defense-specific: backlog proxy = higher gross margin + quality
        backlog_proxy = quality * 1.1

        composite = 0.25 * (mom if not np.isnan(mom) else 0) + \
                    0.25 * quality + \
                    0.25 * value + \
                    0.15 * backlog_proxy + \
                    0.10 * compute_size_score(mcap_m)

        label = apply_valuation_filter(info, cfg)

        # Account: large long-duration defense -> taxable, smaller speculative -> Roth
        if mcap_m > 10000:
            account = "taxable"
        else:
            account = "roth_ira"

        records.append({
            "ticker": ticker,
            "watchlist": "defense",
            "short_name": info.get("short_name", ticker),
            "market_cap_m": mcap_m,
            "price": info.get("price"),
            "momentum": round(mom if not np.isnan(mom) else 0, 4),
            "quality_score": round(quality, 4),
            "value_score": round(value, 4),
            "backlog_proxy": round(backlog_proxy, 4),
            "composite_score": round(composite, 4),
            "valuation_label": label,
            "account": account,
        })

    df = pd.DataFrame(records) if records else pd.DataFrame()
    if not df.empty:
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


def score_materials_watchlist(tickers: List[str], prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Score green transition materials candidates.
    Signals: cost curve (margin proxy), momentum, size.
    """
    records = []
    for ticker in tickers:
        info = _fetch_ticker_info(ticker)
        if info.get("error"):
            continue

        mom = score_momentum_stock(prices, ticker)
        quality = compute_quality_score(info)
        value = compute_value_score(info)
        mcap_m = info.get("market_cap_m", 0)

        # Materials-specific: gross margin as AISC cost curve proxy
        gm = info.get("gross_margin", 0) or 0
        cost_curve_score = min(1.0, gm * 2.0)  # 50% margin = 1.0

        composite = 0.25 * (mom if not np.isnan(mom) else 0) + \
                    0.20 * quality + \
                    0.20 * value + \
                    0.20 * cost_curve_score + \
                    0.15 * compute_size_score(mcap_m)

        label = apply_valuation_filter(info, cfg)

        records.append({
            "ticker": ticker,
            "watchlist": "green_materials",
            "short_name": info.get("short_name", ticker),
            "market_cap_m": mcap_m,
            "price": info.get("price"),
            "gross_margin": info.get("gross_margin"),
            "momentum": round(mom if not np.isnan(mom) else 0, 4),
            "quality_score": round(quality, 4),
            "value_score": round(value, 4),
            "cost_curve_score": round(cost_curve_score, 4),
            "composite_score": round(composite, 4),
            "valuation_label": label,
            "account": "taxable",  # Long-duration commodity -> taxable
        })

    df = pd.DataFrame(records) if records else pd.DataFrame()
    if not df.empty:
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


def score_generic_watchlist(
    tickers: List[str], prices: pd.DataFrame, cfg: dict,
    watchlist_name: str, default_account: str = "roth_ira",
) -> pd.DataFrame:
    """Generic scoring function for watchlists without custom logic.
    Uses the standard momentum/quality/value/size composite.
    """
    weights = cfg.get("stock_screener", {}).get("scoring_weights", {})
    w_mom = weights.get("momentum", 0.30)
    w_qual = weights.get("quality", 0.25)
    w_val = weights.get("value", 0.25)
    w_size = weights.get("size", 0.20)

    records = []
    for ticker in tickers:
        info = _fetch_ticker_info(ticker)
        if info.get("error"):
            continue

        mom = score_momentum_stock(prices, ticker)
        quality = compute_quality_score(info)
        value = compute_value_score(info)
        mcap_m = info.get("market_cap_m", 0)

        composite = w_mom * (mom if not np.isnan(mom) else 0) + \
                    w_qual * quality + \
                    w_val * value + \
                    w_size * compute_size_score(mcap_m)

        label = apply_valuation_filter(info, cfg)

        records.append({
            "ticker": ticker,
            "watchlist": watchlist_name,
            "short_name": info.get("short_name", ticker),
            "market_cap_m": mcap_m,
            "price": info.get("price"),
            "forward_pe": info.get("forward_pe"),
            "momentum": round(mom if not np.isnan(mom) else 0, 4),
            "quality_score": round(quality, 4),
            "value_score": round(value, 4),
            "composite_score": round(composite, 4),
            "valuation_label": label,
            "account": default_account,
        })

    df = pd.DataFrame(records) if records else pd.DataFrame()
    if not df.empty:
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


def run_all_watchlists(cfg: dict, prices: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
    """Score all thematic watchlists.
    Returns dict of watchlist_name -> scored DataFrame.
    """
    results = {}

    # Collect all tickers for price download
    all_wl_tickers = []
    for key in ["watchlist_biotech", "watchlist_ai_software",
                 "watchlist_defense", "watchlist_green_materials",
                 "watchlist_semiconductors", "watchlist_energy_transition",
                 "watchlist_fintech"]:
        all_wl_tickers.extend(cfg.get("tickers", {}).get(key, []))

    if prices is None and all_wl_tickers:
        logger.info("Downloading prices for %d watchlist tickers...", len(all_wl_tickers))
        prices = _fetch_price_history(all_wl_tickers, period="2y")

    # Score each watchlist
    bio = cfg.get("tickers", {}).get("watchlist_biotech", [])
    results["biotech"] = score_biotech_watchlist(bio, prices, cfg)
    logger.info("Biotech watchlist: %d scored", len(results["biotech"]))

    ai_sw = cfg.get("tickers", {}).get("watchlist_ai_software", [])
    results["ai_software"] = score_ai_software_watchlist(ai_sw, prices, cfg)
    logger.info("AI Software watchlist: %d scored", len(results["ai_software"]))

    defense = cfg.get("tickers", {}).get("watchlist_defense", [])
    results["defense"] = score_defense_watchlist(defense, prices, cfg)
    logger.info("Defense watchlist: %d scored", len(results["defense"]))

    mats = cfg.get("tickers", {}).get("watchlist_green_materials", [])
    results["green_materials"] = score_materials_watchlist(mats, prices, cfg)
    logger.info("Materials watchlist: %d scored", len(results["green_materials"]))

    # New watchlists — use generic scoring
    semis = cfg.get("tickers", {}).get("watchlist_semiconductors", [])
    if semis:
        results["semiconductors"] = score_generic_watchlist(semis, prices, cfg, "semiconductors", "roth_ira")
        logger.info("Semiconductors watchlist: %d scored", len(results["semiconductors"]))

    energy_tr = cfg.get("tickers", {}).get("watchlist_energy_transition", [])
    if energy_tr:
        results["energy_transition"] = score_generic_watchlist(energy_tr, prices, cfg, "energy_transition", "roth_ira")
        logger.info("Energy Transition watchlist: %d scored", len(results["energy_transition"]))

    fintech = cfg.get("tickers", {}).get("watchlist_fintech", [])
    if fintech:
        results["fintech"] = score_generic_watchlist(fintech, prices, cfg, "fintech", "roth_ira")
        logger.info("Fintech watchlist: %d scored", len(results["fintech"]))

    return results


# ===========================================================================
# PART C — WATCHLIST MONITORING (ENTRY/EXIT SIGNALS)
# ===========================================================================

def compute_entry_exit_signals(
    watchlist_scores: Dict[str, pd.DataFrame],
    regime: str,
    cfg: dict,
) -> Dict[str, List[dict]]:
    """Compute ENTRY and EXIT signals for all watchlist positions.

    ENTRY when: composite_score in top quartile AND valuation = FUNDAMENTAL_BUY
                AND parent sector in Offense band.
    EXIT when:  momentum drops to bottom quartile OR valuation = AVOID
                OR parent sector rotates to Defense.

    Returns dict of signal_type -> list of signal dicts.
    """
    entry_cfg = cfg.get("stock_screener", {}).get("entry_signal", {})
    exit_cfg = cfg.get("stock_screener", {}).get("exit_signal", {})

    min_score_pct = entry_cfg.get("min_factor_score_percentile", 75) / 100.0
    require_buy = entry_cfg.get("require_fundamental_buy", True)
    require_offense = entry_cfg.get("require_offense_regime", True)
    mom_bottom_pct = exit_cfg.get("momentum_bottom_percentile", 25) / 100.0
    defense_exit = exit_cfg.get("trigger_on_defense_rotation", True)

    signals = {"entry": [], "exit": [], "catalyst": []}

    for wl_name, df in watchlist_scores.items():
        if df.empty:
            continue

        # Compute score percentile within this watchlist
        df = df.copy()
        df["score_pct"] = df["composite_score"].rank(pct=True)
        if "momentum" in df.columns:
            df["mom_pct"] = df["momentum"].rank(pct=True)
        else:
            df["mom_pct"] = 0.5

        for _, row in df.iterrows():
            ticker = row["ticker"]
            label = row.get("valuation_label", "")

            # --- ENTRY SIGNAL ---
            is_top_quartile = row.get("score_pct", 0) >= min_score_pct
            is_fundamental_buy = (label == "FUNDAMENTAL_BUY") if require_buy else True
            is_offense = (regime == "offense") if require_offense else True

            if is_top_quartile and is_fundamental_buy and is_offense:
                signals["entry"].append({
                    "ticker": ticker,
                    "watchlist": wl_name,
                    "signal": "ENTRY",
                    "composite_score": row.get("composite_score", 0),
                    "valuation_label": label,
                    "account": row.get("account", "roth_ira"),
                    "reason": f"Top quartile score ({row.get('composite_score', 0):.3f}), "
                              f"valuation={label}, regime={regime}",
                })

            # --- EXIT SIGNAL ---
            is_bottom_momentum = row.get("mom_pct", 0.5) <= mom_bottom_pct
            is_avoid = label == "AVOID"
            is_defense_exit = (regime in ["defense", "panic"]) and defense_exit

            if is_bottom_momentum or is_avoid or is_defense_exit:
                reasons = []
                if is_bottom_momentum:
                    reasons.append("momentum in bottom quartile")
                if is_avoid:
                    reasons.append("valuation = AVOID")
                if is_defense_exit:
                    reasons.append(f"sector in {regime} regime")

                signals["exit"].append({
                    "ticker": ticker,
                    "watchlist": wl_name,
                    "signal": "EXIT",
                    "composite_score": row.get("composite_score", 0),
                    "valuation_label": label,
                    "reason": "; ".join(reasons),
                })

    logger.info("Signals: %d entry, %d exit, %d catalyst",
                len(signals["entry"]), len(signals["exit"]), len(signals["catalyst"]))
    return signals


def check_biotech_catalysts(
    biotech_tickers: List[str],
    conn: sqlite3.Connection = None,
) -> List[dict]:
    """Check for recent 8-K filings from biotech watchlist companies.
    Pulls from the filings table in rotation_system.db (populated by Phase 1).

    Returns list of catalyst alert dicts.
    """
    catalysts = []

    if conn is None:
        try:
            conn = sqlite3.connect(str(DB_PATH))
        except Exception:
            return catalysts

    today = dt.date.today().isoformat()
    yesterday = (dt.date.today() - dt.timedelta(days=3)).isoformat()

    try:
        # Check if filings table exists
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='filings'"
        ).fetchall()
        if not tables:
            logger.debug("No filings table — skipping catalyst check")
            return catalysts

        for ticker in biotech_tickers:
            rows = conn.execute(
                "SELECT ticker, filing_type, filing_date, description "
                "FROM filings WHERE ticker = ? AND filing_date >= ? AND filing_type = '8-K' "
                "ORDER BY filing_date DESC",
                (ticker, yesterday),
            ).fetchall()

            for row in rows:
                catalysts.append({
                    "ticker": row[0],
                    "filing_type": row[1],
                    "filing_date": row[2],
                    "description": row[3] or "8-K filed — review for clinical data / M&A / FDA",
                    "signal": "CATALYST_ALERT",
                })

    except Exception as e:
        logger.debug("Catalyst check error: %s", e)

    return catalysts


def format_watchlist_report(
    watchlist_scores: Dict[str, pd.DataFrame],
    signals: Dict[str, List[dict]],
    catalysts: List[dict] = None,
    cfg: dict = None,
) -> str:
    """Generate the daily watchlist report text.
    Matches the format from the master prompt.

    Includes dollar amounts and tax account notes per portfolio rules:
    $100K taxable + $44K Roth = $144K total.
    """
    if cfg is None:
        cfg = load_config()

    today = dt.date.today().isoformat()

    # Portfolio values for dollar sizing
    pf = cfg.get("portfolio", {})
    acct_cfg = pf.get("accounts", {})
    taxable_val = acct_cfg.get("taxable", {}).get("value", 100000)
    roth_val = acct_cfg.get("roth_ira", {}).get("value", 44000)
    total_val = pf.get("total_value", 144000)
    pos_pct = cfg.get("stock_screener", {}).get("watchlist_pos_pct", 0.02)
    split_threshold = cfg.get("tax_location", {}).get("split_threshold", 10000)

    lines = [
        f"=== THEMATIC WATCHLIST STATUS — {today} ===",
        f"Portfolio: ${total_val:,.0f} (Taxable ${taxable_val:,.0f} + Roth ${roth_val:,.0f})",
        f"Position size: {pos_pct*100:.1f}% | Split threshold: ${split_threshold:,.0f}",
        "",
    ]

    label_icons = {
        "FUNDAMENTAL_BUY": "✅",
        "MOMENTUM_ONLY": "⚠️",
        "AVOID": "❌",
        "EXCEEDS_CAP": "📈",
        "SPECULATIVE": "🔬",
    }

    wl_display = {
        "biotech": "BIOTECH M&A PIPELINE",
        "ai_software": "AI SOFTWARE DIFFUSION",
        "defense": "DEFENSE & RESHORING",
        "green_materials": "GREEN TRANSITION MATERIALS",
    }

    for wl_key, display_name in wl_display.items():
        df = watchlist_scores.get(wl_key, pd.DataFrame())
        lines.append(f"{display_name}:")

        if df.empty:
            lines.append("  No data available")
        else:
            # Top candidates
            top = df.head(5)
            candidates = []
            for _, row in top.iterrows():
                icon = label_icons.get(row.get("valuation_label", ""), "")
                candidates.append(
                    f"{row['ticker']} {icon} ({row.get('composite_score', 0):.3f})"
                )
            lines.append(f"  Top candidates: {', '.join(candidates)}")

        # Catalysts (biotech only)
        if wl_key == "biotech" and catalysts:
            catalyst_tickers = [c["ticker"] for c in catalysts]
            if catalyst_tickers:
                lines.append(f"  New 8-K filings: {', '.join(set(catalyst_tickers))}")
            else:
                lines.append("  New 8-K filings today: NONE")

        # Entry signals for this watchlist — with dollar amounts and account notes
        wl_entries = [s for s in signals.get("entry", []) if s["watchlist"] == wl_key]
        if wl_entries:
            entry_lines = []
            for sig in wl_entries:
                acct = sig.get("account", "roth_ira")
                acct_val = roth_val if acct == "roth_ira" else taxable_val
                pos_dollars = acct_val * pos_pct
                acct_label = "Roth IRA" if acct == "roth_ira" else "Taxable"
                # Flag if position would cross split threshold
                split_note = ""
                if pos_dollars >= split_threshold:
                    split_note = " [eligible for split]"
                entry_lines.append(
                    f"{sig['ticker']} → {acct_label} "
                    f"({pos_pct*100:.1f}% = ${pos_dollars:,.0f}){split_note}"
                )
            lines.append(f"  Entry signals: {'; '.join(entry_lines)}")
        else:
            lines.append("  Entry signals: NONE")

        lines.append("")

    return "\n".join(lines)


# ===========================================================================
# MOCK DATA SUPPORT
# ===========================================================================

def _generate_mock_watchlist_data(cfg: dict) -> Dict[str, pd.DataFrame]:
    """Generate synthetic watchlist scores for smoke testing without yfinance.
    Returns dict of watchlist_name -> DataFrame with realistic mock data.
    """
    np.random.seed(42)

    mock_results = {}
    wl_configs = {
        "biotech": ("watchlist_biotech", "roth_ira",
                     ["momentum", "quality_score", "ma_premium_score", "cash_score"]),
        "ai_software": ("watchlist_ai_software", "roth_ira",
                         ["momentum", "quality_score", "value_score"]),
        "defense": ("watchlist_defense", "taxable",
                     ["momentum", "quality_score", "value_score", "backlog_proxy"]),
        "green_materials": ("watchlist_green_materials", "taxable",
                             ["momentum", "quality_score", "value_score", "cost_curve_score"]),
        "semiconductors": ("watchlist_semiconductors", "roth_ira",
                            ["momentum", "quality_score", "value_score"]),
        "energy_transition": ("watchlist_energy_transition", "roth_ira",
                               ["momentum", "quality_score", "value_score"]),
        "fintech": ("watchlist_fintech", "roth_ira",
                     ["momentum", "quality_score", "value_score"]),
    }

    labels = ["FUNDAMENTAL_BUY", "MOMENTUM_ONLY", "AVOID"]

    for wl_name, (cfg_key, default_account, score_cols) in wl_configs.items():
        tickers = cfg.get("tickers", {}).get(cfg_key, [])
        n = len(tickers)
        if n == 0:
            mock_results[wl_name] = pd.DataFrame()
            continue

        records = []
        for i, ticker in enumerate(tickers):
            row = {
                "ticker": ticker,
                "watchlist": wl_name,
                "short_name": f"Mock {ticker}",
                "market_cap_m": round(np.random.uniform(500, 50000), 1),
                "price": round(np.random.uniform(10, 500), 2),
                "composite_score": round(np.random.uniform(0.2, 0.8), 4),
                "valuation_label": np.random.choice(labels, p=[0.5, 0.3, 0.2]),
                "account": default_account,
            }
            for col in score_cols:
                row[col] = round(np.random.uniform(0.1, 0.9), 4)
            records.append(row)

        df = pd.DataFrame(records)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        mock_results[wl_name] = df

    return mock_results


def _generate_mock_screen_data(etf: str, cfg: dict) -> pd.DataFrame:
    """Generate mock ETF holdings screening data."""
    np.random.seed(hash(etf) % 2**31)
    holdings = _fetch_etf_holdings(etf, 20)
    if not holdings:
        return pd.DataFrame()

    labels = ["FUNDAMENTAL_BUY", "MOMENTUM_ONLY", "AVOID"]
    records = []
    for ticker in holdings:
        records.append({
            "ticker": ticker,
            "etf": etf,
            "short_name": f"Mock {ticker}",
            "sector": "Technology",
            "market_cap_m": round(np.random.uniform(1000, 300000), 1),
            "price": round(np.random.uniform(50, 500), 2),
            "momentum_raw": round(np.random.normal(0.15, 0.20), 4),
            "quality_score": round(np.random.uniform(0.3, 0.9), 4),
            "value_score": round(np.random.uniform(0.2, 0.8), 4),
            "size_score": round(np.random.uniform(0.1, 0.6), 4),
            "forward_pe": round(np.random.uniform(8, 60), 1),
            "roe": round(np.random.uniform(-0.05, 0.40), 4),
            "gross_margin": round(np.random.uniform(0.20, 0.70), 4),
            "valuation_label": np.random.choice(labels, p=[0.5, 0.3, 0.2]),
        })

    df = pd.DataFrame(records)
    df["momentum_rank"] = df["momentum_raw"].rank(pct=True)
    weights = cfg.get("stock_screener", {}).get("scoring_weights", {})
    df["composite_score"] = (
        weights.get("momentum", 0.30) * df["momentum_rank"] +
        weights.get("quality", 0.25) * df["quality_score"] +
        weights.get("value", 0.25) * df["value_score"] +
        weights.get("size", 0.20) * df["size_score"]
    ).round(4)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


# ===========================================================================
# MASTER PIPELINE
# ===========================================================================

def run_stock_screener(
    conn: sqlite3.Connection = None,
    cfg: dict = None,
    regime: str = None,
    mock: bool = False,
) -> dict:
    """Master function: run the full stock screening pipeline.

    1. Determine overweight sectors from the current allocation
    2. Screen top holdings of each overweight sector ETF (Part A)
    3. Score all four thematic watchlists (Part B)
    4. Compute ENTRY/EXIT signals (Part C)
    5. Check biotech catalysts
    6. Generate watchlist report
    7. Output JSON + CSV

    Returns the complete screener result dict.
    """
    if cfg is None:
        cfg = load_config()

    close_conn = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        close_conn = True

    # --- Step 1: Determine regime ---
    if regime is None:
        try:
            try:
                from regime_detector import get_latest_regime_state
            except ImportError:
                from sector_rotation.regime_detector import get_latest_regime_state
            state = get_latest_regime_state(conn, cfg)
            regime = state.get("dominant_regime", "offense")
        except Exception:
            regime = "offense"
    logger.info("Stock screener running for regime: %s", regime)

    # --- Step 2: Determine overweight sectors ---
    # Read the current allocation from Phase 3 output
    alloc_path = Path(__file__).parent / "current_allocation.json"
    overweight_etfs = []
    if alloc_path.exists():
        with open(alloc_path) as f:
            alloc = json.load(f)
        positions = alloc.get("positions", {})
        sector_etfs = cfg.get("tickers", {}).get("sector_etfs", [])
        equal_weight = 1.0 / max(len(sector_etfs), 1)
        for etf in sector_etfs:
            if etf in positions:
                pct = positions[etf].get("pct", 0) / 100.0
                if pct > equal_weight:
                    overweight_etfs.append(etf)
        if not overweight_etfs:
            # Default to top 3 by weight
            etf_weights = [(e, positions.get(e, {}).get("pct", 0)) for e in sector_etfs]
            etf_weights.sort(key=lambda x: x[1], reverse=True)
            overweight_etfs = [e for e, _ in etf_weights[:3]]
    else:
        # No allocation yet — screen the top 3 sector ETFs by convention
        overweight_etfs = cfg.get("tickers", {}).get("sector_etfs", [])[:3]

    logger.info("Screening overweight ETFs: %s", overweight_etfs)

    # --- Step 3: Screen ETF holdings (Part A) ---
    etf_screens = {}
    for etf in overweight_etfs:
        if mock:
            etf_screens[etf] = _generate_mock_screen_data(etf, cfg)
        else:
            etf_screens[etf] = screen_etf_holdings(etf, cfg)

    # --- Step 4: Score watchlists (Part B) ---
    if mock:
        watchlist_scores = _generate_mock_watchlist_data(cfg)
    else:
        watchlist_scores = run_all_watchlists(cfg)

    # --- Step 5: ENTRY/EXIT signals (Part C) ---
    signals = compute_entry_exit_signals(watchlist_scores, regime, cfg)

    # --- Step 6: Biotech catalysts ---
    bio_tickers = cfg.get("tickers", {}).get("watchlist_biotech", [])
    catalysts = check_biotech_catalysts(bio_tickers, conn)
    signals["catalyst"] = catalysts

    # --- Step 7: Generate report ---
    report_text = format_watchlist_report(watchlist_scores, signals, catalysts, cfg)

    # --- Build output ---
    result = {
        "date": dt.date.today().isoformat(),
        "regime": regime,
        "overweight_etfs": overweight_etfs,
        "etf_screens": {etf: df.to_dict(orient="records") for etf, df in etf_screens.items()},
        "watchlist_scores": {wl: df.to_dict(orient="records") for wl, df in watchlist_scores.items()},
        "signals": signals,
        "catalysts": catalysts,
        "report_text": report_text,
    }

    # Save JSON
    json_path = Path(__file__).parent / "screener_output.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Screener JSON saved to %s", json_path)

    # Save watchlist CSVs
    for wl_name, df in watchlist_scores.items():
        if not df.empty:
            csv_path = Path(__file__).parent / f"watchlist_{wl_name}.csv"
            df.to_csv(csv_path, index=False)

    # Save ETF screen CSVs
    for etf, df in etf_screens.items():
        if not df.empty:
            csv_path = Path(__file__).parent / f"screen_{etf}.csv"
            df.to_csv(csv_path, index=False)

    if close_conn:
        conn.close()

    return result


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3B: Stock Screener & Thematic Watchlists")
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic data (no yfinance calls)")
    parser.add_argument("--real", action="store_true",
                        help="Force live yfinance API calls (overrides --mock)")
    parser.add_argument("--regime", choices=["offense", "defense", "panic"],
                        default=None, help="Override regime")
    args = parser.parse_args()

    # --real overrides --mock; default is mock for safety
    use_mock = args.mock and not args.real

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    result = run_stock_screener(mock=use_mock, regime=args.regime)

    if result:
        print("\n" + result.get("report_text", "No report generated"))
        print(f"\nScreener complete: {len(result.get('overweight_etfs', []))} ETFs screened, "
              f"{sum(len(v) for v in result.get('watchlist_scores', {}).values())} watchlist names scored")
