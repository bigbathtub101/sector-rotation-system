"""
universe_builder.py — Dynamic Stock Universe Builder
=====================================================
Global Sector Rotation System

Replaces hardcoded watchlists (watchlist_ai_software, watchlist_biotech, etc.)
in config.yaml with a dynamic stock universe that refreshes weekly.

The :class:`UniverseBuilder` class screens stocks by sector, industry, market cap,
and average volume using the yfinance screener API, then stores results in SQLite.
A weekly staleness check prevents unnecessary API calls.

SQLite tables created:
    dynamic_universe     — per-ticker-theme rows with screening metadata
    universe_metadata    — key/value store for bookkeeping (e.g. last_refresh_ts)

Usage::

    import yaml
    from sector_rotation.universe_builder import UniverseBuilder

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    builder = UniverseBuilder(db_path="rotation_system.db", config=config)
    universe_df = builder.build_universe()
    biotech_tickers = builder.get_watchlist("biotech")

CLI::

    python -m sector_rotation.universe_builder
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_DAYS: int = 7
_SCREEN_PAGE_SIZE: int = 250
_RATE_LIMIT_SLEEP: float = 1.0
_MAX_TICKERS_PER_THEME: int = 500

DEFAULT_THEMES: Dict[str, Dict[str, Any]] = {
    "biotech": {
        "description": "Healthcare / Biotechnology — small-to-mid cap M&A candidates",
        "sector": "Healthcare",
        "industry": "Biotechnology",
        "min_mcap": 500e6,
        "max_mcap": 8e9,
        "min_volume": 100_000,
        "region": "us",
    },
    "ai_software": {
        "description": "Technology / Software — mid-cap AI and cloud names",
        "sector": "Technology",
        "industries": [
            "Software—Application",
            "Software—Infrastructure",
            "Information Technology Services",
        ],
        "min_mcap": 3e9,
        "max_mcap": 40e9,
        "min_volume": 200_000,
        "region": "us",
    },
    "defense": {
        "description": "Industrials / Aerospace & Defense — NATO spend beneficiaries",
        "sector": "Industrials",
        "industry": "Aerospace & Defense",
        "min_mcap": 1e9,
        "max_mcap": None,
        "min_volume": 100_000,
        "region": "us",
    },
    "green_materials": {
        "description": "Basic Materials — lithium, copper, rare earth transition metals",
        "sector": "Basic Materials",
        "industries": [
            "Other Industrial Metals & Mining",
            "Copper",
            "Aluminum",
            "Steel",
            "Gold",
            "Silver",
            "Specialty Chemicals",
        ],
        "min_mcap": 500e6,
        "max_mcap": None,
        "min_volume": 50_000,
        "region": "us",
    },
    "semiconductors": {
        "description": "Technology / Semiconductors — equipment and fabless leaders",
        "sector": "Technology",
        "industries": [
            "Semiconductors",
            "Semiconductor Equipment & Materials",
        ],
        "min_mcap": 2e9,
        "max_mcap": None,
        "min_volume": 100_000,
        "region": "us",
    },
    "energy_transition": {
        "description": "Energy + Utilities — renewable and nuclear power players",
        "sectors": ["Energy", "Utilities"],
        "industries": [
            "Solar",
            "Utilities—Renewable",
            "Utilities—Independent Power Producers",
            "Oil & Gas Midstream",
            "Oil & Gas E&P",
        ],
        "min_mcap": 500e6,
        "max_mcap": None,
        "min_volume": 75_000,
        "region": "us",
    },
    "fintech": {
        "description": "Financial Services + Technology / Fintech — digital finance disruptors",
        "sectors": ["Financial Services", "Technology"],
        "industries": [
            "Credit Services",
            "Capital Markets",
            "Banks—Regional",
            "Insurance—Specialty",
            "Software—Application",
        ],
        "min_mcap": 1e9,
        "max_mcap": None,
        "min_volume": 150_000,
        "region": "us",
    },
}

_DDL_DYNAMIC_UNIVERSE = """
CREATE TABLE IF NOT EXISTS dynamic_universe (
    ticker       TEXT    NOT NULL,
    theme        TEXT    NOT NULL,
    sector       TEXT,
    industry     TEXT,
    market_cap   REAL,
    avg_volume   REAL,
    last_updated TEXT,
    PRIMARY KEY (ticker, theme)
);
"""

_DDL_UNIVERSE_METADATA = """
CREATE TABLE IF NOT EXISTS universe_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class UniverseBuilder:
    """Build and maintain a dynamic stock universe stored in SQLite."""

    def __init__(self, db_path: str | Path, config: dict) -> None:
        self._db_path = Path(db_path)
        self._config = config
        self._themes: Dict[str, Dict[str, Any]] = self._merge_theme_config()
        self._init_db()

    def build_universe(self, force: bool = False) -> pd.DataFrame:
        if force or self._needs_refresh():
            logger.info("Building dynamic universe (force=%s)...", force)
            all_frames: List[pd.DataFrame] = []
            for theme_name, theme_cfg in self._themes.items():
                logger.info("Processing theme: %s", theme_name)
                try:
                    df = self._build_thematic_universe(theme_cfg)
                    df["theme"] = theme_name
                    if not df.empty:
                        all_frames.append(df)
                        logger.info("  theme=%s  tickers=%d", theme_name, len(df))
                    else:
                        logger.warning("  theme=%s returned 0 tickers", theme_name)
                except Exception as exc:
                    logger.error("Failed to build theme '%s': %s", theme_name, exc, exc_info=True)
            if all_frames:
                combined = pd.concat(all_frames, ignore_index=True)
                self._store_universe(combined)
                self._set_metadata("last_refresh_ts", datetime.now(timezone.utc).isoformat())
                logger.info("Universe build complete — %d total rows across %d themes.", len(combined), combined["theme"].nunique())
            else:
                logger.warning("No themes produced results; universe NOT updated.")
        else:
            logger.info("Universe is fresh (last refresh < %d days ago). Loading from DB.", REFRESH_INTERVAL_DAYS)
        return self._load_universe()

    def get_watchlist(self, theme: str) -> List[str]:
        df = self._load_universe()
        if df.empty:
            return []
        mask = df["theme"] == theme
        return sorted(df.loc[mask, "ticker"].tolist())

    def _needs_refresh(self) -> bool:
        ts_str = self._get_metadata("last_refresh_ts")
        if ts_str is None:
            return True
        try:
            last_ts = datetime.fromisoformat(ts_str)
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - last_ts
            return age > timedelta(days=REFRESH_INTERVAL_DAYS)
        except (ValueError, TypeError):
            return True

    def _screen_by_sector(self, sector: str, industry: Optional[str] = None, min_mcap: float = 500e6, max_mcap: Optional[float] = None, min_volume: float = 100_000, region: str = "us") -> List[str]:
        try:
            from yfinance.screener.screener import screen
            from yfinance import EquityQuery
        except ImportError as exc:
            logger.error("yfinance not available: %s", exc)
            return []
        clauses = [
            EquityQuery("eq", ["sector", sector]),
            EquityQuery("eq", ["region", region]),
            EquityQuery("gt", ["avgdailyvol3m", min_volume]),
        ]
        if min_mcap and max_mcap:
            clauses.append(EquityQuery("btwn", ["intradaymarketcap", min_mcap, max_mcap]))
        elif min_mcap:
            clauses.append(EquityQuery("gt", ["intradaymarketcap", min_mcap]))
        elif max_mcap:
            clauses.append(EquityQuery("lt", ["intradaymarketcap", max_mcap]))
        query = EquityQuery("and", clauses) if len(clauses) > 1 else clauses[0]
        tickers: List[str] = []
        offset = 0
        while len(tickers) < _MAX_TICKERS_PER_THEME:
            try:
                result = screen(query, offset=offset, size=_SCREEN_PAGE_SIZE)
            except Exception as exc:
                logger.warning("screener call failed (sector=%s, offset=%d): %s", sector, offset, exc)
                break
            quotes = result.get("quotes", [])
            if not quotes:
                break
            for q in quotes:
                sym = q.get("symbol")
                if sym:
                    tickers.append(sym)
            total = result.get("total", 0)
            offset += len(quotes)
            if offset >= total or len(quotes) < _SCREEN_PAGE_SIZE:
                break
            time.sleep(_RATE_LIMIT_SLEEP)
        seen: set = set()
        unique: List[str] = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    def _fetch_ticker_metadata(self, tickers: List[str]) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            logger.error("yfinance not available: %s", exc)
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for sym in tickers:
            try:
                info = yf.Ticker(sym).info or {}
                rows.append({"ticker": sym, "sector": info.get("sector"), "industry": info.get("industry"), "market_cap": info.get("marketCap"), "avg_volume": info.get("averageDailyVolume3Month") or info.get("averageVolume")})
                time.sleep(0.1)
            except Exception as exc:
                logger.debug("Could not fetch info for %s: %s", sym, exc)
        if not rows:
            return pd.DataFrame(columns=["ticker", "sector", "industry", "market_cap", "avg_volume"])
        return pd.DataFrame(rows)

    def _build_thematic_universe(self, theme_config: dict) -> pd.DataFrame:
        min_mcap: float = theme_config.get("min_mcap", 500e6)
        max_mcap: Optional[float] = theme_config.get("max_mcap")
        min_volume: float = theme_config.get("min_volume", 100_000)
        region: str = theme_config.get("region", "us")
        raw_sectors = theme_config.get("sectors") or ([theme_config["sector"]] if "sector" in theme_config else [])
        if not raw_sectors:
            logger.warning("theme_config has no 'sector' or 'sectors' key; skipping.")
            return pd.DataFrame()
        raw_industries: List[str] = theme_config.get("industries") or ([theme_config["industry"]] if "industry" in theme_config else [])
        all_tickers: List[str] = []
        for sector in raw_sectors:
            tickers = self._screen_by_sector(sector=sector, min_mcap=min_mcap, max_mcap=max_mcap, min_volume=min_volume, region=region)
            all_tickers.extend(tickers)
        seen: set = set()
        unique_tickers: List[str] = []
        for t in all_tickers:
            if t not in seen:
                seen.add(t)
                unique_tickers.append(t)
        if not unique_tickers:
            return pd.DataFrame()
        meta_df = self._fetch_ticker_metadata(unique_tickers)
        if meta_df.empty:
            now_iso = datetime.now(timezone.utc).isoformat()
            return pd.DataFrame({"ticker": unique_tickers, "sector": None, "industry": None, "market_cap": None, "avg_volume": None, "last_updated": now_iso})
        if raw_industries:
            lower_industries = [i.lower() for i in raw_industries]
            def _industry_matches(ind: Optional[str]) -> bool:
                if not ind:
                    return False
                ind_lower = ind.lower()
                return any(target in ind_lower for target in lower_industries)
            filtered = meta_df[meta_df["industry"].apply(_industry_matches)].copy()
            if filtered.empty:
                filtered = meta_df.copy()
        else:
            filtered = meta_df.copy()
        filtered["last_updated"] = datetime.now(timezone.utc).isoformat()
        return filtered.reset_index(drop=True)

    def _store_universe(self, df: pd.DataFrame) -> None:
        required = {"ticker", "theme", "sector", "industry", "market_cap", "avg_volume", "last_updated"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"DataFrame missing columns: {missing}")
        with sqlite3.connect(self._db_path) as conn:
            themes = df["theme"].unique().tolist()
            placeholders = ",".join("?" * len(themes))
            conn.execute(f"DELETE FROM dynamic_universe WHERE theme IN ({placeholders})", themes)
            df[list(required)].to_sql("dynamic_universe", conn, if_exists="append", index=False, method="multi")
            conn.commit()

    def _load_universe(self) -> pd.DataFrame:
        with sqlite3.connect(self._db_path) as conn:
            try:
                df = pd.read_sql_query("SELECT * FROM dynamic_universe", conn)
            except Exception as exc:
                logger.error("Failed to load dynamic_universe: %s", exc)
                return pd.DataFrame()
        return df

    def _get_metadata(self, key: str) -> Optional[str]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT value FROM universe_metadata WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_metadata(self, key: str, value: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO universe_metadata (key, value) VALUES (?, ?)", (key, value))
            conn.commit()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_DDL_DYNAMIC_UNIVERSE)
            conn.execute(_DDL_UNIVERSE_METADATA)
            conn.commit()

    def _merge_theme_config(self) -> Dict[str, Dict[str, Any]]:
        merged = {name: dict(cfg) for name, cfg in DEFAULT_THEMES.items()}
        overrides: Dict[str, Any] = (self._config.get("universe", {}).get("themes", {}) or {})
        for theme_name, override_cfg in overrides.items():
            if theme_name in merged:
                merged[theme_name].update(override_cfg)
            else:
                merged[theme_name] = dict(override_cfg)
        return merged


def get_sector_constituents(sector_etf_ticker: str) -> List[str]:
    """Attempt to retrieve the constituents of a sector ETF."""
    try:
        import yfinance as yf
    except ImportError as exc:
        logger.error("yfinance not available: %s", exc)
        return []
    try:
        ticker_obj = yf.Ticker(sector_etf_ticker)
        funds_data = ticker_obj.funds_data
        if funds_data is not None:
            holdings = funds_data.top_holdings
            if holdings is not None and not holdings.empty:
                return holdings.index.tolist()
    except Exception as exc:
        logger.debug("funds_data.top_holdings failed for %s: %s", sector_etf_ticker, exc)
    try:
        info = yf.Ticker(sector_etf_ticker).info or {}
        sector_key: Optional[str] = info.get("sectorKey") or info.get("sector")
        if sector_key:
            normalised = sector_key.lower().replace(" ", "-").replace("_", "-")
            sec_obj = yf.Sector(normalised)
            companies = sec_obj.top_companies
            if companies is not None and not companies.empty:
                return companies.index.tolist()
    except Exception as exc:
        logger.debug("Sector.top_companies fallback failed for %s: %s", sector_etf_ticker, exc)
    logger.warning("get_sector_constituents: all strategies exhausted for %s — returning []", sector_etf_ticker)
    return []


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)


if __name__ == "__main__":
    import sys
    import yaml
    _setup_logging()
    log = logging.getLogger("universe_builder.__main__")
    _search_paths = [Path.cwd()] + list(Path.cwd().parents)[:3]
    _config_path: Optional[Path] = None
    for _p in _search_paths:
        candidate = _p / "config.yaml"
        if candidate.exists():
            _config_path = candidate
            break
    if _config_path is None:
        log.error("config.yaml not found.  Run from the project root or supply a path.")
        sys.exit(1)
    log.info("Loading config from %s", _config_path)
    with open(_config_path, "r") as _f:
        _config = yaml.safe_load(_f)
    _db_relative = (_config.get("database", {}).get("path") or "rotation_system.db")
    _db_path = (_config_path.parent / _db_relative).resolve()
    log.info("Database path: %s", _db_path)
    builder = UniverseBuilder(db_path=_db_path, config=_config)
    log.info("Running build_universe(force=True)...")
    universe_df = builder.build_universe(force=True)
    if universe_df.empty:
        log.warning("Universe is empty — check network access and yfinance installation.")
        sys.exit(0)
    print("\n" + "=" * 60)
    print("  DYNAMIC UNIVERSE — TICKER COUNTS BY THEME")
    print("=" * 60)
    counts = universe_df.groupby("theme").size().sort_values(ascending=False)
    for theme, count in counts.items():
        print(f"  {theme:<25} {count:>4} tickers")
    print("-" * 60)
    print(f"  {'TOTAL':<25} {len(universe_df):>4} rows")
    print("=" * 60 + "\n")
    for theme in sorted(universe_df["theme"].unique()):
        sample = builder.get_watchlist(theme)[:8]
        print(f"  {theme}: {', '.join(sample)}" + (" ..." if len(builder.get_watchlist(theme)) > 8 else ""))
    print()
