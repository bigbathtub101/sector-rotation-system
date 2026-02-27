"""
nlp_sentiment.py — Phase 4: NLP Sentiment Pipeline
=====================================================
Global Sector Rotation System

Scores SEC filings (10-K, 8-K) using ProsusAI/finbert and the
Loughran-McDonald financial word list.  Produces per-sector sentiment
signals that feed into the composite allocation weight during Offense
(20 %), while operating in monitoring-only mode during Defense / Panic.

Key design decisions
---------------------
* MD&A extraction first; fallback = truncate to config `max_tokens` tokens.
* HTML / XBRL stripped before any NLP.
* Loughran-McDonald pre-filter keeps only sentences containing ≥ 1
  finance-sentiment word, reducing FinBERT noise on boilerplate.
* Rolling 90-day trend computed from per-filing scores stored in DB.
* "NLP Regime Drift Risk" flag when VIX > 75th-percentile trailing 252 days.
* Mock mode (--mock) for smoke testing without downloading the 440 MB model.

Dependencies: transformers, torch (real mode); pandas, numpy, pyyaml,
              sqlite3, re, html (always)
"""

import argparse
import datetime as dt
import html
import json
import logging
import math
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent
LOG_FILE = LOG_DIR / "nlp_errors.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("nlp_sentiment")

# ---------------------------------------------------------------------------
# CONFIG / DB PATHS
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"
DB_PATH = Path(__file__).parent / "rotation_system.db"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load master configuration from YAML."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", path)
    return cfg


# ===========================================================================
# 1. DATABASE HELPERS
# ===========================================================================

def get_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Connect and ensure the nlp_scores table exists."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS nlp_scores (
            date            TEXT    NOT NULL,
            ticker          TEXT    NOT NULL,
            filing_type     TEXT    NOT NULL,
            filing_date     TEXT    NOT NULL,
            raw_score       REAL,
            lm_positive     INTEGER DEFAULT 0,
            lm_negative     INTEGER DEFAULT 0,
            lm_uncertainty  INTEGER DEFAULT 0,
            confidence      REAL,
            confidence_flag TEXT,
            vix_at_filing   REAL,
            vix_regime      TEXT,
            md_a_found      INTEGER DEFAULT 0,
            tokens_used     INTEGER DEFAULT 0,
            scored_at       TEXT,
            PRIMARY KEY (date, ticker, filing_type)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS nlp_sector_signals (
            date                TEXT NOT NULL,
            sector_etf          TEXT NOT NULL,
            sector_score        REAL,
            sector_confidence   REAL,
            rolling_trend       REAL,
            n_filings           INTEGER,
            regime_weight       REAL,
            weighted_score      REAL,
            drift_risk          INTEGER DEFAULT 0,
            vix_percentile      REAL,
            computed_at         TEXT,
            PRIMARY KEY (date, sector_etf)
        )
    """)

    conn.commit()
    logger.info("NLP tables verified / created.")
    return conn


def fetch_filings(conn: sqlite3.Connection,
                  tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Return filings from DB, optionally filtered by ticker list."""
    query = "SELECT ticker, filing_type, filing_date, raw_text FROM filings"
    params: list = []
    if tickers:
        placeholders = ",".join("?" for _ in tickers)
        query += f" WHERE ticker IN ({placeholders})"
        params = list(tickers)
    query += " ORDER BY filing_date DESC"
    return pd.read_sql_query(query, conn, params=params)


def fetch_vix_series(conn: sqlite3.Connection,
                     lookback_days: int = 252) -> pd.Series:
    """
    Return a date-indexed Series of VIX closing prices for the trailing
    `lookback_days` calendar days.  Falls back to NaN if no data.
    """
    cutoff = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
    df = pd.read_sql_query(
        "SELECT date, adj_close FROM prices "
        "WHERE ticker = '^VIX' AND date >= ? ORDER BY date",
        conn,
        params=[cutoff],
    )
    if df.empty:
        logger.warning("No VIX data found in prices table.")
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["adj_close"]


def fetch_latest_regime(conn: sqlite3.Connection) -> str:
    """Return the most recent dominant_regime from the signals table."""
    row = conn.execute(
        "SELECT signal_data FROM signals WHERE signal_type = 'regime_state' "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    if row:
        data = json.loads(row[0])
        return data.get("dominant_regime", "offense")
    return "offense"


# ===========================================================================
# 2. TEXT PREPROCESSING
# ===========================================================================

# --- 2a. HTML / XBRL stripping -------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")


def strip_html(raw: str) -> str:
    """Remove all HTML / XBRL tags, decode entities, normalise whitespace."""
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


# --- 2b. MD&A extraction --------------------------------------------------

# Common header patterns for MD&A in 10-K / 10-Q filings
_MDA_HEADERS = [
    r"(?i)management[\u2019\u2018']?s\s+discussion\s+and\s+analysis",
    r"(?i)item\s+7[\.\s]*management",
    r"(?i)item\s+7[\.\s]+md\s*&\s*a",
    r"(?i)md\s*&\s*a\s+of\s+financial\s+condition",
]

# End-of-MD&A markers (next section headers)
_MDA_END = [
    r"(?i)item\s+7a",
    r"(?i)item\s+8[\.\s]",
    r"(?i)quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk",
    r"(?i)financial\s+statements?\s+and\s+supplementary\s+data",
]


def extract_mda(text: str) -> Optional[str]:
    """
    Attempt to extract the MD&A section from plain text of a 10-K/10-Q.
    Returns None if not found.
    """
    start_pos = None
    for pat in _MDA_HEADERS:
        m = re.search(pat, text)
        if m:
            start_pos = m.start()
            break

    if start_pos is None:
        return None

    # Find earliest end marker after start
    end_pos = len(text)
    for pat in _MDA_END:
        m = re.search(pat, text[start_pos + 100:])
        if m:
            candidate = start_pos + 100 + m.start()
            end_pos = min(end_pos, candidate)

    mda = text[start_pos:end_pos].strip()
    if len(mda) < 200:          # too short — probably a table-of-contents ref
        return None
    return mda


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Rough word-level truncation to approximate `max_tokens` tokens.
    FinBERT uses WordPiece so actual token count ≈ 1.3 × word count;
    we use 1.3 factor conservatively.
    """
    max_words = int(max_tokens / 1.3)
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def preprocess_filing(raw_text: str, filing_type: str,
                      max_tokens: int = 512) -> Tuple[str, bool]:
    """
    Full preprocessing pipeline for a single filing.

    Returns (processed_text, mda_found).
    """
    # Step 1: strip HTML/XBRL
    plain = strip_html(raw_text)

    if not plain or len(plain.split()) < 20:
        return plain, False

    # Step 2: attempt MD&A extraction (10-K / 10-Q only)
    mda_found = False
    if filing_type in ("10-K", "10-Q"):
        mda = extract_mda(plain)
        if mda:
            plain = mda
            mda_found = True

    # Step 3: truncate to token limit
    processed = truncate_to_tokens(plain, max_tokens)
    return processed, mda_found


# ===========================================================================
# 3. LOUGHRAN-MCDONALD FINANCIAL WORD LISTS
# ===========================================================================

# Core finance-sentiment words from the Loughran-McDonald dictionary.
# We use a curated subset (most-frequent 80 terms in each polarity)
# to keep the module self-contained without downloading the full CSV.

LM_POSITIVE = frozenset([
    "achieve", "achieved", "achieving", "advancement", "advantage",
    "beneficial", "benefit", "benefited", "benefiting", "benefits",
    "better", "bolster", "boost", "creative", "delivered", "earn",
    "earned", "earnings", "efficiency", "enable", "enabled",
    "enhance", "enhanced", "excellent", "exceed", "exceeded",
    "exceeding", "expand", "expanded", "favorable", "gain",
    "gained", "gaining", "gains", "good", "greater", "grew",
    "grow", "growing", "growth", "highest", "improve", "improved",
    "improvement", "improving", "increase", "increased", "increasing",
    "innovation", "innovative", "momentum", "opportunities",
    "opportunity", "optimistic", "outperform", "outperformed",
    "positive", "profitability", "profitable", "progress",
    "prosper", "record", "recover", "recovered", "recovery",
    "reward", "rewarding", "rise", "risen", "rising", "solid",
    "strength", "strengthen", "strong", "stronger", "succeed",
    "succeeded", "success", "successful", "superior", "surpass",
    "surpassed", "upturn", "upward",
])

LM_NEGATIVE = frozenset([
    "abandon", "adverse", "adversely", "challenge", "challenged",
    "challenges", "closure", "concern", "concerns", "decline",
    "declined", "declining", "decrease", "decreased", "decreasing",
    "default", "deficit", "delay", "delayed", "deteriorate",
    "deteriorated", "deteriorating", "difficult", "difficulties",
    "difficulty", "diminish", "diminished", "discontinue", "doubt",
    "downturn", "dropped", "erode", "eroded", "eroding", "erosion",
    "fail", "failed", "failing", "failure", "fell", "fraud",
    "impair", "impaired", "impairment", "inability", "inadequate",
    "investigation", "lawsuit", "layoff", "layoffs", "liability",
    "litigation", "lose", "loss", "losses", "losing", "negative",
    "negatively", "penalty", "problem", "problems", "recession",
    "restructure", "restructuring", "risk", "risks", "severe",
    "shortage", "shrink", "shrinkage", "slowdown", "slowing",
    "struggling", "terminate", "terminated", "threat", "troubled",
    "uncertain", "uncertainties", "uncertainty", "unfavorable",
    "violation", "volatile", "volatility", "warn", "warning",
    "weak", "weaken", "weakened", "weakness", "worse", "worsen",
    "worsened", "worsening", "writedown", "writeoff",
])

LM_UNCERTAINTY = frozenset([
    "approximate", "approximately", "assume", "assumed", "assumes",
    "assuming", "assumption", "assumptions", "believe", "believed",
    "contingency", "contingent", "could", "depend", "depending",
    "depends", "doubt", "estimate", "estimated", "estimates",
    "estimating", "expect", "expected", "expecting", "fluctuate",
    "fluctuated", "fluctuating", "fluctuation", "fluctuations",
    "indefinite", "indefinitely", "may", "might", "nearly",
    "pending", "perhaps", "possible", "possibly", "predict",
    "predicted", "predicting", "prediction", "preliminary",
    "probable", "probably", "project", "projected", "projecting",
    "projection", "projections", "roughly", "seem", "seemed",
    "seemingly", "seems", "somewhat", "suggest", "suggested",
    "suggesting", "suggests", "tentative", "tentatively",
    "uncertain", "uncertainties", "uncertainty", "unclear",
    "undetermined", "unknown", "unpredictable", "unresolved",
    "unsure", "variable", "variability",
])

ALL_LM_WORDS = LM_POSITIVE | LM_NEGATIVE | LM_UNCERTAINTY


def lm_word_counts(text: str) -> Dict[str, int]:
    """Count LM sentiment words in text. Returns dict with pos/neg/unc."""
    words = re.findall(r"[a-z]+", text.lower())
    pos = sum(1 for w in words if w in LM_POSITIVE)
    neg = sum(1 for w in words if w in LM_NEGATIVE)
    unc = sum(1 for w in words if w in LM_UNCERTAINTY)
    return {"positive": pos, "negative": neg, "uncertainty": unc}


def lm_sentence_filter(text: str) -> str:
    """
    Pre-filter: keep only sentences containing ≥ 1 Loughran-McDonald
    sentiment word.  This concentrates FinBERT on opinion-carrying
    sentences and avoids misclassifying boilerplate legalese.
    """
    # Split on period, question-mark, or newline as sentence boundaries
    sentences = re.split(r"(?<=[.?!])\s+|\n+", text)
    filtered = []
    for sent in sentences:
        words = set(re.findall(r"[a-z]+", sent.lower()))
        if words & ALL_LM_WORDS:
            filtered.append(sent.strip())
    result = " ".join(filtered)
    if not result:
        # If no LM words found at all, return original (edge case)
        return text
    return result


# ===========================================================================
# 4. FINBERT SCORING
# ===========================================================================

class FinBERTScorer:
    """
    Wraps ProsusAI/finbert inference.  Call `.score(text)` to get
    (label, score, confidence).  Label ∈ {positive, negative, neutral}.

    In mock mode, returns deterministic pseudo-scores without loading
    the model (saves 440 MB download + GPU/CPU time).
    """

    def __init__(self, model_name: str = "ProsusAI/finbert",
                 mock: bool = False):
        self.mock = mock
        self.model_name = model_name
        self._pipeline = None

        if not mock:
            self._load_model()

    def _load_model(self) -> None:
        """Lazy-load the HuggingFace pipeline."""
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading FinBERT model '%s' ...", self.model_name)
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load FinBERT: %s", exc)
            logger.warning("Falling back to mock mode.")
            self.mock = True

    def score(self, text: str) -> Dict[str, Any]:
        """
        Score a text passage.

        Returns dict:
            label       : str   — "positive" | "negative" | "neutral"
            score       : float — numeric score, −1 to +1
            confidence  : float — model confidence [0, 1]
        """
        if self.mock:
            return self._mock_score(text)

        if not text or len(text.strip()) < 10:
            return {"label": "neutral", "score": 0.0, "confidence": 0.0}

        result = self._pipeline(text[:512 * 4])[0]   # rough char limit
        label = result["label"].lower()
        conf = float(result["score"])

        # Convert to numeric: positive = +conf, negative = −conf, neutral = 0
        if label == "positive":
            numeric = conf
        elif label == "negative":
            numeric = -conf
        else:
            numeric = 0.0

        return {"label": label, "score": round(numeric, 4),
                "confidence": round(conf, 4)}

    @staticmethod
    def _mock_score(text: str) -> Dict[str, Any]:
        """
        Deterministic mock scoring based on Loughran-McDonald word counts.
        Gives a reasonable approximation without the model.
        """
        counts = lm_word_counts(text)
        pos, neg = counts["positive"], counts["negative"]
        total = pos + neg + 1  # avoid div-by-zero

        if pos > neg:
            label = "positive"
            score = round(min((pos - neg) / total, 1.0), 4)
        elif neg > pos:
            label = "negative"
            score = round(max(-(neg - pos) / total, -1.0), 4)
        else:
            label = "neutral"
            score = 0.0

        confidence = round(min(abs(pos - neg) / total + 0.3, 0.99), 4)
        return {"label": label, "score": score, "confidence": confidence}


# ===========================================================================
# 5. SECTOR-ETF → TOP HOLDINGS MAPPING
# ===========================================================================

# Static mapping of sector ETFs to their approximate top-5 holdings
# (by weight).  Updated periodically.  Used to determine which tickers'
# filings represent each sector.  This avoids a live API call.

DEFAULT_SECTOR_HOLDINGS: Dict[str, List[str]] = {
    "XLK":  ["AAPL", "MSFT", "NVDA", "AVGO", "CRM"],
    "XLV":  ["LLY", "UNH", "JNJ", "ABBV", "MRK"],
    "XLE":  ["XOM", "CVX", "EOG", "SLB", "MPC"],
    "XLF":  ["BRK-B", "JPM", "V", "MA", "BAC"],
    "XLI":  ["GE", "CAT", "UNP", "HON", "RTX"],
    "XLB":  ["LIN", "SHW", "APD", "FCX", "ECL"],
    "XLU":  ["NEE", "SO", "DUK", "CEG", "SRE"],
    "XLP":  ["PG", "COST", "WMT", "KO", "PEP"],
    "XLRE": ["PLD", "AMT", "EQIX", "WELL", "SPG"],
    "XLC":  ["META", "GOOGL", "GOOG", "NFLX", "T"],
    "XLY":  ["AMZN", "TSLA", "HD", "MCD", "NKE"],
}


def get_sector_holdings(cfg: dict) -> Dict[str, List[str]]:
    """
    Return mapping of sector ETF → top-N holdings.
    Uses config override `nlp.sector_holdings` if present,
    otherwise falls back to DEFAULT_SECTOR_HOLDINGS.
    """
    override = cfg.get("nlp", {}).get("sector_holdings")
    if override and isinstance(override, dict):
        return override
    return DEFAULT_SECTOR_HOLDINGS.copy()


# ===========================================================================
# 6. VIX REGIME TAGGING & DRIFT RISK
# ===========================================================================

def compute_vix_percentile(vix_series: pd.Series,
                           as_of_date: Optional[str] = None) -> float:
    """
    Return the percentile rank of the current (or `as_of_date`) VIX
    within the trailing data.  Returns NaN if insufficient data.
    """
    if vix_series.empty:
        return float("nan")

    if as_of_date:
        target = pd.Timestamp(as_of_date)
        # Find closest date <= target
        valid = vix_series[vix_series.index <= target]
        if valid.empty:
            return float("nan")
        current_vix = valid.iloc[-1]
        series_for_rank = valid
    else:
        current_vix = vix_series.iloc[-1]
        series_for_rank = vix_series

    from scipy.stats import percentileofscore
    pct = percentileofscore(series_for_rank.dropna().values,
                            current_vix, kind="rank")
    return round(pct, 2)


def tag_vix_regime(vix_value: float, vix_series: pd.Series) -> str:
    """
    Classify the VIX level at time of filing into a regime bucket
    for rolling fine-tuning tagging.

    Uses the same thresholds as regime_detector for consistency:
      VIX < 20       → low_vol
      20 ≤ VIX < 30  → elevated
      VIX ≥ 30       → high_vol
    """
    if math.isnan(vix_value):
        return "unknown"
    if vix_value < 20:
        return "low_vol"
    elif vix_value < 30:
        return "elevated"
    else:
        return "high_vol"


def check_drift_risk(vix_percentile: float,
                     threshold_pct: float) -> bool:
    """
    Return True if VIX is above its `threshold_pct`-th percentile
    trailing 252-day → NLP Regime Drift Risk.
    """
    if math.isnan(vix_percentile):
        return False
    return vix_percentile >= threshold_pct


# ===========================================================================
# 7. CONFIDENCE FLAGS
# ===========================================================================

def compute_confidence_flag(confidence: float, n_filings: int,
                            mda_found: bool) -> str:
    """
    Assign a human-readable confidence level.

    HIGH   : confidence ≥ 0.7 AND mda_found AND n_filings ≥ 3
    MEDIUM : confidence ≥ 0.5 OR (mda_found AND n_filings ≥ 1)
    LOW    : everything else
    """
    if confidence >= 0.7 and mda_found and n_filings >= 3:
        return "HIGH"
    if confidence >= 0.5 or (mda_found and n_filings >= 1):
        return "MEDIUM"
    return "LOW"


# ===========================================================================
# 8. CORE PIPELINE — SCORE FILINGS
# ===========================================================================

def score_single_filing(raw_text: str, filing_type: str,
                        scorer: FinBERTScorer,
                        max_tokens: int = 512) -> Dict[str, Any]:
    """
    Full pipeline for one filing:
      1. strip HTML → 2. extract MD&A → 3. truncate →
      4. LM pre-filter → 5. FinBERT score → 6. LM counts

    Returns dict with all scoring fields.
    """
    # Preprocess
    processed, mda_found = preprocess_filing(raw_text, filing_type, max_tokens)

    if not processed or len(processed.strip()) < 20:
        return {
            "raw_score": 0.0, "confidence": 0.0, "confidence_flag": "LOW",
            "lm_positive": 0, "lm_negative": 0, "lm_uncertainty": 0,
            "md_a_found": False, "tokens_used": 0,
            "label": "neutral",
        }

    # LM pre-filter
    filtered = lm_sentence_filter(processed)

    # Truncate filtered text to token limit (safety net)
    final_text = truncate_to_tokens(filtered, max_tokens)
    tokens_used = len(final_text.split())

    # Score with FinBERT (or mock)
    result = scorer.score(final_text)

    # LM word counts on the original processed text
    lm = lm_word_counts(processed)

    return {
        "raw_score": result["score"],
        "confidence": result["confidence"],
        "confidence_flag": compute_confidence_flag(
            result["confidence"], 1, mda_found),
        "label": result["label"],
        "lm_positive": lm["positive"],
        "lm_negative": lm["negative"],
        "lm_uncertainty": lm["uncertainty"],
        "md_a_found": mda_found,
        "tokens_used": tokens_used,
    }


def score_all_filings(conn: sqlite3.Connection,
                      scorer: FinBERTScorer,
                      cfg: dict,
                      tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Score every filing in the DB (or a subset).
    Stores results in nlp_scores table and returns the DataFrame.
    """
    max_tokens = cfg.get("nlp", {}).get("max_tokens", 512)
    filings_df = fetch_filings(conn, tickers)

    if filings_df.empty:
        logger.warning("No filings found to score.")
        return pd.DataFrame(columns=[
            "date", "ticker", "filing_type", "filing_date", "raw_score",
            "lm_positive", "lm_negative", "lm_uncertainty", "confidence",
            "confidence_flag", "vix_at_filing", "vix_regime", "md_a_found",
            "tokens_used", "scored_at",
        ])

    vix_series = fetch_vix_series(conn)
    now = dt.datetime.now().isoformat()
    today = dt.date.today().isoformat()
    rows = []

    for _, row in filings_df.iterrows():
        ticker = row["ticker"]
        filing_type = row["filing_type"]
        filing_date = row["filing_date"]
        raw_text = row["raw_text"] if pd.notna(row["raw_text"]) else ""

        logger.info("Scoring %s %s (filed %s) ...", ticker,
                     filing_type, filing_date)

        try:
            result = score_single_filing(
                raw_text, filing_type, scorer, max_tokens)
        except Exception as exc:
            logger.error("Error scoring %s %s: %s", ticker, filing_type, exc)
            result = {
                "raw_score": 0.0, "confidence": 0.0,
                "confidence_flag": "LOW",
                "lm_positive": 0, "lm_negative": 0, "lm_uncertainty": 0,
                "md_a_found": False, "tokens_used": 0,
            }

        # VIX at time of filing
        vix_at_filing = float("nan")
        vix_regime_tag = "unknown"
        if not vix_series.empty:
            vix_pct = compute_vix_percentile(vix_series, filing_date)
            mask = vix_series.index <= pd.Timestamp(filing_date)
            valid_vix = vix_series[mask]
            if not valid_vix.empty:
                vix_at_filing = float(valid_vix.iloc[-1])
                vix_regime_tag = tag_vix_regime(vix_at_filing, vix_series)

        record = {
            "date": today,
            "ticker": ticker,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "raw_score": result["raw_score"],
            "lm_positive": result["lm_positive"],
            "lm_negative": result["lm_negative"],
            "lm_uncertainty": result["lm_uncertainty"],
            "confidence": result["confidence"],
            "confidence_flag": result["confidence_flag"],
            "vix_at_filing": None if math.isnan(vix_at_filing) else vix_at_filing,
            "vix_regime": vix_regime_tag,
            "md_a_found": 1 if result["md_a_found"] else 0,
            "tokens_used": result["tokens_used"],
            "scored_at": now,
        }
        rows.append(record)

    scores_df = pd.DataFrame(rows)

    # Upsert into nlp_scores
    cur = conn.cursor()
    for _, r in scores_df.iterrows():
        cur.execute("""
            INSERT OR REPLACE INTO nlp_scores
                (date, ticker, filing_type, filing_date, raw_score,
                 lm_positive, lm_negative, lm_uncertainty,
                 confidence, confidence_flag, vix_at_filing,
                 vix_regime, md_a_found, tokens_used, scored_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["date"], r["ticker"], r["filing_type"], r["filing_date"],
            r["raw_score"], r["lm_positive"], r["lm_negative"],
            r["lm_uncertainty"], r["confidence"], r["confidence_flag"],
            r["vix_at_filing"], r["vix_regime"], r["md_a_found"],
            r["tokens_used"], r["scored_at"],
        ))
    conn.commit()
    logger.info("Scored %d filings → nlp_scores table.", len(scores_df))
    return scores_df


# ===========================================================================
# 9. SECTOR SIGNAL AGGREGATION
# ===========================================================================

def compute_sector_signals(conn: sqlite3.Connection,
                           cfg: dict) -> pd.DataFrame:
    """
    For each sector ETF, aggregate sentiment across top-5 holdings'
    most recent filings.  Compute rolling 90-day trend and apply
    regime-adjusted weight.

    Returns DataFrame stored in nlp_sector_signals.
    """
    nlp_cfg = cfg.get("nlp", {})
    rolling_window = nlp_cfg.get("rolling_sentiment_window", 90)
    vix_drift_pct = nlp_cfg.get("vix_drift_percentile", 75)
    regime_weights = nlp_cfg.get("regime_weights",
                                  {"offense": 0.20, "defense": 0.0, "panic": 0.0})

    sector_holdings = get_sector_holdings(cfg)
    vix_series = fetch_vix_series(conn)
    current_regime = fetch_latest_regime(conn)
    regime_weight = regime_weights.get(current_regime, 0.0)
    today = dt.date.today().isoformat()
    now = dt.datetime.now().isoformat()

    if current_regime in ("defense", "panic"):
        logger.info(
            "Regime=%s → NLP weight=%.2f (monitoring only, "
            "scores computed but not applied).",
            current_regime, regime_weight,
        )

    # VIX percentile for drift risk
    vix_pctl = compute_vix_percentile(vix_series)
    drift_risk = check_drift_risk(vix_pctl, vix_drift_pct)

    # Get all NLP scores
    all_scores = pd.read_sql_query(
        "SELECT * FROM nlp_scores ORDER BY filing_date DESC", conn
    )

    results = []
    for etf, holdings in sector_holdings.items():
        # Filter scores to this sector's holdings
        sector_scores = all_scores[all_scores["ticker"].isin(holdings)]

        # Apply drift-risk override to base weight
        effective_weight = 0.0 if drift_risk else regime_weight

        if sector_scores.empty:
            results.append({
                "date": today, "sector_etf": etf,
                "sector_score": 0.0, "sector_confidence": 0.0,
                "rolling_trend": 0.0, "n_filings": 0,
                "regime_weight": effective_weight,
                "weighted_score": 0.0,
                "drift_risk": 1 if drift_risk else 0,
                "vix_percentile": vix_pctl if not math.isnan(vix_pctl) else None,
                "computed_at": now,
            })
            continue

        # Take most recent filing per ticker
        latest_per_ticker = (
            sector_scores.sort_values("filing_date", ascending=False)
            .drop_duplicates(subset=["ticker"], keep="first")
        )

        # Aggregate: mean score, mean confidence (NaN-safe)
        raw_scores = latest_per_ticker["raw_score"].dropna()
        conf_vals = latest_per_ticker["confidence"].dropna()
        sector_score = float(raw_scores.mean()) if not raw_scores.empty else 0.0
        sector_conf = float(conf_vals.mean()) if not conf_vals.empty else 0.0
        n_filings = len(latest_per_ticker)

        # Rolling 90-day trend: compare current mean to mean of filings
        # that are 45-90 days old.  Positive trend = improving.
        cutoff_recent = (
            dt.date.today() - dt.timedelta(days=rolling_window // 2)
        ).isoformat()
        cutoff_old = (
            dt.date.today() - dt.timedelta(days=rolling_window)
        ).isoformat()

        recent = sector_scores[sector_scores["filing_date"] >= cutoff_recent]
        older = sector_scores[
            (sector_scores["filing_date"] >= cutoff_old)
            & (sector_scores["filing_date"] < cutoff_recent)
        ]

        if not recent.empty and not older.empty:
            rolling_trend = round(
                float(recent["raw_score"].mean() - older["raw_score"].mean()),
                4,
            )
        else:
            rolling_trend = 0.0

        # Regime-adjusted weighted score
        weighted = round(sector_score * effective_weight, 4)

        results.append({
            "date": today,
            "sector_etf": etf,
            "sector_score": round(sector_score, 4),
            "sector_confidence": round(sector_conf, 4),
            "rolling_trend": rolling_trend,
            "n_filings": n_filings,
            "regime_weight": effective_weight,
            "weighted_score": weighted,
            "drift_risk": 1 if drift_risk else 0,
            "vix_percentile": vix_pctl if not math.isnan(vix_pctl) else None,
            "computed_at": now,
        })

    signals_df = pd.DataFrame(results)

    # Store in DB
    cur = conn.cursor()
    for _, r in signals_df.iterrows():
        cur.execute("""
            INSERT OR REPLACE INTO nlp_sector_signals
                (date, sector_etf, sector_score, sector_confidence,
                 rolling_trend, n_filings, regime_weight, weighted_score,
                 drift_risk, vix_percentile, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["date"], r["sector_etf"], r["sector_score"],
            r["sector_confidence"], r["rolling_trend"], r["n_filings"],
            r["regime_weight"], r["weighted_score"], r["drift_risk"],
            r["vix_percentile"], r["computed_at"],
        ))
    conn.commit()
    logger.info("Computed sector signals for %d ETFs.", len(signals_df))
    return signals_df


# ===========================================================================
# 10. HUMAN-READABLE REPORT
# ===========================================================================

def generate_nlp_report(scores_df: pd.DataFrame,
                        signals_df: pd.DataFrame,
                        regime: str) -> str:
    """
    Build a console-friendly NLP Sentiment report.
    """
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  NLP SENTIMENT PIPELINE — PHASE 4 REPORT")
    lines.append(f"  Date: {dt.date.today().isoformat()}   "
                 f"Regime: {regime.upper()}")
    lines.append("=" * 70)

    # --- Per-filing scores ---
    lines.append("")
    lines.append("─── FILING-LEVEL SCORES ───")
    if scores_df.empty:
        lines.append("  (no filings scored)")
    else:
        lines.append(f"  {'Ticker':<8} {'Type':<6} {'Filed':<12} "
                     f"{'Score':>7} {'Conf':>6} {'Flag':<7} "
                     f"{'LM+':>4} {'LM−':>4} {'LM?':>4} "
                     f"{'MD&A':>5} {'VIX Regime':<10}")
        lines.append("  " + "─" * 80)
        for _, r in scores_df.iterrows():
            lines.append(
                f"  {r['ticker']:<8} {r['filing_type']:<6} "
                f"{r['filing_date']:<12} "
                f"{r['raw_score']:>+7.4f} {r['confidence']:>6.4f} "
                f"{r['confidence_flag']:<7} "
                f"{r['lm_positive']:>4} {r['lm_negative']:>4} "
                f"{r['lm_uncertainty']:>4} "
                f"{'  Y  ' if r['md_a_found'] else '  N  '} "
                f"{r['vix_regime']:<10}"
            )

    # --- Sector signals ---
    lines.append("")
    lines.append("─── SECTOR NLP SIGNALS ───")
    if signals_df.empty:
        lines.append("  (no sector signals)")
    else:
        drift_any = signals_df["drift_risk"].any()
        if drift_any:
            lines.append("  ⚠ NLP REGIME DRIFT RISK ACTIVE "
                         "— VIX above 75th percentile trailing 252-day")
            lines.append("    NLP weight overridden to 0% "
                         "(monitoring only)")
            lines.append("")

        lines.append(f"  {'ETF':<6} {'Score':>7} {'Conf':>6} "
                     f"{'Trend':>7} {'#Files':>6} {'Weight':>7} "
                     f"{'Wtd Score':>10} {'Drift':>6}")
        lines.append("  " + "─" * 65)
        for _, r in signals_df.iterrows():
            lines.append(
                f"  {r['sector_etf']:<6} {r['sector_score']:>+7.4f} "
                f"{r['sector_confidence']:>6.4f} "
                f"{r['rolling_trend']:>+7.4f} "
                f"{r['n_filings']:>6} "
                f"{r['regime_weight']:>6.0%} "
                f"{r['weighted_score']:>+10.4f} "
                f"{'  Y ' if r['drift_risk'] else '  N '}"
            )

        lines.append("")
        vix_pct = signals_df["vix_percentile"].iloc[0]
        if vix_pct is not None and not (isinstance(vix_pct, float) and math.isnan(vix_pct)):
            vix_str = f"{vix_pct:.1f}%"
        else:
            vix_str = "N/A"
        lines.append(f"  VIX trailing 252-day percentile: {vix_str}")
        lines.append(f"  Current regime: {regime.upper()}")
        rw = signals_df["regime_weight"].iloc[0]
        lines.append(f"  NLP regime weight: {rw:.0%}"
                     f"{' (monitoring only)' if rw == 0 else ''}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ===========================================================================
# 11. CLI ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 — NLP Sentiment Pipeline")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock FinBERT scorer (no model download)")
    parser.add_argument("--real", action="store_true",
                        help="Use real FinBERT model from Hugging Face")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to SQLite database")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--tickers", type=str, nargs="*", default=None,
                        help="Score only these tickers' filings")
    args = parser.parse_args()

    # Default to mock unless --real is set
    use_mock = True
    if args.real:
        use_mock = False
    elif args.mock:
        use_mock = True

    # Paths
    db_path = Path(args.db) if args.db else DB_PATH
    config_path = Path(args.config) if args.config else CONFIG_PATH

    cfg = load_config(config_path)

    # Connect DB
    conn = get_db(db_path)

    # Init scorer
    model_name = cfg.get("nlp", {}).get("model", "ProsusAI/finbert")
    scorer = FinBERTScorer(model_name=model_name, mock=use_mock)
    mode_label = "MOCK" if scorer.mock else "REAL"
    logger.info("FinBERT scorer initialised in %s mode.", mode_label)

    # Score filings
    scores_df = score_all_filings(conn, scorer, cfg, args.tickers)

    # Compute sector signals
    signals_df = compute_sector_signals(conn, cfg)

    # Get regime
    regime = fetch_latest_regime(conn)

    # Report
    report = generate_nlp_report(scores_df, signals_df, regime)
    print(report)

    conn.close()
    logger.info("Phase 4 NLP pipeline complete.")


if __name__ == "__main__":
    main()
