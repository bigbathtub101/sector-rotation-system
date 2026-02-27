"""
smoke_test_phase4.py — Comprehensive Smoke Tests for NLP Sentiment Pipeline
=============================================================================
Global Sector Rotation System — Phase 4

Tests every component of nlp_sentiment.py using the live DB plus
synthetic fixtures.  Runs entirely in mock mode (no FinBERT download).

Usage:
    python smoke_test_phase4.py          # runs all checks
    python smoke_test_phase4.py -v       # verbose output
"""

import datetime as dt
import json
import math
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure the module is importable
sys.path.insert(0, str(Path(__file__).parent))

from nlp_sentiment import (
    # DB helpers
    get_db,
    fetch_filings,
    fetch_vix_series,
    fetch_latest_regime,
    # Text preprocessing
    strip_html,
    extract_mda,
    truncate_to_tokens,
    preprocess_filing,
    # LM word lists
    LM_POSITIVE,
    LM_NEGATIVE,
    LM_UNCERTAINTY,
    ALL_LM_WORDS,
    lm_word_counts,
    lm_sentence_filter,
    # FinBERT scorer
    FinBERTScorer,
    # Sector holdings
    DEFAULT_SECTOR_HOLDINGS,
    get_sector_holdings,
    # VIX
    compute_vix_percentile,
    tag_vix_regime,
    check_drift_risk,
    # Confidence
    compute_confidence_flag,
    # Pipeline
    score_single_filing,
    score_all_filings,
    compute_sector_signals,
    # Report
    generate_nlp_report,
    # Config
    load_config,
)

# ---------------------------------------------------------------------------
# TEST FRAMEWORK
# ---------------------------------------------------------------------------
VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv

passed = 0
failed = 0
errors = []


def check(name: str, condition: bool, detail: str = ""):
    """Register a single test check."""
    global passed, failed
    if condition:
        passed += 1
        if VERBOSE:
            print(f"  \u2713 {name}")
    else:
        failed += 1
        msg = f"  \u2717 {name}"
        if detail:
            msg += f"  \u2192  {detail}"
        print(msg)
        errors.append(name)


def section(title: str):
    print(f"\n{'\u2500' * 60}")
    print(f"  {title}")
    print(f"{'\u2500' * 60}")


# ===========================================================================
# FIXTURES \u2014 synthetic DB for isolated testing
# ===========================================================================

SAMPLE_10K_HTML = """
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
<p>Item 7. Management's Discussion and Analysis of Financial Condition</p>
<p>Our company achieved strong growth in fiscal year 2025, with revenue
increasing 15% year-over-year. Profitability improved significantly due
to operational efficiency gains and cost reduction initiatives. We
successfully expanded into new markets and delivered record earnings
per share. The favorable macro environment enabled accelerated investment
in innovation and R&amp;D programs.</p>
<p>However, we face challenges from rising interest rates and potential
supply chain disruptions. Litigation risks remain elevated, and adverse
regulatory changes could negatively impact our operations. Losses in the
international segment deteriorated due to currency headwinds and declining
demand. We experienced significant impairment charges related to
restructuring activities.</p>
<p>Management believes that uncertainty around trade policy may cause
fluctuations in our export business. We estimate approximately 10% of
revenue could be affected if projected tariffs materialize. These
projections assume no further escalation.</p>
<p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p>
<p>This section intentionally left blank.</p>
</body>
</html>
"""

SAMPLE_8K_HTML = """
<html>
<body>
<p>FORM 8-K CURRENT REPORT</p>
<p>The company reported record quarterly earnings, with revenue growth
exceeding expectations. Management provided optimistic guidance for the
next quarter, citing strong demand and improved profitability. The Board
approved a dividend increase reflecting confidence in sustainable
growth.</p>
</body>
</html>
"""

SAMPLE_XBRL_ONLY = """
<?xml version='1.0' encoding='ASCII'?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
<ix:nonfraction contextRef="c-1" name="us-gaap:Revenue" unitRef="usd"
    decimals="-6">15000000000</ix:nonfraction>
<ix:nonfraction contextRef="c-1" name="us-gaap:NetIncome" unitRef="usd"
    decimals="-6">2500000000</ix:nonfraction>
</body>
</html>
"""


def create_test_db() -> sqlite3.Connection:
    """Build a temporary in-memory DB with synthetic data."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    # --- prices table (VIX data for 300 days) ---
    cur.execute("""
        CREATE TABLE prices (
            date TEXT, ticker TEXT, open REAL, high REAL, low REAL,
            close REAL, adj_close REAL, volume INTEGER,
            stale_price INTEGER DEFAULT 0, fetched_at TEXT,
            PRIMARY KEY (date, ticker)
        )
    """)
    base = dt.date(2025, 3, 1)
    np.random.seed(42)
    vix_vals = np.random.uniform(14, 35, 300)
    for i, vix in enumerate(vix_vals):
        d = (base + dt.timedelta(days=i)).isoformat()
        cur.execute(
            "INSERT INTO prices VALUES (?,?,?,?,?,?,?,?,?,?)",
            (d, "^VIX", vix, vix + 1, vix - 1, vix, vix, 1000000, 0, "now"),
        )

    # --- filings table ---
    cur.execute("""
        CREATE TABLE filings (
            cik TEXT, ticker TEXT, company_name TEXT, filing_type TEXT,
            filing_date TEXT, accession_number TEXT, primary_document TEXT,
            filing_url TEXT, raw_text TEXT, fetched_at TEXT
        )
    """)
    # Insert synthetic filings for multiple tickers
    test_tickers = [
        ("AAPL", "10-K", "2025-11-01", SAMPLE_10K_HTML),
        ("AAPL", "8-K", "2025-12-15", SAMPLE_8K_HTML),
        ("MSFT", "10-K", "2025-10-15", SAMPLE_10K_HTML),
        ("NVDA", "8-K", "2025-12-01", SAMPLE_8K_HTML),
        ("LLY", "10-K", "2025-09-01", SAMPLE_10K_HTML),
        ("UNH", "8-K", "2025-11-20", SAMPLE_8K_HTML),
        ("XBRL_ONLY", "10-K", "2025-10-01", SAMPLE_XBRL_ONLY),
    ]
    for ticker, ftype, fdate, text in test_tickers:
        cur.execute(
            "INSERT INTO filings VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("000", ticker, f"{ticker} Inc", ftype, fdate,
             "0001-00", "doc.htm", "http://sec.gov", text, "now"),
        )

    # --- signals table (regime data) ---
    cur.execute("""
        CREATE TABLE signals (
            date TEXT, signal_type TEXT, signal_data TEXT, created_at TEXT
        )
    """)
    # Recent offense regime
    for i in range(30):
        d = (dt.date.today() - dt.timedelta(days=30 - i)).isoformat()
        data = json.dumps({
            "dominant_regime": "offense",
            "wedge_volume_percentile": 85,
        })
        cur.execute(
            "INSERT INTO signals VALUES (?,?,?,?)",
            (d, "regime_state", data, "now"),
        )

    # --- allocations table ---
    cur.execute("""
        CREATE TABLE allocations (
            date TEXT, regime TEXT, allocations TEXT,
            taxable_dollars TEXT, roth_dollars TEXT, created_at TEXT
        )
    """)

    conn.commit()

    # Now create NLP tables
    nlp_conn = get_db.__wrapped__(conn) if hasattr(get_db, "__wrapped__") else None
    # Manually create NLP tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nlp_scores (
            date TEXT NOT NULL, ticker TEXT NOT NULL,
            filing_type TEXT NOT NULL, filing_date TEXT NOT NULL,
            raw_score REAL, lm_positive INTEGER DEFAULT 0,
            lm_negative INTEGER DEFAULT 0, lm_uncertainty INTEGER DEFAULT 0,
            confidence REAL, confidence_flag TEXT,
            vix_at_filing REAL, vix_regime TEXT,
            md_a_found INTEGER DEFAULT 0, tokens_used INTEGER DEFAULT 0,
            scored_at TEXT,
            PRIMARY KEY (date, ticker, filing_type)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nlp_sector_signals (
            date TEXT NOT NULL, sector_etf TEXT NOT NULL,
            sector_score REAL, sector_confidence REAL,
            rolling_trend REAL, n_filings INTEGER,
            regime_weight REAL, weighted_score REAL,
            drift_risk INTEGER DEFAULT 0, vix_percentile REAL,
            computed_at TEXT,
            PRIMARY KEY (date, sector_etf)
        )
    """)
    conn.commit()
    return conn


def get_test_config() -> dict:
    """Load real config or build minimal test config."""
    cfg_path = Path(__file__).parent / "config.yaml"
    if cfg_path.exists():
        return load_config(cfg_path)
    return {
        "nlp": {
            "model": "ProsusAI/finbert",
            "max_tokens": 512,
            "rolling_sentiment_window": 90,
            "vix_drift_percentile": 75,
            "regime_weights": {
                "offense": 0.20,
                "defense": 0.00,
                "panic": 0.00,
            },
        },
        "tickers": {"sector_etfs": [
            "XLK", "XLV", "XLE", "XLF", "XLI",
            "XLB", "XLU", "XLP", "XLRE", "XLC", "XLY",
        ]},
        "sec_edgar": {"top_holdings_per_etf": 5},
    }


# ===========================================================================
# TEST SUITES
# ===========================================================================

def test_html_stripping():
    section("1. HTML / XBRL Stripping")

    # Basic HTML
    out = strip_html("<p>Hello <b>world</b></p>")
    check("strip_html removes tags", "Hello" in out and "world" in out)
    check("strip_html no angle brackets", "<" not in out)

    # XBRL
    out = strip_html(SAMPLE_XBRL_ONLY)
    check("strip_html handles XBRL", "15000000000" in out)

    # Entity decoding
    out = strip_html("<p>R&amp;D costs &gt; $1M</p>")
    check("strip_html decodes entities", "R&D" in out and "> $1M" in out)

    # Empty / None-ish
    out = strip_html("")
    check("strip_html handles empty string", out == "")

    # Whitespace normalisation
    out = strip_html("<p>too   many     spaces</p>")
    check("strip_html normalises whitespace", "  " not in out)


def test_mda_extraction():
    section("2. MD&A Extraction")

    plain = strip_html(SAMPLE_10K_HTML)

    mda = extract_mda(plain)
    check("extract_mda finds MD&A section", mda is not None)
    if mda:
        check("MD&A contains growth text", "growth" in mda.lower())
        check("MD&A excludes Item 7A",
              "quantitative and qualitative" not in mda.lower())
        check("MD&A length reasonable", 200 < len(mda) < 5000,
              f"len={len(mda)}")

    # Filing without MD&A
    plain_8k = strip_html(SAMPLE_8K_HTML)
    mda_8k = extract_mda(plain_8k)
    check("extract_mda returns None for 8-K", mda_8k is None)

    # XBRL-only filing
    plain_xbrl = strip_html(SAMPLE_XBRL_ONLY)
    mda_xbrl = extract_mda(plain_xbrl)
    check("extract_mda returns None for XBRL-only", mda_xbrl is None)


def test_token_truncation():
    section("3. Token Truncation")

    long_text = " ".join(["word"] * 1000)
    trunc = truncate_to_tokens(long_text, 512)
    word_count = len(trunc.split())
    max_words = int(512 / 1.3)
    check("truncate respects token limit",
          word_count <= max_words + 1,
          f"words={word_count}, max={max_words}")

    short_text = "just a few words"
    trunc_short = truncate_to_tokens(short_text, 512)
    check("truncate preserves short text", trunc_short == short_text)


def test_preprocess_pipeline():
    section("4. Preprocessing Pipeline (strip \u2192 MD&A \u2192 truncate)")

    # 10-K with MD&A
    text, mda_found = preprocess_filing(SAMPLE_10K_HTML, "10-K", 512)
    check("preprocess_filing returns text", len(text) > 50)
    check("preprocess_filing finds MD&A in 10-K", mda_found is True)

    # 8-K (no MD&A extraction attempted)
    text_8k, mda_8k = preprocess_filing(SAMPLE_8K_HTML, "8-K", 512)
    check("preprocess_filing returns text for 8-K", len(text_8k) > 20)
    check("preprocess_filing mda_found=False for 8-K", mda_8k is False)

    # XBRL-only
    text_xbrl, mda_xbrl = preprocess_filing(SAMPLE_XBRL_ONLY, "10-K", 512)
    check("preprocess_filing handles XBRL-only", text_xbrl is not None)

    # Empty text
    text_empty, _ = preprocess_filing("", "10-K", 512)
    check("preprocess_filing handles empty input", text_empty == "")


def test_lm_word_lists():
    section("5. Loughran-McDonald Word Lists")

    check("LM_POSITIVE is non-empty", len(LM_POSITIVE) > 50)
    check("LM_NEGATIVE is non-empty", len(LM_NEGATIVE) > 50)
    check("LM_UNCERTAINTY is non-empty", len(LM_UNCERTAINTY) > 30)
    check("ALL_LM_WORDS is union",
          ALL_LM_WORDS == LM_POSITIVE | LM_NEGATIVE | LM_UNCERTAINTY)

    # Known words
    check("'growth' in LM_POSITIVE", "growth" in LM_POSITIVE)
    check("'loss' in LM_NEGATIVE", "loss" in LM_NEGATIVE)
    check("'uncertain' in LM_UNCERTAINTY", "uncertain" in LM_UNCERTAINTY)

    # No overlap between pos and neg
    overlap = LM_POSITIVE & LM_NEGATIVE
    check("no overlap between positive and negative",
          len(overlap) == 0, f"overlap: {overlap}")


def test_lm_word_counts():
    section("6. LM Word Counts")

    text = ("Our growth exceeded expectations, delivering record earnings. "
            "However, losses from restructuring and litigation risks remain. "
            "We estimate uncertain future outcomes.")
    counts = lm_word_counts(text)

    check("lm_word_counts detects positives", counts["positive"] >= 3,
          f"pos={counts['positive']}")
    check("lm_word_counts detects negatives", counts["negative"] >= 2,
          f"neg={counts['negative']}")
    check("lm_word_counts detects uncertainty", counts["uncertainty"] >= 1,
          f"unc={counts['uncertainty']}")

    # Empty text
    counts_empty = lm_word_counts("")
    check("lm_word_counts handles empty", counts_empty["positive"] == 0)


def test_lm_sentence_filter():
    section("7. LM Sentence Filter")

    text = ("This is boring boilerplate text about nothing. "
            "Our growth strategy delivered strong results. "
            "The weather was nice on Tuesday. "
            "Losses from impairment were significant.")
    filtered = lm_sentence_filter(text)

    check("filter keeps sentiment sentences",
          "growth" in filtered.lower() or "strong" in filtered.lower())
    check("filter keeps negative sentences",
          "losses" in filtered.lower() or "impairment" in filtered.lower())
    check("filter removes boilerplate",
          "weather" not in filtered.lower())

    # All-boilerplate text returns original
    boring = "The quick brown fox jumped over the lazy dog."
    filtered_boring = lm_sentence_filter(boring)
    check("filter returns original when no LM words", filtered_boring == boring)


def test_finbert_mock_scorer():
    section("8. FinBERT Mock Scorer")

    scorer = FinBERTScorer(mock=True)
    check("mock scorer initialised", scorer.mock is True)

    # Positive text
    pos_result = scorer.score(
        "The company achieved record growth and exceeded all expectations. "
        "Revenue increased significantly with improved profitability.")
    check("mock positive: score > 0", pos_result["score"] > 0,
          f"score={pos_result['score']}")
    check("mock positive: label=positive",
          pos_result["label"] == "positive")
    check("mock positive: confidence in [0,1]",
          0 <= pos_result["confidence"] <= 1)

    # Negative text
    neg_result = scorer.score(
        "The company suffered significant losses due to declining revenue. "
        "Impairment charges and restructuring costs eroded profitability. "
        "Litigation risks and adverse regulatory changes threaten operations.")
    check("mock negative: score < 0", neg_result["score"] < 0,
          f"score={neg_result['score']}")
    check("mock negative: label=negative",
          neg_result["label"] == "negative")

    # Neutral / empty text
    neutral = scorer.score("")
    check("mock empty: score=0", neutral["score"] == 0.0)

    # Short text
    short = scorer.score("hello")
    check("mock short: returns valid dict",
          "score" in short and "confidence" in short)

    # Score bounds
    check("mock score in [-1, 1]",
          -1 <= pos_result["score"] <= 1 and -1 <= neg_result["score"] <= 1)


def test_score_single_filing():
    section("9. score_single_filing Integration")

    scorer = FinBERTScorer(mock=True)

    # 10-K with MD&A
    result = score_single_filing(SAMPLE_10K_HTML, "10-K", scorer, 512)
    check("score_single_filing returns dict", isinstance(result, dict))
    check("score has raw_score key", "raw_score" in result)
    check("score has confidence key", "confidence" in result)
    check("score has confidence_flag", "confidence_flag" in result)
    check("score has lm_positive", "lm_positive" in result)
    check("score has md_a_found", "md_a_found" in result)
    check("10-K MD&A found", result["md_a_found"] is True)
    check("10-K tokens_used > 0", result["tokens_used"] > 0,
          f"tokens={result['tokens_used']}")
    check("10-K raw_score in [-1,1]",
          -1 <= result["raw_score"] <= 1,
          f"score={result['raw_score']}")

    # 8-K
    result_8k = score_single_filing(SAMPLE_8K_HTML, "8-K", scorer, 512)
    check("8-K scores positive", result_8k["raw_score"] > 0,
          f"score={result_8k['raw_score']}")
    check("8-K MD&A not found", result_8k["md_a_found"] is False)

    # XBRL-only
    result_xbrl = score_single_filing(SAMPLE_XBRL_ONLY, "10-K", scorer, 512)
    check("XBRL-only returns valid score",
          -1 <= result_xbrl["raw_score"] <= 1)

    # Empty filing
    result_empty = score_single_filing("", "10-K", scorer, 512)
    check("empty filing score=0", result_empty["raw_score"] == 0.0)
    check("empty filing confidence=0", result_empty["confidence"] == 0.0)


def test_vix_percentile():
    section("10. VIX Percentile & Regime Tagging")

    # Build synthetic VIX series
    dates = pd.date_range("2025-01-01", periods=252, freq="B")
    np.random.seed(42)
    vix_vals = np.random.uniform(15, 30, 252)
    vix_series = pd.Series(vix_vals, index=dates)

    pct = compute_vix_percentile(vix_series)
    check("vix_percentile returns float", isinstance(pct, float))
    check("vix_percentile in [0, 100]", 0 <= pct <= 100, f"pct={pct}")

    # As-of date
    pct_past = compute_vix_percentile(vix_series, "2025-06-15")
    check("vix_percentile as_of returns float", isinstance(pct_past, float))

    # Empty series
    pct_empty = compute_vix_percentile(pd.Series(dtype=float))
    check("vix_percentile empty \u2192 NaN", math.isnan(pct_empty))

    # tag_vix_regime
    check("VIX 15 \u2192 low_vol", tag_vix_regime(15.0, vix_series) == "low_vol")
    check("VIX 25 \u2192 elevated", tag_vix_regime(25.0, vix_series) == "elevated")
    check("VIX 35 \u2192 high_vol", tag_vix_regime(35.0, vix_series) == "high_vol")
    check("VIX NaN \u2192 unknown",
          tag_vix_regime(float("nan"), vix_series) == "unknown")

    # check_drift_risk
    check("drift_risk: 80 >= 75 \u2192 True", check_drift_risk(80.0, 75.0) is True)
    check("drift_risk: 50 < 75 \u2192 False", check_drift_risk(50.0, 75.0) is False)
    check("drift_risk: NaN \u2192 False",
          check_drift_risk(float("nan"), 75.0) is False)


def test_confidence_flags():
    section("11. Confidence Flags")

    check("HIGH: conf\u22650.7, mda, n\u22653",
          compute_confidence_flag(0.85, 3, True) == "HIGH")
    check("MEDIUM: conf\u22650.5",
          compute_confidence_flag(0.6, 1, False) == "MEDIUM")
    check("MEDIUM: mda + n\u22651",
          compute_confidence_flag(0.3, 1, True) == "MEDIUM")
    check("LOW: conf<0.5, no mda",
          compute_confidence_flag(0.3, 0, False) == "LOW")


def test_sector_holdings():
    section("12. Sector Holdings Mapping")

    cfg = get_test_config()
    holdings = get_sector_holdings(cfg)

    check("11 sector ETFs mapped", len(holdings) == 11)

    for etf in ["XLK", "XLV", "XLE", "XLF", "XLI", "XLB",
                "XLU", "XLP", "XLRE", "XLC", "XLY"]:
        check(f"{etf} has holdings", etf in holdings and len(holdings[etf]) > 0)
        check(f"{etf} has ~5 holdings",
              len(holdings[etf]) >= 3, f"n={len(holdings[etf])}")

    # Check specific known holdings
    check("AAPL in XLK", "AAPL" in holdings.get("XLK", []))
    check("LLY in XLV", "LLY" in holdings.get("XLV", []))


def test_full_pipeline_synthetic_db():
    section("13. Full Pipeline with Synthetic DB")

    conn = create_test_db()
    cfg = get_test_config()
    scorer = FinBERTScorer(mock=True)

    # Score all filings
    scores_df = score_all_filings(conn, scorer, cfg)
    check("score_all_filings returns DataFrame",
          isinstance(scores_df, pd.DataFrame))
    check("scored 7 filings", len(scores_df) == 7,
          f"n={len(scores_df)}")

    # Check DB was populated
    db_count = conn.execute("SELECT COUNT(*) FROM nlp_scores").fetchone()[0]
    check("nlp_scores table populated", db_count == 7, f"rows={db_count}")

    # Check AAPL 10-K has MD&A
    aapl_10k = scores_df[
        (scores_df["ticker"] == "AAPL") & (scores_df["filing_type"] == "10-K")
    ]
    check("AAPL 10-K found in scores", len(aapl_10k) == 1)
    if len(aapl_10k) == 1:
        check("AAPL 10-K MD&A found",
              aapl_10k.iloc[0]["md_a_found"] == 1)
        check("AAPL 10-K has VIX regime tag",
              aapl_10k.iloc[0]["vix_regime"] in
              ("low_vol", "elevated", "high_vol", "unknown"))

    # Check XBRL-only filing handled gracefully
    xbrl_row = scores_df[scores_df["ticker"] == "XBRL_ONLY"]
    check("XBRL-only filing scored", len(xbrl_row) == 1)

    # Compute sector signals
    signals_df = compute_sector_signals(conn, cfg)
    check("compute_sector_signals returns DataFrame",
          isinstance(signals_df, pd.DataFrame))
    check("11 sector signals", len(signals_df) == 11,
          f"n={len(signals_df)}")

    # XLK should have data (AAPL, MSFT, NVDA are in holdings)
    xlk = signals_df[signals_df["sector_etf"] == "XLK"]
    check("XLK signal computed", len(xlk) == 1)
    if len(xlk) == 1:
        check("XLK n_filings > 0", xlk.iloc[0]["n_filings"] > 0,
              f"n={xlk.iloc[0]['n_filings']}")
        check("XLK sector_score in [-1,1]",
              -1 <= xlk.iloc[0]["sector_score"] <= 1)
        check("XLK has regime_weight",
              xlk.iloc[0]["regime_weight"] is not None)
        check("XLK has vix_percentile",
              xlk.iloc[0]["vix_percentile"] is not None)

    # XLV should have data (LLY, UNH)
    xlv = signals_df[signals_df["sector_etf"] == "XLV"]
    check("XLV signal computed", len(xlv) == 1)
    if len(xlv) == 1:
        check("XLV n_filings > 0", xlv.iloc[0]["n_filings"] > 0)

    # Drift risk check
    check("drift_risk field exists",
          "drift_risk" in signals_df.columns)
    check("drift_risk is 0 or 1",
          signals_df["drift_risk"].isin([0, 1]).all())

    # DB sector signals stored
    sig_count = conn.execute(
        "SELECT COUNT(*) FROM nlp_sector_signals"
    ).fetchone()[0]
    check("nlp_sector_signals table populated",
          sig_count == 11, f"rows={sig_count}")

    conn.close()


def test_regime_weight_logic():
    section("14. Regime-Weight Logic (Offense / Defense / Panic)")

    cfg = get_test_config()

    # Test each regime
    for regime, expected_weight in [("offense", 0.20), ("defense", 0.0),
                                     ("panic", 0.0)]:
        conn = create_test_db()
        # Override regime in signals table
        conn.execute("DELETE FROM signals")
        data = json.dumps({"dominant_regime": regime,
                           "wedge_volume_percentile": 50})
        conn.execute(
            "INSERT INTO signals VALUES (?, ?, ?, ?)",
            (dt.date.today().isoformat(), "regime_state", data, "now"),
        )
        conn.commit()

        scorer = FinBERTScorer(mock=True)
        score_all_filings(conn, scorer, cfg)
        signals_df = compute_sector_signals(conn, cfg)

        # Check weight \u2014 note: drift_risk may also zero out the weight
        xlk = signals_df[signals_df["sector_etf"] == "XLK"]
        if len(xlk) == 1:
            actual_weight = xlk.iloc[0]["regime_weight"]
            drift = xlk.iloc[0]["drift_risk"]
            if drift:
                # Drift risk overrides to 0
                check(f"{regime} regime: weight=0 (drift active)",
                      actual_weight == 0.0)
            else:
                check(f"{regime} regime: weight={expected_weight}",
                      abs(actual_weight - expected_weight) < 0.001,
                      f"actual={actual_weight}")
        conn.close()


def test_defense_panic_monitoring_only():
    section("15. Defense/Panic = Monitoring Only (weight=0)")

    cfg = get_test_config()

    for regime in ["defense", "panic"]:
        conn = create_test_db()
        # Set regime
        conn.execute("DELETE FROM signals")
        data = json.dumps({"dominant_regime": regime})
        conn.execute(
            "INSERT INTO signals VALUES (?, ?, ?, ?)",
            (dt.date.today().isoformat(), "regime_state", data, "now"),
        )
        # Set VIX low so drift_risk doesn't activate
        conn.execute("DELETE FROM prices WHERE ticker='^VIX'")
        base = dt.date.today() - dt.timedelta(days=260)
        for i in range(252):
            d = (base + dt.timedelta(days=i)).isoformat()
            conn.execute(
                "INSERT INTO prices VALUES (?,?,?,?,?,?,?,?,?,?)",
                (d, "^VIX", 16, 17, 15, 16, 16, 1000000, 0, "now"),
            )
        conn.commit()

        scorer = FinBERTScorer(mock=True)
        score_all_filings(conn, scorer, cfg)
        signals_df = compute_sector_signals(conn, cfg)

        all_zero = (signals_df["weighted_score"] == 0.0).all()
        check(f"{regime}: all weighted_scores = 0", all_zero)

        weights_zero = (signals_df["regime_weight"] == 0.0).all()
        check(f"{regime}: all regime_weights = 0", weights_zero)

        conn.close()


def test_drift_risk_flag():
    section("16. NLP Regime Drift Risk Flag")

    cfg = get_test_config()
    conn = create_test_db()

    # Set VIX very high for recent period to trigger drift
    conn.execute("DELETE FROM prices WHERE ticker='^VIX'")
    base = dt.date.today() - dt.timedelta(days=260)
    for i in range(252):
        d = (base + dt.timedelta(days=i)).isoformat()
        # All VIX at 25 except last 10 days at 40
        vix = 40 if i > 242 else 20
        conn.execute(
            "INSERT INTO prices VALUES (?,?,?,?,?,?,?,?,?,?)",
            (d, "^VIX", vix, vix, vix, vix, vix, 1000000, 0, "now"),
        )
    conn.commit()

    scorer = FinBERTScorer(mock=True)
    score_all_filings(conn, scorer, cfg)
    signals_df = compute_sector_signals(conn, cfg)

    # With VIX at 40 (top of distribution), percentile should be high
    vix_pct = signals_df["vix_percentile"].iloc[0]
    check("drift risk: VIX percentile > 75",
          vix_pct is not None and vix_pct > 75,
          f"pct={vix_pct}")

    drift_active = signals_df["drift_risk"].iloc[0] == 1
    check("drift risk: flag is 1 when VIX high", drift_active)

    if drift_active:
        check("drift risk: weight overridden to 0",
              signals_df["regime_weight"].iloc[0] == 0.0)

    conn.close()


def test_rolling_trend():
    section("17. Rolling 90-Day Sentiment Trend")

    cfg = get_test_config()
    conn = create_test_db()

    # Add filings at different dates to create a trend
    cur = conn.cursor()
    # Old filing (70 days ago) \u2014 negative
    old_date = (dt.date.today() - dt.timedelta(days=70)).isoformat()
    neg_html = ("<html><body><p>Significant losses, declining revenue, "
                "adverse conditions, impairment charges, and failure "
                "to meet targets. Deteriorating margins.</p></body></html>")
    cur.execute(
        "INSERT INTO filings VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("000", "AAPL", "Apple", "8-K", old_date,
         "0002-00", "doc.htm", "http://sec.gov", neg_html, "now"),
    )

    # Recent filing (10 days ago) \u2014 positive
    recent_date = (dt.date.today() - dt.timedelta(days=10)).isoformat()
    pos_html = ("<html><body><p>Record growth, exceeding expectations, "
                "improved profitability, strong momentum, and "
                "excellent performance across all segments.</p></body></html>")
    cur.execute(
        "INSERT INTO filings VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("000", "AAPL", "Apple", "8-K", recent_date,
         "0003-00", "doc2.htm", "http://sec.gov", pos_html, "now"),
    )
    conn.commit()

    scorer = FinBERTScorer(mock=True)
    score_all_filings(conn, scorer, cfg)
    signals_df = compute_sector_signals(conn, cfg)

    xlk = signals_df[signals_df["sector_etf"] == "XLK"]
    if len(xlk) == 1:
        trend = xlk.iloc[0]["rolling_trend"]
        check("rolling_trend computed", trend is not None)
        check("rolling_trend is numeric",
              isinstance(trend, (int, float)))
        # With positive recent + negative older, trend should be positive
        check("rolling_trend positive (improving)",
              trend >= 0.0, f"trend={trend}")

    conn.close()


def test_report_generation():
    section("18. Report Generation")

    conn = create_test_db()
    cfg = get_test_config()
    scorer = FinBERTScorer(mock=True)

    scores_df = score_all_filings(conn, scorer, cfg)
    signals_df = compute_sector_signals(conn, cfg)
    regime = fetch_latest_regime(conn)

    report = generate_nlp_report(scores_df, signals_df, regime)

    check("report is non-empty string",
          isinstance(report, str) and len(report) > 100)
    check("report contains header",
          "NLP SENTIMENT PIPELINE" in report)
    check("report contains regime",
          regime.upper() in report)
    check("report contains filing scores",
          "FILING-LEVEL SCORES" in report)
    check("report contains sector signals",
          "SECTOR NLP SIGNALS" in report)
    check("report contains VIX percentile",
          "VIX trailing" in report)

    # Empty DataFrames
    report_empty = generate_nlp_report(pd.DataFrame(), pd.DataFrame(), "offense")
    check("empty report generates OK",
          "no filings scored" in report_empty.lower())

    conn.close()


def test_live_db_integration():
    section("19. Live DB Integration (if available)")

    db_path = Path(__file__).parent / "rotation_system.db"
    if not db_path.exists():
        check("live DB exists (SKIP \u2014 not found)", True)
        return

    conn = get_db(db_path)
    cfg = get_test_config()
    scorer = FinBERTScorer(mock=True)

    # Check tables were created
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    check("nlp_scores table created", "nlp_scores" in tables)
    check("nlp_sector_signals table created",
          "nlp_sector_signals" in tables)

    # Score filings from live DB
    scores_df = score_all_filings(conn, scorer, cfg)
    check("live DB: filings scored",
          isinstance(scores_df, pd.DataFrame))
    check("live DB: scores >= 0 rows", len(scores_df) >= 0)

    if len(scores_df) > 0:
        check("live DB: all scores in [-1,1]",
              scores_df["raw_score"].between(-1, 1).all())
        check("live DB: all confidence in [0,1]",
              scores_df["confidence"].between(0, 1).all())

    # Sector signals
    signals_df = compute_sector_signals(conn, cfg)
    check("live DB: sector signals computed", len(signals_df) == 11)

    if len(signals_df) > 0:
        check("live DB: all sector_scores in [-1,1]",
              signals_df["sector_score"].between(-1, 1).all())

    # Regime
    regime = fetch_latest_regime(conn)
    check("live DB: regime retrieved",
          regime in ("offense", "defense", "panic"))

    conn.close()


def test_cli_flags():
    section("20. CLI Flags & Config")

    cfg = get_test_config()
    nlp_cfg = cfg.get("nlp", {})

    check("config: model is ProsusAI/finbert",
          nlp_cfg.get("model") == "ProsusAI/finbert")
    check("config: max_tokens = 512",
          nlp_cfg.get("max_tokens") == 512)
    check("config: rolling_sentiment_window = 90",
          nlp_cfg.get("rolling_sentiment_window") == 90)
    check("config: vix_drift_percentile = 75",
          nlp_cfg.get("vix_drift_percentile") == 75)

    rw = nlp_cfg.get("regime_weights", {})
    check("config: offense weight = 0.20",
          abs(rw.get("offense", 0) - 0.20) < 0.001)
    check("config: defense weight = 0.00",
          rw.get("defense", 0) == 0.0)
    check("config: panic weight = 0.00",
          rw.get("panic", 0) == 0.0)


def test_idempotency():
    section("21. Idempotency (re-run does not duplicate)")

    conn = create_test_db()
    cfg = get_test_config()
    scorer = FinBERTScorer(mock=True)

    # Run twice
    score_all_filings(conn, scorer, cfg)
    compute_sector_signals(conn, cfg)
    score_all_filings(conn, scorer, cfg)
    compute_sector_signals(conn, cfg)

    # Check no duplicates
    score_count = conn.execute(
        "SELECT COUNT(*) FROM nlp_scores"
    ).fetchone()[0]
    # 7 unique filings (date, ticker, filing_type)
    check("no duplicate nlp_scores after re-run",
          score_count == 7, f"rows={score_count}")

    sig_count = conn.execute(
        "SELECT COUNT(*) FROM nlp_sector_signals"
    ).fetchone()[0]
    check("no duplicate sector_signals after re-run",
          sig_count == 11, f"rows={sig_count}")

    conn.close()


def test_edge_cases():
    section("22. Edge Cases")

    scorer = FinBERTScorer(mock=True)

    # Very short text
    result = score_single_filing("<p>Hi</p>", "8-K", scorer, 512)
    check("very short filing: score=0", result["raw_score"] == 0.0)

    # All-numbers text
    nums = "<p>" + " ".join(["123.45"] * 100) + "</p>"
    result_nums = score_single_filing(nums, "8-K", scorer, 512)
    check("all-numbers filing: valid result",
          -1 <= result_nums["raw_score"] <= 1)

    # Unicode / special chars
    unicode_text = ("<html><body><p>Revenue grew 15% \u2014 exceeding "
                    "analysts' expectations. The company's \u20ac500M "
                    "investment achieved strong returns.</p></body></html>")
    result_uni = score_single_filing(unicode_text, "8-K", scorer, 512)
    check("unicode filing: scored OK",
          result_uni["raw_score"] != 0 or result_uni["confidence"] >= 0)

    # Massive text
    big_text = "<html><body>" + "<p>growth profit gain</p>" * 10000 + "</body></html>"
    result_big = score_single_filing(big_text, "10-K", scorer, 512)
    check("massive filing: scored without error",
          -1 <= result_big["raw_score"] <= 1)
    check("massive filing: tokens_used <= ~512",
          result_big["tokens_used"] <= 500,
          f"tokens={result_big['tokens_used']}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    global passed, failed

    print("=" * 60)
    print("  SMOKE TEST \u2014 Phase 4: NLP Sentiment Pipeline")
    print(f"  Date: {dt.date.today().isoformat()}")
    print(f"  Mode: MOCK (no FinBERT download)")
    print("=" * 60)

    test_html_stripping()
    test_mda_extraction()
    test_token_truncation()
    test_preprocess_pipeline()
    test_lm_word_lists()
    test_lm_word_counts()
    test_lm_sentence_filter()
    test_finbert_mock_scorer()
    test_score_single_filing()
    test_vix_percentile()
    test_confidence_flags()
    test_sector_holdings()
    test_full_pipeline_synthetic_db()
    test_regime_weight_logic()
    test_defense_panic_monitoring_only()
    test_drift_risk_flag()
    test_rolling_trend()
    test_report_generation()
    test_live_db_integration()
    test_cli_flags()
    test_idempotency()
    test_edge_cases()

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed "
          f"({passed + failed} total)")
    print(f"{'=' * 60}")

    if failed > 0:
        print("\n  FAILED CHECKS:")
        for e in errors:
            print(f"    \u2717 {e}")
        print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
