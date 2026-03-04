"""
test_nlp_sentiment.py — Tests for NLP sentiment pipeline.
============================================================
Covers: module imports, config validation, DB data presence.
"""

import sqlite3

import pytest


class TestNLPModule:
    """Tests for nlp_sentiment module."""

    def test_module_imports(self):
        import nlp_sentiment
        assert hasattr(nlp_sentiment, "score_single_filing")
        assert hasattr(nlp_sentiment, "score_all_filings")
        assert hasattr(nlp_sentiment, "compute_sector_signals")
        assert hasattr(nlp_sentiment, "generate_nlp_report")

    def test_nlp_config(self, cfg):
        assert cfg["nlp"]["model"] == "ProsusAI/finbert"
        assert cfg["nlp"]["max_tokens"] == 512
        assert cfg["nlp"]["rolling_sentiment_window"] == 90

    def test_nlp_scores_in_db(self, db_path):
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM nlp_scores").fetchone()[0]
        conn.close()
        assert count >= 1

    def test_nlp_sector_signals_in_db(self, db_path):
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM nlp_sector_signals").fetchone()[0]
        conn.close()
        assert count >= 5
