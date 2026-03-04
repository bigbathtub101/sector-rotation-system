"""
test_data_feeds.py — Tests for data ingestion pipeline.
=========================================================
Covers: DB schema, price validation, function existence.
"""

import sqlite3

import pytest


class TestDatabaseSchema:
    """Tests for database initialization and schema."""

    def test_required_tables_exist(self, db_path):
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        for tbl in ["prices", "macro_data", "filings", "signals", "allocations"]:
            assert tbl in tables, f"Table '{tbl}' missing from DB"

    def test_prices_populated(self, db_path):
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        conn.close()
        assert count > 100, f"Only {count} price rows"


class TestDataFeedsModule:
    """Tests for data_feeds module functions."""

    def test_module_imports(self):
        import data_feeds
        assert hasattr(data_feeds, "fetch_prices")
        assert hasattr(data_feeds, "fetch_macro_data")
        assert hasattr(data_feeds, "fetch_all_filings")
        assert hasattr(data_feeds, "run_full_ingestion")

    def test_init_database_creates_tables(self, tmp_path):
        from data_feeds import init_database
        db = tmp_path / "test.db"
        conn = init_database(db)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "prices" in tables
        assert "macro_data" in tables
        assert "signals" in tables
