"""
test_stock_screener.py — Tests for stock screening and watchlist scoring.
==========================================================================
Covers: quality scoring, value scoring, size scoring, valuation filter,
momentum, watchlist config, entry/exit signal logic.
"""

import numpy as np
import pandas as pd
import pytest


class TestScoringFunctions:
    """Tests for individual scoring components."""

    def test_quality_score_bounded(self):
        from stock_screener import compute_quality_score
        info_good = {"roe": 0.30, "gross_margin": 0.60, "ocf_yield": 0.08}
        info_bad = {"roe": -0.10, "gross_margin": 0.15, "ocf_yield": 0.01}
        info_empty = {}
        assert 0.0 <= compute_quality_score(info_good) <= 1.0
        assert 0.0 <= compute_quality_score(info_bad) <= 1.0
        assert compute_quality_score(info_empty) == 0.5

    def test_value_score_inversely_related_to_pe(self):
        from stock_screener import compute_value_score
        low_pe = compute_value_score({"forward_pe": 10})
        high_pe = compute_value_score({"forward_pe": 40})
        assert low_pe > high_pe, "Lower P/E should have higher value score"

    def test_size_score_favors_smaller_caps(self):
        from stock_screener import compute_size_score
        small = compute_size_score(1000)   # $1B
        large = compute_size_score(500000)  # $500B
        assert small > large, "Smaller market cap should have higher size score"

    def test_valuation_filter_labels(self, cfg):
        from stock_screener import apply_valuation_filter
        cheap = apply_valuation_filter({"forward_pe": 10}, cfg)
        assert cheap == "FUNDAMENTAL_BUY"
        expensive = apply_valuation_filter({"forward_pe": 80}, cfg)
        assert expensive in ("MOMENTUM_ONLY", "AVOID")


class TestMomentum:
    """Tests for momentum computation."""

    def test_momentum_with_sufficient_data(self):
        from stock_screener import score_momentum_stock
        np.random.seed(42)
        prices = pd.DataFrame({
            "AAPL": np.cumsum(np.random.randn(300) * 0.02) + 150,
        })
        mom = score_momentum_stock(prices, "AAPL", lookback=252, skip=21)
        assert np.isfinite(mom), "Momentum should be finite with sufficient data"

    def test_momentum_nan_with_insufficient_data(self):
        from stock_screener import score_momentum_stock
        prices = pd.DataFrame({"AAPL": [100, 101, 102]})
        mom = score_momentum_stock(prices, "AAPL", lookback=252, skip=21)
        assert np.isnan(mom), "Momentum should be NaN with insufficient data"


class TestWatchlistConfig:
    """Tests for watchlist configuration in config.yaml."""

    @pytest.mark.parametrize("watchlist", [
        "watchlist_biotech",
        "watchlist_ai_software",
        "watchlist_defense",
        "watchlist_green_materials",
    ])
    def test_watchlist_has_minimum_tickers(self, cfg, watchlist):
        tickers = cfg["tickers"].get(watchlist, [])
        assert len(tickers) >= 5, f"{watchlist} has only {len(tickers)} tickers"

    def test_scoring_weights_sum_to_one(self, cfg):
        weights = cfg["stock_screener"]["scoring_weights"]
        w_sum = sum(weights.values())
        assert abs(w_sum - 1.0) < 0.01, f"Scoring weights sum to {w_sum}"


class TestEntryExitSignals:
    """Tests for entry/exit signal logic."""

    def test_entry_signals_require_offense_regime(self, cfg):
        from stock_screener import compute_entry_exit_signals
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "watchlist": ["ai_software", "ai_software"],
            "composite_score": [0.85, 0.75],
            "valuation_label": ["FUNDAMENTAL_BUY", "FUNDAMENTAL_BUY"],
            "momentum": [0.3, 0.2],
            "account": ["taxable", "taxable"],
        })
        signals = compute_entry_exit_signals(
            {"ai_software": df}, regime="defense", cfg=cfg,
        )
        # In defense regime, no entry signals should fire
        assert len(signals.get("entry", [])) == 0

    def test_exit_signals_on_avoid_label(self, cfg):
        from stock_screener import compute_entry_exit_signals
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "watchlist": ["ai_software"],
            "composite_score": [0.30],
            "valuation_label": ["AVOID"],
            "momentum": [-0.1],
            "account": ["taxable"],
        })
        signals = compute_entry_exit_signals(
            {"ai_software": df}, regime="offense", cfg=cfg,
        )
        exit_tickers = [s["ticker"] for s in signals.get("exit", [])]
        assert "AAPL" in exit_tickers
