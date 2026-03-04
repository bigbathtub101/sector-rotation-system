"""
test_regime_detector.py — Tests for regime detection pipeline.
================================================================
Covers: wedge volume computation, regime classification, confirmation
filter, fast shock indicator, probability scoring.
"""

import numpy as np
import pandas as pd
import pytest


class TestWedgeVolume:
    """Tests for wedge volume series computation."""

    def test_compute_wedge_volume_series_returns_series(self, cfg):
        from regime_detector import compute_wedge_volume_series
        dummy_returns = pd.DataFrame(
            np.random.randn(252, 11) * 0.015,
            columns=cfg["tickers"]["sector_etfs"],
        )
        wv = compute_wedge_volume_series(dummy_returns, window=63)
        assert isinstance(wv, pd.Series)
        assert len(wv) > 0

    def test_wedge_volume_nonnegative(self, cfg):
        from regime_detector import compute_wedge_volume_series
        np.random.seed(42)
        dummy_returns = pd.DataFrame(
            np.random.randn(300, 11) * 0.015,
            columns=cfg["tickers"]["sector_etfs"],
        )
        wv = compute_wedge_volume_series(dummy_returns, window=63)
        valid = wv.dropna()
        assert (valid >= 0).all(), "Wedge volume should be non-negative"


class TestRegimeClassification:
    """Tests for regime probability scoring and classification."""

    def test_regime_probabilities_sum_to_one(self, cfg):
        from regime_detector import compute_regime_probabilities
        for pct in [2, 5, 10, 20, 30, 50, 70, 90]:
            probs = compute_regime_probabilities(pct, cfg)
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total} at pct={pct}"

    def test_panic_regime_at_low_percentile(self, cfg):
        from regime_detector import compute_regime_probabilities, get_dominant_regime
        probs = compute_regime_probabilities(2.0, cfg)
        assert get_dominant_regime(probs) == "panic"

    def test_offense_regime_at_high_percentile(self, cfg):
        from regime_detector import compute_regime_probabilities, get_dominant_regime
        probs = compute_regime_probabilities(50.0, cfg)
        assert get_dominant_regime(probs) == "offense"

    def test_nan_percentile_returns_zeros(self, cfg):
        from regime_detector import compute_regime_probabilities
        probs = compute_regime_probabilities(float("nan"), cfg)
        assert all(v == 0.0 for v in probs.values())


class TestConfirmationFilter:
    """Tests for the 2-day confirmation filter."""

    def test_confirmation_filter_prevents_single_day_flip(self):
        from regime_detector import apply_confirmation_filter
        data = pd.DataFrame({
            "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            "dominant_regime": ["offense", "offense", "defense", "offense", "offense"],
        })
        result = apply_confirmation_filter(data, consecutive_days=2)
        # The single-day defense blip should not be confirmed
        assert result["confirmed_regime"].iloc[-1] == "offense"

    def test_confirmation_filter_confirms_after_two_days(self):
        from regime_detector import apply_confirmation_filter
        data = pd.DataFrame({
            "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "dominant_regime": ["offense", "defense", "defense", "defense"],
        })
        result = apply_confirmation_filter(data, consecutive_days=2)
        assert result["confirmed_regime"].iloc[-1] == "defense"


class TestFastShockIndicator:
    """Tests for the VIX/RV fast shock indicator."""

    def test_fast_shock_high_when_ratio_exceeds_threshold(self):
        from regime_detector import compute_fast_shock_indicator
        vix = pd.Series([30.0, 30.0, 30.0], index=pd.date_range("2026-01-01", periods=3))
        rv = pd.Series([0.10, 0.10, 0.10], index=pd.date_range("2026-01-01", periods=3))
        result = compute_fast_shock_indicator(vix, rv, threshold=1.5)
        # VIX=30, RV=10% -> ratio=3.0 > 1.5 -> HIGH
        assert (result["fast_shock_risk"] == "high").all()

    def test_fast_shock_low_in_normal_conditions(self):
        from regime_detector import compute_fast_shock_indicator
        vix = pd.Series([15.0, 15.0, 15.0], index=pd.date_range("2026-01-01", periods=3))
        rv = pd.Series([0.12, 0.12, 0.12], index=pd.date_range("2026-01-01", periods=3))
        result = compute_fast_shock_indicator(vix, rv, threshold=1.5)
        # VIX=15, RV=12% -> ratio=1.25 < 1.5 -> LOW
        assert (result["fast_shock_risk"] == "low").all()


class TestPercentileNormalization:
    """Tests for percentile computation — critical for lookahead bias."""

    def test_percentile_uses_trailing_window_only(self, cfg):
        from regime_detector import compute_wedge_volume_percentile
        np.random.seed(42)
        wv = pd.Series(np.random.rand(500), name="wedge_volume")
        pct = compute_wedge_volume_percentile(wv, lookback=252)
        # Only values at index >= lookback should be non-NaN
        assert pct.iloc[:252].isna().all() or pct.iloc[:252].notna().sum() <= 1
        assert pct.iloc[252:].notna().sum() > 0
