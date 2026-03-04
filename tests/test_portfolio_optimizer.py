"""
test_portfolio_optimizer.py — Tests for CVaR portfolio optimization pipeline.
==============================================================================
Covers: factor scoring, momentum, allocation bands, CVaR optimizer,
tail correlations, dollar allocation, ETF quality filter.
"""

import numpy as np
import pandas as pd
import pytest


class TestFactorScoring:
    """Tests for Fama-French factor scoring."""

    def test_ff_factor_download_or_fallback(self):
        from portfolio_optimizer import download_ff_factors
        ff = download_ff_factors(start_date="2024-01-01")
        assert isinstance(ff, pd.DataFrame)
        assert len(ff) > 0
        assert "Mkt-RF" in ff.columns

    def test_normalize_signal_bounded(self):
        from portfolio_optimizer import _normalize_signal
        assert 0.0 <= _normalize_signal(0.0) <= 1.0
        assert 0.0 <= _normalize_signal(100.0) <= 1.0
        assert 0.0 <= _normalize_signal(-100.0) <= 1.0

    def test_momentum_computation(self, cfg):
        from portfolio_optimizer import compute_momentum
        np.random.seed(42)
        tickers = cfg["tickers"]["sector_etfs"]
        prices = pd.DataFrame(
            np.cumsum(np.random.randn(300, len(tickers)) * 0.01, axis=0) + 100,
            columns=tickers,
        )
        mom = compute_momentum(prices, lookback=252, skip=21)
        assert isinstance(mom, pd.Series)
        assert len(mom) == len(tickers)
        assert mom.min() >= 0.0 and mom.max() <= 1.0


class TestAllocationBands:
    """Tests for regime-conditional allocation bands."""

    def test_allocation_bands_valid_ranges(self, cfg):
        bands = cfg["optimizer"]["allocation_bands"]
        for regime in ["offense", "defense", "panic"]:
            for ac, band_cfg in bands.items():
                if ac == "vix_overlay_notional":
                    continue
                lo, hi = band_cfg[regime]
                assert 0 <= lo <= hi <= 1.0, f"Bad band: {ac}/{regime} = [{lo},{hi}]"

    def test_allocation_bands_allow_full_investment(self, cfg):
        bands = cfg["optimizer"]["allocation_bands"]
        for regime in ["offense", "defense", "panic"]:
            total_hi = sum(
                band_cfg[regime][1]
                for ac, band_cfg in bands.items()
                if ac != "vix_overlay_notional"
            )
            # Upper bounds should allow at least 100% investment
            assert total_hi >= 1.0, f"Upper bounds sum to {total_hi} in {regime}"


class TestDollarAllocation:
    """Tests for dollar allocation across accounts."""

    def test_allocate_dollars_returns_dict(self, cfg):
        from portfolio_optimizer import allocate_dollars
        sample_alloc = {
            "XLK": 0.20, "XLV": 0.15, "INDA": 0.10,
            "SGOV": 0.10, "XLE": 0.10, "XLF": 0.10,
            "XLI": 0.10, "VGK": 0.05, "XBI": 0.05, "BOTZ": 0.05,
        }
        result = allocate_dollars(sample_alloc, cfg)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestETFQualityFilter:
    """Tests for the ETF structural quality filter."""

    def test_expense_ratio_penalty_applied(self, cfg):
        from portfolio_optimizer import apply_etf_quality_filter
        weights = {"KWEB": 0.10, "XLK": 0.10, "SGOV": 0.10}
        filtered = apply_etf_quality_filter(weights, cfg)
        # KWEB has 69 bps expense ratio > 50 bps cap -> should be penalized
        assert filtered.get("KWEB", 0) <= weights["KWEB"]

    def test_quality_filter_preserves_total_weight(self, cfg):
        from portfolio_optimizer import apply_etf_quality_filter
        weights = {"XLK": 0.30, "XLV": 0.30, "SGOV": 0.20, "XLE": 0.20}
        filtered = apply_etf_quality_filter(weights, cfg)
        total = sum(filtered.values())
        assert abs(total - 1.0) < 0.05, f"Total weight after filter: {total}"


class TestCovarianceAndTailCorrelation:
    """Tests for Ledoit-Wolf shrinkage and tail correlations."""

    def test_shrunk_covariance_positive_definite(self, sample_returns):
        from portfolio_optimizer import compute_shrunk_covariance
        cov = compute_shrunk_covariance(sample_returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert (eigenvalues > -1e-10).all(), "Covariance should be positive semi-definite"

    def test_tail_correlations_symmetric(self, sample_returns, cfg):
        from portfolio_optimizer import compute_tail_correlations
        em_tickers = [t for t in cfg["tickers"]["geographic_etfs"] if t in sample_returns.columns]
        if not em_tickers:
            pytest.skip("No EM tickers in sample returns")
        corr, diag = compute_tail_correlations(
            sample_returns, em_tickers,
            percentile=cfg["optimizer"]["tail_correlation_percentile"],
        )
        # Check symmetry
        assert np.allclose(corr.values, corr.values.T, atol=1e-10)
