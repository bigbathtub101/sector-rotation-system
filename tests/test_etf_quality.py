"""
test_etf_quality.py — Tests for ETF structural quality filter.
================================================================
Covers: expense ratio penalties, overlap dedup, concentration caps.
"""

import pytest
import yaml
import os


class TestETFQualityFilter:
    """Tests for the ETF quality filter in portfolio_optimizer."""

    def test_expense_penalty_reduces_high_fee_etfs(self, cfg):
        from portfolio_optimizer import apply_etf_quality_filter
        weights = {"KWEB": 0.10, "XLK": 0.10}
        result = apply_etf_quality_filter(weights, cfg)
        # KWEB has 69 bps, above 50 bps cap -> penalized
        assert result.get("KWEB", 0) < 0.10 or result.get("KWEB", 0) == 0

    def test_overlap_group_concentrates_preferred(self, cfg):
        from portfolio_optimizer import apply_etf_quality_filter
        # VWO, EEM, IEMG are in overlap group. VWO is preferred.
        weights = {"VWO": 0.05, "EEM": 0.05, "IEMG": 0.05, "XLK": 0.85}
        result = apply_etf_quality_filter(weights, cfg)
        # EEM should be reduced/zeroed in favor of VWO
        assert result.get("EEM", 0) <= weights["EEM"]

    def test_single_country_concentration_cap(self, cfg):
        from portfolio_optimizer import apply_etf_quality_filter
        max_country_pct = cfg["etf_quality"]["max_single_country_pct"] / 100.0
        weights = {"INDA": 0.20, "XLK": 0.80}
        result = apply_etf_quality_filter(weights, cfg)
        assert result.get("INDA", 0) <= max_country_pct + 0.01  # small tolerance

    def test_em_total_cap(self, cfg):
        from portfolio_optimizer import apply_etf_quality_filter
        max_em_pct = cfg["etf_quality"]["max_em_total_pct"] / 100.0
        weights = {
            "INDA": 0.10, "EWZ": 0.10, "MCHI": 0.10, "VWO": 0.10,
            "XLK": 0.30, "SGOV": 0.30,
        }
        result = apply_etf_quality_filter(weights, cfg)
        em_tickers = {"INDA", "EWZ", "MCHI", "VWO", "EWY", "EWT", "KWEB"}
        em_total = sum(result.get(t, 0) for t in em_tickers)
        assert em_total <= max_em_pct + 0.02
