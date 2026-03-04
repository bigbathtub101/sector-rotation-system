"""
test_dashboard.py — Tests for Streamlit dashboard.
=====================================================
Covers: module imports (with mocked streamlit), page functions, math helpers.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock streamlit before importing dashboard."""
    st_mock = MagicMock()
    sys.modules["streamlit"] = st_mock
    yield st_mock
    if "dashboard" in sys.modules:
        del sys.modules["dashboard"]
    if sys.modules.get("streamlit") is st_mock:
        del sys.modules["streamlit"]


class TestDashboardModule:
    """Tests for dashboard module."""

    def test_module_imports(self, mock_streamlit):
        import importlib
        if "dashboard" in sys.modules:
            del sys.modules["dashboard"]
        import dashboard
        assert hasattr(dashboard, "page_regime_dashboard")
        assert hasattr(dashboard, "page_portfolio_allocation")
        assert hasattr(dashboard, "page_stock_screener")

    def test_page_functions_exist(self, mock_streamlit):
        import importlib
        if "dashboard" in sys.modules:
            del sys.modules["dashboard"]
        import dashboard
        pages = [
            "page_regime_dashboard",
            "page_portfolio_allocation",
            "page_signal_detail",
            "page_stock_screener",
            "page_alerts_log",
            "page_backtester",
        ]
        for fn_name in pages:
            assert hasattr(dashboard, fn_name), f"Missing {fn_name}"

    def test_max_drawdown_helper(self, mock_streamlit):
        import importlib
        if "dashboard" in sys.modules:
            del sys.modules["dashboard"]
        import dashboard
        if hasattr(dashboard, "_max_drawdown"):
            series = pd.Series([100, 110, 105, 95, 100, 90])
            mdd = dashboard._max_drawdown(series)
            expected = (90 - 110) / 110 * 100
            assert abs(mdd - expected) < 0.5


class TestConfigIntegrity:
    """Tests for config.yaml integrity."""

    def test_mclean_pontiff_decay(self, cfg):
        assert cfg["factor_model"]["mclean_pontiff_decay"] == 0.74

    def test_portfolio_totals(self, cfg):
        taxable = cfg["portfolio"]["accounts"]["taxable"]["value"]
        roth = cfg["portfolio"]["accounts"]["roth_ira"]["value"]
        total = cfg["portfolio"]["total_value"]
        assert taxable + roth == total

    def test_no_hardcoded_secrets(self, cfg):
        import yaml
        config_str = yaml.dump(cfg)
        for bad in ["sk-", "ghp_", "xoxb-", "Bearer "]:
            assert bad not in config_str

    def test_cvar_confidence(self, cfg):
        assert cfg["optimizer"]["cvar_confidence"] == 0.95

    def test_monitor_run_time(self, cfg):
        assert cfg["monitor"]["run_time_et"] == "16:30"
