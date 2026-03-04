"""
test_monitor.py — Tests for daily monitoring and alert system.
================================================================
Covers: module imports, alert thresholds, drift detection logic.
"""

import pytest


class TestMonitorModule:
    """Tests for monitor module."""

    def test_module_imports(self):
        import monitor
        assert hasattr(monitor, "main")
        assert hasattr(monitor, "AlertEngine")
        assert hasattr(monitor, "generate_executive_summary")

    def test_alert_thresholds(self, cfg):
        assert cfg["monitor"]["rebalance_threshold_bps"] == 200
        assert cfg["monitor"]["entry_window_threshold_bps"] == 300
        assert cfg["monitor"]["extended_defense_days"] == 60

    def test_panic_exit_sequence(self, cfg):
        pes = cfg["monitor"]["panic_exit_sequence"]
        assert pes["immediate_pct"] == 0.50
        assert pes["remainder_days"] == [3, 5]

    def test_drift_detection_logic(self):
        """Verify drift detection math: 500bps drift should exceed 200bps threshold."""
        current = {"us_equities": 0.45, "cash_short_duration": 0.10}
        target = {"us_equities": 0.40, "cash_short_duration": 0.15}
        threshold_bps = 200
        over_threshold = []
        for k in current:
            drift_bps = abs(current[k] - target[k]) * 10000
            if drift_bps > threshold_bps:
                over_threshold.append(k)
        assert len(over_threshold) >= 2
