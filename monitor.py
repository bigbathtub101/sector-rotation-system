"""
monitor.py — Phase 5: Monitoring Engine & Alert System
========================================================
Global Sector Rotation System

The master orchestrator that runs daily at 4:30 PM ET, ties every prior
phase together, and produces the Executive Summary a human reads in
under 60 seconds.

Daily Run Sequence
-------------------
1. Pull fresh price data for all ETFs and macro series
2. Recompute wedge volume percentile and regime probabilities
3. Check Fast Shock Risk indicator
4. If regime changed (confirmed): rerun CVaR optimizer with new bands
5. If new allocation drifts > 200 bps in any position: flag REBALANCE
6. Pull latest SEC filings (new 8-Ks today) and score sentiment
7. Log all outputs to rotation_system.db
8. Generate daily report JSON + human-readable Executive Summary

Alert Types
-----------
* REBALANCE        — regime transition + allocation drift > 200 bp
* FAST_SHOCK       — VIX / 21-day RV ratio > 1.5
* ENTRY_WINDOW     — underweight > 300 bp + factor score top quartile
* PANIC_PROTOCOL   — regime = Panic confirmed (staged exit sequence)
* EXTENDED_DEFENSE — 60+ days in Defense/Panic without crisis confirmation

Dependencies: data_feeds, regime_detector, portfolio_optimizer,
              stock_screener, nlp_sentiment, plus delivery (Telegram,
              email, Google Sheets) — all optional/graceful-degrade.
"""

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
import smtplib
import sqlite3
import sys
import traceback
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent
LOG_FILE = LOG_DIR / "monitor.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("monitor")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"
DB_PATH = Path(__file__).parent / "rotation_system.db"
ALERTS_JSON = Path(__file__).parent / "alerts.json"
ALERTS_CSV = Path(__file__).parent / "alerts_history.csv"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

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
    """Connect to the rotation system database."""
    conn = sqlite3.connect(str(db_path))
    # Ensure monitor_runs table exists for audit trail
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monitor_runs (
            run_id      TEXT PRIMARY KEY,
            date        TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            finished_at TEXT,
            status      TEXT DEFAULT 'running',
            regime      TEXT,
            alerts_json TEXT,
            report_text TEXT
        )
    """)
    conn.commit()
    return conn


def fetch_latest_allocation(conn: sqlite3.Connection) -> Optional[Dict]:
    """Return the most recent allocation from the DB."""
    row = conn.execute(
        "SELECT date, regime, allocation_json, dollar_taxable, dollar_roth "
        "FROM allocations ORDER BY date DESC LIMIT 1"
    ).fetchone()
    if row:
        return {
            "date": row[0],
            "regime": row[1],
            "allocations": json.loads(row[2]) if row[2] else {},
            "taxable_dollars": json.loads(row[3]) if row[3] else {},
            "roth_dollars": json.loads(row[4]) if row[4] else {},
        }
    return None


def fetch_latest_regime(conn: sqlite3.Connection) -> Dict:
    """Return the most recent regime state from signals table."""
    row = conn.execute(
        "SELECT date, signal_data FROM signals "
        "WHERE signal_type = 'regime_state' "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    if row:
        data = json.loads(row[1])
        data["date"] = row[0]
        return data
    return {
        "date": None,
        "dominant_regime": "offense",
        "wedge_volume_percentile": 50.0,
        "regime_probabilities": {"panic": 0.0, "defense": 0.0, "offense": 1.0},
        "fast_shock_risk": "low",
        "vix_rv_ratio": 0.0,
        "consecutive_days_in_regime": 0,
        "regime_confirmed": False,
    }


def fetch_regime_history(conn: sqlite3.Connection,
                         days: int = 90) -> pd.DataFrame:
    """Return regime states for the trailing N calendar days."""
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    return pd.read_sql_query(
        "SELECT date, signal_data FROM signals "
        "WHERE signal_type = 'regime_state' AND date >= ? "
        "ORDER BY date ASC",
        conn, params=[cutoff],
    )


def count_consecutive_defensive_days(conn: sqlite3.Connection) -> int:
    """
    Count how many consecutive calendar days the system has been in
    Defense or Panic regime (for Extended Defense Alert).
    """
    rows = conn.execute(
        "SELECT date, signal_data FROM signals "
        "WHERE signal_type = 'regime_state' "
        "ORDER BY date DESC"
    ).fetchall()
    count = 0
    for row in rows:
        data = json.loads(row[1])
        regime = data.get("dominant_regime", "offense")
        if regime in ("defense", "panic"):
            count += 1
        else:
            break
    return count


def fetch_nlp_sector_signals(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the latest NLP sector signals."""
    return pd.read_sql_query(
        "SELECT * FROM nlp_sector_signals "
        "ORDER BY date DESC LIMIT 11",
        conn,
    )


def fetch_latest_factor_signals(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the latest factor scores from signals table."""
    rows = conn.execute(
        "SELECT date, signal_data FROM signals "
        "WHERE signal_type = 'factor_scores' "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    if rows:
        data = json.loads(rows[1])
        return pd.DataFrame(data.get("sector_scores", []))
    return pd.DataFrame()


# ===========================================================================
# 2. PHASE ORCHESTRATION (graceful-degrade wrappers)
# ===========================================================================

def run_data_refresh(conn: sqlite3.Connection, cfg: dict,
                     mock: bool = False) -> Dict[str, Any]:
    """
    Step 1: Pull fresh price data for all ETFs and macro series.
    Wraps data_feeds.py.  Graceful-degrade: returns status dict.
    """
    result = {"step": "data_refresh", "status": "ok", "details": {}}
    if mock:
        result["details"]["mode"] = "mock"
        result["details"]["tickers_refreshed"] = 0
        logger.info("Data refresh: MOCK mode — skipping live pull.")
        return result

    try:
        from data_feeds import (
            fetch_all_prices, fetch_macro_data, fetch_sec_filings
        )
        prices = fetch_all_prices(conn, cfg)
        result["details"]["prices_rows"] = len(prices) if prices is not None else 0

        macro = fetch_macro_data(conn, cfg)
        result["details"]["macro_rows"] = len(macro) if macro is not None else 0

        # SEC filings — only new 8-Ks
        filings = fetch_sec_filings(conn, cfg, filing_types=["8-K"])
        result["details"]["new_filings"] = len(filings) if filings is not None else 0

    except ImportError:
        result["status"] = "skip"
        result["details"]["reason"] = "data_feeds module not importable"
        logger.warning("data_feeds not available — skipping data refresh.")
    except Exception as exc:
        result["status"] = "error"
        result["details"]["error"] = str(exc)
        logger.error("Data refresh failed: %s", exc)
    return result


def run_regime_detection(conn: sqlite3.Connection, cfg: dict,
                         mock: bool = False) -> Dict[str, Any]:
    """
    Step 2-3: Recompute wedge volume percentile, regime probabilities,
    and Fast Shock Risk indicator.
    """
    result = {"step": "regime_detection", "status": "ok", "regime_state": {}}
    if mock:
        # Return whatever is in the DB already
        regime = fetch_latest_regime(conn)
        result["regime_state"] = regime
        result["details"] = {"mode": "mock"}
        return result

    try:
        from regime_detector import (
            compute_daily_regime, get_db as regime_get_db
        )
        regime_state = compute_daily_regime(conn, cfg)
        result["regime_state"] = regime_state
    except ImportError:
        result["status"] = "skip"
        regime = fetch_latest_regime(conn)
        result["regime_state"] = regime
        result["details"] = {"reason": "regime_detector not importable"}
        logger.warning("regime_detector not available — using last known regime.")
    except Exception as exc:
        result["status"] = "error"
        regime = fetch_latest_regime(conn)
        result["regime_state"] = regime
        result["details"] = {"error": str(exc)}
        logger.error("Regime detection failed: %s", exc)
    return result


def run_optimizer(conn: sqlite3.Connection, cfg: dict,
                  regime: str,
                  mock: bool = False) -> Dict[str, Any]:
    """
    Step 4: If regime changed, rerun CVaR optimizer with new bands.
    """
    result = {"step": "optimizer", "status": "ok", "allocation": {}}

    prev_alloc = fetch_latest_allocation(conn)
    prev_regime = prev_alloc["regime"] if prev_alloc else None
    regime_changed = prev_regime != regime

    result["regime_changed"] = regime_changed

    if mock or not regime_changed:
        if prev_alloc:
            result["allocation"] = prev_alloc
        else:
            result["allocation"] = _default_allocation(cfg, regime)
        if not regime_changed:
            result["details"] = {"reason": "regime unchanged — using existing allocation"}
        else:
            result["details"] = {"mode": "mock"}
        return result

    try:
        from portfolio_optimizer import run_optimization
        alloc = run_optimization(conn, cfg, regime)
        result["allocation"] = alloc
    except ImportError:
        result["status"] = "skip"
        result["allocation"] = prev_alloc or _default_allocation(cfg, regime)
        result["details"] = {"reason": "portfolio_optimizer not importable"}
        logger.warning("portfolio_optimizer not available — using last allocation.")
    except Exception as exc:
        result["status"] = "error"
        result["allocation"] = prev_alloc or _default_allocation(cfg, regime)
        result["details"] = {"error": str(exc)}
        logger.error("Optimizer failed: %s", exc)
    return result


def run_nlp_scoring(conn: sqlite3.Connection, cfg: dict,
                    mock: bool = False) -> Dict[str, Any]:
    """
    Step 6: Pull latest SEC filings and score sentiment.
    """
    result = {"step": "nlp_scoring", "status": "ok", "details": {}}
    if mock:
        result["details"]["mode"] = "mock"
        logger.info("NLP scoring: MOCK mode — skipping.")
        return result

    try:
        from nlp_sentiment import (
            FinBERTScorer, score_all_filings, compute_sector_signals
        )
        scorer = FinBERTScorer(mock=True)  # Always mock in daily run for CI
        scores_df = score_all_filings(conn, scorer, cfg)
        signals_df = compute_sector_signals(conn, cfg)
        result["details"]["filings_scored"] = len(scores_df)
        result["details"]["sector_signals"] = len(signals_df)
    except ImportError:
        result["status"] = "skip"
        result["details"]["reason"] = "nlp_sentiment not importable"
        logger.warning("nlp_sentiment not available — skipping NLP scoring.")
    except Exception as exc:
        result["status"] = "error"
        result["details"]["error"] = str(exc)
        logger.error("NLP scoring failed: %s", exc)
    return result


def _default_allocation(cfg: dict, regime: str) -> Dict:
    """
    Build a sensible default allocation from config bands when
    the optimizer is not available.  Uses midpoints of the regime bands.
    """
    bands = cfg.get("optimizer", {}).get("allocation_bands", {})
    total = cfg.get("portfolio", {}).get("total_value", 144000)
    taxable = cfg.get("portfolio", {}).get("accounts", {}).get("taxable", {}).get("value", 100000)
    roth = cfg.get("portfolio", {}).get("accounts", {}).get("roth_ira", {}).get("value", 44000)

    alloc = {}
    for asset_class, regime_bands in bands.items():
        band = regime_bands.get(regime, [0.0, 0.0])
        midpoint = (band[0] + band[1]) / 2.0
        alloc[asset_class] = round(midpoint, 4)

    # Normalize to sum to 1.0
    total_weight = sum(alloc.values())
    if total_weight > 0:
        alloc = {k: round(v / total_weight, 4) for k, v in alloc.items()}

    # Compute dollar amounts
    alloc_dollars = {k: round(v * total, 2) for k, v in alloc.items()}

    # Simple tax-location: cash/ETFs to taxable, thematic to Roth
    taxable_dollars = {}
    roth_dollars = {}
    roth_classes = {"healthcare", "vix_overlay_notional"}
    taxable_remaining = taxable
    roth_remaining = roth

    for asset, dollars in alloc_dollars.items():
        if asset in roth_classes and roth_remaining > 0:
            placed = min(dollars, roth_remaining)
            roth_dollars[asset] = placed
            roth_remaining -= placed
            if dollars > placed:
                taxable_dollars[asset] = dollars - placed
                taxable_remaining -= (dollars - placed)
        else:
            placed = min(dollars, taxable_remaining)
            taxable_dollars[asset] = placed
            taxable_remaining -= placed
            if dollars > placed:
                roth_dollars[asset] = dollars - placed
                roth_remaining -= (dollars - placed)

    return {
        "date": dt.date.today().isoformat(),
        "regime": regime,
        "allocations": alloc,
        "taxable_dollars": taxable_dollars,
        "roth_dollars": roth_dollars,
    }


# ===========================================================================
# 3. ALERT DETECTION
# ===========================================================================

class AlertEngine:
    """
    Evaluates all alert conditions against current system state.
    Returns a list of alert dicts.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        mon = cfg.get("monitor", {})
        self.rebalance_bps = mon.get("rebalance_threshold_bps", 200)
        self.entry_window_bps = mon.get("entry_window_threshold_bps", 300)
        self.extended_defense_days = mon.get("extended_defense_days", 60)
        self.panic_exit = mon.get("panic_exit_sequence", {})
        fast = cfg.get("regime", {}).get("fast_shock", {})
        self.vix_rv_threshold = fast.get("vix_rv_ratio_threshold", 1.5)
        prob = cfg.get("probabilistic_triggers", {})
        self.whipsaw_floor = prob.get("whipsaw_buffer_percentile", 3)

    def evaluate(self,
                 regime_state: Dict,
                 prev_allocation: Optional[Dict],
                 new_allocation: Dict,
                 regime_changed: bool,
                 consecutive_defensive_days: int) -> List[Dict]:
        """Run all alert checks and return list of triggered alerts."""
        alerts: List[Dict] = []
        now = dt.datetime.now().isoformat()

        # --- FAST SHOCK ALERT ---
        vix_rv = regime_state.get("vix_rv_ratio", 0.0)
        if isinstance(vix_rv, (int, float)) and vix_rv > self.vix_rv_threshold:
            alerts.append({
                "type": "FAST_SHOCK",
                "severity": "HIGH",
                "timestamp": now,
                "message": (
                    f"Fast Shock Risk HIGH — VIX/RV ratio {vix_rv:.2f} "
                    f"exceeds {self.vix_rv_threshold:.1f} threshold. "
                    f"Override standing put spread sizing immediately."
                ),
                "data": {"vix_rv_ratio": vix_rv,
                         "threshold": self.vix_rv_threshold},
            })

        # --- PANIC PROTOCOL ALERT ---
        dominant = regime_state.get("dominant_regime", "offense")
        confirmed = regime_state.get("regime_confirmed", False)
        if dominant == "panic" and confirmed:
            imm_pct = self.panic_exit.get("immediate_pct", 0.50)
            rem_days = self.panic_exit.get("remainder_days", [3, 5])
            alerts.append({
                "type": "PANIC_PROTOCOL",
                "severity": "CRITICAL",
                "timestamp": now,
                "message": (
                    f"PANIC REGIME CONFIRMED — Execute staged exit: "
                    f"{imm_pct:.0%} immediate reduction, "
                    f"remainder over {rem_days[0]}–{rem_days[1]} days. "
                    f"Move to {(1-imm_pct):.0%} of current equity positions TODAY."
                ),
                "data": {"immediate_pct": imm_pct,
                         "remainder_days": rem_days},
            })

        # --- REBALANCE ALERT ---
        if regime_changed and prev_allocation:
            max_drift = self._compute_max_drift(
                prev_allocation.get("allocations", {}),
                new_allocation.get("allocations", {}),
            )
            if max_drift >= self.rebalance_bps:
                alerts.append({
                    "type": "REBALANCE",
                    "severity": "HIGH",
                    "timestamp": now,
                    "message": (
                        f"Regime transition detected — max allocation drift "
                        f"{max_drift:.0f} bps exceeds {self.rebalance_bps} bps "
                        f"threshold. REBALANCE REQUIRED."
                    ),
                    "data": {"max_drift_bps": max_drift,
                             "threshold_bps": self.rebalance_bps,
                             "from_regime": prev_allocation.get("regime"),
                             "to_regime": new_allocation.get("regime")},
                })

        # --- ENTRY WINDOW ALERT ---
        if dominant == "offense" and confirmed and prev_allocation:
            entry_alerts = self._check_entry_windows(
                prev_allocation.get("allocations", {}),
                new_allocation.get("allocations", {}),
            )
            alerts.extend(entry_alerts)

        # --- EXTENDED DEFENSE ALERT ---
        if consecutive_defensive_days >= self.extended_defense_days:
            # Check for further wedge volume deterioration
            wv_pct = regime_state.get("wedge_volume_percentile", 50)
            no_crisis = wv_pct > self.whipsaw_floor
            if no_crisis:
                alerts.append({
                    "type": "EXTENDED_DEFENSE",
                    "severity": "MEDIUM",
                    "timestamp": now,
                    "message": (
                        f"System in Defense/Panic for {consecutive_defensive_days} "
                        f"consecutive days WITHOUT crisis confirmation "
                        f"(wedge volume at {wv_pct:.1f}th percentile, "
                        f"above {self.whipsaw_floor}rd floor). "
                        f"Consider partial re-entry at lower Offense allocation band. "
                        f"Review whether signal reflects structural crisis or "
                        f"prolonged orderly correction."
                    ),
                    "data": {"days": consecutive_defensive_days,
                             "wedge_volume_pct": wv_pct,
                             "crisis_floor": self.whipsaw_floor},
                })

        return alerts

    def _compute_max_drift(self, prev: Dict, curr: Dict) -> float:
        """Return the maximum allocation drift in basis points."""
        all_keys = set(prev.keys()) | set(curr.keys())
        max_drift = 0.0
        for k in all_keys:
            old = prev.get(k, 0.0) or 0.0
            new = curr.get(k, 0.0) or 0.0
            drift = abs(new - old) * 10000  # to bps
            max_drift = max(max_drift, drift)
        return max_drift

    def _check_entry_windows(self, prev: Dict, target: Dict) -> List[Dict]:
        """Check for ENTRY_WINDOW conditions."""
        now = dt.datetime.now().isoformat()
        alerts = []
        for asset, target_weight in target.items():
            current = prev.get(asset, 0.0) or 0.0
            target_w = target_weight or 0.0
            if target_w > current:
                underweight_bps = (target_w - current) * 10000
                if underweight_bps >= self.entry_window_bps:
                    alerts.append({
                        "type": "ENTRY_WINDOW",
                        "severity": "MEDIUM",
                        "timestamp": now,
                        "message": (
                            f"Entry window: {asset} is underweight by "
                            f"{underweight_bps:.0f} bps (current {current:.1%} "
                            f"vs target {target_w:.1%}) in Offense regime."
                        ),
                        "data": {"asset": asset,
                                 "current_weight": current,
                                 "target_weight": target_w,
                                 "underweight_bps": underweight_bps},
                    })
        return alerts


# ===========================================================================
# 4. ALERT DELIVERY
# ===========================================================================

def write_alerts_json(alerts: List[Dict], path: Path = ALERTS_JSON) -> None:
    """Write alerts to alerts.json (for GitHub Actions to read)."""
    payload = {
        "generated_at": dt.datetime.now().isoformat(),
        "count": len(alerts),
        "alerts": alerts,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote %d alerts to %s", len(alerts), path)


def append_alerts_csv(alerts: List[Dict], path: Path = ALERTS_CSV) -> None:
    """Append alerts to alerts_history.csv."""
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "type", "severity", "message"])
        for a in alerts:
            writer.writerow([
                a.get("timestamp", ""),
                a.get("type", ""),
                a.get("severity", ""),
                a.get("message", ""),
            ])
    logger.info("Appended %d alerts to %s", len(alerts), path)


def send_telegram(alerts: List[Dict], report_summary: str) -> bool:
    """
    Send alert via Telegram Bot API.
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.info("Telegram credentials not set — skipping.")
        return False

    try:
        import urllib.request
        import urllib.parse

        # Build message
        if alerts:
            msg_parts = [f"🚨 {len(alerts)} ALERT(S)"]
            for a in alerts:
                msg_parts.append(f"\n[{a['severity']}] {a['type']}: {a['message'][:200]}")
        else:
            msg_parts = ["✅ No alerts today."]
        msg_parts.append(f"\n\n📊 Daily regime: {report_summary[:500]}")
        text = "\n".join(msg_parts)[:4000]  # Telegram limit

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }).encode()
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=10)
        logger.info("Telegram message sent.")
        return True
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)
        return False


def send_email(alerts: List[Dict], report_text: str) -> bool:
    """
    Send alert email via Gmail SMTP.
    Requires GMAIL_USERNAME and GMAIL_PASSWORD env vars.
    """
    username = os.environ.get("GMAIL_USERNAME")
    password = os.environ.get("GMAIL_PASSWORD")
    if not username or not password:
        logger.info("Gmail credentials not set — skipping email.")
        return False

    try:
        subject = f"Sector Rotation Alert — {dt.date.today().isoformat()}"
        if alerts:
            subject += f" — {len(alerts)} ALERT(S)"

        body = report_text[:50000]  # Safety cap

        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = username
        msg["To"] = username  # Send to self

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(username, password)
            server.send_message(msg)
        logger.info("Email sent to %s", username)
        return True
    except Exception as exc:
        logger.error("Email send failed: %s", exc)
        return False


def write_google_sheets(allocation: Dict, regime: str,
                        alerts: List[Dict]) -> bool:
    """
    Append a daily row to Google Sheets via gspread.
    Requires GOOGLE_SHEETS_CREDENTIALS env var (service account JSON).
    """
    creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    if not creds_json:
        logger.info("Google Sheets credentials not set — skipping.")
        return False

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds_data = json.loads(creds_json)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        credentials = Credentials.from_service_account_info(creds_data, scopes=scopes)
        gc = gspread.authorize(credentials)

        sh = gc.open("Sector Rotation Daily Log")
        ws = sh.sheet1

        row = [
            dt.date.today().isoformat(),
            regime,
            json.dumps(allocation.get("allocations", {})),
            json.dumps(allocation.get("taxable_dollars", {})),
            json.dumps(allocation.get("roth_dollars", {})),
            len(alerts),
            "; ".join(a["type"] for a in alerts) if alerts else "none",
        ]
        ws.append_row(row)
        logger.info("Google Sheets row appended.")
        return True
    except Exception as exc:
        logger.error("Google Sheets write failed: %s", exc)
        return False


# ===========================================================================
# 5. EXECUTIVE SUMMARY REPORT GENERATOR
# ===========================================================================

def generate_executive_summary(
    regime_state: Dict,
    allocation: Dict,
    alerts: List[Dict],
    cfg: dict,
    nlp_signals: Optional[pd.DataFrame] = None,
    factor_signals: Optional[pd.DataFrame] = None,
    consecutive_defensive_days: int = 0,
) -> str:
    """
    Build the full human-readable Executive Summary.
    Designed to be read in under 60 seconds.
    """
    today = dt.date.today().isoformat()
    total = cfg.get("portfolio", {}).get("total_value", 144000)
    taxable_val = cfg.get("portfolio", {}).get("accounts", {}).get("taxable", {}).get("value", 100000)
    roth_val = cfg.get("portfolio", {}).get("accounts", {}).get("roth_ira", {}).get("value", 44000)

    regime = regime_state.get("dominant_regime", "offense").upper()
    confirmed = "Y" if regime_state.get("regime_confirmed", False) else "N"
    consec = regime_state.get("consecutive_days_in_regime", 0)
    wv_pct = regime_state.get("wedge_volume_percentile", 0)
    vix_rv = regime_state.get("vix_rv_ratio", 0.0)
    fast_shock = regime_state.get("fast_shock_risk", "low").upper()

    # Risk level
    if regime == "PANIC":
        risk_level = "HIGH"
    elif regime == "DEFENSE" or fast_shock == "HIGH":
        risk_level = "ELEVATED"
    else:
        risk_level = "LOW"

    # Action required
    rebalance_alerts = [a for a in alerts if a["type"] in ("REBALANCE", "PANIC_PROTOCOL")]
    if rebalance_alerts:
        action = "YES — REBALANCE"
    elif alerts:
        action = "MONITOR"
    else:
        action = "NO — HOLD"

    alloc = allocation.get("allocations", {})
    taxable_d = allocation.get("taxable_dollars", {})
    roth_d = allocation.get("roth_dollars", {})

    # --- Build category table ---
    # Map asset classes to display names
    DISPLAY_NAMES = {
        "us_equities": "US Equities (ETFs)",
        "intl_developed": "Intl Developed",
        "em_equities": "EM Equities",
        "energy_materials": "Energy / Materials",
        "healthcare": "Healthcare (ETF)",
        "cash_short_duration": "Cash / BIL",
        "vix_overlay_notional": "VIX Overlay (notional)",
    }

    def fmt_dollar(v):
        if v is None or v == 0:
            return "    —   "
        return f"${v:>8,.0f}"

    def fmt_pct(v):
        if v is None or v == 0:
            return "  — "
        return f"{v:>5.1%}"

    rows = []
    total_taxable_check = 0.0
    total_roth_check = 0.0
    for key, display in DISPLAY_NAMES.items():
        pct = alloc.get(key, 0.0) or 0.0
        dollars = round(pct * total, 2)
        tax = taxable_d.get(key, 0.0) or 0.0
        rth = roth_d.get(key, 0.0) or 0.0
        total_taxable_check += tax
        total_roth_check += rth
        rows.append((display, pct, dollars, tax, rth))

    # --- Build report ---
    lines = []
    lines.append("")
    lines.append("╔══════════════════════════════════════════════════════════════════╗")
    lines.append("║         SECTOR ROTATION SYSTEM — EXECUTIVE SUMMARY              ║")
    lines.append(f"║                        {today}                            ║")
    lines.append("╚══════════════════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append(f"MARKET REGIME: {regime}  (Confirmed: {confirmed}, Day {consec} of regime)")
    lines.append(f"Risk Level: {risk_level}")
    lines.append(f"Fast Shock Risk: {fast_shock}  (VIX/RV: {vix_rv:.2f})")
    lines.append("")
    lines.append("━" * 66)
    lines.append(f"TODAY'S ACTION REQUIRED: {action}")
    lines.append("━" * 66)
    lines.append("")

    # Portfolio overview
    lines.append("─" * 66)
    lines.append(f"FULL PORTFOLIO OVERVIEW  (${total:,.0f} total)")
    lines.append("─" * 66)
    lines.append(f"{'Category':<22}│{'Target %':>9} │{'Target $':>10} │{'Taxable':>10} │{'Roth IRA':>9}")
    lines.append("─" * 22 + "┼" + "─" * 10 + "┼" + "─" * 11 + "┼" + "─" * 11 + "┼" + "─" * 10)

    for display, pct, dollars, tax, rth in rows:
        lines.append(
            f"{display:<22}│{fmt_pct(pct):>9} │{fmt_dollar(dollars):>10} │"
            f"{fmt_dollar(tax):>10} │{fmt_dollar(rth):>9}"
        )

    lines.append("─" * 22 + "┼" + "─" * 10 + "┼" + "─" * 11 + "┼" + "─" * 11 + "┼" + "─" * 10)
    total_pct = sum(alloc.get(k, 0) or 0 for k in DISPLAY_NAMES)
    lines.append(
        f"{'TOTAL':<22}│{fmt_pct(total_pct):>9} │"
        f"{fmt_dollar(total):>10} │"
        f"{fmt_dollar(total_taxable_check):>10} │"
        f"{fmt_dollar(total_roth_check):>9}"
    )
    lines.append("")

    # Alerts section
    lines.append("─" * 66)
    lines.append("ALERTS")
    lines.append("─" * 66)
    if alerts:
        for a in alerts:
            icon = "🔴" if a["severity"] == "CRITICAL" else "🟡" if a["severity"] == "HIGH" else "🔵"
            lines.append(f"  {icon} [{a['type']}] {a['message'][:120]}")
    else:
        lines.append("  No alerts today.")
    lines.append("")

    # Signal detail section
    lines.append("═" * 66)
    lines.append("SIGNAL DETAIL (for reference — no action needed to read this)")
    lines.append("═" * 66)
    lines.append("")

    lines.append("REGIME SIGNALS:")
    lines.append(f"  Wedge Volume Percentile:  {wv_pct:.1f}%")
    lines.append(f"  Consecutive days:         {consec}")
    lines.append(f"  Regime confirmed:         {confirmed}")
    lines.append("")

    # Factor signals
    if factor_signals is not None and not factor_signals.empty:
        lines.append("TOP FACTOR SIGNALS:")
        if "sector_etf" in factor_signals.columns and "composite_score" in factor_signals.columns:
            sorted_fs = factor_signals.sort_values("composite_score", ascending=False)
            lines.append(f"  Strongest sector:   {sorted_fs.iloc[0]['sector_etf']} "
                         f"(score: {sorted_fs.iloc[0]['composite_score']:.2f})")
            lines.append(f"  Weakest sector:     {sorted_fs.iloc[-1]['sector_etf']} "
                         f"(score: {sorted_fs.iloc[-1]['composite_score']:.2f})")
    else:
        lines.append("TOP FACTOR SIGNALS:")
        lines.append("  (factor data not available)")
    lines.append("")

    # NLP sentiment
    lines.append("NLP SENTIMENT (active in Offense mode only):")
    if nlp_signals is not None and not nlp_signals.empty and "sector_score" in nlp_signals.columns:
        positive = nlp_signals.sort_values("sector_score", ascending=False)
        negative = nlp_signals.sort_values("sector_score", ascending=True)
        drift_col = nlp_signals.get("drift_risk", pd.Series([0]))
        drift = "HIGH" if drift_col.any() else "LOW"
        lines.append(f"  Most positive:   {positive.iloc[0]['sector_etf']} "
                     f"(+{positive.iloc[0]['sector_score']:.2f})")
        lines.append(f"  Most negative:   {negative.iloc[0]['sector_etf']} "
                     f"({negative.iloc[0]['sector_score']:+.2f})")
        lines.append(f"  Drift risk:      {drift}")
    else:
        lines.append("  (NLP data not available)")
    lines.append("")

    # Next scheduled dates
    next_rebalance = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    # Monthly factor update: next 1st of month
    today_date = dt.date.today()
    if today_date.month == 12:
        next_factor = dt.date(today_date.year + 1, 1, 1).isoformat()
    else:
        next_factor = dt.date(today_date.year, today_date.month + 1, 1).isoformat()

    lines.append(f"NEXT SCHEDULED REBALANCE CHECK: {next_rebalance}")
    lines.append(f"NEXT FACTOR MODEL UPDATE:       {next_factor} (monthly)")
    lines.append("")
    lines.append("═" * 66)

    return "\n".join(lines)


# ===========================================================================
# 6. DAILY RUN LOG
# ===========================================================================

def log_run(conn: sqlite3.Connection, run_id: str, date: str,
            started_at: str, finished_at: str, status: str,
            regime: str, alerts: List[Dict], report: str) -> None:
    """Write a monitor run record to the DB."""
    conn.execute("""
        INSERT OR REPLACE INTO monitor_runs
            (run_id, date, started_at, finished_at, status,
             regime, alerts_json, report_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, date, started_at, finished_at, status,
        regime, json.dumps(alerts, default=str),
        report[:50000],  # Truncate for DB safety
    ))
    conn.commit()


# ===========================================================================
# 7. CLI ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 — Monitoring Engine & Alert System")
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode (no live data pulls)")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to SQLite database")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--no-deliver", action="store_true",
                        help="Skip Telegram/email/Sheets delivery")
    args = parser.parse_args()

    # Paths
    db_path = Path(args.db) if args.db else DB_PATH
    config_path = Path(args.config) if args.config else CONFIG_PATH

    cfg = load_config(config_path)
    conn = get_db(db_path)

    run_id = f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    started_at = dt.datetime.now().isoformat()
    today = dt.date.today().isoformat()

    logger.info("═" * 60)
    logger.info("  MONITOR RUN: %s", run_id)
    logger.info("  Date: %s   Mode: %s", today,
                "MOCK" if args.mock else "LIVE")
    logger.info("═" * 60)

    status = "ok"
    alerts: List[Dict] = []

    try:
        # Step 1: Data refresh
        logger.info("Step 1: Data refresh...")
        data_result = run_data_refresh(conn, cfg, mock=args.mock)
        logger.info("  Data refresh: %s", data_result["status"])

        # Step 2-3: Regime detection
        logger.info("Step 2-3: Regime detection...")
        regime_result = run_regime_detection(conn, cfg, mock=args.mock)
        regime_state = regime_result.get("regime_state", {})
        regime = regime_state.get("dominant_regime", "offense")
        logger.info("  Regime: %s (confirmed: %s)",
                     regime, regime_state.get("regime_confirmed", False))

        # Step 4: Optimizer (if regime changed)
        logger.info("Step 4: Optimizer check...")
        optimizer_result = run_optimizer(conn, cfg, regime, mock=args.mock)
        allocation = optimizer_result.get("allocation", {})
        regime_changed = optimizer_result.get("regime_changed", False)
        logger.info("  Optimizer: %s (regime_changed: %s)",
                     optimizer_result["status"], regime_changed)

        # Step 5: Check allocation drift
        prev_allocation = fetch_latest_allocation(conn)

        # Step 6: NLP scoring
        logger.info("Step 6: NLP scoring...")
        nlp_result = run_nlp_scoring(conn, cfg, mock=args.mock)
        logger.info("  NLP: %s", nlp_result["status"])

        # Fetch supplementary data for report
        try:
            nlp_signals = fetch_nlp_sector_signals(conn)
        except Exception:
            nlp_signals = None

        try:
            factor_signals = fetch_latest_factor_signals(conn)
        except Exception:
            factor_signals = None

        # Count consecutive defensive days
        defensive_days = count_consecutive_defensive_days(conn)

        # Step 7: Alert evaluation
        logger.info("Step 7: Alert evaluation...")
        engine = AlertEngine(cfg)
        alerts = engine.evaluate(
            regime_state=regime_state,
            prev_allocation=prev_allocation,
            new_allocation=allocation,
            regime_changed=regime_changed,
            consecutive_defensive_days=defensive_days,
        )
        logger.info("  Alerts triggered: %d", len(alerts))
        for a in alerts:
            logger.info("    [%s] %s: %s",
                         a["severity"], a["type"], a["message"][:80])

        # Step 8: Generate report
        logger.info("Step 8: Generating Executive Summary...")
        report = generate_executive_summary(
            regime_state=regime_state,
            allocation=allocation,
            alerts=alerts,
            cfg=cfg,
            nlp_signals=nlp_signals,
            factor_signals=factor_signals,
            consecutive_defensive_days=defensive_days,
        )
        print(report)

        # Step 9: Deliver alerts
        logger.info("Step 9: Alert delivery...")
        write_alerts_json(alerts)
        append_alerts_csv(alerts)

        if not args.no_deliver:
            regime_summary = f"{regime} (Day {regime_state.get('consecutive_days_in_regime', 0)})"
            send_telegram(alerts, regime_summary)
            send_email(alerts, report)
            write_google_sheets(allocation, regime, alerts)
        else:
            logger.info("  Delivery skipped (--no-deliver flag).")

        # Log to DB
        log_run(conn, run_id, today, started_at,
                dt.datetime.now().isoformat(), status, regime,
                alerts, report)

    except Exception as exc:
        status = "error"
        logger.error("Monitor run failed: %s\n%s", exc, traceback.format_exc())
        log_run(conn, run_id, today, started_at,
                dt.datetime.now().isoformat(), status,
                "unknown", alerts, str(exc))

    conn.close()
    logger.info("Monitor run %s complete. Status: %s", run_id, status)


if __name__ == "__main__":
    main()
