"""
dashboard.py — Phase 6: Streamlit Dashboard
==============================================
Global Sector Rotation System

A visual interface you can check anytime — from desktop or phone.
Deployable to Streamlit Community Cloud (free).

Pages:
  1. Regime Dashboard   — gauge, probability bars, Fast Shock, timeline
  2. Portfolio Allocation — dual-account table, historical toggle
  3. Signal Detail       — wedge volume history, factor trends, NLP, macro
  4. Stock Screener      — top candidates, watchlists, 8-K filings
  5. Alerts Log          — historical alerts, regime transitions
  6. Backtester          — standard + 3 failure-mode stress tests

Run locally:
  streamlit run dashboard.py

Dependencies: streamlit, plotly, pandas, numpy, pyyaml
"""

import datetime as dt
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# PATHS  (relative to this file)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"
DB_PATH = BASE_DIR / "rotation_system.db"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# DATABASE HELPERS
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

@st.cache_data(ttl=60)
def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_db()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def query_one(sql: str, params: tuple = ()) -> Optional[tuple]:
    conn = get_db()
    try:
        return conn.execute(sql, params).fetchone()
    except Exception:
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_latest_regime() -> Dict:
    row = query_one(
        "SELECT date, signal_data FROM signals "
        "WHERE signal_type = 'regime_state' "
        "ORDER BY date DESC LIMIT 1"
    )
    if row:
        data = json.loads(row[1])
        data["date"] = row[0]
        return data
    return {
        "date": None, "dominant_regime": "offense",
        "wedge_volume_percentile": 50.0,
        "regime_probabilities": {"panic": 0.0, "defense": 0.0, "offense": 1.0},
        "fast_shock_risk": "low", "vix_rv_ratio": 0.0,
        "consecutive_days_in_regime": 0, "regime_confirmed": False,
    }


@st.cache_data(ttl=60)
def load_regime_history(days: int = 730) -> pd.DataFrame:
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    df = query_df(
        "SELECT date, signal_data FROM signals "
        "WHERE signal_type = 'regime_state' AND date >= ? "
        "ORDER BY date ASC",
        (cutoff,),
    )
    if df.empty:
        return pd.DataFrame()
    records = []
    for _, row in df.iterrows():
        d = json.loads(row["signal_data"])
        d["date"] = row["date"]
        records.append(d)
    return pd.DataFrame(records)


@st.cache_data(ttl=60)
def load_latest_allocation() -> Optional[Dict]:
    row = query_one(
        "SELECT date, regime, allocations, taxable_dollars, roth_dollars "
        "FROM allocations ORDER BY date DESC LIMIT 1"
    )
    if row:
        return {
            "date": row[0], "regime": row[1],
            "allocations": json.loads(row[2]) if row[2] else {},
            "taxable_dollars": json.loads(row[3]) if row[3] else {},
            "roth_dollars": json.loads(row[4]) if row[4] else {},
        }
    return None


@st.cache_data(ttl=60)
def load_all_allocations() -> pd.DataFrame:
    return query_df(
        "SELECT date, regime, allocations, taxable_dollars, roth_dollars "
        "FROM allocations ORDER BY date DESC"
    )


@st.cache_data(ttl=60)
def load_factor_history(days: int = 730) -> pd.DataFrame:
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    df = query_df(
        "SELECT date, signal_data FROM signals "
        "WHERE signal_type = 'factor_scores' AND date >= ? "
        "ORDER BY date ASC",
        (cutoff,),
    )
    if df.empty:
        return pd.DataFrame()
    records = []
    for _, row in df.iterrows():
        d = json.loads(row["signal_data"])
        for s in d.get("sector_scores", []):
            s["date"] = row["date"]
            records.append(s)
    return pd.DataFrame(records)


@st.cache_data(ttl=60)
def load_nlp_sector_signals() -> pd.DataFrame:
    return query_df(
        "SELECT * FROM nlp_sector_signals ORDER BY date DESC LIMIT 50"
    )


@st.cache_data(ttl=60)
def load_macro_data() -> pd.DataFrame:
    return query_df("SELECT * FROM macro_data ORDER BY date DESC")


@st.cache_data(ttl=60)
def load_prices(tickers: List[str] = None, days: int = 730) -> pd.DataFrame:
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        return query_df(
            f"SELECT date, ticker, close FROM prices "
            f"WHERE date >= ? AND ticker IN ({placeholders}) "
            f"ORDER BY date ASC",
            (cutoff, *tickers),
        )
    return query_df(
        "SELECT date, ticker, close FROM prices WHERE date >= ? ORDER BY date ASC",
        (cutoff,),
    )


@st.cache_data(ttl=60)
def load_alerts_history() -> pd.DataFrame:
    return query_df(
        "SELECT * FROM monitor_runs ORDER BY date DESC LIMIT 200"
    )


@st.cache_data(ttl=60)
def load_filings() -> pd.DataFrame:
    return query_df("SELECT * FROM filings ORDER BY date DESC LIMIT 50")


@st.cache_data(ttl=60)
def load_nlp_scores() -> pd.DataFrame:
    return query_df("SELECT * FROM nlp_scores ORDER BY date DESC LIMIT 50")


# ===========================================================================
# PAGE SETUP
# ===========================================================================
st.set_page_config(
    page_title="Sector Rotation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Regime colors */
    .regime-offense { color: #00d26a; }
    .regime-defense { color: #ffc107; }
    .regime-panic   { color: #ff4444; }

    /* Severity badges */
    .badge-critical {
        background: #ff4444; color: white; padding: 2px 10px;
        border-radius: 4px; font-weight: 600; font-size: 0.8rem;
    }
    .badge-high {
        background: #ffc107; color: #1e1e2e; padding: 2px 10px;
        border-radius: 4px; font-weight: 600; font-size: 0.8rem;
    }
    .badge-medium {
        background: #17a2b8; color: white; padding: 2px 10px;
        border-radius: 4px; font-weight: 600; font-size: 0.8rem;
    }
    .badge-low {
        background: #6c757d; color: white; padding: 2px 10px;
        border-radius: 4px; font-weight: 600; font-size: 0.8rem;
    }

    /* Table styling */
    .allocation-table th {
        background: #2d2d44;
        color: #8888aa;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# SIDEBAR NAVIGATION
# ===========================================================================
cfg = load_config()

PAGES = [
    "📈 Regime Dashboard",
    "💼 Portfolio Allocation",
    "🔍 Signal Detail",
    "🎯 Stock Screener",
    "🔔 Alerts Log",
    "📊 Backtester",
]

st.sidebar.title("Sector Rotation System")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

# Show current regime in sidebar
regime_state = load_latest_regime()
regime = regime_state.get("dominant_regime", "offense").upper()
regime_color = {"OFFENSE": "#00d26a", "DEFENSE": "#ffc107", "PANIC": "#ff4444"}.get(regime, "#888")
st.sidebar.markdown(f"""
<div style="text-align:center; padding:12px; margin-top:8px;
     background:rgba(255,255,255,0.03); border-radius:8px;
     border:1px solid {regime_color}40;">
    <div style="font-size:0.7rem; color:#888; text-transform:uppercase;
         letter-spacing:2px;">Current Regime</div>
    <div style="font-size:1.6rem; font-weight:700; color:{regime_color};
         margin:4px 0;">{regime}</div>
    <div style="font-size:0.75rem; color:#aaa;">
        Day {regime_state.get('consecutive_days_in_regime', 0)} ·
        {'Confirmed' if regime_state.get('regime_confirmed') else 'Unconfirmed'}
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="text-align:center; padding:8px; margin-top:6px; font-size:0.7rem; color:#666;">
    Data as of {regime_state.get('date', 'N/A')}
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# PAGE 1: REGIME DASHBOARD
# ===========================================================================
def page_regime_dashboard():
    st.title("Regime Dashboard")
    st.caption("Live system state — wedge volume, regime probabilities, and Fast Shock Risk")

    # --- Top metrics row ---
    col1, col2, col3, col4 = st.columns(4)

    wv_pct = regime_state.get("wedge_volume_percentile", 0)
    vix_rv = regime_state.get("vix_rv_ratio", 0)
    fast_shock = regime_state.get("fast_shock_risk", "low").upper()
    consec = regime_state.get("consecutive_days_in_regime", 0)
    confirmed = "YES" if regime_state.get("regime_confirmed") else "NO"

    with col1:
        st.metric("Wedge Volume Percentile", f"{wv_pct:.1f}%")
    with col2:
        st.metric("VIX / RV Ratio", f"{vix_rv:.2f}")
    with col3:
        shock_color = "🔴" if fast_shock == "HIGH" else "🟢"
        st.metric("Fast Shock Risk", f"{shock_color} {fast_shock}")
    with col4:
        st.metric("Regime Day", f"{consec} (Confirmed: {confirmed})")

    st.markdown("---")

    # --- Wedge Volume Gauge ---
    col_gauge, col_probs = st.columns([1, 1])

    with col_gauge:
        st.subheader("Wedge Volume Gauge")
        thresholds = cfg.get("regime", {}).get("thresholds", {})
        panic_upper = thresholds.get("panic_upper", 5)
        defense_upper = thresholds.get("defense_upper", 30)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=wv_pct,
            number={"suffix": "%", "font": {"size": 42}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": regime_color, "thickness": 0.3},
                "steps": [
                    {"range": [0, panic_upper], "color": "rgba(255,68,68,0.3)"},
                    {"range": [panic_upper, defense_upper], "color": "rgba(255,193,7,0.3)"},
                    {"range": [defense_upper, 100], "color": "rgba(0,210,106,0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": wv_pct,
                },
            },
            title={"text": "Wedge Volume Percentile", "font": {"size": 14}},
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(t=40, b=20, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ccc"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_probs:
        st.subheader("Regime Probabilities")
        probs = regime_state.get("regime_probabilities", {})
        prob_data = pd.DataFrame({
            "Regime": ["Offense", "Defense", "Panic"],
            "Probability": [
                probs.get("offense", 0) * 100,
                probs.get("defense", 0) * 100,
                probs.get("panic", 0) * 100,
            ],
        })
        fig_probs = px.bar(
            prob_data, x="Probability", y="Regime",
            orientation="h",
            color="Regime",
            color_discrete_map={
                "Offense": "#00d26a", "Defense": "#ffc107", "Panic": "#ff4444"
            },
            text="Probability",
        )
        fig_probs.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_probs.update_layout(
            height=280, showlegend=False,
            xaxis={"range": [0, 105], "title": "Probability (%)", "showgrid": True},
            yaxis={"title": ""},
            margin=dict(t=40, b=20, l=10, r=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ccc"},
        )
        st.plotly_chart(fig_probs, use_container_width=True)

    # --- Historical Regime Timeline ---
    st.subheader("Historical Regime Timeline")
    hist = load_regime_history()
    if not hist.empty:
        regime_map = {"offense": 3, "defense": 2, "panic": 1}
        color_map = {"offense": "#00d26a", "defense": "#ffc107", "panic": "#ff4444"}

        hist["regime_num"] = hist["dominant_regime"].map(regime_map)
        hist["color"] = hist["dominant_regime"].map(color_map)

        fig_timeline = go.Figure()

        # Regime band shading
        for regime_name, num in regime_map.items():
            mask = hist["dominant_regime"] == regime_name
            if mask.any():
                fig_timeline.add_trace(go.Scatter(
                    x=hist.loc[mask, "date"],
                    y=hist.loc[mask, "regime_num"],
                    mode="markers",
                    marker={"color": color_map[regime_name], "size": 6},
                    name=regime_name.capitalize(),
                    hovertemplate="%{x}<br>Regime: " + regime_name.capitalize() + "<extra></extra>",
                ))

        fig_timeline.update_layout(
            height=250,
            yaxis={
                "tickvals": [1, 2, 3],
                "ticktext": ["Panic", "Defense", "Offense"],
                "title": "",
            },
            xaxis={"title": "Date"},
            margin=dict(t=20, b=40, l=60, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ccc"},
            showlegend=True,
            legend={"orientation": "h", "y": 1.12},
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Wedge volume history overlay
        if "wedge_volume_percentile" in hist.columns:
            st.subheader("Wedge Volume Percentile History")
            fig_wv = go.Figure()
            fig_wv.add_trace(go.Scatter(
                x=hist["date"], y=hist["wedge_volume_percentile"],
                mode="lines", name="Wedge Volume %",
                line={"color": "#6c63ff", "width": 2},
            ))
            # Add threshold lines
            fig_wv.add_hline(y=panic_upper, line_dash="dash",
                            line_color="#ff4444", annotation_text="Panic threshold")
            fig_wv.add_hline(y=defense_upper, line_dash="dash",
                            line_color="#ffc107", annotation_text="Defense threshold")
            fig_wv.update_layout(
                height=280,
                yaxis={"title": "Percentile", "range": [0, 100]},
                xaxis={"title": "Date"},
                margin=dict(t=20, b=40, l=60, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ccc"},
            )
            st.plotly_chart(fig_wv, use_container_width=True)
    else:
        st.info("No regime history data available. Run `monitor.py --mock` to populate.")


# ===========================================================================
# PAGE 2: PORTFOLIO ALLOCATION
# ===========================================================================
def page_portfolio_allocation():
    st.title("Portfolio Allocation")

    alloc = load_latest_allocation()
    total_val = cfg.get("portfolio", {}).get("total_value", 144000)
    taxable_val = cfg.get("portfolio", {}).get("accounts", {}).get("taxable", {}).get("value", 100000)
    roth_val = cfg.get("portfolio", {}).get("accounts", {}).get("roth_ira", {}).get("value", 44000)

    if alloc:
        st.caption(f"As of: {alloc['date']} · Regime: {alloc['regime'].upper()}")

        # --- Top metrics ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Portfolio", f"${total_val:,.0f}")
        m2.metric("Taxable Account", f"${taxable_val:,.0f}")
        m3.metric("Roth IRA", f"${roth_val:,.0f}")

        st.markdown("---")

        # --- Allocation table ---
        DISPLAY_NAMES = {
            "us_equities": "US Equities (ETFs)",
            "intl_developed": "Intl Developed",
            "em_equities": "EM Equities",
            "energy_materials": "Energy / Materials",
            "healthcare": "Healthcare (ETF)",
            "cash_short_duration": "Cash / BIL",
            "vix_overlay_notional": "VIX Overlay (notional)",
        }

        weights = alloc.get("allocations", {})
        tax_d = alloc.get("taxable_dollars", {})
        roth_d = alloc.get("roth_dollars", {})

        rows = []
        for key, display in DISPLAY_NAMES.items():
            pct = weights.get(key, 0) or 0
            target_dollar = round(pct * total_val, 2)
            taxable = tax_d.get(key, 0) or 0
            roth = roth_d.get(key, 0) or 0
            rows.append({
                "Category": display,
                "Target %": f"{pct:.1%}",
                "Target $": f"${target_dollar:,.0f}",
                "Taxable $": f"${taxable:,.0f}" if taxable > 0 else "—",
                "Roth IRA $": f"${roth:,.0f}" if roth > 0 else "—",
            })

        # Total row
        total_pct = sum(weights.get(k, 0) or 0 for k in DISPLAY_NAMES)
        total_tax = sum(tax_d.get(k, 0) or 0 for k in DISPLAY_NAMES)
        total_roth = sum(roth_d.get(k, 0) or 0 for k in DISPLAY_NAMES)
        rows.append({
            "Category": "TOTAL",
            "Target %": f"{total_pct:.1%}",
            "Target $": f"${total_val:,.0f}",
            "Taxable $": f"${total_tax:,.0f}",
            "Roth IRA $": f"${total_roth:,.0f}",
        })

        df_alloc = pd.DataFrame(rows)
        st.dataframe(
            df_alloc,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn(width="large"),
            },
        )

        # --- Pie chart ---
        st.subheader("Allocation Breakdown")
        pie_data = []
        for key, display in DISPLAY_NAMES.items():
            pct = weights.get(key, 0) or 0
            if pct > 0:
                pie_data.append({"Category": display, "Weight": pct})
        if pie_data:
            pie_df = pd.DataFrame(pie_data)
            fig_pie = px.pie(
                pie_df, names="Category", values="Weight",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig_pie.update_traces(textposition="outside", textinfo="label+percent")
            fig_pie.update_layout(
                height=400,
                margin=dict(t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ccc"},
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- Historical allocations toggle ---
        st.markdown("---")
        with st.expander("📅 View Historical Allocations"):
            all_allocs = load_all_allocations()
            if not all_allocs.empty:
                selected_date = st.selectbox(
                    "Select date:",
                    all_allocs["date"].tolist(),
                )
                sel_row = all_allocs[all_allocs["date"] == selected_date].iloc[0]
                hist_weights = json.loads(sel_row["allocations"]) if sel_row["allocations"] else {}
                hist_tax = json.loads(sel_row["taxable_dollars"]) if sel_row["taxable_dollars"] else {}
                hist_roth = json.loads(sel_row["roth_dollars"]) if sel_row["roth_dollars"] else {}

                hist_rows = []
                for key, display in DISPLAY_NAMES.items():
                    pct = hist_weights.get(key, 0) or 0
                    hist_rows.append({
                        "Category": display,
                        "Target %": f"{pct:.1%}",
                        "Target $": f"${pct * total_val:,.0f}",
                        "Taxable $": f"${hist_tax.get(key, 0):,.0f}" if hist_tax.get(key, 0) else "—",
                        "Roth IRA $": f"${hist_roth.get(key, 0):,.0f}" if hist_roth.get(key, 0) else "—",
                    })
                st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No historical allocation data.")
    else:
        st.info("No allocation data available. Run `monitor.py --mock` to populate.")


# ===========================================================================
# PAGE 3: SIGNAL DETAIL
# ===========================================================================
def page_signal_detail():
    st.title("Signal Detail")
    st.caption("Deep dive into regime signals, factor scores, NLP sentiment, and macro indicators")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Wedge Volume", "Factor Scores", "NLP Sentiment", "Macro Indicators"
    ])

    # --- Tab 1: Wedge Volume History ---
    with tab1:
        st.subheader("Wedge Volume Percentile — 2-Year History")
        hist = load_regime_history()
        if not hist.empty and "wedge_volume_percentile" in hist.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["date"], y=hist["wedge_volume_percentile"],
                mode="lines", name="Wedge Volume %",
                line={"color": "#6c63ff", "width": 2},
                fill="tozeroy", fillcolor="rgba(108,99,255,0.1)",
            ))
            thresholds = cfg.get("regime", {}).get("thresholds", {})
            fig.add_hline(y=thresholds.get("panic_upper", 5),
                         line_dash="dash", line_color="#ff4444",
                         annotation_text="Panic (5th)")
            fig.add_hline(y=thresholds.get("defense_upper", 30),
                         line_dash="dash", line_color="#ffc107",
                         annotation_text="Defense (30th)")
            fig.update_layout(
                height=400, yaxis={"title": "Percentile", "range": [0, 100]},
                xaxis={"title": "Date"},
                margin=dict(t=20, b=40, l=60, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ccc"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No wedge volume history available.")

    # --- Tab 2: Factor Score Trends ---
    with tab2:
        st.subheader("Factor Score Trends by Sector")
        factor_df = load_factor_history()
        if not factor_df.empty and "sector_etf" in factor_df.columns:
            sector_filter = st.multiselect(
                "Filter sectors:",
                factor_df["sector_etf"].unique().tolist(),
                default=factor_df["sector_etf"].unique().tolist()[:5],
            )
            filtered = factor_df[factor_df["sector_etf"].isin(sector_filter)]
            if not filtered.empty:
                fig = px.line(
                    filtered, x="date", y="composite_score",
                    color="sector_etf",
                    title="Composite Factor Scores Over Time",
                    labels={"composite_score": "Score", "date": "Date"},
                )
                fig.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#ccc"},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Latest scores table
            latest_date = factor_df["date"].max()
            latest = factor_df[factor_df["date"] == latest_date].sort_values(
                "composite_score", ascending=False
            )
            st.subheader(f"Latest Scores ({latest_date})")
            st.dataframe(latest[["sector_etf", "composite_score"]].reset_index(drop=True),
                        use_container_width=True, hide_index=True)
        else:
            st.info("No factor score data available.")

    # --- Tab 3: NLP Sentiment ---
    with tab3:
        st.subheader("NLP Sentiment by Sector")
        nlp_df = load_nlp_sector_signals()
        if not nlp_df.empty and "sector_score" in nlp_df.columns:
            # Active only in Offense
            current_regime = regime_state.get("dominant_regime", "offense")
            if current_regime != "offense":
                st.warning(f"NLP sentiment is monitoring-only in {current_regime.upper()} mode. "
                          f"Scores shown for reference only (weight = 0%).")

            nlp_weight = cfg.get("nlp", {}).get("regime_weights", {}).get(current_regime, 0)
            st.caption(f"NLP composite weight in current regime: {nlp_weight:.0%}")

            # Bar chart of sector scores
            latest_nlp = nlp_df.drop_duplicates(subset="sector_etf", keep="first")
            fig = px.bar(
                latest_nlp.sort_values("sector_score", ascending=True),
                x="sector_score", y="sector_etf",
                orientation="h",
                color="sector_score",
                color_continuous_scale=["#ff4444", "#ffc107", "#00d26a"],
                color_continuous_midpoint=0,
                labels={"sector_score": "Sentiment Score", "sector_etf": "Sector ETF"},
            )
            fig.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ccc"}, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Drift risk
            drift_col = nlp_df.get("drift_risk", pd.Series([0]))
            if drift_col.any():
                st.error("⚠️ NLP Regime Drift Risk: HIGH — VIX elevated above configured threshold")
            else:
                st.success("NLP Regime Drift Risk: LOW")
        else:
            st.info("No NLP sentiment data available.")

    # --- Tab 4: Macro Indicators ---
    with tab4:
        st.subheader("Macro Indicator Dashboard")
        macro_df = load_macro_data()
        if not macro_df.empty:
            MACRO_NAMES = {
                "FEDFUNDS": "Federal Funds Rate",
                "T10Y2Y": "10Y-2Y Treasury Spread",
                "CPIAUCSL": "CPI (Year-over-Year)",
                "UNRATE": "Unemployment Rate",
                "CFNAI": "Chicago Fed National Activity Index",
                "INDPRO": "Industrial Production",
            }
            for _, row in macro_df.iterrows():
                series = row.get("series_id", row.get("series", ""))
                name = MACRO_NAMES.get(series, series)
                value = row.get("value", "N/A")
                date = row.get("date", "N/A")
                st.metric(name, f"{value}", help=f"As of {date}")
        else:
            st.info("No macro data available. Ensure FRED_API_KEY is set.")


# ===========================================================================
# PAGE 4: STOCK SCREENER
# ===========================================================================
def page_stock_screener():
    st.title("Stock Screener")
    st.caption("Top buy candidates, thematic watchlists, and recent filings")

    tab1, tab2, tab3 = st.tabs([
        "Top Candidates", "Thematic Watchlists", "Recent 8-K Filings"
    ])

    # --- Tab 1: Top Candidates ---
    with tab1:
        st.subheader("Top 5-7 Buy Candidates per Overweight Sector")
        factor_df = load_factor_history()
        if not factor_df.empty and "composite_score" in factor_df.columns:
            latest_date = factor_df["date"].max()
            latest = factor_df[factor_df["date"] == latest_date].sort_values(
                "composite_score", ascending=False
            )
            st.dataframe(latest.head(7).reset_index(drop=True),
                        use_container_width=True, hide_index=True)
        else:
            st.info("No factor score data. Run Phase 3B stock screener to populate.")

    # --- Tab 2: Thematic Watchlists ---
    with tab2:
        st.subheader("Thematic Watchlists")
        tickers = cfg.get("tickers", {})
        watchlists = {
            "🧬 Biotech": tickers.get("watchlist_biotech", []),
            "🤖 AI / Cyber": tickers.get("watchlist_ai_software", []),
            "🛡️ Defense": tickers.get("watchlist_defense", []),
            "⛏️ Green Materials": tickers.get("watchlist_green_materials", []),
        }

        nlp_scores = load_nlp_scores()
        score_map = {}
        if not nlp_scores.empty and "ticker" in nlp_scores.columns:
            for _, row in nlp_scores.iterrows():
                score_map[row["ticker"]] = row.get("sentiment_score", 0)

        for wl_name, wl_tickers in watchlists.items():
            st.markdown(f"#### {wl_name}")
            wl_rows = []
            for t in wl_tickers:
                score = score_map.get(t, None)
                if score is not None:
                    if score > 0.2:
                        label = "✅ BUY"
                    elif score > -0.1:
                        label = "⚠️ HOLD"
                    else:
                        label = "❌ AVOID"
                else:
                    label = "— No data"
                wl_rows.append({
                    "Ticker": t,
                    "NLP Score": f"{score:.3f}" if score is not None else "N/A",
                    "Signal": label,
                })
            st.dataframe(pd.DataFrame(wl_rows), use_container_width=True, hide_index=True)

    # --- Tab 3: Recent 8-K Filings ---
    with tab3:
        st.subheader("Recent 8-K Filings with NLP Summaries")
        filings = load_filings()
        if not filings.empty:
            for _, filing in filings.iterrows():
                ticker = filing.get("ticker", "N/A")
                filing_type = filing.get("filing_type", "N/A")
                date = filing.get("date", "N/A")
                text_col = "raw_text" if "raw_text" in filing.index else "text"
                text = filing.get(text_col, "")
                summary = text[:300] + "..." if len(str(text)) > 300 else text

                with st.expander(f"📄 {ticker} — {filing_type} ({date})"):
                    st.write(summary)

                    # NLP score for this filing
                    score = score_map.get(ticker, None)
                    if score is not None:
                        color = "green" if score > 0.1 else "red" if score < -0.1 else "orange"
                        st.markdown(f"NLP Sentiment: :{color}[{score:.3f}]")
        else:
            st.info("No filings data available.")


# ===========================================================================
# PAGE 5: ALERTS LOG
# ===========================================================================
def page_alerts_log():
    st.title("Alerts Log")
    st.caption("Historical alerts, regime transitions, and rebalance history")

    tab1, tab2 = st.tabs(["Alert History", "Regime Transitions"])

    # --- Tab 1: Alert History ---
    with tab1:
        st.subheader("Monitor Run History")
        runs = load_alerts_history()
        if not runs.empty:
            for _, run in runs.iterrows():
                date = run.get("date", "N/A")
                status = run.get("status", "N/A")
                regime = run.get("regime", "N/A")
                alerts_json = run.get("alerts_json", "[]")

                try:
                    alerts = json.loads(alerts_json) if alerts_json else []
                except (json.JSONDecodeError, TypeError):
                    alerts = []

                status_icon = "✅" if status == "ok" else "❌"
                alert_count = len(alerts)
                alert_badge = f"🔴 {alert_count} alert(s)" if alert_count > 0 else "No alerts"

                with st.expander(f"{status_icon} {date} — {regime.upper() if regime else 'N/A'} — {alert_badge}"):
                    st.write(f"**Status:** {status}")
                    st.write(f"**Started:** {run.get('started_at', 'N/A')}")
                    st.write(f"**Finished:** {run.get('finished_at', 'N/A')}")

                    if alerts:
                        for a in alerts:
                            sev = a.get("severity", "LOW")
                            badge_class = f"badge-{sev.lower()}"
                            st.markdown(
                                f"<span class='{badge_class}'>{sev}</span> "
                                f"**{a.get('type', '')}** — {a.get('message', '')[:200]}",
                                unsafe_allow_html=True,
                            )

                    report = run.get("report_text", "")
                    if report:
                        with st.expander("View Full Report"):
                            st.code(report[:5000], language="text")
        else:
            st.info("No monitor runs recorded yet.")

    # --- Tab 2: Regime Transitions ---
    with tab2:
        st.subheader("Regime Transition Log")
        hist = load_regime_history()
        if not hist.empty:
            transitions = []
            prev_regime = None
            for _, row in hist.iterrows():
                r = row.get("dominant_regime", "offense")
                if r != prev_regime and prev_regime is not None:
                    transitions.append({
                        "Date": row["date"],
                        "From": prev_regime.upper(),
                        "To": r.upper(),
                        "Confirmed": "Yes" if row.get("regime_confirmed") else "No",
                        "Wedge Vol %": f"{row.get('wedge_volume_percentile', 0):.1f}%",
                    })
                prev_regime = r

            if transitions:
                st.dataframe(pd.DataFrame(transitions), use_container_width=True, hide_index=True)
            else:
                st.info("No regime transitions detected in history.")
        else:
            st.info("No regime history data available.")


# ===========================================================================
# PAGE 6: BACKTESTER
# ===========================================================================
def page_backtester():
    st.title("Backtester")
    st.caption("Standard backtest and Failure Mode Stress Tests (Section 9)")

    mclean_pontiff = cfg.get("factor_model", {}).get("mclean_pontiff_decay", 0.74)
    mclean_label = cfg.get("backtest", {}).get("mclean_pontiff_label",
        "(in-sample, apply 26% McLean-Pontiff decay to alpha differential for forward estimate)")

    tab1, tab2 = st.tabs(["Standard Backtest", "Failure Mode Stress Tests"])

    # --- Tab 1: Standard Backtest ---
    with tab1:
        st.subheader("Cumulative Returns vs. SPY Buy-and-Hold")
        st.info(f"⚠️ All performance metrics shown are in-sample. {mclean_label}")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date",
                                        dt.date.today() - dt.timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", dt.date.today())

        if st.button("Run Backtest", type="primary"):
            prices = load_prices(["SPY"], days=(dt.date.today() - start_date).days + 30)
            if not prices.empty:
                spy = prices[prices["ticker"] == "SPY"].copy()
                spy["date"] = pd.to_datetime(spy["date"])
                spy = spy[(spy["date"] >= pd.Timestamp(start_date)) &
                          (spy["date"] <= pd.Timestamp(end_date))]
                spy = spy.sort_values("date")

                if len(spy) > 1:
                    spy["return"] = spy["close"].pct_change()
                    spy["cum_return"] = (1 + spy["return"]).cumprod() - 1

                    # Simulate system returns (apply regime overlay)
                    hist = load_regime_history(days=(dt.date.today() - start_date).days + 30)
                    regime_dates = {}
                    if not hist.empty:
                        for _, row in hist.iterrows():
                            regime_dates[row["date"]] = row.get("dominant_regime", "offense")

                    # System applies cash drag in defense/panic
                    system_returns = []
                    for _, row in spy.iterrows():
                        d = row["date"].strftime("%Y-%m-%d")
                        r = row.get("return", 0) or 0
                        regime = regime_dates.get(d, "offense")
                        if regime == "panic":
                            r = r * 0.15  # 85% cash
                        elif regime == "defense":
                            r = r * 0.55  # 45% cash
                        system_returns.append(r)

                    spy["system_return"] = system_returns
                    spy["system_cum"] = (1 + spy["system_return"]).cumprod() - 1

                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=spy["date"], y=spy["cum_return"] * 100,
                        name="SPY Buy & Hold",
                        line={"color": "#888", "dash": "dot"},
                    ))
                    fig.add_trace(go.Scatter(
                        x=spy["date"], y=spy["system_cum"] * 100,
                        name="Rotation System",
                        line={"color": "#6c63ff", "width": 2},
                    ))
                    fig.update_layout(
                        height=400,
                        yaxis={"title": "Cumulative Return (%)"}, xaxis={"title": "Date"},
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#ccc"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Metrics
                    spy_total = spy["cum_return"].iloc[-1] * 100
                    sys_total = spy["system_cum"].iloc[-1] * 100
                    alpha_raw = sys_total - spy_total

                    # Max drawdown
                    spy_dd = _max_drawdown(spy["close"])
                    sys_prices = (1 + spy["system_return"]).cumprod()
                    sys_dd = _max_drawdown(sys_prices)

                    # Calmar ratio (annualized return / max drawdown)
                    years = len(spy) / 252
                    spy_ann = ((1 + spy_total / 100) ** (1 / max(years, 0.01)) - 1) * 100
                    sys_ann = ((1 + sys_total / 100) ** (1 / max(years, 0.01)) - 1) * 100
                    spy_calmar = spy_ann / abs(spy_dd) if spy_dd != 0 else 0
                    sys_calmar = sys_ann / abs(sys_dd) if sys_dd != 0 else 0

                    # McLean-Pontiff adjusted alpha
                    alpha_adjusted = alpha_raw * mclean_pontiff

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("SPY Return", f"{spy_total:.1f}%")
                    m2.metric("System Return", f"{sys_total:.1f}%")
                    m3.metric("Raw Alpha", f"{alpha_raw:+.1f}%")
                    m4.metric("Adj. Alpha (MP)", f"{alpha_adjusted:+.1f}%",
                             help=mclean_label)

                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("SPY Max DD", f"{spy_dd:.1f}%")
                    m6.metric("System Max DD", f"{sys_dd:.1f}%")
                    m7.metric("SPY Calmar", f"{spy_calmar:.2f}")
                    m8.metric("System Calmar", f"{sys_calmar:.2f}")

                    st.markdown(f"*{mclean_label}*")
                else:
                    st.warning("Not enough price data for selected range.")
            else:
                st.warning("No SPY price data available.")

    # --- Tab 2: Failure Mode Stress Tests ---
    with tab2:
        st.subheader("Failure Mode Stress Tests")
        st.markdown("Three named scenarios from Section 9 of the strategy report:")

        stress_cfg = cfg.get("backtest", {}).get("stress_tests", {})

        # Scenario buttons
        scenario = st.radio(
            "Select scenario:",
            [
                "Policy Shock Test (Section 9.1)",
                "Incomplete Panic Test (Section 9.2)",
                "Extended Bull Market Test (Section 9.3)",
            ],
        )

        if st.button("Run Stress Test", type="primary"):
            if "Policy Shock" in scenario:
                _run_policy_shock_test(stress_cfg, mclean_label)
            elif "Incomplete Panic" in scenario:
                _run_incomplete_panic_test(stress_cfg, mclean_label)
            elif "Extended Bull" in scenario:
                _run_extended_bull_test(stress_cfg, mclean_label)


def _max_drawdown(prices) -> float:
    """Compute max drawdown as a negative percentage."""
    if len(prices) < 2:
        return 0.0
    cummax = prices.cummax()
    dd = (prices - cummax) / cummax
    return dd.min() * 100


def _run_policy_shock_test(stress_cfg: dict, mclean_label: str):
    """Section 9.1: S&P drops >3% with no prior wedge deterioration."""
    st.markdown("### Policy Shock Test (Section 9.1)")
    st.markdown("""
    **Scenario:** The S&P 500 drops more than 3% in a single day, BUT there was
    no prior deterioration of wedge volume below the 15th percentile. The system
    is still in Offense mode when the shock hits.

    **Purpose:** Expose the detection gap — the system can't catch exogenous shocks
    that don't build gradually through correlation changes.
    """)

    shock_pct = stress_cfg.get("policy_shock", {}).get("sp500_drop_threshold", -0.03)
    pre_wv = stress_cfg.get("policy_shock", {}).get("wedge_volume_pre_threshold", 15)

    np.random.seed(91)
    days = 60
    dates = pd.date_range(end=dt.date.today(), periods=days, freq="B")

    # Normal market, then sudden shock on day 45
    returns = np.random.normal(0.0005, 0.008, days)
    returns[44] = shock_pct  # The policy shock day
    returns[45] = -0.015     # Continued selling
    returns[46] = 0.012      # Partial recovery

    prices = 100 * np.cumprod(1 + returns)
    wv_pct = np.full(days, 55.0)
    wv_pct[44:] = np.linspace(55, 8, days - 44)  # Drops AFTER the shock

    # System response: stays in offense (wv was fine before shock)
    system_returns = returns.copy()
    # System doesn't react until day 46 when regime transitions
    # Day 44-45: full exposure, day 46+: transition to defense
    system_returns[46:] *= 0.55

    spy_cum = (pd.Series(1 + returns).cumprod() - 1) * 100
    sys_cum = (pd.Series(1 + system_returns).cumprod() - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=spy_cum, name="SPY", line={"color": "#888", "dash": "dot"}))
    fig.add_trace(go.Scatter(x=dates, y=sys_cum, name="System", line={"color": "#6c63ff"}))
    fig.add_vline(x=dates[44], line_dash="dash", line_color="red",
                  annotation_text="Shock Day")
    fig.update_layout(
        height=350, yaxis={"title": "Cumulative Return (%)"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detection gap
    gap = abs(returns[44]) * 100
    st.error(f"**Detection Gap:** System took the full {gap:.1f}% hit on shock day "
             f"because wedge volume was at {wv_pct[43]:.0f}th percentile "
             f"(above {pre_wv}th threshold). Regime transition didn't begin until "
             f"2 days after the shock.")
    st.markdown(f"""
    **Staged Exit Protocol Response:**
    - Day 0 (shock): No action — system in Offense
    - Day +1: Wedge volume begins dropping → watching for confirmation
    - Day +2: Regime transitions to Defense → 50% cash move
    - Day +3-5: Complete transition to Defense allocation

    *{mclean_label}*
    """)


def _run_incomplete_panic_test(stress_cfg: dict, mclean_label: str):
    """Section 9.2: 2022-analog sustained drawdown without correlation spike."""
    st.markdown("### Incomplete Panic Test (Section 9.2)")
    st.markdown("""
    **Scenario:** A 2022-style year — sustained drawdown (~25%) without the
    correlation spike that would trigger Panic mode. The system stays in Defense
    (elevated cash) the entire year.

    **Purpose:** Show the cash drag cost vs. protection benefit. This is the
    scenario most likely to cause mandate abandonment.
    """)

    analog_year = stress_cfg.get("incomplete_panic", {}).get("analog_year", 2022)

    np.random.seed(2022)
    days = 252
    dates = pd.date_range(end=dt.date.today(), periods=days, freq="B")

    # 2022-style: gradual drawdown with rallies
    trend = np.linspace(0, -0.25, days)
    noise = np.random.normal(0, 0.012, days)
    spy_prices = 100 * np.exp(trend + np.cumsum(noise) * 0.3)
    spy_returns = np.diff(spy_prices) / spy_prices[:-1]
    spy_returns = np.insert(spy_returns, 0, 0)

    # System: in defense (45% equity exposure) for most of the year
    system_returns = spy_returns * 0.55  # 45% cash allocation

    spy_cum = (pd.Series(1 + spy_returns).cumprod() - 1) * 100
    sys_cum = (pd.Series(1 + system_returns).cumprod() - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=spy_cum, name="SPY", line={"color": "#888", "dash": "dot"}))
    fig.add_trace(go.Scatter(x=dates, y=sys_cum, name="System (Defense)", line={"color": "#ffc107"}))
    fig.update_layout(
        height=350, yaxis={"title": "Cumulative Return (%)"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
    )
    st.plotly_chart(fig, use_container_width=True)

    spy_dd = spy_cum.min()
    sys_dd = sys_cum.min()
    protection = abs(spy_dd) - abs(sys_dd)

    # Cash drag on rally days
    rally_mask = spy_returns > 0
    cash_drag = (spy_returns[rally_mask] * 0.45).sum() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("SPY Max Drawdown", f"{spy_dd:.1f}%")
    c2.metric("System Max Drawdown", f"{sys_dd:.1f}%")
    c3.metric("Protection Benefit", f"{protection:.1f}%")

    st.warning(f"**Cash Drag on Rally Days:** {cash_drag:.1f}% of potential gains missed "
               f"due to 45% cash allocation in Defense mode. This is the cost of protection "
               f"when no crisis materializes.")
    st.markdown(f"""
    **Mandate Abandonment Risk:** After {days} trading days in Defense without a Panic event,
    the system's Extended Defense Alert (at day 60) recommends partial re-entry.
    Without discipline to the process, many users would override the system here.

    *{mclean_label}*
    """)


def _run_extended_bull_test(stress_cfg: dict, mclean_label: str):
    """Section 9.3: 10 years with no Panic events."""
    st.markdown("### Extended Bull Market Test (Section 9.3)")
    st.markdown("""
    **Scenario:** 10 years of bull market with no Panic events. The system
    compounds annual drag from VIX roll cost, false-positive Defense triggers,
    and cash buffer — showing cumulative underperformance vs. buy-and-hold.

    **Purpose:** Show the honest long-term cost of insurance when no crisis
    justifies it.
    """)

    drag_range = stress_cfg.get("extended_bull", {}).get("annual_drag_range", [0.005, 0.012])
    duration = stress_cfg.get("extended_bull", {}).get("duration_years", 10)

    np.random.seed(42)
    years = np.arange(0, duration + 1)
    spy_annual = 0.10  # 10% annual return
    drag_low = drag_range[0]
    drag_high = drag_range[1]

    spy_growth = (1 + spy_annual) ** years
    sys_growth_low = (1 + spy_annual - drag_low) ** years
    sys_growth_high = (1 + spy_annual - drag_high) ** years

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=(spy_growth - 1) * 100,
        name="SPY Buy & Hold",
        line={"color": "#888", "dash": "dot", "width": 2},
    ))
    fig.add_trace(go.Scatter(
        x=years, y=(sys_growth_low - 1) * 100,
        name=f"System (min drag {drag_low:.1%}/yr)",
        line={"color": "#00d26a", "width": 2},
    ))
    fig.add_trace(go.Scatter(
        x=years, y=(sys_growth_high - 1) * 100,
        name=f"System (max drag {drag_high:.1%}/yr)",
        line={"color": "#ff4444", "width": 2},
    ))
    # Fill between
    fig.add_trace(go.Scatter(
        x=np.concatenate([years, years[::-1]]),
        y=np.concatenate([(sys_growth_low - 1) * 100, ((sys_growth_high - 1) * 100)[::-1]]),
        fill="toself", fillcolor="rgba(255,68,68,0.1)",
        line={"color": "rgba(0,0,0,0)"},
        showlegend=False,
    ))
    fig.update_layout(
        height=400,
        xaxis={"title": "Years", "dtick": 1},
        yaxis={"title": "Cumulative Return (%)"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dollar impact on $144K
    total_val = cfg.get("portfolio", {}).get("total_value", 144000)
    spy_end = total_val * spy_growth[-1]
    sys_end_low = total_val * sys_growth_low[-1]
    sys_end_high = total_val * sys_growth_high[-1]
    cost_low = spy_end - sys_end_low
    cost_high = spy_end - sys_end_high

    c1, c2, c3 = st.columns(3)
    c1.metric("SPY Final Value", f"${spy_end:,.0f}")
    c2.metric("System (min drag)", f"${sys_end_low:,.0f}",
             delta=f"-${cost_low:,.0f}", delta_color="inverse")
    c3.metric("System (max drag)", f"${sys_end_high:,.0f}",
             delta=f"-${cost_high:,.0f}", delta_color="inverse")

    st.markdown(f"""
    **Drag Breakdown (annual):**
    | Component | Low Estimate | High Estimate |
    |-----------|-------------|---------------|
    | VIX roll cost | 0.2% | 0.5% |
    | False-positive Defense triggers | 0.2% | 0.4% |
    | Cash buffer (BIL underperformance) | 0.1% | 0.3% |
    | **Total annual drag** | **{drag_low:.1%}** | **{drag_high:.1%}** |

    Over {duration} years on ${total_val:,.0f}, this costs **${cost_low:,.0f}–${cost_high:,.0f}**
    in cumulative underperformance vs. buy-and-hold — *if no crisis occurs to justify it.*

    *{mclean_label}*
    """)


# ===========================================================================
# ROUTER
# ===========================================================================
if page == PAGES[0]:
    page_regime_dashboard()
elif page == PAGES[1]:
    page_portfolio_allocation()
elif page == PAGES[2]:
    page_signal_detail()
elif page == PAGES[3]:
    page_stock_screener()
elif page == PAGES[4]:
    page_alerts_log()
elif page == PAGES[5]:
    page_backtester()
