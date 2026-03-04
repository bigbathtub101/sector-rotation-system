"""
sector_rotation — Quantitative Sector Rotation System
========================================================

A quantitative portfolio allocation system combining macroeconomic regime
detection, CVaR-optimized portfolio construction, thematic stock screening,
and NLP sentiment analysis.

Modules
-------
regime_detector
    Wedge Volume regime classification (Offense / Defense / Panic)
portfolio_optimizer
    CVaR optimization with regime-conditional allocation bands
stock_screener
    Factor-based stock scoring + thematic watchlists
data_feeds
    Market data (yfinance), FRED macro, SEC EDGAR filings
nlp_sentiment
    FinBERT sentiment on SEC filings
monitor
    Daily orchestrator and alert system
dashboard
    Streamlit dashboard (6 interactive pages)
walk_forward
    Rolling walk-forward backtest engine
universe_builder
    Dynamic stock universe builder (replaces static watchlists)
transaction_costs
    Bid-ask spread and market impact cost model
"""

__version__ = "0.1.0"
