# Sector Rotation System

[![Open Setup & Backtest in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bigbathtub101/sector-rotation-system/blob/main/notebooks/setup_and_backtest.ipynb)
[![Open Signal Validation in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bigbathtub101/sector-rotation-system/blob/main/notebooks/validate_signals.ipynb)

A quantitative sector rotation and portfolio allocation system that combines macroeconomic regime detection, CVaR-optimized portfolio construction, thematic stock screening, and NLP sentiment analysis — all running on **free, open-source infrastructure**.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [FRED API Key](#2-fred-api-key)
3. [Gmail SMTP Setup](#3-gmail-smtp-setup)
4. [Telegram Bot Setup](#4-telegram-bot-setup)
5. [Google Sheets Setup](#5-google-sheets-setup)
6. [Configuring Your Holdings](#6-configuring-your-holdings)
7. [Streamlit Dashboard Deployment](#7-streamlit-dashboard-deployment)
8. [GitHub Actions Setup](#8-github-actions-setup)
9. [Alert Types Explained](#9-alert-types-explained)

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ data_feeds   │────▶│ regime_detector   │────▶│ portfolio_optimizer  │
│ (yfinance,   │     │ (Wedge Volume,    │     │ (CVaR, allocation    │
│  FRED, SEC)  │     │  Fast Shock)      │     │  bands by regime)    │
└─────────────┘     └──────────────────┘     └─────────────────────┘
       │                                              │
       ▼                                              ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ nlp_sentiment│     │ stock_screener    │     │ monitor             │
│ (FinBERT on  │     │ (factor scoring,  │     │ (daily alerts,      │
│  SEC filings)│     │  thematic lists)  │     │  email, Telegram)   │
└─────────────┘     └──────────────────┘     └─────────────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────────────┐
                                              │ dashboard           │
                                              │ (Streamlit, 6 pages)│
                                              └─────────────────────┘
```

**File Structure:**

| File | Description |
|------|-------------|
| `config.yaml` | All parameters, thresholds, and ticker lists — nothing hardcoded |
| `data_feeds.py` | Market data (yfinance), FRED macro, SEC EDGAR filings |
| `regime_detector.py` | Wedge Volume regime classification (Offense / Defense / Panic) |
| `portfolio_optimizer.py` | CVaR optimization with regime-conditional allocation bands |
| `stock_screener.py` | Factor-based stock scoring + 4 thematic watchlists |
| `nlp_sentiment.py` | FinBERT sentiment on SEC filings, sector-level aggregation |
| `monitor.py` | Daily orchestrator: runs all phases, detects alerts, delivers notifications |
| `dashboard.py` | Streamlit dashboard with 6 interactive pages |
| `rotation_system.db` | SQLite database — prices, macro, filings, signals, allocations |

---

## 1. Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sector-rotation-system.git
cd sector-rotation-system

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt
```

**First run — populate the database with 2 years of historical data:**

```bash
python data_feeds.py --backfill
```

This fetches ~2 years of daily prices for all ETFs and watchlist stocks via yfinance, pulls macro data from FRED (if `FRED_API_KEY` is set), and downloads recent SEC filings. Takes about 2–3 minutes.

**Run the full pipeline once:**

```bash
python monitor.py --run-daily
```

This executes every phase in order (data refresh → regime detection → optimization → screening → NLP → alert check) and prints the Executive Summary to the console.

**Launch the dashboard:**

```bash
streamlit run dashboard.py
```

---

## 2. FRED API Key

The system uses 6 macroeconomic series from the Federal Reserve (FRED). Getting a key is free and takes about 2 minutes.

**Steps:**

1. Go to [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Click **"Request API Key"**
3. Create a free account (or sign in if you have one)
4. Copy your 32-character API key

**Set the environment variable:**

```bash
# macOS/Linux — add to ~/.bashrc or ~/.zshrc
export FRED_API_KEY="your_32_character_key_here"

# Windows PowerShell
$env:FRED_API_KEY = "your_32_character_key_here"
```

> **Without this key:** The system still works — macro data fields will use placeholder values and regime detection will rely solely on market data (prices and VIX). With the key, you get CFNAI, yield curve, CPI, and unemployment data for richer regime scoring.

---

## 3. Gmail SMTP Setup

The GitHub Actions workflow sends email alerts when rebalance signals, regime changes, or panic events are detected.

**Step 1: Enable 2-Factor Authentication on your Gmail**

1. Go to [https://myaccount.google.com/security](https://myaccount.google.com/security)
2. Under "Signing in to Google," enable **2-Step Verification**

**Step 2: Create an App Password**

1. Go to [https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
2. Select app: **Mail**, device: **Other** (enter "Sector Rotation")
3. Click **Generate**
4. Copy the 16-character password (e.g., `abcd efgh ijkl mnop`)

**Step 3: Add as GitHub Actions Secrets**

1. Go to your repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add two secrets:

| Secret Name | Value |
|-------------|-------|
| `GMAIL_USERNAME` | `your_email@gmail.com` |
| `GMAIL_PASSWORD` | `abcdefghijklmnop` (the App Password, no spaces) |

> **Security note:** App Passwords are separate from your Google account password. You can revoke them at any time from the App Passwords page.

---

## 4. Telegram Bot Setup

Telegram alerts give you instant push notifications on your phone.

**Step 1: Create a Telegram Bot**

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Choose a name (e.g., "Sector Rotation Alerts")
4. Choose a username (e.g., `sector_rotation_alert_bot`)
5. BotFather will reply with your **bot token** — looks like `123456789:ABCdefGhIjKlMnOpQrStUvWxYz`

**Step 2: Get Your Chat ID**

1. Start a conversation with your new bot (send it any message)
2. Open this URL in your browser (replace `YOUR_BOT_TOKEN`):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
3. Look for `"chat":{"id":123456789}` — that number is your **chat ID**

**Step 3: Add as GitHub Actions Secrets**

| Secret Name | Value |
|-------------|-------|
| `TELEGRAM_BOT_TOKEN` | `123456789:ABCdefGhIjKlMnOpQrStUvWxYz` |
| `TELEGRAM_CHAT_ID` | `123456789` |

> **Test it:** After adding the secrets, trigger the workflow manually (Actions tab → "Daily Sector Rotation Monitor" → "Run workflow"). If there are any alerts, you'll get a Telegram message.

---

## 5. Google Sheets Setup

The system writes a daily allocation row to Google Sheets, giving you a running log of every day's regime, allocations, and alerts.

**Step 1: Create a Google Cloud Service Account**

1. Go to [https://console.cloud.google.com/](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Enable the **Google Sheets API**: APIs & Services → Library → search "Google Sheets API" → Enable
4. Enable the **Google Drive API**: same process, search "Google Drive API" → Enable
5. Go to **APIs & Services → Credentials → Create Credentials → Service Account**
6. Name it (e.g., "sector-rotation-writer"), click **Create and Continue**
7. Skip the optional role assignment, click **Done**
8. Click on the new service account → **Keys** tab → **Add Key → Create new key → JSON**
9. A `.json` file downloads — this is your credentials file

**Step 2: Share the Spreadsheet**

1. Create a new Google Spreadsheet named **"Sector Rotation Daily Log"** (must match `config.yaml` → `alerts.google_sheets.spreadsheet_name`)
2. Click **Share** and add the service account email (looks like `sector-rotation-writer@your-project.iam.gserviceaccount.com`) with **Editor** access

**Step 3: Add as GitHub Actions Secret**

1. Open the downloaded JSON file in a text editor
2. Copy the **entire** contents (it's one JSON object)
3. In your repo → Settings → Secrets → Actions → New repository secret:

| Secret Name | Value |
|-------------|-------|
| `GOOGLE_SHEETS_CREDENTIALS` | `{"type": "service_account", "project_id": "...", ...}` (paste entire JSON) |

> **Cost:** Completely free. Google Sheets API has a generous free tier (500 requests/100 seconds) that this system will never approach.

---

## 6. Configuring Your Holdings

Edit `config.yaml` to match your actual portfolio so rebalance alerts reflect real dollar amounts.

**Portfolio Size:**

```yaml
portfolio:
  total_value: 144000          # Update to your actual total
  accounts:
    taxable:
      value: 100000            # Your taxable brokerage balance
    roth_ira:
      value: 44000             # Your Roth IRA balance
```

**Tax Location Rules** control which positions go in which account:

```yaml
tax_location:
  roth_ira_first:
    - "individual_stocks_thematic"    # High-turnover thematic picks → Roth
    - "high_turnover"
    - "biotech_smallcap"
  taxable:
    - "broad_etf_longterm"           # Buy-and-hold ETFs → taxable (LTCG)
    - "geographic_etf"               # Foreign tax credit eligible
    - "cash_short_duration"
```

**Watchlists** — add or remove tickers from any of the 4 thematic lists:

```yaml
tickers:
  watchlist_biotech:
    - NBIX
    - EXEL
    # ... add your own picks
  watchlist_ai_software:
    - CRWD
    - NET
    # ...
```

**All thresholds are configurable.** Key ones to review:

| Parameter | Location | Default | What It Does |
|-----------|----------|---------|--------------|
| `regime.thresholds.panic_upper` | `config.yaml` | 5 | Percentile below which = Panic |
| `regime.thresholds.defense_upper` | `config.yaml` | 30 | Percentile below which = Defense |
| `monitor.rebalance_threshold_bps` | `config.yaml` | 200 | Basis points of drift before alert |
| `optimizer.cvar_confidence` | `config.yaml` | 0.95 | CVaR confidence level |
| `factor_model.mclean_pontiff_decay` | `config.yaml` | 0.74 | Post-publication alpha decay |

---

## 7. Streamlit Dashboard Deployment

Deploy the dashboard to **Streamlit Community Cloud** for free, 24/7 access from any browser.

**Steps:**

1. Go to [https://share.streamlit.io/](https://share.streamlit.io/) and sign in with GitHub
2. Click **"New app"**
3. Select your repo: `YOUR_USERNAME/sector-rotation-system`
4. Main file path: `dashboard.py`
5. Python version: `3.11`
6. Click **"Deploy"**

**Add secrets in Streamlit Cloud:**

1. In the app settings (⋮ menu → Settings → Secrets), add:
   ```toml
   FRED_API_KEY = "your_key_here"
   ```
2. The dashboard reads `config.yaml` and `rotation_system.db` from the repo — make sure your DB is committed after running the backfill.

> **Note:** The free Streamlit Community Cloud plan sleeps the app after 7 days of inactivity. It wakes up automatically when you visit it (takes ~30 seconds). For always-on, consider Streamlit's paid plans or self-hosting.

---

## 8. GitHub Actions Setup

The workflow runs `monitor.py` every trading day at 4:30 PM ET and sends alerts via email, Telegram, and Google Sheets.

**Step 1: Fork the Repository**

1. Click the **Fork** button at the top of the repo page
2. Keep the default settings and click **Create fork**

**Step 2: Enable GitHub Actions**

1. In your forked repo, go to the **Actions** tab
2. If prompted, click **"I understand my workflows, go ahead and enable them"**
3. The "Daily Sector Rotation Monitor" workflow should appear

**Step 3: Add All Required Secrets**

Go to Settings → Secrets and variables → Actions, and add:

| Secret | Source | Required? |
|--------|--------|-----------|
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Recommended |
| `GMAIL_USERNAME` | Your Gmail address | For email alerts |
| `GMAIL_PASSWORD` | Gmail App Password (see Section 3) | For email alerts |
| `TELEGRAM_BOT_TOKEN` | @BotFather (see Section 4) | For Telegram alerts |
| `TELEGRAM_CHAT_ID` | Bot API (see Section 4) | For Telegram alerts |
| `GOOGLE_SHEETS_CREDENTIALS` | GCP service account JSON (see Section 5) | For Sheets logging |

**Step 4: Test with a Manual Run**

1. Go to Actions → "Daily Sector Rotation Monitor" → **"Run workflow"** → **"Run workflow"**
2. Watch the run — it should complete in 2–4 minutes
3. Check the **Artifacts** section for the daily report
4. Verify Telegram and email delivery

**Free Tier Limits:**

| Resource | Free Limit | This System Uses |
|----------|------------|------------------|
| GitHub Actions minutes | 2,000 min/month | ~60–80 min/month (3 min × ~22 trading days) |
| Artifact storage | 500 MB | ~5 MB/month |
| yfinance API | Unlimited | ~50 calls/day |
| FRED API | 120 req/min | 6 requests/day |

---

## 9. Alert Types Explained

The monitor detects 5 types of alerts. Here's what each one means and what action it recommends.

### REGIME_CHANGE

**What it means:** The market regime has shifted (e.g., Offense → Defense, Defense → Panic).

**What to do:**
- **Offense → Defense:** Reduce equity exposure. The system will show target allocations with specific dollar amounts for each account. Increase cash/short-duration bonds (BIL). Don't panic-sell, but start moving to the Defense allocation bands over 1–3 days.
- **Defense → Panic:** Immediately execute the panic exit sequence — sell 50% of equity positions now, remainder over 3–5 days. Move to maximum cash allocation.
- **Panic → Defense (upgrade):** Begin cautiously re-entering equities. The system will show which sectors to add first and at what dollar amounts.
- **Defense → Offense:** Increase equity exposure to Offense-level allocation bands. Add international and EM positions.

### REBALANCE

**What it means:** One or more positions have drifted more than 200 basis points (2%) from their target allocation.

**What to do:** Review the specific positions listed in the alert. The system shows current weight vs. target weight and the dollar amount to buy or sell. Execute the trades in the correct accounts (taxable vs. Roth) as shown.

### ENTRY_WINDOW

**What it means:** A sector or thematic position is underweight by more than 300 basis points (3%) during an Offense regime — a potential buying opportunity.

**What to do:** Consider adding to the underweight position. The alert shows the specific ticker, current allocation, target allocation, and dollar amount to deploy. Prioritize positions with high factor scores.

### EXTENDED_DEFENSE

**What it means:** The system has been in Defense or Panic mode for more than 60 consecutive days, suggesting the downturn may be structural.

**What to do:** Reassess your thesis. Review the regime metrics on the dashboard. Consider whether the defensive posture is still appropriate or if conditions are improving. This alert is informational — it doesn't recommend a specific trade, but flags that the current positioning has persisted unusually long.

### NLP_SENTIMENT_SHIFT

**What it means:** FinBERT sentiment analysis on SEC filings has detected a significant shift in sector-level sentiment.

**What to do:** Review the NLP Sentiment page on the dashboard. During Offense regimes, NLP sentiment contributes 20% to the composite score and may tilt allocation weights. During Defense/Panic, NLP is monitoring-only and doesn't affect allocations. Use this as a qualitative overlay — confirm with your own reading of the filings.

---

## Backtest Disclaimer

All backtest results displayed in the dashboard carry the label:

> *(in-sample, apply 26% McLean-Pontiff decay to alpha differential for forward estimate)*

This means: if the backtest shows 2% annual alpha over the benchmark, a realistic forward-looking estimate is `2% × 0.74 = 1.48%` after accounting for historical data-mining bias ([McLean & Pontiff, 2016](https://doi.org/10.1111/jofi.12365)).

---

## License

This project is for personal, non-commercial use. No investment advice is provided. The system outputs recommendations — it does not execute trades. Always verify signals independently before acting.
