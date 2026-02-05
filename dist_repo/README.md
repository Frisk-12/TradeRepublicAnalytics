# Portfolio Analytics Dashboard

Streamlit dashboard for personal portfolio analytics based on uploaded account statements.

## What It Does
- Parses account statement PDFs and builds a session‑only datalake.
- Computes portfolio metrics and visual analytics:
  - Overview (account value, net invested, PnL, contributors, allocations)
  - Performance (TWR, rolling returns, attribution, calendar returns)
  - Risk (volatility, drawdown, risk contributors)
  - Risk Advanced (correlations, beta, tail risk)
  - Diagnostics (pricing and fiscal checks)

## How It Works (High‑Level)
1. Upload PDF statements in the sidebar.
2. The app parses transactions in memory for the session.
3. Pricing, analytics, and charts are computed on the fly.

## Run Locally
```bash
streamlit run script/analytics/app/app.py
```

### Required Environment
- Python 3.10+
- Streamlit
- `OPENFIGI_API_KEY` (required for security master enrichment)

## Notes
- Data is session‑scoped; no files are written by default.
- The app assumes Trade Republic statement formats.

