# Security & Privacy Notice

This dashboard handles **financial data** (account statements, transactions, balances). Use it with care.

## Data Handling
- **Session‑only processing**: data is loaded from uploaded PDFs and processed in memory.
- **No persistence**: the app does not write uploaded statements to disk by default.
- **No caching to disk**: OpenFIGI/JustETF/YFinance/FX caches are disabled in the app.
- **On refresh**: all data is discarded and must be re‑uploaded.

## External Services
The app may call third‑party APIs to enrich data and fetch prices:
- **OpenFIGI** (security master enrichment by ISIN)
- **Yahoo Finance** (equity prices / FX conversion)
- **JustETF** (ETF prices and exposures)
- **Vontobel Markets** (warrant prices)
- **Société Générale Warrants** (warrant prices)

These services may see:
- ISINs or tickers for instruments present in your statements

They should **not** receive personal identifiers (name, address, account number) unless present in instrument identifiers.


## Risks & Recommendations
1. **Confidentiality**: Only upload statements you are allowed to disclose.
2. **Compliance**: Ensure this usage is permitted by your broker’s terms.
3. **Network Exposure**: Running on Streamlit Cloud sends data to their servers.
4. **Third‑party calls**: ISINs/tickers may be sent to OpenFIGI, Yahoo, JustETF, Vontobel, Société Générale.
5. **Sharing**: Do not share public links if the app processes private data.

## If You Need Stronger Guarantees
- Run locally or on a private server.
- Disable external APIs and use offline price sources.
- Add anonymization steps to strip instrument identifiers.

---
For questions or changes to data handling, review `script/analytics/app/app.py` and `script/datalake/build_datalake.py`.
