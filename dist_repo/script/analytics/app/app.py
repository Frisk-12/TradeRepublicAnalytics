#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analytics multi-tab app: Overview + Diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tempfile

from script.analytics.app.diagnostics import render_diagnostics
from script.analytics.app.overview import render_overview
from script.analytics.app.risk import render_risk, render_risk_advanced
from script.analytics.app.performance import render_performance
from script.analytics.core.tables import build_canonical_tables
from script.analytics.core.trades import build_trade_pricing_and_alerts
from script.datalake.build_datalake import build_datalake

DEFAULT_DATALAKE_JSON = "/Users/andreadesogus/Desktop/portfolio_analytics/script/datalake/datalake.json"


st.set_page_config(page_title="Portfolio Analytics", page_icon="📊", layout="wide")
st.cache_data.clear()

st.sidebar.markdown("**Upload account statements (PDF)**")
st.sidebar.caption("Caricando i file accetti che l’app effettui chiamate esterne per prezzi e metadata (OpenFIGI, Yahoo, JustETF, Vontobel, Société Générale).")
consent = st.sidebar.checkbox("I accept the data processing and external calls", value=False)

MAX_FILE_MB = 10
MAX_TOTAL_MB = 25
MAX_UPLOAD_BATCHES = 5
if "upload_batches" not in st.session_state:
    st.session_state.upload_batches = 0

uploaded = st.sidebar.file_uploader(
    "Account statements",
    type=["pdf"],
    accept_multiple_files=True,
    help="I file restano in memoria per la sessione corrente.",
)

if not consent:
    st.info("Per procedere è necessario accettare il trattamento dati e le chiamate esterne.")
    st.stop()

if not uploaded:
    st.info("Carica i PDF degli account statement per iniziare.")
    st.stop()

if st.session_state.upload_batches >= MAX_UPLOAD_BATCHES:
    st.error("Limite di upload raggiunto per la sessione. Ricarica la pagina per ripartire.")
    st.stop()

total_size = sum(getattr(f, "size", 0) for f in uploaded)
if total_size > MAX_TOTAL_MB * 1024 * 1024:
    st.error(f"Dimensione totale troppo grande (max {MAX_TOTAL_MB} MB).")
    st.stop()

for f in uploaded:
    if getattr(f, "size", 0) > MAX_FILE_MB * 1024 * 1024:
        st.error(f"File troppo grande: {f.name} (max {MAX_FILE_MB} MB).")
        st.stop()
    head = f.getvalue()[:5]
    if head != b"%PDF-":
        st.error(f"File non valido (non PDF): {f.name}")
        st.stop()

with st.spinner("Processing uploaded statements..."):
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in uploaded:
            out_path = Path(tmpdir) / f.name
            out_path.write_bytes(f.getvalue())
        try:
            datalake = build_datalake(
                statements_dir=tmpdir,
                openfigi_cache_path=None,
                justetf_cache_path=None,
                yfinance_cache_path=None,
                fx_cache_path=None,
            )
        except SystemExit as exc:
            st.error(str(exc))
            st.stop()
    df_master, df_tx, df_prices_long = build_canonical_tables(datalake)
    df_trade_priced, df_alerts = build_trade_pricing_and_alerts(df_tx, df_prices_long, df_master)

st.session_state.upload_batches += 1

tab_overview, tab_performance, tab_risk, tab_risk_adv, tab_diagnostics = st.tabs(
    ["Overview", "Performance", "Risk", "Risk Advanced", "Diagnostics"]
)
with tab_overview:
    render_overview(df_tx, df_trade_priced, df_prices_long, df_master)
with tab_performance:
    render_performance(df_tx, df_trade_priced, df_prices_long, df_master)
with tab_risk:
    render_risk(df_tx, df_trade_priced, df_prices_long, df_master)
with tab_risk_adv:
    render_risk_advanced(df_tx, df_trade_priced, df_prices_long, df_master)
with tab_diagnostics:
    render_diagnostics(df_master, df_tx, df_trade_priced)
