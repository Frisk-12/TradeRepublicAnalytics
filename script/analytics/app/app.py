#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analytics multi-tab app with session-only processing and PDF reporting."""

from __future__ import annotations

import sys
import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tempfile

from script.analytics.app.about import render_about
from script.analytics.app.annex import render_annex
from script.analytics.app.diagnostics import render_diagnostics
from script.analytics.app.overview import render_overview
from script.analytics.app.risk import render_risk, render_risk_advanced
from script.analytics.app.performance import render_performance
from script.analytics.app.pdf_report import generate_portfolio_pdf_report
from script.analytics.core.tables import build_canonical_tables
from script.analytics.core.trades import build_trade_pricing_and_alerts
from script.datalake.build_datalake import build_datalake


st.set_page_config(page_title="Portfolio Analytics", page_icon="📊", layout="wide")
if "cache_cleared_this_session" not in st.session_state:
    st.cache_data.clear()
    st.session_state.cache_cleared_this_session = True

st.sidebar.markdown("**Upload account statements (PDF)**")
st.sidebar.caption(
    "Session-only processing: files stay in memory/temporary runtime files and are never saved or manually reviewed by us."
)
st.sidebar.caption(
    "Important before consent: upload only statements you are authorized to process. "
    "The app is session-only (no persistent storage), and instrument identifiers can be shared with "
    "external data providers. See the About tab for warnings and transparency details."
)
consent = st.sidebar.checkbox("I accept the data processing and external calls", value=False)

MAX_FILE_MB = 10
MAX_TOTAL_MB = 25
MAX_UPLOAD_BATCHES = 5
if "upload_batches" not in st.session_state:
    st.session_state.upload_batches = 0
if "last_upload_signature" not in st.session_state:
    st.session_state.last_upload_signature = None
if "analytics_payload" not in st.session_state:
    st.session_state.analytics_payload = None
if "report_pdf_bytes" not in st.session_state:
    st.session_state.report_pdf_bytes = None
if "report_generated_at" not in st.session_state:
    st.session_state.report_generated_at = None
if "report_error" not in st.session_state:
    st.session_state.report_error = None

uploaded = st.sidebar.file_uploader(
    "Account statement (single PDF)",
    type=["pdf"],
    help="Single statement per run. Processed in memory for the current session only.",
    disabled=not consent,
)

uploaded_item = None
upload_error = None

if uploaded:
    if st.session_state.upload_batches >= MAX_UPLOAD_BATCHES:
        upload_error = "Upload limit reached for this session. Reload the page to reset."
    else:
        size = int(getattr(uploaded, "size", 0))
        if size > MAX_TOTAL_MB * 1024 * 1024:
            upload_error = f"Total file size too large (max {MAX_TOTAL_MB} MB)."
        else:
            if size > MAX_FILE_MB * 1024 * 1024:
                upload_error = f"File too large: {uploaded.name} (max {MAX_FILE_MB} MB)."
            else:
                content = uploaded.getvalue()
                if content[:5] != b"%PDF-":
                    upload_error = f"Invalid file type (not a PDF): {uploaded.name}"
                else:
                    uploaded_item = {
                        "name": uploaded.name,
                        "size": size,
                        "content": content,
                        "hash": hashlib.sha256(content).hexdigest(),
                    }

if upload_error:
    st.error(upload_error)

if not consent:
    st.info("Accept data processing and external calls to enable uploads and analytics.")
    # Safety-first: if consent is revoked, clear in-session analytics artifacts.
    st.session_state.analytics_payload = None
    st.session_state.report_pdf_bytes = None
    st.session_state.report_generated_at = None
    st.session_state.report_error = None
    st.session_state.last_upload_signature = None
    st.session_state.upload_batches = 0
elif not uploaded:
    st.info("Upload one account statement PDF to generate dashboard analytics and the auto-report.")

if consent and uploaded_item and not upload_error:
    upload_signature = (uploaded_item["name"], uploaded_item["size"], uploaded_item["hash"])

    if st.session_state.last_upload_signature != upload_signature:
        pipeline_error = None
        report_error = None
        pdf_bytes = None
        openfigi_report = {}
        with st.spinner("Processing uploaded statements and building report..."):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    out_path = Path(tmpdir) / uploaded_item["name"]
                    out_path.write_bytes(uploaded_item["content"])
                    datalake = build_datalake(
                        statements_dir=tmpdir,
                        openfigi_cache_path=None,
                        justetf_cache_path=None,
                        yfinance_cache_path=None,
                        fx_cache_path=None,
                    )

                openfigi_report = datalake.get("openfigi_report", {}) or {}
                df_master, df_tx, df_prices_long = build_canonical_tables(datalake)
                df_trade_priced, df_alerts = build_trade_pricing_and_alerts(df_tx, df_prices_long, df_master)
                try:
                    pdf_bytes = generate_portfolio_pdf_report(df_tx, df_trade_priced, df_prices_long, df_master)
                    report_error = None
                except Exception as exc:  # noqa: BLE001
                    pdf_bytes = None
                    report_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "pdftotext" in msg.lower():
                    pipeline_error = (
                        "Statement processing failed: `pdftotext` is not available in runtime. "
                        "For Streamlit Cloud, add `packages.txt` with `poppler-utils`, "
                        "then redeploy. If `PDFTOTEXT_BIN` is set, ensure it points to a valid binary."
                    )
                else:
                    pipeline_error = (
                        "Statement processing failed. "
                        "Ensure the uploaded file is a valid Trade Republic Account Statement PDF. "
                        f"Details: {type(exc).__name__}: {exc}"
                    )

        if pipeline_error:
            st.session_state.analytics_payload = None
            st.session_state.report_pdf_bytes = None
            st.session_state.report_generated_at = None
            st.session_state.report_error = pipeline_error
        else:
            st.session_state.analytics_payload = {
                "df_master": df_master,
                "df_tx": df_tx,
                "df_prices_long": df_prices_long,
                "df_trade_priced": df_trade_priced,
                "df_alerts": df_alerts,
                "openfigi_report": openfigi_report,
            }
            st.session_state.report_pdf_bytes = pdf_bytes
            st.session_state.report_generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            st.session_state.report_error = report_error
            st.session_state.upload_batches += 1
        st.session_state.last_upload_signature = upload_signature
    elif st.session_state.analytics_payload is not None and st.session_state.report_pdf_bytes is None:
        # Avoid reprocessing documents: regenerate PDF from in-session canonical data only.
        payload = st.session_state.analytics_payload
        try:
            st.session_state.report_pdf_bytes = generate_portfolio_pdf_report(
                payload["df_tx"],
                payload["df_trade_priced"],
                payload["df_prices_long"],
                payload["df_master"],
            )
            st.session_state.report_generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            st.session_state.report_error = None
        except Exception as exc:  # noqa: BLE001
            st.session_state.report_error = f"{type(exc).__name__}: {exc}"

payload = st.session_state.analytics_payload
active_payload = payload if consent else None

if consent and st.session_state.report_error and active_payload is None:
    st.error(st.session_state.report_error)
elif active_payload is not None:
    openfigi_report = active_payload.get("openfigi_report") or {}
    if openfigi_report.get("mode") == "cache_only_no_api_key":
        st.warning(
            "OpenFIGI API key not configured: analytics runs with cache/fallback instrument metadata. "
            "Set `OPENFIGI_API_KEY` (for example in GitHub deployment) for fuller enrichment."
        )

tab_about, tab_annex, tab_overview, tab_performance, tab_risk, tab_risk_adv, tab_diagnostics, tab_report = st.tabs(
    ["About", "Annex", "Overview", "Performance", "Risk", "Risk Advanced", "Diagnostics", "Report"]
)

if active_payload is None:
    with tab_about:
        render_about()
    with tab_annex:
        render_annex()
    with tab_overview:
        st.info("No analytics available yet.")
    with tab_performance:
        st.info("No analytics available yet.")
    with tab_risk:
        st.info("No analytics available yet.")
    with tab_risk_adv:
        st.info("No analytics available yet.")
    with tab_diagnostics:
        st.info("No analytics available yet.")
    with tab_report:
        st.info("Upload one statement PDF to auto-generate the portfolio report.")
else:
    with tab_about:
        render_about()
    with tab_annex:
        render_annex()
    df_master = active_payload["df_master"]
    df_tx = active_payload["df_tx"]
    df_prices_long = active_payload["df_prices_long"]
    df_trade_priced = active_payload["df_trade_priced"]

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
    with tab_report:
        st.markdown("### Portfolio Report")
        st.caption(
            "Automatically generated after successful statement processing. "
            "Session-only: no persistent storage and no report files saved on disk."
        )
        if st.session_state.report_generated_at:
            st.caption(f"Generated at: {st.session_state.report_generated_at}")
        if st.session_state.report_pdf_bytes:
            st.download_button(
                label="Download PDF report",
                data=st.session_state.report_pdf_bytes,
                file_name=f"portfolio_report_{datetime.utcnow().date().isoformat()}.pdf",
                mime="application/pdf",
                width="content",
            )
        else:
            st.warning("Report generation failed for this upload.")
            if st.session_state.report_error:
                st.caption(f"Error: {st.session_state.report_error}")
