#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""About tab renderer."""

from __future__ import annotations

import streamlit as st


def render_about() -> None:
    st.markdown(
        """
        <style>
        .about-hero {
            border: 1px solid #dbe7f1;
            background: linear-gradient(135deg, #f7fbff 0%, #edf4fb 100%);
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }
        .about-card {
            border: 1px solid #e4edf6;
            border-radius: 14px;
            padding: 1rem 1rem;
            background: #ffffff;
            min-height: 156px;
        }
        .about-card h4 {
            margin: 0 0 .4rem 0;
            font-size: 1.01rem;
        }
        .about-card p {
            margin: 0;
            color: #3d4f62;
            line-height: 1.46;
            font-size: .93rem;
        }
        .about-note {
            border: 1px solid #d7e5f2;
            background: #f7fbff;
            border-radius: 10px;
            padding: .7rem .85rem;
            color: #355069;
        }
        .about-warn {
            border: 1px solid #f1d4d4;
            background: #fff6f6;
            border-radius: 10px;
            padding: .8rem .9rem;
            color: #6b2f2f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="about-hero">
          <h1 style="margin:0;font-size:2.1rem;">About This Platform</h1>
          <p style="margin:.6rem 0 0 0;color:#334d63;max-width:990px;">
            Deeper portfolio intelligence for Trade Republic investors.
            The platform accepts Trade Republic Account Statement PDFs only,
            expands the broker export into institutional-style performance/risk analytics,
            and automatically generates a clean report for decision support.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="about-card">
              <h4>Stateless by Design</h4>
              <p>No persistent storage for uploaded statements or generated reports.
              Data is processed in session memory and temporary runtime files only.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="about-card">
              <h4>Input Scope</h4>
              <p>Only Trade Republic Account Statement PDFs are supported.
              The app processes one statement per run to keep the workflow controlled and auditable.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="about-card">
              <h4>External Data Enrichment</h4>
              <p>Market and metadata enrichment relies on specialized providers to
              improve price coverage and instrument classification.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="about-card">
              <h4>Risk and Control Focus</h4>
              <p>Performance, drawdown, concentration and diagnostics are built from
              the same canonical layer for consistent decision support.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Dashboard Description")
    st.markdown(
        "- Trade Republic statement-driven dashboard for deeper portfolio monitoring and reporting.\n"
        "- Unified framework across Overview, Performance, Risk, Risk Advanced, and Diagnostics.\n"
        "- Extends standard broker exports with concentration, drawdown, attribution, and data-quality diagnostics.\n"
        "- Full methodology formulas are documented in the **Annex** tab.\n"
        "- Automatic PDF report generation after successful statement processing.\n"
        "- Designed for clarity, reproducibility, and decision support."
    )

    st.markdown("### Warnings")
    st.markdown(
        """
        <div class="about-warn">
          <strong>Read before use</strong><br/>
          Upload only statements you are authorized to process. This tool is not investment advice,
          and output quality depends on statement completeness and external market-data availability.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "- Only **Trade Republic Account Statement PDFs** are accepted as input.\n"
        "- Current workflow supports one account statement per run.\n"
        "- Trade Republic export granularity can limit some analytics depth.\n"
        "- Corporate actions, intraday details, and tax specifics may be partially represented."
    )

    st.markdown("### Transparency")
    st.markdown(
        "- **Trade Republic account statements**: primary transactional source.\n"
        "- **OpenFIGI**: security master enrichment by ISIN.\n"
        "- **Yahoo Finance**: equity prices and FX support.\n"
        "- **JustETF**: ETF metadata and exposure enrichment.\n"
        "- **Vontobel / Societe Generale warrants**: derivative price support when available."
    )
    st.markdown(
        "- No uploaded statement is persisted as an application artifact.\n"
        "- No generated PDF report is permanently saved by the application.\n"
        "- Session refresh/restart clears in-memory analytics context.\n"
        "- External calls share instrument identifiers (ISIN/ticker), not balances/quantities by design.\n"
        "- Project implementation was supported with GPT Codex 5.2 and 5.3."
    )

    st.markdown("### Methodology Assumptions (Italian Baseline)")
    st.markdown(
        "- Fiscal engine assumes Italian capital-gain taxation at **26%** on taxable gains.\n"
        "- Tax loss carry-forward (`zainetto fiscale`) is modeled for **non-UCITS** instruments with FIFO usage and 4-year expiry.\n"
        "- FIFO lot matching is used for sell-side cost basis reconstruction.\n"
        "- Trade model uses a default execution fee of **EUR 1.00** when not otherwise specified.\n"
        "- If market prices are missing for a trade date, statement-implied prices are used as fallback when possible.\n"
        "- For instruments without usable price history (including delisted/unpriced cases), missing-price adjustment is applied on first trade date, effectively treating unavailable-price positions as opened/closed on that date for continuity.\n"
        "- TWR external flows are derived from transfer rows (`Bonifico`) only.\n"
        "- Performance timeline is anchored to the first transfer date when available."
    )

    st.markdown(
        """
        <div class="about-note">
          This platform is built for portfolio analytics and reporting workflows only.
          It is not legal, tax, or investment advice.
        </div>
        """,
        unsafe_allow_html=True,
    )
