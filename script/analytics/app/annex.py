#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Annex tab renderer with formulas and metric definitions."""

from __future__ import annotations

import streamlit as st


def render_annex() -> None:
    st.markdown(
        """
        <style>
        .annex-hero {
            border: 1px solid #dbe7f1;
            background: linear-gradient(135deg, #f8fbff 0%, #edf4fb 100%);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }
        .annex-card {
            border: 1px solid #e2ecf6;
            border-radius: 12px;
            padding: .85rem .95rem;
            background: #ffffff;
            margin-bottom: .6rem;
        }
        .annex-card h4 {
            margin: 0 0 .35rem 0;
            font-size: 1rem;
        }
        .annex-card p {
            margin: 0;
            color: #3f556b;
            line-height: 1.45;
            font-size: .92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="annex-hero">
          <h1 style="margin:0;font-size:2.0rem;">Annex: Methods and Formulas</h1>
          <p style="margin:.5rem 0 0 0;color:#355069;max-width:980px;">
            Technical reference for the key analytics used in this dashboard and in the generated PDF report.
            The formulas below focus on the most non-trivial metrics.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1) Flow-Adjusted Daily Return and TWR")
    st.markdown(
        """
        <div class="annex-card">
          <h4>Daily return with external flows</h4>
          <p>
            Daily portfolio return is adjusted for external flows (deposits/withdrawals) so performance is not distorted by capital movements.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(r"r_t = \frac{V_t - V_{t-1} - F_t}{V_{t-1}}")
    st.markdown("Where:")
    st.markdown("- `V_t`: portfolio/account value at day `t`")
    st.markdown("- `F_t`: external net flow at day `t` (e.g. Bonifico)")
    st.latex(r"\mathrm{TWR} = \prod_{t=1}^{T}(1 + r_t) - 1")

    st.markdown("### 2) Drawdown and Max Drawdown")
    st.latex(r"\mathrm{DD}_t = \frac{V_t}{\max_{s \le t} V_s} - 1")
    st.latex(r"\mathrm{MaxDD} = \min_t(\mathrm{DD}_t)")
    st.markdown("- `Current Drawdown` is the latest value of `DD_t`.")

    st.markdown("### 3) Volatility (Annualized)")
    st.latex(r"\sigma_{\mathrm{ann}} = \mathrm{std}(r_t)\cdot\sqrt{252}")
    st.markdown(
        "- In Risk Advanced, `Daily/Weekly/Monthly` volatility uses rolling windows with 2/7/30 observations."
    )

    st.markdown("### 4) Tail Risk: VaR and Expected Shortfall")
    st.latex(r"\mathrm{VaR}_{95} = Q_{0.05}(r_t), \quad \mathrm{VaR}_{99} = Q_{0.01}(r_t)")
    st.latex(r"\mathrm{ES}_{95} = \mathbb{E}[r_t \mid r_t \le \mathrm{VaR}_{95}]")
    st.latex(r"\mathrm{ES}_{99} = \mathbb{E}[r_t \mid r_t \le \mathrm{VaR}_{99}]")
    st.markdown("- VaR is a percentile threshold, ES is the average loss beyond that threshold.")

    st.markdown("### 5) Risk Proxy and Concentration")
    st.latex(r"\mathrm{RiskProxy}_i = w_i \cdot \sigma_{30d,i}")
    st.markdown("- `w_i`: portfolio weight of instrument `i`")
    st.markdown("- `σ_30d,i`: 30-day annualized volatility of instrument `i`")
    st.markdown("- Risk contributors are ranked by this proxy.")

    st.markdown("### 6) Correlation Metrics")
    st.latex(
        r"\bar{\rho}^{(w)}_t = \frac{2}{n(n-1)}\sum_{i<j}\mathrm{corr}\left(r_{i,t-w+1:t}, r_{j,t-w+1:t}\right)"
    )
    st.markdown("- Rolling average pairwise correlation is shown with Daily/Weekly/Monthly windows.")

    st.markdown("### 7) Operational Assumptions (Summary)")
    st.markdown("- Input scope: only Trade Republic Account Statement PDFs.")
    st.markdown("- Session-only processing: no persistent storage for statements/reports.")
    st.markdown("- Single statement per run.")
    st.markdown("- Italian fiscal baseline assumptions are documented in the About tab.")
