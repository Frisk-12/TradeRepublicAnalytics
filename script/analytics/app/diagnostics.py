#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostics tab renderer."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_diagnostics(df_master: pd.DataFrame, df_tx: pd.DataFrame, df_trade_priced: pd.DataFrame) -> None:
    st.markdown(
        "<h2 style='margin:0 0 .25rem 0;'>Diagnostics</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Trade pricing consistency, fiscal reconciliation, and audit tables.")

    if df_trade_priced.empty:
        st.warning("No trades with type 'Commercio' are available.")
        return

    d = df_trade_priced.copy()

    name_map = dict(zip(df_master["isin"], df_master["name"])) if not df_master.empty else {}
    fallback_name = (
        df_tx.dropna(subset=["isin"])
        .sort_values("date")
        .groupby("isin", as_index=False)["instrument_name"]
        .last()
        .set_index("isin")["instrument_name"]
        .to_dict()
        if "instrument_name" in df_tx.columns and not df_tx.empty
        else {}
    )
    d["name"] = d["isin"].map(name_map).fillna(d["isin"].map(fallback_name)).fillna(d["isin"])

    st.markdown("**Trade Pricing - Summary**")
    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    with k1:
        st.metric("Trades", int(len(d)))
    with k2:
        n_relevant = int(((d["abs_diff_pct"] > 0.005) | (d["diff_eur_abs"] > 5.0)).sum())
        st.metric("Mismatch", n_relevant)
    with k3:
        med_abs_diff_pct = float(d["abs_diff_pct"].median()) if not d.empty else 0.0
        st.metric("Median abs diff", f"{med_abs_diff_pct:.2%}")
    with k4:
        p95_abs_diff_pct = float(d["abs_diff_pct"].quantile(0.95)) if not d.empty else 0.0
        st.metric("P95 abs diff", f"{p95_abs_diff_pct:.2%}")

    st.divider()

    st.markdown("**Fiscal - FIFO and Tax Loss Carry-Forward (Zainetto)**")
    if "minus_pool_eur" not in d.columns:
        st.info("Fiscal fields are not available in this version.")
    else:
        latest_minus = float(d["minus_pool_eur"].dropna().iloc[-1]) if d["minus_pool_eur"].notna().any() else 0.0
        total_minus_added = float(d["minus_added_eur"].sum()) if "minus_added_eur" in d.columns else 0.0
        total_minus_applied = float(d["minus_applied_eur"].sum()) if "minus_applied_eur" in d.columns else 0.0

        f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
        f1.metric("Minus Pool (latest)", f"€ {latest_minus:,.2f}")
        f2.metric("Minus Added (sum)", f"€ {total_minus_added:,.2f}")
        f3.metric("Minus Applied (sum)", f"€ {total_minus_applied:,.2f}")
        f4.metric("Net Minus", f"€ {(total_minus_added - total_minus_applied):,.2f}")

        fiscal_cols = [
            "trade_date",
            "isin",
            "side",
            "quantity",
            "price_used",
            "notional_mkt",
            "capital_gain_eur",
            "minus_added_eur",
            "minus_applied_eur",
            "minus_pool_eur",
            "tax_base_gain_eur",
            "tax_capital_gain_eur",
        ]
        fiscal = d[fiscal_cols].sort_values(["trade_date", "isin"]).reset_index(drop=True)
        with st.expander("Fiscal details (FIFO / tax loss carry-forward)", expanded=False):
            st.dataframe(
                fiscal.style.format(
                    {
                        "price_used": "{:.4f}",
                        "notional_mkt": "{:.2f}",
                        "capital_gain_eur": "{:.2f}",
                        "minus_added_eur": "{:.2f}",
                        "minus_applied_eur": "{:.2f}",
                        "minus_pool_eur": "{:.2f}",
                        "tax_base_gain_eur": "{:.2f}",
                        "tax_capital_gain_eur": "{:.2f}",
                    }
                ),
                width="stretch",
            )

    st.divider()

    st.markdown("**Trade Pricing - Table**")
    isin_text = st.text_input("ISIN contains", "", key="diag_isin_filter")
    if isin_text:
        d = d[d["isin"].str.contains(isin_text.strip().upper(), na=False)]

    show_cols = [
        "trade_date",
        "name",
        "isin",
        "side",
        "quantity",
        "n_aggregated",
        "price_date_used",
        "days_gap",
        "price_used",
        "price_origin",
        "notional_mkt",
        "fee_eur",
        "tax_capital_gain_eur",
        "cash_flow_stmt",
        "cash_flow_model_abs",
        "cash_flow_model_net",
        "diff_eur_abs",
        "diff_pct",
        "tx_id",
    ]

    table = d[show_cols].sort_values(["trade_date", "isin"]).reset_index(drop=True)
    styled = (
        table.style.background_gradient(subset=["diff_eur_abs"], cmap="RdYlGn_r").format(
            {
                "price_used": "{:.4f}",
                "notional_mkt": "{:.2f}",
                "fee_eur": "{:.2f}",
                "tax_capital_gain_eur": "{:.2f}",
                "cash_flow_stmt": "{:.2f}",
                "cash_flow_model_abs": "{:.2f}",
                "cash_flow_model_net": "{:.2f}",
                "diff_eur_abs": "{:.2f}",
                "diff_pct": "{:.2%}",
            }
        )
    )
    st.dataframe(styled, width="stretch")
