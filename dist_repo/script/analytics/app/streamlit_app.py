#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-page Streamlit app for canonical transactions + trade pricing."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.analytics.core.tables import build_canonical_tables, load_datalake
from script.analytics.core.trades import build_trade_pricing_and_alerts

DEFAULT_DATALAKE_JSON = "/Users/andreadesogus/Desktop/portfolio_analytics/script/datalake/datalake.json"
CACHE_VERSION = "v3"


def load_all(path: str, cache_version: str):
    _ = cache_version
    datalake = load_datalake(path)
    df_master, df_tx, df_prices_long = build_canonical_tables(datalake)
    df_trade_priced, df_alerts = build_trade_pricing_and_alerts(df_tx, df_prices_long, df_master)
    return df_master, df_tx, df_prices_long, df_trade_priced, df_alerts

st.set_page_config(page_title="Diagnostics", page_icon="🩺", layout="wide")
st.title("Diagnostics — Trade pricing consistency")

path = st.text_input("Path DATALAKE JSON", DEFAULT_DATALAKE_JSON)
if not Path(path).exists():
    st.error("DATALAKE JSON non trovato.")
    st.stop()

with st.spinner("Loading canonical tables..."):
    df_master, df_tx, df_prices_long, df_trade_priced, _ = load_all(path, CACHE_VERSION)

if df_trade_priced.empty:
    st.warning("Nessun trade di tipo Commercio disponibile.")
    st.stop()

df_trade_priced = df_trade_priced.copy()
# Backward compatibility in case cached/legacy dataframe misses new columns.
if "diff_eur" not in df_trade_priced.columns and "diff_cash_flow" in df_trade_priced.columns:
    df_trade_priced["diff_eur"] = pd.to_numeric(df_trade_priced["diff_cash_flow"], errors="coerce").abs()
if "diff_eur_abs" not in df_trade_priced.columns:
    df_trade_priced["diff_eur_abs"] = pd.to_numeric(df_trade_priced.get("diff_eur"), errors="coerce").abs()
if "diff_pct" not in df_trade_priced.columns:
    denom_raw = (
        df_trade_priced["cash_flow_model_abs"]
        if "cash_flow_model_abs" in df_trade_priced.columns
        else df_trade_priced.get("cash_flow_model")
    )
    denom = pd.Series(pd.to_numeric(denom_raw, errors="coerce"), index=df_trade_priced.index).replace(0, pd.NA)
    df_trade_priced["diff_pct"] = pd.to_numeric(df_trade_priced.get("diff_eur_abs"), errors="coerce") / denom
if "abs_diff_pct" not in df_trade_priced.columns:
    df_trade_priced["abs_diff_pct"] = pd.to_numeric(df_trade_priced["diff_pct"], errors="coerce").abs()
if "fee_eur" not in df_trade_priced.columns:
    df_trade_priced["fee_eur"] = 1.0
if "cash_flow_model_abs" not in df_trade_priced.columns:
    df_trade_priced["cash_flow_model_abs"] = pd.to_numeric(df_trade_priced.get("cash_flow_model"), errors="coerce").abs()
if "price_origin" not in df_trade_priced.columns:
    df_trade_priced["price_origin"] = "unknown"
df_trade_priced["price_origin"] = df_trade_priced["price_origin"].fillna("unknown")
if "n_aggregated" not in df_trade_priced.columns:
    df_trade_priced["n_aggregated"] = 1

if "cash_flow_model_net" not in df_trade_priced.columns:
    df_trade_priced["cash_flow_model_net"] = pd.to_numeric(df_trade_priced.get("cash_flow_model"), errors="coerce")
if "cash_flow_model_abs_net" not in df_trade_priced.columns:
    df_trade_priced["cash_flow_model_abs_net"] = pd.to_numeric(df_trade_priced.get("cash_flow_model_abs"), errors="coerce")
if "tax_capital_gain_eur" not in df_trade_priced.columns:
    df_trade_priced["tax_capital_gain_eur"] = 0.0
if "tax_base_gain_eur" not in df_trade_priced.columns:
    df_trade_priced["tax_base_gain_eur"] = 0.0

# Ensure numeric types; preserve reconciliation already computed by core logic (incl. tax).
for col in [
    "notional_mkt",
    "fee_eur",
    "cash_flow_stmt",
    "cash_flow_model_abs",
    "cash_flow_model",
    "cash_flow_model_net",
    "cash_flow_model_abs_net",
    "diff_cash_flow",
    "diff_eur_abs",
    "diff_pct",
    "abs_diff_pct",
    "capital_gain_eur",
    "minus_added_eur",
    "minus_applied_eur",
    "minus_pool_eur",
    "tax_capital_gain_eur",
    "tax_base_gain_eur",
]:
    if col in df_trade_priced.columns:
        df_trade_priced[col] = pd.to_numeric(df_trade_priced[col], errors="coerce")

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
df_trade_priced["name"] = (
    df_trade_priced["isin"].map(name_map).fillna(df_trade_priced["isin"].map(fallback_name)).fillna(df_trade_priced["isin"])
)

# SECTION 1 - KPI
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Numero trade", int(len(df_trade_priced)))
with k2:
    n_relevant = int(((df_trade_priced["abs_diff_pct"] > 0.005) | (df_trade_priced["diff_eur_abs"] > 5.0)).sum())
    st.metric("Mismatch rilevanti", n_relevant)
with k3:
    med_abs_diff_pct = float(df_trade_priced["abs_diff_pct"].median()) if not df_trade_priced.empty else 0.0
    st.metric("Median abs diff (%)", f"{med_abs_diff_pct:.2%}")
st.caption(f"95° percentile abs diff (%): {df_trade_priced['abs_diff_pct'].quantile(0.95):.2%}")
st.caption("price_origin: justetf | yfinance | vontobel | sogen | account_statement | missing.")

# Filters
isin_text = st.text_input("Filtro ISIN contiene", "")

filtered = df_trade_priced.copy()
if isin_text:
    filtered = filtered[filtered["isin"].str.contains(isin_text.strip().upper(), na=False)]

# SECTION 2 - Trade table
st.subheader("Trade Table")
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

table_df = filtered[show_cols].sort_values(["trade_date", "isin"]).reset_index(drop=True)
styled = (
    table_df
    .style.background_gradient(subset=["diff_eur_abs"], cmap="RdYlGn_r")
    .format({
        "price_used": "{:.4f}",
        "notional_mkt": "{:.2f}",
        "fee_eur": "{:.2f}",
        "tax_capital_gain_eur": "{:.2f}",
        "cash_flow_stmt": "{:.2f}",
        "cash_flow_model_abs": "{:.2f}",
        "cash_flow_model_net": "{:.2f}",
        "diff_eur_abs": "{:.2f}",
        "diff_pct": "{:.2%}",
    })
)
st.dataframe(styled, use_container_width=True)
