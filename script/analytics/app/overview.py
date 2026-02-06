#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overview tab renderer."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from script.analytics.core.overview import build_overview_result


def _compute_contributors(df_trade_priced, df_prices_long, df_master, df_tx):
    name_map = dict(zip(df_master["isin"], df_master["name"])) if df_master is not None and not df_master.empty else {}

    trades = df_trade_priced.copy()
    trades["qty_net"] = trades["quantity"] * trades["side"].str.upper().map({"BUY": 1, "SELL": -1}).fillna(0)
    qty_net = trades.groupby("isin", as_index=False)["qty_net"].sum()

    last_price = (
        df_prices_long.sort_values("date")
        .groupby("isin", as_index=False)
        .last()[["isin", "price"]]
        .rename(columns={"price": "last_price"})
    )

    net_invested = trades.groupby("isin", as_index=False)["cash_flow_model"].sum().rename(columns={"cash_flow_model": "net_invested_raw"})

    contrib = qty_net.merge(last_price, on="isin", how="left").merge(net_invested, on="isin", how="left")
    contrib["net_invested"] = -contrib["net_invested_raw"].fillna(0.0)
    contrib["market_value"] = contrib["qty_net"] * contrib["last_price"]
    contrib["pnl"] = contrib["market_value"] - contrib["net_invested"]
    contrib["pnl_pct"] = contrib.apply(
        lambda r: (r["pnl"] / abs(r["net_invested"])) if r["net_invested"] not in (0, None) and abs(r["net_invested"]) > 1e-12 else 0.0,
        axis=1,
    )
    contrib = contrib[
        contrib["last_price"].notna() | (contrib["qty_net"].abs() < 1e-12)
    ].copy()
    contrib.loc[contrib["last_price"].isna(), "market_value"] = 0.0
    total_portfolio = contrib["market_value"].sum()
    contrib["weight_pct"] = contrib["market_value"] / total_portfolio if total_portfolio else 0.0
    contrib["name"] = contrib["isin"].map(name_map).fillna(contrib["isin"])
    return contrib


def _eur(v: float) -> str:
    return f"€ {v:,.2f}"


def _eur_delta(v: float | None, suffix: str = "") -> str | None:
    if v is None or pd.isna(v):
        return None
    sign = "+" if v >= 0 else "-"
    out = f"{sign}€ {abs(v):,.2f}"
    return f"{out} {suffix}".strip()


def _series_delta(series: pd.Series, lag_days: int, *, precision: int | None = None) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2 or lag_days < 1 or len(s) <= lag_days:
        return None
    if precision is not None:
        s = s.round(precision)
    prev = s.iloc[-1 - lag_days]
    return float(s.iloc[-1] - prev)


def render_overview(df_tx, df_trade_priced, df_prices_long, df_master) -> None:
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #eef2f7;
            border-radius: 14px;
            padding: 14px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    res = build_overview_result(
        df_tx=df_tx,
        df_trade_priced=df_trade_priced,
        df_prices_long=df_prices_long,
        start_date=None,
        end_date=None,
    )

    st.markdown(
        "<h1 style='font-size:2.9rem; margin:0 0 .2rem 0; font-weight:700;'>Overview</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Range: {res.start_date.date().isoformat()} → {res.end_date.date().isoformat()}"
    )

    chart = res.chart_df.sort_values("date").reset_index(drop=True)
    pnl_series = chart["account_value_ex_pm"] - chart["net_invested"]
    delta_port_1d = _series_delta(chart["portfolio_value"], 1, precision=2)
    delta_cash_1d = _series_delta(chart["cash_balance"], 1, precision=2)
    delta_net_30d = _series_delta(chart["net_invested"], 30, precision=2)
    delta_pnl_1d = _series_delta(pnl_series, 1, precision=2)
    delta_port_30d = _series_delta(chart["portfolio_value"], 30, precision=2)
    delta_pnl_30d = _series_delta(pnl_series, 30, precision=2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Portfolio Value",
        _eur(res.portfolio_value),
        delta=_eur_delta(delta_port_1d, "vs prev day"),
        help=(
            "Current market value of open positions: sum(qty_net per ISIN multiplied by latest available price). "
            "Delta uses 1-day change; shown as n/a when insufficient history."
        ),
    )
    c2.metric(
        "Cash Balance",
        _eur(res.cash_balance),
        delta=_eur_delta(delta_cash_1d, "vs prev day"),
        delta_color="off",
        help=(
            "Latest available cash balance from transactions (saldo column). "
            "Delta uses 1-day change; shown as n/a when insufficient history."
        ),
    )
    c3.metric(
        "Net Cash In",
        _eur(res.net_cash_in),
        delta=_eur_delta(delta_net_30d, "last 30d"),
        delta_color="off",
        help=(
            "Net sum of transfers from inception to date. "
            "Includes withdrawals if recorded as negative transfers. "
            "Delta uses 30-day change; shown as n/a when insufficient history."
        ),
    )
    c4.metric(
        "PnL",
        _eur(res.pnl_abs),
        delta=_eur_delta(delta_pnl_1d, "vs prev day"),
        help=(
            "Absolute Profit/Loss: (Account Value - Private Markets) - Net Invested. "
            "Private Markets remain at nominal value until realized. "
            "Delta uses 1-day change; shown as n/a when insufficient history."
        ),
    )
    st.caption(
        "30d movement "
        f"- Portfolio Value: {_eur_delta(delta_port_30d) or 'n/a'} | "
        f"PnL: {_eur_delta(delta_pnl_30d) or 'n/a'}"
    )

    chart = res.chart_df.copy()
    chart["return_pct"] = (chart["account_value_ex_pm"] - chart["net_invested"]) / chart["net_invested"]
    chart.loc[chart["net_invested"] <= 0, "return_pct"] = 0.0

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=chart["date"],
            y=chart["account_value"],
            mode="lines",
            name="Account Value",
            line={"width": 3, "color": "#0F4C81"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart["date"],
            y=chart["net_invested"],
            mode="lines",
            name="Net Invested",
            line={"width": 2, "color": "#3AA76D", "shape": "hv"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart["date"],
            y=chart["return_pct"] * 100.0,
            mode="lines",
            name="Return %",
            line={"width": 2, "color": "#F28E2B"},
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Account Value vs Net Invested",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="EUR",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=520,
    )
    fig.update_yaxes(tickprefix="€ ")
    fig.update_yaxes(
        title_text="Return %",
        secondary_y=True,
        ticksuffix="%",
        showgrid=False,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Contributors**")
    if df_trade_priced.empty or df_prices_long.empty:
        st.info("Insufficient data to compute contributors.")
    else:
        contrib = _compute_contributors(df_trade_priced, df_prices_long, df_master, df_tx)

        top = contrib.sort_values("pnl", ascending=False).head(5)
        bottom = contrib.sort_values("pnl", ascending=True).head(5)

        def _fmt(df):
            return df[["name", "pnl", "pnl_pct", "weight_pct"]].rename(
                columns={"name": "Name", "pnl": "PnL €", "pnl_pct": "PnL %", "weight_pct": "Weight %"}
            )

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Top 5**")
            st.dataframe(
                _fmt(top).style.format(
                    {
                        "PnL €": "€ {:,.2f}",
                        "PnL %": "{:.2%}",
                        "Weight %": "{:.2%}",
                    }
                ).set_properties(subset=["PnL %"], **{"font-weight": "700"}),
                width="stretch",
                hide_index=True,
            )
        with c_right:
            st.markdown("**Bottom 5**")
            st.dataframe(
                _fmt(bottom).style.format(
                    {
                        "PnL €": "€ {:,.2f}",
                        "PnL %": "{:.2%}",
                        "Weight %": "{:.2%}",
                    }
                ).set_properties(subset=["PnL %"], **{"font-weight": "700"}),
                width="stretch",
                hide_index=True,
            )

        with st.expander("All Positions - PnL (weight >= 0.5%)"):
            filtered = contrib[contrib["weight_pct"] >= 0.005].sort_values("pnl", ascending=False)
            st.caption("Showing only positions with weight >= 0.5% of portfolio value.")
            all_rows = _fmt(filtered)
            st.dataframe(
                all_rows.style.format(
                    {
                        "PnL €": "€ {:,.2f}",
                        "PnL %": "{:.2%}",
                        "Weight %": "{:.2%}",
                    }
                ).set_properties(subset=["PnL %"], **{"font-weight": "700"}),
                width="stretch",
                hide_index=True,
            )

    with st.expander("Account Value movement details (table)"):
        detail = res.chart_df.copy()
        detail["account_value_delta"] = detail["account_value"].diff().fillna(detail["account_value"])
        detail = detail.sort_values("date").reset_index(drop=True)
        st.dataframe(
            detail.style.format(
                {
                    "cash_balance": "€ {:,.2f}",
                    "portfolio_value": "€ {:,.2f}",
                    "net_invested": "€ {:,.2f}",
                    "account_value": "€ {:,.2f}",
                    "account_value_delta": "€ {:,.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown("**Allocations**")
    if df_trade_priced.empty or df_prices_long.empty or df_master is None:
        st.info("Insufficient data to compute allocations.")
        return

    # Build current market value per ISIN.
    trades = df_trade_priced.copy()
    trades["qty_net"] = trades["quantity"] * trades["side"].str.upper().map({"BUY": 1, "SELL": -1}).fillna(0)
    qty_net = trades.groupby("isin", as_index=False)["qty_net"].sum()
    last_price = (
        df_prices_long.sort_values("date")
        .groupby("isin", as_index=False)
        .last()[["isin", "price"]]
        .rename(columns={"price": "last_price"})
    )
    mv = qty_net.merge(last_price, on="isin", how="left")
    mv["market_value"] = mv["qty_net"] * mv["last_price"]
    mv = mv[mv["market_value"].notna() & (mv["market_value"] > 0)]

    master = df_master.copy()
    alloc_base = mv.merge(master, on="isin", how="left")
    name_map_tx = (
        df_tx.dropna(subset=["isin", "instrument_name"])
        .sort_values("date")
        .groupby("isin", as_index=False)["instrument_name"]
        .last()
        .set_index("isin")["instrument_name"]
        .to_dict()
        if "instrument_name" in df_tx.columns and not df_tx.empty
        else {}
    )
    def _infer_type(name: str) -> str:
        n = str(name or "").lower()
        if any(k in n for k in ["optionsschein", "warrant", "turbo", "knock"]):
            return "warrant"
        return ""

    alloc_base["instrument_type"] = alloc_base["instrument_type"].fillna("")
    inferred = alloc_base["isin"].map(name_map_tx).map(_infer_type)
    alloc_base["instrument_type"] = alloc_base["instrument_type"].where(alloc_base["instrument_type"] != "", inferred)
    alloc_base["asset_class"] = alloc_base["asset_class"].fillna("other")
    alloc_base["instrument_type"] = alloc_base["instrument_type"].fillna("other")
    alloc_base["security_type"] = alloc_base["security_type"].fillna("")
    alloc_base["security_type2"] = alloc_base["security_type2"].fillna("")

    def _is_warrant_row(r) -> bool:
        st = str(r["security_type"]).lower()
        st2 = str(r["security_type2"]).lower()
        it = str(r["instrument_type"]).lower()
        return any(k in st for k in ["warrant", "option", "certificate", "derivative"]) or any(
            k in st2 for k in ["warrant", "option", "certificate", "derivative"]
        ) or it in {"warrant", "turbo", "derivative"}

    alloc_base["asset_class"] = alloc_base.apply(
        lambda r: "warrant/derivatives" if _is_warrant_row(r) else str(r["asset_class"]).lower(),
        axis=1,
    )

    # Asset class allocation.
    asset_alloc = (
        alloc_base.groupby("asset_class", as_index=False)["market_value"]
        .sum()
        .sort_values("market_value", ascending=False)
    )

    # Separate ETF vs Equity exposures (do not mix nomenclatures).
    etf_base = alloc_base[alloc_base["asset_class"] == "etf"].copy()
    equity_base = alloc_base[alloc_base["asset_class"] == "equity"].copy()

    # Country allocation — ETF
    country_rows_etf = []
    for _, r in etf_base.iterrows():
        mv_val = float(r["market_value"] or 0.0)
        if mv_val <= 0:
            continue
        just = r.get("justetf_exposure") or {}
        if isinstance(just, dict) and just.get("countries"):
            for c in just.get("countries") or []:
                country_rows_etf.append(
                    {
                        "country": c.get("country") or "Unknown",
                        "market_value": mv_val * (float(c.get("weight_pct") or 0.0) / 100.0),
                    }
                )
    country_alloc = pd.DataFrame(country_rows_etf)
    if country_alloc.empty or "country" not in country_alloc.columns:
        country_alloc = pd.DataFrame(columns=["country", "market_value"])
    else:
        country_alloc = country_alloc.groupby("country", as_index=False)["market_value"].sum().sort_values("market_value", ascending=False)

    # Sector allocation — ETF
    sector_rows_etf = []
    for _, r in etf_base.iterrows():
        mv_val = float(r["market_value"] or 0.0)
        if mv_val <= 0:
            continue
        just = r.get("justetf_exposure") or {}
        if isinstance(just, dict) and just.get("sectors"):
            for s in just.get("sectors") or []:
                sector_rows_etf.append(
                    {
                        "sector": s.get("sector") or "Other",
                        "market_value": mv_val * (float(s.get("weight_pct") or 0.0) / 100.0),
                    }
                )
    sector_alloc = pd.DataFrame(sector_rows_etf)
    if sector_alloc.empty or "sector" not in sector_alloc.columns:
        sector_alloc = pd.DataFrame(columns=["sector", "market_value"])
    else:
        sector_alloc = sector_alloc.groupby("sector", as_index=False)["market_value"].sum().sort_values("market_value", ascending=False)

    total_mv = alloc_base["market_value"].sum()
    etf_mv = etf_base["market_value"].sum()
    equity_mv = equity_base["market_value"].sum()
    warr_mv = alloc_base[alloc_base["asset_class"] == "warrant/derivatives"]["market_value"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Portfolio Value", f"€ {total_mv:,.2f}")
    k2.metric("ETF Weight", f"{(etf_mv / total_mv if total_mv else 0.0):.2%}")
    k3.metric("Equity Weight", f"{(equity_mv / total_mv if total_mv else 0.0):.2%}")
    k4.metric("Warrant Weight", f"{(warr_mv / total_mv if total_mv else 0.0):.2%}")

    contrib_alloc = _compute_contributors(df_trade_priced, df_prices_long, df_master, df_tx)
    tab_macro, tab_etf, tab_equity = st.tabs(["Macro", "ETF Exposure", "Equity Exposure"])
    with tab_macro:
        fig_asset = px.pie(asset_alloc, names="asset_class", values="market_value", hole=0.55)
        fig_asset.update_layout(title="Asset Class", legend_title_text="", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_asset, width="stretch")
        st.caption("Asset-class breakdown.")

        def _drop_empty_cols(df: pd.DataFrame) -> pd.DataFrame:
            keep = []
            for c in df.columns:
                s = df[c]
                if pd.api.types.is_numeric_dtype(s):
                    if s.notna().any():
                        keep.append(c)
                else:
                    if s.fillna("").astype(str).str.strip().ne("").any():
                        keep.append(c)
            return df[keep]

        with st.expander("Equity — Details"):
            eq_detail = equity_base.copy()
            if eq_detail.empty:
                st.caption("No equity positions.")
            else:
                eq_detail["name"] = eq_detail["isin"].map(name_map_tx).fillna(eq_detail["name"]).fillna(eq_detail["isin"])
                eq_detail = eq_detail.merge(contrib_alloc[["isin", "pnl", "pnl_pct"]], on="isin", how="left")
                y_cols = ["country", "sector", "industry", "marketCap", "beta"]
                def _eq_info(r):
                    y = r.get("yfinance_profile") or {}
                    return pd.Series({c: y.get(c) for c in y_cols})
                eq_extra = eq_detail.apply(_eq_info, axis=1)
                eq_detail = pd.concat([eq_detail, eq_extra], axis=1)
                eq_detail = eq_detail[
                    ["name", "isin", "market_value", "pnl", "pnl_pct", "country", "sector", "industry", "marketCap", "beta"]
                ]
                eq_detail = _drop_empty_cols(eq_detail)
                fmt_cols = {c: "€ {:,.2f}" for c in ["market_value", "pnl"] if c in eq_detail.columns}
                if "pnl_pct" in eq_detail.columns:
                    fmt_cols["pnl_pct"] = "{:.2%}"
                if "marketCap" in eq_detail.columns:
                    fmt_cols["marketCap"] = "€ {:,.0f}"
                if "beta" in eq_detail.columns:
                    fmt_cols["beta"] = "{:.2f}"
                for c in ["market_value", "pnl", "pnl_pct", "marketCap", "beta"]:
                    if c in eq_detail.columns:
                        eq_detail[c] = eq_detail[c].fillna(0.0)
                eq_styler = eq_detail.sort_values("market_value", ascending=False).style.format(fmt_cols)
                if "pnl_pct" in eq_detail.columns:
                    eq_styler = eq_styler.set_properties(subset=["pnl_pct"], **{"font-weight": "700"})
                st.dataframe(eq_styler, width="stretch", hide_index=True)

        with st.expander("ETF — Details"):
            etf_detail = etf_base.copy()
            if etf_detail.empty:
                st.caption("No ETF positions.")
            else:
                etf_detail["name"] = etf_detail["isin"].map(name_map_tx).fillna(etf_detail["name"]).fillna(etf_detail["isin"])
                etf_detail = etf_detail.merge(contrib_alloc[["isin", "pnl", "pnl_pct"]], on="isin", how="left")
                def _etf_top(r):
                    j = r.get("justetf_exposure") or {}
                    countries = j.get("countries") or []
                    sectors = j.get("sectors") or []
                    top_country = countries[0]["country"] if countries else None
                    top_sector = sectors[0]["sector"] if sectors else None
                    n_holdings = (j.get("meta") or {}).get("n_holdings")
                    return pd.Series({"top_country": top_country, "top_sector": top_sector, "n_holdings": n_holdings})
                etf_extra = etf_detail.apply(_etf_top, axis=1)
                etf_detail = pd.concat([etf_detail, etf_extra], axis=1)
                etf_detail = etf_detail[
                    ["name", "isin", "market_value", "pnl", "pnl_pct", "top_country", "top_sector", "n_holdings"]
                ]
                etf_detail = _drop_empty_cols(etf_detail)
                fmt_cols = {c: "€ {:,.2f}" for c in ["market_value", "pnl"] if c in etf_detail.columns}
                if "pnl_pct" in etf_detail.columns:
                    fmt_cols["pnl_pct"] = "{:.2%}"
                if "n_holdings" in etf_detail.columns:
                    fmt_cols["n_holdings"] = "{:.0f}"
                for c in ["market_value", "pnl", "pnl_pct", "n_holdings"]:
                    if c in etf_detail.columns:
                        etf_detail[c] = etf_detail[c].fillna(0.0)
                etf_styler = etf_detail.sort_values("market_value", ascending=False).style.format(fmt_cols)
                if "pnl_pct" in etf_detail.columns:
                    etf_styler = etf_styler.set_properties(subset=["pnl_pct"], **{"font-weight": "700"})
                st.dataframe(etf_styler, width="stretch", hide_index=True)

        with st.expander("Warrant / Derivatives — Details"):
            w_base = alloc_base[alloc_base["asset_class"] == "warrant/derivatives"].copy()
            if w_base.empty:
                st.caption("No warrant/derivative positions.")
            else:
                w_base["name"] = w_base["isin"].map(name_map_tx).fillna(w_base["name"]).fillna(w_base["isin"])
                w_base = w_base.merge(contrib_alloc[["isin", "pnl", "pnl_pct"]], on="isin", how="left")
                w_base = w_base[["name", "isin", "market_value", "pnl", "pnl_pct"]]
                fmt_cols = {
                    "market_value": "€ {:,.2f}",
                    "pnl": "€ {:,.2f}",
                    "pnl_pct": "{:.2%}",
                }
                w_base[["market_value", "pnl", "pnl_pct"]] = w_base[["market_value", "pnl", "pnl_pct"]].fillna(0.0)
                st.dataframe(
                    w_base.sort_values("market_value", ascending=False).style.format(fmt_cols).set_properties(
                        subset=["pnl_pct"],
                        **{"font-weight": "700"},
                    ),
                    width="stretch",
                    hide_index=True,
                )
    with tab_etf:
        fig_country = px.bar(country_alloc.head(12), x="market_value", y="country", orientation="h")
        fig_country.update_layout(title="Country (Top 12)", xaxis_title="EUR", yaxis_title="")
        st.plotly_chart(fig_country, width="stretch")
        fig_sector = px.bar(sector_alloc.head(12), x="market_value", y="sector", orientation="h")
        fig_sector.update_layout(title="Sector (Top 12)", xaxis_title="EUR", yaxis_title="")
        st.plotly_chart(fig_sector, width="stretch")
    with tab_equity:
        # Country allocation — Equity
        country_rows_eq = []
        for _, r in equity_base.iterrows():
            mv_val = float(r["market_value"] or 0.0)
            if mv_val <= 0:
                continue
            yfin = r.get("yfinance_profile") or {}
            if isinstance(yfin, dict) and yfin.get("country"):
                country_rows_eq.append({"country": yfin.get("country"), "market_value": mv_val})
            else:
                country_rows_eq.append({"country": "Unknown", "market_value": mv_val})
        country_eq = pd.DataFrame(country_rows_eq)
        if country_eq.empty or "country" not in country_eq.columns:
            country_eq = pd.DataFrame(columns=["country", "market_value"])
        else:
            country_eq = country_eq.groupby("country", as_index=False)["market_value"].sum().sort_values("market_value", ascending=False)
        sector_rows_eq = []
        for _, r in equity_base.iterrows():
            mv_val = float(r["market_value"] or 0.0)
            if mv_val <= 0:
                continue
            yfin = r.get("yfinance_profile") or {}
            if isinstance(yfin, dict) and yfin.get("sector"):
                sector_rows_eq.append({"sector": yfin.get("sector"), "market_value": mv_val})
            else:
                sector_rows_eq.append({"sector": "Other", "market_value": mv_val})
        sector_eq = pd.DataFrame(sector_rows_eq)
        if sector_eq.empty or "sector" not in sector_eq.columns:
            sector_eq = pd.DataFrame(columns=["sector", "market_value"])
        else:
            sector_eq = sector_eq.groupby("sector", as_index=False)["market_value"].sum().sort_values("market_value", ascending=False)

        fig_country_eq = px.bar(country_eq.head(12), x="market_value", y="country", orientation="h")
        fig_country_eq.update_layout(title="Country (Top 12)", xaxis_title="EUR", yaxis_title="")
        st.plotly_chart(fig_country_eq, width="stretch")
        fig_sector_eq = px.bar(sector_eq.head(12), x="market_value", y="sector", orientation="h")
        fig_sector_eq.update_layout(title="Sector (Top 12)", xaxis_title="EUR", yaxis_title="")
        st.plotly_chart(fig_sector_eq, width="stretch")

    with st.expander("Allocations — Data Tables"):
        asset_view = asset_alloc.copy()
        asset_view["market_value"] = asset_view["market_value"].fillna(0.0)
        st.dataframe(
            asset_view.style.format({"market_value": "€ {:,.2f}"}),
            width="stretch",
            hide_index=True,
        )
        country_view = country_alloc.copy()
        country_view["market_value"] = country_view["market_value"].fillna(0.0)
        st.dataframe(
            country_view.style.format({"market_value": "€ {:,.2f}"}),
            width="stretch",
            hide_index=True,
        )
        sector_view = sector_alloc.copy()
        sector_view["market_value"] = sector_view["market_value"].fillna(0.0)
        st.dataframe(
            sector_view.style.format({"market_value": "€ {:,.2f}"}),
            width="stretch",
            hide_index=True,
        )
