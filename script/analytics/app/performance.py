#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performance tab renderer (TWR-based)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from script.analytics.core.overview import (
    _cash_balance_series,
    _missing_price_adjustment_series,
    _net_invested_series,
    _portfolio_value_series,
)


def _pct(v: float) -> str:
    return f"{v:.2%}"


def _table_style(styler):
    return styler.set_table_styles(
        [
            {"selector": "th", "props": [("background-color", "#f3f5f8"), ("font-weight", "600"), ("color", "#222")]},
            {"selector": "td", "props": [("padding", "6px 10px")]},
        ]
    )


@st.cache_data(show_spinner=False)
def _build_account_series(df_tx, df_trade_priced, df_prices_long):
    if df_tx.empty and df_prices_long.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    tx_dates = pd.to_datetime(df_tx["date"], errors="coerce") if not df_tx.empty else pd.Series(dtype="datetime64[ns]")
    px_dates = pd.to_datetime(df_prices_long["date"], errors="coerce") if not df_prices_long.empty else pd.Series(dtype="datetime64[ns]")
    candidates = [d for d in [tx_dates.min(), px_dates.min()] if pd.notna(d)]
    start = pd.to_datetime(min(candidates), errors="coerce") if candidates else pd.NaT
    end = pd.Timestamp.today().normalize()
    if pd.isna(start):
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    portfolio_series = _portfolio_value_series(df_trade_priced, df_prices_long, df_tx, start, end)
    cash_series = _cash_balance_series(df_tx, start, end, anchor_date=start, anchor_saldo=0.0)
    cash_series = cash_series.add(_missing_price_adjustment_series(df_trade_priced, df_prices_long, start, end), fill_value=0.0)
    net_invested_series = _net_invested_series(df_tx, start, end)

    common_idx = portfolio_series.index.union(cash_series.index).union(net_invested_series.index)
    portfolio_series = portfolio_series.reindex(common_idx).ffill().fillna(0.0)
    cash_series = cash_series.reindex(common_idx).ffill().fillna(0.0)
    net_invested_series = net_invested_series.reindex(common_idx).ffill().fillna(0.0)
    account_series = cash_series + portfolio_series

    pm_series = pd.Series(0.0, index=account_series.index)
    pm_tx = df_tx.copy()
    if not pm_tx.empty:
        pm_tx["date"] = pd.to_datetime(pm_tx["date"], errors="coerce").dt.normalize()
        pm_tx["description"] = pm_tx["description"].astype(str).str.lower()
        pm_tx = pm_tx[(pm_tx["type"] == "Commercio") & (pm_tx["isin"].isna()) & pm_tx["description"].str.contains("private market")]
        if not pm_tx.empty:
            pm_tx["cash_flow"] = pd.to_numeric(pm_tx["cash_flow"], errors="coerce").fillna(0.0)
            daily_pm = pm_tx.groupby("date", as_index=True)["cash_flow"].sum().reindex(account_series.index, fill_value=0.0)
            pm_series = (-daily_pm).cumsum()

    return account_series, (account_series - pm_series), net_invested_series


@st.cache_data(show_spinner=False)
def _external_flows_series(df_tx: pd.DataFrame, _index: pd.DatetimeIndex) -> pd.Series:
    if df_tx.empty or "type" not in df_tx.columns:
        return pd.Series(0.0, index=_index)
    tx = df_tx.copy()
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce").dt.normalize()
    mask = tx["type"].astype(str).str.lower().eq("bonifico")
    tx = tx[mask]
    if tx.empty:
        return pd.Series(0.0, index=_index)
    if "cash_flow" in tx.columns:
        flow = pd.to_numeric(tx["cash_flow"], errors="coerce")
    else:
        inflow = pd.to_numeric(tx.get("in_entrata"), errors="coerce")
        outflow = pd.to_numeric(tx.get("in_uscita"), errors="coerce")
        flow = inflow.fillna(0.0) - outflow.fillna(0.0)
    daily = flow.groupby(tx["date"]).sum()
    return daily.reindex(_index, fill_value=0.0)


def _twr_returns(account_series_ex_pm: pd.Series, external_flows: pd.Series) -> pd.Series:
    prev = account_series_ex_pm.shift(1)
    flows = external_flows.reindex(account_series_ex_pm.index).fillna(0.0)
    twr = ((account_series_ex_pm - prev - flows) / prev).replace([np.inf, -np.inf], np.nan)
    twr = twr[(prev > 1e-6)].dropna()
    return twr


def render_performance(df_tx, df_trade_priced, df_prices_long, df_master) -> None:
    benchmark_isin = "IE000BI8OT95"
    st.markdown("<h2 style='margin:0 0 .25rem 0;'>Performance</h2>", unsafe_allow_html=True)
    st.markdown("")

    account_series, account_series_ex_pm, net_invested_series = _build_account_series(
        df_tx, df_trade_priced, df_prices_long
    )
    if account_series_ex_pm.empty:
        st.info("Insufficient data for performance analytics.")
        return

    external_flows = _external_flows_series(df_tx, account_series_ex_pm.index)
    twr = _twr_returns(account_series_ex_pm, external_flows)
    if twr.empty:
        st.info("Insufficient data for performance analytics.")
        return

    first_op = None
    if not df_tx.empty:
        tmp_dates = pd.to_datetime(df_tx["date"], errors="coerce")
        if "type" in df_tx.columns:
            bonifico_mask = df_tx["type"].astype(str).str.lower().eq("bonifico")
            if bonifico_mask.any():
                first_op = tmp_dates[bonifico_mask].min()
        if pd.isna(first_op):
            flow_series = None
            if "cash_flow" in df_tx.columns:
                flow_series = pd.to_numeric(df_tx["cash_flow"], errors="coerce")
            elif "cash_flow_model" in df_tx.columns:
                flow_series = pd.to_numeric(df_tx["cash_flow_model"], errors="coerce")
            if flow_series is not None:
                first_op = tmp_dates[(flow_series.fillna(0.0).abs() > 0)].min()
        if pd.isna(first_op):
            first_op = tmp_dates.min()
    if pd.notna(first_op):
        twr = twr[twr.index >= first_op.normalize()]
    equity_curve = (1.0 + twr).cumprod() * 100.0
    total_return = equity_curve.iloc[-1] / 100.0 - 1.0
    days = (equity_curve.index.max() - equity_curve.index.min()).days
    years = days / 365.25 if days > 0 else 0.0
    ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years >= 0.5 else np.nan
    vol = twr.std() * np.sqrt(252)
    sharpe = (twr.mean() * 252) / vol if vol and not np.isnan(vol) else np.nan

    st.markdown("**Performance snapshot**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total Return (TWR)",
        _pct(total_return),
        help="Chain-linked TWR: compounded daily flow-adjusted returns minus 1 (external flows = transfers).",
    )
    c2.metric(
        "Annualized Return",
        _pct(ann_return) if not np.isnan(ann_return) else "n/a",
        help="Annualized total TWR, shown only when the sample period is sufficiently long.",
    )
    c3.metric(
        "Volatility",
        _pct(vol),
        help="Standard deviation of daily TWR returns, annualized with sqrt(252).",
    )
    c4.metric(
        "Sharpe Ratio",
        f"{sharpe:.2f}" if not np.isnan(sharpe) else "n/a",
        help="Sharpe (rf=0): annualized average return divided by annualized volatility.",
    )

    # Secondary realized/unrealized info (non KPI)
    realized_total = pd.to_numeric(df_trade_priced.get("capital_gain_eur"), errors="coerce").sum() if not df_trade_priced.empty else 0.0
    # total_pnl computed later; use account_series_ex_pm vs net invested as proxy if needed
    unrealized_total = np.nan
    if not np.isnan(realized_total):
        unrealized_total = (account_series_ex_pm.iloc[-1] - net_invested_series.iloc[-1]) - realized_total
    info1, info2 = st.columns(2)
    info1.caption(f"Realized PnL: € {realized_total:,.2f}")
    if not np.isnan(unrealized_total):
        info2.caption(f"Unrealized PnL: € {unrealized_total:,.2f}")

    st.markdown("**TWR equity curve**")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode="lines",
            name="Portfolio (TWR, base 100)",
            line={"width": 2.8, "color": "#0F4C81"},
        )
    )

    price_wide = df_prices_long.pivot(index="date", columns="isin", values="price").sort_index()
    bench = price_wide.get(benchmark_isin)
    if bench is not None and bench.dropna().any():
        bench_ret = bench.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        if pd.notna(first_op):
            bench_ret = bench_ret[bench_ret.index >= first_op.normalize()]
        bench_curve = (1.0 + bench_ret).cumprod() * 100.0
        fig.add_trace(
            go.Scatter(
                x=bench_curve.index,
                y=bench_curve.values,
                mode="lines",
                name="MSCI World (base 100)",
                line={"width": 2, "color": "#6C757D"},
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        yaxis_title="Base 100",
        xaxis_title="Date",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Rolling performance**")
    roll_30 = (1.0 + twr).rolling(30).apply(np.prod, raw=True) - 1.0
    roll_90 = (1.0 + twr).rolling(90).apply(np.prod, raw=True) - 1.0
    roll_180 = (1.0 + twr).rolling(180).apply(np.prod, raw=True) - 1.0
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=roll_30.index, y=roll_30.values * 100.0, mode="lines", name="30d"))
    fig_roll.add_trace(go.Scatter(x=roll_90.index, y=roll_90.values * 100.0, mode="lines", name="90d"))
    fig_roll.add_trace(go.Scatter(x=roll_180.index, y=roll_180.values * 100.0, mode="lines", name="180d"))
    fig_roll.update_layout(
        template="plotly_white",
        height=360,
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        yaxis_title="Rolling return (%)",
        xaxis_title="Date",
    )
    st.plotly_chart(fig_roll, width="stretch")

    st.markdown("**Calendar returns (monthly)**")
    month_ret = (1.0 + twr).resample("M").apply(np.prod) - 1.0
    cal = month_ret.to_frame("ret")
    cal["Year"] = cal.index.year
    cal["Month"] = cal.index.month
    pivot = cal.pivot(index="Year", columns="Month", values="ret").sort_index()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = month_names
    ytd = (1.0 + month_ret).groupby(month_ret.index.year).apply(lambda s: s.prod() - 1.0)
    pivot["YTD"] = ytd
    styled = pivot.style.format("{:.2%}").background_gradient(cmap="RdYlGn", axis=None)
    st.dataframe(styled, width="stretch")

    st.markdown("**Performance attribution (light)**")
    # Show both views without toggles to avoid page reloads.
    mode = None
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

    net_invested = trades.groupby("isin", as_index=False)["cash_flow_model"].sum().rename(columns={"cash_flow_model": "net_invested_raw"})
    contrib = mv.merge(net_invested, on="isin", how="left")
    contrib["net_invested"] = -contrib["net_invested_raw"].fillna(0.0)
    contrib.loc[contrib["last_price"].isna() & (contrib["qty_net"].abs() < 1e-12), "market_value"] = 0.0
    contrib["market_value"] = contrib["market_value"].fillna(0.0)
    contrib["pnl"] = contrib["market_value"] - contrib["net_invested"]
    total_mv = contrib["market_value"].sum()
    contrib["weight_pct"] = contrib["market_value"] / total_mv if total_mv else 0.0

    realized_by_isin = (
        trades.groupby("isin", as_index=False)["capital_gain_eur"]
        .sum()
        .rename(columns={"capital_gain_eur": "pnl_realized"})
    )
    contrib = contrib.merge(realized_by_isin, on="isin", how="left")
    contrib["pnl_realized"] = contrib["pnl_realized"].fillna(0.0)
    contrib["pnl_unrealized"] = contrib["pnl"] - contrib["pnl_realized"]

    # Performance extremes per ISIN
    price_wide = df_prices_long.pivot(index="date", columns="isin", values="price").sort_index().ffill()
    last_date = price_wide.index.max() if not price_wide.empty else None
    if last_date is not None:
        ret_30d = price_wide / price_wide.shift(30) - 1.0
        ret_90d = price_wide / price_wide.shift(90) - 1.0
        ytd_start = pd.Timestamp(year=last_date.year, month=1, day=1)
        ytd_base = price_wide.loc[price_wide.index >= ytd_start].iloc[0]
        ret_ytd = price_wide.loc[last_date] / ytd_base - 1.0
        contrib = contrib.merge(ret_30d.loc[last_date].rename("ret_30d"), on="isin", how="left")
        contrib = contrib.merge(ret_90d.loc[last_date].rename("ret_90d"), on="isin", how="left")
        contrib = contrib.merge(ret_ytd.rename("ret_ytd"), on="isin", how="left")

    # A) PnL share
    st.markdown("**PnL share**")
    total_pnl = contrib["pnl"].sum()
    denom = total_pnl if abs(total_pnl) > 1e-8 else contrib["pnl"].abs().sum()
    contrib["pnl_share"] = contrib["pnl"] / denom if denom else 0.0

    # B) Return contribution (TWR-like)
    st.markdown("**Return contribution (TWR-like)**")
    returns_wide = price_wide.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    qty_delta = trades.groupby(["trade_date", "isin"], as_index=False)["qty_net"].sum()
    qty_pivot = qty_delta.pivot(index="trade_date", columns="isin", values="qty_net").reindex(price_wide.index).fillna(0.0)
    qty_cum = qty_pivot.cumsum().ffill().fillna(0.0)
    mv_ts = qty_cum * price_wide
    total_mv_ts = mv_ts.sum(axis=1).replace(0.0, np.nan)
    weight_ts = mv_ts.div(total_mv_ts, axis=0)
    weight_lag = weight_ts.shift(1)
    contrib_ts = weight_lag * returns_wide
    contrib_ts = contrib_ts.where(weight_lag >= 0.005)
    contrib_i = contrib_ts.sum().rename("return_contribution")
    contrib = contrib.merge(contrib_i, on="isin", how="left")
    contrib["return_contribution"] = contrib["return_contribution"].fillna(0.0)

    name_map = dict(zip(df_master["isin"], df_master["name"])) if df_master is not None and not df_master.empty else {}
    contrib["name"] = contrib["isin"].map(name_map).fillna(contrib["isin"])
    # Keep rows even if one of the contribution metrics is missing.
    top_pnl = contrib.sort_values("pnl_share", ascending=False).head(5)
    bottom_pnl = contrib.sort_values("pnl_share", ascending=True).head(5)
    top_ret = contrib.sort_values("return_contribution", ascending=False).head(5)
    bottom_ret = contrib.sort_values("return_contribution", ascending=True).head(5)

    def _fmt_pnl(df):
        cols = ["name", "pnl", "pnl_realized", "pnl_unrealized", "pnl_share", "weight_pct", "ret_30d", "ret_90d", "ret_ytd"]
        return df[cols].rename(
            columns={
                "name": "Name",
                "pnl": "PnL total €",
                "pnl_realized": "PnL realized €",
                "pnl_unrealized": "PnL unrealized €",
                "pnl_share": "PnL share",
                "weight_pct": "Weight %",
                "ret_30d": "Return 30d",
                "ret_90d": "Return 90d",
                "ret_ytd": "Return YTD",
            }
        )

    def _fmt_ret(df):
        cols = ["name", "pnl", "pnl_realized", "pnl_unrealized", "return_contribution", "weight_pct", "ret_30d", "ret_90d", "ret_ytd"]
        return df[cols].rename(
            columns={
                "name": "Name",
                "pnl": "PnL total €",
                "pnl_realized": "PnL realized €",
                "pnl_unrealized": "PnL unrealized €",
                "return_contribution": "Return contrib",
                "weight_pct": "Weight %",
                "ret_30d": "Return 30d",
                "ret_90d": "Return 90d",
                "ret_ytd": "Return YTD",
            }
        )

    c_left, c_right = st.columns(2)
    def _highlight_extremes(df, cols):
        def _style(val, col):
            if pd.isna(val):
                return ""
            s = df[col]
            if s.dropna().empty:
                return ""
            if val >= s.quantile(0.9):
                return "background-color: #e6f4ea"
            if val <= s.quantile(0.1):
                return "background-color: #fdecea"
            return ""
        return df.style.apply(
            lambda row: [ _style(row[c], c) if c in cols else "" for c in df.columns ],
            axis=1,
        )

    with c_left:
        st.markdown("**Top contributors - PnL share**")
        view = _fmt_pnl(top_pnl)
        styler = _highlight_extremes(view, ["Return 30d", "Return 90d", "Return YTD"]).format(
            {
                "PnL total €": "€ {:,.2f}",
                "PnL realized €": "€ {:,.2f}",
                "PnL unrealized €": "€ {:,.2f}",
                "PnL share": "{:.2%}",
                "Weight %": "{:.2%}",
                "Return 30d": "{:.2%}",
                "Return 90d": "{:.2%}",
                "Return YTD": "{:.2%}",
            }
        )
        st.dataframe(_table_style(styler), width="stretch", hide_index=True, height=260)
    with c_right:
        st.markdown("**Bottom contributors - PnL share**")
        view = _fmt_pnl(bottom_pnl)
        styler = _highlight_extremes(view, ["Return 30d", "Return 90d", "Return YTD"]).format(
            {
                "PnL total €": "€ {:,.2f}",
                "PnL realized €": "€ {:,.2f}",
                "PnL unrealized €": "€ {:,.2f}",
                "PnL share": "{:.2%}",
                "Weight %": "{:.2%}",
                "Return 30d": "{:.2%}",
                "Return 90d": "{:.2%}",
                "Return YTD": "{:.2%}",
            }
        )
        st.dataframe(_table_style(styler), width="stretch", hide_index=True, height=260)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Top contributors - Return**")
        view = _fmt_ret(top_ret)
        styler = _highlight_extremes(view, ["Return 30d", "Return 90d", "Return YTD"]).format(
            {
                "PnL total €": "€ {:,.2f}",
                "PnL realized €": "€ {:,.2f}",
                "PnL unrealized €": "€ {:,.2f}",
                "Return contrib": "{:.2%}",
                "Weight %": "{:.2%}",
                "Return 30d": "{:.2%}",
                "Return 90d": "{:.2%}",
                "Return YTD": "{:.2%}",
            }
        )
        st.dataframe(_table_style(styler), width="stretch", hide_index=True, height=260)
    with c_right:
        st.markdown("**Bottom contributors - Return**")
        view = _fmt_ret(bottom_ret)
        styler = _highlight_extremes(view, ["Return 30d", "Return 90d", "Return YTD"]).format(
            {
                "PnL total €": "€ {:,.2f}",
                "PnL realized €": "€ {:,.2f}",
                "PnL unrealized €": "€ {:,.2f}",
                "Return contrib": "{:.2%}",
                "Weight %": "{:.2%}",
                "Return 30d": "{:.2%}",
                "Return 90d": "{:.2%}",
                "Return YTD": "{:.2%}",
            }
        )
        st.dataframe(_table_style(styler), width="stretch", hide_index=True, height=260)

    st.markdown("**Positions to Watch**")
    if not price_wide.empty:
        ret_30d_series = price_wide / price_wide.shift(30) - 1.0
        roll_max_90 = price_wide.rolling(90).max()
        dist_from_max = price_wide / roll_max_90 - 1.0
        latest = price_wide.index.max()
        items = []
        for isin in price_wide.columns:
            r30 = ret_30d_series.loc[latest, isin] if isin in ret_30d_series.columns else np.nan
            hist = ret_30d_series[isin].dropna() if isin in ret_30d_series.columns else pd.Series(dtype=float)
            if hist.empty or pd.isna(r30):
                continue
            pct = (hist.rank(pct=True).iloc[-1]) if not hist.empty else np.nan
            z = (r30 - hist.mean()) / hist.std() if hist.std() and not np.isnan(hist.std()) else np.nan
            dmax = dist_from_max.loc[latest, isin] if isin in dist_from_max.columns else np.nan
            if (pct >= 0.9) or (pct <= 0.1) or (pd.notna(dmax) and dmax <= -0.15):
                name = contrib.loc[contrib["isin"] == isin, "name"].iloc[0] if "name" in contrib.columns and (contrib["isin"] == isin).any() else isin
                items.append(
                    {
                        "name": name,
                        "ret_30d": r30,
                        "pct": pct,
                        "z": z,
                        "dmax": dmax,
                    }
                )
        items = sorted(items, key=lambda x: abs(x["ret_30d"]), reverse=True)[:5]
        if items:
            cards = []
            for it in items:
                tags = []
                if it["pct"] >= 0.9:
                    tags.append("extended to the upside vs 30d history")
                if it["pct"] <= 0.1:
                    tags.append("extended to the downside vs 30d history")
                if pd.notna(it["dmax"]) and it["dmax"] <= -0.15:
                    tags.append("meaningfully below recent 90d high")
                summary = "; ".join(tags)
                cards.append({"Name": it["name"], "Return 30d": it["ret_30d"], "Comment": summary})
            df_cards = pd.DataFrame(cards)
            st.dataframe(
                _table_style(df_cards.style.format({"Return 30d": "{:.2%}"})),
                width="stretch",
                hide_index=True,
                height=240,
            )
        else:
            st.caption("No positions with relevant recent extremes.")
