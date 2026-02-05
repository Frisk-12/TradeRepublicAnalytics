#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Risk tab renderer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from script.analytics.core.overview import (
    _cash_balance_series,
    _missing_price_adjustment_series,
    _net_invested_series,
    _portfolio_value_series,
)


def _eur(v: float) -> str:
    return f"€ {v:,.2f}"


def _pct(v: float) -> str:
    return f"{v:.2%}"


def _short_name(s: str, max_len: int = 14) -> str:
    s = str(s or "")
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


@st.cache_data(show_spinner=False)
def _build_account_series(df_tx, df_trade_priced, df_prices_long):
    if df_tx.empty and df_prices_long.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    tx_dates = pd.to_datetime(df_tx["date"], errors="coerce") if not df_tx.empty else pd.Series(dtype="datetime64[ns]")
    px_dates = pd.to_datetime(df_prices_long["date"], errors="coerce") if not df_prices_long.empty else pd.Series(dtype="datetime64[ns]")
    candidates = [d for d in [tx_dates.min(), px_dates.min()] if pd.notna(d)]
    start = pd.to_datetime(min(candidates), errors="coerce") if candidates else pd.NaT
    end = pd.Timestamp.today().normalize()
    if pd.isna(start):
        return pd.Series(dtype=float), pd.Series(dtype=float)

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
def _risk_metrics(account_series_ex_pm: pd.Series, net_invested_series: pd.Series):
    if account_series_ex_pm.empty:
        return {}

    flows = net_invested_series.diff().reindex(account_series_ex_pm.index).fillna(0.0)
    prev = account_series_ex_pm.shift(1)
    returns = ((account_series_ex_pm - flows) / prev - 1.0).replace([np.inf, -np.inf], np.nan)
    returns = returns[(prev > 0)].dropna()
    vol_full = returns.std() * np.sqrt(252) if not returns.empty else 0.0
    vol_30 = returns.tail(30).std() * np.sqrt(252) if len(returns) >= 2 else 0.0
    vol_60 = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 2 else 0.0

    running_max = account_series_ex_pm.cummax()
    drawdown = account_series_ex_pm / running_max - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    current_dd = float(drawdown.iloc[-1]) if not drawdown.empty else 0.0

    worst_day = float(returns.min()) if not returns.empty else 0.0
    best_day = float(returns.max()) if not returns.empty else 0.0

    trough_date = drawdown.idxmin() if not drawdown.empty else None
    peak_date = None
    if trough_date is not None and trough_date in account_series_ex_pm.index:
        peak_date = account_series_ex_pm.loc[:trough_date].idxmax()

    last_peak_date = running_max.idxmax() if not running_max.empty else None
    regime = "Rising" if vol_30 > vol_60 else "Falling"

    return {
        "returns": returns,
        "drawdown": drawdown,
        "vol_30": vol_30,
        "vol_full": vol_full,
        "vol_60": vol_60,
        "max_dd": max_dd,
        "current_dd": current_dd,
        "worst_day": worst_day,
        "best_day": best_day,
        "trough_date": trough_date,
        "peak_date": peak_date,
        "last_peak_date": last_peak_date,
        "regime": regime,
    }


@st.cache_data(show_spinner=False)
def _risk_by_isin(df_trade_priced, df_prices_long, df_master, df_tx):
    if df_trade_priced.empty or df_prices_long.empty:
        return pd.DataFrame()

    trades = df_trade_priced.copy()
    trades["qty_net"] = trades["quantity"] * trades["side"].str.upper().map({"BUY": 1, "SELL": -1}).fillna(0)
    qty_net = trades.groupby("isin", as_index=False)["qty_net"].sum()

    last_price = (
        df_prices_long.sort_values("date")
        .groupby("isin", as_index=False)
        .last()[["isin", "price", "price_source"]]
        .rename(columns={"price": "last_price"})
    )
    mv = qty_net.merge(last_price, on="isin", how="left")
    mv["market_value"] = mv["qty_net"] * mv["last_price"]
    mv = mv[mv["market_value"].notna() & (mv["market_value"] > 0)]
    total_mv = mv["market_value"].sum()
    mv["weight_pct"] = mv["market_value"] / total_mv if total_mv else 0.0

    price_wide = df_prices_long.pivot(index="date", columns="isin", values="price").sort_index()
    returns = price_wide.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    vol_30 = returns.tail(30).std() * np.sqrt(252)
    worst_day = returns.min()

    running_max = price_wide.cummax()
    dd = price_wide / running_max - 1.0
    max_dd = dd.min()

    name_map = dict(zip(df_master["isin"], df_master["name"])) if df_master is not None and not df_master.empty else {}
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

    risk = mv.merge(vol_30.rename("vol_30d"), on="isin", how="left")
    risk = risk.merge(max_dd.rename("max_dd"), on="isin", how="left")
    risk = risk.merge(worst_day.rename("worst_day"), on="isin", how="left")
    # PnL % for color coding (same definition as Contributors)
    net_invested = trades.groupby("isin", as_index=False)["cash_flow_model"].sum().rename(columns={"cash_flow_model": "net_invested_raw"})
    pnl_base = qty_net.merge(net_invested, on="isin", how="left").merge(last_price, on="isin", how="left")
    pnl_base["net_invested"] = -pnl_base["net_invested_raw"].fillna(0.0)
    pnl_base["market_value"] = pnl_base["qty_net"] * pnl_base["last_price"]
    pnl_base["pnl"] = pnl_base["market_value"] - pnl_base["net_invested"]
    pnl_base["pnl_pct"] = pnl_base.apply(
        lambda r: (r["pnl"] / abs(r["net_invested"])) if r["net_invested"] not in (0, None) and abs(r["net_invested"]) > 1e-12 else np.nan,
        axis=1,
    )
    risk = risk.merge(pnl_base[["isin", "pnl_pct"]], on="isin", how="left")

    risk["name"] = risk["isin"].map(name_map_tx).fillna(risk["isin"].map(name_map)).fillna(risk["isin"])
    if df_master is not None and not df_master.empty and "asset_class" in df_master.columns:
        asset_map = dict(zip(df_master["isin"], df_master["asset_class"]))
        risk["asset_class"] = risk["isin"].map(asset_map).fillna("other")
    else:
        risk["asset_class"] = "other"
    risk["price_source"] = risk["price_source"].fillna("unknown")

    def _flag(r):
        v = r.get("vol_30d")
        d = r.get("max_dd")
        if pd.notna(d) and d < -0.15 or pd.notna(v) and v > 0.35:
            return "🔴"
        if pd.notna(d) and d < -0.08 or pd.notna(v) and v > 0.25:
            return "🟠"
        return "🟢"

    risk["risk_flag"] = risk.apply(_flag, axis=1)
    risk["risk_proxy"] = risk["weight_pct"] * risk["vol_30d"].fillna(0.0)
    return risk[
        [
            "name",
            "isin",
            "weight_pct",
            "vol_30d",
            "max_dd",
            "worst_day",
            "price_source",
            "risk_flag",
            "risk_proxy",
            "asset_class",
            "pnl_pct",
        ]
    ].sort_values("risk_proxy", ascending=False)


def _prepare_risk_inputs(df_tx, df_trade_priced, df_prices_long, df_master):
    account_series, account_series_ex_pm, net_invested_series = _build_account_series(
        df_tx, df_trade_priced, df_prices_long
    )
    metrics = _risk_metrics(account_series_ex_pm, net_invested_series)
    risk = _risk_by_isin(df_trade_priced, df_prices_long, df_master, df_tx)
    return account_series_ex_pm, metrics, risk


def render_risk(df_tx, df_trade_priced, df_prices_long, df_master) -> None:
    st.markdown("<h2 style='margin:0 0 .25rem 0;'>Risk</h2>", unsafe_allow_html=True)
    st.caption("Full period available")

    account_series_ex_pm, metrics, risk = _prepare_risk_inputs(df_tx, df_trade_priced, df_prices_long, df_master)
    if not metrics:
        st.info("Dati insufficienti per calcolare il rischio.")
        return

    print(
        {
            "vol_30": metrics["vol_30"],
            "vol_full": metrics["vol_full"],
            "max_dd": metrics["max_dd"],
            "current_dd": metrics["current_dd"],
            "worst_day": metrics["worst_day"],
            "best_day": metrics["best_day"],
        }
    )

    st.markdown("**Risk snapshot**")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Volatility 30d (ann.)",
        _pct(metrics["vol_30"]),
        help="Deviazione standard dei rendimenti giornalieri flow-adjusted sugli ultimi 30 giorni, annualizzata con √252.",
    )
    k2.metric(
        "Volatility full-period (ann.)",
        _pct(metrics["vol_full"]),
        help="Deviazione standard dei rendimenti giornalieri flow-adjusted su tutto il periodo, annualizzata con √252.",
    )
    k3.metric(
        "Current Drawdown",
        _pct(metrics["current_dd"]),
        help="Drawdown attuale: V_oggi / peak - 1.",
    )
    k4.metric(
        "Worst Day",
        _pct(metrics["worst_day"]),
        help="Peggior rendimento giornaliero flow-adjusted.",
    )
    k5.metric(
        "Best Day",
        _pct(metrics["best_day"]),
        help="Miglior rendimento giornaliero flow-adjusted.",
    )

    insight_left = "Peak-to-trough worst period: "
    if metrics.get("peak_date") is not None and metrics.get("trough_date") is not None:
        insight_left += f"{metrics['peak_date'].date().isoformat()} → {metrics['trough_date'].date().isoformat()}"
    else:
        insight_left += "n/a"
    insight_mid = "Worst day: "
    if metrics.get("returns") is not None and not metrics["returns"].empty:
        worst_date = metrics["returns"].idxmin()
        insight_mid += f"{worst_date.date().isoformat()} ({_pct(metrics['worst_day'])})"
    else:
        insight_mid += "n/a"
    insight_right = f"Most recent volatility regime: {metrics['regime']}"
    st.markdown(f"- {insight_left}\n- {insight_mid}\n- {insight_right}")

    st.markdown("**Equity & Drawdown**")
    drawdown = metrics["drawdown"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
    )
    fig.add_trace(
        go.Scatter(
            x=account_series_ex_pm.index,
            y=account_series_ex_pm.values,
            mode="lines",
            name="Account Value",
            line={"width": 2.8, "color": "#0F4C81"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100.0,
            mode="lines",
            name="Drawdown %",
            line={"width": 1.8, "color": "#D9534F"},
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )
    if metrics.get("trough_date") is not None:
        fig.add_trace(
            go.Scatter(
                x=[metrics["trough_date"]],
                y=[drawdown.loc[metrics["trough_date"]] * 100.0],
                mode="markers",
                name="Max DD",
                marker={"size": 8, "color": "#D9534F"},
            ),
            row=2,
            col=1,
        )
    if metrics.get("last_peak_date") is not None:
        fig.add_trace(
            go.Scatter(
                x=[metrics["last_peak_date"]],
                y=[account_series_ex_pm.loc[metrics["last_peak_date"]]],
                mode="markers",
                name="Last Peak",
                marker={"size": 7, "color": "#2E8B57"},
            ),
            row=1,
            col=1,
        )
    fig.update_layout(
        template="plotly_white",
        height=540,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(tickprefix="€ ", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Risk by ISIN**")
    if risk.empty:
        st.info("Dati insufficienti per la risk table.")
    else:
        q = st.text_input("Search (name/isin)", value="", key="risk_search")
        view = risk.copy()
        if q.strip():
            ql = q.strip().lower()
            view = view[view["name"].str.lower().str.contains(ql) | view["isin"].str.lower().str.contains(ql)]
        view = view.sort_values("risk_proxy", ascending=False)
        view = view.head(15)

        def _style_flag(val):
            if val == "🔴":
                return "background-color: #f8d7da"
            if val == "🟠":
                return "background-color: #fff3cd"
            return "background-color: #d1e7dd"

        display = view[
            ["name", "isin", "weight_pct", "vol_30d", "max_dd", "worst_day", "price_source", "risk_flag"]
        ].copy()
        display["weight_pct"] = display["weight_pct"].fillna(0.0)
        display["vol_30d"] = display["vol_30d"].fillna(0.0)
        display["max_dd"] = display["max_dd"].fillna(0.0)
        display["worst_day"] = display["worst_day"].fillna(0.0)
        st.dataframe(
            display.style.format(
                {
                    "weight_pct": "{:.2%}",
                    "vol_30d": "{:.2%}",
                    "max_dd": "{:.2%}",
                    "worst_day": "{:.2%}",
                }
            ).applymap(_style_flag, subset=["risk_flag"]),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Risk contributors**")
    if not risk.empty:
        st.caption("Proxy rischio = weight % × vol 30d (ann.).")
        rc = risk.copy()
        rc = rc[rc["risk_proxy"].notna()].sort_values("risk_proxy", ascending=False).head(10)
        fig_rc = go.Figure(
            go.Bar(
                x=rc["risk_proxy"],
                y=rc["name"],
                orientation="h",
                marker_color="#2C7FB8",
            )
        )
        fig_rc.update_layout(
            template="plotly_white",
            height=360,
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            xaxis_title="Risk proxy (weight × vol 30d)",
            yaxis_title="",
        )
        st.plotly_chart(fig_rc, use_container_width=True)

        total_risk = risk["risk_proxy"].sum()
        top1 = risk["risk_proxy"].head(1).sum() / total_risk if total_risk else 0.0
        top3 = risk["risk_proxy"].head(3).sum() / total_risk if total_risk else 0.0
        hhi = ((risk["risk_proxy"] / total_risk) ** 2).sum() if total_risk else 0.0

        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": ["Top 1 risk share", "Top 3 risk share", "HHI (risk)"],
                    "Value": [top1, top3, hhi],
                    "Note": [
                        "Quota del rischio totale attribuibile al singolo titolo più rilevante.",
                        "Quota del rischio totale attribuibile ai primi 3 titoli.",
                        "Somma dei quadrati delle quote di rischio (0–1). Più alto = più concentrato.",
                    ],
                }
            ).style.format({"Value": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Diversification & Concentration**")
    if not risk.empty:
        weights = risk["weight_pct"].fillna(0.0).sort_values(ascending=False)
        top1_w = weights.head(1).sum()
        top3_w = weights.head(3).sum()
        top5_w = weights.head(5).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Top 1 weight", _pct(top1_w))
        c2.metric("Top 3 weight", _pct(top3_w))
        c3.metric("Top 5 weight", _pct(top5_w))

        enp = 1.0 / (weights.pow(2).sum()) if weights.sum() > 0 else 0.0
        st.caption(f"Effective Number of Positions (ENP): {enp:.1f} — risk equivalent to ~{enp:.0f} equal positions.")

        scatter = risk.copy()
        scatter["risk_proxy"] = scatter["risk_proxy"].fillna(0.0)
        pnl_pct = scatter["pnl_pct"].fillna(0.0)
        max_abs = float(pnl_pct.abs().quantile(0.95)) if not pnl_pct.empty else 0.0
        if max_abs <= 0:
            max_abs = 0.2
        fig_scatter = go.Figure(
            data=go.Scatter(
                x=scatter["weight_pct"] * 100.0,
                y=scatter["risk_proxy"] * 100.0,
                mode="markers",
                marker=dict(
                    color=pnl_pct * 100.0,
                    colorscale="RdYlGn",
                    cmin=-max_abs * 100.0,
                    cmax=max_abs * 100.0,
                    colorbar=dict(title="PnL %"),
                    size=8,
                ),
                text=scatter["name"],
                hovertemplate="%{text}<br>Weight: %{x:.2f}%<br>Risk proxy: %{y:.2f}%<br>PnL: %{marker.color:.2f}%<extra></extra>",
            )
        )
        fig_scatter.update_layout(
            template="plotly_white",
            height=380,
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            xaxis_title="Weight %",
            yaxis_title="Risk proxy (weight × vol 30d)",
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)



def render_risk_advanced(df_tx, df_trade_priced, df_prices_long, df_master) -> None:
    benchmark_isin = "IE000BI8OT95"
    st.markdown("<h2 style='margin:0 0 .25rem 0;'>Risk Advanced</h2>", unsafe_allow_html=True)
    st.caption("Approfondimento quantitativo con metriche leggere e leggibili.")

    account_series_ex_pm, metrics, risk = _prepare_risk_inputs(df_tx, df_trade_priced, df_prices_long, df_master)
    if df_prices_long.empty or risk.empty or not metrics:
        st.info("Dati insufficienti per Risk Advanced.")
        return

    price_wide = df_prices_long.pivot(index="date", columns="isin", values="price").sort_index()
    returns_wide = price_wide.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    # Correlation & Diversification
    st.markdown("**Correlation & Diversification**")
    top_isins = risk.sort_values("weight_pct", ascending=False)["isin"].head(10).tolist()
    if not returns_wide.empty:
        window = st.select_slider("Correlation window", options=["1M", "3M", "6M"], value="3M")
        days = {"1M": 30, "3M": 90, "6M": 180}[window]
        returns_slice = returns_wide.tail(days)
    else:
        returns_slice = returns_wide
    top_returns = returns_slice[top_isins].dropna(how="all")
    corr_top = top_returns.corr()
    if not corr_top.empty:
        upper = corr_top.where(np.triu(np.ones(corr_top.shape), k=1).astype(bool))
        avg_corr_top = float(upper.stack().mean()) if not upper.stack().empty else 0.0
        # Full portfolio average correlation
        corr_all = returns_wide[risk["isin"].tolist()].corr()
        upper_all = corr_all.where(np.triu(np.ones(corr_all.shape), k=1).astype(bool))
        avg_corr_all = float(upper_all.stack().mean()) if not upper_all.stack().empty else 0.0

        k1, k2 = st.columns(2)
        k1.metric("Average correlation (Top 10)", _pct(avg_corr_top), help="Media della matrice di correlazione (solo triangolo superiore, esclusa diagonale).")
        k2.metric("Average correlation (All)", _pct(avg_corr_all), help="Media della matrice di correlazione per tutte le posizioni.")

        names = [risk.set_index("isin").loc[i, "name"] if i in risk["isin"].values else i for i in corr_top.columns]
        short = [_short_name(n, 12) for n in names]
        custom = np.array(
            [[f"{names[i]} × {names[j]}" for j in range(len(names))] for i in range(len(names))],
            dtype=object,
        )
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_top.values,
                x=short,
                y=short,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                showscale=True,
                hovertemplate="%{customdata}<br>Correlation: %{z:.2f}<extra></extra>",
                customdata=custom,
            )
        )
        fig_corr.update_layout(
            template="plotly_white",
            height=360,
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Market Risk (Beta)
    st.markdown("**Market Risk**")
    bench = returns_wide.get(benchmark_isin)
    if bench is None or bench.dropna().empty:
        st.warning(f"Benchmark prices not available for {benchmark_isin}. Rebuild datalake to include it.")
    else:
        port_ret = metrics["returns"]
        aligned = pd.concat([port_ret.rename("port"), bench.rename("bench")], axis=1).dropna()
        beta_port = aligned["port"].cov(aligned["bench"]) / aligned["bench"].var() if not aligned.empty else 0.0
        st.metric("Portfolio beta", f"{beta_port:.2f}", help="Beta = cov(ritorni portafoglio, benchmark) / var(benchmark).")

        betas = {}
        for isin in risk["isin"].tolist():
            r = returns_wide.get(isin)
            if r is None:
                continue
            tmp = pd.concat([r.rename("asset"), bench.rename("bench")], axis=1).dropna()
            if tmp.empty or tmp["bench"].var() == 0:
                continue
            betas[isin] = tmp["asset"].cov(tmp["bench"]) / tmp["bench"].var()
        risk_beta = risk.copy()
        risk_beta["beta"] = risk_beta["isin"].map(betas)
        scatter_beta = risk_beta.dropna(subset=["beta"])
        fig_beta = go.Figure(
            data=go.Scatter(
                x=scatter_beta["weight_pct"] * 100.0,
                y=scatter_beta["beta"],
                mode="markers",
                marker=dict(size=8, color="#2C7FB8"),
                text=scatter_beta["name"],
                hovertemplate="%{text}<br>Weight: %{x:.2f}%<br>Beta: %{y:.2f}<extra></extra>",
            )
        )
        fig_beta.update_layout(
            template="plotly_white",
            height=320,
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            xaxis_title="Weight %",
            yaxis_title="Beta vs MSCI World",
            showlegend=False,
        )
        st.plotly_chart(fig_beta, use_container_width=True)

        # Rolling correlation among portfolio positions
        st.markdown("**Rolling correlation (portfolio)**")
        st.markdown("Correlazione media mobile tra le posizioni principali (7/30/90 giorni).")

        def _rolling_avg_corr(returns_df: pd.DataFrame, window: int) -> pd.Series:
            vals = []
            idx = returns_df.index
            for i in range(len(idx)):
                if i + 1 < window:
                    vals.append(np.nan)
                    continue
                chunk = returns_df.iloc[i + 1 - window : i + 1].dropna(axis=1, how="any")
                if chunk.shape[1] < 2:
                    vals.append(np.nan)
                    continue
                corr = np.corrcoef(chunk.values, rowvar=False)
                upper = corr[np.triu_indices_from(corr, k=1)]
                vals.append(np.nanmean(upper) if upper.size else np.nan)
            return pd.Series(vals, index=idx)

        base_returns = returns_wide[risk["isin"].tolist()]
        corr_7 = _rolling_avg_corr(base_returns, 7)
        corr_30 = _rolling_avg_corr(base_returns, 30)
        corr_90 = _rolling_avg_corr(base_returns, 90)
        # show last 3 months
        last_end = corr_90.dropna().index.max() if corr_90.notna().any() else base_returns.index.max()
        if pd.notna(last_end):
            start_3m = last_end - pd.Timedelta(days=90)
            corr_7 = corr_7.loc[corr_7.index >= start_3m]
            corr_30 = corr_30.loc[corr_30.index >= start_3m]
            corr_90 = corr_90.loc[corr_90.index >= start_3m]
        fig_corr_roll = go.Figure()
        fig_corr_roll.add_trace(go.Scatter(x=corr_7.index, y=corr_7.values, mode="lines", name="7d"))
        fig_corr_roll.add_trace(go.Scatter(x=corr_30.index, y=corr_30.values, mode="lines", name="30d"))
        fig_corr_roll.add_trace(go.Scatter(x=corr_90.index, y=corr_90.values, mode="lines", name="90d"))
        fig_corr_roll.update_layout(
            template="plotly_white",
            height=300,
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            yaxis_title="Average correlation",
            xaxis_title="Date",
        )
        st.plotly_chart(fig_corr_roll, use_container_width=True)

    # Tail Risk
    st.markdown("**Tail Risk**")
    port_ret = metrics["returns"]
    if port_ret.empty:
        st.info("Dati insufficienti per Tail Risk.")
        return
    var_95 = float(port_ret.quantile(0.05))
    var_99 = float(port_ret.quantile(0.01))
    es_95 = float(port_ret[port_ret <= var_95].mean()) if (port_ret <= var_95).any() else 0.0
    es_99 = float(port_ret[port_ret <= var_99].mean()) if (port_ret <= var_99).any() else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loss threshold 95%", _pct(var_95), help="VaR 95% = perdita che non viene superata nel 95% dei giorni (quantile 5%).")
    c2.metric("Loss threshold 99%", _pct(var_99), help="VaR 99% = perdita che non viene superata nel 99% dei giorni (quantile 1%).")
    c3.metric("Avg loss beyond 95%", _pct(es_95), help="ES 95% = perdita media nei giorni peggiori oltre il VaR 95%.")
    c4.metric("Avg loss beyond 99%", _pct(es_99), help="ES 99% = perdita media nei giorni peggiori oltre il VaR 99%.")

    hist = go.Figure(
        data=go.Histogram(
            x=port_ret * 100.0,
            nbinsx=40,
            marker_color="#6C757D",
        )
    )
    hist.add_vline(x=var_95 * 100.0, line_color="#D9534F", line_dash="dash")
    hist.add_vline(x=var_99 * 100.0, line_color="#B52B27", line_dash="dash")
    hist.add_annotation(x=var_95 * 100.0, y=1.0, yref="paper", text="VaR 95%", showarrow=False, xanchor="left")
    hist.add_annotation(x=var_99 * 100.0, y=1.0, yref="paper", text="VaR 99%", showarrow=False, xanchor="left")
    hist.update_layout(
        template="plotly_white",
        height=320,
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        xaxis_title="Daily return (%)",
        yaxis_title="Frequency",
    )
    st.plotly_chart(hist, use_container_width=True)
