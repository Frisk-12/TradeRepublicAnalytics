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


def _pct_or_na(v: float) -> str:
    return _pct(float(v)) if pd.notna(v) else "n/a"


def _pp_delta(series: pd.Series, lag: int) -> str | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2 or lag < 1 or len(s) <= lag:
        return None
    prev = s.iloc[-1 - lag]
    return f"{(s.iloc[-1] - prev) * 100.0:+.2f} pp"


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
    vol_30_prev = returns.iloc[:-1].tail(30).std() * np.sqrt(252) if len(returns) >= 31 else np.nan

    running_max = account_series_ex_pm.cummax()
    drawdown = account_series_ex_pm / running_max - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    current_dd = float(drawdown.iloc[-1]) if not drawdown.empty else 0.0
    current_dd_prev = float(drawdown.iloc[-2]) if len(drawdown) >= 2 else np.nan

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
        "vol_30_prev": vol_30_prev,
        "max_dd": max_dd,
        "current_dd": current_dd,
        "current_dd_prev": current_dd_prev,
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
        st.info("Insufficient data to compute risk analytics.")
        return

    st.markdown("**Risk snapshot**")
    k1, k2, k3, k4, k5 = st.columns(5)
    vol_delta = None
    if pd.notna(metrics.get("vol_30_prev")):
        vol_delta = f"{(metrics['vol_30'] - metrics['vol_30_prev']) * 100.0:+.2f} pp"
    k1.metric(
        "Volatility 30d (ann.)",
        _pct(metrics["vol_30"]),
        delta=vol_delta,
        help=(
            "Standard deviation of daily flow-adjusted returns over the last 30 days, annualized with sqrt(252). "
            "Delta is in percentage points (pp) versus previous day estimate."
        ),
    )
    k2.metric(
        "Volatility full-period (ann.)",
        _pct(metrics["vol_full"]),
        help="Standard deviation of daily flow-adjusted returns over the full period, annualized with sqrt(252).",
    )
    k3.metric(
        "Current Drawdown",
        _pct(metrics["current_dd"]),
        delta=(f"{(metrics['current_dd'] - metrics['current_dd_prev']) * 100.0:+.2f} pp" if pd.notna(metrics.get("current_dd_prev")) else None),
        help="Current drawdown: current value / peak value - 1. Delta is in percentage points (pp) vs previous day.",
    )
    k4.metric(
        "Worst Day",
        _pct(metrics["worst_day"]),
        help="Worst flow-adjusted daily return.",
    )
    k5.metric(
        "Best Day",
        _pct(metrics["best_day"]),
        help="Best flow-adjusted daily return.",
    )
    st.caption("Delta notation: `pp` means percentage points.")

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
        st.info("Insufficient data for the risk table.")
    else:
        st.caption(
            "Risk Flag heuristic: 🔴 if `max_dd < -15%` or `vol_30d > 35%`; "
            "🟠 if `max_dd < -8%` or `vol_30d > 25%`; otherwise 🟢."
        )
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
        display = display.rename(
            columns={
                "name": "Name",
                "isin": "ISIN",
                "weight_pct": "Weight",
                "vol_30d": "Volatility 30d (ann.)",
                "max_dd": "Max Drawdown",
                "worst_day": "Worst Day",
                "price_source": "Price Source",
                "risk_flag": "Risk Flag",
            }
        )
        st.dataframe(
            display.style.format(
                {
                    "Weight": "{:.2%}",
                    "Volatility 30d (ann.)": "{:.2%}",
                    "Max Drawdown": "{:.2%}",
                    "Worst Day": "{:.2%}",
                }
            ).applymap(_style_flag, subset=["Risk Flag"]),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Risk contributors**")
    if not risk.empty:
        st.caption("Risk proxy = weight % * 30d volatility (annualized).")
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
                        "Share of total risk attributable to the most relevant single position.",
                        "Share of total risk attributable to the top 3 positions.",
                        "Sum of squared risk shares (0-1). Higher means more concentration.",
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
        st.caption(f"Effective Number of Positions (ENP): {enp:.1f} - risk equivalent to about {enp:.0f} equal positions.")

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
    st.markdown("<h2 style='margin:0 0 .25rem 0;'>Risk Advanced</h2>", unsafe_allow_html=True)
    st.caption("Quantitative deep dive with lightweight and readable metrics.")

    account_series_ex_pm, metrics, risk = _prepare_risk_inputs(df_tx, df_trade_priced, df_prices_long, df_master)
    if df_prices_long.empty or risk.empty or not metrics:
        st.info("Insufficient data for Risk Advanced.")
        return

    price_wide = df_prices_long.pivot(index="date", columns="isin", values="price").sort_index()
    returns_wide = price_wide.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    tracked_isins = [isin for isin in risk["isin"].tolist() if isin in returns_wide.columns]
    top_isins = [isin for isin in risk.sort_values("weight_pct", ascending=False)["isin"].head(10).tolist() if isin in returns_wide.columns]
    windows = {"Daily": 2, "Weekly": 7, "Monthly": 30}

    # Volatility regime
    st.markdown("**Volatility regime (portfolio)**")
    port_ret = metrics["returns"].dropna()
    if port_ret.empty:
        st.info("Insufficient data to compute volatility regime.")
    else:
        st.caption(
            "Annualized realized volatility on flow-adjusted returns. "
            "Daily uses a 2-day minimum window to avoid 1-point instability."
        )
        vol_series = {
            label: port_ret.rolling(win).std() * np.sqrt(252) for label, win in windows.items()
        }
        v1, v2, v3 = st.columns(3)
        for col, (label, series) in zip([v1, v2, v3], vol_series.items()):
            last = series.dropna().iloc[-1] if series.notna().any() else np.nan
            col.metric(f"{label} vol (ann.)", _pct_or_na(last), delta=_pp_delta(series, windows[label]))
        st.caption("Delta notation: `pp` means percentage points.")

        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=vol_series["Daily"].index,
                y=vol_series["Daily"].values * 100.0,
                mode="lines",
                name="Daily",
                line={"width": 1.7, "color": "#5A7184"},
            )
        )
        fig_vol.add_trace(
            go.Scatter(
                x=vol_series["Weekly"].index,
                y=vol_series["Weekly"].values * 100.0,
                mode="lines",
                name="Weekly",
                line={"width": 2.0, "color": "#2C7FB8"},
            )
        )
        fig_vol.add_trace(
            go.Scatter(
                x=vol_series["Monthly"].index,
                y=vol_series["Monthly"].values * 100.0,
                mode="lines",
                name="Monthly",
                line={"width": 2.2, "color": "#D9534F"},
            )
        )
        fig_vol.update_layout(
            template="plotly_white",
            height=320,
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            yaxis_title="Volatility (%)",
            xaxis_title="Date",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # Correlation & Diversification
    st.markdown("**Correlation & Diversification**")
    if len(top_isins) < 2:
        st.info("At least two priced open positions are required for correlation analysis.")
    else:
        def _avg_corr(df: pd.DataFrame) -> float:
            corr = df.corr()
            if corr.empty:
                return np.nan
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            stacked = upper.stack()
            return float(stacked.mean()) if not stacked.empty else np.nan

        c_daily = returns_wide[top_isins].tail(windows["Daily"]).dropna(axis=1, how="any")
        c_weekly = returns_wide[top_isins].tail(windows["Weekly"]).dropna(axis=1, how="any")
        c_monthly = returns_wide[top_isins].tail(windows["Monthly"]).dropna(axis=1, how="any")
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg corr (Daily)", _pct_or_na(_avg_corr(c_daily) if c_daily.shape[1] >= 2 else np.nan))
        m2.metric("Avg corr (Weekly)", _pct_or_na(_avg_corr(c_weekly) if c_weekly.shape[1] >= 2 else np.nan))
        m3.metric("Avg corr (Monthly)", _pct_or_na(_avg_corr(c_monthly) if c_monthly.shape[1] >= 2 else np.nan))

        corr_top = c_monthly.corr() if c_monthly.shape[1] >= 2 else pd.DataFrame()
        if not corr_top.empty:
            name_map = risk.set_index("isin")["name"].to_dict()
            names = [name_map.get(i, i) for i in corr_top.columns]
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
        else:
            st.info("Monthly correlation heatmap requires at least two assets with complete recent data.")

        st.markdown("**Rolling correlation (portfolio)**")
        st.caption("Rolling average pairwise correlation across active positions (Daily/Weekly/Monthly windows).")

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

        base_returns = returns_wide[tracked_isins]
        corr_daily = _rolling_avg_corr(base_returns, windows["Daily"])
        corr_weekly = _rolling_avg_corr(base_returns, windows["Weekly"])
        corr_monthly = _rolling_avg_corr(base_returns, windows["Monthly"])
        last_end = base_returns.index.max()
        if pd.notna(last_end):
            start_window = last_end - pd.Timedelta(days=180)
            corr_daily = corr_daily.loc[corr_daily.index >= start_window]
            corr_weekly = corr_weekly.loc[corr_weekly.index >= start_window]
            corr_monthly = corr_monthly.loc[corr_monthly.index >= start_window]
        fig_corr_roll = go.Figure()
        fig_corr_roll.add_trace(go.Scatter(x=corr_daily.index, y=corr_daily.values, mode="lines", name="Daily"))
        fig_corr_roll.add_trace(go.Scatter(x=corr_weekly.index, y=corr_weekly.values, mode="lines", name="Weekly"))
        fig_corr_roll.add_trace(go.Scatter(x=corr_monthly.index, y=corr_monthly.values, mode="lines", name="Monthly"))
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
        st.info("Insufficient data for Tail Risk.")
        return
    var_95 = float(port_ret.quantile(0.05))
    var_99 = float(port_ret.quantile(0.01))
    es_95 = float(port_ret[port_ret <= var_95].mean()) if (port_ret <= var_95).any() else 0.0
    es_99 = float(port_ret[port_ret <= var_99].mean()) if (port_ret <= var_99).any() else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loss threshold 95%", _pct(var_95), help="VaR 95% = loss level not exceeded in 95% of days (5% quantile).")
    c2.metric("Loss threshold 99%", _pct(var_99), help="VaR 99% = loss level not exceeded in 99% of days (1% quantile).")
    c3.metric("Avg loss beyond 95%", _pct(es_95), help="ES 95% = average loss in days beyond VaR 95%.")
    c4.metric("Avg loss beyond 99%", _pct(es_99), help="ES 99% = average loss in days beyond VaR 99%.")

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
