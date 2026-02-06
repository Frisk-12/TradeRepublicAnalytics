#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-quality PDF report generator for the dashboard."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Tuple

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from script.analytics.core.overview import (
    _cash_balance_series,
    _missing_price_adjustment_series,
    _net_invested_series,
    _portfolio_value_series,
    build_overview_result,
)


def _eur(v: float) -> str:
    return f"EUR {v:,.2f}"


def _pct(v: float) -> str:
    return f"{v:.2%}"


def _pp(v: float) -> str:
    return f"{v * 100.0:+.2f} pp"


def _series_delta(series: pd.Series, lag: int) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2 or lag < 1 or len(s) <= lag:
        return None
    prev = s.iloc[-1 - lag]
    return float(s.iloc[-1] - prev)


def _fmt_delta_eur(v: float | None, suffix: str) -> str | None:
    if v is None or pd.isna(v):
        return None
    sign = "+" if v >= 0 else "-"
    return f"{sign}EUR {abs(v):,.2f} {suffix}"


def _fmt_delta_pp(v: float | None, suffix: str) -> str | None:
    if v is None or pd.isna(v):
        return None
    return f"{_pp(v)} {suffix}"


def _fmt_eur_axis(v, _pos) -> str:
    if abs(v) >= 1_000_000:
        return f"EUR {v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"EUR {v / 1_000:.0f}k"
    return f"EUR {v:.0f}"


def _short_label(v: object, max_len: int = 30) -> str:
    s = str(v or "")
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _apply_date_axis(ax, month_interval: int = 2) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def _add_footer(fig: plt.Figure, page: int, total: int = 5) -> None:
    fig.text(
        0.5,
        0.012,
        f"Portfolio Analytics Report  |  Page {page}/{total}",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#5d6f82",
    )


def _monthly_return_table(twr: pd.Series) -> pd.DataFrame:
    if twr.empty:
        return pd.DataFrame()
    month_ret = (1.0 + twr).resample("ME").apply("prod") - 1.0
    if month_ret.empty:
        return pd.DataFrame()
    cal = month_ret.to_frame("ret")
    cal["Year"] = cal.index.year
    cal["Month"] = cal.index.month
    pivot = cal.pivot(index="Year", columns="Month", values="ret").sort_index()
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ytd = (1.0 + month_ret).groupby(month_ret.index.year).apply(lambda s: s.prod() - 1.0)
    pivot["YTD"] = ytd
    return pivot


def _build_contributors(df_trade_priced, df_prices_long, df_master) -> pd.DataFrame:
    if df_trade_priced.empty or df_prices_long.empty:
        return pd.DataFrame()

    trades = df_trade_priced.copy()
    trades["qty_net"] = trades["quantity"] * trades["side"].astype(str).str.upper().map({"BUY": 1, "SELL": -1}).fillna(0)
    qty_net = trades.groupby("isin", as_index=False)["qty_net"].sum()

    last_price = (
        df_prices_long.sort_values("date")
        .groupby("isin", as_index=False)
        .last()[["isin", "price"]]
        .rename(columns={"price": "last_price"})
    )
    net_invested = trades.groupby("isin", as_index=False)["cash_flow_model"].sum().rename(columns={"cash_flow_model": "net_invested_raw"})

    contrib = qty_net.merge(last_price, on="isin", how="left").merge(net_invested, on="isin", how="left")
    contrib["net_invested"] = -pd.to_numeric(contrib["net_invested_raw"], errors="coerce").fillna(0.0)
    contrib["last_price"] = pd.to_numeric(contrib["last_price"], errors="coerce")
    contrib["market_value"] = contrib["qty_net"] * contrib["last_price"]
    contrib.loc[contrib["last_price"].isna() & (contrib["qty_net"].abs() < 1e-12), "market_value"] = 0.0
    contrib["market_value"] = contrib["market_value"].fillna(0.0)
    contrib["pnl"] = contrib["market_value"] - contrib["net_invested"]
    contrib["pnl_pct"] = np.where(
        contrib["net_invested"].abs() > 1e-12,
        contrib["pnl"] / contrib["net_invested"].abs(),
        0.0,
    )
    total_mv = float(contrib["market_value"].sum())
    contrib["weight_pct"] = contrib["market_value"] / total_mv if total_mv else 0.0

    name_map = dict(zip(df_master["isin"], df_master["name"])) if df_master is not None and not df_master.empty else {}
    contrib["name"] = contrib["isin"].map(name_map).fillna(contrib["isin"])
    return contrib.sort_values("market_value", ascending=False)


def _build_asset_allocation(contrib: pd.DataFrame, df_master: pd.DataFrame) -> pd.DataFrame:
    if contrib.empty:
        return pd.DataFrame(columns=["asset_class", "market_value"])
    if df_master is None or df_master.empty:
        out = contrib.copy()
        out = out[out["market_value"] > 0].copy()
        out["asset_class"] = "other"
        return out.groupby("asset_class", as_index=False)["market_value"].sum().sort_values("market_value", ascending=False)

    alloc = contrib.merge(df_master[["isin", "asset_class"]], on="isin", how="left")
    alloc["asset_class"] = alloc["asset_class"].fillna("other").astype(str).str.lower()
    alloc = alloc[alloc["market_value"] > 0].copy()
    return alloc.groupby("asset_class", as_index=False)["market_value"].sum().sort_values("market_value", ascending=False)


def _build_account_series(df_tx, df_trade_priced, df_prices_long) -> Tuple[pd.Series, pd.Series]:
    if df_tx.empty and df_prices_long.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    tx_dates = pd.to_datetime(df_tx["date"], errors="coerce") if not df_tx.empty else pd.Series(dtype="datetime64[ns]")
    px_dates = pd.to_datetime(df_prices_long["date"], errors="coerce") if not df_prices_long.empty else pd.Series(dtype="datetime64[ns]")
    candidates = [d for d in [tx_dates.min(), px_dates.min()] if pd.notna(d)]
    if not candidates:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    start = pd.to_datetime(min(candidates), errors="coerce")
    end = pd.Timestamp.today().normalize()
    if pd.isna(start):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    start = start.normalize()

    # Align to first transfer date when available to avoid pre-funding distortions.
    if not df_tx.empty and "type" in df_tx.columns:
        tx_tmp = df_tx.copy()
        tx_tmp["date"] = pd.to_datetime(tx_tmp["date"], errors="coerce").dt.normalize()
        first_transfer = tx_tmp.loc[tx_tmp["type"].astype(str).str.lower().eq("bonifico"), "date"].min()
        if pd.notna(first_transfer) and first_transfer > start:
            start = pd.Timestamp(first_transfer).normalize()

    portfolio_series = _portfolio_value_series(df_trade_priced, df_prices_long, df_tx, start, end)
    cash_series = _cash_balance_series(df_tx, start, end, anchor_date=start, anchor_saldo=0.0)
    cash_series = cash_series.add(_missing_price_adjustment_series(df_trade_priced, df_prices_long, start, end), fill_value=0.0)
    net_invested = _net_invested_series(df_tx, start, end)

    common_idx = portfolio_series.index.union(cash_series.index).union(net_invested.index)
    portfolio_series = portfolio_series.reindex(common_idx).ffill().fillna(0.0)
    cash_series = cash_series.reindex(common_idx).ffill().fillna(0.0)
    net_invested = net_invested.reindex(common_idx).ffill().fillna(0.0)
    account_series = cash_series + portfolio_series

    pm_series = pd.Series(0.0, index=account_series.index)
    pm_tx = df_tx.copy()
    if not pm_tx.empty:
        pm_tx["date"] = pd.to_datetime(pm_tx["date"], errors="coerce").dt.normalize()
        pm_tx["description"] = pm_tx["description"].astype(str).str.lower()
        pm_tx = pm_tx[
            (pm_tx["type"] == "Commercio")
            & (pm_tx["isin"].isna())
            & pm_tx["description"].str.contains("private market")
        ]
        if not pm_tx.empty:
            pm_tx["cash_flow"] = pd.to_numeric(pm_tx["cash_flow"], errors="coerce").fillna(0.0)
            daily_pm = pm_tx.groupby("date", as_index=True)["cash_flow"].sum().reindex(account_series.index, fill_value=0.0)
            pm_series = (-daily_pm).cumsum()
    return account_series - pm_series, net_invested


def _external_flows_series(df_tx: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    if df_tx.empty or "type" not in df_tx.columns:
        return pd.Series(0.0, index=idx)
    tx = df_tx.copy()
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce").dt.normalize()
    tx = tx[tx["type"].astype(str).str.lower().eq("bonifico")]
    if tx.empty:
        return pd.Series(0.0, index=idx)
    flow = pd.to_numeric(tx.get("cash_flow"), errors="coerce")
    daily = flow.groupby(tx["date"]).sum()
    return daily.reindex(idx, fill_value=0.0)


def _build_twr(account_series: pd.Series, df_tx: pd.DataFrame) -> pd.Series:
    if account_series.empty:
        return pd.Series(dtype=float)
    prev = account_series.shift(1)
    flows = _external_flows_series(df_tx, account_series.index)
    twr = ((account_series - prev - flows) / prev).replace([np.inf, -np.inf], np.nan)
    return twr[(prev > 1e-6)].dropna()


def _max_drawdown(account_series: pd.Series) -> float:
    if account_series.empty:
        return 0.0
    running_max = account_series.cummax()
    drawdown = account_series / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _risk_table(contrib: pd.DataFrame, df_prices_long: pd.DataFrame) -> pd.DataFrame:
    if contrib.empty or df_prices_long.empty:
        return pd.DataFrame(columns=["name", "weight_pct", "vol_30d", "risk_proxy", "pnl_pct"])
    price_wide = df_prices_long.pivot(index="date", columns="isin", values="price").sort_index()
    returns = price_wide.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    vol_30 = (returns.tail(30).std() * np.sqrt(252)).rename("vol_30d")
    risk = contrib.merge(vol_30, on="isin", how="left")
    risk["vol_30d"] = risk["vol_30d"].fillna(0.0)
    risk["risk_proxy"] = risk["weight_pct"].fillna(0.0) * risk["vol_30d"]
    return risk[["name", "weight_pct", "vol_30d", "risk_proxy", "pnl_pct"]].sort_values("risk_proxy", ascending=False)


def _kpi_box(ax, x: float, y: float, w: float, h: float, title: str, value: str, delta: str | None = None) -> None:
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="#ffffff", edgecolor="#d8e5f1", linewidth=1.0))
    ax.text(x + 0.014, y + h - 0.032, title, transform=ax.transAxes, fontsize=9.2, color="#4a5f73", va="top")
    ax.text(x + 0.014, y + 0.056, value, transform=ax.transAxes, fontsize=12.5, color="#0f3f64", fontweight="bold", va="bottom")
    if delta:
        d = delta.strip()
        d_color = "#2e8b57" if d.startswith("+") else "#c64b47" if d.startswith("-") else "#5b6f82"
        ax.text(x + 0.014, y + 0.022, d, transform=ax.transAxes, fontsize=8.2, color=d_color, va="bottom")


def generate_portfolio_pdf_report(df_tx, df_trade_priced, df_prices_long, df_master) -> bytes:
    """Generate an institutional-style PDF report and return bytes."""

    overview = build_overview_result(
        df_tx=df_tx,
        df_trade_priced=df_trade_priced,
        df_prices_long=df_prices_long,
        start_date=None,
        end_date=None,
    )
    contrib = _build_contributors(df_trade_priced, df_prices_long, df_master)
    alloc = _build_asset_allocation(contrib, df_master)
    account_series, net_invested_series = _build_account_series(df_tx, df_trade_priced, df_prices_long)
    twr = _build_twr(account_series, df_tx)
    monthly = _monthly_return_table(twr)
    risk_table = _risk_table(contrib, df_prices_long).head(12)

    total_return = float((1.0 + twr).prod() - 1.0) if not twr.empty else 0.0
    total_days = int((twr.index.max() - twr.index.min()).days) if not twr.empty else 0
    years = total_days / 365.25 if total_days > 0 else 0.0
    ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years >= 0.5 else np.nan
    ann_vol = float(twr.std() * np.sqrt(252)) if not twr.empty else 0.0
    sharpe = (float(twr.mean() * 252) / ann_vol) if ann_vol > 0 and not twr.empty else np.nan
    max_dd = _max_drawdown(account_series)
    var_95 = float(twr.quantile(0.05)) if not twr.empty else 0.0
    var_99 = float(twr.quantile(0.01)) if not twr.empty else 0.0
    es_95 = float(twr[twr <= var_95].mean()) if not twr.empty and (twr <= var_95).any() else 0.0
    es_99 = float(twr[twr <= var_99].mean()) if not twr.empty and (twr <= var_99).any() else 0.0

    source_counts = (
        df_prices_long["price_source"].fillna("unknown").value_counts().rename_axis("source").reset_index(name="rows")
        if "price_source" in df_prices_long.columns and not df_prices_long.empty
        else pd.DataFrame(columns=["source", "rows"])
    )
    quality = pd.DataFrame(
        {
            "Metric": [
                "Transactions rows",
                "Priced trades rows",
                "Price rows",
                "Unique ISIN in master",
                "Unique ISIN in prices",
                "Open positions",
            ],
            "Value": [
                int(len(df_tx)),
                int(len(df_trade_priced)),
                int(len(df_prices_long)),
                int(df_master["isin"].nunique()) if df_master is not None and not df_master.empty else 0,
                int(df_prices_long["isin"].nunique()) if not df_prices_long.empty else 0,
                int((contrib["weight_pct"] > 0).sum()) if not contrib.empty else 0,
            ],
        }
    )

    chart = overview.chart_df.sort_values("date")
    pnl_series = chart["account_value_ex_pm"] - chart["net_invested"] if not chart.empty else pd.Series(dtype=float)
    drawdown = pd.Series(dtype=float)
    if not account_series.empty:
        running_max = account_series.cummax().replace(0.0, np.nan)
        drawdown = (account_series / running_max - 1.0).fillna(0.0)
    max_dd_series = drawdown.cummin() if not drawdown.empty else pd.Series(dtype=float)

    delta_port_1d = _series_delta(chart["portfolio_value"], 1) if "portfolio_value" in chart.columns else None
    delta_cash_1d = _series_delta(chart["cash_balance"], 1) if "cash_balance" in chart.columns else None
    delta_net_30d = _series_delta(chart["net_invested"], 30) if "net_invested" in chart.columns else None
    delta_pnl_1d = _series_delta(pnl_series, 1) if not pnl_series.empty else None

    cum_twr = ((1.0 + twr).cumprod() - 1.0) if not twr.empty else pd.Series(dtype=float)
    delta_total_ret_1d = _series_delta(cum_twr, 1) if not cum_twr.empty else None
    vol_30_series = twr.rolling(30).std() * np.sqrt(252) if not twr.empty else pd.Series(dtype=float)
    delta_vol_1d = _series_delta(vol_30_series, 1) if not vol_30_series.empty else None
    delta_max_dd_1d = _series_delta(max_dd_series, 1) if not max_dd_series.empty else None

    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        # Page 1: Executive summary.
        fig1 = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        ax1 = fig1.add_axes([0, 0, 1, 1])
        ax1.axis("off")
        ax1.add_patch(Rectangle((0, 0.82), 1, 0.18, transform=ax1.transAxes, facecolor="#0f3f64", edgecolor="#0f3f64"))
        ax1.text(0.04, 0.92, "Portfolio Analytics Report", transform=ax1.transAxes, color="white", fontsize=24, fontweight="bold", va="center")
        ax1.text(0.04, 0.865, "Executive Summary", transform=ax1.transAxes, color="#d8e6f2", fontsize=12)
        ax1.text(0.04, 0.79, f"Period: {overview.start_date.date().isoformat()} to {overview.end_date.date().isoformat()}", transform=ax1.transAxes, fontsize=10.8, color="#2f4559")
        ax1.text(0.04, 0.758, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", transform=ax1.transAxes, fontsize=10.0, color="#4d6073")

        cards = [
            ("Portfolio Value", _eur(float(overview.portfolio_value)), _fmt_delta_eur(delta_port_1d, "vs prev day")),
            ("Cash Balance", _eur(float(overview.cash_balance)), _fmt_delta_eur(delta_cash_1d, "vs prev day")),
            ("Net Cash In", _eur(float(overview.net_cash_in)), _fmt_delta_eur(delta_net_30d, "last 30d")),
            ("PnL", _eur(float(overview.pnl_abs)), _fmt_delta_eur(delta_pnl_1d, "vs prev day")),
            ("Total Return", _pct(total_return), _fmt_delta_pp(delta_total_ret_1d, "vs prev day")),
            ("Annualized Return", _pct(ann_return) if not np.isnan(ann_return) else "n/a", None),
            ("Volatility (ann.)", _pct(ann_vol), _fmt_delta_pp(delta_vol_1d, "vs prev day")),
            ("Max Drawdown", _pct(max_dd), _fmt_delta_pp(delta_max_dd_1d, "vs prev day")),
        ]
        x_grid = [0.04, 0.285, 0.53, 0.775]
        y_grid = [0.56, 0.355]
        idx = 0
        for y in y_grid:
            for x in x_grid:
                title, value, delta = cards[idx]
                _kpi_box(ax1, x, y, 0.195, 0.165, title, value, delta=delta)
                idx += 1
        ax1.text(0.04, 0.327, "Delta notation: pp = percentage points.", transform=ax1.transAxes, fontsize=8.8, color="#4d6073")

        ax1.text(0.04, 0.245, "Data handling and constraints", transform=ax1.transAxes, fontsize=12, fontweight="bold", color="#1e3c56")
        ax1.text(
            0.04,
            0.122,
            "Session-only processing: no persistent storage for uploaded statements or generated reports.\n"
            "No statement content is saved, retained, or manually reviewed by us.\n"
            "Single statement workflow per run.\n"
            "Analytics coverage can be constrained by Trade Republic statement granularity and external source availability.",
            transform=ax1.transAxes,
            fontsize=9.8,
            color="#41586d",
            linespacing=1.55,
        )
        _add_footer(fig1, page=1)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # Page 2: Capital trajectory and drawdown.
        fig2 = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        gs2 = fig2.add_gridspec(2, 1, height_ratios=[1.5, 0.9], hspace=0.22)
        ax2a = fig2.add_subplot(gs2[0, 0])
        ax2b = fig2.add_subplot(gs2[1, 0])

        ax2a.plot(chart["date"], chart["account_value"], color="#0f4c81", linewidth=2.4, label="Account Value")
        ax2a.plot(chart["date"], chart["net_invested"], color="#2e8b57", linewidth=2.0, linestyle="--", label="Net Invested")
        ax2a.yaxis.set_major_formatter(FuncFormatter(_fmt_eur_axis))
        ax2a.set_title("Capital Evolution", loc="left", fontsize=12.5)
        ax2a.grid(alpha=0.2)
        ax2a.legend(loc="upper left", frameon=False)
        ax2a_ret = ax2a.twinx()
        denom = chart["net_invested"].replace(0.0, np.nan)
        ret_line = ((chart["account_value_ex_pm"] - chart["net_invested"]) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ax2a_ret.plot(chart["date"], ret_line * 100.0, color="#f28e2b", linewidth=1.7, alpha=0.95)
        ax2a_ret.set_ylabel("Return %")
        _apply_date_axis(ax2a, month_interval=2)

        if not drawdown.empty:
            ax2b.fill_between(drawdown.index, drawdown.values * 100.0, 0.0, color="#d9534f", alpha=0.25)
            ax2b.plot(drawdown.index, drawdown.values * 100.0, color="#c64b47", linewidth=1.8)
        ax2b.axhline(0, color="#2d2d2d", linewidth=0.8)
        ax2b.set_title("Drawdown Profile", loc="left", fontsize=12.0)
        ax2b.set_ylabel("Drawdown %")
        ax2b.grid(alpha=0.2)
        _apply_date_axis(ax2b, month_interval=2)
        _add_footer(fig2, page=2)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # Page 3: Performance analytics.
        fig3 = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        gs3 = fig3.add_gridspec(2, 2, height_ratios=[0.9, 1.1], hspace=0.34, wspace=0.24)
        ax3a = fig3.add_subplot(gs3[0, 0])
        ax3b = fig3.add_subplot(gs3[0, 1])
        ax3c = fig3.add_subplot(gs3[1, :])

        if not twr.empty:
            eq = (1.0 + twr).cumprod() * 100.0
            ax3a.plot(eq.index, eq.values, color="#0f4c81", linewidth=2.2)
            ax3a.set_title("TWR Equity Curve (Base 100)", loc="left", fontsize=12.0)
            ax3a.grid(alpha=0.2)
            _apply_date_axis(ax3a, month_interval=2)
            roll_30 = (1.0 + twr).rolling(30).apply(np.prod, raw=True) - 1.0
            roll_90 = (1.0 + twr).rolling(90).apply(np.prod, raw=True) - 1.0
            roll_180 = (1.0 + twr).rolling(180).apply(np.prod, raw=True) - 1.0
            ax3b.plot(roll_30.index, roll_30.values * 100.0, label="30d", color="#1f78b4")
            ax3b.plot(roll_90.index, roll_90.values * 100.0, label="90d", color="#33a02c")
            ax3b.plot(roll_180.index, roll_180.values * 100.0, label="180d", color="#e31a1c")
            ax3b.set_title("Rolling Returns", loc="left", fontsize=12.0)
            ax3b.grid(alpha=0.2)
            ax3b.legend(frameon=False)
            _apply_date_axis(ax3b, month_interval=2)
        else:
            ax3a.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax3a.axis("off")
            ax3b.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax3b.axis("off")

        ax3c.axis("off")
        ax3c.set_title("Monthly Return Table", loc="left", fontsize=12.0, pad=8)
        if not monthly.empty:
            month_tbl = monthly.copy().tail(6).round(4).fillna(np.nan)
            cell_txt = []
            for _, row in month_tbl.iterrows():
                cell_txt.append([f"{v:.2%}" if pd.notna(v) else "" for v in row.values.tolist()])
            tbl = ax3c.table(
                cellText=cell_txt,
                colLabels=month_tbl.columns.tolist(),
                rowLabels=[str(i) for i in month_tbl.index.tolist()],
                loc="center",
                cellLoc="center",
                rowLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.4)
            tbl.scale(1, 1.38)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor("#d8e3ed")
                if r == 0 or c == -1:
                    cell.set_facecolor("#eaf1f8")
                    cell.set_text_props(weight="bold")
        else:
            ax3c.text(0.5, 0.5, "No monthly return history available.", ha="center", va="center", fontsize=10)
        _add_footer(fig3, page=3)
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        # Page 4: Allocation and contributors.
        fig4 = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        gs4 = fig4.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.22)
        ax4a = fig4.add_subplot(gs4[0, 0])
        ax4b = fig4.add_subplot(gs4[0, 1])
        ax4c = fig4.add_subplot(gs4[1, :])

        if not alloc.empty and alloc["market_value"].sum() > 0:
            alloc_show = alloc.head(6).copy()
            other_mv = float(alloc.iloc[6:]["market_value"].sum()) if len(alloc) > 6 else 0.0
            if other_mv > 0:
                alloc_show = pd.concat([alloc_show, pd.DataFrame([{"asset_class": "other", "market_value": other_mv}])], ignore_index=True)
            ax4a.pie(
                alloc_show["market_value"],
                labels=alloc_show["asset_class"],
                autopct="%1.0f%%",
                startangle=120,
                colors=["#0f4c81", "#2e8b57", "#f28e2b", "#3d7ea6", "#b56576", "#7f8c98", "#c2cad1"][: len(alloc_show)],
                textprops={"fontsize": 8.7},
            )
            ax4a.set_title("Asset Allocation", loc="left", fontsize=12.0)
        else:
            ax4a.text(0.5, 0.5, "No allocation data", ha="center", va="center")
            ax4a.axis("off")

        if not contrib.empty:
            top = contrib.sort_values("pnl", ascending=False).head(6).sort_values("pnl", ascending=True)
            bottom = contrib.sort_values("pnl", ascending=True).head(6)
            merged = pd.concat([bottom, top], ignore_index=True).drop_duplicates(subset=["name"])
            colors = ["#d9534f" if v < 0 else "#2e8b57" for v in merged["pnl"]]
            merged["name_short"] = merged["name"].map(lambda s: _short_label(s, 18))
            ax4b.barh(merged["name_short"], merged["pnl"], color=colors)
            ax4b.xaxis.set_major_formatter(FuncFormatter(_fmt_eur_axis))
            ax4b.set_title("Top/Bottom Contributors by PnL", loc="left", fontsize=12.0)
            ax4b.grid(axis="x", alpha=0.2)
            ax4b.tick_params(axis="y", labelsize=8.3)
        else:
            ax4b.text(0.5, 0.5, "No contributor data", ha="center", va="center")
            ax4b.axis("off")

        ax4c.axis("off")
        ax4c.set_title("Top Holdings Snapshot", loc="left", fontsize=12.0, pad=8)
        if not contrib.empty:
            hold = contrib[["name", "weight_pct", "market_value", "pnl", "pnl_pct"]].head(10).copy()
            hold["name"] = hold["name"].map(lambda s: _short_label(s, 24))
            hold["weight_pct"] = hold["weight_pct"].map(lambda v: f"{v:.2%}")
            hold["market_value"] = hold["market_value"].map(lambda v: f"EUR {v:,.0f}")
            hold["pnl"] = hold["pnl"].map(lambda v: f"EUR {v:,.0f}")
            hold["pnl_pct"] = hold["pnl_pct"].map(lambda v: f"{v:.2%}")
            hold.columns = ["Position", "Weight", "Market Value", "PnL", "PnL %"]
            t_hold = ax4c.table(
                cellText=hold.values,
                colLabels=hold.columns.tolist(),
                loc="center",
                cellLoc="left",
                colLoc="left",
            )
            t_hold.auto_set_font_size(False)
            t_hold.set_fontsize(8.4)
            t_hold.scale(1, 1.16)
            for (r, c), cell in t_hold.get_celld().items():
                cell.set_edgecolor("#d9e3ed")
                if r == 0:
                    cell.set_facecolor("#eaf1f8")
                    cell.set_text_props(weight="bold")
                else:
                    cell.set_facecolor("#ffffff" if r % 2 else "#f8fbfe")
        else:
            ax4c.text(0.5, 0.5, "No holdings data available.", ha="center", va="center", fontsize=10)
        _add_footer(fig4, page=4)
        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)

        # Page 5: Risk and data sources.
        fig5 = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        gs5 = fig5.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.24)
        ax5a = fig5.add_subplot(gs5[0, 0])
        ax5b = fig5.add_subplot(gs5[0, 1])
        ax5c = fig5.add_subplot(gs5[1, 0])
        ax5d = fig5.add_subplot(gs5[1, 1])

        if not twr.empty:
            dist = (twr * 100.0).dropna()
            ax5a.hist(dist, bins=36, color="#6c7a89", alpha=0.88)
            ax5a.axvline(var_95 * 100.0, color="#d9534f", linestyle="--", linewidth=1.2, label="VaR 95%")
            ax5a.axvline(var_99 * 100.0, color="#b52b27", linestyle="--", linewidth=1.2, label="VaR 99%")
            ax5a.set_title("Tail Risk Distribution", loc="left", fontsize=12.0)
            ax5a.set_xlabel("Daily return (%)")
            ax5a.legend(frameon=False)
            ax5a.grid(alpha=0.2)
            if not dist.empty:
                lo = float(dist.quantile(0.01))
                hi = float(dist.quantile(0.99))
                span = max(1.0, hi - lo)
                ax5a.set_xlim(lo - 0.15 * span, hi + 0.15 * span)
            ax5a.text(
                0.02,
                0.98,
                f"ES95: {_pct(es_95)}\nES99: {_pct(es_99)}",
                transform=ax5a.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.28", fc="#ffffff", ec="#d4dfeb"),
            )
        else:
            ax5a.text(0.5, 0.5, "No risk return history", ha="center", va="center")
            ax5a.axis("off")

        if not risk_table.empty:
            rr = risk_table.head(10).sort_values("risk_proxy", ascending=True)
            rr["name_short"] = rr["name"].map(lambda s: _short_label(s, 18))
            ax5b.barh(rr["name_short"], rr["risk_proxy"] * 100.0, color="#2c7fb8")
            ax5b.set_title("Risk Contributors (weight x vol30)", loc="left", fontsize=12.0)
            ax5b.set_xlabel("Risk proxy (%)")
            ax5b.grid(axis="x", alpha=0.2)
            ax5b.tick_params(axis="y", labelsize=8.3)
        else:
            ax5b.text(0.5, 0.5, "No risk contributors", ha="center", va="center")
            ax5b.axis("off")

        ax5c.axis("off")
        ax5c.set_title("Data Quality Snapshot", loc="left", fontsize=12.0, pad=8)
        q_tbl = ax5c.table(
            cellText=quality.values,
            colLabels=quality.columns.tolist(),
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        q_tbl.auto_set_font_size(False)
        q_tbl.set_fontsize(8.7)
        q_tbl.scale(1, 1.18)
        for (r, c), cell in q_tbl.get_celld().items():
            cell.set_edgecolor("#d9e3ed")
            if r == 0:
                cell.set_facecolor("#eaf1f8")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#ffffff" if r % 2 else "#f8fbfe")

        ax5d.axis("off")
        ax5d.set_title("Data Sources, Limits, and Assumptions", loc="left", fontsize=12.0, pad=8)
        source_lines = []
        if not source_counts.empty:
            for _, row in source_counts.head(8).iterrows():
                source_lines.append(f"- {row['source']}: {int(row['rows'])} rows")
        else:
            source_lines.append("- No price-source metadata available")
        source_lines.extend(
            [
                "",
                "Sources:",
                "- Trade Republic statement data",
                "- OpenFIGI, Yahoo Finance, JustETF",
                "- Vontobel / Societe Generale (where available)",
                "",
                "Operational limits:",
                "- Single account statement per run",
                "- No persistent statement/report storage",
                "- Some analytics depend on broker export granularity",
                "",
                "Key assumptions:",
                "- Capital-gain tax model: 26% (Italian baseline)",
                "- Tax loss carry-forward (zainetto) for non-UCITS",
                "- FIFO lot matching on sells",
                "- Missing-price trades may use statement-implied prices",
                "- Unpriced/delisted positions use first-trade-date adjustment",
            ]
        )
        ax5d.text(0.0, 0.98, "\n".join(source_lines), transform=ax5d.transAxes, va="top", fontsize=8.7, color="#385167")

        _add_footer(fig5, page=5)
        pdf.savefig(fig5, bbox_inches="tight")
        plt.close(fig5)

    return buffer.getvalue()
