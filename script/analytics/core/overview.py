#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overview metrics and time series utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_OVERVIEW_START = pd.Timestamp("2025-03-14")


@dataclass
class OverviewResult:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    portfolio_value: float
    cash_balance: float
    net_cash_in: float
    pnl_abs: float
    private_markets_value: float
    chart_df: pd.DataFrame


def _safe_date(v) -> pd.Timestamp:
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).normalize()


def _xnpv(rate: float, amounts: np.ndarray, years: np.ndarray) -> float:
    return float(np.sum(amounts / np.power(1.0 + rate, years)))


def _xirr(flows: pd.DataFrame) -> Optional[float]:
    if flows.empty:
        return None

    f = flows.copy()
    f = f.dropna(subset=["date", "amount"])
    f["amount"] = pd.to_numeric(f["amount"], errors="coerce")
    f = f.dropna(subset=["amount"])
    f = f[f["amount"] != 0.0]
    if f.empty:
        return None

    if not ((f["amount"] > 0).any() and (f["amount"] < 0).any()):
        return None

    f = f.groupby("date", as_index=False)["amount"].sum().sort_values("date")
    t0 = f["date"].min()
    years = (f["date"] - t0).dt.days.to_numpy(dtype=float) / 365.25
    amounts = f["amount"].to_numpy(dtype=float)

    # Newton method first.
    rate = 0.10
    for _ in range(100):
        if rate <= -0.999999:
            break
        denom = np.power(1.0 + rate, years)
        npv = float(np.sum(amounts / denom))
        d_npv = float(np.sum(-years * amounts / ((1.0 + rate) * denom)))
        if abs(npv) < 1e-8:
            return rate
        if abs(d_npv) < 1e-12:
            break
        next_rate = rate - (npv / d_npv)
        if not np.isfinite(next_rate):
            break
        if abs(next_rate - rate) < 1e-10:
            return float(next_rate)
        rate = float(next_rate)

    # Bisection fallback.
    low = -0.9999
    high = 1.0
    f_low = _xnpv(low, amounts, years)
    f_high = _xnpv(high, amounts, years)

    for _ in range(20):
        if np.sign(f_low) != np.sign(f_high):
            break
        high *= 2.0
        if high > 1000:
            return None
        f_high = _xnpv(high, amounts, years)

    if np.sign(f_low) == np.sign(f_high):
        return None

    for _ in range(200):
        mid = (low + high) / 2.0
        f_mid = _xnpv(mid, amounts, years)
        if abs(f_mid) < 1e-10 or (high - low) < 1e-10:
            return float(mid)
        if np.sign(f_mid) == np.sign(f_low):
            low = mid
            f_low = f_mid
        else:
            high = mid

    return float((low + high) / 2.0)


def _normalize_trades(df_trade_priced: pd.DataFrame, df_tx: pd.DataFrame) -> pd.DataFrame:
    if df_trade_priced.empty:
        return pd.DataFrame(columns=["trade_date", "isin", "qty_delta"])

    trades = df_trade_priced.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"], errors="coerce").dt.normalize()
    trades["isin"] = trades["isin"].astype(str)
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")
    side = trades["side"].astype(str).str.upper().str.strip()
    trades["qty_delta"] = np.where(side == "BUY", trades["quantity"], np.where(side == "SELL", -trades["quantity"], 0.0))
    trades = trades.dropna(subset=["trade_date", "isin", "qty_delta"])
    trades = trades[["trade_date", "isin", "qty_delta"]]

    # Add Private Markets pseudo-position (nominal at price 1) from cash flows.
    pm = df_tx.copy()
    pm["date"] = pd.to_datetime(pm["date"], errors="coerce").dt.normalize()
    pm["description"] = pm["description"].astype(str).str.lower()
    pm = pm[(pm["type"] == "Commercio") & (pm["isin"].isna()) & pm["description"].str.contains("private market")]
    if not pm.empty:
        pm["cash_flow"] = pd.to_numeric(pm["cash_flow"], errors="coerce").fillna(0.0)
        pm["qty_delta"] = -pm["cash_flow"]  # buy -> positive qty, sell -> negative qty
        pm = pm.dropna(subset=["date", "qty_delta"])
        pm = pm[["date", "qty_delta"]].rename(columns={"date": "trade_date"})
        pm["isin"] = "PM_PRIVATE_MARKETS"
        pm = pm.groupby(["trade_date", "isin"], as_index=False)["qty_delta"].sum()
        trades = pd.concat([trades, pm], ignore_index=True)

    return trades


def _portfolio_value_series(
    df_trade_priced: pd.DataFrame,
    df_prices_long: pd.DataFrame,
    df_tx: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    idx = pd.date_range(start_date, end_date, freq="D")
    if idx.empty:
        return pd.Series(dtype=float)

    trades = _normalize_trades(df_trade_priced, df_tx=df_tx)
    if trades.empty:
        return pd.Series(0.0, index=idx)

    prices = df_prices_long.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["date", "isin", "price"])
    prices = prices[prices["date"] <= end_date].copy()

    if prices.empty:
        return pd.Series(0.0, index=idx)

    full_start = min(start_date, trades["trade_date"].min(), prices["date"].min())
    full_idx = pd.date_range(full_start, end_date, freq="D")

    deltas = (
        trades[trades["trade_date"] <= end_date]
        .groupby(["trade_date", "isin"], as_index=False)["qty_delta"]
        .sum()
        .pivot(index="trade_date", columns="isin", values="qty_delta")
        .fillna(0.0)
    )
    qty = deltas.reindex(full_idx, fill_value=0.0).cumsum()
    qty = qty.reindex(columns=sorted(qty.columns))

    px = (
        prices[prices["isin"].isin(qty.columns)]
        .groupby(["date", "isin"], as_index=False)["price"]
        .last()
        .pivot(index="date", columns="isin", values="price")
    )
    px = px.reindex(full_idx).ffill()
    px = px.reindex(columns=qty.columns)
    if "PM_PRIVATE_MARKETS" in px.columns:
        px["PM_PRIVATE_MARKETS"] = 1.0

    port = (qty * px).sum(axis=1, min_count=1).fillna(0.0)
    return port.reindex(idx).fillna(0.0)


def _missing_price_adjustment_series(
    df_trade_priced: pd.DataFrame,
    df_prices_long: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    idx = pd.date_range(start_date, end_date, freq="D")
    if idx.empty or df_trade_priced.empty:
        return pd.Series(0.0, index=idx)

    trades = df_trade_priced.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"], errors="coerce").dt.normalize()
    trades["cash_flow_stmt"] = pd.to_numeric(trades["cash_flow_stmt"], errors="coerce").fillna(0.0)

    priced_isins = set(df_prices_long["isin"].dropna().unique().tolist())
    traded_isins = set(trades["isin"].dropna().unique().tolist())
    missing_isins = sorted(traded_isins - priced_isins)
    if not missing_isins:
        return pd.Series(0.0, index=idx)

    adj = pd.Series(0.0, index=idx)
    for isin in missing_isins:
        t = trades[trades["isin"] == isin]
        if t.empty:
            continue
        first_date = t["trade_date"].min()
        if pd.isna(first_date) or first_date < start_date or first_date > end_date:
            continue
        net_cf = float(t["cash_flow_stmt"].sum())
        # Apply algebraic sum on first trade date (treat as opened/closed same day).
        adj.loc[first_date] += net_cf
    return adj


def _cash_balance_series(
    df_tx: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    anchor_date: pd.Timestamp | None = None,
    anchor_saldo: float | None = None,
) -> pd.Series:
    idx = pd.date_range(start_date, end_date, freq="D")
    if idx.empty:
        return pd.Series(dtype=float)

    if df_tx.empty:
        return pd.Series(0.0, index=idx)

    tx = df_tx.copy()
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce").dt.normalize()
    tx["saldo"] = pd.to_numeric(tx["saldo"], errors="coerce")
    tx = tx.dropna(subset=["date"])
    tx = tx[tx["date"] <= end_date]

    if tx.empty:
        return pd.Series(0.0, index=idx)

    # Anchor to explicit date/saldo and accumulate cash flows.
    tx_sorted = tx.sort_values("date")
    tx_sorted["cash_flow"] = pd.to_numeric(tx_sorted["cash_flow"], errors="coerce").fillna(0.0)
    if anchor_date is None or anchor_saldo is None:
        anchor_date = tx_sorted["date"].min()
        anchor_saldo = 0.0

    daily_cf = tx_sorted.groupby("date", as_index=True)["cash_flow"].sum()
    daily_cf = daily_cf.reindex(pd.date_range(anchor_date, end_date, freq="D"), fill_value=0.0)
    cash_from_anchor = daily_cf.cumsum() + anchor_saldo
    cash = cash_from_anchor.reindex(idx).ffill()
    cash = cash.reindex(idx).fillna(0.0)
    return cash


def _net_invested_series(df_tx: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    idx = pd.date_range(start_date, end_date, freq="D")
    if idx.empty or df_tx.empty:
        return pd.Series(0.0, index=idx)

    tx = df_tx.copy()
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce").dt.normalize()
    tx["cash_flow"] = pd.to_numeric(tx["cash_flow"], errors="coerce").fillna(0.0)
    tx["type"] = tx["type"].astype(str)
    tx = tx[(tx["date"] >= start_date) & (tx["date"] <= end_date)]

    daily = (
        tx[tx["type"] == "Bonifico"]
        .groupby("date", as_index=True)["cash_flow"]
        .sum()
    )
    return daily.reindex(idx, fill_value=0.0).cumsum()


def build_overview_result(
    df_tx: pd.DataFrame,
    df_trade_priced: pd.DataFrame,
    df_prices_long: pd.DataFrame,
    start_date,
    end_date,
) -> OverviewResult:
    start = _safe_date(start_date)
    end = _safe_date(end_date)
    if pd.isna(start):
        tx_dates = pd.to_datetime(df_tx.get("date"), errors="coerce").dropna()
        start = tx_dates.min().normalize() if len(tx_dates) else DEFAULT_OVERVIEW_START
    if pd.isna(end):
        end = pd.Timestamp.today().normalize()
    if start > end:
        start, end = end, start

    # Use first Bonifico as logical start to avoid pre-funding balances.
    tx_tmp = df_tx.copy()
    tx_tmp["date"] = pd.to_datetime(tx_tmp["date"], errors="coerce").dt.normalize()
    bonifico_start = tx_tmp.loc[tx_tmp["type"] == "Bonifico", "date"].min()
    if pd.notna(bonifico_start) and bonifico_start > start:
        start = bonifico_start

    portfolio_series = _portfolio_value_series(df_trade_priced, df_prices_long, df_tx, start, end)
    cash_series = _cash_balance_series(
        df_tx,
        start,
        end,
        anchor_date=start,
        anchor_saldo=0.0,
    )
    # Adjust cash for instruments without price history (apply net cash flow on buy date).
    missing_adj = _missing_price_adjustment_series(df_trade_priced, df_prices_long, start, end)
    cash_series = cash_series.add(missing_adj, fill_value=0.0)
    net_invested_series = _net_invested_series(df_tx, start, end)

    common_idx = portfolio_series.index.union(cash_series.index).union(net_invested_series.index)
    portfolio_series = portfolio_series.reindex(common_idx).ffill().fillna(0.0)
    cash_series = cash_series.reindex(common_idx).ffill().fillna(0.0)
    net_invested_series = net_invested_series.reindex(common_idx).ffill().fillna(0.0)
    account_series = cash_series + portfolio_series

    cash_balance = float(cash_series.iloc[-1]) if not cash_series.empty else 0.0
    portfolio_value = float(portfolio_series.iloc[-1]) if not portfolio_series.empty else 0.0

    # Split out Private Markets nominal value for PnL neutrality.
    pm_value = 0.0
    if "PM_PRIVATE_MARKETS" in df_trade_priced.get("isin", pd.Series(dtype=str)).unique().tolist():
        # PM included via pseudo-position; estimate value from positions series if present.
        if "PM_PRIVATE_MARKETS" in df_prices_long.get("isin", pd.Series(dtype=str)).unique().tolist():
            pm_value = 0.0
    if "PM_PRIVATE_MARKETS" in portfolio_series.index.name if False else False:
        pm_value = 0.0

    # Recompute PM value from series if available in chart data.
    pm_series = pd.Series(0.0, index=account_series.index)
    # When PM is enabled, it's already included in portfolio_series at price 1.
    # We reconstruct it from tx cash flows to separate PnL.
    pm_tx = df_tx.copy()
    pm_tx["date"] = pd.to_datetime(pm_tx["date"], errors="coerce").dt.normalize()
    pm_tx["description"] = pm_tx["description"].astype(str).str.lower()
    pm_tx = pm_tx[(pm_tx["type"] == "Commercio") & (pm_tx["isin"].isna()) & pm_tx["description"].str.contains("private market")]
    if not pm_tx.empty:
        pm_tx["cash_flow"] = pd.to_numeric(pm_tx["cash_flow"], errors="coerce").fillna(0.0)
        daily_pm = pm_tx.groupby("date", as_index=True)["cash_flow"].sum().reindex(account_series.index, fill_value=0.0)
        pm_series = (-daily_pm).cumsum()
        pm_value = float(pm_series.iloc[-1]) if not pm_series.empty else 0.0
    net_cash_in = float(net_invested_series.iloc[-1]) if not net_invested_series.empty else 0.0
    pnl_abs = float((account_series.iloc[-1] - pm_value) - net_cash_in) if not account_series.empty else 0.0

    chart_df = pd.DataFrame(
        {
            "date": common_idx,
            "cash_balance": cash_series.values,
            "portfolio_value": portfolio_series.values,
            "net_invested": net_invested_series.values,
            "account_value": account_series.values,
            "private_markets_value": pm_series.reindex(common_idx).values,
            "account_value_ex_pm": (account_series - pm_series).values,
        }
    )

    return OverviewResult(
        start_date=start,
        end_date=end,
        portfolio_value=portfolio_value,
        cash_balance=cash_balance,
        net_cash_in=net_cash_in,
        pnl_abs=pnl_abs,
        private_markets_value=pm_value,
        chart_df=chart_df,
    )
