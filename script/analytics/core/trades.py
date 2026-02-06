#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trade pricing and reconciliation utilities."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DEFAULT_TRADE_FEE_EUR = 1.0


def _aggregate_same_day_trades(tx: pd.DataFrame) -> pd.DataFrame:
    if tx.empty:
        return tx

    grp_cols = ["trade_date", "isin", "side"]
    agg = (
        tx.groupby(grp_cols, dropna=False)
        .agg(
            quantity=("quantity", "sum"),
            cash_flow_stmt=("cash_flow_stmt", "sum"),
            in_entrata=("in_entrata", "sum"),
            in_uscita=("in_uscita", "sum"),
            tx_ids=("tx_id", lambda s: [x for x in s if pd.notna(x)]),
            descriptions=("description", lambda s: [x for x in s if pd.notna(x)]),
        )
        .reset_index()
    )

    def _mk_id(ids):
        ids = sorted(set(ids))
        if len(ids) == 1:
            return ids[0]
        raw = "|".join(ids)
        return "agg:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()

    agg["tx_id"] = agg["tx_ids"].apply(_mk_id)
    agg["description"] = agg["descriptions"].apply(lambda xs: xs[0] if xs else None)
    agg["n_aggregated"] = agg["tx_ids"].apply(len)
    return agg


def _apply_sell_tax_fifo(priced: pd.DataFrame, is_ucits_by_isin: Dict[str, bool]) -> pd.DataFrame:
    out = priced.copy()
    out["capital_gain_eur"] = 0.0
    out["minus_added_eur"] = 0.0
    out["minus_applied_eur"] = 0.0
    out["minus_pool_eur"] = 0.0
    out["tax_capital_gain_eur"] = 0.0
    out["tax_base_gain_eur"] = 0.0
    out["cash_flow_model_net"] = out["cash_flow_model"]
    out["cash_flow_model_abs_net"] = out["cash_flow_model_abs"]

    lot_queues = defaultdict(list)
    minus_pool = []  # [{"amount": float, "expiry": pd.Timestamp}]

    rows = out.sort_values(["trade_date", "tx_id"])
    for ridx, r in rows.iterrows():
        side = str(r.get("side") or "").upper()
        isin = str(r.get("isin") or "")
        qty = float(r.get("quantity") or 0.0)
        if qty <= 0 or not isin:
            continue

        if side == "BUY":
            unit_cost = float(r.get("cash_flow_model_abs") or 0.0) / qty if qty else 0.0
            lot_queues[isin].append({"qty": qty, "unit_cost": unit_cost})
            continue

        if side != "SELL":
            continue

        # Always reduce lots for sells (FIFO).
        queue = lot_queues[isin]

        remaining = qty
        cost_basis = 0.0
        while remaining > 1e-12 and queue:
            lot = queue[0]
            take = min(remaining, lot["qty"])
            cost_basis += take * lot["unit_cost"]
            lot["qty"] -= take
            remaining -= take
            if lot["qty"] <= 1e-12:
                queue.pop(0)

        if remaining > 1e-12:
            cost_basis += 0.0

        sell_notional = float(abs(r.get("notional_mkt") or 0.0))
        fee = float(r.get("fee_eur") or 0.0)
        proceeds_net_fee = max(0.0, sell_notional - fee)

        raw_gain = proceeds_net_fee - cost_basis
        out.at[ridx, "capital_gain_eur"] = raw_gain

        trade_date = pd.to_datetime(r.get("trade_date"), errors="coerce")
        if pd.notna(trade_date):
            minus_pool = [m for m in minus_pool if m["expiry"] >= trade_date]

        is_ucits = bool(is_ucits_by_isin.get(isin, False))
        minus_added = 0.0
        minus_applied = 0.0
        taxable_gain = 0.0

        if raw_gain < 0:
            if not is_ucits:
                minus_added = -raw_gain
                expiry = (trade_date + pd.DateOffset(years=4)) if pd.notna(trade_date) else pd.Timestamp.max
                minus_pool.append({"amount": minus_added, "expiry": expiry})
        else:
            taxable_gain = raw_gain
            if not is_ucits and minus_pool:
                # Apply minus FIFO by expiry (oldest first).
                minus_pool.sort(key=lambda m: m["expiry"])
                remaining_gain = taxable_gain
                for m in minus_pool:
                    if remaining_gain <= 1e-12:
                        break
                    take = min(remaining_gain, m["amount"])
                    m["amount"] -= take
                    remaining_gain -= take
                    minus_applied += take
                minus_pool = [m for m in minus_pool if m["amount"] > 1e-12]
                taxable_gain = max(0.0, taxable_gain - minus_applied)

        tax = 0.26 * max(0.0, taxable_gain)

        out.at[ridx, "minus_added_eur"] = minus_added
        out.at[ridx, "minus_applied_eur"] = minus_applied
        out.at[ridx, "minus_pool_eur"] = float(sum(m["amount"] for m in minus_pool))
        out.at[ridx, "tax_base_gain_eur"] = max(0.0, taxable_gain)
        out.at[ridx, "tax_capital_gain_eur"] = tax

        # Skip applying tax to cash flow when price is statement-implied
        # (statement already includes tax effects), but keep fiscal metrics.
        if tax > 0 and str(r.get("price_origin") or "") != "account_statement":
            net_cf = float(r.get("cash_flow_model") or 0.0) - tax
            out.at[ridx, "cash_flow_model_net"] = net_cf
            out.at[ridx, "cash_flow_model_abs_net"] = abs(net_cf)

    out["diff_cash_flow"] = out["cash_flow_stmt"] - out["cash_flow_model_net"]
    out["diff_eur_abs"] = (out["cash_flow_stmt"].abs() - out["cash_flow_model_abs_net"]).abs()
    denom = out["cash_flow_model_abs_net"].replace(0, np.nan)
    out["diff_pct"] = out["diff_eur_abs"] / denom
    out["abs_diff_pct"] = out["diff_pct"].abs()
    return out


def build_trade_priced(
    df_tx: pd.DataFrame,
    df_prices_long: pd.DataFrame,
    df_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    tx = df_tx.copy()
    prices = df_prices_long.copy()

    if tx.empty:
        return pd.DataFrame(
            columns=[
                "tx_id",
                "trade_date",
                "isin",
                "side",
                "quantity",
                "price_date_used",
                "days_gap",
                "price_used",
                "price_origin",
                "notional_mkt",
                "fee_eur",
                "cash_flow_stmt",
                "cash_flow_model",
                "diff_cash_flow",
                "diff_pct",
                "abs_diff_pct",
                "n_aggregated",
                "severity",
            ]
        )

    tx = tx[(tx["type"] == "Commercio") & tx["isin"].notna()].copy()
    tx["trade_date"] = pd.to_datetime(tx["date"], errors="coerce")
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")
    tx["side"] = tx["side"].astype(str).str.strip().str.upper()
    tx["cash_flow_stmt"] = pd.to_numeric(tx["cash_flow"], errors="coerce").fillna(0.0)
    tx["in_entrata"] = pd.to_numeric(tx["in_entrata"], errors="coerce").fillna(0.0)
    tx["in_uscita"] = pd.to_numeric(tx["in_uscita"], errors="coerce").fillna(0.0)
    tx = tx.dropna(subset=["trade_date", "isin", "side", "quantity"])
    tx = _aggregate_same_day_trades(tx)

    prices = prices.rename(columns={"date": "price_date_used", "price": "price_used"}).copy()
    prices["price_date_used"] = pd.to_datetime(prices["price_date_used"], errors="coerce")
    prices = prices.dropna(subset=["price_date_used", "isin", "price_used"]).sort_values(["isin", "price_date_used"])

    tx = tx.sort_values(["isin", "trade_date"])

    chunks = []
    for isin, tx_g in tx.groupby("isin", dropna=False):
        px_g = prices[prices["isin"] == isin][["price_date_used", "price_used", "price_source"]].copy()
        tx_g = tx_g.sort_values("trade_date")
        px_g = px_g.sort_values("price_date_used")
        if px_g.empty:
            tx_g["price_date_used"] = pd.NaT
            tx_g["price_used"] = np.nan
            tx_g["price_origin"] = "missing"
            chunks.append(tx_g)
            continue
        merged = pd.merge_asof(
            tx_g,
            px_g,
            left_on="trade_date",
            right_on="price_date_used",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["price_origin"] = merged["price_source"].fillna("unknown")
        chunks.append(merged)

    priced = pd.concat(chunks, ignore_index=True) if chunks else tx.copy()

    priced["fee_eur"] = DEFAULT_TRADE_FEE_EUR

    # Fallback statement-implied price when market price is unavailable.
    implied_price = (priced["cash_flow_stmt"].abs() - priced["fee_eur"]).clip(lower=0) / priced["quantity"].abs()
    missing_market_price = priced["price_used"].isna()
    priced.loc[missing_market_price, "price_used"] = implied_price[missing_market_price]
    priced.loc[missing_market_price, "price_origin"] = np.where(
        priced.loc[missing_market_price, "price_used"].notna(),
        "account_statement",
        "missing",
    )
    priced.loc[missing_market_price & priced["price_date_used"].isna(), "price_date_used"] = priced.loc[
        missing_market_price & priced["price_date_used"].isna(), "trade_date"
    ]

    priced["days_gap"] = (priced["trade_date"] - priced["price_date_used"]).dt.days
    priced["notional_mkt"] = priced["quantity"] * priced["price_used"]
    # If notional cannot be computed, do not force a fee.
    priced.loc[priced["notional_mkt"].isna(), "fee_eur"] = 0.0

    side = priced["side"].fillna("").str.upper()
    priced["cash_flow_model_abs"] = np.where(
        side.isin(["BUY", "SELL"]),
        priced["notional_mkt"].abs() + priced["fee_eur"],
        np.nan,
    )
    priced["cash_flow_model"] = np.where(
        side == "BUY",
        -priced["cash_flow_model_abs"],
        np.where(side == "SELL", priced["cash_flow_model_abs"], np.nan),
    )
    is_ucits_by_isin = {}
    if df_master is not None and not df_master.empty:
        def _is_ucits_row(row: pd.Series) -> bool:
            if row.get("justetf_exposure") not in (None, {}, []):
                return True
            asset_class = str(row.get("asset_class") or "").lower()
            instrument_type = str(row.get("instrument_type") or "").lower()
            name = str(row.get("name") or "").lower()
            if asset_class == "etf":
                return True
            if instrument_type in {"etf", "etp"}:
                return True
            return "ucits" in name

        is_ucits_by_isin = {
            str(row.get("isin")): bool(_is_ucits_row(row))
            for _, row in df_master.iterrows()
            if pd.notna(row.get("isin"))
        }
    priced = _apply_sell_tax_fifo(priced, is_ucits_by_isin)

    keep = [
        "tx_id",
        "trade_date",
        "isin",
        "side",
        "quantity",
        "price_date_used",
        "days_gap",
        "price_used",
        "price_origin",
        "notional_mkt",
        "fee_eur",
        "cash_flow_stmt",
        "tax_base_gain_eur",
        "tax_capital_gain_eur",
        "capital_gain_eur",
        "minus_added_eur",
        "minus_applied_eur",
        "minus_pool_eur",
        "cash_flow_model_abs",
        "cash_flow_model",
        "cash_flow_model_net",
        "cash_flow_model_abs_net",
        "diff_cash_flow",
        "diff_eur_abs",
        "diff_pct",
        "abs_diff_pct",
        "n_aggregated",
    ]
    return priced[keep].sort_values(["trade_date", "isin", "tx_id"]).reset_index(drop=True)


def build_alerts(df_trade_priced: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df_trade_priced.empty:
        return pd.DataFrame(columns=["severity", "ref", "message"])

    for _, r in df_trade_priced.iterrows():
        tx_id = r.get("tx_id")
        isin = r.get("isin")
        ref = tx_id or isin

        if pd.isna(r.get("price_used")) or pd.isna(r.get("price_date_used")):
            rows.append(
                {
                    "severity": "CRITICAL",
                    "ref": ref,
                    "message": f"No price available <= trade date for ISIN {isin}",
                }
            )
            continue

        gap = r.get("days_gap")
        if pd.notna(gap) and float(gap) > 5:
            rows.append(
                {
                    "severity": "WARN",
                    "ref": ref,
                    "message": f"Price gap is {int(gap)} days",
                }
            )

        diff_abs = r.get("diff_eur_abs")
        diff_pct = r.get("abs_diff_pct")
        if (pd.notna(diff_abs) and float(diff_abs) > 5.0) or (pd.notna(diff_pct) and float(diff_pct) > 0.005):
            rows.append(
                {
                    "severity": "WARN",
                    "ref": ref,
                    "message": f"Mismatch: diff_abs={float(diff_abs or 0):.2f} EUR, diff_pct={float(diff_pct or 0):.2%}",
                }
            )

    alerts = pd.DataFrame(rows)
    if alerts.empty:
        return pd.DataFrame(columns=["severity", "ref", "message"])

    severity_order = {"CRITICAL": 0, "WARN": 1, "INFO": 2}
    alerts["_ord"] = alerts["severity"].map(severity_order).fillna(9)
    alerts = alerts.sort_values(["_ord", "ref", "message"]).drop(columns=["_ord"]).reset_index(drop=True)
    return alerts


def build_trade_pricing_and_alerts(
    df_tx: pd.DataFrame,
    df_prices_long: pd.DataFrame,
    df_master: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_trade_priced = build_trade_priced(df_tx, df_prices_long, df_master)
    df_alerts = build_alerts(df_trade_priced)
    return df_trade_priced, df_alerts
