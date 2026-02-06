#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical tables builder from DATALAKE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


def load_datalake(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text())


def build_df_master(datalake: Dict[str, object]) -> pd.DataFrame:
    security_master = datalake.get("security_master", {}) or {}
    price_sources = datalake.get("price_sources", {}) or {}

    rows = []
    for isin, sec in security_master.items():
        sec = sec or {}
        rows.append(
            {
                "isin": isin,
                "asset_class": sec.get("asset_class"),
                "instrument_type": sec.get("instrument_type"),
                "market_sector": sec.get("market_sector"),
                "name": sec.get("name"),
                "ticker": sec.get("ticker"),
                "exch_code": sec.get("exch_code"),
                "currency": sec.get("currency"),
                "yfinance_ticker": sec.get("yfinance_ticker"),
                "security_type": sec.get("security_type"),
                "security_type2": sec.get("security_type2"),
                "price_source": price_sources.get(isin),
                "yfinance_profile": sec.get("yfinance_profile"),
                "justetf_exposure": sec.get("justetf_exposure"),
            }
        )

    return pd.DataFrame(rows)


def _iter_transactions(datalake: Dict[str, object]) -> Iterable[Dict[str, object]]:
    statements = datalake.get("statements", {}) or {}
    for payload in statements.values():
        for txn in payload.get("transactions", []) or []:
            yield txn


def build_df_tx(datalake: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for txn in _iter_transactions(datalake):
        trade = txn.get("trade") or {}
        rows.append(
            {
                "tx_id": txn.get("id"),
                "date": pd.to_datetime(txn.get("date"), errors="coerce"),
                "seq": txn.get("seq"),
                "type": txn.get("type"),
                "description": txn.get("description"),
                "cash_flow": float(txn.get("cash_flow") or 0.0),
                "in_entrata": float(txn.get("in_entrata") or 0.0),
                "in_uscita": float(txn.get("in_uscita") or 0.0),
                "saldo": float(txn.get("saldo") or 0.0),
                "isin": trade.get("isin"),
                "side": trade.get("side"),
                "quantity": pd.to_numeric(trade.get("quantity"), errors="coerce"),
                "share_class": trade.get("share_class"),
                "instrument_name": trade.get("instrument_name"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "type", "tx_id"], na_position="last").reset_index(drop=True)
    return df


def build_df_prices_long(datalake: Dict[str, object]) -> pd.DataFrame:
    rows = (datalake.get("prices", {}) or {}).get("long", []) or []
    price_sources = datalake.get("price_sources", {}) or {}
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "isin", "price", "price_source"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price_source"] = df["isin"].map(price_sources)
    df = df.dropna(subset=["date", "isin", "price"]).sort_values(["isin", "date"]).reset_index(drop=True)
    return df[["date", "isin", "price", "price_source"]]


def build_canonical_tables(datalake: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_master = build_df_master(datalake)
    df_tx = build_df_tx(datalake)
    df_prices_long = build_df_prices_long(datalake)
    return df_master, df_tx, df_prices_long
