#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:19:23 2026

@author: andreadesogus
"""

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

HIST_DIR = Path(__file__).resolve().parent
if str(HIST_DIR) not in sys.path:
    sys.path.insert(0, str(HIST_DIR))

from sogen import (
    SocieteGeneraleWarrantsHistoryClient,
    SocGenInstrumentNotFound,
    SocGenBadResponse,
)
from vontobel import VontobelMaxHistoryClient, VontobelInstrumentNotFound, VontobelBadResponse
from justetf_client import JustEtfClient
from yfinance_client import (
    YFinanceHistoryClient,
    YFinanceInstrumentNotFound,
    YFinanceBadResponse,
)
from openfigi.openfigi_api import OpenFigiClient

# Esempio: input ISIN (ordine irrilevante; la cascata gestisce le fonti)
ISINS: List[str] = [
    "DE000FD4TL81",
    "DE000VJ2GXA8",
    "IE00B8HGT870",
]

# Esempio: mapping prodotto SocGen (facoltativo, evita probing)
SOGEN_ISIN_TO_PRODUCT_ID = {
    "DE000FD4TL81": 6088482,
}


def _fill_holiday_gaps_only(wide: pd.DataFrame, *, method: str = "ffill") -> pd.DataFrame:
    if method not in {"ffill", "bfill"}:
        raise ValueError("method must be 'ffill' or 'bfill'")

    out = wide.copy()
    for col in [c for c in out.columns if c != "date"]:
        s = out[col]
        first = s.first_valid_index()
        last = s.last_valid_index()
        if first is None or last is None:
            continue
        if method == "ffill":
            out.loc[first:last, col] = s.loc[first:last].ffill()
        else:
            out.loc[first:last, col] = s.loc[first:last].bfill()
    return out


def _is_valid_frame(df: Optional[pd.DataFrame]) -> bool:
    if df is None:
        return False
    if getattr(df, "empty", True):
        return False
    if "price" in df.columns and df["price"].notna().any():
        return True
    if "mid" in df.columns and df["mid"].notna().any():
        return True
    return False


def _is_etf(security_master: Optional[Dict[str, dict]], isin: str) -> bool:
    if not security_master:
        return False
    info = security_master.get(isin, {})
    asset_class = str(info.get("asset_class", "")).lower()
    instrument_type = str(info.get("instrument_type", "")).lower()
    security_type = str(info.get("security_type", "")).lower()
    security_type2 = str(info.get("security_type2", "")).lower()
    name = str(info.get("name", "")).lower()
    ticker = str(info.get("ticker", "")).lower()
    justetf_exposure = info.get("justetf_exposure")
    if asset_class == "etf" or instrument_type in {"etf", "etp"}:
        return True
    if any(k in security_type for k in ["etf", "etp", "index fund"]):
        return True
    if any(k in security_type2 for k in ["etf", "etp", "fund"]):
        return True
    if any(k in name for k in [" etf", "ucits", "index fund", "exchange traded fund"]):
        return True
    if ticker.endswith(".etf"):
        return True
    if isinstance(justetf_exposure, dict) and bool(justetf_exposure):
        return True
    return False


def _is_stock(security_master: Optional[Dict[str, dict]], isin: str) -> bool:
    if not security_master:
        return False
    info = security_master.get(isin, {})
    asset_class = str(info.get("asset_class", "")).lower()
    instrument_type = str(info.get("instrument_type", "")).lower()
    security_type = str(info.get("security_type", "")).lower()
    security_type2 = str(info.get("security_type2", "")).lower()
    market_sector = str(info.get("market_sector", "")).lower()
    name = str(info.get("name", "")).lower()
    if asset_class == "equity" or instrument_type in {"stock", "equity"}:
        return True
    if market_sector == "equity":
        return True
    if any(k in security_type for k in ["common stock", "equity", "preferred"]):
        return True
    if any(k in security_type2 for k in ["common stock", "equity"]):
        return True
    if " ord " in f" {name} " and "etf" not in name:
        return True
    return False


def _justetf_to_df(series, isin: str) -> pd.DataFrame:
    if not series:
        return pd.DataFrame()
    df = pd.DataFrame([{"date": p.date, "price": p.value, "isin": isin} for p in series])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    return df


def _pick_candidate_for_yahoo(raw: Dict[str, object]) -> Optional[Dict[str, object]]:
    candidates = raw.get("candidates") or []
    best = raw.get("best") if isinstance(raw.get("best"), dict) else None

    if isinstance(best, dict) and best.get("micCode"):
        return best

    # Prefer any candidate with MIC (can be converted to Yahoo ticker)
    for c in candidates:
        if c.get("micCode"):
            return c

    # If no MIC, prefer explicit US candidate when present
    us = OpenFigiClient.best_candidate(candidates, preferred_exch="US")
    if isinstance(us, dict):
        return us

    if isinstance(best, dict):
        return best
    if candidates:
        return candidates[0]
    return None


def _build_yfinance_ticker_candidates(
    isins: List[str], security_master: Optional[Dict[str, dict]]
) -> Dict[str, List[str]]:
    if not security_master:
        return {}

    out: Dict[str, List[str]] = {}
    for isin in isins:
        sec = security_master.get(isin, {}) or {}
        raw = sec.get("raw_best") or {}
        candidates = raw.get("candidates") or []

        ordered: List[Dict[str, object]] = []
        # Prefer US candidate first if available, then best/others.
        us = OpenFigiClient.best_candidate(candidates, preferred_exch="US")
        if us:
            ordered.append(us)

        primary = _pick_candidate_for_yahoo(raw)
        if primary and primary not in ordered:
            ordered.append(primary)
        if not ordered and candidates:
            ordered.extend(candidates[:2])

        tickers: List[str] = []
        for c in ordered:
            t = OpenFigiClient.to_yahoo_symbol(c.get("ticker"), c.get("micCode"))
            if t and t not in tickers:
                tickers.append(t)

        fallback = str(sec.get("ticker") or isin)
        if fallback and fallback not in tickers:
            tickers.append(fallback)

        out[isin] = tickers
    return out


def build_yfinance_ticker_map(
    isins: List[str], security_master: Optional[Dict[str, dict]]
) -> Dict[str, str]:
    candidates = _build_yfinance_ticker_candidates(isins, security_master)
    return {isin: (lst[0] if lst else isin) for isin, lst in candidates.items()}


def fetch_all_prices(
    isins: List[str],
    *,
    security_master: Optional[Dict[str, dict]] = None,
    start: str = "2024-01-01",
    end: Optional[str] = None,
    return_sources: bool = False,
    fx_cache_path: Optional[str] = "/Users/andreadesogus/Desktop/portfolio_analytics/script/histpricetakers/fx_cache.json",
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    source_by_isin: Dict[str, str] = {}
    end = end or date.today().isoformat()

    yfin_candidates = _build_yfinance_ticker_candidates(isins, security_master)
    yfin = YFinanceHistoryClient(fx_cache_path=fx_cache_path)
    vont = VontobelMaxHistoryClient()
    socgen = SocieteGeneraleWarrantsHistoryClient(
        resolver_mode="http_only",
        isin_to_product_id=SOGEN_ISIN_TO_PRODUCT_ID,
    )
    justetf = JustEtfClient(locale="it")

    for isin in isins:
        df = None

        # 0) justetf (ETF only)
        if _is_etf(security_master, isin):
            try:
                series = justetf.fetch_performance_chart(
                    isin=isin,
                    date_from=start,
                    date_to=end,
                    currency="EUR",
                    values_type="MARKET_VALUE",
                    include_dividends=False,
                    reduce_data=False,
                )
            except Exception:
                series = []
            df = _justetf_to_df(series, isin)
            if _is_valid_frame(df):
                df["source"] = "justetf"
                frames.append(df)
                source_by_isin[isin] = "justetf"
                continue

        # 1) yfinance (equity only)
        if _is_stock(security_master, isin):
            for ticker in yfin_candidates.get(isin, []):
                try:
                    df = yfin.fetch_daily_history(
                        isin,
                        as_pandas=True,
                        start=start,
                        end=end,
                        ticker_override=ticker,
                    )
                except (YFinanceInstrumentNotFound, YFinanceBadResponse, ValueError):
                    df = None
                if _is_valid_frame(df):
                    df = df[["date", "price", "isin"]].copy()
                    df["source"] = "yfinance"
                    frames.append(df)
                    source_by_isin[isin] = "yfinance"
                    source_by_isin[f"{isin}__yahoo_ticker"] = ticker
                    break
            if source_by_isin.get(isin) == "yfinance":
                continue

        # 2) vontobel
        try:
            df = vont.fetch_max_daily(isin, as_pandas=True)
        except (VontobelInstrumentNotFound, VontobelBadResponse, ValueError):
            df = None
        if _is_valid_frame(df):
            df = df[["date", "price", "isin"]].copy()
            df["source"] = "vontobel"
            frames.append(df)
            source_by_isin[isin] = "vontobel"
            continue

        # 3) societe generale
        try:
            df = socgen.fetch_daily_history(isin, as_pandas=True)
        except (SocGenInstrumentNotFound, SocGenBadResponse, ValueError):
            df = None
        if _is_valid_frame(df):
            df = df[["date", "mid", "isin"]].rename(columns={"mid": "price"})
            df["source"] = "sogen"
            frames.append(df)
            source_by_isin[isin] = "sogen"

    if not frames:
        empty = pd.DataFrame(columns=["date"])
        return (empty, source_by_isin) if return_sources else empty

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "isin", "source"])
    out = out.groupby(["date", "isin"], as_index=False).last()
    wide = out.pivot(index="date", columns="isin", values="price").reset_index()
    wide = wide.sort_values("date").reset_index(drop=True)
    # Fill only internal gaps (e.g., holidays); keep NaN before listing or after delisting.
    wide = _fill_holiday_gaps_only(wide, method="bfill")
    return (wide, source_by_isin) if return_sources else wide


if __name__ == "__main__":
    df_prices = fetch_all_prices(ISINS)
    print(df_prices.tail())
