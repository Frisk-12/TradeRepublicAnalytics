#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 18:34:24 2026

@author: andreadesogus
"""

from __future__ import annotations

import contextlib
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yfinance as yf


CORE_KEYS = [
    "symbol",
    "shortName",
    "longName",
    "quoteType",
    "currency",
    "exchange",
    "market",
    "sector",
    "industry",
    "country",
    "city",
    "longBusinessSummary",
    "marketCap",
    "enterpriseValue",
    "beta",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "dividendRate",
    "dividendYield",
    "payoutRatio",
    "profitMargins",
    "grossMargins",
    "operatingMargins",
    "returnOnAssets",
    "returnOnEquity",
    "revenueGrowth",
    "earningsGrowth",
    "fullTimeEmployees",
]


def fetch_stock_profile(ticker: str) -> Dict[str, object]:
    t = yf.Ticker(ticker)
    # Silence noisy yfinance stderr/stdout for missing symbols.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        info = t.info or {}
    out = {k: info.get(k) for k in CORE_KEYS}
    out["ticker_input"] = ticker
    out["as_of"] = datetime.utcnow().isoformat() + "Z"
    out["status"] = "matched" if info else "not_found"
    return out


def _load_cache(path: str) -> Dict[str, Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_cache(path: str, cache: Dict[str, Dict[str, object]]) -> None:
    try:
        Path(path).write_text(json.dumps(cache, indent=2, ensure_ascii=True))
    except Exception:
        return


def fetch_stock_profiles(
    tickers: List[str],
    *,
    cache_path: Optional[str] = None,
    report: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    cache = _load_cache(cache_path) if cache_path else {}
    for t in tickers:
        ticker = t.strip()
        if not ticker:
            continue
        if cache_path and ticker in cache:
            out[ticker] = cache[ticker]
            continue
        try:
            out[ticker] = fetch_stock_profile(ticker)
        except Exception as exc:
            out[ticker] = {
                "ticker_input": ticker,
                "as_of": datetime.utcnow().isoformat() + "Z",
                "status": "error",
                "error": str(exc),
            }
            if report is not None:
                report.setdefault("errors", []).append({"ticker": ticker, "error": str(exc)})
        if report is not None:
            report.setdefault("status_counts", {})
            st = out[ticker].get("status", "unknown")
            report["status_counts"][st] = report["status_counts"].get(st, 0) + 1
        if cache_path:
            cache[ticker] = out[ticker]
            _save_cache(cache_path, cache)
    return out


# if __name__ == "__main__":
#     d = fetch_stock_profile("AAPL")
#     for k in list(d.keys())[:10]:
#         print(k, d[k])
