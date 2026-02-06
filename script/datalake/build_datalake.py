#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a unified "datalake" object with instrument master + account statements + price history.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SCRIPT_DIR = SCRIPT_DIR.parent
if str(REPO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPT_DIR))

from statement_enricher.enrich_account_statements_with_openfigi import (  # noqa: E402
    enrich_account_statements,
)
from histpricetakers.main import fetch_all_prices, build_yfinance_ticker_map  # noqa: E402
from histpricetakers.justetf_sectors import fetch_exposures  # noqa: E402
from histpricetakers.yfinance_extra import fetch_stock_profiles  # noqa: E402

DEFAULT_STATEMENTS_DIR = "/Users/andreadesogus/Desktop/portfolio_analytics/data/account_statement"
DEFAULT_OUTPUT_JSON = "/Users/andreadesogus/Desktop/portfolio_analytics/script/datalake/datalake.json"
DEFAULT_JUSTETF_CACHE = "/Users/andreadesogus/Desktop/portfolio_analytics/script/datalake/justetf_exposure_cache.json"
DEFAULT_YFINANCE_CACHE = "/Users/andreadesogus/Desktop/portfolio_analytics/script/datalake/yfinance_profile_cache.json"


def _is_etf_like(sec: Dict[str, object]) -> bool:
    asset_class = str(sec.get("asset_class") or "").lower()
    instrument_type = str(sec.get("instrument_type") or "").lower()
    security_type = str(sec.get("security_type") or "").lower()
    security_type2 = str(sec.get("security_type2") or "").lower()
    name = str(sec.get("name") or "").lower()
    if asset_class == "etf" or instrument_type in {"etf", "etp"}:
        return True
    if any(k in security_type for k in ["etf", "etp", "index fund"]):
        return True
    if any(k in security_type2 for k in ["etf", "etp", "fund"]):
        return True
    if any(k in name for k in [" etf", "ucits", "exchange traded fund", "index fund"]):
        return True
    return False


def build_datalake(
    statements_dir: str = DEFAULT_STATEMENTS_DIR,
    *,
    openfigi_cache_path: str | None = None,
    justetf_cache_path: str | None = DEFAULT_JUSTETF_CACHE,
    yfinance_cache_path: str | None = DEFAULT_YFINANCE_CACHE,
    fx_cache_path: str | None = "/Users/andreadesogus/Desktop/portfolio_analytics/script/histpricetakers/fx_cache.json",
) -> Dict[str, object]:
    enriched = enrich_account_statements(
        statements_dir,
        cache_path=openfigi_cache_path,
    )
    security_master = enriched.get("security_master", {})
    statements = enriched.get("statements", {})
    openfigi_report = enriched.get("openfigi_report", {})

    benchmark_isin = "IE000BI8OT95"
    if benchmark_isin not in security_master:
        security_master[benchmark_isin] = {
            "isin": benchmark_isin,
            "name": "MSCI World UCITS ETF (benchmark)",
            "asset_class": "etf",
            "instrument_type": "etf",
        }

    isins = sorted(security_master.keys())
    prices_wide, source_by_isin = fetch_all_prices(
        isins,
        security_master=security_master,
        return_sources=True,
        fx_cache_path=fx_cache_path,
    )
    prices_wide = prices_wide.copy()
    prices_wide["date"] = prices_wide["date"].astype(str)
    prices_long = prices_wide.melt(id_vars=["date"], var_name="isin", value_name="price").dropna()

    justetf_isins = sorted([isin for isin in isins if _is_etf_like(security_master.get(isin, {}))])
    yfinance_isins = [i for i, src in source_by_isin.items() if src == "yfinance"]

    justetf_report: Dict[str, object] = {}
    yfinance_report: Dict[str, object] = {}
    justetf_payload = (
        fetch_exposures(
            justetf_isins,
            cache_path=justetf_cache_path,
            report=justetf_report,
        )
        if justetf_isins
        else {}
    )
    yfinance_ticker_map = build_yfinance_ticker_map(yfinance_isins, security_master)
    yfinance_tickers = [
        yfinance_ticker_map.get(i) or (security_master.get(i, {}).get("ticker") or i)
        for i in yfinance_isins
    ]
    yfinance_payload = (
        fetch_stock_profiles(
            yfinance_tickers,
            cache_path=yfinance_cache_path,
            report=yfinance_report,
        )
        if yfinance_tickers
        else {}
    )

    # attach spot info to security_master
    for isin in justetf_isins:
        security_master.setdefault(isin, {})["justetf_exposure"] = justetf_payload.get(isin)
    for isin in yfinance_isins:
        ticker = source_by_isin.get(f"{isin}__yahoo_ticker") or yfinance_ticker_map.get(isin) or (
            security_master.get(isin, {}).get("ticker") or isin
        )
        security_master.setdefault(isin, {})["yfinance_ticker"] = ticker
        security_master.setdefault(isin, {})["yfinance_profile"] = yfinance_payload.get(ticker)

    return {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "isin_count": len(isins),
        },
        "security_master": security_master,
        "statements": statements,
        "openfigi_report": openfigi_report,
        "justetf_report": justetf_report,
        "yfinance_report": yfinance_report,
        "price_sources": source_by_isin,
        "prices": {
            "wide": prices_wide.to_dict(orient="list"),
            "long": prices_long.to_dict(orient="records"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified datalake.")
    parser.add_argument(
        "--statements-dir",
        default=DEFAULT_STATEMENTS_DIR,
        help="Directory containing account statement PDFs.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Path to write JSON datalake output.",
    )
    args = parser.parse_args()

    datalake = build_datalake(args.statements_dir)

    global DATALAKE
    DATALAKE = datalake

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(datalake, indent=2, ensure_ascii=True))
    else:
        print(datalake)


if __name__ == "__main__":
    main()
