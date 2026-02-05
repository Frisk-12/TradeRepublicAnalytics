#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich account statement transactions with OpenFIGI metadata by ISIN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SCRIPT_DIR = SCRIPT_DIR.parent
if str(REPO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPT_DIR))

from openfigi.openfigi_api import DEFAULT_API_KEY, OpenFigiClient  # noqa: E402
from account_statements.parse_account_statements import parse_all  # noqa: E402

DEFAULT_STATEMENTS_DIR = "/Users/andreadesogus/Desktop/portfolio_analytics/data/account_statement"
DEFAULT_CACHE_PATH = "/Users/andreadesogus/Desktop/portfolio_analytics/script/statement_enricher/openfigi_cache.json"


def _collect_isins(data: Dict[str, Dict]) -> Set[str]:
    isins: Set[str] = set()
    for payload in data.values():
        for txn in payload.get("transactions", []):
            trade = txn.get("trade") or {}
            isin = trade.get("isin")
            if isin:
                isins.add(isin)
    return isins


def _derive_asset_fields(openfigi: Dict[str, object]) -> Dict[str, Optional[str]]:
    market_sector = (openfigi.get("market_sector") or "").lower()
    security_type = (openfigi.get("security_type") or "").lower()
    security_type2 = (openfigi.get("security_type2") or "").lower()
    name = (openfigi.get("name") or "").lower()
    ticker = (openfigi.get("ticker") or "").lower()

    instrument_type = None
    asset_class = None

    if any(k in name for k in ["warrant", "optionsschein", "certificate", "certific"]):
        instrument_type = "warrant"
        asset_class = "derivative"
    elif "turbo" in name or "knock" in name:
        instrument_type = "turbo"
        asset_class = "derivative"
    elif security_type in {"etp", "etn"} or "etf" in name:
        instrument_type = "etf"
        asset_class = "etf"
    elif market_sector == "equity" and security_type not in {"etp", "etn"}:
        instrument_type = "stock"
        asset_class = "equity"
    elif market_sector == "fixed income" or "bond" in name:
        instrument_type = "bond"
        asset_class = "bond"
    elif security_type2 == "mutual fund" or "fund" in name:
        instrument_type = "fund"
        asset_class = "fund"
    else:
        instrument_type = "other"
        asset_class = "other"

    return {"asset_class": asset_class, "instrument_type": instrument_type}


def _enrich_transactions(data: Dict[str, Dict]) -> None:
    for payload in data.values():
        for txn in payload.get("transactions", []):
            trade = txn.get("trade") or {}
            isin = trade.get("isin")
            if not isin:
                continue
            trade["security_ref"] = isin
            trade["isin"] = isin
            txn["trade"] = trade


def _quality_report(security_master: Dict[str, dict]) -> Dict[str, object]:
    report = {
        "total_isins": len(security_master),
        "status_counts": {},
        "ambiguous": [],
        "errors": [],
        "not_found": [],
    }
    for isin, payload in security_master.items():
        status = payload.get("status") or "not_found"
        report["status_counts"][status] = report["status_counts"].get(status, 0) + 1
        if status == "ambiguous":
            report["ambiguous"].append(isin)
        elif status == "error":
            report["errors"].append(isin)
        elif status == "not_found":
            report["not_found"].append(isin)
    return report


def _load_cache(cache_path: Optional[str]) -> Dict[str, dict]:
    if not cache_path:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache_path: Optional[str], cache: Dict[str, dict]) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=True))


def enrich_account_statements(
    statements_dir: str = DEFAULT_STATEMENTS_DIR,
    api_key: str | None = None,
    cache_path: Optional[str] = DEFAULT_CACHE_PATH,
) -> Dict[str, Dict]:
    data = parse_all(statements_dir)
    isins = sorted(_collect_isins(data))
    if not isins:
        return {"security_master": {}, "statements": data}

    api_key = (api_key or os.getenv("OPENFIGI_API_KEY", "") or DEFAULT_API_KEY).strip()
    if not api_key:
        raise SystemExit("Missing OPENFIGI_API_KEY env var or --api-key.")

    cache = _load_cache(cache_path)
    def _needs_refresh(payload: dict) -> bool:
        raw = payload.get("raw_best")
        if not isinstance(raw, dict):
            return True
        # We need candidates for ticker resolution; older cache entries lack them.
        return "candidates" not in raw

    missing = [isin for isin in isins if isin not in cache or _needs_refresh(cache.get(isin, {}))]

    if missing:
        client = OpenFigiClient(api_key=api_key)
        instruments = client.lookup_isins(missing)
        for inst in instruments:
            cache[inst.isin] = asdict(inst)
        _save_cache(cache_path, cache)

    security_master: Dict[str, dict] = {}
    for isin in isins:
        base = cache.get(isin, {"isin": isin})
        status = base.get("status")
        if not status:
            base["status"] = "not_found"
            base["candidates"] = 0
        derived = _derive_asset_fields(base)
        base.update(derived)
        security_master[isin] = base

    _enrich_transactions(data)
    return {
        "security_master": security_master,
        "statements": data,
        "openfigi_report": _quality_report(security_master),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich account statements with OpenFIGI metadata.")
    parser.add_argument(
        "--statements-dir",
        default=DEFAULT_STATEMENTS_DIR,
        help="Directory containing account statement PDFs.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenFIGI API key (otherwise uses OPENFIGI_API_KEY).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write JSON output.",
    )
    parser.add_argument(
        "--cache-path",
        default=DEFAULT_CACHE_PATH,
        help="Path to OpenFIGI cache JSON.",
    )
    args = parser.parse_args()

    enriched = enrich_account_statements(args.statements_dir, args.api_key, args.cache_path)

    global ENRICHED_ACCOUNT_STATEMENTS
    ENRICHED_ACCOUNT_STATEMENTS = enriched

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=True))
    else:
        print(enriched)


if __name__ == "__main__":
    main()
