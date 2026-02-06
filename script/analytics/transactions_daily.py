#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate transactions by day, preserving type and attaching trade + security metadata.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_DATALAKE_JSON = "/Users/andreadesogus/Desktop/portfolio_analytics/script/datalake/datalake.json"


def _load_datalake(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text())


def _iter_transactions(datalake: Dict[str, object]) -> Iterable[Dict[str, object]]:
    statements = datalake.get("statements", {})
    for payload in statements.values():
        for txn in payload.get("transactions", []):
            yield txn


def _agg_key(txn: Dict[str, object]) -> Tuple:
    trade = txn.get("trade") or {}
    return (
        txn.get("date"),
        txn.get("type"),
        txn.get("description"),
        trade.get("isin"),
        trade.get("side"),
        trade.get("quantity"),
        trade.get("share_class"),
    )


def aggregate_transactions(datalake: Dict[str, object]) -> List[Dict[str, object]]:
    security_master = datalake.get("security_master", {})
    bucket: Dict[Tuple, Dict[str, object]] = {}

    for txn in _iter_transactions(datalake):
        key = _agg_key(txn)
        if key not in bucket:
            trade = txn.get("trade") or {}
            isin = trade.get("isin")
            security = security_master.get(isin) if isin else None
            bucket[key] = {
                "date": txn.get("date"),
                "type": txn.get("type"),
                "description": txn.get("description"),
                "in_entrata": float(txn.get("in_entrata") or 0.0),
                "in_uscita": float(txn.get("in_uscita") or 0.0),
                "cash_flow": float(txn.get("cash_flow") or 0.0),
                "n": 1,
                "trade": trade if trade else None,
                "security": security,
                "tx_ids": [txn.get("id")],
            }
        else:
            entry = bucket[key]
            entry["in_entrata"] += float(txn.get("in_entrata") or 0.0)
            entry["in_uscita"] += float(txn.get("in_uscita") or 0.0)
            entry["cash_flow"] += float(txn.get("cash_flow") or 0.0)
            entry["n"] += 1
            entry["tx_ids"].append(txn.get("id"))

    out = list(bucket.values())
    out.sort(key=lambda x: (x.get("date") or "", x.get("type") or "", x.get("description") or ""))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate transactions by day.")
    parser.add_argument(
        "--datalake-json",
        default=DEFAULT_DATALAKE_JSON,
        help="Path to datalake JSON.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write aggregated output.",
    )
    args = parser.parse_args()

    datalake = _load_datalake(args.datalake_json)
    aggregated = aggregate_transactions(datalake)

    global TRANSACTIONS_DAILY
    TRANSACTIONS_DAILY = aggregated

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(aggregated, indent=2, ensure_ascii=True))
    else:
        print(aggregated)


if __name__ == "__main__":
    main()
