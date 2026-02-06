#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich parsed statement data with OpenFIGI metadata for each ISIN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SCRIPT_DIR = SCRIPT_DIR.parent
if str(REPO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPT_DIR))

from openfigi.openfigi_api import OpenFigiClient, DEFAULT_API_KEY  # noqa: E402
from statement_parser.load_statements import load_statements  # noqa: E402

DEFAULT_STATEMENTS_DIR = "/Users/andreadesogus/Desktop/portfolio_analytics/data/statements"


def enrich_statements(
    statements_dir: str = DEFAULT_STATEMENTS_DIR,
    api_key: str | None = None,
) -> Dict[str, dict]:
    data = load_statements(statements_dir)
    if not data:
        return data

    isins = sorted(data.keys())
    api_key = (api_key or os.getenv("OPENFIGI_API_KEY", "") or DEFAULT_API_KEY).strip()
    if not api_key:
        raise SystemExit("Missing OPENFIGI_API_KEY env var or --api-key.")

    client = OpenFigiClient(api_key=api_key)
    instruments = client.lookup_isins(isins)
    by_isin = {inst.isin: asdict(inst) for inst in instruments}

    for isin, payload in data.items():
        payload["openfigi"] = by_isin.get(isin, {"isin": isin})

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich statements with OpenFIGI metadata.")
    parser.add_argument(
        "--statements-dir",
        default=DEFAULT_STATEMENTS_DIR,
        help="Directory containing PDF statements.",
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
    args = parser.parse_args()

    enriched = enrich_statements(args.statements_dir, args.api_key)

    # keep a named variable available when running in IDEs like Spyder
    global ENRICHED_DATA
    ENRICHED_DATA = enriched

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=True))
    else:
        print(enriched)


if __name__ == "__main__":
    main()
