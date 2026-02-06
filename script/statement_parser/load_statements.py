#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load statement PDFs and expose the parsed dictionary for exploration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

STATEMENT_PARSER_DIR = Path(__file__).resolve().parent
if str(STATEMENT_PARSER_DIR) not in sys.path:
    sys.path.insert(0, str(STATEMENT_PARSER_DIR))

from parse_statements import merge_results, parse_pdf  # noqa: E402

DEFAULT_STATEMENTS_DIR = "/Users/andreadesogus/Desktop/portfolio_analytics/data/statements"


def load_statements(statements_dir: str = DEFAULT_STATEMENTS_DIR) -> Dict[str, dict]:
    statements_path = Path(statements_dir)
    pdfs = sorted(statements_path.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {statements_path}")

    combined: Dict[str, dict] = {}
    for pdf_path in pdfs:
        parsed = parse_pdf(pdf_path)
        combined = merge_results(combined, parsed)

    return combined


if __name__ == "__main__":
    data = load_statements()
    print(f"Loaded {len(data)} ISIN entries")
