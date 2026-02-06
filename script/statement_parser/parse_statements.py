#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse Trade Republic "Statement of securities account" PDFs.

Output: dict keyed by ISIN, with per-ISIN subdicts and entries per statement date.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

STATEMENT_DATE_PATTERNS = [
    re.compile(r"ESTRATTO CONTO TITOLI\s+al\s+(\d{2}\.\d{2}\.\d{4})", re.IGNORECASE),
    re.compile(r"Elenco delle tu[oe] azioni .*?\s+al\s+(\d{2}\.\d{2}\.\d{4})", re.IGNORECASE),
]

POSITION_START_RE = re.compile(r"^\s*([0-9][0-9\.,]*)\s+(Pz\.|Nominale)\s+(.*)$")
DATE_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")


PDFTOTEXT = os.getenv("PDFTOTEXT_BIN", "pdftotext")


def _resolve_pdftotext() -> str:
    candidate = (PDFTOTEXT or "").strip()
    if candidate:
        if Path(candidate).is_file():
            return candidate
        from_path = shutil.which(candidate)
        if from_path:
            return from_path
    fallback = shutil.which("pdftotext")
    if fallback:
        return fallback
    raise RuntimeError(
        "pdftotext binary not found. Install poppler-utils (Linux) "
        "or set PDFTOTEXT_BIN to the executable path."
    )


def extract_text(pdf_path: Path) -> str:
    bin_path = _resolve_pdftotext()
    try:
        result = subprocess.run(
            [bin_path, "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"pdftotext not found at {bin_path}") from exc
    return result.stdout


def parse_statement_date(text: str) -> Optional[str]:
    for pattern in STATEMENT_DATE_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1)
    return None


def split_name_and_price(rest: str) -> (str, Optional[str]):
    m = re.search(r"(.*?)([0-9][0-9\.,]*)\s+([0-9][0-9\.,]*)\s*$", rest)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return rest.strip(), None


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_positions(text: str) -> List[Dict[str, Optional[str]]]:
    lines = text.splitlines()
    positions: List[Dict[str, Optional[str]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = POSITION_START_RE.match(line)
        if not m:
            i += 1
            continue

        quantity = m.group(1).strip()
        unit = m.group(2).strip()
        rest = m.group(3)
        name_part, price_per_unit = split_name_and_price(rest)
        name_lines: List[str] = []
        if name_part:
            name_lines.append(name_part)

        isin: Optional[str] = None
        j = i + 1
        while j < len(lines):
            l = lines[j].strip()
            if not l:
                j += 1
                continue
            if "ISIN:" in l:
                isin = l.split("ISIN:", 1)[1].strip()
                break
            if DATE_RE.search(l):
                l = DATE_RE.sub("", l).strip()
            if l and not l.lower().startswith(("conto titoli", "luogo di custodia", "numero di posizioni")):
                name_lines.append(l)
            j += 1

        if isin:
            name = normalize_spaces(" ".join(name_lines))
            positions.append(
                {
                    "isin": isin,
                    "name": name,
                    "quantity": quantity,
                    "unit": unit,
                    "price_per_unit": price_per_unit,
                }
            )
            i = j + 1
        else:
            i += 1

    return positions


def aggregate_positions(positions: List[Dict[str, Optional[str]]], statement_date: Optional[str]) -> Dict[str, Dict]:
    result: Dict[str, Dict] = {}
    for pos in positions:
        isin = pos["isin"]
        if not isin:
            continue
        entry = {
            "statement_date": statement_date,
            "quantity": pos.get("quantity"),
            "unit": pos.get("unit"),
            "price_per_unit": pos.get("price_per_unit"),
        }
        if isin not in result:
            result[isin] = {
                "name": pos.get("name"),
                "entries": [entry],
            }
        else:
            if pos.get("name") and result[isin].get("name") != pos.get("name"):
                variants = result[isin].setdefault("name_variants", [])
                if pos.get("name") not in variants:
                    variants.append(pos.get("name"))
            result[isin]["entries"].append(entry)
    return result


def parse_pdf(pdf_path: Path) -> Dict[str, Dict]:
    text = extract_text(pdf_path)
    statement_date = parse_statement_date(text)
    positions = parse_positions(text)
    return aggregate_positions(positions, statement_date)


def merge_results(base: Dict[str, Dict], incoming: Dict[str, Dict]) -> Dict[str, Dict]:
    for isin, payload in incoming.items():
        if isin not in base:
            base[isin] = payload
            continue
        base[isin]["entries"].extend(payload.get("entries", []))
        name = payload.get("name")
        if name and base[isin].get("name") != name:
            variants = base[isin].setdefault("name_variants", [])
            if name not in variants:
                variants.append(name)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse portfolio statement PDFs.")
    parser.add_argument(
        "--statements-dir",
        default="/Users/andreadesogus/Desktop/portfolio_analytics/data/statements",
        help="Directory containing PDF statements.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write JSON output.",
    )
    args = parser.parse_args()

    statements_dir = Path(args.statements_dir)
    pdfs = sorted(statements_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {statements_dir}")

    combined: Dict[str, Dict] = {}
    for pdf_path in pdfs:
        parsed = parse_pdf(pdf_path)
        combined = merge_results(combined, parsed)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=True))
    else:
        print(combined)


if __name__ == "__main__":
    main()
