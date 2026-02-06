#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse Trade Republic account statement PDFs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PDFTOTEXT = os.getenv("PDFTOTEXT_BIN", "pdftotext")
_HASH_SEED = "account_statement_v1"

MONTHS = {
    "gen": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "mag": "05",
    "giu": "06",
    "lug": "07",
    "ago": "08",
    "set": "09",
    "ott": "10",
    "nov": "11",
    "dic": "12",
}

KNOWN_TYPES = {
    "bonifico",
    "commercio",
    "interessi",
    "rendimento",
    "imposte",
    "premio",
}


class StatementParseError(RuntimeError):
    """Raised when a statement cannot be parsed as a valid Trade Republic account statement."""


class UnsupportedStatementFormatError(StatementParseError):
    """Raised when the PDF does not match the expected Trade Republic account statement structure."""


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
        "or set environment variable PDFTOTEXT_BIN to the executable path."
    )


def _amount_to_float(value: str) -> float:
    return float(value.replace(".", "").replace(",", "."))


def _parse_date_italian(value: str) -> Optional[str]:
    m = re.match(r"\s*(\d{1,2})\s+([a-z]{3})\s+(\d{4})\s*$", value, flags=re.IGNORECASE)
    if not m:
        return None
    day, mon, year = m.group(1), m.group(2).lower(), m.group(3)
    month_num = MONTHS.get(mon)
    if not month_num:
        return None
    return f"{year}-{month_num}-{int(day):02d}"


def _parse_trade_description(description: str) -> Dict[str, Optional[object]]:
    """
    Parse 'Commercio' descriptions, extracting:
    - side: Buy/Sell
    - isin
    - instrument_name
    - quantity (float)
    - share_class: Acc/Dist/Accumulative if present
    """
    out: Dict[str, Optional[object]] = {
        "side": None,
        "isin": None,
        "instrument_name": None,
        "quantity": None,
        "share_class": None,
    }

    desc = _normalize_spaces(description)

    # side
    m_side = re.search(r"\b(Buy|Sell)\s+trade\b", desc, flags=re.IGNORECASE)
    if m_side:
        out["side"] = m_side.group(1).capitalize()

    # ISIN
    m_isin = re.search(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b", desc)
    if m_isin:
        out["isin"] = m_isin.group(1)

    # quantity
    m_qty = re.search(r"\bquantity:\s*([0-9]+(?:[.,][0-9]+)?)\b", desc, flags=re.IGNORECASE)
    if m_qty:
        try:
            out["quantity"] = float(m_qty.group(1).replace(",", "."))
        except ValueError:
            out["quantity"] = None

    # share class
    share_markers = [
        "acc",
        "acc.",
        "accumulating",
        "accumulative",
        "dist",
        "dist.",
        "distributing",
        "distribution",
        "hedged",
    ]
    for marker in share_markers:
        if re.search(rf"\b{re.escape(marker)}\b", desc, flags=re.IGNORECASE):
            out["share_class"] = marker.replace(".", "").capitalize()
            break
    if out["share_class"] in {"Acc", "Accumulating"}:
        out["share_class"] = "Acc"
    if out["share_class"] in {"Dist", "Distributing"}:
        out["share_class"] = "Dist"

    # instrument name (best-effort)
    name = desc
    name = re.sub(r"^.*?\b(Buy|Sell)\s+trade\b\s+", "", name, flags=re.IGNORECASE)
    if out["isin"]:
        name = name.replace(out["isin"], "").strip()
    name = re.sub(r",?\s*quantity:\s*[0-9]+(?:[.,][0-9]+)?", "", name, flags=re.IGNORECASE).strip()
    out["instrument_name"] = name if name else None

    return out


def _transaction_id(
    filename: str,
    date: Optional[str],
    typ: Optional[str],
    description: Optional[str],
    saldo: Optional[float],
    in_entrata: Optional[float],
    in_uscita: Optional[float],
) -> str:
    import hashlib

    payload = "|".join(
        [
            _HASH_SEED,
            filename or "",
            date or "",
            typ or "",
            description or "",
            f"{saldo:.4f}" if saldo is not None else "",
            f"{in_entrata:.4f}" if in_entrata is not None else "",
            f"{in_uscita:.4f}" if in_uscita is not None else "",
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


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


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _parse_amounts(line: str) -> List[float]:
    values: List[float] = []
    for m in re.finditer(r"([0-9\.]+,[0-9]{2})\s*€", line):
        raw = m.group(1)
        values.append(_amount_to_float(raw))
    return values


def _strip_amounts(line: str) -> str:
    return re.sub(r"[0-9\.]+,[0-9]{2}\s*€", "", line)


def _line_has_footer(line: str) -> bool:
    l = line.strip().lower()
    if (
        l.startswith("trade republic bank")
        or l.startswith("generato il")
        or l.startswith("pagina")
    ):
        return True
    footer_markers = (
        "www.traderepublic.it",
        "sede centrale",
        "direttore generale",
        "registro commerciale",
        "camera di commercio",
        "brunnenstrasse",
        "p. iva",
        "spaces gae aulenti",
        "andreas torner",
        "gernot mittendorfer",
        "christian hecker",
        "thomas pischke",
    )
    return any(marker in l for marker in footer_markers)


def _infer_single_amount_direction(txn_type: Optional[str], description: str) -> str:
    typ = (txn_type or "").strip().lower()
    desc = (description or "").strip().lower()

    if typ in {"bonifico", "interessi", "rendimento", "premio"}:
        return "in"
    if typ in {"imposte"}:
        return "out"
    if typ == "commercio":
        if "sell trade" in desc:
            return "in"
        if "buy trade" in desc or "ordine d'acquisto" in desc:
            return "out"

    inflow_keywords = (
        "incoming",
        "dividend",
        "interest payment",
        "interessi",
        "rendimento",
        "bonus",
        "refund",
        "warrant exercise",
        "sell trade",
    )
    outflow_keywords = (
        "buy trade",
        "tax",
        "stamp duty",
        "imposte",
        "ordine d'acquisto",
    )
    if any(k in desc for k in inflow_keywords):
        return "in"
    if any(k in desc for k in outflow_keywords):
        return "out"
    return "out"


def _line_is_header(line: str) -> bool:
    l = line.strip().lower()
    if not l:
        return False
    header_markers = (
        "trade republic",
        "branch italy",
        "spaces gae aulenti",
        "piazza gae aulenti",
        "torre b",
    )
    if any(marker in l for marker in header_markers):
        return True
    if l.startswith("data") and "descrizione" in l:
        return True
    return False


def _parse_date_range(text: str) -> Optional[Tuple[str, str]]:
    m = re.search(
        r"DATA\s+(\d{2}\s+[a-z]{3}\s+\d{4})\s+-\s+(\d{2}\s+[a-z]{3}\s+\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    start = _parse_date_italian(m.group(1))
    end = _parse_date_italian(m.group(2))
    return start, end


def _parse_account_info(text: str) -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {
        "iban": None,
        "bic": None,
    }
    iban = re.search(r"IBAN\s+([A-Z0-9]+)", text)
    if iban:
        info["iban"] = iban.group(1)
    bic = re.search(r"BIC\s+([A-Z0-9]+)", text)
    if bic:
        info["bic"] = bic.group(1)
    return info


def _parse_summary_block(text: str) -> List[Dict[str, Optional[float]]]:
    lines = text.splitlines()
    summary: List[Dict[str, str]] = []
    in_summary = False
    for line in lines:
        if "ESTRATTO CONTO RIASSUNTIVO" in line:
            in_summary = True
            continue
        if in_summary and "TRANSAZIONI SUL CONTO" in line:
            break
        if not in_summary:
            continue
        if not line.strip():
            continue
        if line.strip().startswith("PRODOTTO"):
            continue
        amounts = _parse_amounts(line)
        if len(amounts) >= 2:
            product = _normalize_spaces(_strip_amounts(line))
            summary.append(
                {
                    "product": product,
                    "saldo_iniziale": amounts[0] if len(amounts) > 0 else None,
                    "in_entrata": amounts[1] if len(amounts) > 1 else None,
                    "in_uscita": amounts[2] if len(amounts) > 2 else None,
                    "saldo_finale": amounts[3] if len(amounts) > 3 else None,
                }
            )
    return summary


def _parse_transactions(text: str) -> List[Dict[str, Optional[object]]]:
    lines = text.splitlines()
    transactions: List[Dict[str, Optional[str]]] = []
    in_tx = False
    pending_day: Optional[str] = None
    pending_month: Optional[str] = None
    current_year: Optional[str] = None
    current: Optional[Dict[str, Optional[str]]] = None
    seq = 0

    def ensure_current() -> Dict[str, Optional[str]]:
        nonlocal current
        if current is None:
            current = {
                "date": None,
                "type": None,
                "description": None,
                "in_entrata": None,
                "in_uscita": None,
                "saldo": None,
            }
        return current

    for raw in lines:
        line = raw.rstrip("\n")
        if "TRANSAZIONI SUL CONTO" in line:
            in_tx = True
            continue
        if not in_tx:
            continue
        if _line_has_footer(line) or _line_is_header(line):
            continue
        if line.strip().startswith("DATA"):
            continue
        if not line.strip():
            continue

        # year-only line
        year_only = re.fullmatch(r"\s*(\d{4})\s*", line)
        if year_only:
            current_year = year_only.group(1)
            if pending_day and pending_month and transactions:
                month_num = MONTHS.get(pending_month.lower())
                if month_num:
                    transactions[-1]["date"] = f"{current_year}-{month_num}-{int(pending_day):02d}"
                pending_day = None
                pending_month = None
            continue

        # year at start of line (may include other text)
        year_start = re.match(r"\s*(\d{4})\b", line)
        if year_start:
            current_year = year_start.group(1)
            line = line[year_start.end() :].strip()
            if pending_day and pending_month and transactions and transactions[-1].get("date") is None:
                month_num = MONTHS.get(pending_month.lower())
                if month_num:
                    transactions[-1]["date"] = f"{current_year}-{month_num}-{int(pending_day):02d}"
                pending_day = None
                pending_month = None

        # detect day + month (possibly at line start)
        dm = re.match(r"\s*(\d{1,2})\s+([a-z]{3})\b", line, flags=re.IGNORECASE)
        if dm:
            pending_day = dm.group(1)
            pending_month = dm.group(2)
            line = line[dm.end() :].strip()

        # detect day only on line
        day_only = re.fullmatch(r"\s*(\d{1,2})\s*", line)
        if day_only:
            pending_day = day_only.group(1)
            continue

        # detect month at line start (may include other text)
        month_start = re.match(r"\s*([a-z]{3})\b", line, flags=re.IGNORECASE)
        if month_start and pending_day and month_start.group(1).lower() in MONTHS:
            pending_month = month_start.group(1)
            line = line[month_start.end() :].strip()

        amounts = _parse_amounts(line)
        text_no_amounts = _normalize_spaces(_strip_amounts(line))

        # extract type if present as first token
        parts = text_no_amounts.split(" ", 1)
        typ = None
        desc = text_no_amounts
        if parts and parts[0].lower() in KNOWN_TYPES:
            typ = parts[0]
            desc = parts[1] if len(parts) > 1 else ""

        # continuation line after amounts: append to last transaction
        if not amounts and current is None and transactions and desc and typ is None and pending_day is None and pending_month is None:
            last = transactions[-1]
            if last.get("description"):
                last["description"] = _normalize_spaces(f"{last['description']} {desc}")
            else:
                last["description"] = desc
            continue

        if amounts:
            txn = ensure_current()
            if typ and not txn["type"]:
                txn["type"] = typ.capitalize()
            if desc:
                if txn["description"]:
                    txn["description"] = _normalize_spaces(f"{txn['description']} {desc}")
                else:
                    txn["description"] = desc

            # assign amounts
            balance = amounts[-1]
            in_amt = None
            out_amt = None
            if len(amounts) == 2:
                direction = _infer_single_amount_direction(txn.get("type"), txn.get("description") or "")
                if direction == "in":
                    in_amt = amounts[0]
                else:
                    out_amt = amounts[0]
            elif len(amounts) >= 3:
                in_amt = amounts[0]
                out_amt = amounts[1]

            txn["in_entrata"] = in_amt if in_amt is not None else 0.0
            txn["in_uscita"] = out_amt if out_amt is not None else 0.0
            txn["saldo"] = balance if balance is not None else 0.0
            txn["cash_flow"] = txn["in_entrata"] - txn["in_uscita"]
            seq += 1
            txn["seq"] = seq

            if txn.get("date") is None and pending_day and pending_month and current_year:
                month_num = MONTHS.get(pending_month.lower())
                if month_num:
                    txn["date"] = f"{current_year}-{month_num}-{int(pending_day):02d}"
                pending_day = None
                pending_month = None

            transactions.append(txn)
            current = None
        else:
            if not text_no_amounts:
                continue
            txn = ensure_current()
            if typ and not txn["type"]:
                txn["type"] = typ.capitalize()
            if desc:
                if txn["description"]:
                    txn["description"] = _normalize_spaces(f"{txn['description']} {desc}")
                else:
                    txn["description"] = desc

    if transactions and transactions[-1].get("date") is None and pending_day and pending_month and current_year:
        month_num = MONTHS.get(pending_month.lower())
        if month_num:
            transactions[-1]["date"] = f"{current_year}-{month_num}-{int(pending_day):02d}"

    # Ensure trade parsing for Commercio rows after all descriptions are finalized.
    for txn in transactions:
        if txn.get("type") == "Commercio" and txn.get("description"):
            txn["trade"] = _parse_trade_description(txn["description"])

    return transactions


def _validate_statement_structure(
    *,
    text: str,
    transactions: List[Dict[str, Optional[object]]],
) -> None:
    text_up = (text or "").upper()
    if "TRADE REPUBLIC" not in text_up:
        raise UnsupportedStatementFormatError(
            "Invalid statement format: missing Trade Republic markers. "
            "Upload a Trade Republic Account Statement PDF."
        )
    if "TRANSAZIONI SUL CONTO" not in text_up:
        raise UnsupportedStatementFormatError(
            "Invalid statement format: account-transactions section not found. "
            "Upload a full Trade Republic Account Statement PDF."
        )
    if not transactions:
        raise UnsupportedStatementFormatError(
            "Invalid statement format: no account transactions were parsed. "
            "The file may not be a Trade Republic Account Statement."
        )

    valid_rows = 0
    for txn in transactions:
        has_date = bool(txn.get("date"))
        has_type = bool(txn.get("type"))
        has_balance = txn.get("saldo") is not None
        if has_date and has_type and has_balance:
            valid_rows += 1
    if valid_rows == 0:
        raise UnsupportedStatementFormatError(
            "Invalid statement format: required transaction fields (date/type/balance) were not detected."
        )


def parse_account_statement(pdf_path: Path) -> Dict[str, object]:
    text = extract_text(pdf_path)
    date_range = _parse_date_range(text)
    info = _parse_account_info(text)
    summary = _parse_summary_block(text)
    transactions = _parse_transactions(text)
    _validate_statement_structure(text=text, transactions=transactions)
    for txn in transactions:
        txn["id"] = _transaction_id(
            pdf_path.name,
            txn.get("date"),
            txn.get("type"),
            txn.get("description"),
            txn.get("saldo"),
            txn.get("in_entrata"),
            txn.get("in_uscita"),
        )

    return {
        "file": pdf_path.name,
        "date_range": date_range,
        "account_info": info,
        "summary": summary,
        "transactions": transactions,
    }


def parse_all(statements_dir: str) -> Dict[str, Dict[str, object]]:
    base = Path(statements_dir)
    pdfs = sorted(base.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {base}")

    out: Dict[str, Dict[str, object]] = {}
    for pdf in pdfs:
        try:
            out[pdf.name] = parse_account_statement(pdf)
        except StatementParseError as exc:
            raise StatementParseError(f"{pdf.name}: {exc}") from exc
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Trade Republic account statements.")
    parser.add_argument(
        "--statements-dir",
        default="/Users/andreadesogus/Desktop/portfolio_analytics/data/account_statement",
        help="Directory containing account statement PDFs.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write JSON output.",
    )
    args = parser.parse_args()

    data = parse_all(args.statements_dir)

    global ACCOUNT_STATEMENTS_DATA
    ACCOUNT_STATEMENTS_DATA = data
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(data, indent=2, ensure_ascii=True))
    else:
        print(data)


if __name__ == "__main__":
    main()
