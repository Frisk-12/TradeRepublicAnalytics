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
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PDFTOTEXT = os.getenv("PDFTOTEXT_BIN", "pdftotext")
_HASH_SEED = "account_statement_v1"

MONTHS = {
    "gen": "01",
    "gennaio": "01",
    "janv": "01",
    "janvier": "01",
    "feb": "02",
    "febbraio": "02",
    "fev": "02",
    "fevr": "02",
    "fevrier": "02",
    "mar": "03",
    "marzo": "03",
    "mars": "03",
    "apr": "04",
    "aprile": "04",
    "avr": "04",
    "avril": "04",
    "mag": "05",
    "maggio": "05",
    "mai": "05",
    "giu": "06",
    "giugno": "06",
    "juin": "06",
    "lug": "07",
    "luglio": "07",
    "juil": "07",
    "juillet": "07",
    "ago": "08",
    "agosto": "08",
    "aou": "08",
    "aout": "08",
    "set": "09",
    "settembre": "09",
    "sept": "09",
    "septembre": "09",
    "ott": "10",
    "ottobre": "10",
    "octobre": "10",
    "nov": "11",
    "novembre": "11",
    "dic": "12",
    "dicembre": "12",
    "decembre": "12",
}

TX_SECTION_MARKERS = (
    "TRANSAZIONI SUL CONTO",
    "TRANSACTIONS DU COMPTE",
    "MOUVEMENTS DU COMPTE",
    "OPERATIONS SUR LE COMPTE",
    "OPERATIONS DU COMPTE",
)

TX_END_MARKERS = (
    "PANORAMICA DEL SALDO",
    "BALANCE OVERVIEW",
    "APERÇU DU SOLDE",
    "NOTE SULL'ESTRATTO CONTO",
    "NOTES ON THE ACCOUNT STATEMENT",
    "NOTES SUR LE RELEVE DE COMPTE",
    "CONTI FIDUCIARI",
    "FONDI DEL MERCATO MONETARIO",
)

SUMMARY_START_MARKERS = (
    "ESTRATTO CONTO RIASSUNTIVO",
    "RESUME DU COMPTE",
    "RELEVE DE COMPTE",
    "RELEVÉ DE COMPTE",
    "SYNTHÈSE DU COMPTE",
)

TYPE_ALIASES = {
    "bonifico": "Bonifico",
    "transfert": "Bonifico",
    "virement": "Bonifico",
    "versement": "Bonifico",
    "commercio": "Commercio",
    "interessi": "Interessi",
    "interet": "Interessi",
    "interets": "Interessi",
    "rendimento": "Rendimento",
    "rendement": "Rendimento",
    "dividende": "Rendimento",
    "imposte": "Imposte",
    "impot": "Imposte",
    "impots": "Imposte",
    "premio": "Premio",
    "prime": "Premio",
}

AMOUNT_SUFFIX_RE = re.compile(
    r"(?P<num>-?(?:\d{1,3}(?:[.,\s\u00A0\u202F]\d{3})+|\d+)(?:[.,]\d{2}))\s*(?:€|EUR)(?!\w)",
    flags=re.IGNORECASE,
)
AMOUNT_PREFIX_RE = re.compile(
    r"(?:€|EUR)\s{0,1}(?P<num>-?(?:\d{1,3}(?:[.,\s\u00A0\u202F]\d{3})+|\d+)(?:[.,]\d{2}))\b",
    flags=re.IGNORECASE,
)
AMOUNT_GENERIC_RE = re.compile(
    r"(?P<num>-?(?:\d{1,3}(?:[.,\s\u00A0\u202F]\d{3})+|\d+)(?:[.,]\d{2}))",
    flags=re.IGNORECASE,
)


class StatementParseError(RuntimeError):
    """Raised when a statement cannot be parsed as a valid Trade Republic account statement."""


class UnsupportedStatementFormatError(StatementParseError):
    """Raised when the PDF does not match the expected Trade Republic account statement structure."""


def _format_unknown_tokens_suffix(tokens: List[str]) -> str:
    if not tokens:
        return ""
    cleaned = sorted({t for t in tokens if t})
    if not cleaned:
        return ""
    joined = ", ".join(cleaned[:25])
    return f" Unrecognized transaction tokens: {joined}."


def _resolve_pdftotext() -> str:
    candidate = (PDFTOTEXT or "").strip()
    if candidate:
        if Path(candidate).is_file():
            return candidate
        from_path = shutil.which(candidate)
        if from_path:
            return from_path
    # Fallback explicit locations often used in cloud runtimes.
    for fixed in ("/usr/bin/pdftotext", "/usr/local/bin/pdftotext"):
        if Path(fixed).is_file():
            return fixed
    fallback = shutil.which("pdftotext")
    if fallback:
        return fallback
    raise RuntimeError(
        "pdftotext binary not found. Install poppler-utils (Linux) "
        "or set environment variable PDFTOTEXT_BIN to the executable path."
    )


def _amount_to_float(value: str) -> float:
    s = str(value or "").strip()
    s = s.replace("\u00a0", " ").replace("\u202f", " ").replace(" ", "")
    neg = s.startswith("-")
    s = s.lstrip("+-")
    last_comma = s.rfind(",")
    last_dot = s.rfind(".")

    if last_comma == -1 and last_dot == -1:
        out = s
    else:
        decimal_sep = "," if last_comma > last_dot else "."
        thousands_sep = "." if decimal_sep == "," else ","
        s = s.replace(thousands_sep, "")
        idx = s.rfind(decimal_sep)
        out = s[:idx].replace(decimal_sep, "") + "." + s[idx + 1 :]

    if neg:
        out = "-" + out
    return float(out)


def _normalize_token(value: str) -> str:
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _month_to_num(token: str) -> Optional[str]:
    key = _normalize_token(token)
    return MONTHS.get(key) or MONTHS.get(key[:3])


def _parse_date_locale(value: str) -> Optional[str]:
    m = re.match(r"\s*(\d{1,2})\s+([A-Za-zÀ-ÿ\.]{3,12})\s+(\d{4})\s*$", value, flags=re.IGNORECASE)
    if not m:
        return None
    day, mon, year = m.group(1), m.group(2), m.group(3)
    month_num = _month_to_num(mon)
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
    else:
        m_side_fr = re.search(r"\b(Achat|Vente)\b", desc, flags=re.IGNORECASE)
        if m_side_fr:
            out["side"] = "Buy" if m_side_fr.group(1).lower().startswith("achat") else "Sell"

    # ISIN
    m_isin = re.search(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b", desc)
    if m_isin:
        out["isin"] = m_isin.group(1)

    # quantity
    m_qty = re.search(
        r"\b(quantity|quantite|quantité|quantita)[: ]\s*([0-9]+(?:[.,][0-9]+)?)\b",
        desc,
        flags=re.IGNORECASE,
    )
    if m_qty:
        try:
            out["quantity"] = float(m_qty.group(2).replace(",", "."))
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
    name = re.sub(r"^.*?\b(Achat|Vente)\b\s+", "", name, flags=re.IGNORECASE)
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
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if stderr:
            first_line = stderr.splitlines()[0]
            detail = first_line[:240]
        else:
            detail = f"exit status {exc.returncode}"
        raise RuntimeError(f"pdftotext execution failed: {detail}") from exc
    return result.stdout


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _parse_amounts(line: str) -> List[float]:
    found: List[Tuple[int, str]] = []
    for m in AMOUNT_SUFFIX_RE.finditer(line):
        found.append((m.start(), m.group("num")))
    for m in AMOUNT_PREFIX_RE.finditer(line):
        found.append((m.start(), m.group("num")))
    if len(found) < 2:
        for m in AMOUNT_GENERIC_RE.finditer(line):
            pos = m.start()
            # Avoid duplicates when a currency-specific regex already matched nearby.
            if any(abs(pos - p) <= 1 for p, _ in found):
                continue
            found.append((pos, m.group("num")))
    found.sort(key=lambda x: x[0])

    values: List[float] = []
    for _, raw in found:
        try:
            values.append(_amount_to_float(raw))
        except ValueError:
            continue
    return values


def _strip_amounts(line: str) -> str:
    out = AMOUNT_SUFFIX_RE.sub("", line)
    out = AMOUNT_PREFIX_RE.sub("", out)
    out = AMOUNT_GENERIC_RE.sub("", out)
    return out


def _line_has_footer(line: str) -> bool:
    l = line.strip().lower()
    l_norm = unicodedata.normalize("NFKD", l)
    l_norm = "".join(ch for ch in l_norm if not unicodedata.combining(ch))
    if (
        l.startswith("trade republic bank")
        or l_norm.startswith("generato il")
        or l_norm.startswith("generated on")
        or l_norm.startswith("genere le")
        or l_norm.startswith("pagina")
        or l_norm.startswith("page")
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


def _line_is_tx_end(line: str) -> bool:
    upper = (line or "").upper()
    return any(marker in upper for marker in TX_END_MARKERS)


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
        "dividende",
        "interessi",
        "interet",
        "interets",
        "rendimento",
        "rendement",
        "prime",
        "sell trade",
        "ordre de vente",
        "vente",
    )
    outflow_keywords = (
        "buy trade",
        "ordre d'achat",
        "achat",
        "impot",
        "impots",
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
    if (l.startswith("data") or l.startswith("date")) and ("descrizione" in l or "description" in l):
        return True
    return False


def _parse_date_range(text: str) -> Optional[Tuple[str, str]]:
    patterns = [
        r"(?:DATA|DATE|PERIODE|P[ÉE]RIODE)\s+(\d{1,2}\s+[A-Za-zÀ-ÿ\.]{3,12}\s+\d{4})\s*-\s*(\d{1,2}\s+[A-Za-zÀ-ÿ\.]{3,12}\s+\d{4})",
        r"(\d{1,2}\s+[A-Za-zÀ-ÿ\.]{3,12}\s+\d{4})\s*-\s*(\d{1,2}\s+[A-Za-zÀ-ÿ\.]{3,12}\s+\d{4})",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if not m:
            continue
        start = _parse_date_locale(m.group(1))
        end = _parse_date_locale(m.group(2))
        if start and end:
            return start, end
    return None


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
        if any(m in line.upper() for m in SUMMARY_START_MARKERS):
            in_summary = True
            continue
        if in_summary and any(m in line.upper() for m in TX_SECTION_MARKERS):
            break
        if not in_summary:
            continue
        if not line.strip():
            continue
        if line.strip().upper().startswith(("PRODOTTO", "PRODUCT", "PRODUIT")):
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


def _collect_unrecognized_transaction_tokens(text: str) -> List[str]:
    lines = text.splitlines()
    in_tx = False
    unknown: set[str] = set()

    known = set(TYPE_ALIASES.keys())
    skip_tokens = {
        "data",
        "date",
        "description",
        "descrizione",
        "solde",
        "saldo",
        "balance",
        "account",
        "compte",
        "type",
        "in",
        "out",
    }

    for raw in lines:
        line = raw.rstrip("\n")
        if any(m in line.upper() for m in TX_SECTION_MARKERS):
            in_tx = True
            continue
        if not in_tx:
            continue
        if _line_is_tx_end(line):
            break
        if _line_has_footer(line) or _line_is_header(line):
            continue
        if not line.strip():
            continue
        if line.strip().lower().startswith(("data", "date")):
            continue

        amounts = _parse_amounts(line)
        if not amounts:
            continue

        text_no_amounts = _normalize_spaces(_strip_amounts(line))
        # Strip leading date chunks to isolate the transaction-type token.
        s = text_no_amounts
        s = re.sub(r"^\s*\d{4}\b", "", s).strip()  # optional leading year
        had_day = False
        m_day = re.match(r"^\s*(\d{1,2})\b", s)
        if m_day:
            had_day = True
            s = s[m_day.end() :].strip()
        if had_day:
            m_mon = re.match(r"^\s*([A-Za-zÀ-ÿ\.]{3,12})\b", s)
            if m_mon and _month_to_num(m_mon.group(1)):
                s = s[m_mon.end() :].strip()
            s = re.sub(r"^\s*\d{4}\b", "", s).strip()  # optional year after day+month
        text_no_amounts = s
        if not text_no_amounts:
            continue
        first = text_no_amounts.split(" ", 1)[0]
        token = _normalize_token(first)
        if not token or token.isdigit() or token in skip_tokens or token in known:
            continue
        # Ignore likely day/month artifacts.
        if token in MONTHS or _month_to_num(token):
            continue
        # Keep only alphabetic-ish short words likely to be transaction types.
        if len(token) <= 18 and re.fullmatch(r"[a-z0-9]+", token):
            unknown.add(token)

    return sorted(unknown)


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
        if any(m in line.upper() for m in TX_SECTION_MARKERS):
            in_tx = True
            continue
        if not in_tx:
            continue
        if _line_is_tx_end(line):
            break
        if _line_has_footer(line) or _line_is_header(line):
            continue
        if line.strip().lower().startswith(("data", "date")):
            continue
        if not line.strip():
            continue

        # year-only line
        year_only = re.fullmatch(r"\s*(\d{4})\s*", line)
        if year_only:
            current_year = year_only.group(1)
            if pending_day and pending_month and transactions:
                month_num = _month_to_num(pending_month)
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
                month_num = _month_to_num(pending_month)
                if month_num:
                    transactions[-1]["date"] = f"{current_year}-{month_num}-{int(pending_day):02d}"
                pending_day = None
                pending_month = None

        # detect full date (day + month + year) at line start
        dmy = re.match(r"\s*(\d{1,2})\s+([A-Za-zÀ-ÿ\.]{3,12})\s+(\d{4})\b", line, flags=re.IGNORECASE)
        if dmy:
            pending_day = dmy.group(1)
            pending_month = dmy.group(2)
            current_year = dmy.group(3)
            line = line[dmy.end() :].strip()

        # detect day + month (possibly at line start)
        dm = re.match(r"\s*(\d{1,2})\s+([A-Za-zÀ-ÿ\.]{3,12})\b", line, flags=re.IGNORECASE)
        if dm:
            pending_day = dm.group(1)
            pending_month = dm.group(2)
            line = line[dm.end() :].strip()
            # Some statements place the year right after day+month on same line.
            ym = re.match(r"\s*(\d{4})\b", line)
            if ym:
                current_year = ym.group(1)
                line = line[ym.end() :].strip()

        # detect day only on line
        day_only = re.fullmatch(r"\s*(\d{1,2})\s*", line)
        if day_only:
            pending_day = day_only.group(1)
            continue

        # detect month at line start (may include other text)
        month_start = re.match(r"\s*([A-Za-zÀ-ÿ\.]{3,12})\b", line, flags=re.IGNORECASE)
        if month_start and pending_day and _month_to_num(month_start.group(1)):
            pending_month = month_start.group(1)
            line = line[month_start.end() :].strip()

        amounts = _parse_amounts(line)
        text_no_amounts = _normalize_spaces(_strip_amounts(line))

        # extract type if present as first token
        parts = text_no_amounts.split(" ", 1)
        typ = None
        desc = text_no_amounts
        if parts:
            normalized = _normalize_token(parts[0])
            canonical = TYPE_ALIASES.get(normalized)
            if canonical:
                typ = canonical
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
                txn["type"] = typ
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
                month_num = _month_to_num(pending_month)
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
                txn["type"] = typ
            if desc:
                if txn["description"]:
                    txn["description"] = _normalize_spaces(f"{txn['description']} {desc}")
                else:
                    txn["description"] = desc

    if transactions and transactions[-1].get("date") is None and pending_day and pending_month and current_year:
        month_num = _month_to_num(pending_month)
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
    unknown_tokens = _collect_unrecognized_transaction_tokens(text)
    suffix = _format_unknown_tokens_suffix(unknown_tokens)
    text_up = (text or "").upper()
    if "TRADE REPUBLIC" not in text_up:
        raise UnsupportedStatementFormatError(
            "Invalid statement format: missing Trade Republic markers. "
            "Upload a Trade Republic Account Statement PDF."
            + suffix
        )
    if not any(marker in text_up for marker in TX_SECTION_MARKERS):
        raise UnsupportedStatementFormatError(
            "Invalid statement format: account-transactions section not found. "
            "Upload a full Trade Republic Account Statement PDF."
            + suffix
        )
    if not transactions:
        raise UnsupportedStatementFormatError(
            "Invalid statement format: no account transactions were parsed. "
            "The file may not be a Trade Republic Account Statement."
            + suffix
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
            + suffix
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
