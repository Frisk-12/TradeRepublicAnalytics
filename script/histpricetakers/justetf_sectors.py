#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:03:01 2026

@author: andreadesogus
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlencode, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE = "https://www.justetf.com"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def _to_float_pct(s: str) -> float | None:
    """Converte '80,40%' -> 80.40 (float)."""
    if s is None:
        return None
    s = s.strip()
    s = s.replace("%", "").replace(" ", "")
    s = s.replace(".", "")     # separatore migliaia eventuale
    s = s.replace(",", ".")    # decimale italiano
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _extract_security_id_from_profile_url(profile_url: str | None) -> str | None:
    """
    Esempio: https://www.justetf.com/it/stock-profiles/US88160R1014 -> US88160R1014
    """
    if not profile_url:
        return None
    m = re.search(r"/stock-profiles/([A-Z0-9]{8,20})", profile_url)
    return m.group(1) if m else None


@dataclass
class JustEtfExposureResult:
    isin: str
    url: str
    meta: dict
    top10: pd.DataFrame        # holding_name, weight_pct, profile_url, security_id
    countries: pd.DataFrame    # country, weight_pct
    sectors: pd.DataFrame      # sector, weight_pct


class JustEtfScraper:
    def __init__(self, locale: str = "it", timeout: int = 30) -> None:
        self.locale = locale
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": UA,
            "Accept-Language": "it-IT,it;q=0.9,en;q=0.8" if locale == "it" else "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": BASE,
        })

    def _build_etf_profile_url(self, isin: str) -> str:
        qs = urlencode({"isin": isin})
        # NOTA: niente #esposizione (anchor client-side)
        return f"{BASE}/{self.locale}/etf-profile.html?{qs}"

    def _fetch_html(self, url: str) -> str:
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.text

    @staticmethod
    def _parse_top10(soup: BeautifulSoup) -> tuple[pd.DataFrame, dict]:
        table = soup.select_one('table[data-testid="etf-holdings_top-holdings_table"]')
        if not table:
            empty = pd.DataFrame(columns=["holding_name", "weight_pct", "profile_url", "security_id"])
            return empty, {}

        rows = []
        for tr in table.select('tr[data-testid="etf-holdings_top-holdings_row"]'):
            a = tr.select_one('a[data-testid="tl_etf-holdings_top-holdings_link_name"]')
            name = a.get_text(strip=True) if a else None
            href = a.get("href") if a else None
            profile_url = urljoin(BASE, href) if href else None

            txt = tr.get_text(" ", strip=True)
            m = re.findall(r"\d{1,3}(?:[.,]\d+)?\s*%", txt)
            weight = _to_float_pct(m[-1]) if m else None

            if name:
                rows.append((name, weight, profile_url, _extract_security_id_from_profile_url(profile_url)))

        df = pd.DataFrame(rows, columns=["holding_name", "weight_pct", "profile_url", "security_id"])

        # META: "Peso delle prime 10 partecipazioni 51,59% su 43"
        container = soup.select_one('div[data-testid="etf-holdings_top-holdings_container"]')
        meta = {}
        if container:
            t = container.get_text(" ", strip=True)

            m_w = re.search(r"Peso delle prime 10 partecipazioni\s+(\d{1,3}(?:[.,]\d+)?\s*%)", t)
            meta["top10_total_weight_pct"] = _to_float_pct(m_w.group(1)) if m_w else None

            m_n = re.search(r"\bsu\s+(\d+)\b", t)
            meta["n_holdings"] = int(m_n.group(1)) if m_n else None

        return df, meta

    @staticmethod
    def _parse_simple_table(
        soup: BeautifulSoup,
        table_testid: str,
        row_testid: str,
        name_testid: str,
        pct_testid: str,
        out_name: str
    ) -> pd.DataFrame:
        table = soup.select_one(f'table[data-testid="{table_testid}"]')
        if not table:
            return pd.DataFrame(columns=[out_name, "weight_pct"])

        rows = []
        for tr in table.select(f'tr[data-testid="{row_testid}"]'):
            name_el = tr.select_one(f'[data-testid="{name_testid}"]')
            pct_el = tr.select_one(f'[data-testid="{pct_testid}"]')

            name = name_el.get_text(strip=True) if name_el else None
            pct = _to_float_pct(pct_el.get_text(strip=True)) if pct_el else None

            if name:
                rows.append((name, pct))

        return pd.DataFrame(rows, columns=[out_name, "weight_pct"])

    def scrape_exposure_by_isin(self, isin: str) -> JustEtfExposureResult:
        """
        Input: ISIN
        Output: top10 holdings + countries + sectors + meta
        """
        isin = isin.strip().upper()
        url = self._build_etf_profile_url(isin)

        html = self._fetch_html(url)
        soup = BeautifulSoup(html, "lxml")

        top10_df, meta = self._parse_top10(soup)

        countries_df = self._parse_simple_table(
            soup,
            table_testid="etf-holdings_countries_table",
            row_testid="etf-holdings_countries_row",
            name_testid="tl_etf-holdings_countries_value_name",
            pct_testid="tl_etf-holdings_countries_value_percentage",
            out_name="country",
        )

        sectors_df = self._parse_simple_table(
            soup,
            table_testid="etf-holdings_sectors_table",
            row_testid="etf-holdings_sectors_row",
            name_testid="tl_etf-holdings_sectors_value_name",
            pct_testid="tl_etf-holdings_sectors_value_percentage",
            out_name="sector",
        )

        return JustEtfExposureResult(
            isin=isin,
            url=url,
            meta=meta,
            top10=top10_df,
            countries=countries_df,
            sectors=sectors_df,
        )


def exposure_to_dict(res: JustEtfExposureResult) -> Dict[str, object]:
    return {
        "isin": res.isin,
        "url": res.url,
        "as_of": datetime.utcnow().isoformat() + "Z",
        "meta": res.meta,
        "top10": res.top10.to_dict(orient="records"),
        "countries": res.countries.to_dict(orient="records"),
        "sectors": res.sectors.to_dict(orient="records"),
    }


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


def fetch_exposures(
    isins: list[str],
    *,
    locale: str = "it",
    cache_path: Optional[str] = None,
    report: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, object]]:
    scraper = JustEtfScraper(locale=locale)
    out: Dict[str, Dict[str, object]] = {}
    cache = _load_cache(cache_path) if cache_path else {}
    for isin in isins:
        isin_norm = isin.strip().upper()
        if cache_path and isin_norm in cache:
            out[isin_norm] = cache[isin_norm]
            continue
        try:
            res = scraper.scrape_exposure_by_isin(isin_norm)
            payload = exposure_to_dict(res)
            payload["status"] = "matched" if payload["top10"] or payload["countries"] or payload["sectors"] else "not_found"
        except Exception as exc:
            payload = {
                "isin": isin_norm,
                "as_of": datetime.utcnow().isoformat() + "Z",
                "status": "error",
                "error": str(exc),
                "top10": [],
                "countries": [],
                "sectors": [],
                "meta": {},
            }
            if report is not None:
                report.setdefault("errors", []).append({"isin": isin_norm, "error": str(exc)})
        out[isin_norm] = payload
        if cache_path:
            cache[isin_norm] = payload
            _save_cache(cache_path, cache)
        if report is not None:
            report.setdefault("status_counts", {})
            st = payload.get("status", "unknown")
            report["status_counts"][st] = report["status_counts"].get(st, 0) + 1
    return out


# ESEMPIO USO
if __name__ == "__main__":
    scraper = JustEtfScraper(locale="it")
    res = scraper.scrape_exposure_by_isin("IE0003A512E4")

    print(res.isin, res.url)
    print("META:", res.meta)
    print("\nTOP10:\n", res.top10)
    print("\nCOUNTRIES:\n", res.countries)
    print("\nSECTORS:\n", res.sectors)
