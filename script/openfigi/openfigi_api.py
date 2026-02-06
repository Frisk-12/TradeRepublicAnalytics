#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 11:34:39 2026

@author: andreadesogus
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

import requests

_env_key = os.getenv("OPENFIGI_API_KEY")
DEFAULT_API_KEY = _env_key.strip() if _env_key and _env_key.strip() else ""


@dataclass(frozen=True)
class OpenFigiInstrument:
    isin: str
    figi: Optional[str] = None
    name: Optional[str] = None
    ticker: Optional[str] = None
    exch_code: Optional[str] = None
    mic_code: Optional[str] = None
    currency: Optional[str] = None
    market_sector: Optional[str] = None
    security_type: Optional[str] = None
    security_type2: Optional[str] = None
    share_class_figi: Optional[str] = None
    composite_figi: Optional[str] = None
    status: Optional[str] = None
    candidates: Optional[int] = None
    raw_best: Optional[Dict[str, Any]] = None


class OpenFigiClient:
    """
    Minimal OpenFIGI mapping client (ISIN -> metadata).
    - Uses batch requests for performance.
    - Retries on transient errors and 429.
    - In-memory cache.
    """

    BASE_URL = "https://api.openfigi.com/v3/mapping"
    # MIC → Yahoo suffix
    MIC_TO_YAHOO_SUFFIX = {
        "XTSE": ".TO",   # Toronto
        "XTSX": ".V",
        "XNEO": ".NE",
        "XMIL": ".MI",
        "XETR": ".DE",
        "XPAR": ".PA",
        "XLON": ".L",
    }

    def __init__(
        self,
        api_key: str,
        session: Optional[requests.Session] = None,
        timeout: float = 20.0,
        max_retries: int = 5,
        backoff_base: float = 0.6,
        batch_size: int = 80,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.batch_size = max(1, min(batch_size, 100))  # OpenFIGI accepts batches; keep safe
        self._s = session or requests.Session()
        self._s.headers.update(
            {
                "X-OPENFIGI-APIKEY": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        self._cache: Dict[str, OpenFigiInstrument] = {}

    def lookup_isin(self, isin: str) -> OpenFigiInstrument:
        """Single ISIN lookup (still goes through batch pipeline)."""
        res = self.lookup_isins([isin])
        return res[0]

    def lookup_isins(self, isins: Iterable[str]) -> List[OpenFigiInstrument]:
        """Batch lookup; preserves input order."""
        isins_list = [self._normalize_isin(x) for x in isins]
        out: List[OpenFigiInstrument] = []

        # serve from cache first
        missing: List[str] = []
        for isin in isins_list:
            cached = self._cache.get(isin)
            if cached is None:
                missing.append(isin)

        # fetch missing in chunks
        if missing:
            for chunk in _chunks(missing, self.batch_size):
                fetched = self._fetch_chunk(chunk)
                for inst in fetched:
                    self._cache[inst.isin] = inst

        # rebuild output in original order
        for isin in isins_list:
            out.append(self._cache.get(isin) or OpenFigiInstrument(isin=isin))
        return out

    def _fetch_chunk(self, isins: List[str]) -> List[OpenFigiInstrument]:
        payload = [{"idType": "ID_ISIN", "idValue": isin} for isin in isins]

        for attempt in range(self.max_retries + 1):
            r = None
            try:
                r = self._s.post(self.BASE_URL, json=payload, timeout=self.timeout)

                # Rate limit / transient handling
                if r.status_code == 429:
                    wait = self._retry_after_seconds(r) or self._backoff(attempt)
                    time.sleep(wait)
                    continue
                if 500 <= r.status_code < 600:
                    time.sleep(self._backoff(attempt))
                    continue

                r.raise_for_status()
                data = r.json()
                return self._parse_mapping_response(isins, data)

            except (requests.Timeout, requests.ConnectionError):
                time.sleep(self._backoff(attempt))
                continue
            except requests.HTTPError:
                # non-retryable client errors (400/401/403/404 etc.)
                return [OpenFigiInstrument(isin=x, status="error", candidates=0) for x in isins]
            except ValueError:
                # JSON decode error or unexpected payload
                return [OpenFigiInstrument(isin=x, status="error", candidates=0) for x in isins]

        # exhausted retries
        return [OpenFigiInstrument(isin=x, status="error", candidates=0) for x in isins]

    @staticmethod
    def _parse_mapping_response(isins: List[str], data: Any) -> List[OpenFigiInstrument]:
        """
        OpenFIGI returns a list aligned with the request payload.
        Each item can have 'data': [ {match}, ... ].
        We'll pick the first match if present (best-effort).
        """
        results: List[OpenFigiInstrument] = []

        if not isinstance(data, list):
            return [OpenFigiInstrument(isin=x, status="error", candidates=0) for x in isins]

        for isin, item in zip(isins, data):
            best = None
            status = None
            candidates = 0
            raw_best = None
            if isinstance(item, dict):
                matches = item.get("data")
                if isinstance(matches, list) and matches:
                    best = matches[0]  # best-effort: first match
                    raw_best = {"best": best, "candidates": matches}
                    candidates = len(matches)
                    status = "matched" if candidates == 1 else "ambiguous"
                elif item.get("error"):
                    status = "error"
                    candidates = 0
                else:
                    status = "not_found"
                    candidates = 0
            else:
                status = "error"
                candidates = 0

            if not isinstance(best, dict):
                results.append(OpenFigiInstrument(isin=isin, status=status, candidates=candidates))
                continue

            results.append(
                OpenFigiInstrument(
                    isin=isin,
                    figi=best.get("figi"),
                    name=best.get("name"),
                    ticker=best.get("ticker"),
                    exch_code=best.get("exchCode"),
                    mic_code=best.get("micCode"),
                    currency=best.get("currency"),
                    market_sector=best.get("marketSector"),
                    security_type=best.get("securityType"),
                    security_type2=best.get("securityType2"),
                    share_class_figi=best.get("shareClassFIGI"),
                    composite_figi=best.get("compositeFIGI"),
                    status=status,
                    candidates=candidates,
                    raw_best=raw_best,
                )
            )

        return results

    @staticmethod
    def _normalize_isin(isin: str) -> str:
        return isin.strip().upper()

    # --------------------------------------------------------
    # Helpers for Yahoo symbol resolution from candidates
    # --------------------------------------------------------
    @classmethod
    def to_yahoo_symbol(cls, ticker: Optional[str], mic_code: Optional[str]) -> Optional[str]:
        if not ticker:
            return None
        return f"{ticker}{cls.MIC_TO_YAHOO_SUFFIX.get((mic_code or '').upper(), '')}"

    @staticmethod
    def best_candidate(
        candidates: List[Dict[str, Any]],
        preferred_mic: Optional[str] = None,
        preferred_exch: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if preferred_mic:
            pm = preferred_mic.upper()
            for c in candidates:
                if (c.get("micCode") or "").upper() == pm:
                    return c

        if preferred_exch:
            pe = preferred_exch.upper()
            for c in candidates:
                if (c.get("exchCode") or "").upper() == pe:
                    return c

        return candidates[0] if candidates else None

    def _backoff(self, attempt: int) -> float:
        # exponential backoff with small cap
        return min(10.0, self.backoff_base * (2 ** attempt))

    @staticmethod
    def _retry_after_seconds(resp: requests.Response) -> Optional[float]:
        ra = resp.headers.get("Retry-After")
        if not ra:
            return None
        try:
            return float(ra)
        except ValueError:
            return None


def _chunks(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]
