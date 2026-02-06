#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 16:26:17 2026

@author: andreadesogus
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


@dataclass(frozen=True)
class JustEtfPoint:
    date: str          # ISO YYYY-MM-DD
    value: float       # numeric (raw)


class JustEtfClient:
    """
    Minimal JustETF API client for performance chart series.

    Flow:
      1) GET profile page to obtain cookies + XSRF-TOKEN
      2) GET /api/etfs/{isin}/performance-chart with X-XSRF-TOKEN header
    """

    BASE = "https://www.justetf.com"

    def __init__(
        self,
        locale: str = "it",
        timeout: float = 20.0,
        max_retries: int = 4,
        backoff_base: float = 0.6,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.locale = locale
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        self.s = session or requests.Session()
        # headers "browser-like" ma sobri (niente cookie hardcoded)
        self.s.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; portfolio-analytics/1.0; +https://example.invalid)",
                "Accept": "application/json, text/plain, */*",
            }
        )

        # small in-memory caches
        self._xsrf_cache: Dict[str, str] = {}          # locale -> xsrf token
        self._series_cache: Dict[Tuple, List[JustEtfPoint]] = {}

    def fetch_performance_chart(
        self,
        isin: str,
        date_from: str,
        date_to: str,
        currency: str = "EUR",
        values_type: str = "MARKET_VALUE",
        include_dividends: bool = False,
        reduce_data: bool = False,
    ) -> List[JustEtfPoint]:
        """
        Returns list of (date, value) points. 'value' is the raw series value.

        Notes:
        - 'MARKET_VALUE' appears to be price-like for ETF.
        - include_dividends=False => price only; True may provide total return-like.
        """
        isin = isin.strip().upper()
        cache_key = (isin, date_from, date_to, currency, values_type, include_dividends, reduce_data, self.locale)
        if cache_key in self._series_cache:
            return self._series_cache[cache_key]

        self._ensure_session_tokens()

        url = f"{self.BASE}/api/etfs/{isin}/performance-chart"
        params = {
            "locale": self.locale,
            "currency": currency,
            "valuesType": values_type,
            "reduceData": str(reduce_data).lower(),
            "includeDividends": str(include_dividends).lower(),
            "features": "DIVIDENDS",
            "dateFrom": date_from,
            "dateTo": date_to,
        }

        data = self._get_json_with_retries(url, params=params)
        series = self._parse_series(data)
        self._series_cache[cache_key] = series
        return series

    # ----------------- internals -----------------

    def _ensure_session_tokens(self) -> None:
        """Fetch profile page once per locale to get XSRF token cookies."""
        if self.locale in self._xsrf_cache:
            return

        # Hit any profile page without caring about content; we just want cookies.
        # We can use a generic landing in the locale; profile is fine too.
        url = f"{self.BASE}/{self.locale}/etf-profile.html"
        # some deployments accept without ISIN; but safest is to pass a dummy param and tolerate 200/302
        r = self.s.get(url, params={"isin": "IE00B4L5Y983"}, timeout=self.timeout, allow_redirects=True)
        r.raise_for_status()

        xsrf = self.s.cookies.get("XSRF-TOKEN")
        if not xsrf:
            # sometimes the token is set later; try homepage fallback
            r2 = self.s.get(f"{self.BASE}/{self.locale}/", timeout=self.timeout, allow_redirects=True)
            r2.raise_for_status()
            xsrf = self.s.cookies.get("XSRF-TOKEN")

        if not xsrf:
            raise RuntimeError("JustETF: missing XSRF-TOKEN cookie; site flow may have changed.")

        self._xsrf_cache[self.locale] = xsrf
        # Required by many stacks: send token back as header
        self.s.headers["X-XSRF-TOKEN"] = xsrf

    def _get_json_with_retries(self, url: str, params: Dict[str, Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                if r.status_code in (429, 502, 503, 504):
                    time.sleep(self._backoff(attempt))
                    continue
                if r.status_code == 403:
                    # token could be expired; refresh once
                    self._xsrf_cache.pop(self.locale, None)
                    self._ensure_session_tokens()
                    time.sleep(self._backoff(attempt))
                    continue

                r.raise_for_status()
                return r.json()

            except (requests.Timeout, requests.ConnectionError):
                time.sleep(self._backoff(attempt))
                continue
            except ValueError as e:
                raise RuntimeError(f"JustETF: invalid JSON: {e}") from e

        raise RuntimeError("JustETF: retries exhausted.")

    def _parse_series(self, data: Any) -> List[JustEtfPoint]:
        """
        Expected structure (based on your screenshot):
          { ..., "series": [ { "date": "YYYY-MM-DD", "value": { "raw": 8.65, ... } }, ... ] }
        """
        if not isinstance(data, dict):
            return []
        raw_series = data.get("series")
        if not isinstance(raw_series, list):
            return []

        out: List[JustEtfPoint] = []
        for item in raw_series:
            if not isinstance(item, dict):
                continue
            d = item.get("date")
            v = item.get("value", {})
            if isinstance(v, dict):
                val = v.get("raw")
            else:
                val = None
            if isinstance(d, str) and isinstance(val, (int, float)):
                out.append(JustEtfPoint(date=d, value=float(val)))
        return out

    def _backoff(self, attempt: int) -> float:
        return min(8.0, self.backoff_base * (2 ** attempt))



# client = JustEtfClient(locale="it")
# series = client.fetch_performance_chart(
#     isin="IE0003A512E4",
#     date_from="2024-04-12",
#     date_to="2026-02-02",
#     currency="EUR",
#     values_type="MARKET_VALUE",
#     include_dividends=False,
#     reduce_data=False,
# )
# print(series[:3], "...", series[-1])
