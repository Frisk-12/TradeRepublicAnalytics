#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:04:58 2026

@author: andreadesogus
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class SocGenPriceFetchError(RuntimeError): ...
class SocGenInstrumentNotFound(SocGenPriceFetchError): ...
class SocGenBadResponse(SocGenPriceFetchError): ...


@dataclass(frozen=True)
class SocGenDailyBar:
    date_utc: str
    bid: float
    ask: float
    underlying_price: Optional[float] = None
    index_price: Optional[float] = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


class SocieteGeneraleWarrantsHistoryClient:
    BASE = "https://it.warrants.com"
    HISTORY_ENDPOINT = "/EmcWebApi/api/Prices/History"
    PRODUCT_DETAILS_PATH = "/product-details/{isin_lower}"

    _RE_PID_IN_URL = re.compile(r"[?&]productId=(\d+)\b", re.IGNORECASE)
    _RE_PID_IN_HTML = re.compile(
        r'("productId"\s*:\s*|productId\s*=\s*|productId\s*:\s*)(\d+)', re.IGNORECASE
    )

    def __init__(
        self,
        *,
        timeout_s: float = 20.0,
        max_retries: int = 4,
        backoff_base_s: float = 0.6,
        resolver_mode: str = "auto",  # "auto" | "http_only" | "playwright_only"
        playwright_headless: bool = True,
        isin_to_product_id: Optional[Dict[str, int]] = None,
        user_agent: str = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        session: Optional[requests.Session] = None,
    ) -> None:
        if resolver_mode not in {"auto", "http_only", "playwright_only"}:
            raise ValueError("resolver_mode must be one of: auto, http_only, playwright_only")

        self._timeout_s = float(timeout_s)
        self._max_retries = int(max_retries)
        self._backoff_base_s = float(backoff_base_s)
        self._resolver_mode = resolver_mode
        self._playwright_headless = bool(playwright_headless)
        self._session = session or requests.Session()

        self._headers = {
            "accept": "application/json, text/plain, */*",
            "user-agent": user_agent,
            "x-requested-with": "XMLHttpRequest",
        }

        # Optional local override cache (still "ISIN-only" for callers)
        self._isin_to_pid = {k.upper(): int(v) for k, v in (isin_to_product_id or {}).items()}

    # -------------------- Public --------------------

    def fetch_daily_history(self, isin: str, *, as_pandas: bool = True):
        isin_norm = self._normalize_isin(isin)
        pid = self._resolve_product_id(isin_norm)
        bars = self._download_history(pid)

        if not as_pandas:
            return bars

        try:
            import pandas as pd  # type: ignore
        except Exception:
            return bars

        df = pd.DataFrame([{
            "date_utc": b.date_utc,
            "bid": b.bid,
            "ask": b.ask,
            "mid": b.mid,
            "underlying_price": b.underlying_price,
            "index_price": b.index_price
        } for b in bars])

        df["date"] = pd.to_datetime(df["date_utc"], utc=True).dt.date
        df["isin"] = isin_norm
        df["product_id"] = pid
        return df.sort_values("date").reset_index(drop=True)

    # -------------------- Helpers --------------------

    @staticmethod
    def _normalize_isin(isin: str) -> str:
        s = isin.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{12}", s):
            raise ValueError(f"Invalid ISIN: {isin!r}")
        return s

    def _request_json(self, url: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                r = self._session.get(url, params=params, headers=self._headers, timeout=self._timeout_s)
                if r.status_code == 404:
                    raise SocGenInstrumentNotFound(f"404 at {url}")
                r.raise_for_status()
                if "application/json" not in (r.headers.get("content-type", "").lower()):
                    raise SocGenBadResponse(f"Expected JSON, got {r.headers.get('content-type')!r} url={url}")
                return r.json()
            except (requests.RequestException, json.JSONDecodeError, SocGenBadResponse) as e:
                last_err = e
                if attempt < self._max_retries:
                    time.sleep(self._backoff_base_s * (2 ** (attempt - 1)))
                    continue
                break
        raise SocGenBadResponse(f"Failed GET JSON {url}: {last_err}")

    def _request_text(self, url: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                r = self._session.get(url, headers=self._headers, timeout=self._timeout_s)
                if r.status_code == 404:
                    raise SocGenInstrumentNotFound(f"404 at {url}")
                r.raise_for_status()
                return r.text
            except (requests.RequestException, SocGenInstrumentNotFound) as e:
                last_err = e
                if attempt < self._max_retries:
                    time.sleep(self._backoff_base_s * (2 ** (attempt - 1)))
                    continue
                break
        raise SocGenBadResponse(f"Failed GET TEXT {url}: {last_err}")

    # -------------------- Resolution --------------------

    @lru_cache(maxsize=512)
    def _resolve_product_id(self, isin: str) -> int:
        # 0) local override cache (fastest, and still ISIN-only API)
        if isin in self._isin_to_pid:
            return self._isin_to_pid[isin]

        if self._resolver_mode == "playwright_only":
            pid = self._resolve_via_playwright(isin)
            if pid is None:
                raise SocGenInstrumentNotFound(f"Playwright could not resolve productId for {isin}")
            return pid

        # 1) http-first
        pid = self._resolve_via_http_probes(isin)
        if pid is not None:
            return pid

        # 2) html scrape (best-effort)
        pid = self._resolve_via_html(isin)
        if pid is not None:
            return pid

        # 3) playwright fallback (only in auto)
        if self._resolver_mode == "auto":
            pid = self._resolve_via_playwright(isin)
            if pid is not None:
                return pid

        raise SocGenInstrumentNotFound(
            f"Could not resolve productId for ISIN={isin}. "
            f"Tip: install playwright OR provide isin_to_product_id mapping."
        )

    def _resolve_via_http_probes(self, isin: str) -> Optional[int]:
        # Best-effort: these may or may not exist
        endpoints = [
            f"{self.BASE}/EmcWebApi/api/Products/ByIsin",
            f"{self.BASE}/EmcWebApi/api/Product/ByIsin",
            f"{self.BASE}/EmcWebApi/api/Search",
            f"{self.BASE}/EmcWebApi/api/Products/Search",
        ]
        param_sets = [{"isin": isin}, {"q": isin}, {"query": isin}, {"term": isin}]
        for url in endpoints:
            for params in param_sets:
                try:
                    data = self._request_json(url, params=params)
                except (SocGenInstrumentNotFound, SocGenBadResponse):
                    continue
                pid = self._extract_pid_from_payload(data, target_isin=isin)
                if pid is not None:
                    return pid
        return None

    def _resolve_via_html(self, isin: str) -> Optional[int]:
        url = self.BASE + self.PRODUCT_DETAILS_PATH.format(isin_lower=isin.lower())
        html = self._request_text(url)
        m = self._RE_PID_IN_HTML.search(html)
        if m:
            try:
                return int(m.group(2))
            except Exception:
                return None
        return None

    def _resolve_via_playwright(self, isin: str) -> Optional[int]:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception:
            # IMPORTANT: do not crash with SocGenBadResponse here; return None
            # so user can use http_only or mapping without playwright.
            return None

        url = self.BASE + self.PRODUCT_DETAILS_PATH.format(isin_lower=isin.lower())
        found: List[int] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self._playwright_headless)
            context = browser.new_context(user_agent=self._headers["user-agent"])
            page = context.new_page()

            def on_request(req):
                u = req.url
                if "EmcWebApi" in u and "productId=" in u:
                    mm = self._RE_PID_IN_URL.search(u)
                    if mm:
                        try:
                            found.append(int(mm.group(1)))
                        except Exception:
                            pass

            page.on("request", on_request)
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(2500)
            browser.close()

        return found[0] if found else None

    @staticmethod
    def _extract_pid_from_payload(data: Any, *, target_isin: str) -> Optional[int]:
        def try_dict(d: Dict[str, Any]) -> Optional[int]:
            for k in ("productId", "ProductId", "productID"):
                if k in d and str(d[k]).isdigit():
                    isin_val = (d.get("isin") or d.get("ISIN") or "").strip().upper()
                    if isin_val and isin_val != target_isin:
                        return None
                    return int(d[k])
            for k in ("data", "result", "item", "product", "Product", "instrument"):
                v = d.get(k)
                if isinstance(v, dict):
                    pid = try_dict(v)
                    if pid is not None:
                        return pid
            return None

        if isinstance(data, dict):
            pid = try_dict(data)
            if pid is not None:
                return pid
            for k in ("items", "results", "data"):
                if isinstance(data.get(k), list):
                    for it in data[k]:
                        if isinstance(it, dict):
                            pid = try_dict(it)
                            if pid is not None:
                                return pid

        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict):
                    pid = try_dict(it)
                    if pid is not None:
                        return pid

        return None

    # -------------------- History --------------------

    def _download_history(self, product_id: int) -> List[SocGenDailyBar]:
        url = self.BASE + self.HISTORY_ENDPOINT
        data = self._request_json(url, params={"productId": product_id})
        if not isinstance(data, list):
            raise SocGenBadResponse(f"Unexpected history payload type for productId={product_id}")

        out: List[SocGenDailyBar] = []
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            try:
                out.append(SocGenDailyBar(
                    date_utc=str(row["Date"]),
                    bid=float(row["Bid"]),
                    ask=float(row["Ask"]),
                    underlying_price=float(row["UnderlyingPrice"]) if row.get("UnderlyingPrice") is not None else None,
                    index_price=float(row["IndexPrice"]) if row.get("IndexPrice") is not None else None,
                ))
            except Exception as e:
                raise SocGenBadResponse(f"Bad row #{i} for productId={product_id}: {row}") from e

        if not out:
            raise SocGenBadResponse(f"Empty history for productId={product_id}")
        return out




