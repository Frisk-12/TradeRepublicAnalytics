from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class VontobelFetchError(RuntimeError):
    pass


class VontobelInstrumentNotFound(VontobelFetchError):
    pass


class VontobelBadResponse(VontobelFetchError):
    pass


@dataclass(frozen=True)
class VontobelPoint:
    timestamp_ms: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    mid: Optional[float] = None


class VontobelMaxHistoryClient:
    """
    Vontobel Markets historical prices (MAX chart) fetcher.

    ✅ Input: ISIN only
    ✅ Deterministic endpoint (no probing):
        /api/v1/charts/products/{ISIN}/detail/6/0?c=it-it&it=1

    From your DevTools:
      - chartType=6 corresponds to MAX
      - priceType=0 works for product series (returns points with bid in your example)

    Output:
      - If pandas is available: DataFrame (daily or intraday)
      - Else: list[VontobelPoint]
    """

    BASE = "https://markets.vontobel.com"
    ENDPOINT = "/api/v1/charts/products/{isin}/detail/6/0"

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        *,
        locale: str = "it-it",
        it_flag: int = 1,
        timeout_s: float = 20.0,
        max_retries: int = 4,
        backoff_base_s: float = 0.6,
        user_agent: str = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    ) -> None:
        self._session = session or requests.Session()
        self._locale = locale
        self._it_flag = int(it_flag)
        self._timeout_s = float(timeout_s)
        self._max_retries = int(max_retries)
        self._backoff_base_s = float(backoff_base_s)
        self._headers = {
            "accept": "application/json",
            "user-agent": user_agent,
        }

    # ---------------- Public API ----------------

    def fetch_max_intraday(self, isin: str, *, as_pandas: bool = True):
        """
        Returns the MAX intraday time series as provided by Vontobel.
        """
        isin_norm = self._normalize_isin(isin)
        payload = self._get_max_payload(isin_norm)
        points = self._parse_points(payload)

        if not as_pandas:
            return points

        return self._points_to_df(points, isin_norm)

    def fetch_max_daily(self, isin: str, *, as_pandas: bool = True, price_preference=("mid", "last", "bid", "ask")):
        """
        Returns daily series derived from MAX intraday series:
        - one row per day
        - daily close = last observation of the day
        - price field chosen by preference among available columns
        """
        isin_norm = self._normalize_isin(isin)
        payload = self._get_max_payload(isin_norm)
        points = self._parse_points(payload)

        if not as_pandas:
            # If pandas not requested, still return points (caller can aggregate)
            return points

        try:
            import pandas as pd  # type: ignore
        except Exception:
            return points

        df = self._points_to_df(points, isin_norm)

        # choose field
        source_field = None
        for f in price_preference:
            if f in df.columns and df[f].notna().any():
                source_field = f
                break
        if source_field is None:
            raise VontobelBadResponse(f"No usable price field found for ISIN={isin_norm} in {list(df.columns)}")

        df = df.sort_values("timestamp_ms")
        df["date"] = df["timestamp_utc"].dt.date

        daily = df.groupby("date", as_index=False).tail(1).copy()
        daily["price"] = daily[source_field]
        daily["source_field"] = source_field
        daily["chart_type"] = 6
        daily["price_type"] = 0
        daily["isin"] = isin_norm

        return daily[["date", "price", "source_field", "isin", "chart_type", "price_type"]].sort_values("date").reset_index(drop=True)

    # ---------------- Internals ----------------

    @staticmethod
    def _normalize_isin(isin: str) -> str:
        s = isin.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{12}", s):
            raise ValueError(f"Invalid ISIN: {isin!r}")
        return s

    def _build_url(self, isin: str) -> str:
        return f"{self.BASE}{self.ENDPOINT.format(isin=isin)}?c={self._locale}&it={self._it_flag}"

    def _request_json(self, url: str) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                r = self._session.get(url, headers=self._headers, timeout=self._timeout_s)
                if r.status_code == 404:
                    raise VontobelInstrumentNotFound(f"ISIN not found (404): {url}")
                r.raise_for_status()

                ctype = (r.headers.get("content-type") or "").lower()
                if "application/json" not in ctype:
                    raise VontobelBadResponse(f"Expected JSON, got content-type={ctype!r} url={url}")

                data = r.json()
                if not isinstance(data, dict):
                    raise VontobelBadResponse(f"Unexpected JSON root type={type(data).__name__} url={url}")
                return data

            except (requests.RequestException, json.JSONDecodeError, VontobelBadResponse) as e:
                last_err = e
                if attempt < self._max_retries:
                    time.sleep(self._backoff_base_s * (2 ** (attempt - 1)))
                    continue
                break

        raise VontobelBadResponse(f"Failed to fetch Vontobel JSON url={url}: {last_err}")

    def _get_max_payload(self, isin: str) -> Dict[str, Any]:
        url = self._build_url(isin)
        data = self._request_json(url)

        if data.get("isSuccess") is not True:
            raise VontobelInstrumentNotFound(f"isSuccess != true for ISIN={isin}. errorCode={data.get('errorCode')!r}")

        pl = data.get("payload")
        if not isinstance(pl, dict):
            raise VontobelBadResponse(f"Missing/invalid 'payload' for ISIN={isin}")

        # Optional sanity check: ensure series contains the requested ISIN
        series = pl.get("series")
        if isinstance(series, list) and series:
            s0 = series[0]
            if isinstance(s0, dict):
                pid = s0.get("priceIdentifier")
                if pid and str(pid).upper() != isin.upper():
                    logger.debug("Series priceIdentifier=%r != requested ISIN=%r (continuing)", pid, isin)

        return pl

    def _parse_points(self, payload: Dict[str, Any]) -> List[VontobelPoint]:
        series = payload.get("series")
        if not isinstance(series, list) or not series or not isinstance(series[0], dict):
            raise VontobelBadResponse("Missing/invalid payload.series[0]")
    
        pts = series[0].get("points")
        if not isinstance(pts, list):
            raise VontobelBadResponse("Missing/invalid payload.series[0].points")
    
        out: List[VontobelPoint] = []
        for row in pts:
            if not isinstance(row, dict):
                continue
            ts = row.get("timestamp")
            if ts is None:
                continue
            try:
                ts_ms = int(ts)
            except Exception:
                continue
    
            def f(key: str) -> Optional[float]:
                v = row.get(key)
                if v is None:
                    return None
                try:
                    return float(v)
                except Exception:
                    return None
    
            out.append(VontobelPoint(
                timestamp_ms=ts_ms,
                bid=f("bid"),
                ask=f("ask"),
                last=f("last"),
                mid=f("mid"),
            ))
    
        if not out:
            raise VontobelBadResponse("No parsable points returned in payload.series[0].points")
    
        out.sort(key=lambda p: p.timestamp_ms)
        return out


    def _points_to_df(self, points: List[VontobelPoint], isin: str):
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise VontobelBadResponse("pandas is required for as_pandas=True") from e

        df = pd.DataFrame([{
            "timestamp_ms": p.timestamp_ms,
            "bid": p.bid,
            "ask": p.ask,
            "last": p.last,
            "mid": p.mid,
        } for p in points])

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df["isin"] = isin
        df["chart_type"] = 6
        df["price_type"] = 0

        return df[["timestamp_utc", "timestamp_ms", "bid", "ask", "last", "mid", "isin", "chart_type", "price_type"]].sort_values("timestamp_ms").reset_index(drop=True)


# ---------------- Example ----------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = VontobelMaxHistoryClient()
    isin = "DE000VJ2GXA8"

    df_daily = client.fetch_max_daily(isin, as_pandas=True)
    print(df_daily.tail())
