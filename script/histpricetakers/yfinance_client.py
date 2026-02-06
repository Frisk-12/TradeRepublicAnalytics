from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, List

logger = logging.getLogger(__name__)


class YFinanceFetchError(RuntimeError):
    pass


class YFinanceInstrumentNotFound(YFinanceFetchError):
    pass


class YFinanceBadResponse(YFinanceFetchError):
    pass


@dataclass(frozen=True)
class YFinanceDailyBar:
    date: str
    close: float


class YFinanceHistoryClient:
    """
    Robust yfinance historical prices fetcher (daily).

    Input:
      - ISIN or mapped ticker

    Output:
      - If pandas available and as_pandas=True: DataFrame with columns [date, price, isin, ticker]
      - Else: list[YFinanceDailyBar]
    """

    def __init__(
        self,
        *,
        isin_to_ticker: Optional[Dict[str, str]] = None,
        auto_adjust: bool = False,
        prefer_adj_close: bool = False,
        base_currency: str = "EUR",
        fx_cache_path: Optional[str] = "/Users/andreadesogus/Desktop/portfolio_analytics/script/histpricetakers/fx_cache.json",
    ) -> None:
        self._isin_to_ticker = {k.upper(): v for k, v in (isin_to_ticker or {}).items()}
        self._auto_adjust = bool(auto_adjust)
        self._prefer_adj_close = bool(prefer_adj_close)
        self._base_currency = base_currency.upper()
        self._fx_cache: Dict[str, "pd.DataFrame"] = {}
        self._fx_cache_path = fx_cache_path
        self._fx_cache_file: Dict[str, List[Dict[str, Union[str, float]]]] = {}
        if fx_cache_path:
            self._fx_cache_file = self._load_fx_cache(fx_cache_path)

    def fetch_daily_history(
        self,
        isin: str,
        *,
        start: Optional[str] = None,   # "YYYY-MM-DD"
        end: Optional[str] = None,     # "YYYY-MM-DD"
        as_pandas: bool = True,
        convert_to_base: bool = True,
        ticker_override: Optional[str] = None,
    ):
        isin_norm = self._normalize_isin(isin)
        ticker = ticker_override or self._isin_to_ticker.get(isin_norm, isin_norm)

        try:
            import yfinance as yf  # type: ignore
        except Exception as e:
            raise YFinanceBadResponse("yfinance is required to fetch data") from e

        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None

        # Prefer yf.Ticker().history(): di solito evita MultiIndex e sorprese di download()
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=self._auto_adjust,
                actions=False,
            )
        except Exception as e:
            raise YFinanceBadResponse(f"yfinance error for ticker={ticker!r}") from e

        if hist is None or getattr(hist, "empty", True):
            raise YFinanceInstrumentNotFound(f"No data returned for ISIN={isin_norm} ticker={ticker!r}")

        # Alcune versioni possono comunque restituire MultiIndex (raro con history, più comune con download).
        # Normalizziamo: se MultiIndex, proviamo a estrarre la "slice" giusta.
        hist = self._ensure_single_ticker_frame(hist, ticker)

        # Se auto_adjust=True, spesso "Adj Close" non serve; se auto_adjust=False può essere utile.
        price_col_candidates = []
        if self._prefer_adj_close and "Adj Close" in hist.columns:
            price_col_candidates = ["Adj Close", "Close"]
        else:
            price_col_candidates = ["Close", "Adj Close"]

        price_col = next((c for c in price_col_candidates if c in hist.columns), None)
        if price_col is None:
            raise YFinanceBadResponse(
                f"Missing price column (Close/Adj Close) for ISIN={isin_norm} ticker={ticker!r}. "
                f"Columns={list(hist.columns)[:20]}"
            )

        # Convert index -> date
        # - può essere tz-aware DatetimeIndex
        # - può contenere timestamp intraday (ma interval=1d di norma no)
        if pd is None or not as_pandas:
            bars: List[YFinanceDailyBar] = []
            for idx, row in hist.iterrows():
                try:
                    # idx può essere Timestamp
                    d = getattr(idx, "date", None)
                    d_str = str(d() if callable(d) else idx)  # fallback
                    bars.append(YFinanceDailyBar(date=d_str[:10], close=float(row[price_col])))
                except Exception:
                    continue

            if not bars:
                raise YFinanceBadResponse(f"No parsable rows for ISIN={isin_norm} ticker={ticker!r}")

            if convert_to_base:
                ccy = self._resolve_currency(tk) or self._infer_currency_from_ticker(ticker)
                if not ccy:
                    raise YFinanceBadResponse(
                        f"Unable to resolve currency for ISIN={isin_norm} ticker={ticker!r}"
                    )
                fx_map = self._get_fx_map(ccy, self._base_currency, start, end)
                if ccy != self._base_currency and not fx_map:
                    raise YFinanceBadResponse(
                        f"Missing FX series for {ccy}/{self._base_currency} ISIN={isin_norm} ticker={ticker!r}"
                    )
                if fx_map:
                    last_rate = None
                    fx_dates = sorted(fx_map.keys())
                    fx_idx = 0
                    for i, bar in enumerate(bars):
                        # forward fill
                        while fx_idx < len(fx_dates) and fx_dates[fx_idx] <= bar.date:
                            last_rate = fx_map[fx_dates[fx_idx]]
                            fx_idx += 1
                        if last_rate is None and fx_dates:
                            # backfill from first available
                            last_rate = fx_map[fx_dates[0]]
                        if last_rate:
                            bars[i] = YFinanceDailyBar(date=bar.date, close=bar.close * last_rate)
            return bars

        # pandas path
        out = hist.reset_index().copy()

        # rename time column
        if "Date" in out.columns:
            out.rename(columns={"Date": "date"}, inplace=True)
        elif "Datetime" in out.columns:
            out.rename(columns={"Datetime": "date"}, inplace=True)
        else:
            # fallback: first column is index
            out.rename(columns={out.columns[0]: "date"}, inplace=True)

        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        out = out.dropna(subset=["date"])

        out["price_ccy"] = out[price_col].astype(float)
        out["currency"] = self._resolve_currency(tk) or self._infer_currency_from_ticker(ticker)
        out["isin"] = isin_norm
        out["ticker"] = ticker

        resolved_ccy = out["currency"].iloc[0] if not out.empty else None
        if convert_to_base and not resolved_ccy:
            raise YFinanceBadResponse(
                f"Unable to resolve currency for ISIN={isin_norm} ticker={ticker!r}"
            )

        if convert_to_base and resolved_ccy and resolved_ccy != self._base_currency:
            fx = self._get_fx_series(resolved_ccy, self._base_currency, start, end)
            if fx is None or fx.empty:
                raise YFinanceBadResponse(
                    f"Missing FX series for {resolved_ccy}/{self._base_currency} ISIN={isin_norm} ticker={ticker!r}"
                )
            out = out.merge(fx, on="date", how="left")
            out["fx_rate"] = out["fx_rate"].ffill().bfill()
            out["price"] = out["price_ccy"] * out["fx_rate"]
        else:
            out["fx_rate"] = 1.0
            out["price"] = out["price_ccy"]

        return (
            out[["date", "price", "price_ccy", "currency", "fx_rate", "isin", "ticker"]]
            .sort_values("date")
            .reset_index(drop=True)
        )

    # ---------------- Internals ----------------

    @staticmethod
    def _normalize_isin(isin: str) -> str:
        s = isin.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{12}", s):
            raise ValueError(f"Invalid ISIN: {isin!r}")
        return s

    @staticmethod
    def _ensure_single_ticker_frame(df, ticker: str):
        """
        If df has MultiIndex columns like ('Close','AAPL') or ('AAPL','Close'),
        try to extract the sub-frame for the intended ticker.

        Works with:
          - MultiIndex columns level order (field, ticker)
          - MultiIndex columns level order (ticker, field)
        """
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return df

        cols = getattr(df, "columns", None)
        if cols is None:
            return df

        if not isinstance(cols, pd.MultiIndex):
            return df

        # Attempt (field, ticker)
        if ticker in cols.get_level_values(-1):
            try:
                sub = df.xs(ticker, axis=1, level=-1, drop_level=True)
                if getattr(sub, "empty", False) is False:
                    return sub
            except Exception:
                pass

        # Attempt (ticker, field)
        if ticker in cols.get_level_values(0):
            try:
                sub = df.xs(ticker, axis=1, level=0, drop_level=True)
                if getattr(sub, "empty", False) is False:
                    return sub
            except Exception:
                pass

        # If we can't extract, return original (caller will likely raise on missing Close)
        return df

    @staticmethod
    def _resolve_currency(tk) -> Optional[str]:
        currency = None
        try:
            fast = getattr(tk, "fast_info", None)
            if fast and isinstance(fast, dict):
                currency = fast.get("currency")
        except Exception:
            currency = None
        if not currency:
            try:
                info = getattr(tk, "info", {}) or {}
                currency = info.get("currency")
            except Exception:
                currency = None
        return str(currency).upper() if currency else None

    @staticmethod
    def _infer_currency_from_ticker(ticker: str) -> Optional[str]:
        t = str(ticker or "").upper()
        if "." not in t:
            return None
        suffix = t.rsplit(".", 1)[1]
        suffix_to_ccy = {
            "TO": "CAD",
            "V": "CAD",
            "SW": "CHF",
            "MI": "EUR",
            "PA": "EUR",
            "DE": "EUR",
            "AS": "EUR",
            "BR": "EUR",
            "LS": "EUR",
            "MC": "EUR",
            "VI": "EUR",
            "F": "EUR",
            "BE": "EUR",
            "HA": "EUR",
            "MU": "EUR",
            "DU": "EUR",
            "HM": "EUR",
            "SG": "EUR",
        }
        return suffix_to_ccy.get(suffix)

    def _get_fx_series(
        self,
        ccy: str,
        base: str,
        start: Optional[str],
        end: Optional[str],
    ):
        try:
            import pandas as pd  # type: ignore
            import yfinance as yf  # type: ignore
        except Exception:
            return None

        ccy = ccy.upper()
        base = base.upper()
        if ccy == base:
            return None

        cache_key = f"{ccy}_{base}_{start}_{end}"
        if cache_key in self._fx_cache:
            return self._fx_cache[cache_key]
        if cache_key in self._fx_cache_file:
            cached = self._fx_cache_file[cache_key]
            df = pd.DataFrame(cached)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                df = df.dropna(subset=["date"])
                self._fx_cache[cache_key] = df
                return df

        def _fetch_fx(ticker: str, invert: bool = False):
            fx_hist = yf.Ticker(ticker).history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            if fx_hist is None or getattr(fx_hist, "empty", True):
                return None
            fx = fx_hist.reset_index().copy()
            col = "Close" if "Close" in fx.columns else fx.columns[0]
            fx.rename(columns={fx.columns[0]: "date"}, inplace=True)
            if "Date" in fx.columns:
                fx.rename(columns={"Date": "date"}, inplace=True)
            elif "Datetime" in fx.columns:
                fx.rename(columns={"Datetime": "date"}, inplace=True)
            fx["date"] = pd.to_datetime(fx["date"], errors="coerce").dt.date
            fx = fx.dropna(subset=["date"])
            fx["fx_rate"] = fx[col].astype(float)
            if invert:
                fx["fx_rate"] = 1.0 / fx["fx_rate"]
            return fx[["date", "fx_rate"]]

        # Prefer CCYBASE=X (base per ccy). If missing, try BASECCY=X and invert.
        direct = _fetch_fx(f"{ccy}{base}=X", invert=False)
        if direct is None or direct.empty:
            inverse = _fetch_fx(f"{base}{ccy}=X", invert=True)
            fx = inverse
        else:
            fx = direct

        if fx is None or fx.empty:
            fx = self._fetch_fx_fallback(ccy, base, start, end)

        if fx is not None:
            self._fx_cache[cache_key] = fx
            self._write_fx_cache(cache_key, fx)
        return fx

    def _get_fx_map(
        self,
        ccy: str,
        base: str,
        start: Optional[str],
        end: Optional[str],
    ) -> Optional[Dict[str, float]]:
        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None

        if ccy.upper() == base.upper():
            return None

        if pd is not None:
            fx = self._get_fx_series(ccy, base, start, end)
            if fx is None or fx.empty:
                return None
            return {str(row["date"]): float(row["fx_rate"]) for _, row in fx.iterrows()}

        # no pandas: try cache file or fallback fetch
        cache_key = f"{ccy.upper()}_{base.upper()}_{start}_{end}"
        if cache_key in self._fx_cache_file:
            return {row["date"]: float(row["fx_rate"]) for row in self._fx_cache_file[cache_key]}
        fx = self._fetch_fx_fallback(ccy, base, start, end)
        if fx is None:
            return None
        return {row["date"]: float(row["fx_rate"]) for row in fx}

    def _fetch_fx_fallback(
        self,
        ccy: str,
        base: str,
        start: Optional[str],
        end: Optional[str],
    ):
        try:
            import requests  # type: ignore
            import pandas as pd  # type: ignore
        except Exception:
            return None

        ccy = ccy.upper()
        base = base.upper()
        if not start or not end:
            return None

        url = f"https://api.frankfurter.app/{start}..{end}?from={ccy}&to={base}"
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None

        rates = data.get("rates", {}) if isinstance(data, dict) else {}
        if not isinstance(rates, dict) or not rates:
            return None

        rows = []
        for d, payload in rates.items():
            if not isinstance(payload, dict):
                continue
            rate = payload.get(base)
            if rate is None:
                continue
            rows.append({"date": d, "fx_rate": float(rate)})

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.dropna(subset=["date"])
        return df[["date", "fx_rate"]]

    def _load_fx_cache(self, path: str) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        p = Path(path)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    def _write_fx_cache(self, cache_key: str, fx_df) -> None:
        if not self._fx_cache_path:
            return
        try:
            rows = [
                {"date": str(row["date"]), "fx_rate": float(row["fx_rate"])}
                for _, row in fx_df.iterrows()
            ]
        except Exception:
            return
        self._fx_cache_file[cache_key] = rows
        try:
            Path(self._fx_cache_path).write_text(json.dumps(self._fx_cache_file, indent=2))
        except Exception:
            return
