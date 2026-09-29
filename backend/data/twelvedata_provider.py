from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from backend.config import settings
from backend.data.base import DataResult, MarketDataProvider, ProviderError

TWELVE_DATA_BASE_URL = "https://api.twelvedata.com/time_series"

# Twelve Data's "outputsize" parameter is literally a row count, which
# maps directly onto this app's existing period vocabulary the same way
# Stooq's provider maps periods to a day count.
_PERIOD_TO_OUTPUTSIZE = {
    "1d": 3, "5d": 7, "1mo": 25, "3mo": 68, "6mo": 132,
    "1y": 260, "3y": 780, "5y": 1300, "max": 5000,  # 5000 is Twelve Data's own hard cap
}


class TwelveDataProvider(MarketDataProvider):
    """A real, authenticated (not scraping-based) fallback market data
    source, added after a real deployment found Yahoo Finance and Stooq
    both returning HTTP 429s from Render's shared cloud IP range — even
    after adding a realistic browser User-Agent, ruling out headers as
    the cause and pointing to IP reputation specifically. An
    authenticated REST API with its own key isn't fingerprinted as
    anonymous scraping traffic the way those two are, so it shouldn't
    hit the same block. Free tier: 800 requests/day, 8/minute — genuinely
    workable for a personal research tool. Only activates when
    TWELVE_DATA_API_KEY is set (via an environment variable); otherwise
    reports itself unavailable rather than erroring, so a local setup
    without a key is completely unaffected — yfinance and Stooq keep
    working there exactly as before, since they aren't blocked from a
    home internet connection."""

    name = "twelvedata"

    def _api_key(self) -> str:
        return settings.TWELVE_DATA_API_KEY

    def get_quote(self, ticker: str) -> DataResult:
        hist_result = self.get_historical(ticker, period="5d")
        if not hist_result.success or hist_result.data is None or hist_result.data.empty:
            return DataResult(
                data=None, source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day", success=False,
                error=hist_result.error or "Twelve Data has no data for this ticker", quality="ERROR",
            )
        df = hist_result.data
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else None
        data = {
            "ticker": ticker.upper(),
            "price": float(last["close"]),
            "previous_close": float(prev["close"]) if prev is not None else None,
            "day_high": float(last["high"]),
            "day_low": float(last["low"]),
            "volume": int(last["volume"]) if last["volume"] == last["volume"] else None,
            "market_cap": None,
            "currency": "USD",
        }
        return DataResult(data=data, source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="end_of_day")

    def get_historical(self, ticker: str, period: str, interval: str = "1d") -> DataResult:
        if not self._api_key():
            return DataResult(
                data=None, source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day", success=False,
                error="Twelve Data provider has no API key configured", quality="ERROR",
            )
        if interval != "1d":
            return DataResult(
                data=None, source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day", success=False,
                error="Twelve Data fallback only supports daily bars here", quality="ERROR",
            )
        try:
            outputsize = _PERIOD_TO_OUTPUTSIZE.get(period, 260)
            resp = requests.get(TWELVE_DATA_BASE_URL, params={
                "symbol": ticker.upper(),
                "interval": "1day",
                "outputsize": outputsize,
                "order": "asc",
                "adjust": "splits",  # matches this app's own split-adjustment fix (see yfinance_provider.py) — never raw, unadjusted prices
                "apikey": self._api_key(),
            }, timeout=15)
            resp.raise_for_status()
            payload = resp.json()

            if payload.get("status") != "ok":
                raise ProviderError(payload.get("message", f"Twelve Data returned an error for {ticker}"))

            values = payload.get("values", [])
            if not values:
                raise ProviderError(f"'{ticker}' has no data on Twelve Data")

            df = pd.DataFrame(values)
            df = df.rename(columns={"datetime": "date"})
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["adj_close"] = df["close"]  # already split-adjusted via adjust=splits above
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df = df[["date", "open", "high", "low", "close", "adj_close", "volume"]]

            return DataResult(
                data=df.reset_index(drop=True),
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day",
            )
        except ProviderError as exc:
            return DataResult(
                data=None, source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day", success=False, error=str(exc), quality="ERROR",
            )
        except requests.exceptions.RequestException as exc:
            return DataResult(
                data=None, source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day", success=False,
                error=f"Could not reach Twelve Data for {ticker}: {exc}", quality="ERROR",
            )
        except Exception as exc:  # noqa: BLE001
            return DataResult(
                data=None, source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day", success=False, error=str(exc), quality="ERROR",
            )

    def search(self, query: str) -> DataResult:
        return DataResult(
            data=[], source=self.name, fetched_at=datetime.now(timezone.utc),
            timeliness="cached", success=False,
            error="Twelve Data provider does not support search here", quality="ERROR",
        )
