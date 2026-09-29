from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd
import requests

from backend.data.base import DataResult, MarketDataProvider, ProviderError

STOOQ_CSV_URL = "https://stooq.com/q/d/l/"

# Stooq doesn't take yfinance-style periods; we always pull max history
# and trim locally, which keeps this provider simple as a fallback.
_PERIOD_TO_DAYS = {
    "1d": 5, "5d": 10, "1mo": 35, "3mo": 100, "6mo": 200,
    "1y": 400, "3y": 1200, "5y": 2000, "max": None,
}


class StooqProvider(MarketDataProvider):
    """Fallback market data source used when yfinance fails or rate-limits.
    Stooq has no dedicated quote endpoint, so a "quote" here is just the
    most recent daily close from its historical CSV — always labeled
    end-of-day, never presented as live."""

    name = "stooq"

    def get_quote(self, ticker: str) -> DataResult:
        hist_result = self.get_historical(ticker, period="5d")
        if not hist_result.success or hist_result.data is None or hist_result.data.empty:
            return DataResult(
                data=None,
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day",
                success=False,
                error=hist_result.error or "Stooq has no data for this ticker",
                quality="ERROR",
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
        return DataResult(
            data=data,
            source=self.name,
            fetched_at=datetime.now(timezone.utc),
            timeliness="end_of_day",
        )

    def get_historical(self, ticker: str, period: str, interval: str = "1d") -> DataResult:
        if interval != "1d":
            return DataResult(
                data=None, source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day", success=False,
                error="Stooq fallback only supports daily bars", quality="ERROR",
            )
        try:
            symbol = f"{ticker.lower()}.us"
            resp = requests.get(STOOQ_CSV_URL, params={"s": symbol, "i": "d"}, timeout=10)
            if resp.status_code == 404:
                # Stooq 404s on a URL that simply doesn't correspond to any
                # ticker — a clean, human-readable message instead of the
                # raw HTTP exception text (e.g. a typo like "APPL" instead
                # of "AAPL").
                raise ProviderError(f"'{ticker}' doesn't look like a valid ticker (no such symbol on Stooq)")
            resp.raise_for_status()
            text = resp.text
            if "Date" not in text or text.strip().startswith("No data"):
                raise ProviderError(f"'{ticker}' doesn't look like a valid ticker (Stooq has no data for it)")

            df = pd.read_csv(io.StringIO(text))
            df.columns = [c.lower() for c in df.columns]
            df["adj_close"] = df["close"]  # Stooq's free CSV has no adjusted series
            df = df[["date", "open", "high", "low", "close", "adj_close", "volume"]]

            days = _PERIOD_TO_DAYS.get(period)
            if days:
                df = df.tail(days)

            if df.empty:
                raise ProviderError(f"'{ticker}' doesn't look like a valid ticker (Stooq returned an empty series)")

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
                error=f"Could not reach Stooq for {ticker}: {exc}", quality="ERROR",
            )
        except Exception as exc:  # noqa: BLE001
            return DataResult(
                data=None, source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day", success=False, error=str(exc), quality="ERROR",
            )

    def search(self, query: str) -> DataResult:
        return DataResult(
            data=[],
            source=self.name,
            fetched_at=datetime.now(timezone.utc),
            timeliness="cached",
            success=False,
            error="Stooq provider does not support search",
            quality="ERROR",
        )
