"""
Service layer sitting on top of MarketDataProvider implementations.

Responsibilities:
  * Query every configured provider (not just the first one) and pick
    the best-scoring result via backend.data.quality — freshness,
    completeness, and coverage, not registration order. The comparison
    itself is attached to the returned DataResult so the choice is
    never a black box.
  * Cache historical bars in SQLite so identical requests don't re-hit
    the network.
  * Never fabricate data — a failure is reported as a failure, with
    the DataResult metadata surfaced all the way to the API response.

Note on scope: yfinance and Stooq are the only two sources here with
free, legitimate, machine-readable OHLCV/quote data. Reuters, Bloomberg,
CNBC, Seeking Alpha, and MarketWatch don't publish free structured price
APIs (Reuters/Bloomberg are paid terminals; the others only offer news,
which is handled separately in backend/news/) — scraping their HTML
price pages would be fragile and risks violating their terms of
service, so this app doesn't do that.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from backend.config import settings
from backend.data.base import DataResult, MarketDataProvider
from backend.data.quality import pick_best_historical
from backend.data.stooq_provider import StooqProvider
from backend.data.twelvedata_provider import TwelveDataProvider
from backend.data.yfinance_provider import YFinanceProvider
from backend.database.db import db_cursor

HIST_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


class MarketDataService:
    def __init__(self, providers: Optional[list[MarketDataProvider]] = None):
        # No priority order here on purpose — every provider is queried
        # and scored on every cache miss; see get_historical/get_quote.
        self.providers = providers or [YFinanceProvider(), StooqProvider(), TwelveDataProvider()]

    # ---------------------------------------------------------------- quote
    def get_quote(self, ticker: str) -> DataResult:
        # A "quote" is derived from each provider's most recent daily
        # bar (see yfinance_provider/stooq_provider) — so we can reuse
        # the exact same historical comparison logic here instead of a
        # separate, less rigorous quote-specific check.
        hist_result = self.get_historical(ticker, period="5d", interval="1d")
        if not hist_result.success or hist_result.data is None or hist_result.data.empty:
            return DataResult(
                data=None, source=hist_result.source, fetched_at=datetime.now(timezone.utc),
                timeliness="delayed_15m", success=False, error=hist_result.error, quality="ERROR",
            )

        df = hist_result.data
        last, prev = df.iloc[-1], df.iloc[-2] if len(df) >= 2 else None

        # A NaN closing price (e.g. a market-holiday data gap, like the
        # 500 error this exact scenario caused on Labor Day) is not a
        # usable quote — report it as unavailable honestly rather than
        # silently passing NaN through, which would later crash JSON
        # serialization for the entire page requesting this quote.
        if pd.isna(last["close"]):
            return DataResult(
                data=None, source=hist_result.source, fetched_at=datetime.now(timezone.utc),
                timeliness="delayed_15m", success=False,
                error=f"Latest bar for {ticker} has no valid closing price (likely a market-holiday data gap).",
                quality="ERROR",
            )

        data = {
            "ticker": ticker.upper(),
            "price": float(last["close"]),
            "previous_close": float(prev["close"]) if prev is not None and pd.notna(prev["close"]) else None,
            "day_high": float(last["high"]) if pd.notna(last["high"]) else None,
            "day_low": float(last["low"]) if pd.notna(last["low"]) else None,
            "volume": int(last["volume"]) if pd.notna(last["volume"]) else None,
            "market_cap": None,
            "currency": "USD",
        }
        result = DataResult(
            data=data, source=hist_result.source, fetched_at=datetime.now(timezone.utc),
            timeliness=hist_result.timeliness,
        )
        result.provider_comparison = getattr(hist_result, "provider_comparison", None)  # type: ignore[attr-defined]
        return result

    # ------------------------------------------------------------ historical
    def get_historical(self, ticker: str, period: str, interval: str = "1d",
                        force_refresh: bool = False) -> DataResult:
        ticker = ticker.upper()

        # Minimum row counts we require the cache to already have before
        # trusting it for a given period — otherwise a ticker whose cache
        # was first populated by a short request (e.g. a 5-day quote
        # lookup) would incorrectly "satisfy" a much larger request (e.g.
        # a 1-year regime/indicator calculation) with too little history,
        # silently starving downstream analysis instead of re-fetching.
        min_rows_for_period = {
            "1d": 2, "5d": 5, "1mo": 18, "3mo": 55, "6mo": 110,
            "1y": 200, "3y": 600, "5y": 1000, "max": 1,
        }

        if not force_refresh and interval == "1d":
            cached = self._read_cache(ticker)
            required_rows = min_rows_for_period.get(period, 0)
            if cached is not None and len(cached) >= required_rows and self._cache_is_fresh(ticker):
                trimmed = self._trim_to_period(cached, period)
                return DataResult(
                    data=trimmed,
                    source="local_cache",
                    fetched_at=datetime.now(timezone.utc),
                    timeliness="cached",
                )

        # Query every provider (not just until one succeeds) so we can
        # actually compare quality rather than accept whichever
        # happened to be tried first. Always request a generously large
        # window ("5y") for daily data regardless of what the caller
        # actually asked for — this is what gets cached, and is then
        # trimmed locally to the caller's requested period. Without
        # this, whichever period request happens to populate the cache
        # first (e.g. a 5-day quote lookup) would leave the cache too
        # shallow for later, larger requests (e.g. a 1-year regime
        # calculation) to ever be satisfied from it.
        fetch_period = period if (interval != "1d" or period == "max") else "5y"

        candidates: list[tuple[str, Optional[pd.DataFrame]]] = []
        failures: list[DataResult] = []

        for provider in self.providers:
            result = provider.get_historical(ticker, fetch_period, interval)
            if result.success and result.data is not None and not result.data.empty:
                candidates.append((result.source, result.data))
            else:
                failures.append(result)

        if not candidates:
            # Every provider failed outright — return the most informative failure.
            return failures[-1] if failures else DataResult(
                data=None, source="none", fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day", success=False, error="No providers returned data", quality="ERROR",
            )

        comparison = pick_best_historical(candidates, fetch_period)
        winner_name = comparison["winner"]
        winner_df = next(df for name, df in candidates if name == winner_name)

        # Strip trailing rows with no valid closing price BEFORE caching
        # — this is the actual fix for a real, recurring bug: a market-
        # holiday data gap (e.g. Labor Day) leaves the "latest" row with
        # a NaN close, and that NaN then silently propagates into EVERY
        # downstream consumer that reads "the last row" — quotes,
        # market regime, sector rankings, the scanner, and more. Fixing
        # it once here, at the source, means no individual consumer
        # needs its own NaN guard, and none can be missed the way the
        # regime/sector endpoints were the first time this was patched
        # (that fix only covered get_quote(), not this shared source).
        # Stripped from the cache too, so a cached read is equally safe.
        if interval == "1d" and not winner_df.empty:
            valid_mask = winner_df["close"].notna()
            if not valid_mask.iloc[-1]:
                last_valid_idx = valid_mask[valid_mask].index.max() if valid_mask.any() else None
                winner_df = winner_df.loc[:last_valid_idx] if last_valid_idx is not None else winner_df.iloc[0:0]

        if interval == "1d":
            self._write_cache(ticker, winner_df, winner_name)
            winner_df = self._trim_to_period(winner_df, period)

        result = DataResult(
            data=winner_df, source=winner_name, fetched_at=datetime.now(timezone.utc),
            timeliness="end_of_day" if interval == "1d" else "delayed_15m",
        )
        result.provider_comparison = comparison["comparison"]  # type: ignore[attr-defined]
        return result

    def _read_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        with db_cursor() as cur:
            cur.execute(
                "SELECT date, open, high, low, close, adj_close, volume "
                "FROM historical_prices WHERE ticker = ? ORDER BY date ASC",
                (ticker,),
            )
            rows = cur.fetchall()
        if not rows:
            return None
        return pd.DataFrame(rows, columns=HIST_COLUMNS)

    def _cache_is_fresh(self, ticker: str) -> bool:
        with db_cursor() as cur:
            cur.execute(
                "SELECT MAX(fetched_at) as latest FROM historical_prices WHERE ticker = ?",
                (ticker,),
            )
            row = cur.fetchone()
        if not row or not row["latest"]:
            return False
        fetched_at = datetime.fromisoformat(row["latest"])
        age = datetime.now(timezone.utc) - fetched_at
        return age < timedelta(hours=settings.HISTORICAL_CACHE_TTL_HOURS)

    def _write_cache(self, ticker: str, df: pd.DataFrame, source: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                ticker, r["date"], r["open"], r["high"], r["low"],
                r["close"], r["adj_close"], int(r["volume"]) if pd.notna(r["volume"]) else None,
                source, now,
            )
            for _, r in df.iterrows()
        ]
        with db_cursor() as cur:
            cur.executemany(
                """INSERT INTO historical_prices
                   (ticker, date, open, high, low, close, adj_close, volume, source, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, date) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, adj_close=excluded.adj_close,
                     volume=excluded.volume, source=excluded.source, fetched_at=excluded.fetched_at
                """,
                rows,
            )

    @staticmethod
    def _trim_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
        days_map = {
            "1d": 2, "5d": 6, "1mo": 23, "3mo": 66, "6mo": 132,
            "1y": 260, "3y": 780, "5y": 1300, "max": None,
        }
        days = days_map.get(period)
        if days is None:
            return df
        return df.tail(days).reset_index(drop=True)

    # ------------------------------------------------------------- search
    def search(self, query: str) -> DataResult:
        last: Optional[DataResult] = None
        for provider in self.providers:
            result = provider.search(query)
            last = result
            if result.success and result.data:
                return result
        return last


market_data_service = MarketDataService()
