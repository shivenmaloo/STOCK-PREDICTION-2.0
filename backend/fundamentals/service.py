"""
Service layer for fundamentals: caches results in SQLite since
fundamentals change slowly (quarterly, roughly) and the underlying
provider is the flakiest data source in the app.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from backend.data.base import DataResult
from backend.data.fundamentals_provider import YFinanceFundamentalsProvider
from backend.database.db import db_cursor

CACHE_TTL_HOURS = 24 * 3  # fundamentals are refreshed at most every 3 days


class FundamentalsService:
    def __init__(self, provider: Optional[YFinanceFundamentalsProvider] = None):
        self.provider = provider or YFinanceFundamentalsProvider()

    def get_fundamentals(self, ticker: str, force_refresh: bool = False) -> DataResult:
        ticker = ticker.upper()

        if not force_refresh:
            cached = self._read_cache(ticker)
            if cached is not None:
                return cached

        result = self.provider.get_fundamentals(ticker)
        if result.success:
            self._write_cache(ticker, result)
            return result

        # Provider failed — fall back to a stale cache entry if we have one,
        # rather than showing nothing at all. We label it clearly as stale.
        stale = self._read_cache(ticker, ignore_ttl=True)
        if stale is not None:
            stale.quality = "WARNING"
            stale.error = f"Live fetch failed ({result.error}); showing cached data."
            return stale

        return result

    def _read_cache(self, ticker: str, ignore_ttl: bool = False) -> Optional[DataResult]:
        with db_cursor() as cur:
            cur.execute(
                "SELECT data_json, source, fetched_at FROM fundamentals "
                "WHERE ticker = ? ORDER BY as_of DESC LIMIT 1",
                (ticker,),
            )
            row = cur.fetchone()
        if not row:
            return None

        fetched_at = datetime.fromisoformat(row["fetched_at"])
        if not ignore_ttl:
            age = datetime.now(timezone.utc) - fetched_at
            if age > timedelta(hours=CACHE_TTL_HOURS):
                return None

        return DataResult(
            data=json.loads(row["data_json"]),
            source=f"{row['source']} (cached)",
            fetched_at=fetched_at,
            timeliness="cached",
        )

    def _write_cache(self, ticker: str, result: DataResult) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with db_cursor() as cur:
            cur.execute(
                """INSERT INTO fundamentals (ticker, as_of, data_json, source, fetched_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, as_of) DO UPDATE SET
                     data_json=excluded.data_json, source=excluded.source, fetched_at=excluded.fetched_at
                """,
                (ticker, now[:10], json.dumps(result.data), result.source, now),
            )

    def has_cached(self, ticker: str) -> bool:
        return self._read_cache(ticker.upper(), ignore_ttl=True) is not None


fundamentals_service = FundamentalsService()
