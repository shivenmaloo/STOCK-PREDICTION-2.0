from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from backend.data.base import DataResult
from backend.data.sec_provider import sec_provider
from backend.database.db import db_cursor
from backend.sec.analysis import build_financial_trend

FILINGS_CACHE_TTL_HOURS = 12
FACTS_CACHE_TTL_HOURS = 24 * 3  # XBRL facts only change when a new filing lands


class SECService:
    def get_filings(self, ticker: str, force_refresh: bool = False) -> DataResult:
        ticker = ticker.upper()
        cache_key = f"sec_filings:{ticker}"

        if not force_refresh:
            cached = self._read_cache(cache_key, FILINGS_CACHE_TTL_HOURS)
            if cached is not None:
                return cached

        result = sec_provider.get_filings(ticker)
        if result.success:
            self._write_cache(cache_key, result)
        return result

    def get_financial_trend(self, ticker: str, force_refresh: bool = False) -> dict:
        ticker = ticker.upper()
        cache_key = f"sec_facts:{ticker}"

        cik = sec_provider.get_cik(ticker)
        if cik is None:
            return {
                "success": False,
                "error": f"{ticker} does not appear to be SEC-registered under this ticker, or the SEC ticker map is unavailable.",
                "trend": None,
            }

        result: Optional[DataResult] = None
        if not force_refresh:
            result = self._read_cache(cache_key, FACTS_CACHE_TTL_HOURS)

        if result is None:
            result = sec_provider.get_company_facts(ticker)
            if result.success:
                self._write_cache(cache_key, result)

        if not result.success or not result.data:
            return {"success": False, "error": result.error, "trend": None}

        trend = build_financial_trend(result.data, cik)
        return {"success": True, "error": None, "trend": trend, "meta": result.meta()}

    def _read_cache(self, key: str, ttl_hours: int) -> Optional[DataResult]:
        with db_cursor() as cur:
            cur.execute(
                "SELECT value, fetched_at FROM sec_cache WHERE cache_key = ?",
                (key,),
            )
            row = cur.fetchone()
        if not row:
            return None
        fetched_at = datetime.fromisoformat(row["fetched_at"])
        if datetime.now(timezone.utc) - fetched_at > timedelta(hours=ttl_hours):
            return None
        return DataResult(
            data=json.loads(row["value"]), source="sec_edgar (cached)",
            fetched_at=fetched_at, timeliness="cached",
        )

    def _write_cache(self, key: str, result: DataResult) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with db_cursor() as cur:
            cur.execute(
                """INSERT INTO sec_cache (cache_key, value, fetched_at) VALUES (?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET value=excluded.value, fetched_at=excluded.fetched_at""",
                (key, json.dumps(result.data), now),
            )


sec_service = SECService()
