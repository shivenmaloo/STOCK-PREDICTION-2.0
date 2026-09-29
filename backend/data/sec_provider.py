"""
SEC EDGAR provider — a genuine primary source, completely free, no API
key required. SEC's fair-access policy just asks for a descriptive
User-Agent identifying the app (https://www.sec.gov/os/webmaster-faq#developers)
and reasonable request rates, which this respects.

Two endpoints matter here:
  - `submissions` API: metadata for every filing a company has made
    (form type, date, accession number) — used to build the filings list.
  - `companyfacts` API: every XBRL-tagged financial figure a company has
    ever reported, extracted directly from their filings by the SEC
    itself. This is what lets us pull real revenue/margin/EPS trends
    without scraping or parsing raw filing text.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from backend.data.base import DataResult, ProviderError

# SEC requests a descriptive User-Agent identifying the application and
# a contact method. SEC's compliance middleware actively rejects
# User-Agents that don't look like "AppName contact@email.com" — a
# missing "@" is a common cause of unexplained 403s from data.sec.gov.
SEC_HEADERS = {"User-Agent": "StockAIResearchTerminal/1.0 (local research tool; contact@stockai.local)"}

# `browse-edgar` is what powers https://www.sec.gov/search-filings —
# it accepts a ticker symbol directly (no need to download the full
# ~10MB ticker-to-CIK directory just to look up one company), and is
# tried first. The bulk ticker map is only a fallback for the rare
# case this lookup doesn't resolve.
BROWSE_EDGAR_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

RELEVANT_FORMS = ["10-K", "10-Q", "8-K", "DEF 14A", "4"]


class SECProvider:
    name = "sec_edgar"
    _ticker_map_cache: Optional[dict] = None
    _ticker_map_error: Optional[str] = None
    _cik_lookup_cache: dict[str, Optional[int]] = {}

    def _lookup_cik_via_browse_edgar(self, ticker: str) -> Optional[int]:
        """Fast path: ask SEC's own company-search endpoint (the same
        one behind sec.gov/search-filings) for this ticker directly.
        One small request, no large file download."""
        try:
            resp = requests.get(
                BROWSE_EDGAR_URL,
                params={"action": "getcompany", "CIK": ticker, "type": "10-K",
                        "dateb": "", "owner": "include", "count": "1", "output": "atom"},
                headers=SEC_HEADERS, timeout=10,
            )
            if resp.status_code != 200:
                return None
            match = re.search(r"<CIK>(\d+)</CIK>", resp.text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        except Exception:  # noqa: BLE001
            pass
        return None

    def _load_ticker_map(self) -> dict:
        if SECProvider._ticker_map_cache is not None:
            return SECProvider._ticker_map_cache

        # Try the persistent (SQLite) cache first — the ticker directory
        # is ~10MB and changes rarely, so a fresh network fetch on every
        # single app restart is both slow and an unnecessary point of
        # failure.
        persisted = self._read_persisted_ticker_map()
        if persisted is not None:
            SECProvider._ticker_map_cache = persisted
            return persisted

        try:
            resp = requests.get(TICKER_MAP_URL, headers=SEC_HEADERS, timeout=20)
            if resp.status_code == 403:
                raise ProviderError(
                    "SEC returned 403 Forbidden loading the ticker directory — "
                    "this usually means the User-Agent header was rejected by SEC's "
                    "automated compliance filter."
                )
            resp.raise_for_status()
            raw = resp.json()
            mapping = {entry["ticker"].upper(): entry["cik_str"] for entry in raw.values()}
            SECProvider._ticker_map_cache = mapping
            SECProvider._ticker_map_error = None
            self._write_persisted_ticker_map(mapping)
            return mapping
        except Exception as exc:  # noqa: BLE001
            SECProvider._ticker_map_error = str(exc)
            raise

    def _read_persisted_ticker_map(self) -> Optional[dict]:
        try:
            from backend.database.db import db_cursor
            with db_cursor() as cur:
                cur.execute("SELECT value, fetched_at FROM sec_cache WHERE cache_key = 'ticker_map'")
                row = cur.fetchone()
            if not row:
                return None
            fetched_at = datetime.fromisoformat(row["fetched_at"])
            if datetime.now(timezone.utc) - fetched_at > timedelta(days=7):
                return None
            return json.loads(row["value"])
        except Exception:  # noqa: BLE001
            return None

    def _write_persisted_ticker_map(self, mapping: dict) -> None:
        try:
            from backend.database.db import db_cursor
            now = datetime.now(timezone.utc).isoformat()
            with db_cursor() as cur:
                cur.execute(
                    """INSERT INTO sec_cache (cache_key, value, fetched_at) VALUES ('ticker_map', ?, ?)
                       ON CONFLICT(cache_key) DO UPDATE SET value=excluded.value, fetched_at=excluded.fetched_at""",
                    (json.dumps(mapping), now),
                )
        except Exception:  # noqa: BLE001
            pass

    def get_cik(self, ticker: str) -> Optional[int]:
        ticker = ticker.upper()
        if ticker in SECProvider._cik_lookup_cache:
            return SECProvider._cik_lookup_cache[ticker]

        # Fast path: direct ticker search against browse-edgar.
        cik = self._lookup_cik_via_browse_edgar(ticker)
        if cik is not None:
            SECProvider._cik_lookup_cache[ticker] = cik
            return cik

        # Fallback: the full ticker-to-CIK directory (covers cases where
        # browse-edgar's ticker resolution doesn't hit, e.g. some very
        # recently listed companies).
        try:
            mapping = self._load_ticker_map()
        except Exception:  # noqa: BLE001
            SECProvider._cik_lookup_cache[ticker] = None
            return None
        cik = mapping.get(ticker)
        SECProvider._cik_lookup_cache[ticker] = cik
        return cik

    def _cik_error_message(self, ticker: str) -> str:
        """Distinguishes 'the whole ticker directory failed to load'
        (a real infrastructure problem) from 'this specific ticker
        just isn't in it' (expected for some tickers) — the previous
        version conflated these into one confusing message."""
        if SECProvider._ticker_map_error:
            return f"Could not load the SEC ticker directory: {SECProvider._ticker_map_error}"
        return f"{ticker} was not found in the SEC ticker directory — it may not be SEC-registered under this exact symbol."

    def get_filings(self, ticker: str, forms: Optional[list[str]] = None, limit: int = 30) -> DataResult:
        try:
            cik = self.get_cik(ticker)
            if cik is None:
                raise ProviderError(self._cik_error_message(ticker))

            resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=SEC_HEADERS, timeout=15)
            if resp.status_code == 403:
                raise ProviderError("SEC returned 403 Forbidden fetching filings — the User-Agent header was likely rejected.")
            resp.raise_for_status()
            data = resp.json()
            recent = data.get("filings", {}).get("recent", {})

            forms_list = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])
            report_dates = recent.get("reportDate", [])

            wanted = set(forms) if forms else set(RELEVANT_FORMS)
            items = []
            for i in range(len(forms_list)):
                form = forms_list[i]
                if form not in wanted:
                    continue
                accession_nodash = accessions[i].replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_docs[i]}"
                index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/"
                items.append({
                    "form": form,
                    "filed_at": dates[i],
                    "report_date": report_dates[i] if i < len(report_dates) else None,
                    "accession_number": accessions[i],
                    "document_url": doc_url,
                    "index_url": index_url,
                })
                if len(items) >= limit:
                    break

            return DataResult(
                data=items, source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="end_of_day",
            )
        except Exception as exc:  # noqa: BLE001
            return DataResult(
                data=[], source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day", success=False, error=str(exc), quality="ERROR",
            )

    def get_company_facts(self, ticker: str) -> DataResult:
        try:
            cik = self.get_cik(ticker)
            if cik is None:
                raise ProviderError(self._cik_error_message(ticker))

            resp = requests.get(COMPANY_FACTS_URL.format(cik=cik), headers=SEC_HEADERS, timeout=25)
            if resp.status_code == 403:
                raise ProviderError("SEC returned 403 Forbidden fetching company facts — the User-Agent header was likely rejected.")
            resp.raise_for_status()
            data = resp.json()
            data["_cik"] = cik  # stash for building source links downstream
            return DataResult(
                data=data, source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="end_of_day",
            )
        except Exception as exc:  # noqa: BLE001
            return DataResult(
                data=None, source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day", success=False, error=str(exc), quality="ERROR",
            )


sec_provider = SECProvider()
