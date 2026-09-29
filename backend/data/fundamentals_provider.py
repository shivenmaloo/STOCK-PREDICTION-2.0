from __future__ import annotations

from datetime import datetime, timezone

import yfinance as yf

from backend.data.base import DataResult, FundamentalsProvider, ProviderError

# The subset of yfinance's `.info` dict we actually use, mapped to
# stable internal field names so the rest of the app never touches
# yfinance's (occasionally inconsistent) raw key names directly.
_FIELD_MAP = {
    "market_cap": "marketCap",
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "peg_ratio": "trailingPegRatio",
    "price_to_sales": "priceToSalesTrailing12Months",
    "price_to_book": "priceToBook",
    "ev_to_ebitda": "enterpriseToEbitda",
    "dividend_yield": "dividendYield",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "gross_margin": "grossMargins",
    "operating_margin": "operatingMargins",
    "net_margin": "profitMargins",
    "return_on_equity": "returnOnEquity",
    "return_on_assets": "returnOnAssets",
    "debt_to_equity": "debtToEquity",
    "current_ratio": "currentRatio",
    "free_cashflow": "freeCashflow",
    "total_cash": "totalCash",
    "total_debt": "totalDebt",
    "shares_outstanding": "sharesOutstanding",
    "beta": "beta",
    "sector": "sector",
    "industry": "industry",
    "short_name": "shortName",
    "long_name": "longName",
}


class YFinanceFundamentalsProvider(FundamentalsProvider):
    """Fundamentals are the least reliable data yfinance offers — the
    underlying Yahoo `quoteSummary` endpoint is the same one that's
    proven flaky for live quotes. We surface failures honestly rather
    than guessing, and cache successes in SQLite (see FundamentalsService)
    since this data changes slowly and is expensive to re-fetch."""

    name = "yfinance"

    def get_fundamentals(self, ticker: str) -> DataResult:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if not info or not isinstance(info, dict) or len(info) < 3:
                raise ProviderError(f"No fundamentals data returned for {ticker}")

            data = {key: info.get(src_key) for key, src_key in _FIELD_MAP.items()}
            # A ticker with literally nothing populated is treated as a failure,
            # not a "GOOD" result with every field null.
            if all(v is None for v in data.values()):
                raise ProviderError(f"Fundamentals fields all empty for {ticker}")

            return DataResult(
                data=data,
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day",
            )
        except Exception as exc:  # noqa: BLE001
            return DataResult(
                data=None,
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day",
                success=False,
                error=str(exc),
                quality="ERROR",
            )
