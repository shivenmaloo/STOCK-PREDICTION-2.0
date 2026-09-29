from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

from backend.data.base import DataResult, MarketDataProvider, ProviderError

# A worthwhile-but-unproven attempt to reduce Yahoo Finance's IP-reputation-based
# blocking, found via a real deployment: cloud hosts share IP ranges across many
# different apps, and yfinance's default request signature (including its
# User-Agent) is easy for Yahoo to fingerprint as automated traffic. This can't
# be verified as a real fix from this sandbox — Yahoo's blocking decision is
# ultimately about IP reputation, which no header alone can guarantee around —
# but a realistic browser User-Agent is a legitimate, low-risk thing to try
# that sometimes measurably helps. Shared across every request from this
# provider rather than rebuilt each time, since creating a new requests.Session
# per call would defeat the point of it looking like a persistent browser session.
_shared_session = requests.Session()
_shared_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
})

# yfinance period strings the UI is allowed to request.
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "3y", "5y", "max"}


class YFinanceProvider(MarketDataProvider):
    """Primary market data source. Free, no API key, but data is
    delayed (typically ~15 minutes) and occasionally rate-limited."""

    name = "yfinance"

    def get_quote(self, ticker: str) -> DataResult:
        # `fast_info` hits a Yahoo endpoint that's proven flaky/rate-limited
        # in practice. Recent daily bars from `history()` are far more
        # reliable, so we derive the "quote" from the latest close instead.
        # This is honestly labeled end-of-day/delayed rather than real-time.
        try:
            t = yf.Ticker(ticker, session=_shared_session)
            hist = t.history(period="5d", interval="1d", auto_adjust=True)
            if hist is None or hist.empty:
                raise ProviderError(f"No recent price data returned for {ticker}")
            hist = hist.dropna(subset=["Close"])
            if hist.empty:
                raise ProviderError(f"No valid recent closes for {ticker}")

            last = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) >= 2 else None

            market_cap = None
            try:
                market_cap = t.fast_info.get("market_cap")
            except Exception:  # noqa: BLE001 - market cap is a nice-to-have, never fatal
                pass

            data = {
                "ticker": ticker.upper(),
                "price": float(last["Close"]),
                "previous_close": float(prev["Close"]) if prev is not None else None,
                "day_high": float(last["High"]),
                "day_low": float(last["Low"]),
                "volume": int(last["Volume"]) if pd.notna(last["Volume"]) else None,
                "market_cap": market_cap,
                "currency": "USD",
            }
            return DataResult(
                data=data,
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day",
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary, must not crash caller
            return DataResult(
                data=None,
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day",
                success=False,
                error=str(exc),
                quality="ERROR",
            )

    def get_historical(self, ticker: str, period: str, interval: str = "1d") -> DataResult:
        if period not in VALID_PERIODS:
            period = "1y"
        try:
            t = yf.Ticker(ticker, session=_shared_session)
            hist = t.history(period=period, interval=interval, auto_adjust=True)
            if hist is None or hist.empty:
                raise ProviderError(f"No historical data returned for {ticker}")

            hist = hist.reset_index()
            date_col = "Date" if "Date" in hist.columns else "Datetime"
            # Real, structural bug found via testing: previously fetched
            # with auto_adjust=False, meaning "close" (and open/high/low)
            # were RAW, unadjusted prices, while a separate "adj_close"
            # was fetched but never actually used anywhere downstream —
            # every feature, indicator, and backtest P&L calculation used
            # the unadjusted "close". For any ticker with a real stock
            # split in its history, this silently produced a massive,
            # entirely artificial price "jump" with zero real economic
            # meaning — proven directly: a synthetic 1-for-20 reverse
            # split alone distorted a true -28.5% return into a fake
            # +1329.9% one. With auto_adjust=True, yfinance returns
            # OHLC values that are ALL consistently split-and-dividend
            # adjusted (Close and Adj Close become the same series), so
            # "close" is now the economically correct price throughout.
            df = pd.DataFrame({
                "date": pd.to_datetime(hist[date_col]).dt.strftime("%Y-%m-%d"),
                "open": hist["Open"],
                "high": hist["High"],
                "low": hist["Low"],
                "close": hist["Close"],
                "adj_close": hist["Close"],  # identical to close now that everything is pre-adjusted;
                # kept as its own column since downstream code/tests already expect it to exist
                "volume": hist["Volume"],
            })
            return DataResult(
                data=df,
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                timeliness="end_of_day" if interval == "1d" else "delayed_15m",
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

    def search(self, query: str) -> DataResult:
        try:
            results = yf.Search(query, max_results=10).quotes
            data = [
                {
                    "ticker": r.get("symbol"),
                    "name": r.get("shortname") or r.get("longname"),
                    "exchange": r.get("exchange"),
                    "type": r.get("quoteType"),
                }
                for r in results
                if r.get("symbol")
            ]
            return DataResult(
                data=data,
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                timeliness="cached",
            )
        except Exception as exc:  # noqa: BLE001
            return DataResult(
                data=[],
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                timeliness="cached",
                success=False,
                error=str(exc),
                quality="ERROR",
            )
