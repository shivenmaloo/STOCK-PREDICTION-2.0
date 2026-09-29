"""
Provider abstractions (spec section 3).

Every concrete provider implements one of these interfaces so the
service layer can fall back from one source to another without the
rest of the app knowing which provider actually answered.

Phase 1 ships MarketDataProvider implementations (yfinance, Stooq).
NewsDataProvider / FundamentalsProvider / EconomicDataProvider are
defined here now so Phase 2+ providers slot in without touching the
service layer's call sites.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd


@dataclass
class DataResult:
    """
    Wraps any data returned from a provider with the honesty metadata
    the spec requires (section 3 / 25): where it came from, when it
    was fetched, and whether it's real-time, delayed, or end-of-day.
    """
    data: Any
    source: str
    fetched_at: datetime
    timeliness: str  # 'real_time' | 'delayed_15m' | 'end_of_day' | 'cached'
    success: bool = True
    error: Optional[str] = None
    quality: str = "GOOD"  # 'GOOD' | 'WARNING' | 'ERROR'

    def meta(self) -> dict:
        return {
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "timeliness": self.timeliness,
            "success": self.success,
            "error": self.error,
            "quality": self.quality,
        }


class ProviderError(Exception):
    """Raised by a provider on failure so the service layer can fall back."""


class MarketDataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def get_quote(self, ticker: str) -> DataResult:
        """Latest price snapshot for a ticker."""

    @abstractmethod
    def get_historical(self, ticker: str, period: str, interval: str = "1d") -> DataResult:
        """OHLCV history as a DataFrame with columns:
        date, open, high, low, close, adj_close, volume."""

    @abstractmethod
    def search(self, query: str) -> DataResult:
        """Ticker/name search results."""


class NewsDataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def get_news(self, ticker: str, limit: int = 20) -> DataResult:
        """Recent headlines relevant to a ticker."""


class FundamentalsProvider(ABC):
    name: str = "base"

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> DataResult:
        """Latest available fundamentals snapshot."""


class EconomicDataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def get_series(self, series_id: str) -> DataResult:
        """A named macro series (e.g. 10Y treasury yield)."""
