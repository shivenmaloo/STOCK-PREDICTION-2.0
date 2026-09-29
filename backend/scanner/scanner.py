"""
Stock scanner (spec section 18).

Pulls the whole scanner universe in ONE batched yfinance call (instead
of one request per ticker) to stay fast and avoid the rate-limiting
issues that plague per-ticker requests. Technical/return filters run
against this batch. Fundamental filters (market cap, P/E, growth) only
apply to tickers that already have cached fundamentals — a ticker
without cached fundamentals is excluded rather than guessed at, and
the response says how many were skipped for that reason so nothing is
silently hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from backend.fundamentals.service import fundamentals_service
from backend.indicators.technical import relative_volume, rsi
from backend.scanner.universe import SCANNER_UNIVERSE


@dataclass
class ScanFilters:
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[float] = None
    min_relative_volume: Optional[float] = None
    min_rsi: Optional[float] = None
    max_rsi: Optional[float] = None
    min_return_1d: Optional[float] = None
    min_return_1w: Optional[float] = None
    min_return_1m: Optional[float] = None
    min_return_3m: Optional[float] = None
    min_return_1y: Optional[float] = None
    max_pe: Optional[float] = None
    min_revenue_growth: Optional[float] = None
    above_sma50: Optional[bool] = None
    above_sma200: Optional[bool] = None


def _fetch_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=tickers, period="1y", interval="1d",
        group_by="ticker", auto_adjust=False, threads=True, progress=False,
    )
    result = {}
    for ticker in tickers:
        try:
            df = raw[ticker].dropna(how="all") if len(tickers) > 1 else raw.dropna(how="all")
            if df is None or df.empty:
                continue
            df = df.rename(columns=str.lower)
            result[ticker] = df
        except Exception:  # noqa: BLE001 - a single bad ticker shouldn't kill the batch
            continue
    return result


def _compute_row(ticker: str, df: pd.DataFrame) -> Optional[dict]:
    if df.empty or len(df) < 5:
        return None
    close = df["close"]
    last = float(close.iloc[-1])
    volume = float(df["volume"].iloc[-1])

    def ret(n_days: int) -> Optional[float]:
        if len(close) <= n_days:
            return None
        return float((last / close.iloc[-1 - n_days] - 1) * 100)

    rel_vol_series = relative_volume(df["volume"], 20)
    rel_vol = float(rel_vol_series.iloc[-1]) if pd.notna(rel_vol_series.iloc[-1]) else None

    rsi_series = rsi(close, 14)
    rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None

    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    return {
        "ticker": ticker,
        "price": round(last, 2),
        "volume": int(volume) if volume == volume else None,
        "relative_volume": round(rel_vol, 2) if rel_vol is not None else None,
        "rsi_14": round(rsi_val, 1) if rsi_val is not None else None,
        "return_1d_pct": _round_or_none(ret(1)),
        "return_1w_pct": _round_or_none(ret(5)),
        "return_1m_pct": _round_or_none(ret(21)),
        "return_3m_pct": _round_or_none(ret(63)),
        "return_1y_pct": _round_or_none(ret(251)),
        "above_sma50": bool(sma50 and last > sma50) if sma50 else None,
        "above_sma200": bool(sma200 and last > sma200) if sma200 else None,
        "market_cap": None,
        "trailing_pe": None,
        "revenue_growth": None,
        "fundamentals_available": False,
    }


def _round_or_none(v: Optional[float]) -> Optional[float]:
    return round(v, 2) if v is not None else None


def _passes_filters(row: dict, f: ScanFilters) -> bool:
    checks = [
        (f.min_market_cap, row["market_cap"], lambda a, b: b is not None and b >= a),
        (f.max_market_cap, row["market_cap"], lambda a, b: b is not None and b <= a),
        (f.min_price, row["price"], lambda a, b: b is not None and b >= a),
        (f.max_price, row["price"], lambda a, b: b is not None and b <= a),
        (f.min_volume, row["volume"], lambda a, b: b is not None and b >= a),
        (f.min_relative_volume, row["relative_volume"], lambda a, b: b is not None and b >= a),
        (f.min_rsi, row["rsi_14"], lambda a, b: b is not None and b >= a),
        (f.max_rsi, row["rsi_14"], lambda a, b: b is not None and b <= a),
        (f.min_return_1d, row["return_1d_pct"], lambda a, b: b is not None and b >= a),
        (f.min_return_1w, row["return_1w_pct"], lambda a, b: b is not None and b >= a),
        (f.min_return_1m, row["return_1m_pct"], lambda a, b: b is not None and b >= a),
        (f.min_return_3m, row["return_3m_pct"], lambda a, b: b is not None and b >= a),
        (f.min_return_1y, row["return_1y_pct"], lambda a, b: b is not None and b >= a),
        (f.max_pe, row["trailing_pe"], lambda a, b: b is not None and b > 0 and b <= a),
        (f.min_revenue_growth, row["revenue_growth"], lambda a, b: b is not None and b >= a),
    ]
    for threshold, value, cmp in checks:
        if threshold is not None and not cmp(threshold, value):
            return False
    if f.above_sma50 is True and row["above_sma50"] is not True:
        return False
    if f.above_sma200 is True and row["above_sma200"] is not True:
        return False
    return True


def _match_reasons(row: dict, f: ScanFilters) -> list[str]:
    """Human-readable list of which filters this row actually satisfied —
    so a match isn't just an opaque pass/fail."""
    reasons = []
    if f.min_rsi is not None or f.max_rsi is not None:
        reasons.append(f"RSI {row['rsi_14']}")
    if f.min_relative_volume is not None:
        reasons.append(f"{row['relative_volume']}x relative volume")
    if f.min_return_1m is not None:
        reasons.append(f"{row['return_1m_pct']:+.1f}% over 1 month")
    if f.min_return_3m is not None:
        reasons.append(f"{row['return_3m_pct']:+.1f}% over 3 months")
    if f.above_sma50 is True:
        reasons.append("above 50-day average")
    if f.above_sma200 is True:
        reasons.append("above 200-day average")
    if f.min_price is not None:
        reasons.append(f"priced at {row['price']}")
    return reasons


def run_scan(filters: ScanFilters, universe: Optional[list[str]] = None,
             use_cached_fundamentals: bool = True, limit: int = 50) -> dict:
    universe = universe or SCANNER_UNIVERSE
    batch = _fetch_batch(universe)

    rows = []
    for ticker, df in batch.items():
        row = _compute_row(ticker, df)
        if row is None:
            continue
        rows.append(row)

    needs_fundamentals = any([
        filters.min_market_cap, filters.max_market_cap, filters.max_pe, filters.min_revenue_growth,
    ])
    fundamentals_skipped = 0
    if needs_fundamentals and use_cached_fundamentals:
        for row in rows:
            if fundamentals_service.has_cached(row["ticker"]):
                cached = fundamentals_service.get_fundamentals(row["ticker"])
                if cached.success and cached.data:
                    row["market_cap"] = cached.data.get("market_cap")
                    row["trailing_pe"] = cached.data.get("trailing_pe")
                    row["revenue_growth"] = cached.data.get("revenue_growth")
                    row["fundamentals_available"] = True
            else:
                fundamentals_skipped += 1

    matched = [r for r in rows if _passes_filters(r, filters)]
    for r in matched:
        r["match_reasons"] = _match_reasons(r, filters)
    matched.sort(key=lambda r: (r["return_1m_pct"] if r["return_1m_pct"] is not None else -999), reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(universe),
        "scanned": len(rows),
        "matched": len(matched),
        "results": matched[:limit],
        "note": (
            f"{fundamentals_skipped} tickers were excluded from fundamental filters "
            f"(market cap / P/E / revenue growth) because no cached fundamentals were "
            f"available for them yet — visit their Stock Analyzer page once to cache them."
        ) if needs_fundamentals and fundamentals_skipped else None,
    }
