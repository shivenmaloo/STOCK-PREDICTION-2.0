"""
Objective quality scoring for comparing market-data provider results —
this is the "analysis within an analysis" the app needed: instead of a
fixed provider order with fallback-only-on-crash, every successful
result from every configured provider is scored on the same criteria,
and the highest-scoring one wins. The comparison itself is exposed to
the API/UI so the choice is never a black box.

Scoring criteria (each documented, not tuned/fitted):
  - Freshness: how many calendar days old is the most recent bar?
    Fresher data scores higher, capped so ancient data can't win just
    by having more rows.
  - Completeness: what fraction of rows have every OHLCV field
    present (no gaps/NaNs)?
  - Coverage: how many rows were actually returned relative to what
    was requested (a provider that silently truncates history scores
    lower)?
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

# Rough trading-day counts per requested period, used to judge coverage.
_EXPECTED_ROWS = {
    "1d": 1, "5d": 5, "1mo": 21, "3mo": 63, "6mo": 126,
    "1y": 252, "3y": 756, "5y": 1260, "max": None,
}


def score_historical_result(df: Optional[pd.DataFrame], period: str, source_name: str) -> dict:
    """Returns a dict with `score` (higher is better, -1 means unusable)
    and `reasons` — a human-readable trace of exactly why it scored
    that way, so the comparison is inspectable, not a black box."""
    if df is None or df.empty:
        return {"source": source_name, "score": -1.0, "reasons": ["no data returned"]}

    reasons = []
    score = 0.0

    # --- Freshness ---
    try:
        last_date = pd.to_datetime(df["date"].iloc[-1])
        age_days = (pd.Timestamp.now(tz=None).normalize() - last_date.normalize()).days
    except Exception:  # noqa: BLE001
        age_days = None

    if age_days is None:
        reasons.append("could not determine freshness")
    elif age_days <= 1:
        score += 3.0
        reasons.append(f"very fresh (last bar {age_days}d old)")
    elif age_days <= 4:
        score += 1.5
        reasons.append(f"reasonably fresh (last bar {age_days}d old)")
    else:
        reasons.append(f"stale (last bar {age_days}d old)")

    # --- Completeness ---
    cols_present = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if cols_present:
        completeness = float(df[cols_present].notna().all(axis=1).mean())
    else:
        completeness = 0.0
    score += completeness * 2.0
    reasons.append(f"{completeness * 100:.0f}% of rows have complete OHLCV data")

    # --- Coverage vs. what was requested ---
    expected = _EXPECTED_ROWS.get(period)
    if expected:
        coverage = min(len(df) / expected, 1.0)
        score += coverage * 1.0
        reasons.append(f"{len(df)} rows returned (~{coverage * 100:.0f}% of the ~{expected} expected for '{period}')")
    else:
        reasons.append(f"{len(df)} rows returned")

    return {"source": source_name, "score": round(score, 2), "reasons": reasons,
            "age_days": age_days, "completeness": round(completeness, 3), "rows": len(df)}


def pick_best_historical(candidates: list[tuple[str, Optional[pd.DataFrame]]], period: str) -> dict:
    """Scores every candidate (source_name, dataframe) pair and returns
    {"winner": source_name, "comparison": [scores...]} — comparison is
    sorted best-first so the UI can show the full reasoning, not just
    the final pick."""
    scored = [score_historical_result(df, period, name) for name, df in candidates]
    scored.sort(key=lambda s: s["score"], reverse=True)
    winner = scored[0]["source"] if scored and scored[0]["score"] > -1 else None
    return {"winner": winner, "comparison": scored}
