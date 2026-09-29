"""
Live-forward IV history tracking — the honest answer to "is current
IV cheap or expensive relative to the past 52 weeks."

We don't have a point-in-time options archive (yfinance only returns
the CURRENT chain, never historical chains), so a real 52-week
percentile can't be computed retroactively — the same limitation
already handled for news sentiment (backend/news/accuracy_tracker.py).
This logs a real ATM IV reading every time the surface is computed for
a ticker, and IV rank becomes genuinely meaningful once enough history
has accumulated — reported honestly as "building up" until then,
never faked from a single reading.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backend.database.db import db_cursor

MIN_READINGS_FOR_MEANINGFUL_RANK = 20
HISTORY_WINDOW_DAYS = 365


def log_iv_reading(ticker: str, atm_iv_30d: float | None, put_call_skew_30d: float | None) -> None:
    """Best-effort — a logging failure must never break the surface
    computation that triggered it."""
    if atm_iv_30d is None:
        return
    try:
        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO iv_history_log (ticker, logged_at, atm_iv_30d, put_call_skew_30d) VALUES (?, ?, ?, ?)",
                (ticker.upper(), datetime.now(timezone.utc).isoformat(), atm_iv_30d, put_call_skew_30d),
            )
    except Exception:  # noqa: BLE001
        pass


def get_iv_rank(ticker: str, current_atm_iv: float | None) -> dict:
    """Returns where current_atm_iv sits relative to this ticker's own
    logged history (within the last 365 days) — a genuine percentile,
    not a fabricated one. Honestly reports 'not enough history yet'
    below MIN_READINGS_FOR_MEANINGFUL_RANK rather than presenting a
    percentile computed from a handful of points as if it meant
    something statistically."""
    if current_atm_iv is None:
        return {"available": False, "reason": "No current ATM IV to rank."}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=HISTORY_WINDOW_DAYS)).isoformat()
    with db_cursor() as cur:
        cur.execute(
            "SELECT atm_iv_30d FROM iv_history_log WHERE ticker = ? AND logged_at >= ? AND atm_iv_30d IS NOT NULL",
            (ticker.upper(), cutoff),
        )
        history = [row["atm_iv_30d"] for row in cur.fetchall()]

    if len(history) < MIN_READINGS_FOR_MEANINGFUL_RANK:
        return {
            "available": False,
            "n_readings": len(history),
            "reason": f"Only {len(history)} reading(s) logged so far — needs at least "
                      f"{MIN_READINGS_FOR_MEANINGFUL_RANK} to report a meaningful rank. This builds up automatically over time.",
        }

    below_or_equal = sum(1 for h in history if h <= current_atm_iv)
    percentile = round(below_or_equal / len(history) * 100, 1)
    return {
        "available": True,
        "percentile": percentile,
        "n_readings": len(history),
        "min_observed": round(min(history), 4),
        "max_observed": round(max(history), 4),
        "interpretation": ("historically elevated" if percentile >= 70 else
                            "historically low" if percentile <= 30 else "in its typical range"),
    }
