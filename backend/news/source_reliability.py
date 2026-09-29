"""
Per-SOURCE news reliability tracking — the honest answer to "should
some news outlets count more than others."

Mirrors backend/news/accuracy_tracker.py's exact methodology
(log real reads forward, check them against real outcomes once their
horizon elapses, build up a genuine accuracy record over time) at a
finer grain: per individual source (Reuters, Seeking Alpha, a random
aggregator, etc.) rather than the aggregate reading for a ticker.

Same honesty principle throughout: we have no historical, point-in-
time archive of which outlet said what, so a source's reliability
can't be validated retroactively — only real reads, logged from today
forward and checked against real outcomes, count as evidence.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from backend.data.market_data_service import market_data_service
from backend.database.db import db_cursor

DIRECTION_DEAD_ZONE_PCT = 0.5  # matches accuracy_tracker.py's own threshold, for consistency
MIN_READINGS_FOR_RELIABLE_SOURCE_SCORE = 20  # matches the same threshold used for the aggregate news vote


def log_source_reading(source_name: str, ticker: str, impact_score: float, direction: str,
                        reference_price: float, horizon_days: int) -> None:
    """Best-effort — a logging failure must never break the prediction
    that triggered it. Only ever called for OPINIONATED articles
    (direction != 'neutral') — a source that said nothing directional
    has nothing to be scored on."""
    if direction == "neutral":
        return
    try:
        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO news_source_reliability_log "
                "(source_name, ticker, logged_at, impact_score, direction, reference_price, horizon_days) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (source_name, ticker.upper(), datetime.now(timezone.utc).isoformat(),
                 impact_score, direction, reference_price, horizon_days),
            )
    except Exception:  # noqa: BLE001
        pass


def evaluate_matured_source_readings() -> dict:
    """Finds logged source readings whose horizon has elapsed and
    haven't been evaluated yet, checks each against the real realized
    return, and records whether that SPECIFIC source's call was
    correct — independent of what the aggregate or any other source
    said about the same ticker at the same time."""
    with db_cursor() as cur:
        cur.execute("""SELECT id, source_name, ticker, logged_at, direction, reference_price, horizon_days
                       FROM news_source_reliability_log
                       WHERE id NOT IN (SELECT log_id FROM news_source_reliability_results)""")
        pending = [dict(r) for r in cur.fetchall()]

    evaluated, skipped = 0, 0
    for row in pending:
        logged = datetime.fromisoformat(row["logged_at"])
        matured_at = logged + timedelta(days=row["horizon_days"] * 1.5)  # calendar-day buffer for weekends
        if datetime.now(timezone.utc) < matured_at:
            skipped += 1
            continue

        hist = market_data_service.get_historical(row["ticker"], period="1y", interval="1d")
        if not hist.success or hist.data is None or hist.data.empty:
            skipped += 1
            continue

        df = hist.data
        logged_date = logged.date().isoformat()
        on_or_after = df[df["date"] >= logged_date]
        if len(on_or_after) <= row["horizon_days"]:
            skipped += 1
            continue

        exit_price = float(on_or_after["close"].iloc[row["horizon_days"]])
        actual_return = (exit_price / row["reference_price"] - 1) * 100
        actual_direction = ("bullish" if actual_return > DIRECTION_DEAD_ZONE_PCT
                             else "bearish" if actual_return < -DIRECTION_DEAD_ZONE_PCT else "neutral")
        was_correct = None if actual_direction == "neutral" else int(row["direction"] == actual_direction)

        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO news_source_reliability_results (log_id, evaluated_at, actual_return, was_correct) "
                "VALUES (?, ?, ?, ?)",
                (row["id"], datetime.now(timezone.utc).isoformat(), round(actual_return, 3), was_correct),
            )
        evaluated += 1

    return {"evaluated": evaluated, "skipped_not_yet_matured_or_no_data": skipped}


def get_source_reliability(source_name: str) -> dict:
    """The real, tracked accuracy for ONE specific source, across all
    tickers it's been evaluated on — mirrors
    accuracy_tracker.get_news_sentiment_accuracy()'s exact honesty
    contract: reports a real percentage once there's enough evidence,
    and says so plainly when there isn't, rather than presenting a
    number computed from a handful of reads as if it meant something
    statistically."""
    with db_cursor() as cur:
        cur.execute(
            """SELECT r.was_correct FROM news_source_reliability_log l
               JOIN news_source_reliability_results r ON r.log_id = l.id
               WHERE l.source_name = ? AND r.was_correct IS NOT NULL""",
            (source_name,),
        )
        rows = cur.fetchall()

    if len(rows) < MIN_READINGS_FOR_RELIABLE_SOURCE_SCORE:
        return {
            "source_name": source_name, "available": False, "n_evaluated": len(rows),
            "reliability": None,
            "note": f"Only {len(rows)} evaluated reading(s) so far — needs at least "
                    f"{MIN_READINGS_FOR_RELIABLE_SOURCE_SCORE} before this source's reliability means anything statistically.",
        }

    correct = sum(r["was_correct"] for r in rows)
    return {
        "source_name": source_name, "available": True, "n_evaluated": len(rows),
        "reliability": round(correct / len(rows), 4), "note": None,
    }


def get_all_tracked_sources() -> list[str]:
    """Every source that's had at least one reading logged, real or
    still pending evaluation — used to build a full reliability board
    without needing to know source names in advance."""
    with db_cursor() as cur:
        cur.execute("SELECT DISTINCT source_name FROM news_source_reliability_log ORDER BY source_name")
        return [row["source_name"] for row in cur.fetchall()]
