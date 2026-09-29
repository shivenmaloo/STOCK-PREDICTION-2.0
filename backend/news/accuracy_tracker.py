"""
Live, forward-looking accuracy tracking for the news sentiment engine
(backend/news/sentiment.py) — the honest answer to "how do we know
this correlates with real price moves."

We deliberately do NOT claim any retroactive validation, because we
don't have a point-in-time news archive to check historical accuracy
against (the same reason news was never fed into ML training — see
backend/forecasting/news_context.py). What we CAN do, and do here:
log every real news read the moment a live prediction computes one,
then check it against what actually happened once the same horizon
has elapsed — building a genuine, growing, non-fabricated accuracy
record starting from today forward.

This mirrors backend/forecasting/service.py's evaluate_matured_predictions()
almost exactly, on purpose — it's the same honest methodology, just
applied to news sentiment instead of the ML ensemble.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from backend.data.market_data_service import market_data_service
from backend.database.db import db_cursor

DIRECTION_DEAD_ZONE_PCT = 0.5  # matches the ML prediction accuracy tracker's own threshold, for consistency


def log_news_sentiment(ticker: str, impact_score: float, news_direction: str,
                        reference_price: float, horizon_days: int) -> None:
    """Best-effort — a logging failure must never break the prediction
    that triggered it."""
    try:
        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO news_sentiment_log (ticker, logged_at, impact_score, news_direction, reference_price, horizon_days) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ticker.upper(), datetime.now(timezone.utc).isoformat(), impact_score, news_direction,
                 reference_price, horizon_days),
            )
    except Exception:  # noqa: BLE001
        pass


def evaluate_matured_news_sentiment(ticker: Optional[str] = None) -> dict:
    """Finds logged news reads whose horizon has now elapsed and
    haven't been evaluated yet, compares the implied direction against
    the real realized return, and records the result."""
    with db_cursor() as cur:
        query = """SELECT id, ticker, logged_at, impact_score, news_direction, reference_price, horizon_days
                   FROM news_sentiment_log WHERE id NOT IN (SELECT log_id FROM news_sentiment_results)"""
        params = ()
        if ticker:
            query += " AND ticker = ?"
            params = (ticker.upper(),)
        cur.execute(query, params)
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
        # A logged "neutral" news read isn't a directional call, so it
        # can't be scored right/wrong against a directional outcome —
        # only bullish/bearish reads get a correctness verdict.
        was_correct = None if row["news_direction"] == "neutral" else int(row["news_direction"] == actual_direction)

        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO news_sentiment_results (log_id, evaluated_at, actual_return, was_correct) VALUES (?, ?, ?, ?)",
                (row["id"], datetime.now(timezone.utc).isoformat(), round(actual_return, 3), was_correct),
            )
        evaluated += 1

    return {"evaluated": evaluated, "skipped_not_yet_matured_or_no_data": skipped}


def get_news_sentiment_accuracy(ticker: Optional[str] = None) -> dict:
    with db_cursor() as cur:
        query = """SELECT l.news_direction, r.was_correct FROM news_sentiment_log l
                   JOIN news_sentiment_results r ON r.log_id = l.id
                   WHERE r.was_correct IS NOT NULL"""
        params = ()
        if ticker:
            query += " AND l.ticker = ?"
            params = (ticker.upper(),)
        cur.execute(query, params)
        rows = cur.fetchall()

    if not rows:
        return {
            "n_evaluated": 0,
            "accuracy": None,
            "note": "No news reads have matured yet — this builds up over time as real predictions are made and their horizons elapse.",
        }

    correct = sum(r["was_correct"] for r in rows)
    return {
        "n_evaluated": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "note": None,
    }
