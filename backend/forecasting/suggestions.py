"""
Stock suggestions — surfaces which tickers look most interesting right
now, ranked by everything the app already computes: signal strength,
confidence, and whether current news agrees. This is not a new
parallel system — it's a ranking layer over the exact same
predictions the Predictions page produces, so a suggestion here means
exactly what it would mean if you ran that ticker yourself.

Two sources, kept clearly separate:
  - "Tracked" suggestions: your own watchlist + portfolio.
  - "Discovery" suggestions: the Scanner's ~120-ticker liquid-stock
    universe, MINUS whatever's already tracked (so nothing is listed
    twice). This is what actually answers "suggest something new" —
    the tracked-only version never surfaced a ticker you hadn't
    already added yourself.

Discovery is deliberately gradual, not instant: training a real,
walk-forward-validated ensemble takes real time (tens of seconds per
ticker), so scanning ~120 tickers on demand isn't practical for
something meant to load quickly. A background job (see
run_discovery_sweep) works through a handful of the universe's
tickers every few hours, so coverage genuinely builds up over the
first day or so of running the app, rather than requiring you to sit
and wait for a big batch job — see get_discovery_coverage() for an
honest, visible read on how much of the universe has fresh data
right now.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from backend.database.db import db_cursor
from backend.scanner.universe import SCANNER_UNIVERSE

logger = logging.getLogger("stock_ai.suggestions")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False

# How strongly each signal tier counts toward a suggestion — mirrors
# the same tiers classify_signal() produces, so "why is this
# suggested" always traces back to a real signal you'd see on the
# Predictions page itself.
SIGNAL_WEIGHTS = {
    "STRONG_BULLISH_SETUP": 3,
    "BULLISH_WATCH": 1,
    "NEUTRAL": 0,
    "BEARISH_WATCH": -1,
    "HIGH_RISK_POSSIBLE_EXIT": -3,
}

MAX_PREDICTION_AGE_HOURS = 48  # don't surface a suggestion from a stale prediction


def _get_tracked_tickers() -> list[str]:
    with db_cursor() as cur:
        cur.execute("SELECT ticker FROM watchlist")
        watchlist = [r["ticker"] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT ticker FROM portfolio")
        portfolio = [r["ticker"] for r in cur.fetchall()]
    return sorted(set(watchlist) | set(portfolio))


def _get_latest_predictions() -> list[dict]:
    """Most recent prediction per ticker, regardless of horizon —
    good enough for ranking "what looks interesting right now"."""
    with db_cursor() as cur:
        cur.execute("""
            SELECT p.* FROM predictions p
            INNER JOIN (
                SELECT ticker, MAX(created_at) as max_created
                FROM predictions GROUP BY ticker
            ) latest ON p.ticker = latest.ticker AND p.created_at = latest.max_created
        """)
        return [dict(r) for r in cur.fetchall()]


def _score_predictions(candidate_tickers: set[str], predictions: list[dict], limit: int, now: datetime) -> dict:
    """Shared scoring/ranking logic used by both the tracked and
    discovery suggestion sources, so they behave identically and stay
    consistent with each other — the only difference between the two
    is which set of tickers gets passed in here."""
    scored = []
    seen_tickers = set()
    stale_tickers = []
    for p in predictions:
        if p["ticker"] not in candidate_tickers:
            continue
        seen_tickers.add(p["ticker"])
        created = datetime.fromisoformat(p["created_at"])
        age_hours = (now - created).total_seconds() / 3600
        if age_hours > MAX_PREDICTION_AGE_HOURS:
            stale_tickers.append(p["ticker"])
            continue  # too stale to suggest from without re-running it

        signal_weight = SIGNAL_WEIGHTS.get(p["signal"], 0)
        score = signal_weight * p["confidence"]

        scored.append({
            "ticker": p["ticker"],
            "signal": p["signal"],
            "confidence": round(p["confidence"], 3),
            "expected_return_pct": p["expected_return"],
            "horizon_days": p["horizon_days"],
            "age_hours": round(age_hours, 1),
            "score": round(score, 4),
        })

    never_predicted = candidate_tickers - seen_tickers
    stale_or_missing = sorted(never_predicted | set(stale_tickers))

    bullish = sorted([s for s in scored if s["score"] > 0], key=lambda s: s["score"], reverse=True)[:limit]
    risk_flags = sorted([s for s in scored if s["score"] < 0], key=lambda s: s["score"])[:limit]

    return {"bullish": bullish, "risk_flags": risk_flags, "stale_or_missing": stale_or_missing}


def get_stock_suggestions(limit: int = 6) -> dict:
    tracked = set(_get_tracked_tickers())
    if not tracked:
        return {
            "bullish": [], "risk_flags": [], "stale_or_missing": [],
            "note": "Your watchlist and portfolio are both empty — add some tickers to get suggestions.",
        }

    predictions = _get_latest_predictions()
    result = _score_predictions(tracked, predictions, limit, datetime.now(timezone.utc))
    result["note"] = None
    return result


def get_discovery_suggestions(limit: int = 6) -> dict:
    """The actual answer to 'suggest something new' — ranked over the
    Scanner's liquid-stock universe, excluding anything you've already
    tracked (that's what get_stock_suggestions is for, so nothing
    shows up twice)."""
    tracked = set(_get_tracked_tickers())
    universe = set(SCANNER_UNIVERSE) - tracked

    predictions = _get_latest_predictions()
    result = _score_predictions(universe, predictions, limit, datetime.now(timezone.utc))
    result["note"] = None
    return result


def get_discovery_coverage() -> dict:
    """How much of the discovery universe currently has a fresh
    prediction — an honest, visible measure of 'how filled-in is this
    right now', since coverage builds up gradually rather than all at
    once."""
    tracked = set(_get_tracked_tickers())
    universe = set(SCANNER_UNIVERSE) - tracked
    if not universe:
        return {"covered": 0, "total": 0, "pct": None}

    predictions = _get_latest_predictions()
    now = datetime.now(timezone.utc)
    covered = 0
    for p in predictions:
        if p["ticker"] not in universe:
            continue
        age_hours = (now - datetime.fromisoformat(p["created_at"])).total_seconds() / 3600
        if age_hours <= MAX_PREDICTION_AGE_HOURS:
            covered += 1

    return {"covered": covered, "total": len(universe), "pct": round(covered / len(universe) * 100, 1)}


def run_discovery_sweep(batch_size: int = 8, horizon_days: int = 5) -> dict:
    """Trains/refreshes a small batch of the discovery universe's
    stalest (or never-yet-trained) tickers. Meant to be called
    periodically by the background scheduler on a slow cadence — NOT
    something that tries to cover the whole universe in one run, since
    that would mean many real training runs back to back on your
    machine. Prioritizes tickers with no prediction at all, then the
    oldest ones, so coverage genuinely improves each time this runs."""
    from backend.forecasting.service import prediction_service

    tracked = set(_get_tracked_tickers())
    universe = set(SCANNER_UNIVERSE) - tracked

    predictions = _get_latest_predictions()
    last_seen = {p["ticker"]: p["created_at"] for p in predictions}

    def sort_key(ticker: str):
        # Never-predicted tickers sort first (empty string sorts before any real timestamp).
        return last_seen.get(ticker, "")

    candidates = sorted(universe, key=sort_key)[:batch_size]

    refreshed, failed = [], []
    for ticker in candidates:
        try:
            result = prediction_service.get_or_train(ticker, horizon_days)
            if result.get("success"):
                refreshed.append(ticker)
            else:
                failed.append(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.info(f"[DISCOVERY] Failed to train {ticker}: {exc}")
            failed.append(ticker)

    return {"refreshed": refreshed, "failed": failed}


def refresh_tracked_ticker_predictions(horizon_days: int = 5, max_tickers: int = 15) -> dict:
    """Runs get_or_train() (which already respects the 24h freshness
    cache — this is cheap for anything already fresh) across your
    tracked tickers, so suggestions have real data without you having
    to manually predict each one. Capped at max_tickers per run so a
    large watchlist can't turn this into a multi-minute background job
    that starves everything else."""
    from backend.forecasting.service import prediction_service

    tracked = _get_tracked_tickers()[:max_tickers]
    refreshed, failed = [], []
    for ticker in tracked:
        try:
            result = prediction_service.get_or_train(ticker, horizon_days)
            if result.get("success"):
                refreshed.append(ticker)
            else:
                failed.append(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.info(f"[SUGGESTIONS] Failed to refresh {ticker}: {exc}")
            failed.append(ticker)

    return {"refreshed": refreshed, "failed": failed}
