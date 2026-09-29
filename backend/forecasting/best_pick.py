"""
The autonomous "what does the app currently favor most" engine.

Deliberately NOT a new scoring system — it reuses the exact same
signal scoring already used by Suggestions and Discovery, just
narrowed down to a single headline candidate. This is what actually
makes "continuously scanning, even when you're not looking" honest:
the underlying predictions are kept fresh by the existing background
scheduler (discovery sweep + suggestions refresh, both already
running on their own schedules) regardless of whether anyone is
looking at the app — this module is just a live, on-demand view over
that continuously-refreshed data, not a new thing that runs on its
own separate schedule.

Descriptive, not prescriptive, on purpose, for the same reason held
throughout the rest of this app: this surfaces what the app's own
signals currently favor most. It does not — and should not — tell
anyone to act on it. See classify_signal()/SIGNAL_DISPLAY for why this
app has never used "buy"/"sell" language anywhere else, and this is no
exception.
"""
from __future__ import annotations

from datetime import datetime, timezone

from backend.forecasting.suggestions import get_stock_suggestions, get_discovery_suggestions


def get_current_best_pick() -> dict | None:
    """The single strongest bullish candidate right now, across both
    your tracked tickers and the wider discovery universe. Returns
    None honestly if nothing bullish is currently showing anywhere —
    that's a real, meaningful state (the market's just not offering a
    strong setup right now), not an error to hide."""
    tracked = get_stock_suggestions(limit=10)
    discovered = get_discovery_suggestions(limit=10)

    tracked_bullish = tracked.get("bullish") or []
    discovered_bullish = discovered.get("bullish") or []

    all_candidates = [(c, "tracked") for c in tracked_bullish] + [(c, "discovered") for c in discovered_bullish]
    if not all_candidates:
        return None

    best, source = max(all_candidates, key=lambda pair: pair[0]["score"])

    return {
        "ticker": best["ticker"],
        "signal": best["signal"],
        "confidence": best["confidence"],
        "expected_return_pct": best["expected_return_pct"],
        "horizon_days": best["horizon_days"],
        "score": best["score"],
        "age_hours": best["age_hours"],
        "source": source,  # "tracked" (already on your watchlist/portfolio) or "discovered" (found in the wider scan)
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
