import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture()
def isolated_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    from backend.config import settings
    settings.DATABASE_PATH = tmp.name
    from backend.database.db import init_db
    init_db()
    yield tmp.name
    os.remove(tmp.name)


def _seed_prediction(ticker, signal, confidence, expected_return, horizon_days=5):
    from backend.database.db import db_cursor
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO predictions (ticker, horizon_days, signal, confidence, expected_return, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ticker, horizon_days, signal, confidence, expected_return, datetime.now(timezone.utc).isoformat()),
        )


def _add_watchlist(*tickers):
    from backend.database.db import db_cursor
    with db_cursor() as cur:
        for t in tickers:
            cur.execute("INSERT INTO watchlist (ticker, added_at) VALUES (?, ?)", (t, datetime.now(timezone.utc).isoformat()))


def test_picks_the_genuinely_strongest_bullish_candidate_across_both_sources(isolated_db):
    from backend.forecasting.best_pick import get_current_best_pick
    _add_watchlist("AAPL")
    _seed_prediction("AAPL", "BULLISH_WATCH", 0.5, 2.0)
    _seed_prediction("NVDA", "STRONG_BULLISH_SETUP", 0.8, 6.0)
    _seed_prediction("XOM", "BEARISH_WATCH", 0.6, -3.0)

    best = get_current_best_pick()
    assert best is not None
    assert best["ticker"] == "NVDA"
    assert best["source"] == "discovered"


def test_returns_none_honestly_when_nothing_bullish_exists(isolated_db):
    from backend.forecasting.best_pick import get_current_best_pick
    _seed_prediction("XOM", "BEARISH_WATCH", 0.6, -3.0)
    _seed_prediction("NEE", "NEUTRAL", 0.1, 0.2)

    assert get_current_best_pick() is None


def test_returns_none_with_no_predictions_at_all(isolated_db):
    from backend.forecasting.best_pick import get_current_best_pick
    assert get_current_best_pick() is None


def test_tracked_source_is_correctly_labeled(isolated_db):
    from backend.forecasting.best_pick import get_current_best_pick
    _add_watchlist("AAPL")
    _seed_prediction("AAPL", "STRONG_BULLISH_SETUP", 0.8, 5.0)

    best = get_current_best_pick()
    assert best["source"] == "tracked"


def test_result_never_recommends_action_only_describes_state(isolated_db):
    """A structural check on the app's own stated principle: the
    result must never contain imperative buy/sell language — only the
    same descriptive signal tiers used everywhere else in the app."""
    from backend.forecasting.best_pick import get_current_best_pick
    _seed_prediction("NVDA", "STRONG_BULLISH_SETUP", 0.8, 6.0)

    best = get_current_best_pick()
    assert best["signal"] in {"STRONG_BULLISH_SETUP", "BULLISH_WATCH"}
    assert "buy" not in str(best).lower()
    assert "sell" not in str(best).lower()
