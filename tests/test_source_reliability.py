import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tests.conftest import business_day_anchor


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


def _seed_log(source_name, ticker, impact_score, direction, reference_price, horizon_days=5, days_ago=10):
    from backend.database.db import db_cursor
    logged_time = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO news_source_reliability_log (source_name, ticker, logged_at, impact_score, direction, "
            "reference_price, horizon_days) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source_name, ticker, logged_time, impact_score, direction, reference_price, horizon_days),
        )


def _install_provider(close_prices):
    from backend.data.base import DataResult, MarketDataProvider
    import backend.data.market_data_service as mds_module

    today = business_day_anchor()
    dates = pd.date_range(end=today, periods=len(close_prices), freq="B")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": close_prices, "high": close_prices,
                        "low": close_prices, "close": close_prices, "adj_close": close_prices,
                        "volume": [1_000_000] * len(close_prices)})

    class FakeProvider(MarketDataProvider):
        name = "fake"

        def get_quote(self, ticker):
            raise NotImplementedError

        def get_historical(self, ticker, period, interval="1d"):
            return DataResult(data=df.copy(), source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="end_of_day")

        def search(self, query):
            raise NotImplementedError

    mds_module.market_data_service.providers = [FakeProvider()]


def test_log_source_reading_writes_a_row(isolated_db):
    from backend.news.source_reliability import log_source_reading
    from backend.database.db import db_cursor
    log_source_reading("Reuters", "TESTCO", 30.0, "bullish", 100.0, 5)
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) as c FROM news_source_reliability_log WHERE source_name = 'Reuters'")
        assert cur.fetchone()["c"] == 1


def test_neutral_direction_is_never_logged(isolated_db):
    """A source that expressed no directional opinion has nothing to
    be evaluated on — logging it would be scoring an absence of a call."""
    from backend.news.source_reliability import log_source_reading
    from backend.database.db import db_cursor
    log_source_reading("Reuters", "TESTCO", 5.0, "neutral", 100.0, 5)
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) as c FROM news_source_reliability_log")
        assert cur.fetchone()["c"] == 0


def test_log_source_reading_never_raises_on_db_failure(isolated_db, monkeypatch):
    from backend.news import source_reliability

    def broken_cursor():
        raise RuntimeError("simulated db failure")

    monkeypatch.setattr(source_reliability, "db_cursor", broken_cursor)
    source_reliability.log_source_reading("Reuters", "TESTCO", 30.0, "bullish", 100.0, 5)  # must not raise


def test_evaluate_correctly_identifies_right_and_wrong_source_calls(isolated_db):
    from backend.news.source_reliability import evaluate_matured_source_readings, get_source_reliability

    for _ in range(20):
        _seed_log("Reuters", "TESTCO", 30.0, "bullish", 100.0)  # Reuters: right every time
    for _ in range(20):
        _seed_log("RandomBlog", "TESTCO", -30.0, "bearish", 100.0)  # RandomBlog: wrong every time
    _install_provider(np.linspace(100, 115, 30))  # price genuinely rises

    result = evaluate_matured_source_readings()
    assert result["evaluated"] == 40

    reuters = get_source_reliability("Reuters")
    blog = get_source_reliability("RandomBlog")
    assert reuters["available"] is True
    assert reuters["reliability"] == 1.0
    assert blog["available"] is True
    assert blog["reliability"] == 0.0


def test_sources_are_tracked_completely_independently(isolated_db):
    """A real correctness requirement: one source's track record must
    never bleed into another's — that's the entire point of tracking
    per-source rather than in aggregate."""
    from backend.news.source_reliability import evaluate_matured_source_readings, get_source_reliability

    for _ in range(20):
        _seed_log("GoodSource", "TESTCO", 30.0, "bullish", 100.0)
    for _ in range(20):
        _seed_log("BadSource", "TESTCO", -30.0, "bearish", 100.0)
    _install_provider(np.linspace(100, 115, 30))

    evaluate_matured_source_readings()
    good = get_source_reliability("GoodSource")
    bad = get_source_reliability("BadSource")
    assert good["reliability"] == 1.0
    assert bad["reliability"] == 0.0
    assert good["n_evaluated"] == 20
    assert bad["n_evaluated"] == 20


def test_insufficient_evidence_reports_honest_unavailable_state(isolated_db):
    from backend.news.source_reliability import get_source_reliability
    result = get_source_reliability("NeverSeenBefore")
    assert result["available"] is False
    assert result["reliability"] is None
    assert result["n_evaluated"] == 0


def test_reliability_below_threshold_reports_unavailable_even_with_some_data(isolated_db):
    """A handful of evaluated reads isn't statistically meaningful —
    must not present a percentage as if it were, matching the exact
    same principle already used for the aggregate news accuracy tracker."""
    from backend.news.source_reliability import evaluate_matured_source_readings, get_source_reliability

    for _ in range(5):  # below MIN_READINGS_FOR_RELIABLE_SOURCE_SCORE (20)
        _seed_log("NewSource", "TESTCO", 30.0, "bullish", 100.0)
    _install_provider(np.linspace(100, 115, 30))
    evaluate_matured_source_readings()

    result = get_source_reliability("NewSource")
    assert result["available"] is False
    assert result["n_evaluated"] == 5


def test_not_yet_matured_readings_are_skipped(isolated_db):
    from backend.news.source_reliability import evaluate_matured_source_readings

    _seed_log("Reuters", "TESTCO", 30.0, "bullish", 100.0, horizon_days=20, days_ago=1)  # nowhere near matured
    _install_provider(np.linspace(100, 115, 30))
    result = evaluate_matured_source_readings()
    assert result["evaluated"] == 0
    assert result["skipped_not_yet_matured_or_no_data"] == 1


def test_neutral_actual_outcome_is_never_scored_right_or_wrong(isolated_db):
    """If the real outcome itself was a dead-zone/flat move, no
    direction call — bullish or bearish — can honestly be marked
    right or wrong against it."""
    from backend.news.source_reliability import evaluate_matured_source_readings
    from backend.database.db import db_cursor

    _seed_log("Reuters", "TESTCO", 30.0, "bullish", 100.0)
    _install_provider([100.0] * 30)  # flat, no real move at all
    evaluate_matured_source_readings()

    with db_cursor() as cur:
        cur.execute("SELECT was_correct FROM news_source_reliability_results")
        row = cur.fetchone()
        assert row["was_correct"] is None


def test_get_all_tracked_sources_returns_every_distinct_source(isolated_db):
    from backend.news.source_reliability import get_all_tracked_sources
    _seed_log("Reuters", "AAA", 30.0, "bullish", 100.0)
    _seed_log("Seeking Alpha", "BBB", -20.0, "bearish", 50.0)
    _seed_log("Reuters", "CCC", 15.0, "bullish", 200.0)  # same source, different ticker

    sources = get_all_tracked_sources()
    assert set(sources) == {"Reuters", "Seeking Alpha"}
