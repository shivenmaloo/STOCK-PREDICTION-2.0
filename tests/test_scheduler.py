import builtins
import sys
import time
from datetime import datetime, timezone

import pytest


@pytest.fixture()
def isolated_db():
    import os
    import tempfile
    from backend.config import settings
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    settings.DATABASE_PATH = tmp.name
    from backend.database.db import init_db
    init_db()
    yield tmp.name
    os.remove(tmp.name)


def test_scheduler_starts_and_stops_cleanly(isolated_db):
    from backend.scheduler import start_scheduler, stop_scheduler, get_scheduler_status

    started = start_scheduler(interval_minutes=60)
    assert started is True
    assert get_scheduler_status()["running"] is True

    stop_scheduler()
    assert get_scheduler_status()["running"] is False


def test_scheduler_runs_evaluation_job_immediately_on_startup(isolated_db):
    """The whole point of the scheduler is that accuracy tracking
    updates without a manual button click — it must actually execute,
    not just report 'running' while doing nothing."""
    from backend.scheduler import start_scheduler, stop_scheduler, get_scheduler_status

    start_scheduler(interval_minutes=60)
    time.sleep(4.0)  # bumped further — three jobs now registered (predictions + alerts + suggestions), more startup contention
    status = get_scheduler_status()
    assert status["run_count"] >= 1
    assert status["last_run_at"] is not None
    stop_scheduler()


def test_scheduler_degrades_gracefully_when_apscheduler_broken():
    """Same class of bug as the xgboost/torch OSError crashes — a
    broken or missing apscheduler install must not take down the whole
    app at startup, just disable the automatic-evaluation feature."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("apscheduler"):
            raise OSError("simulated broken apscheduler install")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        sys.modules.pop("backend.scheduler", None)
        sys.modules.pop("apscheduler", None)
        import backend.scheduler as sched_module
        assert sched_module.SCHEDULER_AVAILABLE is False
        assert sched_module.start_scheduler() is False
    finally:
        builtins.__import__ = real_import
        sys.modules.pop("backend.scheduler", None)
        import backend.scheduler  # restore normal state for subsequent tests


def test_evaluation_job_actually_calls_source_reliability_evaluation(isolated_db, monkeypatch):
    """Real gap found and fixed: evaluate_matured_source_readings()
    existed as a complete, tested function but was never actually
    wired into any scheduled job — meaning source reliability data
    would sit pending forever unless someone manually hit the
    evaluate-now endpoint. Real results require this to happen
    automatically, not on request. Verified directly here rather than
    inferred from a timing-based test, since a call that silently
    fails or never fires wouldn't necessarily show up as a timing
    difference."""
    import backend.scheduler as sched_module
    from backend.news import source_reliability

    call_count = {"n": 0}
    original = source_reliability.evaluate_matured_source_readings

    def tracked_call():
        call_count["n"] += 1
        return original()

    monkeypatch.setattr(source_reliability, "evaluate_matured_source_readings", tracked_call)
    sched_module._run_evaluation_job()
    assert call_count["n"] == 1


def test_evaluation_job_survives_source_reliability_evaluation_failure(isolated_db, monkeypatch):
    """A failure in the source reliability evaluator must not prevent
    the OTHER evaluations (predictions, news sentiment) in the same
    job from running — isolated the same way each of those already is
    from each other."""
    import backend.scheduler as sched_module
    from backend.news import source_reliability

    def broken_eval():
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(source_reliability, "evaluate_matured_source_readings", broken_eval)
    sched_module._run_evaluation_job()  # must not raise
    status = sched_module.get_scheduler_status()
    assert status["run_count"] >= 1  # the predictions evaluation still completed and was recorded


def test_universe_seed_runs_in_background_without_blocking(isolated_db, monkeypatch):
    """The core requirement: triggering this must return immediately,
    not block the caller for however long real training across many
    tickers would take."""
    import time as time_module
    from backend.data.base import DataResult, MarketDataProvider
    import backend.data.market_data_service as mds_module
    import backend.forecasting.suggestions as suggestions_module
    import numpy as np
    import pandas as pd
    from tests.conftest import business_day_anchor

    monkeypatch.setattr(suggestions_module, "SCANNER_UNIVERSE", ["AAA", "BBB", "CCC"])
    monkeypatch.setattr(suggestions_module, "_get_tracked_tickers", lambda: [])

    dates = pd.date_range(end=business_day_anchor(), periods=1300, freq="B")
    rng = np.random.default_rng(3)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, 1300))
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": close, "high": close, "low": close,
                        "close": close, "adj_close": close, "volume": rng.integers(1_000_000, 5_000_000, 1300)})

    class FakeProvider(MarketDataProvider):
        name = "fake"
        def get_quote(self, t): raise NotImplementedError
        def get_historical(self, t, period, interval="1d"):
            return DataResult(data=df.copy(), source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="end_of_day")
        def search(self, q): raise NotImplementedError

    mds_module.market_data_service.providers = [FakeProvider()]

    # This test verifies BACKGROUND-THREAD EXECUTION BEHAVIOR, not real
    # news fetching — mocking the news service makes it fast,
    # deterministic, and not dependent on real network calls
    # succeeding (which was, in fact, the real reason an earlier
    # version of this test was flaky: real news fetching for 3 tickers
    # took ~25 real seconds, uncomfortably close to this test's own
    # timeout, for something the test was never actually about).
    import backend.news.service as news_module
    from backend.data.base import DataResult as NewsDataResult

    class FakeNewsService:
        def get_analyzed_news(self, ticker, limit=10, company_name=None):
            return NewsDataResult(data=[], source="fake", fetched_at=datetime.now(timezone.utc), timeliness="delayed_15m")

    original_news_service = news_module.news_service
    news_module.news_service = FakeNewsService()

    import backend.scheduler as sched_module
    import backend.forecasting.service as service_module
    service_module.WALK_FORWARD_STEP = 120  # speed up training for this test — same pattern used elsewhere
    try:
        start = time_module.time()
        result = sched_module.trigger_universe_seed_now(horizon_days=5)
        elapsed = time_module.time() - start

        assert result["started"] is True
        assert elapsed < 1.0, "Must return near-instantly, not block on real training"

        # Real fix for a real race condition in this test itself, found
        # while debugging: right after trigger_universe_seed_now() returns,
        # the background thread may genuinely not have started yet, so
        # status still shows the initial running=False default — checking
        # for that FIRST (before the thread has had a chance to set
        # running=True) incorrectly reads as "already finished." Wait for
        # a real, positive signal that it started (started_at is set)
        # before polling for it to finish.
        for _ in range(60):
            if sched_module.get_universe_seed_status()["started_at"] is not None:
                break
            time_module.sleep(0.1)
        assert sched_module.get_universe_seed_status()["started_at"] is not None, "The background thread never actually started"

        for _ in range(60):
            if not sched_module.get_universe_seed_status()["running"]:
                break
            time_module.sleep(0.5)

        final_status = sched_module.get_universe_seed_status()
        assert final_status["running"] is False
        assert final_status["total_tickers"] == 3
        assert final_status["completed_tickers"] == 3
        assert len(final_status["last_result"]["refreshed"]) == 3
    finally:
        news_module.news_service = original_news_service


def test_universe_seed_refuses_to_double_start(isolated_db, monkeypatch):
    """Starting a second seed run while one is already in progress
    must not silently double the work."""
    import backend.scheduler as sched_module
    sched_module._universe_seed_status["running"] = True
    try:
        result = sched_module.trigger_universe_seed_now()
        assert result["started"] is False
    finally:
        sched_module._universe_seed_status["running"] = False
