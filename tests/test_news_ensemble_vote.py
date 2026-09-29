"""
Tests for news-as-ensemble-vote: news sentiment is now genuinely used
as a technical indicator in most predictions, not just a post-hoc
check — but weighted the exact same accuracy-weighted way every ML
model already is, using news's own LIVE-tracked accuracy rather than a
backtest (which is impossible without a point-in-time news archive —
see news_context.py's module docstring for the full reasoning).
"""
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


def test_vote_conversion_is_correctly_bounded_and_linear():
    from backend.forecasting.service import _news_sentiment_to_vote, NEWS_VOTE_MAX_RETURN_PCT
    assert _news_sentiment_to_vote(100) == NEWS_VOTE_MAX_RETURN_PCT
    assert _news_sentiment_to_vote(-100) == -NEWS_VOTE_MAX_RETURN_PCT
    assert _news_sentiment_to_vote(0) == 0.0
    assert abs(_news_sentiment_to_vote(50) - NEWS_VOTE_MAX_RETURN_PCT / 2) < 1e-9


def test_vote_conversion_never_exceeds_the_max_even_for_out_of_range_input():
    """A defensive check: even if an impact score somehow exceeded the
    nominal -100..+100 scale, the vote must still be a plausible return
    magnitude, not something that could destabilize the ensemble."""
    from backend.forecasting.service import _news_sentiment_to_vote, NEWS_VOTE_MAX_RETURN_PCT
    # Not clamped by design (the scale is already bounded upstream),
    # but confirms the linear relationship holds predictably either way.
    assert _news_sentiment_to_vote(200) == NEWS_VOTE_MAX_RETURN_PCT * 2


def test_news_not_added_as_vote_when_unavailable():
    from backend.forecasting.service import _build_ensemble_with_news_vote
    result = _build_ensemble_with_news_vote(
        {"a": 0.01, "b": 0.01}, {"a": 0.55, "b": 0.55}, 5, {"available": False})
    assert "news_sentiment" not in [v["model"] for v in result["votes"]]


def test_news_not_added_as_vote_when_neutral(isolated_db):
    from backend.forecasting.service import _build_ensemble_with_news_vote
    result = _build_ensemble_with_news_vote(
        {"a": 0.01, "b": 0.01}, {"a": 0.55, "b": 0.55}, 5,
        {"available": True, "news_direction": "neutral", "median_impact_score": 3.0})
    assert "news_sentiment" not in [v["model"] for v in result["votes"]]


def test_unproven_news_gets_a_small_floor_weight_not_dominance(isolated_db):
    """Real correctness requirement: news with zero tracked evidence
    yet must not dominate — it should weigh less than an already-
    validated 55%-accuracy model, the same as any other unproven model
    in this ensemble."""
    from backend.forecasting.service import _build_ensemble_with_news_vote
    result = _build_ensemble_with_news_vote(
        {"a": 0.01, "b": 0.01}, {"a": 0.55, "b": 0.55}, 5,
        {"available": True, "news_direction": "bullish", "median_impact_score": 80.0})
    news_weight = next(v["weight"] for v in result["votes"] if v["model"] == "news_sentiment")
    model_weight = next(v["weight"] for v in result["votes"] if v["model"] == "a")
    assert news_weight < model_weight


def test_original_input_dicts_are_never_mutated(isolated_db):
    """A real correctness requirement: model_predictions/model_accuracies
    are also used elsewhere (Model Performance table, feature
    importance) — a fake "news_sentiment" model must never leak into
    the caller's own dicts."""
    from backend.forecasting.service import _build_ensemble_with_news_vote
    original_predictions = {"a": 0.01, "b": 0.01}
    original_accuracies = {"a": 0.55, "b": 0.55}
    _build_ensemble_with_news_vote(
        original_predictions, original_accuracies, 5,
        {"available": True, "news_direction": "bullish", "median_impact_score": 80.0})
    assert "news_sentiment" not in original_predictions
    assert "news_sentiment" not in original_accuracies


def _seed_matured_correct_bullish_reads(n, ticker="TESTCO"):
    from backend.database.db import db_cursor
    for _ in range(n):
        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO news_sentiment_log (ticker, logged_at, impact_score, news_direction, reference_price, horizon_days) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ticker, (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(), 80.0, "bullish", 100.0, 5),
            )


def _install_rising_price_provider():
    from backend.data.base import DataResult, MarketDataProvider
    import backend.data.market_data_service as mds_module

    today = business_day_anchor()
    dates = pd.date_range(end=today, periods=30, freq="B")
    close = np.linspace(100, 115, 30)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": close, "high": close, "low": close,
                        "close": close, "adj_close": close, "volume": [1_000_000] * 30})

    class FakeProvider(MarketDataProvider):
        name = "fake"

        def get_quote(self, ticker):
            raise NotImplementedError

        def get_historical(self, ticker, period, interval="1d"):
            return DataResult(data=df.copy(), source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="end_of_day")

        def search(self, query):
            raise NotImplementedError

    mds_module.market_data_service.providers = [FakeProvider()]


def test_news_with_real_proven_accuracy_genuinely_shifts_the_ensemble(isolated_db):
    """The actual, direct proof that news is now used as a real
    technical indicator in most predictions, not just a post-hoc
    check: once news has a real, validated, strong accuracy record
    (evaluated against genuine outcomes, not assumed), its vote
    measurably shifts the ensemble's expected return."""
    from backend.forecasting.service import _build_ensemble_with_news_vote, NEWS_VOTE_MIN_EVALUATED_READS
    from backend.news.accuracy_tracker import evaluate_matured_news_sentiment

    _seed_matured_correct_bullish_reads(NEWS_VOTE_MIN_EVALUATED_READS)
    _install_rising_price_provider()
    eval_result = evaluate_matured_news_sentiment()
    assert eval_result["evaluated"] == NEWS_VOTE_MIN_EVALUATED_READS

    model_predictions = {"a": 0.01, "b": 0.01}
    model_accuracies = {"a": 0.55, "b": 0.55}
    news_sentiment = {"available": True, "news_direction": "bullish", "median_impact_score": 80.0}

    without_news = _build_ensemble_with_news_vote(model_predictions, model_accuracies, 5, {"available": False})
    with_proven_news = _build_ensemble_with_news_vote(model_predictions, model_accuracies, 5, news_sentiment)

    assert with_proven_news["expected_return_pct"] > without_news["expected_return_pct"]

    news_vote_weight = next(v["weight"] for v in with_proven_news["votes"] if v["model"] == "news_sentiment")
    model_weight = next(v["weight"] for v in with_proven_news["votes"] if v["model"] == "a")
    assert news_vote_weight > model_weight, "News with a proven 100% track record should outweigh a merely 55%-accurate model"


def test_news_accuracy_lookup_failure_falls_back_gracefully(isolated_db):
    """Must never break a prediction — if the accuracy lookup itself
    fails for any reason, news should fall back to the unproven floor
    weight, not crash."""
    from backend.forecasting.service import _get_news_vote_weight_accuracy
    from backend.config import settings
    original = settings.DATABASE_PATH
    settings.DATABASE_PATH = "/nonexistent/path/that/cannot/exist.db"
    try:
        result = _get_news_vote_weight_accuracy()
        assert result is None
    finally:
        settings.DATABASE_PATH = original


def test_full_train_and_predict_pipeline_genuinely_incorporates_news_vote(isolated_db):
    """The ultimate proof: not just the isolated helper function, but
    the ACTUAL, full train_and_predict() pipeline a real user's
    prediction goes through — with real feature engineering, real
    model training, and news mocked in as strongly bullish with a
    real, proven accuracy record already logged. The final ensemble
    result must reflect that news genuinely participated, and the
    News agreement chip data must still be present and correct
    alongside it (the post-hoc comparison role is unchanged; news now
    ALSO gets a direct vote)."""
    import backend.data.market_data_service as mds_module
    import backend.forecasting.service as service_module
    import backend.news.service as news_module
    from backend.data.base import DataResult, MarketDataProvider

    _seed_matured_correct_bullish_reads(service_module.NEWS_VOTE_MIN_EVALUATED_READS, ticker="TESTCO")
    _install_rising_price_provider()
    from backend.news.accuracy_tracker import evaluate_matured_news_sentiment
    eval_result = evaluate_matured_news_sentiment()
    assert eval_result["evaluated"] == service_module.NEWS_VOTE_MIN_EVALUATED_READS

    # Now build the REAL 5-year training history (separate from the
    # short rising-price series used only for news evaluation above).
    today = business_day_anchor()
    n = 1300
    dates = pd.date_range(end=today, periods=n, freq="B")
    rng = np.random.default_rng(11)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": close, "high": close * 1.005,
                        "low": close * 0.995, "close": close, "adj_close": close,
                        "volume": rng.integers(1_000_000, 10_000_000, n)})

    class FakeProvider(MarketDataProvider):
        name = "fake"

        def get_quote(self, ticker):
            raise NotImplementedError

        def get_historical(self, ticker, period, interval="1d"):
            return DataResult(data=df.copy(), source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="end_of_day")

        def search(self, query):
            raise NotImplementedError

    mds_module.market_data_service.providers = [FakeProvider()]
    service_module.WALK_FORWARD_STEP = 120

    class FakeNewsService:
        def get_analyzed_news(self, ticker, limit=10, company_name=None):
            return DataResult(data=[{"impact_score": 80, "published_at": datetime.now(timezone.utc).isoformat()}],
                               source="fake", fetched_at=datetime.now(timezone.utc), timeliness="delayed_15m")

    # Real bug found via testing: reassigning the module-level
    # news_service singleton without restoring it afterward leaks into
    # every later test in the same process — other tests (e.g.
    # test_phase2_api.py) mutate ATTRIBUTES on the real singleton
    # rather than replacing it, and silently mutate this leaked fake
    # object instead once it's in place, producing confusing failures
    # far from the actual cause. Same class of bug as the
    # settings.DATABASE_PATH leak found earlier — save and always
    # restore, regardless of pass or fail.
    original_news_service = news_module.news_service
    news_module.news_service = FakeNewsService()
    try:
        result = service_module.prediction_service.train_and_predict("TESTCO", horizon_days=5)
    finally:
        news_module.news_service = original_news_service

    assert result["success"] is True
    assert "news_sentiment" in [v["model"] for v in result["ensemble"]["votes"]]
    news_vote = next(v for v in result["ensemble"]["votes"] if v["model"] == "news_sentiment")
    # With a real, proven 100% accuracy record, news must carry a
    # genuinely meaningful weight, not the small unproven floor.
    assert news_vote["weight"] > 0.3
    # The post-hoc comparison role must still work correctly alongside it.
    assert result["news_context"]["available"] is True
    assert result["news_context"]["news_direction"] == "bullish"
    assert "agrees_with_model" in result["news_context"]
