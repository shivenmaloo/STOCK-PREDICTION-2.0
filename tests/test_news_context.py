from datetime import datetime, timezone

from tests.conftest import business_day_anchor


class _FakeNewsService:
    def __init__(self, items=None, raise_error=False):
        self.items = items or []
        self.raise_error = raise_error

    def get_analyzed_news(self, ticker, limit=10, company_name=None):
        if self.raise_error:
            raise ConnectionError("simulated network failure")
        from backend.data.base import DataResult
        return DataResult(data=self.items, source="fake", fetched_at=datetime.now(timezone.utc), timeliness="delayed_15m")


class _FakeFundamentalsService:
    """Keeps tests fast and network-independent — real company-name
    lookup is tested separately."""
    def get_fundamentals(self, ticker):
        from backend.data.base import DataResult
        return DataResult(data={"short_name": f"{ticker} Corp"}, source="fake",
                           fetched_at=datetime.now(timezone.utc), timeliness="end_of_day")


def test_fetch_news_sentiment_returns_raw_reading_with_no_ensemble_dependency():
    """The core point of the split: fetch_news_sentiment() must work
    completely independently of any ensemble result — no
    agrees_with_model, no summary yet, just the raw sentiment reading.
    This is what makes it usable BEFORE the ensemble is built (as an
    ensemble vote), not just after."""
    from backend.forecasting.news_context import fetch_news_sentiment
    fake = _FakeNewsService([{"impact_score": 30, "published_at": datetime.now(timezone.utc).isoformat()}])
    result = fetch_news_sentiment("NVDA", news_service=fake, fundamentals_service=_FakeFundamentalsService())

    assert result["available"] is True
    assert result["news_direction"] == "bullish"
    assert "agrees_with_model" not in result
    assert "summary" not in result


def test_compare_news_to_ensemble_adds_comparison_without_refetching():
    """compare_news_to_ensemble() must work purely from an
    already-fetched news_sentiment dict — no news_service needed at
    all, proving it does zero I/O of its own."""
    from backend.forecasting.news_context import compare_news_to_ensemble
    news_sentiment = {
        "available": True, "article_count": 3, "avg_impact_score": 25.0, "median_impact_score": 22.0,
        "source_agreement_pct": 100.0, "news_direction": "bullish", "coverage_note": "",
    }
    result = compare_news_to_ensemble(news_sentiment, ensemble_direction="bullish")
    assert result["agrees_with_model"] is True
    assert "agrees with the model" in result["summary"]


def test_compare_news_to_ensemble_passes_through_unavailable_untouched():
    from backend.forecasting.news_context import compare_news_to_ensemble
    unavailable = {"available": False, "reason": "No recent news available for this ticker."}
    result = compare_news_to_ensemble(unavailable, ensemble_direction="bullish")
    assert result == unavailable


def test_get_news_context_wrapper_matches_split_functions_composed_manually():
    """The backward-compatible wrapper must produce byte-for-byte the
    same result as calling the two split functions manually in
    sequence — proving the refactor didn't change behavior, just
    reorganized it."""
    from backend.forecasting.news_context import get_news_context, fetch_news_sentiment, compare_news_to_ensemble
    fake = _FakeNewsService([{"impact_score": 30, "published_at": datetime.now(timezone.utc).isoformat()}])

    via_wrapper = get_news_context("NVDA", ensemble_direction="bullish", news_service=fake, fundamentals_service=_FakeFundamentalsService())

    fake2 = _FakeNewsService([{"impact_score": 30, "published_at": datetime.now(timezone.utc).isoformat()}])
    sentiment = fetch_news_sentiment("NVDA", news_service=fake2, fundamentals_service=_FakeFundamentalsService())
    via_manual_composition = compare_news_to_ensemble(sentiment, ensemble_direction="bullish")

    assert via_wrapper == via_manual_composition


def test_no_signal_articles_do_not_dilute_a_genuine_directional_read():
    """The actual, empirically-found root cause of 'news is almost
    never used in predictions': realistic everyday headline samples
    have MOST articles scoring exactly 0.0 (no finance-relevant
    keywords matched at all — routine announcements, not real
    non-events). Before this fix, even a genuine +88 earnings beat and
    a genuine -94 downgrade mixed into a 10-article sample still
    produced a median of exactly 0.0, because the median required the
    BULK of all 10 articles to agree — and 6 of them had no opinion at
    all. No-signal articles must be excluded from the consensus
    calculation entirely; they represent an absence of opinion, not a
    real neutral one to dilute a genuine signal with."""
    from backend.forecasting.news_context import fetch_news_sentiment
    now = datetime.now(timezone.utc).isoformat()
    items = (
        [{"impact_score": 0.0, "published_at": now} for _ in range(6)]  # routine, no-signal coverage
        + [{"impact_score": 16.2, "published_at": now}, {"impact_score": 18.0, "published_at": now}]  # mildly positive
        + [{"impact_score": 88.2, "published_at": now}]  # a genuine earnings beat
    )
    fake = _FakeNewsService(items)
    result = fetch_news_sentiment("NVDA", news_service=fake, fundamentals_service=_FakeFundamentalsService())

    assert result["available"] is True
    assert result["n_opinionated_articles"] == 3  # the 6 zero-scoring articles are correctly excluded
    assert result["median_impact_score"] > 0  # genuinely reflects the real, if mixed, positive signal
    assert result["news_direction"] == "bullish"


def test_all_no_signal_articles_produces_an_honest_neutral_not_a_crash():
    """When every single article is genuinely routine/no-signal, this
    must produce a real, honest neutral read (there simply isn't a
    directional opinion in current coverage) — not a division-by-zero
    crash, and not a fabricated non-neutral read."""
    from backend.forecasting.news_context import fetch_news_sentiment, compare_news_to_ensemble
    now = datetime.now(timezone.utc).isoformat()
    items = [{"impact_score": 0.0, "published_at": now} for _ in range(8)]
    fake = _FakeNewsService(items)
    result = fetch_news_sentiment("NVDA", news_service=fake, fundamentals_service=_FakeFundamentalsService())

    assert result["available"] is True
    assert result["n_opinionated_articles"] == 0
    assert result["news_direction"] == "neutral"
    assert result["source_agreement_pct"] is None

    # Must not crash when compared against an ensemble direction either.
    compared = compare_news_to_ensemble(result, ensemble_direction="bullish")
    assert "no directional stance" in compared["summary"] or "no opinion" in compared["summary"]


def test_source_agreement_pct_is_computed_only_over_opinionated_articles():
    """A real correctness requirement: the consensus percentage must
    reflect agreement among articles that actually had an opinion, not
    be diluted by counting no-signal articles as agreeing or
    disagreeing with anything."""
    from backend.forecasting.news_context import fetch_news_sentiment
    now = datetime.now(timezone.utc).isoformat()
    items = (
        [{"impact_score": 0.0, "published_at": now} for _ in range(5)]
        + [{"impact_score": 20.0, "published_at": now}, {"impact_score": 25.0, "published_at": now}]  # both bullish
    )
    fake = _FakeNewsService(items)
    result = fetch_news_sentiment("NVDA", news_service=fake, fundamentals_service=_FakeFundamentalsService())
    assert result["source_agreement_pct"] == 100.0  # both OPINIONATED articles agree, not 2/7


def test_proven_reliable_source_carries_more_weight_than_unproven_source(tmp_path):
    """The direct proof that per-source reliability is genuinely wired
    into the aggregation, not just tracked and ignored: a source with
    a proven, strong track record must carry more weight than an
    unproven one — this is what makes 'only trust sources with real
    impact/weight' an honest description of actual behavior."""
    from backend.config import settings
    original_db_path = settings.DATABASE_PATH
    try:
        settings.DATABASE_PATH = str(tmp_path / "test.db")
        from backend.database.db import init_db
        init_db()
        from backend.database.db import db_cursor
        from datetime import timedelta
        import numpy as np
        import pandas as pd
        from backend.data.base import DataResult, MarketDataProvider
        import backend.data.market_data_service as mds_module
        from backend.news.source_reliability import evaluate_matured_source_readings
        from backend.forecasting.news_context import _get_source_reliability_weight

        for _ in range(20):
            with db_cursor() as cur:
                cur.execute(
                    "INSERT INTO news_source_reliability_log (source_name, ticker, logged_at, impact_score, "
                    "direction, reference_price, horizon_days) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("GoodSource", "TESTCO", (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
                     25.0, "bullish", 100.0, 5),
                )

        close = np.linspace(100, 115, 30)
        dates = pd.date_range(end=business_day_anchor(), periods=30, freq="B")
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
        eval_result = evaluate_matured_source_readings()
        assert eval_result["evaluated"] == 20

        good_weight = _get_source_reliability_weight("GoodSource")
        unproven_weight = _get_source_reliability_weight("UnprovenSource")
        assert good_weight > unproven_weight
        assert good_weight == 1.5
        assert unproven_weight == 1.0
    finally:
        settings.DATABASE_PATH = original_db_path


def test_news_agrees_with_bullish_model():
    from backend.forecasting.news_context import get_news_context
    fake = _FakeNewsService([{"impact_score": 40}, {"impact_score": 25}, {"impact_score": 10}])
    result = get_news_context("NVDA", ensemble_direction="bullish", news_service=fake, fundamentals_service=_FakeFundamentalsService())
    assert result["available"] is True
    assert result["news_direction"] == "bullish"
    assert result["agrees_with_model"] is True


def test_news_contradicts_bullish_model():
    from backend.forecasting.news_context import get_news_context
    fake = _FakeNewsService([{"impact_score": -50}, {"impact_score": -30}, {"impact_score": -20}])
    result = get_news_context("NVDA", ensemble_direction="bullish", news_service=fake, fundamentals_service=_FakeFundamentalsService())
    assert result["news_direction"] == "bearish"
    assert result["agrees_with_model"] is False
    assert "counter" in result["summary"]


def test_news_fetch_failure_degrades_gracefully():
    from backend.forecasting.news_context import get_news_context
    fake = _FakeNewsService(raise_error=True)
    result = get_news_context("OBSCURETICKER", ensemble_direction="bullish", news_service=fake, fundamentals_service=_FakeFundamentalsService())
    assert result["available"] is False  # never raises, never breaks the prediction


def test_no_news_available_degrades_gracefully():
    from backend.forecasting.news_context import get_news_context
    fake = _FakeNewsService(items=[])
    result = get_news_context("OBSCURETICKER", ensemble_direction="bullish", news_service=fake, fundamentals_service=_FakeFundamentalsService())
    assert result["available"] is False


def test_neutral_comparisons_are_not_forced():
    from backend.forecasting.news_context import get_news_context
    fake = _FakeNewsService([{"impact_score": 2}, {"impact_score": -3}])
    result = get_news_context("NVDA", ensemble_direction="neutral", news_service=fake, fundamentals_service=_FakeFundamentalsService())
    assert result["news_direction"] == "neutral"
    assert result["agrees_with_model"] is None  # neither agree nor disagree — not a meaningful comparison


def test_threshold_is_configurable_via_settings(tmp_path):
    """Real, direct answer to a real question: is the +/-15 cutoff
    fixed, or adjustable? It's a setting, not a hardcoded constant —
    lowering it should classify borderline news as directional that
    the default threshold would have called neutral."""
    from backend.config import settings
    original_db_path = settings.DATABASE_PATH
    try:
        settings.DATABASE_PATH = str(tmp_path / "test.db")
        from backend.database.db import init_db
        init_db()
        from backend.settings_service import set_setting
        from backend.forecasting.news_context import get_news_context

        borderline_impact = 10  # below the default 15 threshold, above a lowered one
        fake = _FakeNewsService([{"impact_score": borderline_impact}])

        default_result = get_news_context("NVDA", ensemble_direction="bullish", news_service=fake, fundamentals_service=_FakeFundamentalsService())
        assert default_result["news_direction"] == "neutral"  # 10 < default threshold of 15

        set_setting("news_direction_threshold", "5")
        lowered_result = get_news_context("NVDA", ensemble_direction="bullish", news_service=fake, fundamentals_service=_FakeFundamentalsService())
        assert lowered_result["news_direction"] == "bullish"  # 10 > lowered threshold of 5
    finally:
        # Real test-isolation bug found via testing: without restoring
        # this, every later test in the process — not just this file —
        # would keep reading from this test's temp database and its
        # lowered threshold, silently corrupting unrelated results.
        settings.DATABASE_PATH = original_db_path


def test_threshold_lookup_failure_falls_back_to_default_of_15():
    """The settings lookup must never break the news comparison itself
    — if it fails for any reason (no DB, corrupted settings, etc.),
    this must silently fall back to the original default rather than
    crash or leave news_direction undefined."""
    from backend.forecasting.news_context import get_news_context
    # No init_db() called here — deliberately simulates settings being unavailable.
    fake = _FakeNewsService([{"impact_score": 20}])  # clearly above the default 15
    result = get_news_context("NVDA", ensemble_direction="bullish", news_service=fake, fundamentals_service=_FakeFundamentalsService())
    assert result["news_direction"] == "bullish"  # confirms the 15 default was actually applied, not a crash


def test_single_extreme_outlier_article_does_not_flip_the_classification():
    """The actual, direct answer to a real question: does one loud
    article dominate the read, or does the system reflect what most
    sources actually say? Five ordinary, mildly-mixed articles plus
    one wildly extreme outlier — a naive mean crosses the bullish
    threshold from that one article alone; the median-based
    classification this module actually uses must not."""
    from backend.forecasting.news_context import get_news_context
    now = datetime.now(timezone.utc).isoformat()
    items = [
        {"impact_score": 10, "published_at": now}, {"impact_score": -5, "published_at": now},
        {"impact_score": 5, "published_at": now}, {"impact_score": 8, "published_at": now},
        {"impact_score": -3, "published_at": now}, {"impact_score": 95, "published_at": now},  # the outlier
    ]
    fake = _FakeNewsService(items)
    result = get_news_context("NVDA", ensemble_direction="bullish", news_service=fake, fundamentals_service=_FakeFundamentalsService())

    assert result["avg_impact_score"] > 15, "Sanity check: the mean alone would have crossed the bullish threshold"
    assert result["news_direction"] != "bullish", "One extreme outlier must not single-handedly flip the classification"
    assert result["median_impact_score"] < result["avg_impact_score"]


def test_source_agreement_pct_reflects_genuine_consensus():
    """When most articles genuinely agree, source_agreement_pct should
    be high; when they're mixed, it should reflect that honestly."""
    from backend.forecasting.news_context import get_news_context
    now = datetime.now(timezone.utc).isoformat()
    # Four clearly bullish articles, one mildly bearish
    items = [
        {"impact_score": 30, "published_at": now}, {"impact_score": 25, "published_at": now},
        {"impact_score": 35, "published_at": now}, {"impact_score": 28, "published_at": now},
        {"impact_score": -5, "published_at": now},
    ]
    fake = _FakeNewsService(items)
    result = get_news_context("NVDA", ensemble_direction="bullish", news_service=fake, fundamentals_service=_FakeFundamentalsService())
    assert result["news_direction"] == "bullish"
    assert result["source_agreement_pct"] == 80.0  # 4 of 5 articles are positive


def test_weighted_median_matches_manual_calculation_for_unweighted_case():
    from backend.forecasting.news_context import _weighted_median
    # Equal weights should reduce to a plain median
    pairs = [(10, 1.0), (-5, 1.0), (5, 1.0), (8, 1.0), (-3, 1.0), (95, 1.0)]
    median = _weighted_median(pairs)
    import statistics
    assert median == statistics.median(sorted(p[0] for p in pairs))


def test_weighted_median_handles_empty_input():
    from backend.forecasting.news_context import _weighted_median
    assert _weighted_median([]) == 0.0


def test_news_context_never_appears_as_a_trained_feature():
    """The core methodological guarantee: news must never enter the
    feature matrix used for training/walk-forward validation, only the
    live, post-hoc prediction bundle."""
    from backend.forecasting.features import FEATURE_COLUMNS
    for col in FEATURE_COLUMNS:
        assert "news" not in col.lower() and "sentiment" not in col.lower()


def test_company_name_is_resolved_and_passed_to_news_search():
    """Real gap found via testing: searching news for a raw ticker like
    'NVDA' misses articles that only say 'NVIDIA' — the vast majority
    of real coverage. The company name must be looked up and passed
    through to the actual search call."""
    from backend.forecasting.news_context import get_news_context

    captured = {}

    class CapturingNewsService:
        def get_analyzed_news(self, ticker, limit=10, company_name=None):
            captured["ticker"] = ticker
            captured["company_name"] = company_name
            from backend.data.base import DataResult
            return DataResult(data=[{"impact_score": 10}], source="fake",
                               fetched_at=datetime.now(timezone.utc), timeliness="delayed_15m")

    class FakeFundamentals:
        def get_fundamentals(self, ticker):
            from backend.data.base import DataResult
            return DataResult(data={"short_name": "NVIDIA Corporation"}, source="fake",
                               fetched_at=datetime.now(timezone.utc), timeliness="end_of_day")

    get_news_context("NVDA", ensemble_direction="bullish",
                      news_service=CapturingNewsService(), fundamentals_service=FakeFundamentals())

    assert captured["ticker"] == "NVDA"
    assert captured["company_name"] == "NVIDIA Corporation"


def test_company_name_lookup_failure_falls_back_gracefully():
    """If the fundamentals lookup fails for any reason, news search
    should still proceed ticker-only rather than breaking entirely."""
    from backend.forecasting.news_context import get_news_context

    class FailingFundamentals:
        def get_fundamentals(self, ticker):
            raise ConnectionError("simulated failure")

    fake_news = _FakeNewsService([{"impact_score": 10}])
    result = get_news_context("NVDA", ensemble_direction="bullish",
                               news_service=fake_news, fundamentals_service=FailingFundamentals())
    assert result["available"] is True  # degraded gracefully, still got a result


def test_recent_article_outweighs_stale_weekend_article():
    """The core adaptation of the weekend-rollup idea for a LIVE (not
    trained) check: a stale Saturday rumor must not count equally with
    this morning's real news. Flat-mean of -60 and +40 would be -10
    (bearish-leaning); recency-weighted should let the recent +40
    dominate and land clearly positive instead."""
    from datetime import datetime, timedelta, timezone
    from backend.forecasting.news_context import get_news_context

    now = datetime.now(timezone.utc)
    items = [
        {"impact_score": -60, "published_at": (now - timedelta(days=2, hours=10)).isoformat()},
        {"impact_score": 40, "published_at": (now - timedelta(hours=2)).isoformat()},
    ]
    result = get_news_context("TESTCO", "bullish", news_service=_FakeNewsService(items),
                               fundamentals_service=_FakeFundamentalsService())
    assert result["avg_impact_score"] > 0, "Recency weighting must let the recent article dominate the stale one"


def test_weekend_spanning_window_is_flagged_transparently():
    from datetime import datetime, timedelta, timezone
    from backend.forecasting.news_context import get_news_context

    now = datetime.now(timezone.utc)
    items = [
        {"impact_score": 20, "published_at": (now - timedelta(days=2, hours=10)).isoformat()},
        {"impact_score": 25, "published_at": (now - timedelta(hours=2)).isoformat()},
    ]
    result = get_news_context("TESTCO", "bullish", news_service=_FakeNewsService(items),
                               fundamentals_service=_FakeFundamentalsService())
    assert "spans about" in result["summary"]


def test_narrow_same_day_window_is_not_flagged():
    from datetime import datetime, timedelta, timezone
    from backend.forecasting.news_context import get_news_context

    now = datetime.now(timezone.utc)
    items = [
        {"impact_score": 20, "published_at": (now - timedelta(hours=1)).isoformat()},
        {"impact_score": 25, "published_at": (now - timedelta(hours=3)).isoformat()},
    ]
    result = get_news_context("TESTCO", "bullish", news_service=_FakeNewsService(items),
                               fundamentals_service=_FakeFundamentalsService())
    assert "spans about" not in result["summary"]


def test_missing_or_malformed_published_at_degrades_gracefully():
    """An article with no timestamp, or a malformed one, must still be
    counted (at neutral weight) rather than dropped or crashing —
    losing a real, already-analyzed article over a missing date field
    would throw away genuine signal."""
    from backend.forecasting.news_context import get_news_context

    items = [
        {"impact_score": 30},  # no published_at at all
        {"impact_score": -10, "published_at": "not-a-real-date"},  # malformed
    ]
    result = get_news_context("TESTCO", "bullish", news_service=_FakeNewsService(items),
                               fundamentals_service=_FakeFundamentalsService())
    assert result["available"] is True
    assert result["article_count"] == 2


def test_weekend_span_summary_survives_downstream_truncation_in_plain_summary():
    """Real bug found and fixed during development: build_plain_summary()
    extracts a headline from news_context['summary'] by splitting on the
    FIRST em-dash. The weekend-span coverage note originally also used
    an em-dash internally, so a weekend-spanning summary got silently
    truncated mid-parenthetical the moment it reached the plain-English
    summary — cut off before ever mentioning the gap it was flagging in
    the first place. Fixed by removing the internal dash; this proves
    the fix holds through the actual downstream consumer, not just the
    news_context module in isolation."""
    from datetime import datetime, timedelta, timezone
    from backend.forecasting.news_context import get_news_context
    from backend.forecasting.summary import build_plain_summary

    now = datetime.now(timezone.utc)
    items = [
        {"impact_score": 20, "published_at": (now - timedelta(days=2, hours=10)).isoformat()},
        {"impact_score": 25, "published_at": (now - timedelta(hours=2)).isoformat()},
    ]
    news_context = get_news_context("TESTCO", "bullish", news_service=_FakeNewsService(items),
                                     fundamentals_service=_FakeFundamentalsService())
    assert "spans about" in news_context["summary"]

    plain = build_plain_summary(
        ticker="TESTCO", horizon_days=5,
        signal={"signal": "BULLISH_WATCH", "reason": "test"},
        ensemble={"direction": "bullish", "confidence": 0.6, "expected_return_pct": 1.5,
                  "model_agreement": "3/4", "low_confidence_disagreement": False},
        news_context=news_context,
        position_sizing={"suggested_position_pct": None},
        relaxed_mode=False, out_of_distribution={"is_out_of_distribution": False},
    )
    assert "weekend or holiday gap" in plain, "The gap mention must survive all the way into the plain-English summary, not get truncated"
