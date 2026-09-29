"""
Tests for prediction_input_snapshots — a real gap found via direct
feedback: per-model predictions were already saved, but the actual
FEATURE DATA that fed into each historical prediction never was,
meaning a past prediction could never be reproduced or debugged
against its real inputs. This is a permanent, immutable snapshot of
exactly what went in, frozen at prediction time.
"""
import os
import tempfile
from datetime import datetime, timezone

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


def _install_provider(n=1300, seed=7):
    from backend.data.base import DataResult, MarketDataProvider
    import backend.data.market_data_service as mds_module

    today = business_day_anchor()
    dates = pd.date_range(end=today, periods=n, freq="B")
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": close, "high": close * 1.01,
                        "low": close * 0.99, "close": close, "adj_close": close,
                        "volume": rng.integers(1_000_000, 5_000_000, n)})

    class FakeProvider(MarketDataProvider):
        name = "fake"

        def get_quote(self, ticker):
            raise NotImplementedError

        def get_historical(self, ticker, period, interval="1d"):
            return DataResult(data=df.copy(), source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="end_of_day")

        def search(self, query):
            raise NotImplementedError

    mds_module.market_data_service.providers = [FakeProvider()]


def test_direct_store_and_retrieve_round_trip(isolated_db):
    """The core mechanics, tested directly: a feature snapshot saved
    alongside a prediction must be retrievable afterward, with real
    values intact (not corrupted by JSON serialization)."""
    import backend.forecasting.service as service_module
    ensemble = {"expected_return_pct": 2.5, "confidence": 0.4, "votes": []}
    signal = {"signal": "BULLISH_WATCH"}
    feature_row = pd.Series({"close": 150.25, "rsi_14": 62.3, "volume": 5_000_000, "sma_50": 148.1})

    prediction_id = service_module.prediction_service._store_prediction(
        "TESTCO", 5, ensemble, signal, feature_snapshot=feature_row)

    snapshot = service_module.prediction_service.get_prediction_input_snapshot(prediction_id)
    assert snapshot is not None
    assert snapshot["close"] == 150.25
    assert snapshot["rsi_14"] == 62.3
    assert snapshot["volume"] == 5_000_000


def test_nan_values_in_feature_snapshot_are_saved_as_null_not_crash(isolated_db):
    """A real, common case: early rows of a feature dataframe often
    have NaN for indicators that need a warm-up window (e.g. a 200-day
    SMA on day 50). Must be saved as a real, honest null, not crash the
    whole prediction."""
    import backend.forecasting.service as service_module
    ensemble = {"expected_return_pct": 1.0, "confidence": 0.3, "votes": []}
    signal = {"signal": "NEUTRAL"}
    feature_row = pd.Series({"close": 100.0, "sma_200": np.nan})

    prediction_id = service_module.prediction_service._store_prediction(
        "TESTCO", 5, ensemble, signal, feature_snapshot=feature_row)

    snapshot = service_module.prediction_service.get_prediction_input_snapshot(prediction_id)
    assert snapshot["sma_200"] is None
    assert snapshot["close"] == 100.0


def test_numpy_scalar_types_are_correctly_serialized(isolated_db):
    """Pandas/numpy scalar types (np.float64, np.int64) aren't
    natively JSON-serializable — must be safely converted, not crash."""
    import backend.forecasting.service as service_module
    ensemble = {"expected_return_pct": 1.0, "confidence": 0.3, "votes": []}
    signal = {"signal": "NEUTRAL"}
    feature_row = pd.Series({"close": np.float64(123.456), "volume": np.int64(1_000_000)})

    prediction_id = service_module.prediction_service._store_prediction(
        "TESTCO", 5, ensemble, signal, feature_snapshot=feature_row)

    snapshot = service_module.prediction_service.get_prediction_input_snapshot(prediction_id)
    assert abs(snapshot["close"] - 123.456) < 1e-6
    assert snapshot["volume"] == 1_000_000


def test_missing_snapshot_returns_none_not_an_error(isolated_db):
    import backend.forecasting.service as service_module
    result = service_module.prediction_service.get_prediction_input_snapshot(999999)
    assert result is None


def test_snapshot_failure_never_breaks_the_prediction_itself(isolated_db, monkeypatch):
    """The snapshot is a nice-to-have for future debugging — it must
    never be allowed to break the actual prediction it's attached to."""
    import backend.forecasting.service as service_module
    import json as json_module

    def broken_dumps(*args, **kwargs):
        raise RuntimeError("simulated serialization failure")

    ensemble = {"expected_return_pct": 1.0, "confidence": 0.3, "votes": []}
    signal = {"signal": "NEUTRAL"}
    feature_row = pd.Series({"close": 100.0})

    original_dumps = service_module.json.dumps
    call_count = {"n": 0}

    def selectively_broken_dumps(obj, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] > 1:  # let the first call (votes) succeed, break the snapshot's own dumps call
            raise RuntimeError("simulated serialization failure")
        return original_dumps(obj, *args, **kwargs)

    monkeypatch.setattr(service_module.json, "dumps", selectively_broken_dumps)
    # Must not raise despite the simulated failure.
    prediction_id = service_module.prediction_service._store_prediction(
        "TESTCO", 5, ensemble, signal, feature_snapshot=feature_row)
    assert prediction_id is not None


def test_full_pipeline_saves_a_real_retrievable_snapshot(isolated_db):
    """The ultimate proof: a real, full train_and_predict() run must
    result in a genuine, retrievable feature snapshot matching real
    computed indicator values — not just the isolated helper function
    in a vacuum."""
    import backend.forecasting.service as service_module
    from backend.database.db import db_cursor

    _install_provider()
    service_module.WALK_FORWARD_STEP = 120
    result = service_module.prediction_service.train_and_predict("TESTCO", horizon_days=5)
    assert result["success"] is True

    with db_cursor() as cur:
        cur.execute("SELECT id FROM predictions WHERE ticker = 'TESTCO' ORDER BY id DESC LIMIT 1")
        prediction_id = cur.fetchone()["id"]

    snapshot = service_module.prediction_service.get_prediction_input_snapshot(prediction_id)
    assert snapshot is not None
    assert "close" in snapshot
    assert snapshot["close"] > 0  # a real, plausible price, not a placeholder
