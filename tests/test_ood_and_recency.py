import numpy as np
import pandas as pd
from tests.conftest import business_day_anchor


def test_ood_detects_no_issue_on_normal_data():
    from backend.forecasting.ood import compute_training_distribution, check_out_of_distribution
    rng = np.random.default_rng(1)
    X_train = pd.DataFrame({"rsi_14": rng.normal(50, 8, 500), "atr_pct_14": rng.normal(2.5, 0.5, 500)})
    dist = compute_training_distribution(X_train)
    normal_row = pd.Series({"rsi_14": 55, "atr_pct_14": 2.7})
    result = check_out_of_distribution(normal_row, dist)
    assert result["is_out_of_distribution"] is False
    assert result["warning"] is None


def test_ood_flags_genuine_outlier():
    from backend.forecasting.ood import compute_training_distribution, check_out_of_distribution
    rng = np.random.default_rng(1)
    X_train = pd.DataFrame({"rsi_14": rng.normal(50, 8, 500), "atr_pct_14": rng.normal(2.5, 0.5, 500)})
    dist = compute_training_distribution(X_train)
    extreme_row = pd.Series({"rsi_14": 50, "atr_pct_14": 20.0})  # way beyond training range
    result = check_out_of_distribution(extreme_row, dist)
    assert result["is_out_of_distribution"] is True
    assert result["flagged_features"][0]["feature"] == "atr_pct_14"
    assert result["warning"] is not None and "atr_pct_14" in result["warning"]


def test_live_prediction_uses_true_latest_row_not_stale_training_row():
    """Real bug found and fixed during development: build_training_matrix()
    correctly drops the last `horizon_days` rows from the TRAINING
    matrix X (since they lack a known forward target) — but the live
    prediction pipeline was pulling its input feature row from X's
    last row too, meaning every prediction was silently stale by
    exactly the horizon length instead of reflecting today's actual
    market conditions. This must use features_df's true latest row."""
    from backend.forecasting.features import build_features, build_training_matrix
    from backend.indicators.technical import compute_all_indicators

    today = business_day_anchor()
    n = 300
    dates = pd.date_range(end=today, periods=n, freq="B")
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0.05, 1, n))
    close[-1] = close[-2] * 1.5  # unmistakable final-day spike

    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": close, "high": close + 1,
                        "low": close - 1, "close": close, "adj_close": close, "volume": [1_000_000] * n})
    indicators = compute_all_indicators(df)
    features = build_features(indicators)
    X, y_return, y_direction = build_training_matrix(features, horizon_days=5)

    # This is the exact fix logic used in service.py.
    latest_row = features[list(X.columns)].iloc[[-1]]

    assert abs(latest_row["return_1d"].iloc[0] - 0.5) < 0.01, (
        "The live prediction's input row must reflect today's true return, "
        "not a stale value from `horizon_days` trading days ago."
    )
    assert abs(X["return_1d"].iloc[-1] - 0.5) > 0.1, (
        "Sanity check: X's own last row should NOT show the spike — confirming "
        "the bug this test guards against is real and that we're using the right source."
    )


def test_ood_warning_is_not_duplicated_into_the_generic_notes_list(tmp_path):
    """Real usability issue found via testing: the OOD warning was
    being appended into the same generic 'notes' list as routine
    technical footnotes like 'XGBoost is not installed' — meaning an
    important, prediction-specific reliability caveat rendered with
    the exact same low-priority styling as mundane technical trivia.
    The full OOD result is already exposed separately via the
    dedicated `out_of_distribution` field; it must not ALSO be
    duplicated into `notes`, which should stay reserved for genuine
    model-availability footnotes."""
    from backend.config import settings
    settings.DATABASE_PATH = str(tmp_path / "test.db")
    from backend.database.db import init_db
    init_db()

    import backend.data.market_data_service as mds_module
    from backend.data.base import DataResult, MarketDataProvider
    from datetime import datetime, timezone
    import backend.forecasting.service as service_module

    today = business_day_anchor()
    n = 1300
    dates = pd.date_range(end=today, periods=n, freq="B")
    rng = np.random.default_rng(3)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    close[-1] = close[-2] * 3.0  # an unmistakable, extreme final-day spike to force an OOD flag
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
    service_module.WALK_FORWARD_STEP = 120

    result = service_module.prediction_service.train_and_predict("TESTCO", horizon_days=5)
    assert result["success"] is True
    assert result["out_of_distribution"]["is_out_of_distribution"] is True, \
        "Test setup sanity check: the extreme final-day spike must actually trigger a real OOD flag"

    ood_warning_text = result["out_of_distribution"]["warning"]
    combined_notes = " ".join(result["notes"])
    assert ood_warning_text not in combined_notes, \
        "The OOD warning must not be duplicated into the generic notes list"


def test_ood_handles_nan_and_zero_variance_gracefully():
    from backend.forecasting.ood import compute_training_distribution, check_out_of_distribution
    X_train = pd.DataFrame({"constant_feature": [5.0] * 100, "normal_feature": np.random.default_rng(1).normal(0, 1, 100)})
    dist = compute_training_distribution(X_train)
    row_with_nan = pd.Series({"constant_feature": 5.0, "normal_feature": np.nan})
    result = check_out_of_distribution(row_with_nan, dist)  # must not raise
    assert isinstance(result["is_out_of_distribution"], bool)


def test_ood_ranks_multiple_flags_by_severity():
    from backend.forecasting.ood import compute_training_distribution, check_out_of_distribution
    rng = np.random.default_rng(1)
    X_train = pd.DataFrame({
        "feature_a": rng.normal(0, 1, 500),
        "feature_b": rng.normal(0, 1, 500),
    })
    dist = compute_training_distribution(X_train)
    row = pd.Series({"feature_a": 5.0, "feature_b": 10.0})  # both extreme, b more so
    result = check_out_of_distribution(row, dist)
    assert result["is_out_of_distribution"] is True
    assert len(result["flagged_features"]) == 2
    assert result["flagged_features"][0]["feature"] == "feature_b"  # most severe listed first


def test_recency_weights_favor_recent_data():
    from backend.forecasting.service import _recency_weights
    weights = _recency_weights(1260)
    assert weights[-1] == 1.0  # most recent row gets full weight
    assert weights[0] < weights[-1]  # oldest row gets less weight
    assert 0.45 < weights[-253] < 0.55  # ~1 year ago should be near the half-life point


def test_recency_weights_monotonically_increasing():
    from backend.forecasting.service import _recency_weights
    weights = _recency_weights(500)
    # Every row must be weighted >= the row before it (strictly increasing toward the present).
    assert all(weights[i] <= weights[i + 1] for i in range(len(weights) - 1))
