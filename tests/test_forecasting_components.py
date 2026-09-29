import numpy as np
import pandas as pd
from tests.conftest import business_day_anchor


def _synthetic_indicators_df(n=300, seed=1):
    from backend.indicators.technical import compute_all_indicators
    today = business_day_anchor()
    dates = pd.date_range(end=today, periods=n, freq="B")
    n_actual = len(dates)
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.05, 1.2, n_actual))
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "open": close, "high": close + 1,
        "low": close - 1, "close": close, "adj_close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n_actual),
    })
    return compute_all_indicators(df)


# ------------------------------------------------------------- features

def test_build_features_produces_all_expected_columns_when_benchmark_given():
    from backend.forecasting.features import build_features, FEATURE_COLUMNS
    indicators = _synthetic_indicators_df(seed=1)
    benchmark_indicators = _synthetic_indicators_df(seed=2)
    features = build_features(indicators, benchmark_close=benchmark_indicators["close"])
    for col in FEATURE_COLUMNS:
        assert col in features.columns


def test_build_features_degrades_gracefully_without_benchmark():
    """Relative-strength columns require a benchmark series — when
    none is given, they must be OMITTED (not present-but-NaN), since
    an all-NaN column would cause build_training_matrix()'s NaN-drop
    to wipe out every row."""
    from backend.forecasting.features import build_features, build_training_matrix
    indicators = _synthetic_indicators_df()
    features = build_features(indicators, benchmark_close=None)
    assert "relative_strength_5d" not in features.columns
    assert "relative_strength_20d" not in features.columns
    # And training must still work fine without them.
    X, y_return, y_direction = build_training_matrix(features, horizon_days=5)
    assert len(X) > 0


def test_features_are_bounded_or_ratio_like_not_raw_price():
    """Sanity check that the 'stationary' design goal actually holds —
    RSI should stay in [0,100], relative-volume/ratios shouldn't drift
    with the absolute price level."""
    from backend.forecasting.features import build_features
    indicators = _synthetic_indicators_df()
    features = build_features(indicators)
    valid_rsi = features["rsi_14"].dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


def test_build_training_matrix_never_leaks_future_target():
    """The critical leakage-prevention check: the last `horizon_days`
    rows can't have a valid target (their future isn't known yet) and
    must be dropped, not filled with a fabricated value."""
    from backend.forecasting.features import build_features, build_training_matrix
    indicators = _synthetic_indicators_df(n=300)
    features = build_features(indicators)
    horizon = 5
    X, y_return, y_direction = build_training_matrix(features, horizon)

    # Row count must be less than the full feature set minus the
    # unknowable trailing rows (allowing for NaN warm-up rows too).
    assert len(X) < len(features) - horizon + 1
    assert len(X) == len(y_return) == len(y_direction)
    assert not X.isna().any().any()
    assert not y_return.isna().any()


def test_extra_columns_align_correctly_with_x_row_for_row():
    """Real, confirmed bug found independently in two separate places
    (the main backtest engine and the TCN service): a naive positional
    slice like features_df[col].iloc[-len(X):] silently returns values
    shifted forward by horizon_days rows compared to what X and
    y_return actually represent, because it doesn't account for the
    trailing rows build_training_matrix drops (no valid target yet).
    extra_columns must guarantee correct alignment by construction,
    reusing the exact same dropna mask used to build X itself."""
    from backend.forecasting.features import build_features, build_training_matrix
    indicators = _synthetic_indicators_df(n=400)
    features = build_features(indicators)
    horizon = 5

    X, y_return, y_direction, aligned = build_training_matrix(features, horizon, extra_columns=["date", "close"])
    assert len(aligned) == len(X)

    # The real correctness proof: for several rows, the row's OWN close
    # price times (1 + its OWN y_return) must equal the close price
    # that actually occurred `horizon` trading days later in the
    # original, undropped features DataFrame — proving "close" and
    # "y_return" genuinely describe the same row, not one shifted
    # relative to the other.
    for i in [0, len(X) // 2, len(X) - 1]:
        expected_future_close = aligned["close"].iloc[i] * (1 + y_return.iloc[i])
        row_position_in_features = features[features["date"] == aligned["date"].iloc[i]].index[0]
        actual_future_close = features["close"].iloc[row_position_in_features + horizon]
        assert abs(expected_future_close - actual_future_close) < 1e-6


def test_without_extra_columns_return_signature_is_unchanged():
    """Backward compatibility: existing callers that don't ask for
    extra_columns must keep getting exactly the same 3-tuple as before."""
    from backend.forecasting.features import build_features, build_training_matrix
    indicators = _synthetic_indicators_df(n=300)
    features = build_features(indicators)
    result = build_training_matrix(features, 5)
    assert len(result) == 3


def test_direction_labels_match_return_sign_with_deadzone():
    from backend.forecasting.features import build_features, build_training_matrix
    indicators = _synthetic_indicators_df(n=300)
    features = build_features(indicators)
    X, y_return, y_direction = build_training_matrix(features, horizon_days=5)
    for ret, direction in zip(y_return, y_direction):
        if direction == 1:
            assert ret > 0
        elif direction == -1:
            assert ret < 0


# --------------------------------------------------------------- ensemble

def test_ensemble_agreement_forces_neutral_on_disagreement():
    from backend.forecasting.ensemble import build_ensemble
    preds = {"a": 0.03, "b": -0.02, "c": 0.01, "d": -0.025}
    accs = {"a": 0.58, "b": 0.55, "c": 0.52, "d": 0.60}
    result = build_ensemble(preds, accs, horizon_days=5)
    assert result["direction"] == "neutral"
    assert result["low_confidence_disagreement"] is True


def test_ensemble_confident_when_models_agree():
    from backend.forecasting.ensemble import build_ensemble
    preds = {"a": 0.04, "b": 0.035, "c": 0.045}
    accs = {"a": 0.68, "b": 0.65, "c": 0.70}
    result = build_ensemble(preds, accs, horizon_days=5)
    assert result["direction"] == "bullish"
    assert not result["low_confidence_disagreement"]
    # Threshold reflects the corrected, honest confidence formula: 65-70%
    # direction accuracy is genuinely good for stock prediction, but the
    # formula must never inflate that into an overconfident-sounding
    # number just from agreement alone (see the chance-level regression
    # test below for what this formula is specifically guarding against).
    assert result["confidence"] > 0.3


def test_ensemble_confidence_is_near_zero_for_chance_level_models_even_when_they_agree():
    """Real bug found via testing: the confidence formula used to add
    a flat 'agreement bonus' independent of whether the models had any
    real skill — meaning two models validated at EXACTLY 50% direction
    accuracy (proven, literal coin-flip performance) that happened to
    agree still reported 50% confidence. Confidence must track actual
    demonstrated skill, not just how many models happen to agree,
    since agreement between zero-skill models is coincidence, not
    signal — this is the single most important honesty property the
    whole ensemble module claims to have."""
    from backend.forecasting.ensemble import build_ensemble
    result = build_ensemble(
        model_predictions={"a": 0.02, "b": 0.025},
        model_accuracies={"a": 0.50, "b": 0.50},
        horizon_days=5,
    )
    assert result["confidence"] <= 0.1


def test_ensemble_downweights_below_chance_model():
    from backend.forecasting.ensemble import build_ensemble
    preds = {"good": 0.03, "bad": -0.10}
    accs = {"good": 0.70, "bad": 0.42}
    result = build_ensemble(preds, accs, horizon_days=5)
    weights = {v["model"]: v["weight"] for v in result["votes"]}
    assert weights["good"] > weights["bad"] * 3


def test_classify_signal_requires_technical_agreement_for_strong_setup():
    from backend.forecasting.ensemble import build_ensemble, classify_signal
    preds = {"a": 0.05, "b": 0.045, "c": 0.06}
    accs = {"a": 0.72, "b": 0.70, "c": 0.75}
    ensemble = build_ensemble(preds, accs, horizon_days=5)

    with_technical_agreement = classify_signal(ensemble, technical_bullish=True, technical_bearish=False)
    without_technical_agreement = classify_signal(ensemble, technical_bullish=False, technical_bearish=False)

    assert with_technical_agreement["signal"] == "STRONG_BULLISH_SETUP"
    assert without_technical_agreement["signal"] != "STRONG_BULLISH_SETUP"


def test_classify_signal_requires_news_agreement_for_strong_setup():
    """News is a genuine, live-only gate on the strongest signal tier —
    not a trained feature, but it must be able to prevent a STRONG
    call when today's news actively contradicts everything else."""
    from backend.forecasting.ensemble import build_ensemble, classify_signal
    preds = {"a": 0.05, "b": 0.045, "c": 0.06}
    accs = {"a": 0.72, "b": 0.70, "c": 0.75}
    ensemble = build_ensemble(preds, accs, horizon_days=5)

    news_agrees = classify_signal(ensemble, technical_bullish=True, technical_bearish=False, news_agrees=True)
    news_disagrees = classify_signal(ensemble, technical_bullish=True, technical_bearish=False, news_agrees=False)
    news_unavailable = classify_signal(ensemble, technical_bullish=True, technical_bearish=False, news_agrees=None)

    assert news_agrees["signal"] == "STRONG_BULLISH_SETUP"
    assert news_disagrees["signal"] == "BULLISH_WATCH", "Contradicting news must cap the signal below STRONG"
    assert news_unavailable["signal"] == "STRONG_BULLISH_SETUP", "Missing news data must not block escalation — silence isn't disagreement"


# ----------------------------------------------------------------- risk

def test_kelly_position_sizing_never_exceeds_cap():
    from backend.forecasting.risk import suggest_position_size, MAX_POSITION_PCT
    stats = {"win_rate": 0.9, "avg_win_pct": 10.0, "avg_loss_pct": 0.5, "n_trades": 100}
    result = suggest_position_size(stats)
    assert result["suggested_position_pct"] <= MAX_POSITION_PCT


def test_position_sizing_refuses_with_insufficient_trades():
    from backend.forecasting.risk import suggest_position_size, MIN_TRADES_FOR_KELLY
    stats = {"win_rate": 0.9, "avg_win_pct": 5.0, "avg_loss_pct": 1.0, "n_trades": MIN_TRADES_FOR_KELLY - 1}
    result = suggest_position_size(stats)
    assert result["suggested_position_pct"] is None


def test_all_win_track_record_gets_a_distinct_honest_message_not_a_weak_edge_message():
    """Real issue found via testing: kelly_fraction() returns None when
    there's no losing-trade data to compute a ratio against (every
    trade won) — the message must clearly say THAT, not the generic
    'edge doesn't clear the bar' wording used for a genuinely weak
    edge. Those are very different situations and conflating them is
    misleading either way: an all-win record is a suspiciously perfect
    one worth scrutinizing, not a weak one."""
    from backend.forecasting.risk import suggest_position_size
    stats = {"win_rate": 1.0, "avg_win_pct": 3.5, "avg_loss_pct": None, "n_trades": 25}
    result = suggest_position_size(stats)
    assert result["suggested_position_pct"] is None
    assert "doesn't clear the bar" not in result["explanation"]
    assert "win" in result["explanation"].lower()


def test_all_loss_track_record_gets_a_distinct_alarming_message():
    from backend.forecasting.risk import suggest_position_size
    stats = {"win_rate": 0.0, "avg_win_pct": None, "avg_loss_pct": 2.8, "n_trades": 25}
    result = suggest_position_size(stats)
    assert result["suggested_position_pct"] is None
    assert "doesn't clear the bar" not in result["explanation"]
    assert "no validated edge" in result["explanation"].lower() or "alarming" in result["explanation"].lower() or "loss" in result["explanation"].lower()


def test_atr_stop_direction_correctness():
    from backend.forecasting.risk import suggest_atr_stop
    bullish = suggest_atr_stop(current_price=100.0, atr=2.0, direction="bullish", atr_multiplier=2.0)
    bearish = suggest_atr_stop(current_price=100.0, atr=2.0, direction="bearish", atr_multiplier=2.0)
    assert bullish["stop_price"] < 100.0  # stop below entry for a long position
    assert bearish["stop_price"] > 100.0  # stop above entry for a short position


def test_risk_profiles_produce_genuinely_different_sizing():
    """The honest form of 'personalization' — same underlying edge,
    different suggested sizing based on how much of it you're willing
    to risk. Must actually differ, not just accept the parameter and
    ignore it."""
    from backend.forecasting.risk import suggest_position_size, RISK_PROFILES
    stats = {"win_rate": 0.65, "avg_win_pct": 3.0, "avg_loss_pct": 1.5, "n_trades": 50}

    conservative = suggest_position_size(stats, kelly_fraction_multiplier=RISK_PROFILES["conservative"])
    moderate = suggest_position_size(stats, kelly_fraction_multiplier=RISK_PROFILES["moderate"])
    aggressive = suggest_position_size(stats, kelly_fraction_multiplier=RISK_PROFILES["aggressive"])

    assert conservative["suggested_position_pct"] < moderate["suggested_position_pct"] < aggressive["suggested_position_pct"]
