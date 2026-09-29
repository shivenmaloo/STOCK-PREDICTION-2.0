"""
Orchestration layer for the whole prediction pipeline (spec sections
9, 11, 21, 26).

Ties together: fetching data -> feature engineering -> walk-forward
validation -> final model fitting -> persistence to disk -> live
prediction -> ensemble -> signal classification -> risk sizing ->
storing the prediction for later accuracy tracking.

Models are cached to disk and reused across requests within a
freshness window (spec section 26: "do not retrain every time the
page loads") — training only happens when there's no fresh saved
model, or when explicitly requested.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from backend.data.market_data_service import market_data_service
from backend.database.db import db_cursor
from backend.indicators.technical import compute_all_indicators
from backend.forecasting.features import build_features, build_training_matrix, FEATURE_COLUMNS
from backend.forecasting.validation import (
    run_walk_forward, trading_stats_from_walk_forward,
    compute_conformal_radius, compute_naive_baseline_comparison,
)
from backend.forecasting.models import (
    get_available_sklearn_models, get_available_xgboost_model,
    get_xgboost_feature_importance, XGBOOST_AVAILABLE,
)
from backend.forecasting.lstm_model import make_lstm, TORCH_AVAILABLE
from backend.forecasting.ensemble import build_ensemble, classify_signal
from backend.forecasting.risk import suggest_position_size, suggest_atr_stop, RISK_PROFILES, DEFAULT_KELLY_FRACTION
from backend.forecasting.adaptive import compute_per_model_correctness, get_live_model_accuracy, blend_accuracy
from backend.forecasting.ood import compute_training_distribution, check_out_of_distribution
from backend.forecasting.news_context import fetch_news_sentiment, compare_news_to_ensemble
from backend.news.accuracy_tracker import log_news_sentiment, get_news_sentiment_accuracy
from backend.news.source_reliability import log_source_reading
from backend.forecasting.summary import build_plain_summary

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Progress logging — every stage of training prints a [PREDICT] line
# to the terminal, so a slow or stuck run is immediately diagnosable
# (which stage it's in) instead of being an opaque, unexplained wait.
logger = logging.getLogger("stock_ai.predict")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

BENCHMARK_TICKER = "SPY"  # used for the relative-strength feature

MODEL_FRESHNESS_HOURS = 24  # don't retrain more than once a day per ticker+horizon
MIN_ROWS_FOR_TRAINING = 300  # preferred: ~1.2 years of raw trading days
MIN_ROWS_RELAXED_FLOOR = 120  # absolute floor (~6 months) — below this, there's genuinely not enough to validate anything meaningful
WALK_FORWARD_MIN_TRAIN = 200
WALK_FORWARD_STEP = 10  # ~2 trading weeks per validation step — deliberately finer than before (was 21, ~1
# month): roughly doubles the number of independent walk-forward test windows for a more robust accuracy
# estimate, at the direct cost of proportionally more training time — a tradeoff explicitly chosen for
# thoroughness over speed. Still cheap enough for tree-based models to run this granularity comfortably.

# News-as-ensemble-vote — added as one more vote, weighted the exact
# same accuracy-weighted way as every ML model, using news's OWN
# live-tracked accuracy (backend/news/accuracy_tracker.py) rather than
# a walk-forward backtest — the latter is genuinely impossible without
# a point-in-time news archive we don't have (see news_context.py's
# module docstring for the full reasoning), but a LIVE, forward-only
# accuracy record carries no such leakage risk, since it's built
# entirely from real outcomes as they actually happen, going forward.
NEWS_VOTE_MAX_RETURN_PCT = 0.03  # the return magnitude a maximally-strong news read (+/-100 impact) implies as its vote
NEWS_VOTE_MIN_EVALUATED_READS = 20  # matches MIN_READINGS_FOR_MEANINGFUL_RANK's own threshold in iv_history.py — the
# same "don't trust a percentage computed from a handful of data points" principle applied here


def _news_sentiment_to_vote(median_impact_score: float) -> float:
    """Converts a -100..+100 impact score into a return-like value
    directly comparable to what the ML models vote with — a modest,
    fixed maximum (NEWS_VOTE_MAX_RETURN_PCT) so news can never propose
    a wildly larger move than any real model would, regardless of how
    extreme its own sentiment score gets."""
    return (median_impact_score / 100) * NEWS_VOTE_MAX_RETURN_PCT


def _get_news_vote_weight_accuracy() -> float | None:
    """News's ensemble weight comes from its OWN live-tracked
    directional accuracy — the exact same accuracy-weighting principle
    used for every ML model, just fed a live-tracked number instead of
    a walk-forward one. Returns None (falls back to the same small,
    unproven-model floor weight every other model gets) until there's
    real, meaningful evidence — a raw percentage from only a handful
    of matured reads is not evidence, it's noise, and treating it as
    real would let a lucky early streak inappropriately dominate real
    predictions before news has actually proven anything."""
    try:
        result = get_news_sentiment_accuracy()
        if result["n_evaluated"] < NEWS_VOTE_MIN_EVALUATED_READS:
            return None
        return result["accuracy"]
    except Exception:  # noqa: BLE001 — this must never break a prediction
        return None


def _build_ensemble_with_news_vote(model_predictions: dict, model_accuracies: dict,
                                    horizon_days: int, news_sentiment: dict) -> dict:
    """Builds the ensemble exactly as before, PLUS one additional vote
    from news sentiment when it's genuinely available and directional
    — added to COPIES of the input dicts, never the originals, since
    model_predictions/model_accuracies are also used elsewhere (the
    Model Performance table, feature importance) where a fake "model"
    called news_sentiment has no business appearing. The ensemble math
    itself (backend/forecasting/ensemble.py) is completely unchanged —
    this only ever adds one more entry to its input dicts, reusing the
    exact same, already-tested accuracy-weighting logic for every
    vote, model or not."""
    predictions_for_ensemble = dict(model_predictions)
    accuracies_for_ensemble = dict(model_accuracies)

    if news_sentiment.get("available") and news_sentiment.get("news_direction") != "neutral":
        predictions_for_ensemble["news_sentiment"] = _news_sentiment_to_vote(news_sentiment["median_impact_score"])
        accuracies_for_ensemble["news_sentiment"] = _get_news_vote_weight_accuracy()

    return build_ensemble(predictions_for_ensemble, accuracies_for_ensemble, horizon_days)
# LSTM refits a fresh neural network from scratch on every single walk-
# forward split (no warm-start), which makes the same monthly step size
# extremely expensive over 5 years of history (~50 full retrains). A
# coarser step here is still a fully valid, non-overlapping,
# chronological walk-forward evaluation — just fewer, larger windows —
# and cuts a several-minute wait down to something reasonable.
LSTM_WALK_FORWARD_STEP = 60  # ~2 trading months per validation step — deliberately finer than before (was
# 90, ~1 quarter): more validation windows for a more robust accuracy estimate, at proportionally more
# training time. Kept coarser than the tree-based models' own step (10) since LSTM training is genuinely
# far more expensive per split — this is the more conservative end of the same "more thorough, more time"
# tradeoff, not the same aggressive granularity applied uniformly regardless of cost.
LSTM_EPOCHS = 10  # halved from the original 20 — still enough to fit a small 1-layer LSTM


def _model_path(ticker: str, horizon: int, model_name: str) -> Path:
    ext = "pt" if model_name == "lstm" else "joblib"
    return MODELS_DIR / f"{ticker.upper()}_{horizon}d_{model_name}.{ext}"


def _fetch_aligned_benchmark(dates: pd.Series) -> Optional[pd.Series]:
    """Fetches SPY's close price for the same date range as the ticker
    being modeled, aligned row-for-row. Returns None (rather than
    raising) if the benchmark can't be fetched or aligned — relative
    strength is a genuine improvement, not a hard requirement, and
    build_features() already degrades gracefully when this is None."""
    try:
        bench_result = market_data_service.get_historical(BENCHMARK_TICKER, period="5y", interval="1d")
        if not bench_result.success or bench_result.data is None or bench_result.data.empty:
            return None
        bench_df = bench_result.data[["date", "close"]].rename(columns={"close": "bench_close"})
        aligned = pd.DataFrame({"date": dates}).merge(bench_df, on="date", how="left")
        aligned["bench_close"] = aligned["bench_close"].ffill().bfill()
        if aligned["bench_close"].isna().any():
            return None  # couldn't align cleanly — don't silently hand back a partially-null series
        return aligned["bench_close"]
    except Exception:  # noqa: BLE001
        return None


def _predict_live_value(model, name: str, X_full: pd.DataFrame, latest_row: pd.DataFrame) -> float:
    """Returns a single live prediction from `model`. Most models
    (linear/RF/GBM/XGBoost) only need the single most recent feature
    row. LSTM is structurally different: its architecture needs a
    sequence of `seq_len` preceding rows to build the temporal window
    it was trained on. Passing it only the single latest row — as
    every other model correctly receives — silently produces a
    prediction of 0.0 (neutral) every time, since LSTMModel.predict()
    has no history to build a window from and defaults to neutral for
    any row lacking that context.

    This was a real, previously undetected bug: the LSTM's live
    prediction had never actually reflected real market conditions,
    regardless of what the market was doing — found while fixing the
    unrelated LSTM-cache-reload gap and tracing through exactly what
    the single-row input would produce."""
    if name == "lstm":
        seq_len = getattr(model, "seq_len", 20)
        context = X_full.tail(seq_len)
        preds = model.predict(context.values)
        return float(preds[-1])
    return float(model.predict(latest_row.values)[0])


def _recency_weights(n_rows: int, half_life_days: int = 252) -> np.ndarray:
    """Exponentially decaying sample weights favoring recent rows —
    row 0 (oldest) gets the least weight, the last row (most recent)
    gets weight 1.0. half_life_days=252 (~1 trading year) means data
    from a year ago carries half the weight of today's."""
    ages = np.arange(n_rows - 1, -1, -1)  # most recent row has age 0
    return np.power(0.5, ages / half_life_days)


def get_available_model_factories() -> dict:
    factories = dict(get_available_sklearn_models())
    factories.update(get_available_xgboost_model())
    if TORCH_AVAILABLE:
        factories["lstm"] = lambda: make_lstm(seq_len=20, epochs=LSTM_EPOCHS)
    return factories


class PredictionService:
    def get_or_train(self, ticker: str, horizon_days: int, force_retrain: bool = False,
                      risk_profile: str = "moderate") -> dict:
        """Main entry point: returns a full prediction bundle, training
        fresh models only if needed (no saved version, or the saved
        version is stale)."""
        ticker = ticker.upper()

        existing = self._load_latest_model_version(ticker, horizon_days)
        is_fresh = existing is not None and (
            datetime.now(timezone.utc) - datetime.fromisoformat(existing["trained_at"])
        ) < timedelta(hours=MODEL_FRESHNESS_HOURS)

        if not force_retrain and is_fresh:
            prediction = self._predict_with_saved_models(ticker, horizon_days, existing, risk_profile)
            if prediction is not None:
                return prediction
            # Saved models couldn't be loaded/used for some reason — fall through to retrain.

        return self.train_and_predict(ticker, horizon_days, risk_profile)

    def train_and_predict(self, ticker: str, horizon_days: int, risk_profile: str = "moderate") -> dict:
        ticker = ticker.upper()
        logger.info(f"[PREDICT] === Starting train_and_predict for {ticker} ({horizon_days}D) ===")

        logger.info(f"[PREDICT] Fetching 5y historical data for {ticker}...")
        hist_result = market_data_service.get_historical(ticker, period="5y", interval="1d")
        available_rows = len(hist_result.data) if hist_result.data is not None else 0

        if not hist_result.success or available_rows < MIN_ROWS_RELAXED_FLOOR:
            logger.info(f"[PREDICT] FAILED — insufficient historical data for {ticker} ({available_rows} rows)")
            return {
                "success": False,
                "error": (
                    f"Not enough historical data for {ticker} to train prediction models — "
                    f"only {available_rows} trading days available, need at least "
                    f"{MIN_ROWS_RELAXED_FLOOR} (roughly 6 months) even in relaxed mode for a "
                    f"recently-listed ticker. This is usually a genuinely new listing, not a bug."
                ),
            }
        logger.info(f"[PREDICT] Got {available_rows} rows of historical data (source: {hist_result.source})")

        logger.info("[PREDICT] Computing technical indicators and building features...")
        indicators_df = compute_all_indicators(hist_result.data)
        benchmark_close = _fetch_aligned_benchmark(indicators_df["date"]) if ticker != BENCHMARK_TICKER else None
        if benchmark_close is None and ticker != BENCHMARK_TICKER:
            logger.info(f"[PREDICT] Benchmark ({BENCHMARK_TICKER}) unavailable — proceeding without relative-strength feature")
        features_df = build_features(indicators_df, benchmark_close=benchmark_close)
        X, y_return, y_direction = build_training_matrix(features_df, horizon_days)
        logger.info(f"[PREDICT] Built {len(X)} complete feature rows")

        min_feature_rows_relaxed = 60  # below this, even relaxed mode can't produce a single honest walk-forward split
        if len(X) < min_feature_rows_relaxed:
            logger.info(f"[PREDICT] FAILED — too few complete feature rows for {ticker} even for relaxed mode")
            return {
                "success": False,
                "error": (
                    f"Not enough complete feature rows for {ticker} after removing warm-up/NaN "
                    f"periods ({len(X)} rows) — even relaxed mode needs at least {min_feature_rows_relaxed}."
                ),
            }

        # Two-tier data sufficiency: try the full, preferred walk-forward
        # setup first; if there isn't enough FEATURE data for it to
        # actually produce any real splits (this must be judged on
        # len(X) — the post-warm-up feature row count — not the raw
        # historical row count, since indicators like the 200-day SMA
        # alone consume ~200 rows of warm-up before producing a single
        # value), fall back to a scaled-down but still genuinely
        # walk-forward validated setup instead of silently producing
        # zero splits or refusing outright. Clearly labeled either way.
        relaxed_mode = len(X) < (WALK_FORWARD_MIN_TRAIN + WALK_FORWARD_STEP)
        if relaxed_mode:
            logger.info(f"[PREDICT] Only {len(X)} feature rows available (need {WALK_FORWARD_MIN_TRAIN + WALK_FORWARD_STEP}+ "
                        f"for standard validation) — using RELAXED mode with scaled-down walk-forward validation")

        # Scale the walk-forward window/step to whatever feature data is
        # actually available in relaxed mode, aiming for at least a
        # handful of real validation splits rather than zero.
        if relaxed_mode:
            wf_min_train = max(40, int(len(X) * 0.5))
            wf_step = max(5, int(len(X) * 0.12))
            lstm_step = max(15, int(len(X) * 0.3))
        else:
            wf_min_train = WALK_FORWARD_MIN_TRAIN
            wf_step = WALK_FORWARD_STEP
            lstm_step = LSTM_WALK_FORWARD_STEP

        factories = get_available_model_factories()
        logger.info(f"[PREDICT] Models to train: {list(factories.keys())}")
        unavailable_notes = []
        if not XGBOOST_AVAILABLE:
            unavailable_notes.append("XGBoost is not installed — Model B skipped.")
        if not TORCH_AVAILABLE:
            unavailable_notes.append("PyTorch is not installed — LSTM (Model C) skipped.")
        if relaxed_mode:
            unavailable_notes.append(
                f"RELAXED MODE: only {available_rows} trading days of history were available for {ticker} "
                f"(preferred: {MIN_ROWS_FOR_TRAINING}+) — walk-forward validation used a scaled-down window "
                f"with fewer splits than usual. Treat this prediction as lower-confidence than one for a "
                f"ticker with full history."
            )

        walk_forward_results = {}
        trading_stats = {}
        final_models = {}
        feature_importance = {}
        conformal_radii = {}  # per-model prediction-interval radius, from genuine walk-forward residuals
        naive_baseline_checks = {}  # per-model "does it actually beat predicting zero return" sanity check

        for name, factory in factories.items():
            try:
                step = lstm_step if name == "lstm" else wf_step
                logger.info(f"[PREDICT] Walk-forward validating '{name}' (step={step})...")
                t0 = datetime.now()
                wf_result = run_walk_forward(
                    factory, X, y_return, y_direction, model_name=name,
                    min_train_size=wf_min_train, step_size=step,
                )
                elapsed = (datetime.now() - t0).total_seconds()
                logger.info(f"[PREDICT] '{name}' walk-forward done in {elapsed:.1f}s "
                            f"({wf_result.n_splits} splits, accuracy={wf_result.direction_accuracy})")
                walk_forward_results[name] = wf_result
                trading_stats[name] = trading_stats_from_walk_forward(wf_result)
                conformal_radii[name] = compute_conformal_radius(wf_result, confidence=0.80)
                naive_baseline_checks[name] = compute_naive_baseline_comparison(wf_result)

                # Refit on the FULL available dataset for the actual live
                # prediction — walk-forward already gave us an honest
                # estimate of how well this model type generalizes;
                # this final fit uses all available history since we're
                # not evaluating it further, we're using it.
                #
                # Recency-weighted: markets change character over a 5-year
                # window (a 2021 regime isn't necessarily representative
                # of today), so more recent rows get proportionally more
                # influence on the final model via exponential sample
                # weighting — a half-life of ~1 trading year means data
                # from 4 years ago carries roughly 1/16th the weight of
                # yesterday's. This only affects the FINAL deployed model,
                # never the walk-forward validation above, which must
                # stay methodologically untouched to remain an honest
                # measure of out-of-sample accuracy.
                logger.info(f"[PREDICT] Refitting '{name}' on full dataset (recency-weighted)...")
                final_model = factory()
                sample_weight = _recency_weights(len(X))
                try:
                    final_model.fit(X.values, y_return.values, sample_weight=sample_weight)
                except TypeError:
                    # Some model types (e.g. our custom LSTM wrapper) don't
                    # accept sample_weight — fall back to an unweighted fit
                    # rather than failing this model entirely.
                    final_model.fit(X.values, y_return.values)
                final_models[name] = final_model
                logger.info(f"[PREDICT] '{name}' complete")

                if name == "xgboost":
                    feature_importance[name] = get_xgboost_feature_importance(final_model, list(X.columns))
            except Exception as exc:  # noqa: BLE001 — one model failing shouldn't kill the whole run
                logger.info(f"[PREDICT] '{name}' FAILED: {exc}")
                unavailable_notes.append(f"{name} failed during training: {exc}")

        logger.info(f"[PREDICT] All models done. Building ensemble and saving...")
        if not final_models:
            return {"success": False, "error": "All models failed to train.", "notes": unavailable_notes}

        # Predict from the TRUE most recent feature row — this must come
        # from features_df, not X. X has already dropped the last
        # `horizon_days` rows (correctly, since they lack a known
        # forward-looking target for training), but that means X's
        # "last" row is stale by exactly the horizon length. A live
        # prediction needs today's actual market conditions, not data
        # from `horizon_days` trading days ago. Same feature columns as
        # X (guaranteeing compatibility with the trained model), just
        # the genuinely current row.
        latest_row = features_df[list(X.columns)].iloc[[-1]]
        if latest_row.isna().any(axis=None):
            logger.info("[PREDICT] WARNING — most recent feature row has missing values; filling from the prior row")
            latest_row = features_df[list(X.columns)].ffill().iloc[[-1]]
        # Full (untruncated) feature history, up to and including today —
        # this is what LSTM needs for its windowing context. Deliberately
        # NOT `X`, which has already dropped the most recent rows (correct
        # for training, since they lack a known target — but exactly the
        # kind of staleness that must never leak into the live prediction
        # input, the same lesson already learned and fixed once for
        # `latest_row` itself above).
        full_features_for_prediction = features_df[list(X.columns)].ffill()
        model_predictions = {}
        for name, model in final_models.items():
            try:
                pred = _predict_live_value(model, name, full_features_for_prediction, latest_row)
                model_predictions[name] = pred
            except Exception as exc:  # noqa: BLE001
                unavailable_notes.append(f"{name} failed at prediction time: {exc}")

        # Out-of-distribution check: is today's market condition genuinely
        # unlike anything the model trained on? This is a direct response
        # to a real limitation found during development — tree-based
        # models can't extrapolate beyond their training range, so this
        # flags it plainly rather than silently letting the model guess.
        training_distribution = compute_training_distribution(X)
        ood_result = check_out_of_distribution(latest_row.iloc[0], training_distribution)
        # Deliberately NOT appended to unavailable_notes — that list is
        # generic small-text footnotes ("XGBoost isn't installed"), and
        # an OOD warning is a meaningfully more important, prediction-
        # specific reliability caveat that deserves its own prominent
        # display, not to be buried alongside routine technical notes
        # (a real usability issue found via testing — see how the
        # frontend renders the separate `out_of_distribution` field).

        walk_forward_accuracies = {
            name: walk_forward_results[name].direction_accuracy
            for name in model_predictions if name in walk_forward_results
        }
        model_accuracies, accuracy_notes = self._blend_with_live_accuracy(walk_forward_accuracies, ticker)

        # Fetched BEFORE the ensemble now (not after) specifically so
        # it can be used as one of the ensemble's own votes — see
        # _build_ensemble_with_news_vote() and the module docstring on
        # news_context.py for why this carries no leakage risk despite
        # genuinely influencing the prediction: it's a live, forward-
        # only signal weighted by its own live-tracked accuracy, never
        # a trained, backtested feature.
        news_sentiment = fetch_news_sentiment(ticker)

        ensemble = _build_ensemble_with_news_vote(model_predictions, model_accuracies, horizon_days, news_sentiment)

        # Conformal prediction interval: a weighted combination of each
        # contributing model's own genuine walk-forward residual
        # quantile (weighted the same way their point predictions are
        # weighted in the ensemble above), giving an interval with an
        # approximate real coverage guarantee — not just the raw
        # min/max spread of model opinions (which prediction_range_pct
        # already shows, and still does, alongside this).
        ensemble_weights = {v["model"]: v["weight"] for v in ensemble["votes"]}
        valid_radii = {name: r for name, r in conformal_radii.items() if r is not None and name in ensemble_weights}
        if valid_radii:
            total_w = sum(ensemble_weights[name] for name in valid_radii) or 1.0
            conformal_radius_pct = sum(conformal_radii[name] * ensemble_weights[name] for name in valid_radii) / total_w * 100
            ensemble["conformal_interval_pct"] = [
                round(ensemble["expected_return_pct"] - conformal_radius_pct, 2),
                round(ensemble["expected_return_pct"] + conformal_radius_pct, 2),
            ]
            ensemble["conformal_confidence"] = 0.80
        else:
            ensemble["conformal_interval_pct"] = None
            ensemble["conformal_confidence"] = None

        latest_indicators = indicators_df.iloc[-1]
        technical_bullish = bool(
            pd.notna(latest_indicators.get("sma_50")) and pd.notna(latest_indicators.get("close")) and
            latest_indicators["close"] > latest_indicators["sma_50"] and
            pd.notna(latest_indicators.get("rsi_14")) and latest_indicators["rsi_14"] > 50
        )
        technical_bearish = bool(
            pd.notna(latest_indicators.get("sma_50")) and pd.notna(latest_indicators.get("close")) and
            latest_indicators["close"] < latest_indicators["sma_50"] and
            pd.notna(latest_indicators.get("rsi_14")) and latest_indicators["rsi_14"] < 50
        )

        # Post-hoc COMPARISON against the final ensemble direction —
        # cheap and pure, reusing the news_sentiment already fetched
        # above rather than fetching again. Still one of the factors
        # classify_signal() checks before escalating to the strongest
        # signal tiers, alongside technical agreement — that gating
        # role is unchanged; news now ALSO gets a direct, weighted vote
        # in the ensemble itself when it has enough tracked accuracy to
        # earn one (see _build_ensemble_with_news_vote above).
        news_context = compare_news_to_ensemble(news_sentiment, ensemble["direction"])
        if news_context.get("available"):
            log_news_sentiment(ticker, news_context["avg_impact_score"], news_context["news_direction"],
                                float(latest_indicators["close"]), horizon_days)
            # Per-SOURCE logging alongside the aggregate — each
            # opinionated article's own call gets tracked independently,
            # building the real evidence _get_source_reliability_weight()
            # uses to decide which sources actually deserve extra trust.
            for source_name, impact_score, direction in news_context.get("source_readings", []):
                log_source_reading(source_name, ticker, impact_score, direction,
                                    float(latest_indicators["close"]), horizon_days)

        signal = classify_signal(ensemble, technical_bullish, technical_bearish,
                                  news_agrees=news_context.get("agrees_with_model"))

        # Exposed for the frontend's "does everything actually agree"
        # summary — the same technical_bullish/technical_bearish flags
        # classify_signal() already used internally, just surfaced
        # honestly instead of being silently discarded after use.
        if ensemble["direction"] == "neutral" or (not technical_bullish and not technical_bearish):
            technical_agrees: Optional[bool] = None
        else:
            technical_agrees = (ensemble["direction"] == "bullish" and technical_bullish) or \
                                (ensemble["direction"] == "bearish" and technical_bearish)

        best_model_for_risk = max(
            (n for n in trading_stats if trading_stats[n]["n_trades"] > 0),
            key=lambda n: model_accuracies.get(n) or 0, default=None,
        )
        kelly_multiplier = RISK_PROFILES.get(risk_profile, DEFAULT_KELLY_FRACTION)
        position_sizing = (
            suggest_position_size(trading_stats[best_model_for_risk], kelly_fraction_multiplier=kelly_multiplier)
            if best_model_for_risk else {"suggested_position_pct": None, "explanation": "No model had enough validated trades."}
        )
        position_sizing["risk_profile"] = risk_profile
        atr_stop = suggest_atr_stop(
            current_price=float(latest_indicators["close"]),
            atr=float(latest_indicators["atr_14"]) if pd.notna(latest_indicators.get("atr_14")) else None,
            direction=ensemble["direction"],
        )

        self._save_models(ticker, horizon_days, final_models, walk_forward_results, trading_stats)

        prediction_id = self._store_prediction(ticker, horizon_days, ensemble, signal, feature_snapshot=latest_indicators)
        logger.info(f"[PREDICT] === DONE for {ticker}: signal={signal['signal']}, "
                    f"confidence={ensemble['confidence']} ===")

        prediction_notes = unavailable_notes + accuracy_notes
        if news_context.get("available") and news_context.get("agrees_with_model") is False:
            prediction_notes.append(f"News check: {news_context['summary']}")

        plain_summary = build_plain_summary(
            ticker, horizon_days, signal, ensemble, news_context, position_sizing,
            relaxed_mode, ood_result,
        )

        return {
            "success": True,
            "ticker": ticker,
            "horizon_days": horizon_days,
            "as_of_date": str(features_df["date"].iloc[-1]),
            "signal": signal,
            "plain_summary": plain_summary,
            "ensemble": ensemble,
            "model_performance": {
                name: {**walk_forward_results[name].as_dict(), **trading_stats.get(name, {}),
                       **naive_baseline_checks.get(name, {})}
                for name in walk_forward_results
            },
            "feature_importance": feature_importance,
            "position_sizing": position_sizing,
            "atr_stop": atr_stop,
            "notes": prediction_notes,
            "prediction_id": prediction_id,
            "relaxed_mode": relaxed_mode,
            "available_trading_days": available_rows,
            "out_of_distribution": ood_result,
            "news_context": news_context,
            "technical_agrees": technical_agrees,
        }

    # ------------------------------------------------------------ persistence
    # ---------------------------------------------------------- adaptive weighting
    def _blend_with_live_accuracy(self, walk_forward_accuracies: dict, ticker: str) -> tuple[dict, list[str]]:
        """For each model, blends its walk-forward (backtested) accuracy
        with its real, tracked live accuracy on this ticker (or
        globally, if not enough ticker-specific history yet) — this is
        the actual mechanism that makes the app's predictions adapt as
        it's used for real, rather than every retrain starting from a
        blank slate."""
        blended = {}
        notes = []
        for name, wf_acc in walk_forward_accuracies.items():
            live = get_live_model_accuracy(name, ticker=ticker)
            result = blend_accuracy(wf_acc, live)
            blended[name] = result["blended_accuracy"]
            if live["n"] >= 3:  # only worth mentioning once there's a real sample
                notes.append(f"{name}: {result['note']}")
        return blended, notes

    def _save_models(self, ticker: str, horizon: int, models: dict, wf_results: dict, trading_stats: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for name, model in models.items():
            path = _model_path(ticker, horizon, name)
            try:
                if name == "lstm" and TORCH_AVAILABLE:
                    import torch
                    # Save the normalization stats (feature_mean/std)
                    # alongside the network weights — without these, a
                    # reloaded model would normalize new inputs
                    # incorrectly (or not at all), silently producing
                    # wrong predictions rather than an obvious error.
                    # This was previously a known gap that made the
                    # cached-prediction path skip LSTM entirely; fixing
                    # it here is what makes reusing a cached LSTM
                    # prediction possible at all.
                    torch.save({
                        "state_dict": model.model.state_dict(),
                        "feature_mean": model.feature_mean.tolist(),
                        "feature_std": model.feature_std.tolist(),
                    }, path)
                else:
                    joblib.dump(model, path)
            except Exception:  # noqa: BLE001 — persistence failure shouldn't break the live prediction
                continue

            with db_cursor() as cur:
                cur.execute(
                    """INSERT INTO model_versions
                       (ticker, model_type, trained_at, horizon_days, validation_metrics_json, test_metrics_json, file_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (ticker, name, now, horizon,
                     json.dumps(wf_results[name].as_dict()) if name in wf_results else "{}",
                     json.dumps(trading_stats.get(name, {})), str(path)),
                )

    def _load_latest_model_version(self, ticker: str, horizon: int) -> Optional[dict]:
        with db_cursor() as cur:
            cur.execute(
                """SELECT trained_at FROM model_versions
                   WHERE ticker = ? AND horizon_days = ? ORDER BY trained_at DESC LIMIT 1""",
                (ticker, horizon),
            )
            row = cur.fetchone()
        return {"trained_at": row["trained_at"]} if row else None

    def _predict_with_saved_models(self, ticker: str, horizon: int, meta: dict, risk_profile: str = "moderate") -> Optional[dict]:
        # For simplicity and correctness (avoiding subtly-stale feature
        # windows), a "fresh" cache hit still recomputes today's feature
        # row from current data, but reuses saved model weights instead
        # of retraining them — this is the expensive part we're saving.
        try:
            hist_result = market_data_service.get_historical(ticker, period="5y", interval="1d")
            if not hist_result.success or hist_result.data is None:
                return None
            indicators_df = compute_all_indicators(hist_result.data)
            benchmark_close = _fetch_aligned_benchmark(indicators_df["date"]) if ticker != BENCHMARK_TICKER else None
            features_df = build_features(indicators_df, benchmark_close=benchmark_close)
            X, y_return, y_direction = build_training_matrix(features_df, horizon)
            if X.empty:
                return None
            # Same fix as train_and_predict() above — must use features_df's
            # true latest row, not X's (which is stale by `horizon` days).
            latest_row = features_df[list(X.columns)].iloc[[-1]]
            if latest_row.isna().any(axis=None):
                latest_row = features_df[list(X.columns)].ffill().iloc[[-1]]
            full_features_for_prediction = features_df[list(X.columns)].ffill()

            with db_cursor() as cur:
                cur.execute(
                    """SELECT model_type, file_path, validation_metrics_json, test_metrics_json
                       FROM model_versions WHERE ticker = ? AND horizon_days = ?
                       AND trained_at = (SELECT MAX(trained_at) FROM model_versions WHERE ticker = ? AND horizon_days = ?)""",
                    (ticker, horizon, ticker, horizon),
                )
                rows = cur.fetchall()

            if not rows:
                return None

            model_predictions, model_accuracies, trading_stats_map, feature_importance = {}, {}, {}, {}
            for row in rows:
                name, path = row["model_type"], row["file_path"]
                try:
                    if name == "lstm":
                        if not TORCH_AVAILABLE:
                            continue
                        import torch
                        import numpy as _np
                        from backend.forecasting.lstm_model import _LSTMNet
                        saved = torch.load(path)
                        net = _LSTMNet(n_features=len(FEATURE_COLUMNS))
                        net.load_state_dict(saved["state_dict"])
                        model = make_lstm(seq_len=20)
                        model.model = net
                        # Restore the normalization stats used at training
                        # time — without these, a reloaded model would
                        # normalize new inputs incorrectly (fixed gap;
                        # this used to be skipped entirely for this exact
                        # reason).
                        model.feature_mean = _np.array(saved["feature_mean"])
                        model.feature_std = _np.array(saved["feature_std"])
                    else:
                        model = joblib.load(path)
                    pred = _predict_live_value(model, name, full_features_for_prediction, latest_row)
                    model_predictions[name] = pred
                    metrics = json.loads(row["validation_metrics_json"] or "{}")
                    model_accuracies[name] = metrics.get("direction_accuracy")
                    trading_stats_map[name] = json.loads(row["test_metrics_json"] or "{}")
                    if name == "xgboost":
                        feature_importance[name] = get_xgboost_feature_importance(model, list(X.columns))
                except Exception:  # noqa: BLE001
                    continue

            if not model_predictions:
                return None

            training_distribution = compute_training_distribution(X)
            ood_result = check_out_of_distribution(latest_row.iloc[0], training_distribution)

            model_accuracies, accuracy_notes = self._blend_with_live_accuracy(model_accuracies, ticker)
            # Same reordering as the fresh-train path above: fetch news
            # BEFORE the ensemble so it can be used as one of its
            # votes, then compare against the FINAL direction afterward.
            news_sentiment = fetch_news_sentiment(ticker)
            ensemble = _build_ensemble_with_news_vote(model_predictions, model_accuracies, horizon, news_sentiment)
            latest_indicators = indicators_df.iloc[-1]
            technical_bullish = bool(
                pd.notna(latest_indicators.get("sma_50")) and latest_indicators["close"] > latest_indicators["sma_50"]
            )
            technical_bearish = bool(
                pd.notna(latest_indicators.get("sma_50")) and latest_indicators["close"] < latest_indicators["sma_50"]
            )
            news_context = compare_news_to_ensemble(news_sentiment, ensemble["direction"])
            if news_context.get("available"):
                log_news_sentiment(ticker, news_context["avg_impact_score"], news_context["news_direction"],
                                    float(latest_indicators["close"]), horizon)
                for source_name, impact_score, direction in news_context.get("source_readings", []):
                    log_source_reading(source_name, ticker, impact_score, direction,
                                        float(latest_indicators["close"]), horizon)
            signal = classify_signal(ensemble, technical_bullish, technical_bearish,
                                      news_agrees=news_context.get("agrees_with_model"))

            if ensemble["direction"] == "neutral" or (not technical_bullish and not technical_bearish):
                technical_agrees: Optional[bool] = None
            else:
                technical_agrees = (ensemble["direction"] == "bullish" and technical_bullish) or \
                                    (ensemble["direction"] == "bearish" and technical_bearish)

            best_model = max(
                (n for n in trading_stats_map if trading_stats_map[n].get("n_trades", 0) > 0),
                key=lambda n: model_accuracies.get(n) or 0, default=None,
            )
            kelly_multiplier = RISK_PROFILES.get(risk_profile, DEFAULT_KELLY_FRACTION)
            position_sizing = (
                suggest_position_size(trading_stats_map[best_model], kelly_fraction_multiplier=kelly_multiplier) if best_model
                else {"suggested_position_pct": None, "explanation": "No model had enough validated trades."}
            )
            position_sizing["risk_profile"] = risk_profile
            atr_stop = suggest_atr_stop(
                current_price=float(latest_indicators["close"]),
                atr=float(latest_indicators["atr_14"]) if pd.notna(latest_indicators.get("atr_14")) else None,
                direction=ensemble["direction"],
            )

            prediction_id = self._store_prediction(ticker, horizon, ensemble, signal, feature_snapshot=latest_indicators)
            cached_notes = ["Using saved models (trained within the last 24h) rather than retraining."] + accuracy_notes
            if ood_result["is_out_of_distribution"]:
                cached_notes.append(ood_result["warning"])
            if news_context.get("available") and news_context.get("agrees_with_model") is False:
                cached_notes.append(f"News check: {news_context['summary']}")

            plain_summary = build_plain_summary(
                ticker, horizon, signal, ensemble, news_context, position_sizing,
                relaxed_mode=False, out_of_distribution=ood_result,
            )

            return {
                "success": True, "ticker": ticker, "horizon_days": horizon,
                "as_of_date": str(features_df["date"].iloc[-1]),
                "signal": signal, "plain_summary": plain_summary, "ensemble": ensemble,
                "model_performance": {
                    name: {"model_name": name, **trading_stats_map.get(name, {}),
                           "direction_accuracy": model_accuracies.get(name)}
                    for name in model_predictions
                },
                "feature_importance": feature_importance,
                "position_sizing": position_sizing, "atr_stop": atr_stop,
                "notes": cached_notes,
                "prediction_id": prediction_id, "from_cache": True,
                "out_of_distribution": ood_result,
                "news_context": news_context,
                "technical_agrees": technical_agrees,
            }
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------- prediction history
    def _store_prediction(self, ticker: str, horizon: int, ensemble: dict, signal: dict,
                           feature_snapshot: Optional[pd.Series] = None) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with db_cursor() as cur:
            cur.execute(
                """INSERT INTO predictions
                   (ticker, created_at, horizon_days, expected_return, confidence, signal, model_breakdown_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ticker, now, horizon, ensemble["expected_return_pct"], ensemble["confidence"],
                 signal["signal"], json.dumps(ensemble["votes"])),
            )
            prediction_id = cur.lastrowid

            if feature_snapshot is not None:
                try:
                    feature_dict = {}
                    for key, value in feature_snapshot.items():
                        if pd.isna(value):
                            feature_dict[key] = None
                        elif isinstance(value, (pd.Timestamp,)):
                            feature_dict[key] = value.isoformat()
                        elif isinstance(value, (int, float, str, bool)):
                            feature_dict[key] = value
                        else:
                            # numpy scalar types (np.float64, np.int64, etc.) — cast to
                            # a plain Python type rather than risk a silent json failure
                            feature_dict[key] = float(value) if hasattr(value, "item") else str(value)
                    cur.execute(
                        "INSERT INTO prediction_input_snapshots (prediction_id, feature_json, created_at) VALUES (?, ?, ?)",
                        (prediction_id, json.dumps(feature_dict), now),
                    )
                except Exception as exc:  # noqa: BLE001 — a snapshot failure must never break the prediction itself
                    logger.warning(f"[PREDICT] Failed to save feature snapshot for {ticker}: {exc}")

            return prediction_id

    def get_prediction_input_snapshot(self, prediction_id: int) -> Optional[dict]:
        """Retrieves the exact, frozen feature data used for a specific
        past prediction — the real record needed to debug or reproduce
        it, rather than re-fetching data that may have shifted since."""
        with db_cursor() as cur:
            cur.execute("SELECT feature_json FROM prediction_input_snapshots WHERE prediction_id = ?", (prediction_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row["feature_json"])

    def evaluate_matured_predictions(self, ticker: Optional[str] = None) -> dict:
        """Finds stored predictions whose horizon has now elapsed (i.e.
        we can know the real outcome) and haven't been evaluated yet,
        compares predicted vs. actual return, and records the result —
        the data behind the Prediction Accuracy tracking (spec section
        21). This is a manual/on-demand evaluation (no background
        scheduler yet — that's Phase 4), triggered via API."""
        with db_cursor() as cur:
            query = """SELECT id, ticker, created_at, horizon_days, expected_return, signal, model_breakdown_json
                       FROM predictions WHERE id NOT IN (SELECT prediction_id FROM prediction_results)"""
            params = ()
            if ticker:
                query += " AND ticker = ?"
                params = (ticker.upper(),)
            cur.execute(query, params)
            pending = [dict(r) for r in cur.fetchall()]

        evaluated, skipped = 0, 0
        for pred in pending:
            created = datetime.fromisoformat(pred["created_at"])
            matured_at = created + timedelta(days=pred["horizon_days"] * 1.5)  # calendar-day buffer for weekends
            if datetime.now(timezone.utc) < matured_at:
                skipped += 1
                continue

            hist = market_data_service.get_historical(pred["ticker"], period="1y", interval="1d")
            if not hist.success or hist.data is None or hist.data.empty:
                skipped += 1
                continue

            df = hist.data
            created_date = created.date().isoformat()
            on_or_after = df[df["date"] >= created_date]
            if len(on_or_after) <= pred["horizon_days"]:
                skipped += 1
                continue

            entry_price = float(on_or_after["close"].iloc[0])
            exit_price = float(on_or_after["close"].iloc[pred["horizon_days"]])
            actual_return = (exit_price / entry_price - 1) * 100

            predicted_direction = 1 if pred["expected_return"] > 0.5 else (-1 if pred["expected_return"] < -0.5 else 0)
            actual_direction = 1 if actual_return > 0.5 else (-1 if actual_return < -0.5 else 0)
            was_correct = int(predicted_direction == actual_direction)

            # Also score each individual model's vote (not just the
            # ensemble's) against reality — this per-model track record
            # is what backend.forecasting.adaptive uses to adapt future
            # ensemble weighting toward whichever models have actually
            # been right, on real outcomes, not just backtested ones.
            per_model_correct = {}
            try:
                model_breakdown = json.loads(pred.get("model_breakdown_json") or "[]")
                per_model_correct = compute_per_model_correctness(model_breakdown, actual_direction)
            except Exception:  # noqa: BLE001
                pass  # missing/malformed breakdown shouldn't block the ensemble-level result from being recorded

            with db_cursor() as cur:
                cur.execute(
                    """INSERT INTO prediction_results
                       (prediction_id, evaluated_at, actual_return, was_correct, per_model_correct_json)
                       VALUES (?, ?, ?, ?, ?)""",
                    (pred["id"], datetime.now(timezone.utc).isoformat(), round(actual_return, 3),
                     was_correct, json.dumps(per_model_correct)),
                )
            evaluated += 1

        return {"evaluated": evaluated, "skipped_not_yet_matured_or_no_data": skipped}

    def get_prediction_accuracy(self, ticker: Optional[str] = None) -> dict:
        with db_cursor() as cur:
            query = """SELECT p.horizon_days, r.was_correct FROM predictions p
                       JOIN prediction_results r ON r.prediction_id = p.id"""
            params = ()
            if ticker:
                query += " WHERE p.ticker = ?"
                params = (ticker.upper(),)
            cur.execute(query, params)
            rows = cur.fetchall()

        by_horizon: dict[int, list[int]] = {}
        for row in rows:
            by_horizon.setdefault(row["horizon_days"], []).append(row["was_correct"])

        return {
            str(horizon): {
                "n_evaluated": len(results),
                "accuracy": round(sum(results) / len(results), 3) if results else None,
            }
            for horizon, results in by_horizon.items()
        }


prediction_service = PredictionService()
