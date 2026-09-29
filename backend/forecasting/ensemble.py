"""
Model D — Ensemble (spec sections 9, 11-13, 39).

Combines whichever models are available (Model A baseline, Model B
XGBoost, Model C LSTM — some may be missing if optional dependencies
aren't installed) into one weighted prediction, plus the signal-level
classification the UI actually displays.

Two design commitments carried over directly from the original spec,
because they matter more than model sophistication:

  1. Weights are NOT arbitrary. Each model's weight in the ensemble is
     its own walk-forward-validated direction accuracy (clipped above
     0), normalized to sum to 1 — a model that validated at 65%
     accuracy gets more say than one that validated at 51%. A model
     that didn't beat a coin flip gets ~0 weight rather than dragging
     the ensemble down.

  2. If the models genuinely disagree, the output says so and returns
     NEUTRAL / LOW CONFIDENCE — it does not force a directional call
     when the evidence is contradictory. This is the single most
     important honesty requirement in the entire prediction system.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelVote:
    model_name: str
    predicted_return: float
    direction: str  # "bullish" | "neutral" | "bearish"
    validated_accuracy: float | None
    weight: float


def _direction_from_return(predicted_return: float, dead_zone: float) -> str:
    if predicted_return > dead_zone:
        return "bullish"
    if predicted_return < -dead_zone:
        return "bearish"
    return "neutral"


def build_ensemble(model_predictions: dict[str, float],
                    model_accuracies: dict[str, float | None],
                    horizon_days: int) -> dict:
    """model_predictions: {model_name: predicted_return}
    model_accuracies: {model_name: walk-forward direction_accuracy or None}

    Returns the full ensemble breakdown: weighted expected return,
    prediction range, per-model votes, agreement count, and confidence
    — all derived from the actual inputs, nothing invented."""
    dead_zone = 0.01 if horizon_days <= 5 else 0.02

    votes: list[ModelVote] = []
    for name, pred in model_predictions.items():
        acc = model_accuracies.get(name)
        # A model with no validation accuracy (e.g. validation couldn't
        # run due to insufficient history) gets a small nonzero floor
        # weight rather than zero, so it still contributes a vote to
        # the agreement count even though it barely influences the
        # weighted average.
        raw_weight = max(acc - 0.5, 0.0) if acc is not None else 0.02
        votes.append(ModelVote(
            model_name=name, predicted_return=pred,
            direction=_direction_from_return(pred, dead_zone),
            validated_accuracy=acc, weight=raw_weight,
        ))

    total_weight = sum(v.weight for v in votes)
    if total_weight <= 0:
        # No model validated better than a coin flip — every vote gets
        # equal say rather than the ensemble silently doing nothing.
        for v in votes:
            v.weight = 1.0 / len(votes) if votes else 0.0
        total_weight = 1.0 if votes else 0.0

    for v in votes:
        v.weight = v.weight / total_weight if total_weight > 0 else 0.0

    weighted_return = sum(v.predicted_return * v.weight for v in votes)

    # Prediction range from the actual spread of model predictions —
    # not a fabricated confidence interval, just min/max of what the
    # models themselves actually said.
    all_preds = [v.predicted_return for v in votes]
    pred_low, pred_high = (min(all_preds), max(all_preds)) if all_preds else (None, None)

    bullish_votes = sum(1 for v in votes if v.direction == "bullish")
    bearish_votes = sum(1 for v in votes if v.direction == "bearish")
    neutral_votes = sum(1 for v in votes if v.direction == "neutral")
    n_models = len(votes)

    ensemble_direction = _direction_from_return(weighted_return, dead_zone)
    agreeing = {"bullish": bullish_votes, "bearish": bearish_votes, "neutral": neutral_votes}[ensemble_direction]
    agreement_ratio = agreeing / n_models if n_models > 0 else 0.0

    # Average validated accuracy of models that agree with the ensemble
    # direction — this, combined with agreement ratio, drives confidence.
    agreeing_accuracies = [v.validated_accuracy for v in votes
                            if v.direction == ensemble_direction and v.validated_accuracy is not None]
    avg_agreeing_accuracy = sum(agreeing_accuracies) / len(agreeing_accuracies) if agreeing_accuracies else 0.5

    # Confidence formula (documented, not a black box): purely
    # proportional to how much better than chance the agreeing models
    # validated at, scaled by how many models actually agree. Neither
    # factor alone is enough — models with zero demonstrated skill
    # (accuracy at exactly chance) must produce ~0 confidence
    # regardless of how many of them happen to agree, since agreement
    # between models with no real edge is just as likely to be
    # coincidence as signal. (A prior version of this formula added a
    # flat "agreement bonus" independent of accuracy, which meant two
    # models validated at literally coin-flip accuracy that happened
    # to agree still reported 50% confidence — found and fixed via
    # testing, since confidence claiming to mean something when the
    # underlying models have proven zero skill is exactly the kind of
    # false precision this whole module exists to avoid.)
    confidence = round(min(0.95, max(0.05,
        (avg_agreeing_accuracy - 0.5) * 2 * agreement_ratio
    )), 3)

    # The core honesty requirement: don't force a call when models
    # disagree. If agreement is weak, downgrade to neutral regardless
    # of what the weighted average technically computed to.
    low_confidence = agreement_ratio < 0.5 or n_models < 2
    final_direction = "neutral" if low_confidence else ensemble_direction

    return {
        "expected_return_pct": round(weighted_return * 100, 2),
        "prediction_range_pct": [
            round(pred_low * 100, 2) if pred_low is not None else None,
            round(pred_high * 100, 2) if pred_high is not None else None,
        ],
        "direction": final_direction,
        "confidence": confidence if not low_confidence else round(confidence * 0.5, 3),
        "model_agreement": f"{agreeing}/{n_models}",
        "agreement_ratio": round(agreement_ratio, 3),
        "low_confidence_disagreement": low_confidence,
        "votes": [
            {"model": v.model_name, "predicted_return_pct": round(v.predicted_return * 100, 2),
             "direction": v.direction, "validated_accuracy": v.validated_accuracy, "weight": round(v.weight, 3)}
            for v in votes
        ],
    }


SIGNAL_THRESHOLDS = {
    # Recalibrated for the corrected confidence formula (see the fix
    # above and its regression test): confidence now scales purely
    # from (accuracy - 0.5) * 2 * agreement_ratio, with no flat
    # "agreement bonus," so realistic accuracy levels (55-80% direction
    # accuracy is the genuinely achievable range for stock prediction)
    # top out around 0.10-0.60 confidence, not 0.50-0.95 the way the
    # old, inflated formula did. These thresholds are picked so the
    # signal tiers stay meaningfully reachable at realistic accuracy
    # levels while still requiring real, validated skill — not just
    # models agreeing with each other regardless of whether they've
    # demonstrated any actual edge.
    "strong_bullish_min_confidence": 0.40,
    "strong_bullish_min_return_pct": 3.0,
    "bullish_watch_min_confidence": 0.15,
    "bullish_watch_min_return_pct": 1.0,
    "bearish_watch_min_confidence": 0.15,
    "bearish_watch_min_return_pct": -1.0,
    "high_risk_min_confidence": 0.40,
    "high_risk_max_return_pct": -3.0,
}


def classify_signal(ensemble: dict, technical_bullish: bool, technical_bearish: bool,
                     thresholds: dict | None = None, news_agrees: bool | None = None) -> dict:
    """Maps the ensemble output to one of the six signal levels (spec
    section 12), requiring MULTIPLE factors to agree — never just the
    raw model output alone. A signal only escalates to "strong" when
    the models, the existing technical-indicator read (from
    backend/indicators/technical.py), AND current news sentiment (when
    available) all point the same direction.

    news_agrees is deliberately a THIRD independent input here, not a
    trained feature — it comes from backend.forecasting.news_context,
    which only ever looks at TODAY's news. Requiring it (when available)
    for the strongest signal tiers is honest: it uses only information
    that genuinely exists right now, never touches historical training
    data, and mirrors exactly how a careful analyst would double-check
    a technical setup against the headlines before calling it "strong"
    conviction. When news_agrees is None (no news available, or the
    news read was neutral), it simply doesn't gate anything — silence
    isn't treated as disagreement."""
    t = {**SIGNAL_THRESHOLDS, **(thresholds or {})}

    direction = ensemble["direction"]
    confidence = ensemble["confidence"]
    expected_return = ensemble["expected_return_pct"]
    news_contradicts = news_agrees is False

    if ensemble["low_confidence_disagreement"]:
        signal = "NEUTRAL"
        reason = "Models disagree on direction — showing neutral rather than forcing a call."
    elif direction == "bullish":
        if (confidence >= t["strong_bullish_min_confidence"] and
                expected_return >= t["strong_bullish_min_return_pct"] and technical_bullish and
                not news_contradicts):
            signal = "STRONG_BULLISH_SETUP"
            reason = "High model confidence, strong expected return, technical trend, and current news all agree bullish."
        elif confidence >= t["bullish_watch_min_confidence"] and expected_return >= t["bullish_watch_min_return_pct"]:
            signal = "BULLISH_WATCH"
            if news_contradicts:
                reason = "Models lean bullish, but recent news sentiment runs the other way — kept at Watch rather than Strong."
            else:
                reason = "Models lean bullish with reasonable confidence."
        else:
            signal = "NEUTRAL"
            reason = "Bullish lean, but confidence or expected return too weak to escalate."
    elif direction == "bearish":
        if (confidence >= t["high_risk_min_confidence"] and
                expected_return <= t["high_risk_max_return_pct"] and technical_bearish and
                not news_contradicts):
            signal = "HIGH_RISK_POSSIBLE_EXIT"
            reason = "High model confidence, sharply negative expected return, technical breakdown, and current news all agree bearish."
        elif confidence >= t["bearish_watch_min_confidence"] and expected_return <= t["bearish_watch_min_return_pct"]:
            signal = "BEARISH_WATCH"
            if news_contradicts:
                reason = "Models lean bearish, but recent news sentiment runs the other way — kept at Watch rather than High Risk."
            else:
                reason = "Models lean bearish with reasonable confidence."
        else:
            signal = "NEUTRAL"
            reason = "Bearish lean, but confidence or expected return too weak to escalate."
    else:
        signal = "NEUTRAL"
        reason = "No clear directional edge from the models."

    return {"signal": signal, "reason": reason}
