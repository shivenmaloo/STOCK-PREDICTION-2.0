"""
Adaptive ensemble weighting — this is what makes the app's predictions
genuinely evolve as it's used, in an honest way (as opposed to fake
"personalization" based on clicks, which wouldn't make predictions
more accurate, just more manipulated-feeling).

The mechanism: every prediction stores which direction each individual
model voted (in predictions.model_breakdown_json). Once that
prediction's horizon matures and gets evaluated
(PredictionService.evaluate_matured_predictions), we now also check
whether EACH model's vote — not just the ensemble's final call — was
actually correct, and store that per-model.

Over time, this builds a real, live track record per model per ticker.
When building a new prediction, that live track record is blended with
the walk-forward validation accuracy (which is a backtest on
historical data) using a shrinkage approach: the live track record
only meaningfully shifts a model's ensemble weight once enough real
predictions have actually matured and been checked — a single lucky or
unlucky live prediction doesn't swing anything. This is standard
Bayesian shrinkage, not a fabricated confidence boost.
"""
from __future__ import annotations

import json
from typing import Optional

from backend.database.db import db_cursor

# How much "pseudo-weight" the walk-forward validated accuracy carries,
# expressed as an equivalent number of live predictions. A model needs
# roughly this many real evaluated predictions before its live track
# record meaningfully outweighs the backtest — deliberately
# conservative, since a handful of live predictions is a much smaller,
# noisier sample than a full walk-forward validation run.
WALK_FORWARD_PSEUDO_COUNT = 15
MIN_LIVE_SAMPLE_FOR_ANY_INFLUENCE = 3


def compute_per_model_correctness(model_breakdown: list[dict], actual_direction: int) -> dict[str, int]:
    """Given the stored votes from a matured prediction and the real
    outcome's direction (+1/0/-1), returns {model_name: 1 or 0} for
    whether each individual model's vote matched reality — not just
    whether the ensemble's final call did."""
    direction_to_sign = {"bullish": 1, "bearish": -1, "neutral": 0}
    result = {}
    for vote in model_breakdown:
        model_dir = direction_to_sign.get(vote.get("direction"), 0)
        result[vote["model"]] = int(model_dir == actual_direction and model_dir != 0)
    return result


def get_live_model_accuracy(model_name: str, ticker: Optional[str] = None) -> dict:
    """Returns {'accuracy': float|None, 'n': int, 'scope': 'ticker'|'global'|'none'}.
    Tries ticker-specific history first; if there isn't enough of it
    yet, falls back to this model's accuracy across ALL tickers (still
    a real, honest signal — just less specific) rather than pretending
    to have ticker-specific insight it doesn't have."""

    def _query(ticker_filter: Optional[str]) -> tuple[int, int]:
        with db_cursor() as cur:
            if ticker_filter:
                cur.execute(
                    """SELECT r.per_model_correct_json FROM prediction_results r
                       JOIN predictions p ON p.id = r.prediction_id
                       WHERE p.ticker = ? AND r.per_model_correct_json IS NOT NULL""",
                    (ticker_filter.upper(),),
                )
            else:
                cur.execute(
                    """SELECT per_model_correct_json FROM prediction_results
                       WHERE per_model_correct_json IS NOT NULL""",
                )
            rows = cur.fetchall()

        hits, total = 0, 0
        for row in rows:
            per_model = json.loads(row["per_model_correct_json"] or "{}")
            if model_name in per_model:
                total += 1
                hits += per_model[model_name]
        return hits, total

    if ticker:
        hits, n = _query(ticker)
        if n >= MIN_LIVE_SAMPLE_FOR_ANY_INFLUENCE:
            return {"accuracy": round(hits / n, 4), "n": n, "scope": "ticker"}

    hits, n = _query(None)
    if n >= MIN_LIVE_SAMPLE_FOR_ANY_INFLUENCE:
        return {"accuracy": round(hits / n, 4), "n": n, "scope": "global"}

    return {"accuracy": None, "n": n, "scope": "none"}


def blend_accuracy(walk_forward_accuracy: Optional[float], live: dict) -> dict:
    """Shrinkage blend: the live accuracy pulls the final number toward
    itself proportionally to how many real predictions back it up,
    capped so a small live sample can't dominate a much more thorough
    backtest. Returns the blended accuracy plus a note explaining
    exactly how it was derived — never a silent number."""
    if live["accuracy"] is None or live["n"] < MIN_LIVE_SAMPLE_FOR_ANY_INFLUENCE:
        return {
            "blended_accuracy": walk_forward_accuracy,
            "note": "Based on walk-forward validation only — not enough live predictions have matured yet to adjust this.",
        }

    if walk_forward_accuracy is None:
        return {
            "blended_accuracy": live["accuracy"],
            "note": f"Based on {live['n']} live evaluated predictions ({live['scope']} scope) — no walk-forward baseline was available.",
        }

    wf_weight = WALK_FORWARD_PSEUDO_COUNT
    live_weight = live["n"]
    blended = (walk_forward_accuracy * wf_weight + live["accuracy"] * live_weight) / (wf_weight + live_weight)

    return {
        "blended_accuracy": round(blended, 4),
        "note": (
            f"Blended from walk-forward validation ({walk_forward_accuracy * 100:.1f}%) and "
            f"{live['n']} live evaluated predictions ({live['scope']} scope, {live['accuracy'] * 100:.1f}% accurate) — "
            f"live track record carries more weight as more predictions mature."
        ),
    }
