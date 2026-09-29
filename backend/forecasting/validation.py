"""
Walk-forward / expanding-window validation (spec section 10).

This is the single most important correctness requirement in the
whole prediction system. Financial time series MUST NOT be randomly
shuffled for train/test splits — a random split lets the model "see
the future" relative to some of its training examples (e.g. training
on Tuesday's data while testing on Monday's), which produces
suspiciously good-looking metrics that don't hold up in real trading.

How this prevents leakage:
  - Splits are strictly chronological: every training set only contains
    rows that occurred BEFORE its corresponding test set.
  - The window expands forward through time (walk-forward), never
    backward, never overlapping in the wrong direction.
  - Feature engineering (backend/forecasting/features.py) already only
    uses rolling windows ending at or before the current row, and
    targets are dropped for rows whose future outcome isn't known yet
    — so there's no leakage baked into the data itself either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd


def walk_forward_splits(n_rows: int, min_train_size: int, step_size: int
                         ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns a list of (train_idx, test_idx) index arrays. Each split's
    train indices are a strict, expanding prefix of the data; each
    split's test indices are the next `step_size` rows immediately
    after — never shuffled, never overlapping with train."""
    splits = []
    i = min_train_size
    while i < n_rows:
        train_idx = np.arange(0, i)
        test_idx = np.arange(i, min(i + step_size, n_rows))
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
        i += step_size
    return splits


@dataclass
class WalkForwardResult:
    model_name: str
    n_splits: int
    direction_accuracy: Optional[float]
    mae: Optional[float]
    predicted_returns: list[float] = field(default_factory=list)
    actual_returns: list[float] = field(default_factory=list)
    predicted_directions: list[int] = field(default_factory=list)
    actual_directions: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_splits": self.n_splits,
            "direction_accuracy": self.direction_accuracy,
            "mae": self.mae,
        }


def run_walk_forward(
    model_factory: Callable[[], object],
    X: pd.DataFrame, y_return: pd.Series, y_direction: pd.Series,
    model_name: str, min_train_size: int = 250, step_size: int = 21,
) -> WalkForwardResult:
    """Trains a fresh model on each expanding window and evaluates it
    ONLY on the immediately-following unseen block, exactly mimicking
    how the model would actually be used in production: train on
    everything known so far, predict the near future, then (in the
    next iteration) fold that period into training and predict the
    next block after it.

    A fresh model is fit on every split (not incrementally updated) —
    slower, but it means every single prediction in the results below
    came from a model that never saw that data point during training,
    which is the whole point."""
    n = len(X)
    splits = walk_forward_splits(n, min_train_size, step_size)

    result = WalkForwardResult(model_name=model_name, n_splits=len(splits),
                                direction_accuracy=None, mae=None)

    if not splits:
        return result

    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[train_idx], y_return.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test_return = y_return.iloc[test_idx]
        y_test_direction = y_direction.iloc[test_idx]

        model = model_factory()
        model.fit(X_train.values, y_train.values)
        preds = model.predict(X_test.values)

        result.predicted_returns.extend(np.asarray(preds).tolist())
        result.actual_returns.extend(y_test_return.tolist())

        pred_direction = np.sign(np.where(np.abs(preds) < 1e-6, 0, preds))
        result.predicted_directions.extend(pred_direction.astype(int).tolist())
        result.actual_directions.extend(y_test_direction.tolist())

    if result.actual_returns:
        errors = np.abs(np.array(result.predicted_returns) - np.array(result.actual_returns))
        result.mae = round(float(errors.mean()), 5)

        # Direction accuracy only counted on rows where the actual
        # outcome wasn't in the "neutral" dead-zone — scoring a model
        # against genuinely ambiguous flat periods isn't meaningful.
        pred_dir = np.array(result.predicted_directions)
        actual_dir = np.array(result.actual_directions)
        scored_mask = actual_dir != 0
        if scored_mask.sum() > 0:
            result.direction_accuracy = round(
                float((pred_dir[scored_mask] == actual_dir[scored_mask]).mean()), 4
            )

    return result


def compute_conformal_radius(result: WalkForwardResult, confidence: float = 0.80) -> Optional[float]:
    """Split conformal prediction: uses this model's own walk-forward
    out-of-sample residuals (|actual - predicted| return, across every
    historical test block) to derive a prediction interval radius with
    an approximate coverage guarantee — e.g. at confidence=0.80, ~80%
    of historical out-of-sample outcomes fell within ±radius of the
    prediction.

    This is more robust than the textbook version of this technique
    (which typically calibrates on one held-out slice of data): here
    the residuals come from ~40 different rolling walk-forward windows
    spanning multiple market regimes, not one arbitrary time period
    that could just be luckier or unluckier than usual.

    Returns None if there isn't enough held-out history yet for the
    quantile to be meaningful — an honest "not enough data" rather
    than a fabricated interval from a handful of residuals."""
    residuals = np.abs(np.array(result.predicted_returns) - np.array(result.actual_returns))
    if len(residuals) < 20:
        return None
    return float(np.quantile(residuals, confidence))


def compute_naive_baseline_comparison(result: WalkForwardResult) -> dict:
    """Checks whether this model actually beats the simplest possible
    baseline — predicting zero return (no change) — over the exact
    same out-of-sample windows. A model that can't beat "predict
    nothing happens" hasn't learned anything useful, regardless of how
    good its other metrics look."""
    actual = np.array(result.actual_returns)
    predicted = np.array(result.predicted_returns)
    if len(actual) == 0:
        return {"beats_naive_baseline": None, "naive_mae": None, "model_mae": None}

    naive_mae = float(np.abs(actual).mean())  # MAE of always predicting 0
    model_mae = float(np.abs(actual - predicted).mean())
    return {
        "beats_naive_baseline": model_mae < naive_mae,
        "naive_mae": round(naive_mae, 5),
        "model_mae": round(model_mae, 5),
    }


def trading_stats_from_walk_forward(result: WalkForwardResult) -> dict:
    """Converts walk-forward results into win-rate / avg-win / avg-loss
    statistics — genuine out-of-sample numbers (not fabricated), used
    by the risk-sizing module. 'Win' here means the model's predicted
    direction matched the actual direction on a non-neutral outcome."""
    pred_dir = np.array(result.predicted_directions)
    actual_ret = np.array(result.actual_returns)
    actual_dir = np.array(result.actual_directions)

    traded_mask = pred_dir != 0  # only count periods the model actually took a directional view
    if traded_mask.sum() == 0:
        return {"win_rate": None, "avg_win_pct": None, "avg_loss_pct": None, "n_trades": 0}

    traded_returns = actual_ret[traded_mask]
    traded_pred_dir = pred_dir[traded_mask]
    traded_actual_dir = actual_dir[traded_mask]

    wins_mask = traded_pred_dir == traded_actual_dir
    wins = traded_returns[wins_mask]
    losses = traded_returns[~wins_mask]

    win_rate = float(wins_mask.mean())
    avg_win = float(np.abs(wins).mean()) if len(wins) > 0 else None
    avg_loss = float(np.abs(losses).mean()) if len(losses) > 0 else None

    return {
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(avg_win * 100, 3) if avg_win is not None else None,
        "avg_loss_pct": round(avg_loss * 100, 3) if avg_loss is not None else None,
        "n_trades": int(traded_mask.sum()),
    }
