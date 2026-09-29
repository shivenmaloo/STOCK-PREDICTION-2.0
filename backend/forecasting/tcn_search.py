"""
Ongoing TCN + risk-map configuration search — "keep training it to
find the actual best solution," done honestly.

This directly targets a real, well-known failure mode: trying enough
configurations that one looks good purely by chance is a form of
overfitting to the search process itself, not just to training data —
exactly the "AI often over-optimizes / hallucinates patterns" problem
that real quant researchers have to guard against by hand. The
safeguards here are not optional extras, they're the actual point:

  1. SEARCH/HOLDOUT SPLIT: a ticker's history is split chronologically
     into a SEARCH period (used to compare candidate configs against
     each other) and a HOLDOUT period (the most recent slice, never
     touched while comparing candidates). A config only gets ranked
     using the search period.
  2. GENUINE OUT-OF-SAMPLE SCORING: each candidate's search-period
     score comes from real walk-forward validation (chronological,
     expanding-window — see backend/forecasting/validation.py), not a
     single train/test split, and requires a minimum number of splits
     before being trusted at all.
  3. EVERY TRIAL PERSISTED: nothing is silently discarded — the full
     search history is auditable, so a suspicious result can be
     inspected rather than hidden inside a single "winner."
  4. ONE-TIME HOLDOUT CHECK: only once a best config is chosen does it
     ever get evaluated on the holdout period — and BOTH numbers
     (search score and holdout score) are always shown side by side.
     A large gap between them is the visible, honest signal that the
     search overfit, not something this module tries to hide.
  5. INCREMENTAL, NOT ALL-AT-ONCE: the background scheduler runs a
     handful of new candidates per pass (same conservative pattern
     already used for the discovery sweep), so this builds up over
     time rather than hammering the CPU in one big sweep.
"""
from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from backend.database.db import db_cursor

logger = logging.getLogger("stock_ai.tcn_search")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False

HOLDOUT_FRACTION = 0.2  # the most recent 20% of history is never touched during search
MIN_SEARCH_SPLITS_TO_TRUST = 3  # a candidate scored on fewer walk-forward splits than this is too noisy to trust


@dataclass(frozen=True)
class TCNSearchConfig:
    # TCN architecture
    num_layers: int = 4
    kernel_size: int = 3
    hidden_channels: int = 16
    dropout: float = 0.2
    window_len: int = 64
    # Risk map
    dead_zone_threshold: float = 0.1
    target_volatility: float = 0.02
    max_position_pct: float = 0.20

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "TCNSearchConfig":
        return TCNSearchConfig(**json.loads(s))


# The search space — deliberately modest (a few reasonable values per
# dimension, not a sprawling grid) since each candidate means real
# model training, not a free lookup.
SEARCH_SPACE = {
    "num_layers": [3, 4, 5],
    "kernel_size": [2, 3, 5],
    "hidden_channels": [16, 32],
    "dropout": [0.1, 0.2, 0.3],
    "dead_zone_threshold": [0.0, 0.05, 0.1, 0.15],
}


def split_search_and_holdout(n_rows: int, holdout_fraction: float = HOLDOUT_FRACTION) -> tuple[int, int]:
    """Returns (search_end_idx, holdout_start_idx) — chronological,
    non-overlapping. search_end_idx == holdout_start_idx; the split is
    a single cut point, with the search period being everything before
    it and the holdout period everything from it onward."""
    holdout_start_idx = int(n_rows * (1 - holdout_fraction))
    return holdout_start_idx, holdout_start_idx


def get_existing_trial_configs(ticker: str, horizon_days: int) -> set[str]:
    with db_cursor() as cur:
        cur.execute("SELECT config_json FROM tcn_search_trials WHERE ticker = ? AND horizon_days = ?",
                     (ticker.upper(), horizon_days))
        return {row["config_json"] for row in cur.fetchall()}


def generate_candidate_configs(ticker: str, horizon_days: int, n_new: int = 3) -> list[TCNSearchConfig]:
    """Proposes up to n_new configs from SEARCH_SPACE that haven't
    already been tried for this ticker/horizon — a plain grid walk,
    deduplicated against trial history, not random sampling, so the
    search space gets covered systematically over time rather than
    potentially re-rolling the same few configs repeatedly."""
    already_tried = get_existing_trial_configs(ticker, horizon_days)

    keys = list(SEARCH_SPACE.keys())
    all_combos = itertools.product(*(SEARCH_SPACE[k] for k in keys))

    candidates = []
    for combo in all_combos:
        kwargs = dict(zip(keys, combo))
        config = TCNSearchConfig(**kwargs)
        if config.to_json() in already_tried:
            continue
        candidates.append(config)
        if len(candidates) >= n_new:
            break
    return candidates


def record_trial(ticker: str, horizon_days: int, config: TCNSearchConfig,
                  search_score: float, n_search_splits: int, n_search_trades: int) -> int:
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO tcn_search_trials (ticker, horizon_days, created_at, config_json, search_score, "
            "n_search_splits, n_search_trades) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticker.upper(), horizon_days, datetime.now(timezone.utc).isoformat(),
             config.to_json(), search_score, n_search_splits, n_search_trades),
        )
        trial_id = cur.lastrowid

    # Update "best so far" if this trial genuinely beats it — only
    # trials with enough splits to trust are eligible at all.
    if n_search_splits >= MIN_SEARCH_SPLITS_TO_TRUST:
        current_best = get_best_config(ticker, horizon_days)
        if current_best is None or search_score > current_best["search_score"]:
            with db_cursor() as cur:
                cur.execute(
                    "INSERT INTO tcn_search_best (ticker, horizon_days, trial_id, updated_at, holdout_score, holdout_evaluated_at) "
                    "VALUES (?, ?, ?, ?, NULL, NULL) "
                    "ON CONFLICT(ticker, horizon_days) DO UPDATE SET "
                    "trial_id = excluded.trial_id, updated_at = excluded.updated_at, "
                    "holdout_score = NULL, holdout_evaluated_at = NULL",
                    (ticker.upper(), horizon_days, trial_id, datetime.now(timezone.utc).isoformat()),
                )
            logger.info(f"[TCN SEARCH] New best config for {ticker} ({horizon_days}D): trial {trial_id}, score {search_score:.4f}")

    return trial_id


def get_best_config(ticker: str, horizon_days: int) -> dict | None:
    with db_cursor() as cur:
        cur.execute("""
            SELECT b.trial_id, b.holdout_score, b.holdout_evaluated_at, b.updated_at,
                   t.config_json, t.search_score, t.n_search_splits, t.n_search_trades
            FROM tcn_search_best b JOIN tcn_search_trials t ON t.id = b.trial_id
            WHERE b.ticker = ? AND b.horizon_days = ?
        """, (ticker.upper(), horizon_days))
        row = cur.fetchone()
    if row is None:
        return None
    result = dict(row)
    result["config"] = TCNSearchConfig.from_json(result.pop("config_json"))
    return result


def record_holdout_evaluation(ticker: str, horizon_days: int, holdout_score: float) -> None:
    """Records the ONE-TIME holdout check for the current best config.
    Called at most once per new best config, never repeatedly against
    the same holdout data — re-checking the same holdout period
    against many candidates would make the holdout just another search
    period, defeating its entire purpose."""
    with db_cursor() as cur:
        cur.execute(
            "UPDATE tcn_search_best SET holdout_score = ?, holdout_evaluated_at = ? "
            "WHERE ticker = ? AND horizon_days = ?",
            (holdout_score, datetime.now(timezone.utc).isoformat(), ticker.upper(), horizon_days),
        )


def get_trial_history(ticker: str, horizon_days: int, limit: int = 50) -> list[dict]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, created_at, config_json, search_score, n_search_splits, n_search_trades "
            "FROM tcn_search_trials WHERE ticker = ? AND horizon_days = ? ORDER BY created_at DESC LIMIT ?",
            (ticker.upper(), horizon_days, limit),
        )
        rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["config"] = TCNSearchConfig.from_json(r.pop("config_json"))
    return rows


def run_search_iteration(ticker: str, horizon_days: int = 5, n_candidates: int = 2) -> dict:
    """The actual background-job entry point: tries a handful of new
    candidate configs (real model training per candidate — this is
    real CPU work, not a quick check, which is why n_candidates is
    small and this runs on a slow scheduler cadence), then — only if
    the current best config hasn't had its one-time holdout check yet
    — runs that single holdout evaluation. Deferred import of
    tcn_service to avoid a circular dependency (tcn_service imports
    from this module for TCNSearchConfig/split_search_and_holdout)."""
    from backend.forecasting import tcn_service

    if not tcn_service.is_available():
        return {"evaluated": 0, "reason": "PyTorch is not installed — TCN search unavailable."}

    candidates = generate_candidate_configs(ticker, horizon_days, n_new=n_candidates)
    evaluated = 0
    for config in candidates:
        result = tcn_service.evaluate_config_on_search_period(ticker, horizon_days, config)
        if result is None:
            continue
        record_trial(ticker, horizon_days, config, result["score"], result["n_splits"], result["n_trades"])
        evaluated += 1

    best = get_best_config(ticker, horizon_days)
    holdout_checked = False
    if best is not None and best["holdout_score"] is None:
        holdout_result = tcn_service.evaluate_best_config_on_holdout(ticker, horizon_days, best["config"])
        if holdout_result is not None:
            record_holdout_evaluation(ticker, horizon_days, holdout_result["score"])
            holdout_checked = True

    return {"candidates_tried": len(candidates), "evaluated": evaluated, "holdout_checked": holdout_checked}
