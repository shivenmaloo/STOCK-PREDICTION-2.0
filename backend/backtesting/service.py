"""
Backtesting engine (Phase 5, spec sections 22-25).

This is NOT the same thing as walk-forward validation (which scores
"how accurate was each model, on average, out-of-sample") — this
simulates actually TRADING on those predictions over time: an equity
curve, drawdown, win rate, and a comparison against simply buying and
holding the same stock over the same period.

The one rule that matters more than anything else here, carried over
directly from everything already built in Phases 3-4: every decision
the simulated strategy makes — which models to trust more, how big a
position to take — must only ever use information that was actually
available at that point in the simulated timeline. Concretely:

  - Each model's contribution to the ensemble at a given point in time
    is weighted by its accuracy on walk-forward splits BEFORE that
    point only — never including the split currently being traded,
    and never including future splits.
  - Position sizing (fractional Kelly) at a given point uses only the
    win-rate/avg-win/avg-loss from trades already closed before that
    point — the same "matured, evaluated predictions" logic used
    elsewhere in the app, just replayed historically instead of live.

This is slower and more complex than just replaying stored walk-forward
predictions naively, but it's the only way to call the result an
honest backtest rather than a backtest that quietly cheats.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.data.market_data_service import market_data_service
from backend.database.db import db_cursor
from backend.indicators.technical import compute_all_indicators
from backend.forecasting.features import build_features, build_training_matrix
from backend.forecasting.validation import walk_forward_splits
from backend.forecasting.models import get_available_sklearn_models, get_available_xgboost_model
from backend.forecasting.risk import kelly_fraction, DEFAULT_KELLY_FRACTION, MAX_POSITION_PCT, MIN_TRADES_FOR_KELLY
from backend.forecasting.service import _fetch_aligned_benchmark, BENCHMARK_TICKER

logger = logging.getLogger("stock_ai.backtest")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False

STARTING_CAPITAL = 10_000.0
MIN_ROWS_FOR_BACKTEST = 300
BACKTEST_MIN_TRAIN = 200
BACKTEST_STEP = 21
ROUND_TRIP_TRANSACTION_COST_PCT = 0.001  # 0.10% round-trip — a conservative, disclosed estimate
# for a liquid US equity (bid-ask spread plus minimal/zero commission at a modern retail broker).
# Not a live, ticker-specific slippage model — just the difference between "ignoring costs
# entirely" (a well-known, textbook way a backtest can overstate real performance) and a
# reasonable, stated assumption. Applied per round-trip trade, scaled by position size like
# any other cost that scales with how much capital was actually deployed.


def _get_backtest_model_factories() -> dict:
    # Deliberately excludes the LSTM — a backtest re-trains a fresh
    # model set per split just like walk-forward validation does, and
    # doing that with a neural net across many splits would make this
    # take a very long time for what's meant to be a fairly quick
    # analysis tool. Tree/linear models only, consistent with keeping
    # this fast enough to actually use.
    factories = dict(get_available_sklearn_models())
    factories.update(get_available_xgboost_model())
    return factories


def _simulate_one_trade(direction: int, actual_return: float, closed_trade_returns: list[float],
                         kelly_fraction_multiplier: float) -> tuple[float, float | None, float]:
    """Shared by the ensemble's own simulation and each individual
    model's parallel simulation, so both use IDENTICAL sizing logic —
    the only thing that should differ between them is which
    direction/history feeds in, never the mechanics of turning that
    into a position size and P&L. Returns (realized_return_on_position,
    net_trade_pl_pct_or_None, position_pct) — the second value is None
    when no trade was taken at all (direction == 0)."""
    if direction == 0:
        return 0.0, None, 0.0

    if len(closed_trade_returns) >= MIN_TRADES_FOR_KELLY:
        wins = [r for r in closed_trade_returns if r > 0]
        losses = [r for r in closed_trade_returns if r <= 0]
        win_rate = len(wins) / len(closed_trade_returns)
        avg_win = (sum(wins) / len(wins)) if wins else None
        avg_loss = (abs(sum(losses) / len(losses))) if losses else None
        f_star = kelly_fraction(win_rate, avg_win, avg_loss) if (avg_win and avg_loss) else None
        position_pct = min(max(f_star or 0, 0) * kelly_fraction_multiplier, MAX_POSITION_PCT / 100)
    else:
        position_pct = 0.02  # small default stake until there's enough real history to size properly

    gross_trade_pl_pct = direction * actual_return
    net_trade_pl_pct = gross_trade_pl_pct - ROUND_TRIP_TRANSACTION_COST_PCT
    realized_return_on_position = net_trade_pl_pct * position_pct
    return realized_return_on_position, net_trade_pl_pct, position_pct


def run_backtest(ticker: str, horizon_days: int, kelly_fraction_multiplier: float = DEFAULT_KELLY_FRACTION) -> dict:
    ticker = ticker.upper()
    logger.info(f"[BACKTEST] === Starting backtest for {ticker} ({horizon_days}D) ===")

    # Real inconsistency found and fixed via testing: train_and_predict()
    # already has a proven "relaxed mode" for short-history tickers
    # (down to a 120-row absolute floor), but this backtest engine had
    # no equivalent — it refused outright below a fixed 300-row bar.
    # That meant a ticker like a recent IPO could get a real prediction
    # but never a backtest, which read as broken rather than as the
    # genuinely different data requirement it actually was. Mirrors
    # the exact same relaxed-mode scaling already proven in
    # backend/forecasting/service.py, not a new, separately-invented
    # threshold scheme.
    MIN_ROWS_RELAXED_FLOOR = 120  # matches train_and_predict()'s own floor, for consistency across the app
    hist_result = market_data_service.get_historical(ticker, period="5y", interval="1d")
    available_rows = len(hist_result.data) if hist_result.data is not None else 0
    if not hist_result.success or available_rows < MIN_ROWS_RELAXED_FLOOR:
        return {"success": False, "error": (
            f"Not enough historical data for {ticker} to run a backtest — only {available_rows} trading "
            f"days available, need at least {MIN_ROWS_RELAXED_FLOOR} (roughly 6 months) even in relaxed mode."
        )}

    indicators_df = compute_all_indicators(hist_result.data)
    benchmark_close = _fetch_aligned_benchmark(indicators_df["date"]) if ticker != BENCHMARK_TICKER else None
    features_df = build_features(indicators_df, benchmark_close=benchmark_close)
    X, y_return, y_direction, aligned = build_training_matrix(features_df, horizon_days, extra_columns=["date", "close"])
    dates = aligned["date"]
    close_prices = aligned["close"]

    min_feature_rows_relaxed = 60  # matches train_and_predict()'s own hard floor
    if len(X) < min_feature_rows_relaxed:
        return {"success": False, "error": (
            f"Not enough complete feature rows for {ticker} after removing warm-up/NaN periods "
            f"({len(X)} rows) — even relaxed mode needs at least {min_feature_rows_relaxed}."
        )}

    relaxed_mode = len(X) < (BACKTEST_MIN_TRAIN + BACKTEST_STEP)
    if relaxed_mode:
        logger.info(f"[BACKTEST] Only {len(X)} feature rows available (need {BACKTEST_MIN_TRAIN + BACKTEST_STEP}+ "
                    f"for standard backtesting) — using RELAXED mode with scaled-down walk-forward validation")
        effective_min_train = max(40, int(len(X) * 0.5))
        effective_step = max(5, int(len(X) * 0.12))
    else:
        effective_min_train = BACKTEST_MIN_TRAIN
        effective_step = BACKTEST_STEP

    if len(X) < effective_min_train + effective_step:
        return {"success": False, "error": f"Not enough complete feature rows for {ticker} to run a backtest."}

    factories = _get_backtest_model_factories()
    splits = walk_forward_splits(len(X), effective_min_train, effective_step)
    logger.info(f"[BACKTEST] {len(splits)} splits, {len(factories)} models: {list(factories.keys())}")

    # Running, split-by-split accuracy per model — used ONLY to weight
    # PAST splits' models when scoring the CURRENT split, never
    # including the current or future splits' outcomes.
    model_correct_history: dict[str, list[int]] = {name: [] for name in factories}
    # Running trade outcomes — used the same way for position sizing.
    closed_trade_returns: list[float] = []

    equity = STARTING_CAPITAL
    equity_curve: list[dict] = []
    trade_log: list[dict] = []
    dead_zone = 0.01 if horizon_days <= 5 else 0.02
    first_tested_row_pos: int | None = None
    last_tested_row_pos: int | None = None

    # Per-model tracking, run in parallel with the ensemble above —
    # the real answer to "if I'd traded on Model X alone, how would it
    # have actually done historically?" Each model gets its own
    # independent equity curve, its own Kelly-sized positions (using
    # ONLY that model's own closed-trade history, never the
    # ensemble's), and its own trade log — same discipline as the
    # ensemble, just isolated per model so the comparison is genuinely
    # apples-to-apples.
    per_model_equity: dict[str, float] = {name: STARTING_CAPITAL for name in factories}
    per_model_equity_curve: dict[str, list[dict]] = {name: [] for name in factories}
    per_model_closed_trades: dict[str, list[float]] = {name: [] for name in factories}
    per_model_trade_counts: dict[str, int] = {name: 0 for name in factories}
    per_model_correct_counts: dict[str, int] = {name: 0 for name in factories}

    for split_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y_return.iloc[train_idx]
        X_test = X.iloc[test_idx]

        trained = {}
        for name, factory in factories.items():
            try:
                model = factory()
                model.fit(X_train.values, y_train.values)
                trained[name] = model
            except Exception as exc:  # noqa: BLE001
                logger.info(f"[BACKTEST] {name} failed on split {split_idx}: {exc}")

        if not trained:
            continue

        # Weights for THIS split, derived only from splits 0..split_idx-1.
        weights = {}
        for name in trained:
            history = model_correct_history[name]
            acc = (sum(history) / len(history)) if history else 0.5
            weights[name] = max(acc - 0.5, 0.02)
        total_weight = sum(weights.values()) or 1.0
        weights = {k: v / total_weight for k, v in weights.items()}

        preds_by_model = {name: model.predict(X_test.values) for name, model in trained.items()}

        for i, row_pos in enumerate(test_idx):
            if first_tested_row_pos is None:
                first_tested_row_pos = row_pos
            last_tested_row_pos = row_pos

            weighted_return = sum(preds_by_model[name][i] * weights[name] for name in trained)
            direction = 1 if weighted_return > dead_zone else -1 if weighted_return < -dead_zone else 0
            actual_return = float(y_return.iloc[row_pos])
            actual_direction = 1 if actual_return > dead_zone else -1 if actual_return < -dead_zone else 0

            trade_taken = direction != 0
            realized_return_on_position, net_trade_pl_pct, position_pct = _simulate_one_trade(
                direction, actual_return, closed_trade_returns, kelly_fraction_multiplier)

            if trade_taken:
                equity *= (1 + realized_return_on_position)
                trade_log.append({
                    "date": str(dates.iloc[row_pos]), "direction": "long" if direction == 1 else "short",
                    "position_pct": round(position_pct * 100, 2),
                    "actual_return_pct": round(actual_return * 100, 2),
                    "trade_pl_pct": round(realized_return_on_position * 100, 3), "correct": direction == actual_direction,
                })
                # Kept net of costs so future Kelly sizing is appropriately more
                # conservative once costs are accounted for — a real edge has to
                # overcome them to be worth sizing up on.
                closed_trade_returns.append(net_trade_pl_pct)

            equity_curve.append({"date": str(dates.iloc[row_pos]), "equity": round(equity, 2)})

            # Per-model simulation: each model gets the EXACT same
            # treatment (own direction call, own dead-zone, own Kelly
            # history, same cost assumption) as the ensemble above —
            # the only difference is which prediction feeds in.
            for name in trained:
                model_pred = preds_by_model[name][i]
                model_direction = 1 if model_pred > dead_zone else -1 if model_pred < -dead_zone else 0
                model_return, model_net_pl, _ = _simulate_one_trade(
                    model_direction, actual_return, per_model_closed_trades[name], kelly_fraction_multiplier)
                if model_direction != 0:
                    per_model_equity[name] *= (1 + model_return)
                    per_model_closed_trades[name].append(model_net_pl)
                    per_model_trade_counts[name] += 1
                    if model_direction == actual_direction:
                        per_model_correct_counts[name] += 1
                per_model_equity_curve[name].append({"date": str(dates.iloc[row_pos]), "equity": round(per_model_equity[name], 2)})

        # Now that this split's true outcomes are known, fold them into
        # each model's running accuracy — available for FUTURE splits only.
        for name in trained:
            preds = preds_by_model[name]
            for i, row_pos in enumerate(test_idx):
                pred_dir = 1 if preds[i] > dead_zone else -1 if preds[i] < -dead_zone else 0
                actual_dir = 1 if y_return.iloc[row_pos] > dead_zone else -1 if y_return.iloc[row_pos] < -dead_zone else 0
                if actual_dir != 0:
                    model_correct_history[name].append(int(pred_dir == actual_dir))

    if not equity_curve:
        return {"success": False, "error": "Backtest produced no trades — not enough validated history to simulate."}

    stats = _compute_stats(equity_curve, trade_log, close_prices, first_tested_row_pos, last_tested_row_pos)
    logger.info(f"[BACKTEST] === DONE for {ticker}: total_return={stats['total_return_pct']}%, "
                f"vs buy-and-hold={stats['buy_and_hold_return_pct']}% ===")

    # The direct answer to "which model would actually have been best,
    # and by how much": each model's own final return, using the exact
    # same starting capital and cost assumptions as the ensemble, so
    # the comparison is genuinely apples-to-apples — not a proxy
    # metric, the real simulated outcome of trading on that model alone.
    per_model_results = {}
    from backend.backtesting.sanity_checks import check_backtest_sanity
    first_date = datetime.fromisoformat(str(equity_curve[0]["date"])) if equity_curve else None
    last_date = datetime.fromisoformat(str(equity_curve[-1]["date"])) if equity_curve else None
    n_days_tested = max((last_date - first_date).days, 1) if first_date and last_date else 1

    for name in factories:
        curve = per_model_equity_curve[name]
        if not curve:
            per_model_results[name] = {"total_return_pct": None, "n_trades": 0, "win_rate": None,
                                        "note": "This model never produced a usable prediction across any split."}
            continue
        final_equity = curve[-1]["equity"]
        total_return_pct = round((final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2)
        n_trades = per_model_trade_counts[name]
        win_rate = round(per_model_correct_counts[name] / n_trades, 4) if n_trades > 0 else None

        # Same automated safety net as the ensemble/buy-and-hold stats
        # above — a per-model return is just as capable of hitting the
        # same kind of bug that produced the originally reported
        # impossible number, so it gets the same check, not a pass.
        model_max_drawdown = 0.0
        running_max = STARTING_CAPITAL
        for point in curve:
            running_max = max(running_max, point["equity"])
            model_max_drawdown = min(model_max_drawdown, (point["equity"] - running_max) / running_max * 100)
        model_sanity_warnings = check_backtest_sanity(total_return_pct, model_max_drawdown, n_days_tested, label=name)

        per_model_results[name] = {
            "total_return_pct": total_return_pct, "n_trades": n_trades, "win_rate": win_rate,
            "final_equity": round(final_equity, 2), "max_drawdown_pct": round(model_max_drawdown, 2),
            "equity_curve": curve, "note": None, "sanity_warnings": model_sanity_warnings,
        }

    result = {
        "success": True, "ticker": ticker, "horizon_days": horizon_days,
        "starting_capital": STARTING_CAPITAL, "equity_curve": equity_curve,
        "trade_log": trade_log[-200:],  # cap returned trade log size — equity curve already tells the full story
        "stats": stats,
        "n_splits": len(splits), "models_used": list(factories.keys()),
        "relaxed_mode_note": (
            f"RELAXED MODE: only {available_rows} trading days of history were available for {ticker} "
            f"(preferred: {MIN_ROWS_FOR_BACKTEST}+) — this backtest used a scaled-down walk-forward setup "
            f"with fewer splits than usual. Treat these results as lower-confidence than a ticker with full history."
        ) if relaxed_mode else None,
        "per_model_results": per_model_results,
    }

    with db_cursor() as cur:
        import json as _json
        cur.execute(
            "INSERT INTO backtests (created_at, ticker, strategy_json, start_date, end_date, results_json) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), ticker,
             _json.dumps({"horizon_days": horizon_days, "kelly_fraction_multiplier": kelly_fraction_multiplier}),
             equity_curve[0]["date"], equity_curve[-1]["date"], _json.dumps(stats)),
        )

    return result


def _compute_stats(equity_curve: list[dict], trade_log: list[dict], close_prices: pd.Series,
                    first_tested_row_pos: int, last_tested_row_pos: int) -> dict:
    equities = np.array([p["equity"] for p in equity_curve])
    total_return_pct = (equities[-1] / STARTING_CAPITAL - 1) * 100

    running_max = np.maximum.accumulate(equities)
    drawdowns = (equities - running_max) / running_max
    max_drawdown_pct = float(drawdowns.min()) * 100

    n_trades = len(trade_log)
    wins = [t for t in trade_log if t["correct"]]
    win_rate = (len(wins) / n_trades) if n_trades else None

    # Sharpe-like ratio from per-point equity returns (not annualized to
    # a precise trading-day convention — this is a rough, honestly-labeled
    # risk-adjusted-return indicator, not a precise finance-grade Sharpe).
    point_returns = np.diff(equities) / equities[:-1]
    sharpe = (point_returns.mean() / point_returns.std() * np.sqrt(252)) if len(point_returns) > 1 and point_returns.std() > 0 else None

    # Buy-and-hold comparison over the EXACT SAME date range the backtest
    # actually traded — using the real first/last row positions tested,
    # not row 0 of the full feature history. Using the wrong range here
    # was a real bug: it silently compared the strategy's return over
    # ~3.5 tested years against buy-and-hold's return over the full
    # ~5 year history, an apples-to-oranges comparison that made the
    # strategy look far worse (or better) than it actually was relative
    # to simply holding the stock over the SAME period.
    start_price = close_prices.iloc[first_tested_row_pos]
    end_price = close_prices.iloc[last_tested_row_pos]
    buy_and_hold_return_pct = (end_price / start_price - 1) * 100

    # Automated sanity check — the safety net that should have caught
    # the reported +4,032,923% buy-and-hold bug automatically, rather
    # than relying on someone noticing an absurd number. Computed from
    # the REAL calendar span actually tested (not just a row count),
    # since whether a return is unusual depends entirely on how much
    # time it happened over.
    from backend.backtesting.sanity_checks import check_backtest_sanity
    first_date = pd.Timestamp(equity_curve[0]["date"])
    last_date = pd.Timestamp(equity_curve[-1]["date"])
    n_days_tested = max((last_date - first_date).days, 1)

    sanity_warnings = []
    sanity_warnings += check_backtest_sanity(float(total_return_pct), max_drawdown_pct, n_days_tested, label="strategy")
    sanity_warnings += check_backtest_sanity(float(buy_and_hold_return_pct), max_drawdown_pct, n_days_tested, label="buy_and_hold")

    return {
        "total_return_pct": round(total_return_pct, 2),
        "buy_and_hold_return_pct": round(float(buy_and_hold_return_pct), 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(float(sharpe), 2) if sharpe is not None else None,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "n_trades": n_trades,
        "final_equity": round(float(equities[-1]), 2),
        "sanity_warnings": sanity_warnings,
    }
