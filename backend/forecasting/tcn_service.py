"""
Orchestrates the TCN + risk-map: fetches data, reuses the app's
existing stationary feature set (see the inspection note in
tcn_model.py about why raw OHLCV was replaced with it), runs walk-
forward validation using TCNModel, and produces either a search-period
score (for tcn_search.py) or a full 5-way backtest comparison (for the
live UI).

Everything here requires PyTorch — if it isn't installed, every public
function returns a clear "unavailable" result rather than raising an
unhandled exception, the same graceful-degradation pattern already
used for the LSTM everywhere else in this app.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backend.data.market_data_service import market_data_service
from backend.indicators.technical import compute_all_indicators
from backend.forecasting.features import build_features, build_training_matrix
from backend.forecasting.validation import walk_forward_splits
from backend.forecasting.tcn_model import TCNModel, TORCH_AVAILABLE
from backend.forecasting.risk_map import RiskMapConfig
from backend.forecasting.tcn_search import TCNSearchConfig, split_search_and_holdout
from backend.backtesting.tcn_comparison import run_five_way_comparison, score_config_for_search

logger = logging.getLogger("stock_ai.tcn_service")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False

MIN_ROWS_FOR_TCN = 400  # enough history for a 64-bar window AND multiple walk-forward splits
TCN_WF_MIN_TRAIN = 200
TCN_WF_STEP = 21
MIN_TRADES_TO_TRUST = 20


def is_available() -> bool:
    return TORCH_AVAILABLE


def _prepare_features(ticker: str, horizon_days: int):
    """Fetches data and builds (X, y_return, close_prices) using the
    app's existing, already-tested stationary feature set. Returns
    None if there isn't enough history — never raises for an ordinary
    data-availability reason.

    close_prices comes from build_training_matrix's extra_columns
    parameter, which guarantees correct row-for-row alignment with X
    by construction (reusing the exact same dropna mask internally) —
    not a positional slice, which was a real, confirmed bug (found
    independently in both this module and the main backtest engine):
    it silently shifts every returned value forward by horizon_days
    rows. Needed for an honest buy-and-hold comparison — see the bug
    note on run_comparison_for_ticker below about why y_return itself
    must never be used for that."""
    hist_result = market_data_service.get_historical(ticker, period="5y", interval="1d")
    if not hist_result.success or hist_result.data is None or len(hist_result.data) < MIN_ROWS_FOR_TCN:
        return None

    indicators_df = compute_all_indicators(hist_result.data)
    features_df = build_features(indicators_df)
    X, y_return, _, aligned = build_training_matrix(features_df, horizon_days, extra_columns=["close"])
    if len(X) < MIN_ROWS_FOR_TCN:
        return None

    close_prices = aligned["close"]
    return X, y_return, close_prices


def _run_walk_forward_tcn(X: pd.DataFrame, y_return: pd.Series, config: TCNSearchConfig,
                           min_train: int, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Trains a fresh TCN on each expanding window (same discipline as
    every other model in this app — see validation.run_walk_forward),
    collecting out-of-sample probabilities/returns/volatility across
    every split. Returns (probabilities, actual_returns, volatilities,
    n_splits). A split where training fails (e.g. not enough rows for
    this particular window_len) is skipped, not fatal to the whole run."""
    splits = walk_forward_splits(len(X), min_train, step)
    all_probs, all_returns, all_vols = [], [], []

    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[train_idx], y_return.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y_return.iloc[test_idx]

        model = TCNModel(window_len=config.window_len, num_layers=config.num_layers,
                          kernel_size=config.kernel_size, hidden_channels=config.hidden_channels,
                          dropout=config.dropout)
        try:
            model.fit(X_train.values, y_train.values)
        except ValueError as exc:
            logger.info(f"[TCN] Split skipped (not enough rows for window_len={config.window_len}): {exc}")
            continue

        # Real bug found via testing: predict_proba()'s windowing only
        # looks backward WITHIN the array it's handed. X_test alone is
        # only `step` rows (e.g. 21) — far short of window_len (e.g.
        # 64) — so almost every prediction in the test block would
        # silently default to 0.5 (the documented "not enough history"
        # fallback), the same staleness/insufficient-context bug
        # already found and fixed for the LSTM's live prediction path.
        # Fix: hand predict_proba() the tail of the TRAINING data
        # immediately preceding the test block too, so even the FIRST
        # test row has a genuine window_len of real history — then
        # keep only the predictions for the actual test rows.
        context_start = max(0, test_idx[0] - config.window_len)
        X_context_and_test = X.iloc[context_start:test_idx[-1] + 1]
        probs_full = model.predict_proba(X_context_and_test.values)
        probs = probs_full[-len(test_idx):]

        all_probs.extend(probs.tolist())
        all_returns.extend(y_test.tolist())
        all_vols.extend(X_test["atr_pct_14"].tolist())

    return np.array(all_probs), np.array(all_returns), np.array(all_vols), len(splits)


def evaluate_config_on_search_period(ticker: str, horizon_days: int, config: TCNSearchConfig) -> dict | None:
    """Used by the ongoing search (tcn_search.py) to score one
    candidate config — restricted to the SEARCH portion of history
    only (see split_search_and_holdout), never the holdout tail."""
    if not TORCH_AVAILABLE:
        return None
    prepared = _prepare_features(ticker, horizon_days)
    if prepared is None:
        return None
    X, y_return, _ = prepared

    search_end, _ = split_search_and_holdout(len(X))
    X_search, y_search = X.iloc[:search_end], y_return.iloc[:search_end]

    probs, returns, vols, n_splits = _run_walk_forward_tcn(X_search, y_search, config, TCN_WF_MIN_TRAIN, TCN_WF_STEP)
    if len(probs) < MIN_TRADES_TO_TRUST:
        return None

    risk_config = RiskMapConfig(dead_zone_threshold=config.dead_zone_threshold,
                                 target_volatility=config.target_volatility,
                                 max_position_pct=config.max_position_pct)
    score = score_config_for_search(probs, returns, vols, risk_config)
    return {"score": score, "n_splits": n_splits, "n_trades": len(probs)}


def evaluate_best_config_on_holdout(ticker: str, horizon_days: int, config: TCNSearchConfig) -> dict | None:
    """The one-time holdout check — same walk-forward discipline,
    restricted to ONLY the holdout tail (which the search process
    above never touches)."""
    if not TORCH_AVAILABLE:
        return None
    prepared = _prepare_features(ticker, horizon_days)
    if prepared is None:
        return None
    X, y_return, _ = prepared

    _, holdout_start = split_search_and_holdout(len(X))
    X_holdout, y_holdout = X.iloc[holdout_start:], y_return.iloc[holdout_start:]
    if len(X_holdout) < TCN_WF_MIN_TRAIN + TCN_WF_STEP:
        return None

    # The holdout period needs its OWN small walk-forward pass (still
    # chronological, still never touching the search period) since a
    # single train/test split would waste most of a short holdout tail.
    probs, returns, vols, n_splits = _run_walk_forward_tcn(X_holdout, y_holdout, config,
                                                            min_train=len(X_holdout) // 2, step=max(TCN_WF_STEP // 2, 10))
    if len(probs) < MIN_TRADES_TO_TRUST:
        return None

    risk_config = RiskMapConfig(dead_zone_threshold=config.dead_zone_threshold,
                                 target_volatility=config.target_volatility,
                                 max_position_pct=config.max_position_pct)
    score = score_config_for_search(probs, returns, vols, risk_config)
    return {"score": score, "n_splits": n_splits, "n_trades": len(probs)}


def run_comparison_for_ticker(ticker: str, horizon_days: int, config: TCNSearchConfig | None = None) -> dict:
    """The live, UI-facing 5-way comparison — runs over the FULL
    available history (not restricted to search/holdout, since this is
    a research display, not part of the ongoing search's own scoring)."""
    if not TORCH_AVAILABLE:
        return {"success": False, "error": "PyTorch is not installed — the TCN module is unavailable."}

    from backend.forecasting.tcn_search import get_best_config
    if config is None:
        best = get_best_config(ticker, horizon_days)
        config = best["config"] if best else TCNSearchConfig()

    prepared = _prepare_features(ticker, horizon_days)
    if prepared is None:
        return {"success": False, "error": f"Not enough historical data for {ticker} to run the TCN."}
    X, y_return, close_prices = prepared

    probs, returns, vols, n_splits = _run_walk_forward_tcn(X, y_return, config, TCN_WF_MIN_TRAIN, TCN_WF_STEP)
    if len(probs) < MIN_TRADES_TO_TRUST:
        return {"success": False, "error": "Not enough validated out-of-sample predictions to run a meaningful comparison."}

    # Real bug found via testing: y_return is an OVERLAPPING multi-day
    # forward return (row i covers days [i, i+horizon), row i+1 covers
    # [i+1, i+horizon+1) — heavily overlapping with row i). Naively
    # compounding these values day-by-day as if they were sequential,
    # non-overlapping daily returns multi-counts the same underlying
    # price moves many times over, producing an absurd, meaningless
    # compounding explosion — exactly what a user reported seeing
    # (buy-and-hold showing +4,032,923%). The correct, honest
    # buy-and-hold comparison is a simple start-to-end PRICE RATIO over
    # the exact same range that was actually tested — the same
    # approach already proven correct in the main backtest engine
    # (backend/backtesting/service.py), just re-derived here since
    # this module computes its own walk-forward splits independently.
    splits = walk_forward_splits(len(X), TCN_WF_MIN_TRAIN, TCN_WF_STEP)
    first_tested_row = splits[0][1][0]
    last_tested_row = splits[-1][1][-1]
    start_price = float(close_prices.iloc[first_tested_row])
    end_price = float(close_prices.iloc[last_tested_row])
    buy_and_hold_return_pct = round((end_price / start_price - 1) * 100, 3)

    risk_config = RiskMapConfig(dead_zone_threshold=config.dead_zone_threshold,
                                 target_volatility=config.target_volatility,
                                 max_position_pct=config.max_position_pct)
    comparison = run_five_way_comparison(probs, returns, vols, risk_config, buy_and_hold_return_pct=buy_and_hold_return_pct)

    return {
        "success": True, "ticker": ticker.upper(), "horizon_days": horizon_days,
        "config": config.__dict__, "n_splits": n_splits, "n_predictions": len(probs),
        "comparison": comparison,
    }
