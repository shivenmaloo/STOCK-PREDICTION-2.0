"""
Runs the TCN + risk-map through the same walk-forward discipline as
the rest of the app's backtesting, and produces the 5-way comparison:
buy-and-hold vs raw model signal vs +dead-zone vs +vol-scaling vs
+vol-scaling+exposure-cap.

Also used by backend/forecasting/tcn_search.py to score candidate
configs during the search-period evaluation — the same simulation
logic either way, so a search-period score means exactly what a
comparison-page score means.
"""
from __future__ import annotations

import numpy as np

from backend.forecasting.risk_map import RiskMapConfig, process_signal

ROUND_TRIP_TRANSACTION_COST_PCT = 0.001  # matches the main backtest engine's assumption, for a fair comparison

COMPARISON_VARIANTS = {
    "buy_and_hold": None,  # handled separately — not signal-driven at all
    "raw_model_signal": {"transform"},
    "model_plus_dead_zone": {"transform", "dead_zone"},
    "model_plus_vol_scaling": {"transform", "dead_zone", "vol_scaling"},
    "model_plus_vol_scaling_and_cap": {"transform", "dead_zone", "vol_scaling", "exposure_cap"},
}


def simulate_positions(probabilities: np.ndarray, actual_returns: np.ndarray, volatilities: np.ndarray,
                        config: RiskMapConfig, stages: set[str] | None) -> dict:
    """Applies process_signal() row by row — each row uses only its
    OWN probability/volatility/actual_return, nothing from any other
    row, so this is trivially leak-free by construction (there's
    nothing here that could reach across rows even if it tried).
    Simulates the resulting equity curve with the same transaction
    cost assumption as the main backtest engine, so results are
    directly comparable."""
    equity = 1.0
    equity_curve = [round(equity, 6)]
    trade_returns = []
    max_abs_position = 0.0

    for p, ret, vol in zip(probabilities, actual_returns, volatilities):
        result = process_signal(float(p), float(vol) if vol is not None else None, config, stages=stages)
        position = result["final_position"]
        max_abs_position = max(max_abs_position, abs(position))
        if abs(position) > 1e-9:
            gross = position * ret
            cost = abs(position) * ROUND_TRIP_TRANSACTION_COST_PCT
            net = gross - cost
            equity *= (1 + net)
            trade_returns.append(net)
        equity_curve.append(round(equity, 6))

    n_trades = len(trade_returns)
    win_rate = (sum(1 for r in trade_returns if r > 0) / n_trades) if n_trades else None
    total_return_pct = (equity - 1) * 100

    running_max = np.maximum.accumulate(np.array(equity_curve))
    drawdowns = (np.array(equity_curve) - running_max) / running_max
    max_drawdown_pct = float(drawdowns.min()) * 100

    point_returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe = (point_returns.mean() / point_returns.std() * np.sqrt(252)) if len(point_returns) > 1 and point_returns.std() > 0 else None

    return {
        "total_return_pct": round(total_return_pct, 3),
        "max_drawdown_pct": round(max_drawdown_pct, 3),
        "sharpe_ratio": round(float(sharpe), 3) if sharpe is not None else None,
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "avg_trade_return_pct": round(float(np.mean(trade_returns)) * 100, 4) if trade_returns else None,
        "max_abs_exposure_pct": round(max_abs_position * 100, 2),
        "equity_curve": equity_curve,
    }


def simulate_buy_and_hold(actual_returns: np.ndarray) -> dict:
    equity = 1.0
    equity_curve = [round(equity, 6)]
    for ret in actual_returns:
        equity *= (1 + ret)
        equity_curve.append(round(equity, 6))
    running_max = np.maximum.accumulate(np.array(equity_curve))
    drawdowns = (np.array(equity_curve) - running_max) / running_max
    return {
        "total_return_pct": round((equity - 1) * 100, 3),
        "max_drawdown_pct": round(float(drawdowns.min()) * 100, 3),
        "sharpe_ratio": None,  # buy-and-hold isn't a "trades" series in the same sense — not meaningfully comparable here
        "n_trades": None,
        "win_rate": None,
        "avg_trade_return_pct": None,
        "max_abs_exposure_pct": 100.0,
        "equity_curve": equity_curve,
    }


def run_five_way_comparison(probabilities: np.ndarray, actual_returns: np.ndarray, volatilities: np.ndarray,
                             config: RiskMapConfig | None = None, buy_and_hold_return_pct: float | None = None) -> dict:
    """The actual comparison the spec asked for. Every variant after
    buy-and-hold reuses the exact same probabilities/returns/volatility
    series — the only thing that changes between them is which risk-map
    stages are active, so any difference in the results comes purely
    from the risk management, not from a different underlying model
    run for each one.

    buy_and_hold_return_pct should be a properly pre-computed,
    start-to-end PRICE RATIO over the exact tested date range (see
    backend/forecasting/tcn_service.py for why) — passing it directly
    avoids a real bug found via testing: actual_returns here is an
    OVERLAPPING multi-day forward return (not a sequential daily one),
    and naively compounding it day-by-day the way simulate_buy_and_hold
    does for genuinely sequential returns multi-counts the same price
    moves many times over, producing an absurd result. If not provided
    (e.g. when called with genuinely sequential returns from some other
    context), falls back to the original compounding approach."""
    config = config or RiskMapConfig()
    if buy_and_hold_return_pct is not None:
        results = {"buy_and_hold": {
            "total_return_pct": buy_and_hold_return_pct,
            "max_drawdown_pct": None, "sharpe_ratio": None, "n_trades": None,
            "win_rate": None, "avg_trade_return_pct": None, "max_abs_exposure_pct": 100.0,
            "equity_curve": None,
        }}
    else:
        results = {"buy_and_hold": simulate_buy_and_hold(actual_returns)}
    for name, stages in COMPARISON_VARIANTS.items():
        if stages is None:
            continue
        results[name] = simulate_positions(probabilities, actual_returns, volatilities, config, stages)

    # Automated sanity check — the same safety net now wired into the
    # main backtest engine, applied here too since this module had the
    # exact same underlying bug (overlapping-return compounding) before
    # being fixed. Approximates calendar days from trading-day count
    # (len(actual_returns) is trading days) since this module doesn't
    # carry real dates the way the main backtest engine does.
    from backend.backtesting.sanity_checks import check_backtest_sanity
    n_days_tested = max(int(len(actual_returns) * 365.25 / 252), 1)
    for name, result in results.items():
        if result.get("max_drawdown_pct") is None:
            continue  # buy-and-hold's precomputed path doesn't carry a drawdown to check
        result["sanity_warnings"] = check_backtest_sanity(
            result["total_return_pct"], result["max_drawdown_pct"], n_days_tested, label=name)

    return results


def score_config_for_search(probabilities: np.ndarray, actual_returns: np.ndarray, volatilities: np.ndarray,
                             config: RiskMapConfig) -> float:
    """The single scalar score used to rank candidate configs during
    the ongoing search (backend/forecasting/tcn_search.py) — the full
    risk-map pipeline's total return over the search period. Uses the
    FULL pipeline (not just raw signal) since the search is explicitly
    searching over risk-map parameters too, not just model architecture."""
    result = simulate_positions(probabilities, actual_returns, volatilities, config, stages=None)
    return result["total_return_pct"]
