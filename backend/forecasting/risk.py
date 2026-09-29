"""
Dynamic risk sizing — position sizing and stop-loss suggestions derived
from a model's own walk-forward validation track record, not arbitrary
fixed amounts.

Two techniques, both standard and both documented rather than
black-boxed:

  1. Fractional Kelly Criterion: uses the win rate and average win/loss
     size from the model's actual out-of-sample walk-forward results
     (backend/forecasting/validation.trading_stats_from_walk_forward)
     to size a position. Full Kelly is famously aggressive and
     over-sized for real use, so this applies a conservative fraction
     (25% of full Kelly by default) and hard-caps the suggestion.

  2. ATR-based stop distance: uses the ticker's own recent Average True
     Range (already computed in backend/indicators/technical.py) to
     size a stop-loss distance that's proportional to how volatile the
     stock actually is, rather than an arbitrary fixed percentage.

This produces a SUGGESTION for the user to evaluate, never an
instruction, and every number shown traces back to real data — no
fabricated risk assumptions.
"""
from __future__ import annotations

DEFAULT_KELLY_FRACTION = 0.25  # apply 25% of full Kelly — standard conservative practice
MAX_POSITION_PCT = 20.0  # hard cap regardless of what Kelly suggests
MIN_TRADES_FOR_KELLY = 20  # below this, the win-rate/avg-win-loss stats aren't trustworthy

# Risk-profile personalization: this changes how much of the model's
# statistically-derived edge you're willing to size into, NOT what the
# model predicts. That distinction matters — the prediction itself
# should never bend to suit what a user wants to hear, but how much of
# your own capital you'd risk on a given edge is a legitimate personal
# choice, the same way two investors can look at the same opportunity
# and reasonably size it differently.
RISK_PROFILES = {
    "conservative": 0.15,
    "moderate": DEFAULT_KELLY_FRACTION,
    "aggressive": 0.40,
}


def kelly_fraction(win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> float | None:
    """Standard Kelly formula for a binary bet with asymmetric payoff:
    f* = W - (1-W) / (avg_win/avg_loss)
    Returns None if the inputs don't support a meaningful calculation
    (e.g. no losing trades to measure, so the ratio is undefined)."""
    if avg_loss_pct is None or avg_loss_pct <= 0 or avg_win_pct is None or avg_win_pct <= 0:
        return None
    payoff_ratio = avg_win_pct / avg_loss_pct
    f_star = win_rate - (1 - win_rate) / payoff_ratio
    return f_star


def suggest_position_size(trading_stats: dict, kelly_fraction_multiplier: float = DEFAULT_KELLY_FRACTION) -> dict:
    """trading_stats: output of validation.trading_stats_from_walk_forward()
    — {win_rate, avg_win_pct, avg_loss_pct, n_trades}."""
    n_trades = trading_stats.get("n_trades", 0)
    win_rate = trading_stats.get("win_rate")
    avg_win = trading_stats.get("avg_win_pct")
    avg_loss = trading_stats.get("avg_loss_pct")

    if n_trades < MIN_TRADES_FOR_KELLY or win_rate is None:
        return {
            "suggested_position_pct": None,
            "full_kelly_pct": None,
            "explanation": (
                f"Only {n_trades} out-of-sample trade(s) in the walk-forward validation history — "
                f"need at least {MIN_TRADES_FOR_KELLY} before a position-sizing suggestion is "
                f"statistically meaningful. Showing no suggestion rather than guessing."
            ),
        }

    f_star = kelly_fraction(win_rate, avg_win, avg_loss)
    if f_star is None:
        # Distinct from "the edge doesn't clear the bar" (a weak but
        # measurable edge) — this is specifically "we can't compute a
        # meaningful win/loss ratio at all," which happens only when
        # every trade landed on the same side (all wins or all losses).
        # Conflating the two would be misleading: an all-win record
        # isn't a weak edge, it's a suspiciously perfect one worth
        # scrutinizing on its own terms, and an all-loss record is a
        # genuinely alarming result, not a marginal one.
        if avg_loss is None and avg_win is not None:
            reason = (f"Every one of the {n_trades} out-of-sample trades was a win — there's no losing-trade "
                      f"data to size a position against, since Kelly sizing needs to weigh both sides. "
                      f"A perfect track record over real out-of-sample data is unusual enough to double-check "
                      f"rather than act on directly.")
        elif avg_win is None and avg_loss is not None:
            reason = (f"Every one of the {n_trades} out-of-sample trades was a loss — this model has no "
                      f"validated edge on this ticker/horizon at all; sizing a position from this record "
                      f"isn't just weak, it's actively discouraged by its own history.")
        else:
            reason = "Not enough win/loss variation in the validation history to compute a meaningful position size."
        return {"suggested_position_pct": None, "full_kelly_pct": None, "explanation": reason}
    if f_star <= 0:
        return {
            "suggested_position_pct": 0.0,
            "full_kelly_pct": round(f_star * 100, 1) if f_star is not None else None,
            "explanation": (
                f"Win rate ({win_rate * 100:.0f}%) and win/loss size don't support a positive "
                f"Kelly-sized position based on this model's validated track record — "
                f"the edge, if any, doesn't clear the bar for a sizing recommendation."
            ),
        }

    fractional_kelly_pct = f_star * kelly_fraction_multiplier * 100
    capped_pct = round(min(fractional_kelly_pct, MAX_POSITION_PCT), 1)

    return {
        "suggested_position_pct": capped_pct,
        "full_kelly_pct": round(f_star * 100, 1),
        "kelly_fraction_used": kelly_fraction_multiplier,
        "based_on_n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "explanation": (
            f"Based on {n_trades} out-of-sample walk-forward trades: {win_rate * 100:.0f}% win rate, "
            f"average win {avg_win:.2f}%, average loss {avg_loss:.2f}%. Full Kelly suggests "
            f"{f_star * 100:.1f}% of capital; using {kelly_fraction_multiplier * 100:.0f}% of that "
            f"(fractional Kelly, standard practice since full Kelly is historically too aggressive "
            f"for real use) gives {capped_pct}%, capped at {MAX_POSITION_PCT}% regardless."
        ),
    }


def suggest_atr_stop(current_price: float, atr: float, direction: str, atr_multiplier: float = 2.0) -> dict:
    """A stop-loss distance proportional to the ticker's own recent
    volatility (ATR) rather than an arbitrary fixed percentage — a
    stock that moves 5% a day needs a wider stop than one that moves
    0.5% a day, or a fixed-percent stop gets hit by normal noise."""
    if current_price is None or atr is None or current_price <= 0:
        return {"stop_price": None, "explanation": "Not enough price/ATR data to suggest a stop."}

    stop_distance = atr * atr_multiplier
    if direction == "bullish":
        stop_price = current_price - stop_distance
    elif direction == "bearish":
        stop_price = current_price + stop_distance
    else:
        return {"stop_price": None, "explanation": "No clear direction to base a stop on."}

    stop_pct = (stop_distance / current_price) * 100
    return {
        "stop_price": round(stop_price, 2),
        "stop_distance_pct": round(stop_pct, 2),
        "atr_multiplier": atr_multiplier,
        "explanation": (
            f"{atr_multiplier}x the current ATR (${atr:.2f}) from the current price "
            f"(${current_price:.2f}), giving room for normal volatility while still limiting "
            f"downside to roughly {stop_pct:.1f}% if hit."
        ),
    }
