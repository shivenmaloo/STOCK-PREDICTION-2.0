"""
Automated sanity checks for backtest-style results — the safety net
that should have caught the reported +4,032,923% buy-and-hold bug
automatically, rather than relying on a person noticing an absurd
number. Used by both backend/backtesting/service.py and
backend/backtesting/tcn_comparison.py, since both independently
produce the same kind of equity-curve/return statistics and both
had (unrelated) bugs that could have produced impossible output.

Two tiers, deliberately different:
  - HARD invariants (always true for any correct equity simulation,
    e.g. drawdown can't be positive) raise ValueError — something is
    definitely broken in the code, not just an unusual market outcome.
  - SOFT flags (extreme but not strictly impossible, e.g. a very high
    annualized return) are returned as warnings, not raised — a real
    stock genuinely can 10x in a year, so this doesn't block the
    result, it just surfaces "this is unusual, double-check it"
    instead of silently presenting an extreme number as routine.
"""
from __future__ import annotations

# An annualized return beyond this is not literally impossible (a real
# stock can multi-bag in a year), but it's rare enough that a result
# this extreme deserves a visible flag rather than silent display —
# exactly the category the reported bug fell into.
EXTREME_ANNUALIZED_RETURN_PCT = 500.0
# Exactly -100% (total wipeout) is a real, valid market outcome — a
# stock genuinely can go to zero. Only strictly BELOW -100% is a
# mathematical impossibility (can't lose more than everything).


def implied_annualized_return_pct(total_return_pct: float, n_days_tested: int) -> float | None:
    """Converts a total return over the tested period into an
    annualized rate — the correct, scale-invariant way to judge
    whether a return is unusual, since a total return that's fine over
    10 years would be extraordinary over 3 months."""
    if n_days_tested <= 0:
        return None
    years = n_days_tested / 365.25
    if years <= 0:
        return None
    growth_factor = 1 + total_return_pct / 100
    if growth_factor <= 0:
        return -100.0  # total wipeout or worse — already at the floor
    return (growth_factor ** (1 / years) - 1) * 100


def check_backtest_sanity(total_return_pct: float, max_drawdown_pct: float, n_days_tested: int,
                           label: str = "result") -> list[str]:
    """Returns a list of human-readable warnings for anything
    suspicious. Raises ValueError for anything that's a mathematical
    impossibility, not just an unusual outcome — those indicate a real
    bug in the code that produced the number, not a surprising but
    valid market result."""
    warnings: list[str] = []

    if max_drawdown_pct > 0:
        raise ValueError(f"{label}: max_drawdown_pct={max_drawdown_pct} is positive — "
                          f"drawdown is a decline from a peak and can never be positive. This is a bug, not a market outcome.")
    if max_drawdown_pct < -100:
        raise ValueError(f"{label}: max_drawdown_pct={max_drawdown_pct} implies losing more than 100% of equity, "
                          f"which is impossible in a long-only simulation without leverage. This is a bug, not a market outcome.")

    annualized = implied_annualized_return_pct(total_return_pct, n_days_tested)
    if annualized is not None:
        if annualized > EXTREME_ANNUALIZED_RETURN_PCT:
            warnings.append(
                f"{label}: implies an annualized return of {annualized:.0f}% — extremely high. "
                f"Real, but rare; worth double-checking before treating this as routine."
            )
        elif annualized < -100:
            raise ValueError(f"{label}: implied annualized return of {annualized:.1f}% is below -100%, "
                              f"which is impossible (can't lose more than all of the equity). This is a bug, not a market outcome.")

    return warnings
