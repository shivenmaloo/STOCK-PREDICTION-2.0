import pytest

from backend.backtesting.sanity_checks import check_backtest_sanity, implied_annualized_return_pct


def test_catches_the_exact_originally_reported_bug():
    """The actual point of this whole module: the real user-reported
    bug (buy-and-hold showing +4,032,923% over ~851 trading days) must
    be automatically flagged, not silently displayed."""
    warnings = check_backtest_sanity(total_return_pct=4032923, max_drawdown_pct=-30, n_days_tested=851, label="buy_and_hold")
    assert len(warnings) > 0


def test_normal_sane_result_produces_no_warnings():
    warnings = check_backtest_sanity(total_return_pct=180, max_drawdown_pct=-25, n_days_tested=1250, label="test")
    assert warnings == []


def test_positive_drawdown_raises_as_a_bug_not_a_warning():
    """Drawdown is a decline from a peak — it can never be positive.
    A positive value means the code that computed it is broken, so
    this must raise, not just warn."""
    with pytest.raises(ValueError, match="positive"):
        check_backtest_sanity(total_return_pct=10, max_drawdown_pct=5, n_days_tested=500, label="test")


def test_drawdown_below_negative_100_percent_raises():
    """Can't lose more than 100% of equity without leverage/margin —
    a drawdown beyond -100% is a bug, not a market outcome."""
    with pytest.raises(ValueError, match="impossible"):
        check_backtest_sanity(total_return_pct=10, max_drawdown_pct=-150, n_days_tested=500, label="test")


def test_extreme_but_real_outcome_warns_without_blocking():
    """A rare, genuinely possible outcome (a big multi-bagger in a
    short period) should be flagged as unusual, not treated as an
    error — the distinction between 'impossible' and 'extreme' matters."""
    warnings = check_backtest_sanity(total_return_pct=1900, max_drawdown_pct=-40, n_days_tested=365, label="test")
    assert len(warnings) > 0


def test_total_wipeout_does_not_crash():
    """A -100% total return (complete wipeout) is a valid, if extreme,
    real outcome and must not crash the annualization math (which
    would otherwise take a negative base to a fractional power)."""
    warnings = check_backtest_sanity(total_return_pct=-100, max_drawdown_pct=-100, n_days_tested=500, label="test")
    assert isinstance(warnings, list)  # must not raise


def test_zero_days_tested_does_not_crash():
    result = implied_annualized_return_pct(total_return_pct=10, n_days_tested=0)
    assert result is None


def test_annualization_matches_manual_calculation():
    """100% total return over exactly 1 year must annualize to exactly 100%."""
    result = implied_annualized_return_pct(total_return_pct=100, n_days_tested=365)
    assert abs(result - 100) < 1.0  # small tolerance for the 365 vs 365.25 day-count convention


def test_same_total_return_over_longer_period_annualizes_lower():
    """A scale-invariance sanity check on the sanity checker itself:
    the same total return spread over a longer period must imply a
    LOWER annualized rate — otherwise the whole point of annualizing
    (making returns comparable across different time spans) is broken."""
    short = implied_annualized_return_pct(total_return_pct=200, n_days_tested=365)
    long = implied_annualized_return_pct(total_return_pct=200, n_days_tested=365 * 5)
    assert long < short
