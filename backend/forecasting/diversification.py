"""
Correlation-aware diversification for Discovery.

The insight this operationalizes: a well-established quant finance
consensus that strict risk management and low correlation across many
smaller bets matters more than finding one single "best" predictive
signal. Discovery's raw ranking has no idea whether its top 5 picks
are secretly all the same bet (five tech stocks that move together on
the same macro news) — this selects a set that's both individually
strong AND mutually diversifying, so a slightly-lower-scoring but
genuinely uncorrelated pick can beat out a redundant "me too" pick
riding the same underlying factor.

This is NOT full mean-variance portfolio optimization (Markowitz) —
that requires return forecasts and a covariance matrix treated as
ground truth, which is a much heavier, more fragile assumption than
what this app is built to claim. This is the lighter-weight, more
honest version: a simple greedy selection that trades off signal
strength against realized historical correlation, transparent about
exactly what it's trading off at each step.
"""
from __future__ import annotations

import logging

import pandas as pd

from backend.data.market_data_service import market_data_service

logger = logging.getLogger("stock_ai.diversification")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False

DEFAULT_CORRELATION_PENALTY = 0.5
CORRELATION_LOOKBACK_DAYS = 90
MIN_RETURN_POINTS_FOR_CORRELATION = 20


def _fetch_return_series(ticker: str) -> pd.Series | None:
    try:
        hist = market_data_service.get_historical(ticker, period="6mo", interval="1d")
    except Exception:  # noqa: BLE001
        return None
    if not hist.success or hist.data is None or hist.data.empty:
        return None

    df = hist.data.tail(CORRELATION_LOOKBACK_DAYS + 1)
    if len(df) < MIN_RETURN_POINTS_FOR_CORRELATION + 1:
        return None

    returns = df["close"].pct_change().dropna()
    returns.index = pd.to_datetime(df["date"].iloc[1:].values)
    return returns if len(returns) >= MIN_RETURN_POINTS_FOR_CORRELATION else None


def build_correlation_matrix(tickers: list[str]) -> pd.DataFrame:
    """Best-effort — any ticker whose history can't be fetched is
    simply excluded from the matrix rather than failing the whole
    thing; the selection below treats a missing correlation as 0
    (neither penalized nor favored), a reasonable neutral default."""
    series = {}
    for ticker in tickers:
        s = _fetch_return_series(ticker)
        if s is not None:
            series[ticker] = s
    if len(series) < 2:
        return pd.DataFrame()
    return pd.DataFrame(series).corr()


def select_diversified_picks(candidates: list[dict], limit: int = 5,
                              correlation_penalty: float = DEFAULT_CORRELATION_PENALTY,
                              corr_matrix: pd.DataFrame | None = None) -> dict:
    """Greedy selection: always takes the single highest-scoring
    candidate first, then repeatedly adds whichever remaining
    candidate maximizes (normalized_score - correlation_penalty *
    avg_correlation_with_already_selected) — so a strong-but-redundant
    pick can lose out to a slightly weaker, genuinely diversifying one.

    corr_matrix can be injected directly (used by tests to avoid
    needing real network access); production calls fetch it live."""
    if not candidates:
        return {"picks": [], "correlation_data_available": False}

    if corr_matrix is None:
        corr_matrix = build_correlation_matrix([c["ticker"] for c in candidates])

    ranked = sorted(candidates, key=lambda c: abs(c["score"]), reverse=True)
    max_score = abs(ranked[0]["score"]) or 1.0

    selected: list[dict] = []
    remaining = list(ranked)

    first = dict(remaining.pop(0))
    first["avg_correlation_with_selected"] = None
    selected.append(first)

    while remaining and len(selected) < limit:
        best_candidate, best_adjusted, best_avg_corr = None, -float("inf"), None
        for c in remaining:
            normalized_score = abs(c["score"]) / max_score
            corrs = []
            if not corr_matrix.empty and c["ticker"] in corr_matrix.index:
                for s in selected:
                    if s["ticker"] in corr_matrix.columns:
                        val = corr_matrix.loc[c["ticker"], s["ticker"]]
                        if pd.notna(val):
                            corrs.append(val)
            avg_corr = (sum(corrs) / len(corrs)) if corrs else 0.0
            adjusted = normalized_score - correlation_penalty * avg_corr
            if adjusted > best_adjusted:
                best_candidate, best_adjusted, best_avg_corr = c, adjusted, avg_corr

        remaining.remove(best_candidate)
        picked = dict(best_candidate)
        picked["avg_correlation_with_selected"] = round(best_avg_corr, 3)
        selected.append(picked)

    return {"picks": selected, "correlation_data_available": not corr_matrix.empty}
