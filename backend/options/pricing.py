"""
Black-Scholes pricing and implied volatility solving — the
mathematical core of the volatility surface module.

Deliberately scoped to what actually informs stock-direction
prediction (ATM IV, put/call skew, IV rank) rather than a full
options-trading toolkit. No Greeks, no SVI/SABR parametric fitting —
those serve pricing/hedging an option itself, not predicting where the
underlying stock is headed, which is the one thing this whole app is
built to do.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

IV_MIN = 0.001
IV_MAX = 5.0
MIN_TIME_TO_EXPIRY_DAYS = 7  # excludes near-dated options — high-gamma noise, not a stable signal


def black_scholes_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Standard Black-Scholes European call price.
    S: spot price, K: strike, T: time to expiry in years,
    r: risk-free rate (annualized), sigma: volatility (annualized)."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)  # a European call at/after expiry (or zero vol) is worth exactly its intrinsic value
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def solve_implied_volatility(market_price: float, S: float, K: float, T: float, r: float) -> float | None:
    """Solves for the sigma that makes black_scholes_call_price match
    the observed market price, using Brent's method (scipy.brentq) —
    a bracketing root-finder, not Newton-Raphson, specifically because
    it can't diverge or fail to converge the way a gradient-based
    method can on a poorly-scaled starting guess; it just needs the
    root to be bracketed within [IV_MIN, IV_MAX], which it always is
    for any real, arbitrage-free market price.

    Returns None (not zero, not a fabricated number) if the price
    itself is outside what ANY volatility in [IV_MIN, IV_MAX] could
    produce, or if brentq still fails to converge for any other
    reason — this is a real "give up honestly" case, not something to
    paper over."""
    if T <= 0 or S <= 0 or K <= 0 or market_price <= 0:
        return None

    def price_diff(sigma: float) -> float:
        return black_scholes_call_price(S, K, T, r, sigma) - market_price

    try:
        low, high = price_diff(IV_MIN), price_diff(IV_MAX)
        if low * high > 0:
            # The market price isn't bracketed by any volatility in
            # our allowed range at all — no real root to find here,
            # not a convergence failure to retry.
            return None
        return brentq(price_diff, IV_MIN, IV_MAX, xtol=1e-6)
    except (ValueError, RuntimeError):
        return None
