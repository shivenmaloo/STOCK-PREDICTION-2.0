"""
Options chain preprocessing, quality filtering, and 2D surface
interpolation — turns a raw, noisy options chain into a clean,
smoothed (time-to-expiry, moneyness, IV) surface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from backend.options.pricing import solve_implied_volatility, MIN_TIME_TO_EXPIRY_DAYS

MIN_VOLUME = 1
MIN_OPEN_INTEREST = 5
MAX_BID_ASK_SPREAD_PCT = 0.25  # spread wider than 25% of mid-price is too illiquid to trust
DAYS_PER_YEAR = 365.25

STANDARD_MONEYNESS_POINTS = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
STANDARD_TENOR_DAYS = [30, 60, 90, 180]


def preprocess_chain(raw_chain: pd.DataFrame, spot_price: float, expiry_days: int,
                      risk_free_rate: float = 0.045) -> pd.DataFrame:
    """Takes one expiry's raw call-options DataFrame (matching
    yfinance's option_chain().calls column layout: strike, bid, ask,
    lastPrice, volume, openInterest) plus that expiry's days-to-expiry,
    and returns a clean DataFrame with columns [strike, moneyness,
    time_to_expiry_years, mid_price, implied_vol] — every quality/
    arbitrage-violating/non-convergent row already dropped, not
    flagged for later filtering."""
    if expiry_days < MIN_TIME_TO_EXPIRY_DAYS:
        return pd.DataFrame(columns=["strike", "moneyness", "time_to_expiry_years", "mid_price", "implied_vol"])

    df = raw_chain.copy()

    # Liquidity/quality filters — applied before any pricing math, so
    # a genuinely illiquid quote never even reaches the IV solver.
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df[df["volume"].fillna(0) >= MIN_VOLUME]
    df = df[df["openInterest"].fillna(0) >= MIN_OPEN_INTEREST]

    mid_price = (df["bid"] + df["ask"]) / 2
    spread_pct = (df["ask"] - df["bid"]) / mid_price
    df = df[spread_pct <= MAX_BID_ASK_SPREAD_PCT]

    if df.empty:
        return pd.DataFrame(columns=["strike", "moneyness", "time_to_expiry_years", "mid_price", "implied_vol"])

    mid_price = (df["bid"] + df["ask"]) / 2
    intrinsic_value = np.maximum(0, spot_price - df["strike"])
    df = df[mid_price >= intrinsic_value]  # arbitrage filter: price must be at least intrinsic value

    if df.empty:
        return pd.DataFrame(columns=["strike", "moneyness", "time_to_expiry_years", "mid_price", "implied_vol"])

    T = expiry_days / DAYS_PER_YEAR
    rows = []
    for _, row in df.iterrows():
        mid = (row["bid"] + row["ask"]) / 2
        iv = solve_implied_volatility(mid, spot_price, row["strike"], T, risk_free_rate)
        if iv is None:
            continue  # non-convergent — dropped honestly, not fabricated
        rows.append({
            "strike": row["strike"],
            "moneyness": row["strike"] / spot_price,
            "time_to_expiry_years": T,
            "time_to_expiry_days": expiry_days,
            "mid_price": mid,
            "implied_vol": iv,
        })

    return pd.DataFrame(rows)


def interpolate_surface(points_df: pd.DataFrame, moneyness_grid: np.ndarray, tenor_grid_days: np.ndarray) -> np.ndarray | None:
    """Interpolates sparse (time_to_expiry_days, moneyness, implied_vol)
    scatter points onto a uniform grid via cubic griddata, falling back
    to linear interpolation for any points cubic leaves as NaN (typical
    at the edges of the observed data, where cubic has no valid
    simplex to interpolate within). Returns None if there isn't enough
    real data to interpolate anything at all.

    Real options chains can be genuinely degenerate — e.g. every
    surviving contract sharing (or nearly sharing) the same time to
    expiry, which makes the underlying Delaunay triangulation
    mathematically undefined in 2D. scipy raises a hard QhullError in
    that case rather than returning NaN, which would otherwise crash
    the entire request over what is, from the user's perspective, just
    a sparse/unlucky chain. Caught and degraded gracefully: try linear
    alone (a much less demanding triangulation), and if even that
    fails, report None honestly rather than crash."""
    if len(points_df) < 4:  # griddata needs at least a few real points to interpolate from at all
        return None

    grid_t, grid_m = np.meshgrid(tenor_grid_days, moneyness_grid)
    points = points_df[["time_to_expiry_days", "moneyness"]].values
    values = points_df["implied_vol"].values

    try:
        cubic = griddata(points, values, (grid_t, grid_m), method="cubic")
    except Exception:  # noqa: BLE001 — QhullError on degenerate/collinear input, or any other triangulation failure
        cubic = np.full(grid_t.shape, np.nan)

    try:
        linear = griddata(points, values, (grid_t, grid_m), method="linear")
    except Exception:  # noqa: BLE001
        linear = np.full(grid_t.shape, np.nan)

    result = np.where(np.isnan(cubic), linear, cubic)
    if np.all(np.isnan(result)):
        return None
    return result


def get_surface_features(points_df: pd.DataFrame, spot_price: float) -> dict | None:
    """Samples the interpolated surface at the standardized moneyness/
    tenor grid used for ML feature ingestion — a flattened 28-value
    vector (7 moneyness points x 4 tenors), plus the headline scalars:
    ATM IV and the 30-day put/call-style skew (IV at 0.95 moneyness
    minus IV at 1.05 moneyness — a real, established options-market
    sentiment signal). Returns None if there wasn't enough real data
    to interpolate a meaningful surface at all.

    Real-world options chains are often too sparse or unevenly
    distributed for cubic (or even linear) interpolation to reach
    every standard grid point cleanly — a less-liquid name might have
    plenty of near-dated contracts but almost nothing exactly at 30
    days, especially away from the money. Rather than silently
    reporting these as unavailable when perfectly good NEARBY real
    data exists, the headline stats (ATM IV, skew) fall back to the
    closest ACTUAL observed contract when the exact grid point is NaN
    — or even when the full grid interpolation fails outright (e.g.
    genuinely degenerate/collinear data) — transparently, with the
    real tenor/moneyness actually used reported alongside it. The full
    feature_vector used for ML ingestion is NOT adjusted this way —
    consistency across tickers at fixed grid points matters more
    there, and NaN is a legitimate, handleable value for a model to
    see; it comes back as all-None when the grid itself couldn't be
    built, rather than fabricated from nearest-neighbor values."""
    if points_df.empty:
        return None

    surface = interpolate_surface(points_df, np.array(STANDARD_MONEYNESS_POINTS), np.array(STANDARD_TENOR_DAYS))
    grid_available = surface is not None
    feature_vector = (surface.flatten(order="F").tolist() if grid_available
                       else [np.nan] * (len(STANDARD_MONEYNESS_POINTS) * len(STANDARD_TENOR_DAYS)))

    def _sample_with_fallback(moneyness: float, tenor_days: int) -> dict:
        if grid_available:
            m_idx = STANDARD_MONEYNESS_POINTS.index(moneyness)
            t_idx = STANDARD_TENOR_DAYS.index(tenor_days)
            val = surface[m_idx, t_idx]
            if not np.isnan(val):
                return {"value": round(float(val), 4), "used_fallback": False}

        nearest = _nearest_observed_contract(points_df, moneyness, tenor_days)
        if nearest is None:
            return {"value": None, "used_fallback": False}
        return {"value": round(nearest["implied_vol"], 4), "used_fallback": True,
                "actual_moneyness": nearest["actual_moneyness"], "actual_tenor_days": nearest["actual_tenor_days"]}

    atm = _sample_with_fallback(1.00, 30)
    iv_95 = _sample_with_fallback(0.95, 30)
    iv_105 = _sample_with_fallback(1.05, 30)

    skew = None
    if iv_95["value"] is not None and iv_105["value"] is not None:
        skew = round(iv_95["value"] - iv_105["value"], 4)

    data_quality_notes = []
    if atm["used_fallback"] and atm["value"] is not None:
        data_quality_notes.append(f"ATM IV uses the closest available contract "
                                   f"({atm['actual_tenor_days']:.0f}D, {atm['actual_moneyness']:.2f} moneyness) "
                                   f"rather than an exact 30D/ATM point — this ticker's chain doesn't have "
                                   f"enough liquid coverage exactly there.")
    if (iv_95["used_fallback"] or iv_105["used_fallback"]) and skew is not None:
        data_quality_notes.append("Skew uses the closest available contracts near 0.95/1.05 moneyness, "
                                   "not an exact 30D sample, for the same reason.")

    return {
        "spot_price": spot_price,
        "atm_iv_30d": atm["value"],
        "put_call_skew_30d": skew,
        "feature_vector": [None if np.isnan(v) else round(float(v), 4) for v in feature_vector],
        "moneyness_points": STANDARD_MONEYNESS_POINTS,
        "tenor_points_days": STANDARD_TENOR_DAYS,
        "data_quality_notes": data_quality_notes,
    }


def _nearest_observed_contract(points_df: pd.DataFrame, target_moneyness: float, target_tenor_days: float,
                                moneyness_scale: float = 0.05, tenor_scale: float = 15.0) -> dict | None:
    """Finds the closest ACTUAL observed contract to a target
    (moneyness, tenor) point by normalized distance — moneyness and
    tenor live on very different numeric scales (roughly 0.6 wide vs.
    hundreds of days wide), so each is divided by a characteristic
    scale (the standard grid's own spacing) before comparing, making
    the two dimensions roughly comparable in the distance metric."""
    if points_df.empty:
        return None
    dist_sq = (((points_df["moneyness"] - target_moneyness) / moneyness_scale) ** 2 +
               ((points_df["time_to_expiry_days"] - target_tenor_days) / tenor_scale) ** 2)
    idx = dist_sq.idxmin()
    row = points_df.loc[idx]
    return {
        "implied_vol": float(row["implied_vol"]),
        "actual_moneyness": round(float(row["moneyness"]), 3),
        "actual_tenor_days": round(float(row["time_to_expiry_days"]), 1),
    }
