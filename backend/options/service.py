"""
Orchestrates the full IV surface computation for a ticker: fetches
real options chain data via yfinance, runs every available expiry
through the preprocessing/IV-solving pipeline, computes surface
features, logs the ATM IV reading for rank tracking, and returns a
structured result ready for the API/frontend.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

from backend.data.market_data_service import market_data_service
from backend.options.surface import (
    preprocess_chain, get_surface_features, interpolate_surface, STANDARD_MONEYNESS_POINTS,
)
from backend.options.iv_history import log_iv_reading, get_iv_rank

logger = logging.getLogger("stock_ai.options")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False

# A stated, reasonable proxy for the short-term risk-free rate rather
# than fetching a live Treasury yield — that's a separate data source
# with its own reliability considerations, and the IV solver isn't
# meaningfully sensitive to small differences here (a percentage-point
# error in r moves recovered IV by a small fraction of a percent).
RISK_FREE_RATE_PROXY = 0.045
MAX_EXPIRIES_TO_PROCESS = 6  # keeps a single request from fetching an unbounded number of chains
VIZ_GRID_MONEYNESS_MIN, VIZ_GRID_MONEYNESS_MAX = 0.70, 1.30
VIZ_GRID_POINTS = 30


def fetch_and_build_surface(ticker: str, yf_ticker_factory=None, today: date | None = None) -> dict:
    """yf_ticker_factory and today are injectable for testing — lets
    tests supply a fake object with the same .options/.option_chain()
    interface, and a fixed "today," without needing real network
    access or the result depending on which day the test happens to
    run on."""
    ticker = ticker.upper()
    if yf_ticker_factory is None:
        import yfinance as yf
        yf_ticker_factory = yf.Ticker
    today = today or datetime.now().date()

    quote_result = market_data_service.get_quote(ticker)
    if not quote_result.success:
        return {"success": False, "error": f"Couldn't get a current price for {ticker}."}
    spot_price = quote_result.data["price"]

    try:
        yf_ticker = yf_ticker_factory(ticker)
        expiries = yf_ticker.options
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Couldn't fetch options expiries for {ticker}: {exc}"}

    if not expiries:
        return {"success": False, "error": f"{ticker} doesn't appear to have listed options."}

    all_points = []
    expiries_processed = []
    for expiry_str in expiries[:MAX_EXPIRIES_TO_PROCESS]:
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        expiry_days = (expiry_date - today).days

        try:
            chain = yf_ticker.option_chain(expiry_str)
            calls = chain.calls
        except Exception as exc:  # noqa: BLE001
            logger.info(f"[OPTIONS] Failed to fetch chain for {ticker} {expiry_str}: {exc}")
            continue

        processed = preprocess_chain(calls, spot_price, expiry_days, RISK_FREE_RATE_PROXY)
        if not processed.empty:
            all_points.append(processed)
            expiries_processed.append(expiry_str)

    if not all_points:
        return {"success": False, "error": f"No usable options data survived quality filtering for {ticker}."}

    full_df = pd.concat(all_points, ignore_index=True)

    features = get_surface_features(full_df, spot_price)
    if features is None:
        return {"success": False, "error": f"Not enough data to interpolate a surface for {ticker}."}

    log_iv_reading(ticker, features["atm_iv_30d"], features["put_call_skew_30d"])
    iv_rank = get_iv_rank(ticker, features["atm_iv_30d"])

    # A separate, finer grid purely for the 3D visualization — the
    # standardized 7x4 grid in get_surface_features() stays fixed for
    # ML-feature consistency across tickers/runs, but a chart benefits
    # from a smoother, wider mesh spanning the actual observed tenors.
    min_tenor = int(full_df["time_to_expiry_days"].min())
    max_tenor = int(full_df["time_to_expiry_days"].max())
    viz_tenor_grid = np.linspace(min_tenor, max_tenor, VIZ_GRID_POINTS)
    viz_moneyness_grid = np.linspace(VIZ_GRID_MONEYNESS_MIN, VIZ_GRID_MONEYNESS_MAX, VIZ_GRID_POINTS)
    viz_surface = interpolate_surface(full_df, viz_moneyness_grid, viz_tenor_grid)

    return {
        "success": True,
        "ticker": ticker,
        "spot_price": spot_price,
        "expiries_processed": expiries_processed,
        "n_contracts_used": len(full_df),
        "atm_iv_30d": features["atm_iv_30d"],
        "put_call_skew_30d": features["put_call_skew_30d"],
        "iv_rank": iv_rank,
        "feature_vector": features["feature_vector"],
        "moneyness_points": features["moneyness_points"],
        "tenor_points_days": features["tenor_points_days"],
        "visualization": {
            "moneyness_grid": viz_moneyness_grid.tolist(),
            "tenor_grid_days": viz_tenor_grid.tolist(),
            "iv_surface": [[None if (v is None or np.isnan(v)) else round(float(v), 4) for v in row]
                           for row in viz_surface] if viz_surface is not None else None,
        },
    }
