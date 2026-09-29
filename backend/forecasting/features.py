"""
Feature engineering for the prediction models (spec section 9).

Everything here is deliberately STATIONARY — returns, ratios, and
z-scores rather than raw price levels — because raw prices are
non-stationary (their mean/variance drift over time), which is exactly
the kind of thing that makes a model look good in-sample and fall
apart out-of-sample. A model trained on "AAPL trades around $180"
learns nothing transferable; one trained on "RSI is elevated and price
is 2 standard deviations above its 20-day mean" generalizes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_20d",
    "rsi_14", "macd_hist_norm", "stoch_k",
    "bb_width_pct", "atr_pct_14", "relative_volume",
    "price_vs_sma50", "price_vs_sma200", "sma50_vs_sma200",
    "momentum_10_norm", "volatility_ratio",
    "zscore_20d", "zscore_60d",
    "drawdown_pct",
    "relative_strength_5d", "relative_strength_20d",
]


def build_features(indicators_df: pd.DataFrame, benchmark_close: pd.Series | None = None) -> pd.DataFrame:
    """Takes the output of compute_all_indicators() (which already has
    SMA/RSI/MACD/ATR/etc.) and derives the stationary feature set used
    by every model. Returns a DataFrame aligned to indicators_df's
    index with one column per feature in FEATURE_COLUMNS, plus 'date'
    and 'close' passed through for convenience."""
    df = indicators_df.copy().reset_index(drop=True)
    close = df["close"]

    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"]
    out["close"] = close

    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_20d"] = close.pct_change(20)

    out["rsi_14"] = df.get("rsi_14")
    # MACD histogram is in price units — normalize by price so it's
    # comparable across tickers of wildly different share prices.
    out["macd_hist_norm"] = df.get("macd_hist") / close.replace(0, np.nan)
    out["stoch_k"] = df.get("stoch_k")

    out["bb_width_pct"] = df.get("bb_width_pct")
    out["atr_pct_14"] = df.get("atr_pct_14")
    out["relative_volume"] = df.get("relative_volume")

    sma50 = df.get("sma_50")
    sma200 = df.get("sma_200")
    out["price_vs_sma50"] = (close - sma50) / sma50.replace(0, np.nan)
    out["price_vs_sma200"] = (close - sma200) / sma200.replace(0, np.nan)
    out["sma50_vs_sma200"] = (sma50 - sma200) / sma200.replace(0, np.nan)

    out["momentum_10_norm"] = df.get("momentum_10") / close.replace(0, np.nan)

    # Volatility ratio: short-term realized vol vs. longer-term — a
    # ratio above 1 means volatility is currently expanding relative
    # to its own recent history, a genuinely stationary regime signal.
    log_ret = np.log(close / close.shift(1))
    vol_short = log_ret.rolling(10).std()
    vol_long = log_ret.rolling(60).std()
    out["volatility_ratio"] = vol_short / vol_long.replace(0, np.nan)

    # Rolling z-scores of price relative to its own recent distribution.
    for window in (20, 60):
        roll_mean = close.rolling(window).mean()
        roll_std = close.rolling(window).std()
        out[f"zscore_{window}d"] = (close - roll_mean) / roll_std.replace(0, np.nan)

    out["drawdown_pct"] = df.get("drawdown_pct")

    # Relative strength vs. the broad market — a stock outperforming or
    # underperforming SPY over the same window is a genuinely different
    # (and well-documented) signal than its own absolute return, and
    # this feature was completely absent before: every model was
    # scoring each ticker in total isolation, blind to whether "up 2%"
    # meant leading the market or lagging a market that was up 5%.
    # No look-ahead risk here — for any historical date, this uses only
    # the benchmark's own price as of that same date.
    #
    # Only added when a benchmark series is actually available — adding
    # these columns pre-filled with NaN when it's not would cause
    # build_training_matrix()'s NaN-dropping to wipe out every row,
    # since it requires ALL feature columns present in the frame to be
    # non-null. build_training_matrix() only uses whichever of
    # FEATURE_COLUMNS actually exist in this DataFrame, so simply
    # omitting these two when there's no benchmark degrades gracefully
    # instead of breaking training entirely.
    if benchmark_close is not None and len(benchmark_close) == len(close):
        bench = benchmark_close.reset_index(drop=True)
        stock_ret_5d = close.pct_change(5)
        stock_ret_20d = close.pct_change(20)
        bench_ret_5d = bench.pct_change(5)
        bench_ret_20d = bench.pct_change(20)
        out["relative_strength_5d"] = stock_ret_5d - bench_ret_5d
        out["relative_strength_20d"] = stock_ret_20d - bench_ret_20d

    return out


def build_training_matrix(features_df: pd.DataFrame, horizon_days: int, extra_columns: list[str] | None = None
                           ) -> tuple[pd.DataFrame, pd.Series, pd.Series] | tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Builds (X, y_return, y_direction) for a given prediction horizon.

    y_return: forward return over `horizon_days` trading days —
      (close[t+h] / close[t]) - 1. This is the regression target.
    y_direction: +1 / 0 / -1 classification derived from y_return with
      a small dead-zone around zero (so tiny noise-level moves aren't
      forced into a bullish/bearish bucket).

    Rows without a complete feature set (early rolling-window NaNs) or
    without a complete forward-looking target (the last `horizon_days`
    rows, which don't have a known future yet) are dropped — this is
    the leakage-prevention step: we NEVER train on a row whose target
    depends on data that didn't exist yet at prediction time, and we
    never fabricate a target for the most recent rows.

    extra_columns: if provided (e.g. ["date", "close"]), a 4th value
    is returned — those columns from the SAME rows retained in X,
    correctly aligned. This exists specifically so callers never have
    to reach for `features_df[col].iloc[-len(X):]` themselves: that
    naive positional slice is WRONG (a real bug found via testing,
    since fixed in two separate places) — it takes the last len(X)
    rows of the ORIGINAL, undropped DataFrame, which includes the tail
    rows X itself excludes (no valid target yet) and misses the actual
    head of X's rows, silently shifting every returned value forward
    by horizon_days rows. Requesting extra_columns here guarantees
    correct alignment by construction, since it reuses the exact same
    mask used to build X itself.
    """
    df = features_df.copy()
    close = df["close"]

    forward_return = close.shift(-horizon_days) / close - 1
    df["y_return"] = forward_return

    # Dead-zone threshold scales mildly with horizon — a 1-day move of
    # 0.3% is noise; a 20-day move of 0.3% is also arguably noise, so
    # we don't scale it up proportionally, just give slightly more
    # room for longer horizons.
    dead_zone = 0.01 if horizon_days <= 5 else 0.02
    df["y_direction"] = np.select(
        [df["y_return"] > dead_zone, df["y_return"] < -dead_zone],
        [1, -1], default=0,
    )

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    complete = df.dropna(subset=feature_cols + ["y_return"])

    X = complete[feature_cols].reset_index(drop=True)
    y_return = complete["y_return"].reset_index(drop=True)
    y_direction = complete["y_direction"].reset_index(drop=True)

    if extra_columns:
        extras = complete[extra_columns].reset_index(drop=True)
        return X, y_return, y_direction, extras
    return X, y_return, y_direction
