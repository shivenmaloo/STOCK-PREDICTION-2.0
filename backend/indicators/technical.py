"""
Technical indicator calculations, computed directly with pandas/NumPy
(no ta-lib dependency, so it installs cleanly everywhere).

All functions take/return pandas Series or add columns to a DataFrame
that must already have: open, high, low, close, volume (lowercase).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50)  # flat series -> neutral RSI rather than NaN


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram})


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    percent_k = 100 * (close - lowest_low) / denom
    percent_d = percent_k.rolling(d_window).mean()
    return pd.DataFrame({"stoch_k": percent_k, "stoch_d": percent_d})


def roc(close: pd.Series, window: int = 12) -> pd.Series:
    return (close / close.shift(window) - 1) * 100


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    return close - close.shift(window)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return (atr(high, low, close, window) / close) * 100


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width_pct = ((upper - lower) / mid.replace(0, np.nan)) * 100
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width_pct": width_pct})


def historical_volatility(close: pd.Series, window: int = 20, trading_days: int = 252) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(trading_days) * 100


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_smoothed = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr_smoothed)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr_smoothed)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    return pd.DataFrame({"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di})


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).fillna(0).cumsum()


def volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    return sma(volume, window)


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    avg = volume_sma(volume, window)
    return volume / avg.replace(0, np.nan)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Given an OHLCV DataFrame (date, open, high, low, close, adj_close,
    volume), return a copy with every indicator column added."""
    out = df.copy().reset_index(drop=True)
    close, high, low, vol = out["close"], out["high"], out["low"], out["volume"]

    # Moving averages
    for w in (10, 20, 50, 100, 200):
        out[f"sma_{w}"] = sma(close, w)
    for w in (9, 12, 21, 26, 50):
        out[f"ema_{w}"] = ema(close, w)

    # Momentum
    out["rsi_14"] = rsi(close, 14)
    macd_df = macd(close)
    out = pd.concat([out, macd_df], axis=1)
    stoch_df = stochastic(high, low, close)
    out = pd.concat([out, stoch_df], axis=1)
    out["roc_12"] = roc(close, 12)
    out["momentum_10"] = momentum(close, 10)

    # Volatility
    out["atr_14"] = atr(high, low, close, 14)
    out["atr_pct_14"] = atr_percent(high, low, close, 14)
    bb_df = bollinger_bands(close)
    out = pd.concat([out, bb_df], axis=1)
    out["hist_vol_20"] = historical_volatility(close, 20)

    # Trend
    adx_df = adx(high, low, close, 14)
    out = pd.concat([out, adx_df], axis=1)

    # Volume
    out["obv"] = obv(close, vol)
    out["volume_sma_20"] = volume_sma(vol, 20)
    out["relative_volume"] = relative_volume(vol, 20)
    out["volume_spike"] = out["relative_volume"] > 2.0

    # Returns / gaps / drawdown
    out["daily_return_pct"] = close.pct_change() * 100
    out["gap_pct"] = (out["open"] - close.shift(1)) / close.shift(1) * 100
    running_max = close.cummax()
    out["drawdown_pct"] = (close - running_max) / running_max * 100

    return out


def detect_price_structure(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Lightweight support/resistance + breakout/breakdown + swing-point
    detection over the most recent `lookback` bars."""
    recent = df.tail(lookback)
    if recent.empty or len(recent) < 5:
        return {
            "support": None, "resistance": None,
            "breakout": False, "breakdown": False,
            "structure": "insufficient_data",
        }

    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    last_close = float(df["close"].iloc[-1])
    prev_high = float(recent["high"].iloc[:-1].max())
    prev_low = float(recent["low"].iloc[:-1].min())

    breakout = last_close > prev_high
    breakdown = last_close < prev_low

    # crude swing structure: compare the two most recent local highs/lows
    highs = recent["high"].values
    lows = recent["low"].values
    mid = len(highs) // 2
    first_half_high, second_half_high = highs[:mid].max(), highs[mid:].max()
    first_half_low, second_half_low = lows[:mid].min(), lows[mid:].min()

    if second_half_high > first_half_high and second_half_low > first_half_low:
        structure = "higher_highs_higher_lows"
    elif second_half_high < first_half_high and second_half_low < first_half_low:
        structure = "lower_highs_lower_lows"
    else:
        structure = "mixed"

    return {
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "breakout": bool(breakout),
        "breakdown": bool(breakdown),
        "structure": structure,
    }


def interpret_technicals(latest: pd.Series, structure: dict) -> str:
    """Generate the plain-English interpretation the spec asks for
    (section 5) instead of just dumping numbers on the user."""
    parts = []

    rsi_val = latest.get("rsi_14")
    if pd.notna(rsi_val):
        if rsi_val >= 70:
            parts.append(f"RSI is {rsi_val:.0f}, in overbought territory")
        elif rsi_val <= 30:
            parts.append(f"RSI is {rsi_val:.0f}, in oversold territory")
        else:
            parts.append(f"RSI is {rsi_val:.0f}")

    close = latest.get("close")
    sma50, sma200 = latest.get("sma_50"), latest.get("sma_200")
    if pd.notna(close) and pd.notna(sma50) and pd.notna(sma200):
        if close > sma50 and close > sma200:
            parts.append("price is above both the 50-day and 200-day moving averages")
        elif close < sma50 and close < sma200:
            parts.append("price is below both the 50-day and 200-day moving averages")
        else:
            parts.append("price is mixed relative to the 50-day and 200-day moving averages")

    rel_vol = latest.get("relative_volume")
    if pd.notna(rel_vol):
        parts.append(f"relative volume is {rel_vol:.1f}x average")

    macd_hist = latest.get("macd_hist")
    if pd.notna(macd_hist):
        parts.append(f"MACD histogram is {'positive' if macd_hist > 0 else 'negative'}")

    if structure.get("breakout"):
        parts.append("price just broke out above recent resistance")
    elif structure.get("breakdown"):
        parts.append("price just broke down below recent support")

    if not parts:
        return "Not enough history yet to interpret technicals for this ticker."

    lead = ", ".join(parts[:-1])
    tail = parts[-1]
    sentence = f"{lead}, and {tail}." if lead else f"{tail.capitalize()}."

    # crude directional lean for the closing sentence
    bullish_votes = sum([
        pd.notna(rsi_val) and rsi_val > 55,
        pd.notna(close) and pd.notna(sma50) and close > sma50,
        pd.notna(macd_hist) and macd_hist > 0,
        structure.get("breakout", False),
    ])
    bearish_votes = sum([
        pd.notna(rsi_val) and rsi_val < 45,
        pd.notna(close) and pd.notna(sma50) and close < sma50,
        pd.notna(macd_hist) and macd_hist < 0,
        structure.get("breakdown", False),
    ])
    if bullish_votes >= 3:
        lean = "This supports a bullish momentum regime."
    elif bearish_votes >= 3:
        lean = "This supports a bearish momentum regime."
    else:
        lean = "This paints a mixed picture with no strong momentum lean."

    return f"{sentence[0].upper()}{sentence[1:]} {lean}"
