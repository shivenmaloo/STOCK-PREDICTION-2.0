"""
Market regime model (spec section 16).

Classifies the overall market as Strong Bull / Bull / Neutral / Bear /
Strong Bear using: trend of SPY/QQQ/DIA/IWM relative to their 50/200-day
moving averages, recent 1-month momentum, and the VIX level as a fear
gauge. Rules-based and fully explained — not a trained model.
"""
from __future__ import annotations

from backend.data.market_data_service import market_data_service
from backend.indicators.technical import sma

BENCHMARKS = ["SPY", "QQQ", "DIA", "IWM"]


def _benchmark_signal(ticker: str) -> dict | None:
    result = market_data_service.get_historical(ticker, period="1y", interval="1d")
    if not result.success or result.data is None or result.data.empty or len(result.data) < 60:
        return None

    df = result.data
    close = df["close"]
    sma50 = sma(close, 50).iloc[-1]
    sma200 = sma(close, 200).iloc[-1] if len(close) >= 200 else None
    last = close.iloc[-1]
    month_ago_idx = max(0, len(close) - 22)
    momentum_1m = (last / close.iloc[month_ago_idx] - 1) * 100 if close.iloc[month_ago_idx] else 0

    above_50 = bool(sma50 == sma50 and last > sma50)
    above_200 = bool(sma200 == sma200 and last > sma200) if sma200 is not None else None

    return {
        "ticker": ticker,
        "last": round(float(last), 2),
        "above_50sma": above_50,
        "above_200sma": above_200,
        "momentum_1m_pct": round(float(momentum_1m), 2),
    }


def _vix_level() -> float | None:
    result = market_data_service.get_quote("^VIX")
    if result.success and result.data:
        return result.data.get("price")
    # fall back to last historical close if the quote path fails
    hist = market_data_service.get_historical("^VIX", period="5d")
    if hist.success and hist.data is not None and not hist.data.empty:
        return float(hist.data["close"].iloc[-1])
    return None


def classify_market_regime() -> dict:
    signals = []
    for b in BENCHMARKS:
        sig = _benchmark_signal(b)
        if sig:
            signals.append(sig)

    vix = _vix_level()

    if not signals:
        return {
            "regime": "UNKNOWN",
            "confidence": "low",
            "signals": [],
            "vix": round(vix, 2) if vix is not None else None,
            "explanation": "Not enough benchmark data was available to classify the market regime right now.",
        }

    bull_votes = sum(1 for s in signals if s["above_50sma"] and (s["above_200sma"] in (True, None)))
    bear_votes = sum(1 for s in signals if not s["above_50sma"] and s["above_200sma"] is False)
    avg_momentum = sum(s["momentum_1m_pct"] for s in signals) / len(signals)

    score = 0
    score += bull_votes - bear_votes
    if avg_momentum > 3:
        score += 1
    elif avg_momentum < -3:
        score -= 1
    if vix is not None:
        if vix < 15:
            score += 1
        elif vix > 30:
            score -= 2
        elif vix > 22:
            score -= 1

    if score >= 4:
        regime = "STRONG_BULL"
    elif score >= 2:
        regime = "BULL"
    elif score <= -4:
        regime = "STRONG_BEAR"
    elif score <= -2:
        regime = "BEAR"
    else:
        regime = "NEUTRAL"

    parts = [
        f"{bull_votes}/{len(signals)} major indexes are above their 50-day (and 200-day, where available) moving averages",
        f"average 1-month momentum across indexes is {avg_momentum:+.1f}%",
    ]
    if vix is not None:
        parts.append(f"VIX is at {vix:.1f}")
    explanation = "; ".join(parts) + f". This supports a {regime.replace('_', ' ').title()} regime classification."

    return {
        "regime": regime,
        "confidence": "high" if len(signals) == len(BENCHMARKS) else "medium",
        "signals": signals,
        "vix": round(vix, 2) if vix is not None else None,
        "explanation": explanation,
    }
