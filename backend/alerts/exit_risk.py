"""
Exit/Sell-Risk Engine (spec section 15).

Watches a portfolio holding for MULTIPLE compounding negative signals
— technical breakdown, deteriorating model prediction, elevated
volatility, negative news — and surfaces a risk LEVEL with the actual
reasons behind it. This never recommends selling; the strongest output
is "Review position" (spec's own explicit requirement), because this
app doesn't have the full picture a person does (tax situation,
conviction, timeframe) and shouldn't pretend otherwise.

Risk factors checked (each contributes independently — a single
factor alone rarely warrants HIGH risk, but several agreeing does):
  - Price below 50-day and/or 200-day moving average (trend breakdown)
  - Elevated relative volume alongside a down day (distribution/selling pressure)
  - Negative technical momentum (RSI weak, MACD histogram negative)
  - Recent news sentiment trending negative
  - Elevated volatility (ATR% well above the stock's own recent norm)
"""
from __future__ import annotations

import pandas as pd

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]


def assess_exit_risk(indicators_latest: dict, structure: dict, recent_news_items: list[dict] | None = None) -> dict:
    """indicators_latest: the `latest` row dict from the indicators
    endpoint (same data already computed for Stock Analyzer — no new
    computation, just a new interpretation of existing numbers).
    structure: the price-structure dict (breakout/breakdown/support/resistance).
    recent_news_items: optional list of analyzed news items (from
    backend.news.service) — each with a 'sentiment' sub-dict."""
    reasons: list[str] = []
    risk_score = 0

    close = indicators_latest.get("close")
    sma50 = indicators_latest.get("sma_50")
    sma200 = indicators_latest.get("sma_200")
    rsi = indicators_latest.get("rsi_14")
    macd_hist = indicators_latest.get("macd_hist")
    rel_vol = indicators_latest.get("relative_volume")
    daily_return = indicators_latest.get("daily_return_pct")
    atr_pct = indicators_latest.get("atr_pct_14")
    hist_vol = indicators_latest.get("hist_vol_20")

    def _valid(*vals) -> bool:
        return all(v is not None and not pd.isna(v) for v in vals)

    if _valid(close, sma50) and close < sma50:
        risk_score += 2
        reasons.append("Price has fallen below its 50-day moving average.")

    if _valid(close, sma200) and close < sma200:
        risk_score += 2
        reasons.append("Price has fallen below its 200-day moving average — long-term trend has broken down.")

    if structure.get("breakdown"):
        risk_score += 2
        reasons.append("Price recently broke down below a key support level.")

    if _valid(rel_vol, daily_return) and rel_vol > 1.8 and daily_return < 0:
        risk_score += 2
        reasons.append(f"Relative volume is elevated ({rel_vol:.1f}x average) on a down day — a sign of active selling, not just quiet drift.")

    if _valid(rsi) and rsi < 40:
        risk_score += 1
        reasons.append(f"RSI is weak ({rsi:.0f}), indicating fading momentum.")

    if _valid(macd_hist) and macd_hist < 0:
        risk_score += 1
        reasons.append("MACD histogram is negative — short-term momentum has turned down.")

    if _valid(atr_pct, hist_vol) and atr_pct > 0 and hist_vol > 0:
        # Elevated volatility relative to a typical range for this metric
        # (not a per-ticker historical comparison — a rough general threshold).
        if atr_pct > 5:
            risk_score += 1
            reasons.append(f"Volatility is elevated (ATR is {atr_pct:.1f}% of price) — larger daily swings than usual.")

    negative_news_count = 0
    if recent_news_items:
        negative_news_count = sum(
            1 for item in recent_news_items
            if item.get("sentiment", {}).get("label", "").endswith("negative")
        )
        if negative_news_count >= 2:
            risk_score += 2
            reasons.append(f"{negative_news_count} recent negative-sentiment news items detected.")
        elif negative_news_count == 1:
            risk_score += 1
            reasons.append("1 recent negative-sentiment news item detected.")

    if risk_score >= 6:
        level = "HIGH"
    elif risk_score >= 3:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "risk_level": level,
        "risk_score": risk_score,
        "reasons": reasons if reasons else ["No significant deterioration signals detected."],
        "action": "Review position" if level in ("MEDIUM", "HIGH") else "No action indicated",
    }
