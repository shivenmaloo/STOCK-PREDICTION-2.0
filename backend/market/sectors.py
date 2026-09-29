"""
Sector analysis (spec section 17), using the standard SPDR sector ETFs
as liquid proxies for each GICS sector — the same approach professional
research desks use when they don't want to average hundreds of
individual constituents.
"""
from __future__ import annotations

from backend.data.market_data_service import market_data_service

SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Materials": "XLB",
}


def _sector_momentum(ticker: str) -> dict | None:
    result = market_data_service.get_historical(ticker, period="3mo", interval="1d")
    if not result.success or result.data is None or result.data.empty or len(result.data) < 22:
        return None

    close = result.data["close"]
    last = close.iloc[-1]
    ret_1m = (last / close.iloc[max(0, len(close) - 22)] - 1) * 100
    ret_3m = (last / close.iloc[0] - 1) * 100
    daily_returns = close.pct_change().dropna()
    volatility = float(daily_returns.std() * (252 ** 0.5) * 100) if len(daily_returns) > 5 else None

    # Rotation signal: is the sector's momentum picking up or fading?
    # Compare the last month's pace against the average monthly pace over
    # the full 3-month window — a simple but honest "is this accelerating"
    # check, not a fitted model.
    monthly_pace_3m = ret_3m / 3
    if monthly_pace_3m == 0:
        rotation = "steady"
    elif ret_1m > monthly_pace_3m * 1.3:
        rotation = "accelerating"
    elif ret_1m < monthly_pace_3m * 0.7:
        rotation = "decelerating"
    else:
        rotation = "steady"

    return {
        "return_1m_pct": round(float(ret_1m), 2),
        "return_3m_pct": round(float(ret_3m), 2),
        "volatility_ann_pct": round(volatility, 1) if volatility is not None else None,
        "rotation": rotation,
    }


def rank_sectors() -> dict:
    rows = []
    for sector, etf in SECTOR_ETFS.items():
        stats = _sector_momentum(etf)
        if stats is None:
            rows.append({"sector": sector, "etf": etf, "data_available": False})
            continue
        rows.append({"sector": sector, "etf": etf, "data_available": True, **stats})

    available = [r for r in rows if r["data_available"]]
    ranked = sorted(available, key=lambda r: r["return_1m_pct"], reverse=True)

    top = ranked[:3]
    bottom = ranked[-3:] if len(ranked) >= 3 else []
    accelerating = [r["sector"] for r in available if r["rotation"] == "accelerating"]
    decelerating = [r["sector"] for r in available if r["rotation"] == "decelerating"]

    return {
        "sectors": ranked,
        "top_sectors": [r["sector"] for r in top],
        "weakest_sectors": [r["sector"] for r in reversed(bottom)],
        "accelerating_sectors": accelerating,
        "decelerating_sectors": decelerating,
        "unavailable": [r["sector"] for r in rows if not r["data_available"]],
    }
