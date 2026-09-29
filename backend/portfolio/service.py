"""
Portfolio tracking (spec section 14).

Purely manual entry — this app never places or simulates trades on
your behalf, it just tracks positions you tell it about and shows
their current standing using the same live market data as the rest
of the app. No holding is ever auto-sold or auto-modified; the exit-
risk engine (backend/alerts/exit_risk.py) only ever surfaces a
"review position" flag, never an instruction to act.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from backend.data.market_data_service import market_data_service
from backend.database.db import db_cursor


def add_holding(ticker: str, shares: float, entry_price: float,
                 entry_date: Optional[str] = None, notes: str = "") -> int:
    ticker = ticker.upper()
    entry_date = entry_date or datetime.now(timezone.utc).date().isoformat()
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO portfolio (ticker, shares, entry_price, entry_date, notes) VALUES (?, ?, ?, ?, ?)",
            (ticker, shares, entry_price, entry_date, notes),
        )
        return cur.lastrowid


def remove_holding(holding_id: int) -> bool:
    with db_cursor() as cur:
        cur.execute("DELETE FROM portfolio WHERE id = ?", (holding_id,))
        return cur.rowcount > 0


def list_holdings_raw() -> list[dict]:
    with db_cursor() as cur:
        cur.execute("SELECT id, ticker, shares, entry_price, entry_date, notes FROM portfolio ORDER BY entry_date DESC")
        return [dict(r) for r in cur.fetchall()]


def get_portfolio_summary() -> dict:
    """Returns every holding enriched with live price, unrealized P&L,
    and portfolio allocation — all derived from the same market data
    service (and its honest source/quality metadata) used everywhere
    else in the app. A holding whose live price can't be fetched shows
    that plainly rather than silently using stale/fabricated numbers."""
    holdings = list_holdings_raw()
    if not holdings:
        return {"holdings": [], "total_value": 0.0, "total_cost_basis": 0.0,
                "total_unrealized_pl": 0.0, "total_unrealized_pl_pct": None}

    enriched = []
    total_value = 0.0
    total_cost_basis = 0.0
    any_price_missing = False

    for h in holdings:
        quote_result = market_data_service.get_quote(h["ticker"])
        cost_basis = h["shares"] * h["entry_price"]
        total_cost_basis += cost_basis

        if quote_result.success and quote_result.data and quote_result.data.get("price") is not None:
            current_price = quote_result.data["price"]
            current_value = h["shares"] * current_price
            unrealized_pl = current_value - cost_basis
            unrealized_pl_pct = (unrealized_pl / cost_basis * 100) if cost_basis else None
            total_value += current_value
            enriched.append({
                **h,
                "current_price": round(current_price, 2),
                "current_value": round(current_value, 2),
                "cost_basis": round(cost_basis, 2),
                "unrealized_pl": round(unrealized_pl, 2),
                "unrealized_pl_pct": round(unrealized_pl_pct, 2) if unrealized_pl_pct is not None else None,
                "price_source": quote_result.source,
                "price_available": True,
            })
        else:
            any_price_missing = True
            enriched.append({
                **h,
                "current_price": None, "current_value": None,
                "cost_basis": round(cost_basis, 2),
                "unrealized_pl": None, "unrealized_pl_pct": None,
                "price_source": None, "price_available": False,
                "price_error": quote_result.error,
            })

    for e in enriched:
        e["allocation_pct"] = round(e["current_value"] / total_value * 100, 1) if (total_value and e["current_value"] is not None) else None

    total_unrealized_pl = total_value - total_cost_basis if not any_price_missing else None
    total_unrealized_pl_pct = (
        round(total_unrealized_pl / total_cost_basis * 100, 2)
        if total_unrealized_pl is not None and total_cost_basis else None
    )

    return {
        "holdings": enriched,
        "total_value": round(total_value, 2) if not any_price_missing else None,
        "total_cost_basis": round(total_cost_basis, 2),
        "total_unrealized_pl": round(total_unrealized_pl, 2) if total_unrealized_pl is not None else None,
        "total_unrealized_pl_pct": total_unrealized_pl_pct,
        "note": "Some holdings' live prices are unavailable — totals shown as unavailable rather than partial/misleading." if any_price_missing else None,
    }
