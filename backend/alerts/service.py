"""
Alert generation (spec section 13).

Checks the watchlist and portfolio for MATERIAL changes — a signal
flip, a portfolio position's risk level escalating — and stores an
alert only when something actually changed since the last check, with
the specific reasons attached. This is what prevents alert spam: the
service tracks each ticker's last-known signal/risk-level and only
fires when that value is genuinely different from before, never on
every single scheduler tick just because a check ran.

State tracking uses its own internal alert_type suffix ("_state") so
it never shows up in the user-facing alerts list (get_alerts()
excludes it) — every check updates the state row regardless of
whether anything changed, but a real, visible alert is only written
when the state has genuinely moved since the previous check.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from backend.alerts.exit_risk import assess_exit_risk
from backend.data.market_data_service import market_data_service
from backend.database.db import db_cursor
from backend.indicators.technical import compute_all_indicators, detect_price_structure

logger = logging.getLogger("stock_ai.alerts")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

STATE_SUFFIX = "_state"


def _get_last_known_value(ticker: str, state_key: str) -> str | None:
    with db_cursor() as cur:
        cur.execute(
            "SELECT reasons_json FROM alerts WHERE ticker = ? AND alert_type = ? ORDER BY created_at DESC LIMIT 1",
            (ticker, state_key + STATE_SUFFIX),
        )
        row = cur.fetchone()
    if not row:
        return None
    try:
        return json.loads(row["reasons_json"]).get("value")
    except Exception:  # noqa: BLE001
        return None


def _record_state(ticker: str, state_key: str, value: str) -> None:
    """Always called, every check — this is what next time's comparison
    reads. Never shown in the user-facing alerts list."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO alerts (ticker, created_at, alert_type, title, reasons_json) VALUES (?, ?, ?, ?, ?)",
            (ticker, datetime.now(timezone.utc).isoformat(), state_key + STATE_SUFFIX, "", json.dumps({"value": value})),
        )


def _store_alert(ticker: str, alert_type: str, title: str, reasons: list[str]) -> None:
    """A real, user-facing alert — only called when something material
    actually changed."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO alerts (ticker, created_at, alert_type, title, reasons_json) VALUES (?, ?, ?, ?, ?)",
            (ticker, datetime.now(timezone.utc).isoformat(), alert_type, title, json.dumps({"reasons": reasons})),
        )
    logger.info(f"[ALERTS] {ticker}: {title}")


def check_watchlist_technical_changes() -> int:
    """Checks every watchlisted ticker's technical structure (breakout/
    breakdown) and fires a real alert only when it's DIFFERENT from the
    last recorded state for that ticker — not every time this runs."""
    with db_cursor() as cur:
        cur.execute("SELECT ticker FROM watchlist")
        tickers = [r["ticker"] for r in cur.fetchall()]

    fired = 0
    for ticker in tickers:
        # force_refresh=True — alerts exist specifically to detect
        # change, so this check must never be satisfied by a
        # potentially-hours-old cached snapshot the way a UI page load
        # reasonably can be.
        hist_result = market_data_service.get_historical(ticker, period="6mo", interval="1d", force_refresh=True)
        if not hist_result.success or hist_result.data is None or hist_result.data.empty:
            continue

        indicators_df = compute_all_indicators(hist_result.data)
        structure = detect_price_structure(indicators_df)
        latest = indicators_df.iloc[-1]

        current_state = "breakout" if structure.get("breakout") else "breakdown" if structure.get("breakdown") else "neutral"
        last_state = _get_last_known_value(ticker, "technical_change")

        # Always record the current state so the NEXT check has
        # something correct to compare against, regardless of whether
        # a visible alert fires this time.
        _record_state(ticker, "technical_change", current_state)

        if last_state is not None and last_state != current_state and current_state != "neutral":
            direction = "bullish" if current_state == "breakout" else "bearish"
            price = latest.get("close")
            reasons = [
                f"Price structure moved from '{last_state}' to '{current_state}'.",
                f"Current price: ${price:.2f}" if price is not None else "Price data available in chart.",
            ]
            if structure.get("resistance"):
                reasons.append(f"Resistance level: ${structure['resistance']:.2f}")
            if structure.get("support"):
                reasons.append(f"Support level: ${structure['support']:.2f}")

            _store_alert(ticker, direction, f"{ticker} — technical structure changed to {current_state}", reasons)
            fired += 1

    return fired


def check_portfolio_exit_risk() -> int:
    """Checks every portfolio holding's exit risk and fires a real
    alert only when the risk LEVEL has changed since the last check
    (e.g. LOW -> MEDIUM, or MEDIUM -> HIGH) — not every time it's
    still sitting at the same level."""
    from backend.portfolio.service import list_holdings_raw

    holdings = list_holdings_raw()
    fired = 0

    for holding in holdings:
        ticker = holding["ticker"]
        hist_result = market_data_service.get_historical(ticker, period="6mo", interval="1d", force_refresh=True)
        if not hist_result.success or hist_result.data is None or hist_result.data.empty:
            continue

        indicators_df = compute_all_indicators(hist_result.data)
        structure = detect_price_structure(indicators_df)
        latest = indicators_df.iloc[-1].to_dict()

        risk = assess_exit_risk(latest, structure)
        last_level = _get_last_known_value(ticker, "exit_risk")

        _record_state(ticker, "exit_risk", risk["risk_level"])

        if last_level is not None and last_level != risk["risk_level"] and risk["risk_level"] in ("MEDIUM", "HIGH"):
            _store_alert(
                ticker, "exit_risk",
                f"{ticker} position risk changed: {last_level} → {risk['risk_level']}",
                risk["reasons"],
            )
            fired += 1

    return fired


def run_all_alert_checks() -> dict:
    logger.info("[ALERTS] Running scheduled alert checks...")
    technical_fired = check_watchlist_technical_changes()
    exit_risk_fired = check_portfolio_exit_risk()
    logger.info(f"[ALERTS] Done — {technical_fired} technical alert(s), {exit_risk_fired} exit-risk alert(s)")
    return {"technical_alerts_fired": technical_fired, "exit_risk_alerts_fired": exit_risk_fired}


def get_alerts(unread_only: bool = False, limit: int = 50) -> list[dict]:
    query = f"SELECT id, ticker, created_at, alert_type, title, reasons_json, read FROM alerts WHERE alert_type NOT LIKE '%{STATE_SUFFIX}'"
    if unread_only:
        query += " AND read = 0"
    query += " ORDER BY created_at DESC LIMIT ?"
    with db_cursor() as cur:
        cur.execute(query, (limit,))
        rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        payload = json.loads(r.pop("reasons_json") or "{}")
        r["reasons"] = payload.get("reasons", [])
    return rows


def mark_alert_read(alert_id: int) -> bool:
    with db_cursor() as cur:
        cur.execute("UPDATE alerts SET read = 1 WHERE id = ?", (alert_id,))
        return cur.rowcount > 0


def mark_all_alerts_read() -> int:
    with db_cursor() as cur:
        cur.execute(f"UPDATE alerts SET read = 1 WHERE read = 0 AND alert_type NOT LIKE '%{STATE_SUFFIX}'")
        return cur.rowcount


def count_unread_alerts() -> int:
    with db_cursor() as cur:
        cur.execute(f"SELECT COUNT(*) as c FROM alerts WHERE read = 0 AND alert_type NOT LIKE '%{STATE_SUFFIX}'")
        return cur.fetchone()["c"]
