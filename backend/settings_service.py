"""
Persisted user preferences — server-side (not browser localStorage),
so settings stay consistent regardless of which browser or device you
use to open the app. Deliberately a plain key-value store rather than
a rigid schema: new settings can be added without a migration.
"""
from __future__ import annotations

from backend.database.db import db_cursor

DEFAULTS = {
    "default_risk_profile": "moderate",
    "browser_notifications_enabled": "false",
    "alert_check_interval_minutes": "15",
    "news_direction_threshold": "15",  # |avg impact score| needed before news counts as directional, not neutral —
    # a starting guess, not a value validated against real outcomes yet; see backend/news/accuracy_tracker.py,
    # which is what should eventually settle this with evidence instead of a guess.
}


def get_all_settings() -> dict:
    with db_cursor() as cur:
        cur.execute("SELECT key, value FROM settings")
        stored = {row["key"]: row["value"] for row in cur.fetchall()}
    return {**DEFAULTS, **stored}


def get_setting(key: str) -> str | None:
    with db_cursor() as cur:
        cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
    return row["value"] if row else DEFAULTS.get(key)


def set_setting(key: str, value: str) -> None:
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )


def clear_historical_cache() -> int:
    """Clears the SQLite historical price cache — forces the next
    request for any ticker to fetch fresh data instead of reusing
    whatever's cached, useful if data ever looks stale or wrong."""
    with db_cursor() as cur:
        cur.execute("DELETE FROM historical_prices")
        return cur.rowcount


def get_cache_stats() -> dict:
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(DISTINCT ticker) as tickers, COUNT(*) as rows FROM historical_prices")
        row = cur.fetchone()
        cur.execute("SELECT COUNT(*) as c FROM fundamentals")
        fund_count = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM sec_cache")
        sec_count = cur.fetchone()["c"]
    return {
        "cached_tickers": row["tickers"] or 0,
        "cached_price_rows": row["rows"] or 0,
        "cached_fundamentals": fund_count,
        "cached_sec_filings": sec_count,
    }
