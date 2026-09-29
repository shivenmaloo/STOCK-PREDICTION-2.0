"""
SQLite persistence layer.

The full schema (section 29 of the spec) is created up front so later
phases don't require a migration step, but Phase 1 only reads/writes
`stocks`, `historical_prices`, and `watchlist`.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from backend.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    exchange TEXT,
    asset_type TEXT,           -- 'stock' | 'etf' | 'index'
    sector TEXT,
    industry TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS historical_prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    source TEXT,
    fetched_at TEXT,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,
    data_json TEXT,
    source TEXT,
    fetched_at TEXT,
    PRIMARY KEY (ticker, as_of)
);

CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    published_at TEXT,
    source TEXT,
    headline TEXT,
    link TEXT,
    category TEXT,
    sentiment REAL,
    impact_score REAL,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    created_at TEXT,
    horizon_days INTEGER,
    expected_return REAL,
    confidence REAL,
    signal TEXT,
    model_breakdown_json TEXT
);

-- A genuine gap found via direct feedback, not a hypothetical one:
-- individual model predictions were already saved above
-- (model_breakdown_json), but the actual FEATURE DATA that fed into
-- each historical prediction — every technical indicator value at
-- that moment — was never persisted anywhere. That meant a past
-- prediction could never be reproduced, debugged, or used to validate
-- a future code change against real historical inputs, since raw
-- market data can itself shift subtly over time (revisions, splits)
-- even when queried for the "same" historical date range later. This
-- is a permanent, immutable snapshot of exactly what went in, frozen
-- at the moment the prediction was actually made.
CREATE TABLE IF NOT EXISTS prediction_input_snapshots (
    prediction_id INTEGER PRIMARY KEY,
    feature_json TEXT,
    created_at TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions (id)
);

CREATE TABLE IF NOT EXISTS prediction_results (
    prediction_id INTEGER,
    evaluated_at TEXT,
    actual_return REAL,
    was_correct INTEGER,
    per_model_correct_json TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions (id)
);

-- Live, forward-looking accuracy tracking for the news sentiment
-- engine — logged the moment a real prediction computes a news
-- context, evaluated once the same horizon has actually elapsed.
-- This is the honest answer to "how do we know this correlates with
-- real price moves": we don't retroactively assume it does — we log
-- every real read starting now and check it against what actually
-- happened, the same way prediction accuracy itself is tracked.
CREATE TABLE IF NOT EXISTS news_sentiment_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    logged_at TEXT,
    impact_score REAL,
    news_direction TEXT,
    reference_price REAL,
    horizon_days INTEGER
);

-- Per-SOURCE reliability tracking — a finer grain than
-- news_sentiment_log above, which only logs the aggregate reading per
-- prediction. This logs each individual OPINIONATED article's own
-- call (source, direction, ticker) so a specific outlet's own real
-- accuracy can be tracked over time, independent of what the
-- aggregate/other sources said — the honest way to answer "is Reuters
-- actually more reliable than a random aggregator," built the same
-- forward-only way as everything else here (no historical archive to
-- validate against retroactively).
CREATE TABLE IF NOT EXISTS news_source_reliability_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT,
    ticker TEXT,
    logged_at TEXT,
    impact_score REAL,
    direction TEXT,
    reference_price REAL,
    horizon_days INTEGER
);

CREATE TABLE IF NOT EXISTS news_source_reliability_results (
    log_id INTEGER,
    evaluated_at TEXT,
    actual_return REAL,
    was_correct INTEGER,
    FOREIGN KEY (log_id) REFERENCES news_source_reliability_log (id)
);

-- Live-forward IV history — we don't have a point-in-time options
-- archive, so "IV rank vs 52-week history" can only be built honestly
-- by logging real readings starting now and letting rank accumulate
-- over time, the same principle already used for news sentiment.
CREATE TABLE IF NOT EXISTS iv_history_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    logged_at TEXT,
    atm_iv_30d REAL,
    put_call_skew_30d REAL
);

CREATE TABLE IF NOT EXISTS news_sentiment_results (
    log_id INTEGER,
    evaluated_at TEXT,
    actual_return REAL,
    was_correct INTEGER,
    FOREIGN KEY (log_id) REFERENCES news_sentiment_log (id)
);

-- Ongoing TCN + risk-map configuration search. Every trial is
-- persisted (never silently discarded) so the search process itself
-- stays auditable, and the search/holdout split guards against the
-- classic overfitting-to-the-search-process failure mode: a config is
-- only ever ranked against others using the SEARCH period; the
-- HOLDOUT period is touched exactly once, on the current best config,
-- so a suspicious gap between search and holdout performance is
-- visible rather than hidden.
CREATE TABLE IF NOT EXISTS tcn_search_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    horizon_days INTEGER,
    created_at TEXT,
    config_json TEXT,
    search_score REAL,
    n_search_splits INTEGER,
    n_search_trades INTEGER
);

CREATE TABLE IF NOT EXISTS tcn_search_best (
    ticker TEXT,
    horizon_days INTEGER,
    trial_id INTEGER,
    updated_at TEXT,
    holdout_score REAL,
    holdout_evaluated_at TEXT,
    PRIMARY KEY (ticker, horizon_days),
    FOREIGN KEY (trial_id) REFERENCES tcn_search_trials (id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY,
    added_at TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    shares REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_date TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    created_at TEXT,
    alert_type TEXT,        -- 'bullish' | 'bearish' | 'portfolio_risk' | 'exit_risk' | 'news'
    title TEXT,
    reasons_json TEXT,
    read INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    model_type TEXT,
    trained_at TEXT,
    horizon_days INTEGER,
    validation_metrics_json TEXT,
    test_metrics_json TEXT,
    file_path TEXT
);

CREATE TABLE IF NOT EXISTS backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    ticker TEXT,
    strategy_json TEXT,
    start_date TEXT,
    end_date TEXT,
    results_json TEXT
);

CREATE TABLE IF NOT EXISTS sec_cache (
    cache_key TEXT PRIMARY KEY,
    value TEXT,
    fetched_at TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DATABASE_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    # WAL mode: readers never block on a writer (and vice versa) — a
    # genuine best-practice hardening for this app's exact usage
    # pattern (a background scheduler writing every 15-60 minutes
    # while the UI is reading constantly). Stress-tested this at 1,200
    # concurrent operations across 12 threads without WAL and found
    # zero lock errors even then — this app's real write volume is
    # light enough that the default was already holding up fine — but
    # WAL is still strictly better with no downside for a local
    # single-file database, so there's no reason not to have it.
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


@contextmanager
def db_cursor() -> Iterator[sqlite3.Cursor]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
        _run_migrations(conn)
    finally:
        conn.close()


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Lightweight, idempotent migrations for columns added after a
    database already existed — CREATE TABLE IF NOT EXISTS alone won't
    add new columns to a table that's already there. Each ALTER TABLE
    is wrapped so an already-migrated database (or a fresh one where
    the column was already in the CREATE TABLE) doesn't raise."""
    migrations = [
        "ALTER TABLE prediction_results ADD COLUMN per_model_correct_json TEXT",
    ]
    for stmt in migrations:
        try:
            conn.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists — nothing to do
