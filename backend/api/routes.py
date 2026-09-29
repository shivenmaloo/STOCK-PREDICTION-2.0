from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.data.market_data_service import market_data_service
from backend.database.db import db_cursor
from backend.indicators.technical import compute_all_indicators, detect_price_structure, interpret_technicals
from backend.fundamentals.service import fundamentals_service
from backend.fundamentals.scoring import compute_fundamental_score
from backend.news.service import news_service
from backend.market.regime import classify_market_regime
from backend.market.sectors import rank_sectors
from backend.scanner.scanner import ScanFilters, run_scan
from backend.sec.service import sec_service
from backend.forecasting.service import prediction_service
from backend.forecasting.models import XGBOOST_AVAILABLE
from backend.forecasting.lstm_model import TORCH_AVAILABLE
from backend.backtesting.service import run_backtest
import backend.settings_service as settings_service
import backend.chat.service as chat_service
from backend.forecasting.suggestions import (
    get_stock_suggestions, refresh_tracked_ticker_predictions,
    get_discovery_suggestions, get_discovery_coverage, run_discovery_sweep,
)
from backend.news.accuracy_tracker import get_news_sentiment_accuracy, evaluate_matured_news_sentiment
from backend.news.source_reliability import get_source_reliability, get_all_tracked_sources, evaluate_matured_source_readings
from backend.forecasting.diversification import select_diversified_picks
from backend.forecasting.best_pick import get_current_best_pick
import backend.forecasting.tcn_service as tcn_service
from backend.forecasting.tcn_search import (
    run_search_iteration, get_best_config as get_tcn_best_config, get_trial_history as get_tcn_trial_history,
)
from backend.scheduler import get_scheduler_status, trigger_universe_seed_now, get_universe_seed_status
import backend.portfolio.service as portfolio_service
import backend.alerts.service as alerts_service
from backend.alerts.exit_risk import assess_exit_risk

router = APIRouter(prefix="/api")

# A handful of major indexes/benchmarks the dashboard always shows,
# in addition to whatever the user has watchlisted.
MARKET_BENCHMARKS = ["SPY", "QQQ", "DIA", "IWM", "^VIX"]


def _clean_for_json(value):
    """Replace NaN/inf with None so FastAPI's JSON encoder doesn't choke."""
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    records = df.replace({np.nan: None}).to_dict(orient="records")
    return records


@router.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


# --------------------------------------------------------------------- search
@router.get("/search")
def search_ticker(q: str = Query(..., min_length=1)):
    result = market_data_service.search(q)
    return {
        "query": q,
        "results": result.data or [],
        "meta": result.meta(),
    }


# --------------------------------------------------------------------- quote
@router.get("/stock/{ticker}/quote")
def get_quote(ticker: str):
    result = market_data_service.get_quote(ticker)
    if not result.success:
        return {
            "ticker": ticker.upper(),
            "quote": None,
            "meta": result.meta(),
            "warning": "Live quote unavailable. Showing no data rather than a fabricated price.",
        }
    # Defense in depth: get_quote() already refuses to report success on
    # a NaN price, but this sanitizes every field anyway — the same
    # pattern used for indicators/exit-risk — so a NaN anywhere else in
    # the quote (e.g. day_high on a partial data gap) can never crash
    # JSON serialization for the whole page requesting it.
    clean_quote = {k: _clean_for_json(v) for k, v in result.data.items()}
    return {
        "ticker": ticker.upper(), "quote": clean_quote, "meta": result.meta(),
        "provider_comparison": getattr(result, "provider_comparison", None),
    }


# ---------------------------------------------------------------- historical
@router.get("/stock/{ticker}/historical")
def get_historical(ticker: str, period: str = "1y", interval: str = "1d"):
    result = market_data_service.get_historical(ticker, period, interval)
    if not result.success or result.data is None or result.data.empty:
        raise HTTPException(
            status_code=502,
            detail=f"No historical data available for {ticker.upper()} "
                    f"({result.error or 'all providers failed'}).",
        )
    return {
        "ticker": ticker.upper(),
        "period": period,
        "interval": interval,
        "bars": _df_to_records(result.data),
        "meta": result.meta(),
        "provider_comparison": getattr(result, "provider_comparison", None),
    }


# ----------------------------------------------------------------- indicators
@router.get("/stock/{ticker}/indicators")
def get_indicators(ticker: str, period: str = "1y"):
    result = market_data_service.get_historical(ticker, period, "1d")
    if not result.success or result.data is None or result.data.empty:
        raise HTTPException(
            status_code=502,
            detail=f"No historical data available for {ticker.upper()} "
                    f"({result.error or 'all providers failed'}).",
        )

    enriched = compute_all_indicators(result.data)
    structure = detect_price_structure(enriched)
    latest = enriched.iloc[-1]
    interpretation = interpret_technicals(latest, structure)

    return {
        "ticker": ticker.upper(),
        "period": period,
        "bars": _df_to_records(enriched),
        "latest": {k: _clean_for_json(v) for k, v in latest.to_dict().items()},
        "structure": structure,
        "provider_comparison": getattr(result, "provider_comparison", None),
        "interpretation": interpretation,
        "meta": result.meta(),
    }


# ------------------------------------------------------------------ watchlist
@router.get("/watchlist")
def list_watchlist():
    with db_cursor() as cur:
        cur.execute("SELECT ticker, added_at, notes FROM watchlist ORDER BY added_at ASC")
        rows = [dict(r) for r in cur.fetchall()]
    return {"watchlist": rows}


@router.post("/watchlist/{ticker}")
def add_to_watchlist(ticker: str, notes: str = ""):
    ticker = ticker.upper()
    with db_cursor() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO watchlist (ticker, added_at, notes) VALUES (?, ?, ?)",
            (ticker, datetime.now(timezone.utc).isoformat(), notes),
        )
    return {"ticker": ticker, "added": True}


@router.delete("/watchlist/{ticker}")
def remove_from_watchlist(ticker: str):
    ticker = ticker.upper()
    with db_cursor() as cur:
        cur.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
    return {"ticker": ticker, "removed": True}


# ------------------------------------------------------------------ fundamentals
@router.get("/stock/{ticker}/fundamentals")
def get_fundamentals(ticker: str, force_refresh: bool = False):
    result = fundamentals_service.get_fundamentals(ticker, force_refresh=force_refresh)
    if not result.success or not result.data:
        return {
            "ticker": ticker.upper(),
            "fundamentals": None,
            "score": None,
            "meta": result.meta(),
            "warning": "Fundamentals unavailable for this ticker right now.",
        }
    score = compute_fundamental_score(result.data)
    return {
        "ticker": ticker.upper(),
        "fundamentals": result.data,
        "score": score,
        "meta": result.meta(),
    }


# ------------------------------------------------------------------------ news
@router.get("/stock/{ticker}/news")
def get_stock_news(ticker: str, limit: int = 15):
    # Use a cached company name if we have one, to help the general-market
    # RSS matching catch headlines that use the company name rather than
    # the ticker symbol.
    company_name = None
    cached_fundamentals = fundamentals_service.get_fundamentals(ticker) if fundamentals_service.has_cached(ticker) else None
    if cached_fundamentals and cached_fundamentals.success and cached_fundamentals.data:
        company_name = cached_fundamentals.data.get("short_name")

    result = news_service.get_analyzed_news(ticker, limit=limit, company_name=company_name)
    source_diagnostics = getattr(result, "source_diagnostics", [])
    fallback_used = getattr(result, "fallback_used", False)
    if not result.success or not result.data:
        return {
            "ticker": ticker.upper(),
            "items": [],
            "meta": result.meta(),
            "warning": "No news available for this ticker right now.",
            "source_diagnostics": source_diagnostics,
            "fallback_used": fallback_used,
        }
    return {
        "ticker": ticker.upper(),
        "items": result.data,
        "meta": result.meta(),
        "source_diagnostics": source_diagnostics,
        "fallback_used": fallback_used,
    }


@router.get("/news/sentiment-accuracy")
def get_news_accuracy(ticker: Optional[str] = None):
    """The honest answer to 'how accurate is this system' — a live,
    growing accuracy record built forward from real predictions, not a
    retroactive claim (we can't retroactively validate against a
    historical news archive we don't have — see
    backend/news/accuracy_tracker.py)."""
    return get_news_sentiment_accuracy(ticker=ticker.upper() if ticker else None)


@router.post("/news/sentiment-accuracy/evaluate-now")
def evaluate_news_accuracy_now(ticker: Optional[str] = None):
    return evaluate_matured_news_sentiment(ticker=ticker.upper() if ticker else None)


@router.get("/news/source-reliability")
def get_all_source_reliability():
    """Every source that's had at least one real reading logged, with
    its own tracked reliability — the actual data behind 'which
    sources actually deserve trust,' built forward from real outcomes,
    same honesty principle as the aggregate accuracy tracker above."""
    sources = get_all_tracked_sources()
    return {"sources": [get_source_reliability(name) for name in sources]}


@router.post("/news/source-reliability/evaluate-now")
def evaluate_source_reliability_now():
    return evaluate_matured_source_readings()


# --------------------------------------------------------------------- sec
@router.get("/stock/{ticker}/sec/filings")
def get_sec_filings(ticker: str, forms: Optional[str] = None):
    result = sec_service.get_filings(ticker)
    if not result.success:
        return {
            "ticker": ticker.upper(),
            "filings": [],
            "meta": result.meta(),
            "warning": "SEC filings unavailable for this ticker right now.",
        }
    filings = result.data
    if forms:
        wanted = {f.strip() for f in forms.split(",")}
        filings = [f for f in filings if f["form"] in wanted]
    return {"ticker": ticker.upper(), "filings": filings, "meta": result.meta()}


@router.get("/stock/{ticker}/sec/financials")
def get_sec_financials(ticker: str, force_refresh: bool = False):
    result = sec_service.get_financial_trend(ticker, force_refresh=force_refresh)
    if not result["success"]:
        return {
            "ticker": ticker.upper(),
            "trend": None,
            "warning": result["error"] or "SEC financial data unavailable for this ticker.",
        }
    return {
        "ticker": ticker.upper(),
        "trend": result["trend"],
        "meta": result.get("meta"),
    }


# ---------------------------------------------------------------------- market
@router.get("/market/regime")
def get_market_regime():
    return classify_market_regime()


@router.get("/market/sectors")
def get_sector_rankings():
    return rank_sectors()


# -------------------------------------------------------------------- scanner
@router.get("/scanner")
def scan_stocks(
    min_market_cap: Optional[float] = None,
    max_market_cap: Optional[float] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_volume: Optional[float] = None,
    min_relative_volume: Optional[float] = None,
    min_rsi: Optional[float] = None,
    max_rsi: Optional[float] = None,
    min_return_1d: Optional[float] = None,
    min_return_1w: Optional[float] = None,
    min_return_1m: Optional[float] = None,
    min_return_3m: Optional[float] = None,
    min_return_1y: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    above_sma50: Optional[bool] = None,
    above_sma200: Optional[bool] = None,
    limit: int = 50,
):
    filters = ScanFilters(
        min_market_cap=min_market_cap, max_market_cap=max_market_cap,
        min_price=min_price, max_price=max_price,
        min_volume=min_volume, min_relative_volume=min_relative_volume,
        min_rsi=min_rsi, max_rsi=max_rsi,
        min_return_1d=min_return_1d, min_return_1w=min_return_1w,
        min_return_1m=min_return_1m, min_return_3m=min_return_3m, min_return_1y=min_return_1y,
        max_pe=max_pe, min_revenue_growth=min_revenue_growth,
        above_sma50=above_sma50, above_sma200=above_sma200,
    )
    try:
        return run_scan(filters, limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Scanner failed: {exc}")


# ------------------------------------------------------------------- dashboard
@router.get("/dashboard")
def dashboard():
    with db_cursor() as cur:
        cur.execute("SELECT ticker FROM watchlist ORDER BY added_at ASC")
        watchlist_tickers = [r["ticker"] for r in cur.fetchall()]

    def snapshot(ticker: str) -> dict:
        quote_result = market_data_service.get_quote(ticker)
        if not quote_result.success:
            return {"ticker": ticker, "quote": None, "meta": quote_result.meta()}
        # Sanitize every field — same defense-in-depth pattern as the
        # standalone quote endpoint. This is the exact function whose
        # unsanitized NaN price crashed the entire Dashboard on a
        # market-holiday data gap (root cause now also fixed directly
        # in market_data_service.get_quote()).
        q = {k: _clean_for_json(v) for k, v in quote_result.data.items()}
        change_pct = None
        if q.get("price") is not None and q.get("previous_close"):
            change_pct = (q["price"] - q["previous_close"]) / q["previous_close"] * 100
        return {
            "ticker": ticker,
            "quote": q,
            "change_pct": round(change_pct, 2) if change_pct is not None else None,
            "meta": quote_result.meta(),
        }

    benchmarks = [snapshot(t) for t in MARKET_BENCHMARKS]
    watchlist = [snapshot(t) for t in watchlist_tickers]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmarks": benchmarks,
        "watchlist": watchlist,
    }


# -------------------------------------------------------------- predictions
@router.get("/stock/{ticker}/predict")
def predict_stock(ticker: str, horizon: int = 5, force_retrain: bool = False, risk_profile: str = "moderate"):
    if horizon not in (1, 5, 20):
        raise HTTPException(status_code=422, detail="horizon must be one of 1, 5, or 20 (trading days)")
    from backend.forecasting.risk import RISK_PROFILES
    if risk_profile not in RISK_PROFILES:
        raise HTTPException(status_code=422, detail=f"risk_profile must be one of {list(RISK_PROFILES)}")
    try:
        result = prediction_service.get_or_train(ticker, horizon, force_retrain=force_retrain, risk_profile=risk_profile)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction pipeline failed: {exc}")

    if not result.get("success"):
        raise HTTPException(status_code=422, detail=result.get("error", "Prediction failed"))
    return result


@router.get("/stock/{ticker}/predictions/history")
def get_prediction_history(ticker: str, limit: int = 30):
    with db_cursor() as cur:
        cur.execute(
            """SELECT id, created_at, horizon_days, expected_return, confidence, signal
               FROM predictions WHERE ticker = ? ORDER BY created_at DESC LIMIT ?""",
            (ticker.upper(), limit),
        )
        predictions = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """SELECT p.id as prediction_id, r.actual_return, r.was_correct, r.evaluated_at
               FROM predictions p JOIN prediction_results r ON r.prediction_id = p.id
               WHERE p.ticker = ?""",
            (ticker.upper(),),
        )
        results_by_id = {r["prediction_id"]: dict(r) for r in cur.fetchall()}

    for p in predictions:
        p["result"] = results_by_id.get(p["id"])

    return {"ticker": ticker.upper(), "predictions": predictions}


@router.post("/predictions/evaluate")
def evaluate_predictions(ticker: Optional[str] = None):
    result = prediction_service.evaluate_matured_predictions(ticker)
    return result


@router.get("/predictions/accuracy")
def get_prediction_accuracy(ticker: Optional[str] = None):
    return {"ticker": ticker.upper() if ticker else "all", "accuracy_by_horizon": prediction_service.get_prediction_accuracy(ticker)}


@router.get("/scheduler/status")
def scheduler_status():
    return get_scheduler_status()


@router.post("/discovery/seed-universe-now")
def seed_universe_now(horizon_days: int = 5):
    """Explicit, on-demand alternative to the regular gradual discovery
    sweep (8 tickers every 3 hours): trains real predictions across the
    ENTIRE scanner universe at once, in the background. This is the
    concrete answer to 'how do I get real evidence accumulating faster
    and broader' — it doesn't fabricate anything, it just starts more
    real clocks ticking sooner across more tickers, all at once, rather
    than one at a time as you happen to check them."""
    return trigger_universe_seed_now(horizon_days=horizon_days)


@router.get("/discovery/seed-universe-status")
def seed_universe_status():
    return get_universe_seed_status()


@router.get("/models/availability")
def get_model_availability():
    return {
        "sklearn_baseline": True,  # always available, core dependency
        "xgboost": XGBOOST_AVAILABLE,
        "lstm_pytorch": TORCH_AVAILABLE,
        "notes": [
            note for note in [
                None if XGBOOST_AVAILABLE else "XGBoost not installed — run: pip install xgboost",
                None if TORCH_AVAILABLE else "PyTorch not installed — run: pip install torch (may not have a wheel for very new Python versions yet)",
            ] if note
        ],
    }


@router.get("/stock/{ticker}/models")
def get_model_versions(ticker: str):
    with db_cursor() as cur:
        cur.execute(
            """SELECT model_type, trained_at, horizon_days, validation_metrics_json, test_metrics_json
               FROM model_versions WHERE ticker = ? ORDER BY trained_at DESC LIMIT 20""",
            (ticker.upper(),),
        )
        rows = [dict(r) for r in cur.fetchall()]

    import json as _json
    for r in rows:
        r["validation_metrics"] = _json.loads(r.pop("validation_metrics_json") or "{}")
        r["trading_stats"] = _json.loads(r.pop("test_metrics_json") or "{}")

    return {"ticker": ticker.upper(), "model_versions": rows}


# -------------------------------------------------------------- portfolio
@router.get("/portfolio")
def get_portfolio():
    return portfolio_service.get_portfolio_summary()


@router.post("/portfolio")
def add_portfolio_holding(ticker: str, shares: float, entry_price: float,
                            entry_date: Optional[str] = None, notes: str = ""):
    if shares <= 0 or entry_price <= 0:
        raise HTTPException(status_code=422, detail="shares and entry_price must both be positive")
    holding_id = portfolio_service.add_holding(ticker, shares, entry_price, entry_date, notes)
    return {"id": holding_id, "ticker": ticker.upper(), "added": True}


@router.delete("/portfolio/{holding_id}")
def delete_portfolio_holding(holding_id: int):
    removed = portfolio_service.remove_holding(holding_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No holding with id {holding_id}")
    return {"id": holding_id, "removed": True}


@router.get("/portfolio/{ticker}/exit-risk")
def get_exit_risk(ticker: str):
    result = market_data_service.get_historical(ticker, period="6mo", interval="1d")
    if not result.success or result.data is None or result.data.empty:
        raise HTTPException(status_code=502, detail=f"No historical data available for {ticker.upper()}")
    enriched = compute_all_indicators(result.data)
    structure = detect_price_structure(enriched)
    latest = {k: _clean_for_json(v) for k, v in enriched.iloc[-1].to_dict().items()}

    news_items = None
    try:
        news_result = news_service.get_analyzed_news(ticker, limit=5)
        if news_result.success:
            news_items = news_result.data
    except Exception:  # noqa: BLE001
        pass  # exit-risk assessment works fine without news — it's a supplementary signal, not required

    risk = assess_exit_risk(latest, structure, recent_news_items=news_items)
    return {"ticker": ticker.upper(), **risk}


# ----------------------------------------------------------------- alerts
@router.get("/alerts")
def get_alerts_endpoint(unread_only: bool = False, limit: int = 50):
    return {"alerts": alerts_service.get_alerts(unread_only=unread_only, limit=limit),
            "unread_count": alerts_service.count_unread_alerts()}


@router.post("/alerts/{alert_id}/read")
def mark_alert_read_endpoint(alert_id: int):
    marked = alerts_service.mark_alert_read(alert_id)
    if not marked:
        raise HTTPException(status_code=404, detail=f"No alert with id {alert_id}")
    return {"id": alert_id, "read": True}


@router.post("/alerts/read-all")
def mark_all_alerts_read_endpoint():
    count = alerts_service.mark_all_alerts_read()
    return {"marked_read": count}


@router.post("/alerts/check-now")
def run_alert_checks_now():
    """Manual trigger, mirroring the scheduler's automatic checks —
    useful right after adding a new watchlist ticker or portfolio
    holding, rather than waiting for the next scheduled run."""
    return alerts_service.run_all_alert_checks()


# ------------------------------------------------------------- backtesting
@router.get("/stock/{ticker}/backtest")
def get_backtest(ticker: str, horizon: int = 5):
    if horizon not in (1, 5, 20):
        raise HTTPException(status_code=422, detail="horizon must be one of 1, 5, or 20 (trading days)")
    try:
        result = run_backtest(ticker, horizon)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")

    if not result.get("success"):
        raise HTTPException(status_code=422, detail=result.get("error", "Backtest failed"))
    return result


@router.get("/backtests/history")
def get_backtest_history(ticker: Optional[str] = None, limit: int = 20):
    import json as _json
    query = "SELECT id, created_at, ticker, strategy_json, start_date, end_date, results_json FROM backtests"
    params: tuple = ()
    if ticker:
        query += " WHERE ticker = ?"
        params = (ticker.upper(),)
    query += " ORDER BY created_at DESC LIMIT ?"
    with db_cursor() as cur:
        cur.execute(query, params + (limit,))
        rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["strategy"] = _json.loads(r.pop("strategy_json") or "{}")
        r["stats"] = _json.loads(r.pop("results_json") or "{}")
    return {"backtests": rows}


# ---------------------------------------------------------------- settings
@router.get("/settings")
def get_settings():
    return settings_service.get_all_settings()


@router.post("/settings")
def update_settings(key: str, value: str):
    valid_keys = set(settings_service.DEFAULTS.keys())
    if key not in valid_keys:
        raise HTTPException(status_code=422, detail=f"Unknown setting '{key}'. Valid keys: {sorted(valid_keys)}")
    settings_service.set_setting(key, value)
    return {"key": key, "value": value, "updated": True}


@router.get("/settings/cache-stats")
def get_cache_stats():
    return settings_service.get_cache_stats()


@router.post("/settings/clear-cache")
def clear_cache():
    cleared = settings_service.clear_historical_cache()
    return {"cleared_rows": cleared}


# -------------------------------------------------------------------- chat
class ChatRequest(BaseModel):
    messages: list[dict]
    context: Optional[dict] = None


@router.get("/chat/status")
def get_chat_status():
    return {"configured": chat_service.is_configured()}


@router.post("/chat")
def post_chat_message(req: ChatRequest):
    result = chat_service.send_chat_message(req.messages, req.context)
    if not result.get("success"):
        raise HTTPException(status_code=503, detail=result.get("error", "Chat failed"))
    return result


# ------------------------------------------------------------ suggestions
@router.get("/suggestions")
def get_suggestions(limit: int = 6):
    return get_stock_suggestions(limit=limit)


@router.post("/suggestions/refresh")
def refresh_suggestions():
    """Manually triggers predictions for your tracked tickers (respects
    the existing 24h freshness cache, so already-fresh ones are
    essentially free) — useful right after adding new watchlist
    tickers, rather than waiting for the next scheduled background run."""
    return refresh_tracked_ticker_predictions()


@router.get("/discovery")
def get_discovery(limit: int = 6):
    """The actual 'suggest new stock investments' feature — ranked over
    the Scanner's liquid-stock universe, excluding anything already in
    your watchlist/portfolio (that's what /suggestions is for)."""
    return get_discovery_suggestions(limit=limit)


@router.get("/discovery/coverage")
def get_discovery_coverage_endpoint():
    """How much of the discovery universe currently has a fresh
    prediction — honest visibility into 'how filled-in is this right
    now', since coverage builds up gradually rather than all at once."""
    return get_discovery_coverage()


@router.post("/discovery/sweep")
def trigger_discovery_sweep():
    """Manually triggers one batch of the gradual discovery sweep —
    useful if you don't want to wait for the next scheduled run."""
    return run_discovery_sweep()


@router.get("/discovery/diversified")
def get_diversified_discovery(limit: int = 5, correlation_penalty: float = 0.5):
    """Operationalizes a real quant finance consensus: low correlation
    across many smaller bets matters more than one 'best' signal.
    Selects from the same bullish Discovery candidates, but trades off
    raw signal strength against actual historical return correlation —
    a strong-but-redundant pick can lose out to a genuinely
    diversifying one. Real network cost (fetches price history to
    compute correlation), so this is its own endpoint rather than
    baked into the default /discovery response."""
    discovery = get_discovery_suggestions(limit=max(limit * 3, 10))  # wider pool to choose diversified picks from
    return select_diversified_picks(discovery["bullish"], limit=limit, correlation_penalty=correlation_penalty)


# ----------------------------------------------------------------- TCN
@router.get("/tcn/status")
def get_tcn_status():
    return {"available": tcn_service.is_available()}


@router.get("/tcn/{ticker}/best-config")
def get_tcn_best_config_endpoint(ticker: str, horizon: int = 5):
    """The current best config found by the ongoing search, plus its
    search-period score and — once evaluated — its one-time holdout
    score alongside it, so the gap between them (if any) is visible
    rather than hidden."""
    best = get_tcn_best_config(ticker, horizon)
    if best is None:
        return {"found": False, "note": "No search trials yet for this ticker/horizon — trigger a search or wait for the background job."}
    return {"found": True, **best, "config": best["config"].__dict__}


@router.get("/tcn/{ticker}/trial-history")
def get_tcn_trial_history_endpoint(ticker: str, horizon: int = 5, limit: int = 50):
    """Every trial ever run, in full — nothing silently discarded, so
    the search process itself stays auditable."""
    history = get_tcn_trial_history(ticker, horizon, limit=limit)
    return {"trials": [{**h, "config": h["config"].__dict__} for h in history]}


@router.post("/tcn/{ticker}/search")
def trigger_tcn_search(ticker: str, horizon: int = 5, n_candidates: int = 2):
    """Manually triggers one search iteration — real model training
    per candidate, so this can take a while. Useful if you don't want
    to wait for the next scheduled background run."""
    if not tcn_service.is_available():
        raise HTTPException(status_code=503, detail="PyTorch is not installed — TCN search is unavailable.")
    return run_search_iteration(ticker, horizon_days=horizon, n_candidates=n_candidates)


@router.get("/tcn/{ticker}/comparison")
def get_tcn_comparison(ticker: str, horizon: int = 5):
    """The 5-way comparison: buy-and-hold vs raw model signal vs
    +dead-zone vs +vol-scaling vs +vol-scaling+exposure-cap, all using
    the current best config found by the ongoing search (or sensible
    defaults if nothing's been searched yet for this ticker)."""
    if not tcn_service.is_available():
        raise HTTPException(status_code=503, detail="PyTorch is not installed — the TCN module is unavailable.")
    result = tcn_service.run_comparison_for_ticker(ticker, horizon)
    if not result.get("success"):
        raise HTTPException(status_code=422, detail=result.get("error", "TCN comparison failed"))
    return result


@router.get("/discovery/best-pick")
def get_best_pick():
    """The autonomous 'what does the app currently favor most' read —
    always available on demand, kept fresh by the existing background
    scheduler (discovery sweep + suggestions refresh) regardless of
    whether anyone is actively using the app. Returns null honestly
    when nothing bullish is currently showing anywhere, rather than
    forcing a pick that doesn't exist."""
    result = get_current_best_pick()
    return {"best_pick": result}
