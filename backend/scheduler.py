"""
Background scheduler — this is what makes prediction accuracy tracking
(and therefore the adaptive ensemble weighting in
backend/forecasting/adaptive.py) update automatically as the app runs,
rather than only when someone manually clicks "Evaluate matured
predictions."

Scope, stated plainly: this checks for and evaluates matured
predictions periodically. It does NOT automatically retrain models in
the background (that would mean re-running expensive walk-forward
validation on a timer for every ticker anyone has ever looked at,
which isn't a reasonable thing for a personal app to do to your
laptop's CPU unattended) — model retraining still only happens when
you actually request a prediction and the cached one has gone stale.
Full background monitoring/alerting is Phase 4; this is a deliberately
narrow, useful slice of it pulled forward now.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("stock_ai.scheduler")

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except Exception:  # noqa: BLE001
    # Deliberately broad, not just ImportError — same lesson learned
    # from the xgboost/torch optional-dependency crashes: a partially
    # broken install of an optional package must never be allowed to
    # take down the whole app at startup.
    SCHEDULER_AVAILABLE = False

_scheduler: Optional["BackgroundScheduler"] = None
_status = {"last_run_at": None, "last_result": None, "run_count": 0}
_alert_status = {"last_run_at": None, "last_result": None, "run_count": 0}
_suggestions_status = {"last_run_at": None, "last_result": None, "run_count": 0}
_discovery_status = {"last_run_at": None, "last_result": None, "run_count": 0}
_tcn_search_status = {"last_run_at": None, "last_result": None, "run_count": 0}
_universe_seed_status = {"running": False, "started_at": None, "finished_at": None,
                          "total_tickers": 0, "completed_tickers": 0, "last_result": None}


def _run_universe_seed_job(horizon_days: int) -> None:
    """Runs in a background thread, not the request thread — the
    actual worker behind trigger_universe_seed_now(). Deliberately
    NOT paced/batched the way the regular scheduled discovery sweep
    is: this is the on-demand, explicit-opt-in version for when
    someone has decided the one-time cost is worth it, specifically to
    get real prediction and news evidence accumulating across the
    whole tracked universe as fast as possible, rather than the ~2
    days the regular 8-tickers-per-3-hours pace would take."""
    from backend.forecasting.suggestions import SCANNER_UNIVERSE, _get_tracked_tickers
    from backend.forecasting.service import prediction_service

    tracked = set(_get_tracked_tickers())
    universe = sorted(set(SCANNER_UNIVERSE) - tracked)

    _universe_seed_status["running"] = True
    _universe_seed_status["started_at"] = datetime.now(timezone.utc).isoformat()
    _universe_seed_status["finished_at"] = None
    _universe_seed_status["total_tickers"] = len(universe)
    _universe_seed_status["completed_tickers"] = 0

    refreshed, failed = [], []
    for ticker in universe:
        try:
            # get_or_train() already respects the 24h freshness cache —
            # re-running this later is cheap for anything already fresh,
            # not a redundant full retrain every time.
            result = prediction_service.get_or_train(ticker, horizon_days)
            if result.get("success"):
                refreshed.append(ticker)
            else:
                logger.info(f"[UNIVERSE SEED] {ticker} did not succeed: {result.get('error')}")
                failed.append(ticker)
        except Exception as exc:  # noqa: BLE001 — one ticker failing must not stop the whole seed run
            logger.info(f"[UNIVERSE SEED] Failed to train {ticker}: {exc}")
            failed.append(ticker)
        _universe_seed_status["completed_tickers"] += 1

    _universe_seed_status["running"] = False
    _universe_seed_status["finished_at"] = datetime.now(timezone.utc).isoformat()
    _universe_seed_status["last_result"] = {"refreshed": refreshed, "failed": failed}
    logger.info(f"[UNIVERSE SEED] Done — {len(refreshed)} refreshed, {len(failed)} failed")


def trigger_universe_seed_now(horizon_days: int = 5) -> dict:
    """Starts seeding real predictions across the entire scanner
    universe in a background thread, returning immediately rather than
    blocking the request for what could be many minutes — up to ~120
    real training runs. Refuses to start a second run on top of one
    already in progress rather than silently doubling the work."""
    if _universe_seed_status["running"]:
        return {"started": False, "reason": "A universe seed is already running.", "status": get_universe_seed_status()}

    thread = threading.Thread(target=_run_universe_seed_job, args=(horizon_days,), daemon=True)
    thread.start()
    return {"started": True, "status": get_universe_seed_status()}


def get_universe_seed_status() -> dict:
    return dict(_universe_seed_status)


def _run_evaluation_job() -> None:
    from backend.forecasting.service import prediction_service
    from backend.news.accuracy_tracker import evaluate_matured_news_sentiment
    try:
        result = prediction_service.evaluate_matured_predictions()
        _status["last_run_at"] = datetime.now(timezone.utc).isoformat()
        _status["last_result"] = result
        _status["run_count"] += 1
        logger.info(f"[SCHEDULER] Auto-evaluated matured predictions: {result}")
    except Exception as exc:  # noqa: BLE001 — a failed scheduled run must not kill the scheduler itself
        logger.info(f"[SCHEDULER] Evaluation job failed: {exc}")

    try:
        news_result = evaluate_matured_news_sentiment()
        logger.info(f"[SCHEDULER] Auto-evaluated matured news sentiment reads: {news_result}")
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[SCHEDULER] News sentiment evaluation job failed: {exc}")

    try:
        from backend.news.source_reliability import evaluate_matured_source_readings
        source_result = evaluate_matured_source_readings()
        logger.info(f"[SCHEDULER] Auto-evaluated matured source reliability reads: {source_result}")
    except Exception as exc:  # noqa: BLE001 — a real gap found and fixed: this evaluator existed but was
        # never actually wired into any scheduled job, meaning source reliability data would sit pending
        # forever unless someone manually hit the evaluate-now endpoint. Now runs on the same cadence as
        # the other evaluations, isolated the same way so a failure here can't take down the others.
        logger.info(f"[SCHEDULER] Source reliability evaluation job failed: {exc}")


def _run_alert_checks_job() -> None:
    from backend.alerts.service import run_all_alert_checks
    try:
        result = run_all_alert_checks()
        _alert_status["last_run_at"] = datetime.now(timezone.utc).isoformat()
        _alert_status["last_result"] = result
        _alert_status["run_count"] += 1
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[SCHEDULER] Alert check job failed: {exc}")


def _run_suggestions_refresh_job() -> None:
    # Deliberately NOT run immediately on startup (see start_scheduler)
    # and on a much longer interval than the other jobs — this can
    # mean genuinely training several real ML models back to back,
    # which is real CPU work, not a quick DB check like the other two
    # jobs. get_or_train() already skips anything trained within the
    # last 24h, so most runs of this are cheap; only brand-new tracked
    # tickers actually trigger real training.
    from backend.forecasting.suggestions import refresh_tracked_ticker_predictions
    try:
        result = refresh_tracked_ticker_predictions()
        _suggestions_status["last_run_at"] = datetime.now(timezone.utc).isoformat()
        _suggestions_status["last_result"] = result
        _suggestions_status["run_count"] += 1
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[SCHEDULER] Suggestions refresh job failed: {exc}")


def _run_discovery_sweep_job() -> None:
    # Same caution as the suggestions job (real training, not a quick
    # check) — a small batch per run, on a slow cadence, so coverage of
    # the discovery universe builds up gradually over the first day or
    # so of running the app instead of hammering the CPU all at once.
    from backend.forecasting.suggestions import run_discovery_sweep
    try:
        result = run_discovery_sweep()
        _discovery_status["last_run_at"] = datetime.now(timezone.utc).isoformat()
        _discovery_status["last_result"] = result
        _discovery_status["run_count"] += 1
        logger.info(f"[SCHEDULER] Discovery sweep: {result}")
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[SCHEDULER] Discovery sweep job failed: {exc}")


def _run_tcn_search_job() -> None:
    # Heavier than every other background job here — each candidate
    # config means a full TCN training run, per ticker. Deliberately
    # the most conservative batch size of anything in this scheduler:
    # at most 2 tracked tickers, 2 new candidates each, per run. No-ops
    # entirely (cheaply) if PyTorch isn't installed.
    from backend.forecasting.suggestions import _get_tracked_tickers
    from backend.forecasting.tcn_search import run_search_iteration
    try:
        tickers = _get_tracked_tickers()[:2]
        results = {ticker: run_search_iteration(ticker, horizon_days=5, n_candidates=2) for ticker in tickers}
        _tcn_search_status["last_run_at"] = datetime.now(timezone.utc).isoformat()
        _tcn_search_status["last_result"] = results
        _tcn_search_status["run_count"] += 1
        logger.info(f"[SCHEDULER] TCN search iteration: {results}")
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[SCHEDULER] TCN search job failed: {exc}")


def start_scheduler(interval_minutes: int = 60, alert_interval_minutes: int = 15,
                     suggestions_interval_minutes: int = 240, discovery_interval_minutes: int = 180,
                     tcn_search_interval_minutes: int = 360) -> bool:
    """Returns True if the scheduler actually started. Safe to call
    even if apscheduler isn't installed — just no-ops and logs why.
    Alerts check more frequently than prediction evaluation (15 min vs
    60 min) since detecting a technical/risk change promptly is the
    whole point, while prediction outcomes only mature over days.
    Suggestions and discovery refresh on long intervals since both can
    mean real model training; TCN search is the heaviest of all
    (multiple real training runs per pass), so it gets the longest
    interval — every 6 hours by default."""
    global _scheduler
    if not SCHEDULER_AVAILABLE:
        logger.info("[SCHEDULER] apscheduler not installed — automatic prediction evaluation and alerts disabled. "
                    "Run: pip install apscheduler. You can still evaluate manually from the Models page.")
        return False
    if _scheduler is not None:
        return True

    _scheduler = BackgroundScheduler(daemon=True)
    # `next_run_time=now` makes each job check once immediately on
    # startup too, rather than making you wait a full interval —
    # except the suggestions/discovery/tcn-search jobs, which
    # deliberately wait for their first normal interval rather than
    # potentially spawning several real training runs the instant the
    # app starts up.
    _scheduler.add_job(_run_evaluation_job, "interval", minutes=interval_minutes,
                        next_run_time=datetime.now(), id="evaluate_matured_predictions")
    _scheduler.add_job(_run_alert_checks_job, "interval", minutes=alert_interval_minutes,
                        next_run_time=datetime.now(), id="run_alert_checks")
    _scheduler.add_job(_run_suggestions_refresh_job, "interval", minutes=suggestions_interval_minutes,
                        id="refresh_suggestions")
    _scheduler.add_job(_run_discovery_sweep_job, "interval", minutes=discovery_interval_minutes,
                        id="discovery_sweep")
    _scheduler.add_job(_run_tcn_search_job, "interval", minutes=tcn_search_interval_minutes,
                        id="tcn_search")
    _scheduler.start()
    logger.info(f"[SCHEDULER] Started — predictions every {interval_minutes}min, alerts every "
                f"{alert_interval_minutes}min, suggestions every {suggestions_interval_minutes}min, "
                f"discovery every {discovery_interval_minutes}min, TCN search every {tcn_search_interval_minutes}min")
    return True


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def get_scheduler_status() -> dict:
    return {
        "available": SCHEDULER_AVAILABLE,
        "running": _scheduler is not None and _scheduler.running,
        "run_count": _status["run_count"],
        "last_run_at": _status["last_run_at"],
        "last_result": _status["last_result"],
        "alerts": {
            "run_count": _alert_status["run_count"],
            "last_run_at": _alert_status["last_run_at"],
            "last_result": _alert_status["last_result"],
        },
        "suggestions": {
            "run_count": _suggestions_status["run_count"],
            "last_run_at": _suggestions_status["last_run_at"],
            "last_result": _suggestions_status["last_result"],
        },
        "discovery": {
            "run_count": _discovery_status["run_count"],
            "last_run_at": _discovery_status["last_run_at"],
            "last_result": _discovery_status["last_result"],
        },
        "tcn_search": {
            "run_count": _tcn_search_status["run_count"],
            "last_run_at": _tcn_search_status["last_run_at"],
            "last_result": _tcn_search_status["last_result"],
        },
    }
