"""
Model A (baseline sklearn) and Model B (XGBoost) — spec section 9.

XGBoost is an optional dependency: if it isn't installed, XGBOOST_AVAILABLE
is False and get_available_models() simply omits it — the rest of the
app keeps working with the baseline models, and the UI is told exactly
what's unavailable rather than the app crashing on import.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:  # noqa: BLE001
    # Deliberately broad, not just ImportError: on macOS, xgboost's
    # compiled binary depends on the system OpenMP library (libomp),
    # which isn't included by default. When it's missing, `import
    # xgboost` fails with a low-level library-loading error (OSError),
    # not ImportError — a narrower except clause here would let that
    # crash propagate up and take down the whole app at startup
    # instead of just marking XGBoost as unavailable.
    XGBOOST_AVAILABLE = False


def make_linear_regression():
    return LinearRegression()


def make_random_forest():
    # n_jobs intentionally left at 1 (no parallel worker processes).
    # n_jobs=-1 was found to hang indefinitely (0% CPU, no progress,
    # no error — a true deadlock, not just slowness) on a real Mac
    # running a brand-new Python version, almost certainly a joblib/
    # loky multiprocessing compatibility issue that hasn't caught up
    # to that Python release yet. This dataset is small enough
    # (a few thousand rows, ~17 features) that single-process fitting
    # is still fast — reliability matters far more here than the
    # marginal speed a parallel backend would add.
    return RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)


def make_gradient_boosting():
    return GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)


def make_xgboost():
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed")
    # n_jobs=1, same reasoning as random_forest above — avoiding a
    # confirmed multiprocessing deadlock rather than chasing marginal
    # speed on an already-small dataset.
    return XGBRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective="reg:squarederror", n_jobs=1,
    )


def get_available_sklearn_models() -> dict:
    """Model A — always available (sklearn is a core dependency)."""
    return {
        "linear_regression": make_linear_regression,
        "random_forest": make_random_forest,
        "gradient_boosting": make_gradient_boosting,
    }


def get_available_xgboost_model() -> dict:
    """Model B — only included if xgboost is actually installed."""
    if not XGBOOST_AVAILABLE:
        return {}
    return {"xgboost": make_xgboost}


def get_xgboost_feature_importance(model, feature_names: list[str]) -> list[dict]:
    """Feature importance for XGBoost (spec section 27) — genuine
    gain-based importance from the trained model, not invented."""
    if not XGBOOST_AVAILABLE or not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    total = importances.sum()
    if total <= 0:
        return []
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return [
        {"feature": name, "importance_pct": round(float(imp) / float(total) * 100, 1)}
        for name, imp in ranked if imp > 0
    ]
