from __future__ import annotations

import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as api_router
from backend.database.db import init_db
from backend.scheduler import start_scheduler, stop_scheduler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _sanitize_nan(obj: Any) -> Any:
    """Recursively replaces NaN/Infinity with None throughout any
    nested dict/list/tuple structure.

    Why this exists as a GLOBAL fix rather than a per-endpoint one: the
    same underlying bug (a market-holiday data gap leaving a NaN price
    that crashes Python's strict JSON encoder — 'Out of range float
    values are not JSON compliant: nan') showed up in two completely
    separate endpoints (the dashboard's quotes, then independently in
    Market regime and Sector rankings) because each one reads "the
    latest price" through a different code path. Patching endpoints
    one at a time as they're discovered is exactly how the second one
    got missed. This sanitizes EVERY API response automatically, so no
    endpoint — including ones not yet written — can hit this class of
    crash again."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nan(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return super().render(_sanitize_nan(content))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(
    title="Stock AI Research Terminal",
    description="Local research/prediction dashboard. Not financial advice.",
    version="0.1.0-phase1",
    lifespan=lifespan,
    default_response_class=SafeJSONResponse,
)

app.include_router(api_router)


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    # This is a local dev tool that changes often — never let the browser
    # cache the frontend, or you end up debugging a "bug" that's actually
    # just a stale cached copy of app.js.
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


# Static assets (css/js) under /static, index.html served at root.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")
