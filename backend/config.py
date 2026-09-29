"""
Central app configuration, loaded from .env.

Every setting has a safe default so the app runs with zero
configuration — API keys are purely additive.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of current working dir.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "stock_ai.db"))

    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    TWELVE_DATA_API_KEY: str = os.getenv("TWELVE_DATA_API_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")  # powers the AI Assistant chat — optional, chat is disabled without it

    POLL_INTERVAL_SECONDS: int = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))

    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Historical price cache freshness, in hours, before we re-hit the provider.
    HISTORICAL_CACHE_TTL_HOURS: int = int(os.getenv("HISTORICAL_CACHE_TTL_HOURS", "6"))


settings = Settings()

# Make sure the data directory exists on import.
Path(settings.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
