"""
AI Assistant chat (Phase 5) — lets you ask questions about the app,
how to interpret a result, or general concepts (what's a Sharpe ratio,
why does relative strength matter, etc.) in plain conversation.

Uses your own Anthropic API key (set ANTHROPIC_API_KEY in .env — never
hardcoded, never stored in the database). Without a key configured,
the chat endpoint returns a clear, actionable message instead of a
cryptic error — the rest of the app works completely normally either way.

Scope, stated honestly: this assistant can explain concepts and help
you interpret whatever data you paste in or the app attaches as
context (a prediction, a backtest result, a portfolio snapshot). It
does NOT have live tool access to query the app's data on its own —
it only sees what's explicitly included in the request. That's a
deliberate simplicity trade-off, not an oversight.
"""
from __future__ import annotations

import json
import logging

import requests

from backend.config import settings

logger = logging.getLogger("stock_ai.chat")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-5"
ANTHROPIC_VERSION = "2023-06-01"
MAX_TOKENS = 1024
MAX_HISTORY_MESSAGES = 20  # keep the request small — this is a chat helper, not a long-document tool

SYSTEM_PROMPT = """You are the built-in AI Assistant for a local Stock AI Research Terminal app. \
You help the person understand the app and interpret what it shows them.

What the app actually does (so you can explain it accurately):
- Dashboard, Watchlist, Stock Analyzer: live quotes, candlestick charts, and 40+ technical indicators (RSI, MACD, Bollinger Bands, ATR, moving averages, etc.) with plain-English interpretation.
- Fundamentals: a 0-100 score from growth, profitability, valuation, and financial health.
- SEC filings: real filing links and XBRL-based financial trend analysis.
- News: multi-source headlines with rules-based sentiment/impact scoring.
- Market: bull/bear/neutral regime classification and sector rotation ranking.
- Scanner: batch screening across ~150 liquid US stocks with preset and custom filters.
- Predictions: an ensemble of machine learning models (Linear Regression, Random Forest, Gradient Boosting, XGBoost, and optionally an LSTM neural network) trained on STATIONARY features (returns, ratios, z-scores — never raw price levels) and validated with WALK-FORWARD validation (chronological, expanding-window, never randomly shuffled — this prevents the model from ever training on data from "the future" relative to what it's being tested on). Predictions include an expected return, a confidence score, per-model vote breakdown, an out-of-distribution warning when current conditions are unlike the training history, and a live (non-trained) news cross-check.
- Risk sizing: a fractional Kelly Criterion position-size suggestion derived from the model's own real, validated trade history — never a made-up number, and it refuses to suggest anything until there's enough real track record (20+ trades).
- Portfolio: manually-entered holdings with live P&L; an Exit-Risk engine that flags compounding red flags (never recommends selling — the strongest thing it ever says is "review position").
- Alerts: a background scheduler that only notifies on a genuine material change (a signal flip, a risk escalation), never repeatedly on an unchanged condition.
- Backtesting: simulates actually trading on the model's historical out-of-sample predictions, producing an equity curve and a fair buy-and-hold comparison over the exact same period.
- Settings: default risk profile, cache management, and diagnostics.

Guidelines for how you respond:
- Explain concepts clearly and in plain language — assume the person may not have a finance or ML background, but don't talk down to them either.
- If given context data (a prediction result, a backtest result, portfolio data) in the conversation, use the SPECIFIC numbers in it when answering — don't give a generic answer when you have real data to reference.
- This app never gives financial advice, and neither should you. Explain what the data shows and what it means; never tell the person what to buy, sell, or do with their money. If asked directly for a recommendation, explain what the app's own data suggests and remind them this is informational, not advice.
- Be honest about the app's real limitations (data comes from Yahoo Finance/Stooq with no live-tick data, LSTM requires PyTorch, some tickers get "relaxed mode" with less rigorous validation due to limited history, etc.) rather than overselling what it does.
- Keep answers reasonably concise unless the person is asking for a deep explanation."""


def is_configured() -> bool:
    return bool(settings.ANTHROPIC_API_KEY)


def send_chat_message(messages: list[dict], context: dict | None = None) -> dict:
    """messages: [{"role": "user"|"assistant", "content": "..."}]
    context: optional dict (e.g. a prediction or backtest result) —
    if given, it's summarized and prepended so the assistant can
    answer with real specifics rather than a generic response."""
    if not is_configured():
        return {
            "success": False,
            "error": (
                "No Anthropic API key configured. Add ANTHROPIC_API_KEY to your .env file "
                "to enable the AI Assistant — everything else in the app works without it."
            ),
        }

    if not messages:
        return {"success": False, "error": "No message provided."}

    trimmed = messages[-MAX_HISTORY_MESSAGES:]

    api_messages = []
    if context:
        context_note = f"[Context from the app — the person is currently looking at this data:]\n{json.dumps(context, indent=2, default=str)[:4000]}"
        api_messages.append({"role": "user", "content": context_note})
        api_messages.append({"role": "assistant", "content": "Got it, I can see that data — go ahead with your question."})
    api_messages.extend(trimmed)

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": settings.ANTHROPIC_API_KEY,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": MAX_TOKENS,
                "system": SYSTEM_PROMPT,
                "messages": api_messages,
            },
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        logger.info(f"[CHAT] Request failed: {exc}")
        return {"success": False, "error": f"Couldn't reach the Anthropic API: {exc}"}

    if response.status_code != 200:
        detail = response.text[:300]
        logger.info(f"[CHAT] API error {response.status_code}: {detail}")
        return {"success": False, "error": f"Anthropic API returned an error ({response.status_code}): {detail}"}

    data = response.json()
    reply_text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")

    return {"success": True, "reply": reply_text, "usage": data.get("usage")}
