from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

from backend.data.base import DataResult, NewsDataProvider
from backend.data.news_provider import (
    CNBC_FEED_URL,
    MARKETWATCH_FEED_URL,
    GoogleNewsSearchProvider,
    NamedOutletRSSProvider,
    RSSNewsProvider,
    SeekingAlphaProvider,
    SourceDiagnostics,
    YFinanceNewsProvider,
)
from backend.database.db import db_cursor
from backend.news.sentiment import classify_source_tier, score_news_impact

logger = logging.getLogger("stock_ai.news")

_DUPLICATE_SIMILARITY_THRESHOLD = 0.82


def _normalize_headline(headline: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", headline.lower()).strip()


def _dedupe_articles(items: list[dict]) -> list[dict]:
    """Detects near-duplicate headlines reporting the same event across
    multiple sources (spec section 4/15) and folds them into one entry,
    keeping the highest-tier source and noting how many outlets covered it."""
    kept: list[dict] = []
    normalized_kept: list[str] = []

    for item in items:
        norm = _normalize_headline(item.get("headline", ""))
        match_idx = None
        for i, existing_norm in enumerate(normalized_kept):
            if SequenceMatcher(None, norm, existing_norm).ratio() >= _DUPLICATE_SIMILARITY_THRESHOLD:
                match_idx = i
                break

        if match_idx is None:
            item["also_reported_by"] = []
            kept.append(item)
            normalized_kept.append(norm)
        else:
            existing = kept[match_idx]
            existing.setdefault("also_reported_by", []).append(item.get("source"))
            # Prefer the higher-tier (lower number) source as the primary display item.
            if item.get("source_tier", 3) < existing.get("source_tier", 3):
                also_reported = existing["also_reported_by"] + [existing.get("source")]
                item["also_reported_by"] = [s for s in also_reported if s]
                kept[match_idx] = item
                normalized_kept[match_idx] = norm

    return kept


def _sort_key(item: dict):
    """Higher-tier sources first; within the same tier, most recent first."""
    tier = item.get("source_tier", 3)
    published_at = item.get("published_at")
    try:
        ts = datetime.fromisoformat(published_at).timestamp() if published_at else 0
    except (ValueError, TypeError):
        ts = 0
    return (tier, -ts)


class NewsService:
    """News source priority:

        1. Google News (preferred) + direct-outlet feeds (Seeking Alpha,
           CNBC, MarketWatch) — all genuinely independent of Yahoo,
           always attempted in parallel-by-sequence and merged together
           since more real coverage is strictly better than picking one.
        2. Yahoo (yfinance's own News API, then Yahoo's RSS feed) is
           used ONLY as a last resort — if and only if every source in
           step 1 failed to produce anything. This is logged loudly,
           never silent.

    Every attempt, successful or not, produces a SourceDiagnostics
    entry (HTTP status, byte count, entry count, error) that the API
    exposes for the debug panel — nothing about this pipeline should
    ever be a black box.

    Note on Reuters/Bloomberg: both discontinued their free public RSS
    feeds years ago and offer no free API alternative, so there's no
    direct-outlet source for them here. Their content still surfaces
    through Google News' own aggregation when Google has indexed it."""

    def __init__(self, google_provider: Optional[GoogleNewsSearchProvider] = None,
                 direct_outlet_providers: Optional[list[NewsDataProvider]] = None,
                 yahoo_providers: Optional[list[NewsDataProvider]] = None):
        self.google_provider = google_provider or GoogleNewsSearchProvider()
        self.direct_outlet_providers = direct_outlet_providers or [
            SeekingAlphaProvider(),
            NamedOutletRSSProvider("CNBC", CNBC_FEED_URL),
            NamedOutletRSSProvider("MarketWatch", MARKETWATCH_FEED_URL),
        ]
        self.yahoo_providers = yahoo_providers or [YFinanceNewsProvider(), RSSNewsProvider()]

    def get_analyzed_news(self, ticker: str, limit: int = 15, company_name: Optional[str] = None) -> DataResult:
        ticker = ticker.upper()
        all_items: list[dict] = []
        sources_used: list[str] = []
        diagnostics: list[SourceDiagnostics] = []
        fallback_used = False

        # Step 1: Google News — the preferred source.
        logger.info(f"[NEWS] === Fetching news for {ticker} ===")
        google_result, google_diag = self.google_provider.get_news(ticker, limit=limit, company_name=company_name)
        diagnostics.append(google_diag)
        preferred_succeeded = google_result.success and bool(google_result.data)
        if preferred_succeeded:
            all_items.extend(google_result.data)
            sources_used.append(google_result.source)
        else:
            logger.info(f"[NEWS] Google News did not return usable results ({google_diag.status}: {google_diag.error})")

        # Step 1b: direct-outlet feeds — always attempted and merged in
        # alongside Google News, not treated as fallback, since each is
        # a genuinely independent source of real coverage.
        for provider in self.direct_outlet_providers:
            result, diag = provider.get_news(ticker, limit=limit, company_name=company_name)
            diagnostics.append(diag)
            if result.success and result.data:
                all_items.extend(result.data)
                sources_used.append(result.source)
                preferred_succeeded = True
            else:
                logger.info(f"[NEWS] {provider.name} did not return usable results ({diag.status}: {diag.error})")

        # Step 2: Yahoo — last resort, only if NOTHING above worked.
        if not preferred_succeeded:
            logger.info("[NEWS] All preferred/direct sources failed — falling back to Yahoo Finance")
            for provider in self.yahoo_providers:
                result, diag = provider.get_news(ticker, limit)
                diagnostics.append(diag)
                if result.success and result.data:
                    all_items.extend(result.data)
                    sources_used.append(result.source)
                    fallback_used = True
                    break
                logger.info(f"[NEWS] {provider.name} did not return usable results ({diag.status}: {diag.error})")

        if not all_items:
            error_summary = "; ".join(f"{d.name}: {d.error}" for d in diagnostics if d.error)
            logger.info(f"[NEWS] All sources failed for {ticker}: {error_summary}")
            result = DataResult(
                data=[], source="none", fetched_at=datetime.now(timezone.utc),
                timeliness="delayed_15m", success=False, error=error_summary, quality="ERROR",
            )
            result.source_diagnostics = [d.as_dict() for d in diagnostics]  # type: ignore[attr-defined]
            result.fallback_used = fallback_used  # type: ignore[attr-defined]
            return result

        enriched = self._enrich(ticker, all_items)
        deduped = _dedupe_articles(enriched)
        deduped.sort(key=_sort_key)

        logger.info(
            f"[NEWS] Final result for {ticker}: {len(deduped)} unique articles from [{', '.join(sources_used)}]"
            f"{' (Yahoo fallback used)' if fallback_used else ''}"
        )

        result = DataResult(
            data=deduped[:limit],
            source=" + ".join(sources_used),
            fetched_at=datetime.now(timezone.utc),
            timeliness="delayed_15m",
            quality="GOOD" if not fallback_used else "WARNING",
        )
        result.source_diagnostics = [d.as_dict() for d in diagnostics]  # type: ignore[attr-defined]
        result.fallback_used = fallback_used  # type: ignore[attr-defined]
        return result

    def _enrich(self, ticker: str, items: list[dict]) -> list[dict]:
        enriched = []
        now = datetime.now(timezone.utc).isoformat()
        rows_to_store = []
        for item in items:
            headline = item.get("headline", "")
            analysis = score_news_impact(headline)
            source_tier = classify_source_tier(item.get("source", ""))
            row = {**item, **analysis, "source_tier": source_tier}
            enriched.append(row)
            rows_to_store.append((
                ticker, item.get("published_at"), item.get("source"), headline,
                item.get("link"), analysis["category"], analysis["sentiment"]["score"],
                analysis["impact_score"], now,
            ))

        # Best-effort history logging — never blocks the response if it fails.
        try:
            with db_cursor() as cur:
                cur.executemany(
                    """INSERT INTO news (ticker, published_at, source, headline, link,
                                          category, sentiment, impact_score, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows_to_store,
                )
        except Exception:  # noqa: BLE001
            pass

        return enriched


news_service = NewsService()
