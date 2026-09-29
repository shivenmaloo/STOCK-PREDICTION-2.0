from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

import feedparser
import requests
import yfinance as yf

from backend.data.base import DataResult, NewsDataProvider, ProviderError

# Server-side diagnostic logging for the news pipeline — every fetch
# attempt logs its own outcome so a failure is never silent. Configured
# with its own handler (rather than relying on root/uvicorn logging
# config) so these lines are guaranteed visible in the terminal
# regardless of how the app was started.
logger = logging.getLogger("stock_ai.news")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class SourceDiagnostics:
    """Everything the debug panel needs to show exactly what happened
    for one news source, on one request — never just 'it worked' or
    'it didn't', but the actual HTTP status, byte count, entry count,
    and error, per spec."""

    def __init__(self, name: str):
        self.name = name
        self.status = "NOT_ATTEMPTED"  # SUCCESS | EMPTY | HTTP_ERROR | RATE_LIMITED | FORBIDDEN | NETWORK_ERROR | PARSE_ERROR
        self.http_status: Optional[int] = None
        self.request_url: Optional[str] = None
        self.response_size_bytes: Optional[int] = None
        self.entries_found: int = 0
        self.valid_articles: int = 0
        self.publisher_breakdown: dict[str, int] = {}
        self.error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "name": self.name, "status": self.status, "http_status": self.http_status,
            "request_url": self.request_url, "response_size_bytes": self.response_size_bytes,
            "entries_found": self.entries_found, "valid_articles": self.valid_articles,
            "publisher_breakdown": self.publisher_breakdown, "error": self.error,
        }


class YFinanceNewsProvider(NewsDataProvider):
    """Ticker-specific source: yfinance's Ticker.news, which proxies
    Yahoo Finance's own aggregated headlines for a ticker."""

    name = "yfinance"

    def get_news(self, ticker: str, limit: int = 20) -> tuple[DataResult, SourceDiagnostics]:
        diag = SourceDiagnostics("Yahoo Finance (yfinance)")
        logger.info(f"[NEWS] yfinance: fetching ticker-specific news for {ticker}")
        try:
            t = yf.Ticker(ticker)
            raw = t.news or []
            diag.entries_found = len(raw)
            if not raw:
                diag.status = "EMPTY"
                diag.error = "yfinance returned zero news items"
                logger.info(f"[NEWS] yfinance: 0 entries for {ticker}")
                raise ProviderError(diag.error)

            items = []
            for item in raw[:limit]:
                content = item.get("content", item)  # yfinance has changed this shape across versions
                title = content.get("title") or item.get("title")
                if not title:
                    continue
                link = (
                    (content.get("clickThroughUrl") or {}).get("url")
                    or (content.get("canonicalUrl") or {}).get("url")
                    or item.get("link")
                )
                pub_date = content.get("pubDate") or item.get("providerPublishTime")
                published_at = _normalize_timestamp(pub_date)
                source = (content.get("provider") or {}).get("displayName") or item.get("publisher") or "Unknown"

                items.append({
                    "headline": title,
                    "link": link,
                    "source": source,
                    "published_at": published_at,
                })
                diag.publisher_breakdown[source] = diag.publisher_breakdown.get(source, 0) + 1

            if not items:
                diag.status = "EMPTY"
                diag.error = "yfinance payload had entries but none had usable titles"
                raise ProviderError(diag.error)

            diag.valid_articles = len(items)
            diag.status = "SUCCESS"
            logger.info(f"[NEWS] yfinance: SUCCESS — {len(items)} valid articles for {ticker}")

            return DataResult(
                data=items, source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="delayed_15m",
            ), diag
        except Exception as exc:  # noqa: BLE001
            if diag.status == "NOT_ATTEMPTED":
                diag.status = "NETWORK_ERROR"
            diag.error = diag.error or str(exc)
            logger.info(f"[NEWS] yfinance: FAILED — {diag.error}")
            return DataResult(
                data=[], source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="delayed_15m", success=False, error=diag.error, quality="ERROR",
            ), diag


class RSSNewsProvider(NewsDataProvider):
    """Secondary Yahoo-derived fallback: Yahoo Finance's public
    per-ticker RSS feed, used only if yfinance's own News API fails
    outright. No API key, but less structured."""

    name = "yahoo_rss"
    FEED_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"

    def get_news(self, ticker: str, limit: int = 20) -> tuple[DataResult, SourceDiagnostics]:
        diag = SourceDiagnostics("Yahoo Finance RSS")
        url = f"{self.FEED_URL}?s={ticker}&region=US&lang=en-US"
        diag.request_url = url
        logger.info(f"[NEWS] yahoo_rss: fetching {url}")
        try:
            parsed = feedparser.parse(url)
            diag.entries_found = len(parsed.entries)
            if parsed.bozo and not parsed.entries:
                diag.status = "PARSE_ERROR"
                diag.error = f"RSS feed unreachable or malformed for {ticker} ({getattr(parsed, 'bozo_exception', 'unknown')})"
                raise ProviderError(diag.error)
            if not parsed.entries:
                diag.status = "EMPTY"
                diag.error = f"No RSS entries for {ticker}"
                raise ProviderError(diag.error)

            items = []
            for entry in parsed.entries[:limit]:
                published_at = None
                if getattr(entry, "published_parsed", None):
                    published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
                items.append({
                    "headline": entry.get("title", ""),
                    "link": entry.get("link"),
                    "source": "Yahoo Finance RSS",
                    "published_at": published_at,
                })

            diag.valid_articles = len(items)
            diag.publisher_breakdown["Yahoo Finance RSS"] = len(items)
            diag.status = "SUCCESS"
            logger.info(f"[NEWS] yahoo_rss: SUCCESS — {len(items)} articles for {ticker}")

            return DataResult(
                data=items, source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="delayed_15m",
            ), diag
        except Exception as exc:  # noqa: BLE001
            if diag.status == "NOT_ATTEMPTED":
                diag.status = "NETWORK_ERROR"
            diag.error = diag.error or str(exc)
            logger.info(f"[NEWS] yahoo_rss: FAILED — {diag.error}")
            return DataResult(
                data=[], source=self.name, fetched_at=datetime.now(timezone.utc),
                timeliness="delayed_15m", success=False, error=diag.error, quality="ERROR",
            ), diag


# Google News' search RSS — a genuinely independent pipeline from
# Yahoo's syndication network. It indexes and returns real, distinctly
# attributed publishers (Reuters, Bloomberg, Barron's, individual
# newsroom sites, etc.) for a specific search query, rather than
# filtering one static general-topics feed by keyword.
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

# feedparser's default fetch uses a generic non-browser User-Agent, which
# some servers quietly rate-limit or reject. We fetch the XML ourselves
# with a browser-like header and full status/diagnostic visibility
# instead of letting feedparser fetch (and potentially silently fail on)
# the URL itself.
_BROWSER_USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# Words that don't help identify a specific company and should be
# stripped when deriving a search/match term from a company's legal name.
_CORP_SUFFIXES = {"inc", "inc.", "corp", "corp.", "corporation", "co", "co.", "ltd", "ltd.",
                   "plc", "llc", "group", "holdings", "holding", "company", "the", "class", "a"}


def _build_relevance_terms(ticker: str, company_name: Optional[str]) -> list[tuple[str, bool]]:
    """Returns (term, case_sensitive) pairs used to check whether a
    headline is actually about this company, not just topically
    adjacent. Very short tickers (1-2 letters, e.g. 'T', 'F', 'C') are
    matched case-sensitively to cut down on false positives against
    common words, since they'd otherwise match constantly."""
    ticker = ticker.upper()
    terms: list[tuple[str, bool]] = [(ticker, len(ticker) <= 2)]

    if company_name:
        words = re.findall(r"[A-Za-z0-9&]+", company_name)
        significant = [w for w in words if w.lower() not in _CORP_SUFFIXES and len(w) >= 3]
        if significant:
            terms.append((significant[0], False))
            if len(significant) > 1:
                terms.append((" ".join(significant[:2]), False))

    return terms


def _is_relevant(headline: str, terms: list[tuple[str, bool]]) -> bool:
    for term, case_sensitive in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, headline, 0 if case_sensitive else re.IGNORECASE):
            return True
    return False


class GoogleNewsSearchProvider(NewsDataProvider):
    """Searches Google News for a ticker (and optionally company name)
    and returns real, attributed headlines from whatever outlets Google
    News has indexed — this is the PREFERRED/primary source; the
    Yahoo-derived providers above are only meant to fill in when this
    fails, never to silently replace it."""

    name = "google_news"

    def get_news(self, ticker: str, limit: int = 20, company_name: Optional[str] = None) -> tuple[DataResult, SourceDiagnostics]:
        diag = SourceDiagnostics("Google News RSS")
        query = f"{company_name} stock" if company_name else f"{ticker} stock"
        params = urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        url = f"{GOOGLE_NEWS_RSS_URL}?{params}"
        diag.request_url = url

        logger.info(f"[NEWS] Searching Google News for: {query}")
        logger.info("[NEWS] Google RSS request started")

        try:
            resp = requests.get(url, headers={"User-Agent": _BROWSER_USER_AGENT}, timeout=10)
        except requests.exceptions.Timeout:
            diag.status = "NETWORK_ERROR"
            diag.error = f"Request to Google News timed out after 10s (url: {url})"
            logger.info(f"[NEWS] Google RSS FAILED — timeout")
            return _empty_failure(self.name, diag.error), diag
        except requests.exceptions.RequestException as exc:
            diag.status = "NETWORK_ERROR"
            diag.error = f"Network error reaching Google News: {exc}"
            logger.info(f"[NEWS] Google RSS FAILED — network error: {exc}")
            return _empty_failure(self.name, diag.error), diag

        diag.http_status = resp.status_code
        diag.response_size_bytes = len(resp.content)
        logger.info(f"[NEWS] HTTP status: {resp.status_code}")
        logger.info(f"[NEWS] Response size: {len(resp.content):,} bytes")

        if resp.status_code == 429:
            diag.status = "RATE_LIMITED"
            diag.error = f"Google News rate-limited the request (HTTP 429) for '{query}'"
            logger.info("[NEWS] Google RSS FAILED — rate limited (429)")
            return _empty_failure(self.name, diag.error), diag
        if resp.status_code == 403:
            diag.status = "FORBIDDEN"
            diag.error = f"Google News returned HTTP 403 Forbidden for '{query}'"
            logger.info("[NEWS] Google RSS FAILED — forbidden (403)")
            return _empty_failure(self.name, diag.error), diag
        if resp.status_code != 200:
            diag.status = "HTTP_ERROR"
            diag.error = f"Google News returned HTTP {resp.status_code} for '{query}'"
            logger.info(f"[NEWS] Google RSS FAILED — HTTP {resp.status_code}")
            return _empty_failure(self.name, diag.error), diag

        try:
            parsed = feedparser.parse(resp.content)
        except Exception as exc:  # noqa: BLE001
            diag.status = "PARSE_ERROR"
            diag.error = f"feedparser could not parse the response: {exc}"
            logger.info(f"[NEWS] Google RSS FAILED — parse error: {exc}")
            return _empty_failure(self.name, diag.error), diag

        diag.entries_found = len(parsed.entries)
        logger.info(f"[NEWS] Parsed RSS entries: {len(parsed.entries)}")

        if parsed.bozo and not parsed.entries:
            diag.status = "PARSE_ERROR"
            diag.error = f"Malformed RSS/XML response ({getattr(parsed, 'bozo_exception', 'unknown parse error')})"
            logger.info(f"[NEWS] Google RSS FAILED — {diag.error}")
            return _empty_failure(self.name, diag.error), diag

        if not parsed.entries:
            diag.status = "EMPTY"
            diag.error = f"Google News returned HTTP 200 but zero entries for '{query}'"
            logger.info("[NEWS] Google RSS: 0 entries returned (HTTP 200, empty result set)")
            return _empty_failure(self.name, diag.error), diag

        items = []
        dropped_irrelevant = 0
        relevance_terms = _build_relevance_terms(ticker, company_name)

        for entry in parsed.entries[:limit * 2]:  # pull extra since some will be filtered out
            title = entry.get("title", "")
            # Google News RSS titles are usually "Headline - Publisher";
            # prefer the structured <source> tag when feedparser exposes
            # it, and fall back to splitting the title suffix. Never
            # invent a publisher we can't actually identify.
            source_name = None
            source_obj = entry.get("source")
            if isinstance(source_obj, dict):
                source_name = source_obj.get("title")
            if not source_name and " - " in title:
                title, _, source_name = title.rpartition(" - ")
            if not source_name:
                source_name = "Google News (source unspecified)"

            # Google's own search relevance is loose — it will happily
            # return "related market context" articles that never
            # actually mention the company. Require the ticker or
            # company name to literally appear in the headline before
            # accepting the result, so an AAPL search can't surface a
            # Lakers tax-shelter story just because Google decided it
            # was topically adjacent.
            if not _is_relevant(title, relevance_terms):
                dropped_irrelevant += 1
                continue

            published_at = None
            if getattr(entry, "published_parsed", None):
                published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()

            items.append({
                "headline": title.strip(),
                "link": entry.get("link"),
                "source": source_name,
                "published_at": published_at,
            })
            diag.publisher_breakdown[source_name] = diag.publisher_breakdown.get(source_name, 0) + 1
            if len(items) >= limit:
                break

        if dropped_irrelevant:
            logger.info(f"[NEWS] Dropped {dropped_irrelevant} Google News results that didn't actually mention {ticker}/{company_name or ''}")

        if not items:
            diag.status = "EMPTY"
            diag.error = (
                f"Google News returned {diag.entries_found} entries for '{query}' but none "
                f"mentioned {ticker} by name — likely all loosely-related market context, not filtered out"
            )
            logger.info(f"[NEWS] Google RSS: all {diag.entries_found} entries failed the relevance check")
            return _empty_failure(self.name, diag.error), diag

        diag.valid_articles = len(items)
        diag.status = "SUCCESS"
        breakdown_str = ", ".join(f"{k}: {v}" for k, v in sorted(diag.publisher_breakdown.items(), key=lambda x: -x[1]))
        logger.info(f"[NEWS] Valid articles after filtering: {len(items)}")
        logger.info(f"[NEWS] Publisher breakdown — {breakdown_str}")
        logger.info(f"[NEWS] Google News: SUCCESS — {len(items)} results")

        return DataResult(
            data=items, source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="delayed_15m",
        ), diag


def _empty_failure(source_name: str, error: str) -> DataResult:
    return DataResult(
        data=[], source=source_name, fetched_at=datetime.now(timezone.utc),
        timeliness="delayed_15m", success=False, error=error, quality="ERROR",
    )


class SeekingAlphaProvider(NewsDataProvider):
    """Seeking Alpha publishes a per-ticker RSS feed — this is a
    genuinely ticker-scoped source (the ticker is in the URL itself,
    not matched by keyword), so no relevance filtering is needed here
    the way it is for general-topic feeds. Note: Seeking Alpha is a
    contributor/analysis platform, not a wire service — its articles
    include opinion and analysis pieces alongside straight news."""

    name = "Seeking Alpha"
    FEED_URL = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"

    def get_news(self, ticker: str, limit: int = 20, company_name: Optional[str] = None) -> tuple[DataResult, SourceDiagnostics]:
        diag = SourceDiagnostics(self.name)
        url = self.FEED_URL.format(ticker=ticker.upper())
        diag.request_url = url
        logger.info(f"[NEWS] Fetching Seeking Alpha feed for {ticker}")

        try:
            resp = requests.get(url, headers={"User-Agent": _BROWSER_USER_AGENT}, timeout=10)
        except requests.exceptions.Timeout:
            diag.status, diag.error = "NETWORK_ERROR", f"Seeking Alpha request timed out for {ticker}"
            logger.info(f"[NEWS] Seeking Alpha FAILED — timeout")
            return _empty_failure(self.name, diag.error), diag
        except requests.exceptions.RequestException as exc:
            diag.status, diag.error = "NETWORK_ERROR", f"Network error reaching Seeking Alpha: {exc}"
            logger.info(f"[NEWS] Seeking Alpha FAILED — {exc}")
            return _empty_failure(self.name, diag.error), diag

        diag.http_status = resp.status_code
        diag.response_size_bytes = len(resp.content)
        logger.info(f"[NEWS] Seeking Alpha HTTP status: {resp.status_code}, size: {len(resp.content):,} bytes")

        if resp.status_code == 404:
            diag.status, diag.error = "EMPTY", f"No Seeking Alpha feed found for {ticker} (HTTP 404)"
            return _empty_failure(self.name, diag.error), diag
        if resp.status_code == 403:
            diag.status, diag.error = "FORBIDDEN", f"Seeking Alpha returned HTTP 403 for {ticker}"
            return _empty_failure(self.name, diag.error), diag
        if resp.status_code == 429:
            diag.status, diag.error = "RATE_LIMITED", f"Seeking Alpha rate-limited the request for {ticker}"
            return _empty_failure(self.name, diag.error), diag
        if resp.status_code != 200:
            diag.status, diag.error = "HTTP_ERROR", f"Seeking Alpha returned HTTP {resp.status_code} for {ticker}"
            return _empty_failure(self.name, diag.error), diag

        try:
            parsed = feedparser.parse(resp.content)
        except Exception as exc:  # noqa: BLE001
            diag.status, diag.error = "PARSE_ERROR", f"feedparser could not parse Seeking Alpha response: {exc}"
            return _empty_failure(self.name, diag.error), diag

        diag.entries_found = len(parsed.entries)
        if not parsed.entries:
            diag.status, diag.error = "EMPTY", f"Seeking Alpha returned HTTP 200 but zero entries for {ticker}"
            return _empty_failure(self.name, diag.error), diag

        items = []
        for entry in parsed.entries[:limit]:
            published_at = None
            if getattr(entry, "published_parsed", None):
                published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
            items.append({
                "headline": entry.get("title", ""),
                "link": entry.get("link"),
                "source": self.name,
                "published_at": published_at,
            })

        diag.valid_articles = len(items)
        diag.publisher_breakdown[self.name] = len(items)
        diag.status = "SUCCESS"
        logger.info(f"[NEWS] Seeking Alpha: SUCCESS — {len(items)} results for {ticker}")

        return DataResult(
            data=items, source=self.name, fetched_at=datetime.now(timezone.utc), timeliness="delayed_15m",
        ), diag


class NamedOutletRSSProvider(NewsDataProvider):
    """A single named outlet's general-topics RSS feed (not ticker-
    specific), filtered down to entries that actually mention the
    ticker or company name via the same relevance check used for
    Google News — otherwise a general front-page feed would return
    mostly-irrelevant headlines for any given company."""

    def __init__(self, display_name: str, feed_url: str):
        self._display_name = display_name
        self.feed_url = feed_url

    @property
    def name(self) -> str:
        return self._display_name

    def get_news(self, ticker: str, limit: int = 20, company_name: Optional[str] = None) -> tuple[DataResult, SourceDiagnostics]:
        diag = SourceDiagnostics(self._display_name)
        diag.request_url = self.feed_url
        logger.info(f"[NEWS] Fetching {self._display_name} general feed")

        try:
            resp = requests.get(self.feed_url, headers={"User-Agent": _BROWSER_USER_AGENT}, timeout=10)
        except requests.exceptions.Timeout:
            diag.status, diag.error = "NETWORK_ERROR", f"{self._display_name} request timed out"
            return _empty_failure(self._display_name, diag.error), diag
        except requests.exceptions.RequestException as exc:
            diag.status, diag.error = "NETWORK_ERROR", f"Network error reaching {self._display_name}: {exc}"
            return _empty_failure(self._display_name, diag.error), diag

        diag.http_status = resp.status_code
        diag.response_size_bytes = len(resp.content)
        logger.info(f"[NEWS] {self._display_name} HTTP status: {resp.status_code}, size: {len(resp.content):,} bytes")

        if resp.status_code == 403:
            diag.status, diag.error = "FORBIDDEN", f"{self._display_name} returned HTTP 403"
            return _empty_failure(self._display_name, diag.error), diag
        if resp.status_code == 429:
            diag.status, diag.error = "RATE_LIMITED", f"{self._display_name} rate-limited the request"
            return _empty_failure(self._display_name, diag.error), diag
        if resp.status_code != 200:
            diag.status, diag.error = "HTTP_ERROR", f"{self._display_name} returned HTTP {resp.status_code}"
            return _empty_failure(self._display_name, diag.error), diag

        try:
            parsed = feedparser.parse(resp.content)
        except Exception as exc:  # noqa: BLE001
            diag.status, diag.error = "PARSE_ERROR", f"feedparser could not parse {self._display_name} response: {exc}"
            return _empty_failure(self._display_name, diag.error), diag

        diag.entries_found = len(parsed.entries)
        if not parsed.entries:
            diag.status, diag.error = "EMPTY", f"{self._display_name} feed returned HTTP 200 but zero entries"
            return _empty_failure(self._display_name, diag.error), diag

        terms = _build_relevance_terms(ticker, company_name)
        items = []
        dropped = 0
        for entry in parsed.entries:
            title = entry.get("title", "")
            if not _is_relevant(title, terms):
                dropped += 1
                continue
            published_at = None
            if getattr(entry, "published_parsed", None):
                published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
            items.append({
                "headline": title, "link": entry.get("link"),
                "source": self._display_name, "published_at": published_at,
            })
            if len(items) >= limit:
                break

        if dropped:
            logger.info(f"[NEWS] {self._display_name}: dropped {dropped} entries not mentioning {ticker}")

        if not items:
            diag.status = "EMPTY"
            diag.error = f"{self._display_name} had {diag.entries_found} general headlines but none mentioned {ticker}"
            logger.info(f"[NEWS] {self._display_name}: 0 relevant results after filtering {diag.entries_found} general headlines")
            return _empty_failure(self._display_name, diag.error), diag

        diag.valid_articles = len(items)
        diag.publisher_breakdown[self._display_name] = len(items)
        diag.status = "SUCCESS"
        logger.info(f"[NEWS] {self._display_name}: SUCCESS — {len(items)} relevant results")

        return DataResult(
            data=items, source=self._display_name, fetched_at=datetime.now(timezone.utc), timeliness="delayed_15m",
        ), diag


# CNBC and MarketWatch both publish general-topics RSS feeds (not
# ticker-specific) — filtered down via NamedOutletRSSProvider above.
# Reuters and Bloomberg are deliberately NOT included here: both shut
# down their free public RSS feeds years ago, and there's no free API
# alternative. Their content still reaches the app through Google
# News' own aggregation (which does index Reuters/Bloomberg stories),
# just not via a direct feed — pretending otherwise would mean either
# fabricating an endpoint or quietly scraping a paywalled site, neither
# of which this app does.
CNBC_FEED_URL = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"
MARKETWATCH_FEED_URL = "https://www.marketwatch.com/rss/topstories"


def _normalize_timestamp(value) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        if isinstance(value, str):
            # yfinance sometimes returns ISO8601 strings already
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except Exception:  # noqa: BLE001
        return None
    return None
