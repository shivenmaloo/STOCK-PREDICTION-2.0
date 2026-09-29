"""
Live news context for predictions — deliberately POST-HOC, never a
trained feature.

Why not a real feature: training the model on news sentiment would
require knowing what the sentiment was on every historical date going
back years, which we don't have (no point-in-time news archive — our
providers only return *recent* headlines). Faking it by applying
today's sentiment to historical training rows would be a serious form
of look-ahead bias: the model would be trained as if it "knew" news
sentiment on days it couldn't actually have known.

What this does instead: fetches CURRENT news only, aggregates it into
a single directional read, and compares that against what the
(purely price/technical-trained) ensemble already concluded. This
never changes the model's prediction, confidence, or weighting — it's
an independent, second opinion presented alongside the model's own
read, exactly the way a human analyst would cross-check a technical
signal against the headlines before acting on it.

One adaptation genuinely worth having, though: the "recent N articles"
window can span a much wider calendar gap right after a weekend or
holiday than on an ordinary trading day — Monday morning's "10 most
recent articles" might stretch back through Saturday and Sunday, while
Tuesday's covers barely more than one day. Averaging every article in
that window equally would let a single stale Saturday rumor count the
same as this morning's actual earnings report. Since every article
already carries a real publish timestamp, this recency-weights the
impact score (more recent = more weight) and is transparent in the
summary about how wide a window is actually being averaged over —
this is a live-only adaptation of the same underlying idea, not the
historical-training version, which still isn't possible without a
point-in-time archive we don't have.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

RECENCY_HALF_LIFE_HOURS = 18  # an article twice this old counts for roughly half as much


def _get_source_reliability_weight(source_name: str) -> float:
    """Turns a source's own tracked reliability into a weight
    multiplier: 1.0 (perfectly neutral, unchanged from before this
    feature existed) for a source with no real evidence yet, scaling
    up to 1.5 for a source proven consistently correct, and down to
    0.5 for one proven consistently wrong — proportional, not a hard
    on/off cutoff, since a source is never fully EXCLUDED just for
    being new; it simply hasn't earned extra trust (or distrust) yet.
    Never raises — a reliability lookup failure must never break the
    news read that triggered it."""
    try:
        from backend.news.source_reliability import get_source_reliability
        result = get_source_reliability(source_name)
        if not result["available"]:
            return 1.0
        return 0.5 + result["reliability"]
    except Exception:  # noqa: BLE001
        return 1.0


def _weighted_median(values_and_weights: list[tuple[float, float]]) -> float:
    """A weighted median — the value at which half the total weight
    lies on either side. Used instead of a weighted mean specifically
    because a mean can be dragged far from what most sources actually
    say by a single extreme outlier article (proven directly: five
    ordinary articles averaging mildly positive, plus one wildly
    bullish outlier, can push a MEAN past the bullish threshold even
    though the median — reflecting what the majority of sources
    actually said — stays mild). The median is robust to exactly that
    failure mode: one loud outlier can't drag it past where the bulk
    of real coverage actually sits.

    Matches the standard median convention for the equal-weight case:
    when the cumulative weight lands exactly on the halfway point (an
    even count of equally-weighted items), the median is the average
    of that value and the next one — not just whichever one happens
    to be reached first."""
    if not values_and_weights:
        return 0.0
    sorted_pairs = sorted(values_and_weights, key=lambda p: p[0])
    total_weight = sum(w for _, w in sorted_pairs)
    if total_weight <= 0:
        return 0.0
    half = total_weight / 2
    cumulative = 0.0
    for i, (value, weight) in enumerate(sorted_pairs):
        cumulative += weight
        if cumulative == half and i + 1 < len(sorted_pairs):
            return (value + sorted_pairs[i + 1][0]) / 2
        if cumulative > half:
            return value
    return sorted_pairs[-1][0]


def fetch_news_sentiment(ticker: str, news_service=None, fundamentals_service=None) -> dict:
    """The pure sentiment-fetching half of what used to be a single
    get_news_context() call — deliberately split out so it can run
    BEFORE the ensemble is built (needed to use news as an ensemble
    vote — see backend/forecasting/service.py), not just after it.
    Returns {available, article_count, avg_impact_score,
    median_impact_score, source_agreement_pct, news_direction,
    coverage_note} with NO dependency on any ensemble result —
    compare_news_to_ensemble() below adds that comparison afterward,
    cheaply, without re-fetching or re-scoring anything."""
    if news_service is None:
        from backend.news.service import news_service as default_news_service
        news_service = default_news_service

    # Resolve the company name so the news search isn't limited to
    # headlines that literally contain the ticker symbol — a real gap
    # found via testing: "NVDA" search misses articles that only say
    # "NVIDIA", which is most of them. Best-effort; falls back to a
    # ticker-only search if fundamentals lookup fails for any reason.
    company_name = None
    try:
        if fundamentals_service is None:
            from backend.fundamentals.service import fundamentals_service as default_fundamentals_service
            fundamentals_service = default_fundamentals_service
        fund_result = fundamentals_service.get_fundamentals(ticker)
        if fund_result.success and fund_result.data:
            company_name = fund_result.data.get("short_name") or fund_result.data.get("long_name")
    except Exception:  # noqa: BLE001
        pass  # ticker-only search is a fine fallback, not worth failing the whole prediction over

    try:
        # Deliberately increased from 10 to 25 (more thorough analysis,
        # explicitly chosen over speed): a larger sample gives the
        # median-based consensus calculation more real, opinionated
        # articles to work with, which matters directly given that
        # routine, no-signal articles are now excluded from that
        # calculation — a bigger raw pull means more genuine signal
        # survives that filtering, not just more noise.
        result = news_service.get_analyzed_news(ticker, limit=25, company_name=company_name)
    except Exception:  # noqa: BLE001
        return {"available": False, "reason": "News fetch failed."}

    if not result.success or not result.data:
        return {"available": False, "reason": "No recent news available for this ticker."}

    items = result.data
    scored_items = [item for item in items if "impact_score" in item]
    if not scored_items:
        return {"available": False, "reason": "News found but no analyzable impact scores."}

    now = datetime.now(timezone.utc)
    weighted_sum, weight_total = 0.0, 0.0
    score_weight_pairs: list[tuple[float, float]] = []
    parsed_dates: list[datetime] = []
    # Collected so the caller (service.py) can log each opinionated
    # article's own source-level call for future reliability
    # evaluation — see backend/news/source_reliability.py.
    source_readings: list[tuple[str, float, str]] = []  # (source_name, impact_score, direction)

    # Real, empirically-proven bug found via testing: an article that
    # scores EXACTLY 0.0 didn't express a directional opinion at all —
    # no finance-relevant keywords matched (routine headlines like
    # "company announces board appointment" or "to present at investor
    # conference" score exactly zero, not "mildly neutral"). Testing
    # against realistic, everyday headline samples showed 6 of 10
    # scoring exactly 0.0 — and critically, even with a genuine +88
    # earnings beat AND a genuine -94 downgrade mixed into the same
    # 10-article sample, the MEDIAN still came out to exactly 0.0,
    # because it requires the BULK of articles to agree, and most
    # articles on an ordinary day simply have no opinion to agree with.
    # This was very likely the real, dominant reason news rarely
    # cleared the classification threshold at all — diluted into
    # apparent neutrality by routine coverage, not because genuinely
    # notable news was actually absent. The fix: exclude true no-signal
    # articles from the consensus calculation entirely — they represent
    # an ABSENCE of opinion, not a real "neutral" opinion to average in.
    n_no_signal_articles = 0
    for item in scored_items:
        if item["impact_score"] == 0.0:
            n_no_signal_articles += 1
            continue

        age_hours = None
        published_at = item.get("published_at")
        if published_at:
            try:
                published_dt = datetime.fromisoformat(published_at)
                if published_dt.tzinfo is None:
                    published_dt = published_dt.replace(tzinfo=timezone.utc)
                age_hours = max((now - published_dt).total_seconds() / 3600, 0)
                parsed_dates.append(published_dt)
            except (ValueError, TypeError):
                age_hours = None
        # An article with no usable timestamp still counts, just at
        # neutral (not-recency-boosted, not-penalized) weight, rather
        # than being dropped — a missing date isn't a reason to throw
        # away a real, already-analyzed article.
        recency_weight = 0.5 ** (age_hours / RECENCY_HALF_LIFE_HOURS) if age_hours is not None else 0.5

        # Reliability weighting: a source with a PROVEN good track
        # record counts more; a source with a PROVEN bad one counts
        # less; a source with no real evidence yet stays neutral (the
        # same weight it would have had before this feature existed) —
        # never fully excluded just for being new, since that would
        # mean no source could ever earn a track record in the first
        # place. See backend/news/source_reliability.py.
        source_name = item.get("source")
        reliability_weight = _get_source_reliability_weight(source_name) if source_name else 1.0

        weight = recency_weight * reliability_weight
        weighted_sum += item["impact_score"] * weight
        weight_total += weight
        score_weight_pairs.append((item["impact_score"], weight))

        article_direction = "bullish" if item["impact_score"] > 0 else "bearish"
        source_readings.append((source_name or "Unknown", item["impact_score"], article_direction))

    n_opinionated_articles = len(score_weight_pairs)
    if n_opinionated_articles == 0:
        # Every single article was genuinely no-signal — this is an
        # honest, real neutral read (there simply isn't a directional
        # opinion in the current coverage), not a bug or an error.
        return {
            "available": True, "article_count": len(scored_items), "n_opinionated_articles": 0,
            "avg_impact_score": 0.0, "median_impact_score": 0.0, "source_agreement_pct": None,
            "news_direction": "neutral", "coverage_note": "", "source_readings": [],
        }

    avg_impact = weighted_sum / weight_total if weight_total > 0 else 0.0

    # The actual basis for classification: a weighted MEDIAN, not the
    # mean above (kept only for transparency/display). "Best overall
    # sentiment" should mean what the bulk of real coverage actually
    # says, not whatever a single unusually extreme article happens to
    # push the average toward — the median is specifically robust to
    # that failure mode. See _weighted_median()'s docstring for the
    # concrete proof of why this matters. Computed only over the
    # OPINIONATED articles now, for the reason explained above.
    median_impact = _weighted_median(score_weight_pairs)

    # Transparency: how many of the OPINIONATED articles actually agree
    # in DIRECTION with the median-based read — a genuine consensus
    # measure, not just a number. If the read is bullish but most
    # opinionated articles are bearish, that's visible here rather than
    # hidden behind one aggregate figure.
    if median_impact > 0:
        agreeing_count = sum(1 for score, _ in score_weight_pairs if score > 0)
    elif median_impact < 0:
        agreeing_count = sum(1 for score, _ in score_weight_pairs if score < 0)
    else:
        agreeing_count = sum(1 for score, _ in score_weight_pairs if score == 0)
    source_agreement_pct = round(agreeing_count / len(score_weight_pairs) * 100, 1)

    # Transparency about the actual window being averaged over — this
    # is the part that matters more right after a weekend/holiday than
    # on an ordinary day.
    coverage_note = ""
    if len(parsed_dates) >= 2:
        span_hours = (max(parsed_dates) - min(parsed_dates)).total_seconds() / 3600
        if span_hours > 30:  # meaningfully wider than a single trading day
            span_days = round(span_hours / 24, 1)
            coverage_note = f" (spans about {span_days} days, likely including a weekend or holiday gap)"

    try:
        from backend.settings_service import get_setting
        threshold = float(get_setting("news_direction_threshold") or 15)
    except Exception:  # noqa: BLE001 — a settings lookup must never break the news comparison itself
        threshold = 15.0

    if median_impact >= threshold:
        news_direction = "bullish"
    elif median_impact <= -threshold:
        news_direction = "bearish"
    else:
        news_direction = "neutral"

    return {
        "available": True,
        "article_count": len(scored_items),
        "n_opinionated_articles": n_opinionated_articles,
        "avg_impact_score": round(avg_impact, 1),
        "median_impact_score": round(median_impact, 1),
        "source_agreement_pct": source_agreement_pct,
        "news_direction": news_direction,
        "coverage_note": coverage_note,
        "source_readings": source_readings,
    }


def compare_news_to_ensemble(news_sentiment: dict, ensemble_direction: str) -> dict:
    """Cheap, pure, no I/O — takes an ALREADY-fetched news_sentiment
    dict (from fetch_news_sentiment) and adds the agrees_with_model
    comparison and summary text. Split out from fetching specifically
    so this can run AFTER the ensemble is built (which may now use
    news as one of its own inputs — see service.py), without needing a
    second fetch or a circular dependency between the two."""
    if not news_sentiment.get("available"):
        return news_sentiment

    news_direction = news_sentiment["news_direction"]
    agrees_with_model: Optional[bool]
    if ensemble_direction == "neutral" or news_direction == "neutral":
        agrees_with_model = None  # not a meaningful comparison either way
    else:
        agrees_with_model = (ensemble_direction == news_direction)

    n_opinionated = news_sentiment.get("n_opinionated_articles", news_sentiment["article_count"])
    if n_opinionated == 0:
        summary = (
            f"{news_sentiment['article_count']} recent article(s) found, but none contained language the "
            f"scoring engine recognizes as directional (routine announcements, not market-moving news) "
            f"— reads neutral because there's genuinely no opinion to read, not because of a tie."
        )
    else:
        agreement_text = f"{news_sentiment['source_agreement_pct']:.0f}% of sources agree" if news_sentiment["source_agreement_pct"] is not None else "no clear majority"
        summary = (
            f"{n_opinionated} of {news_sentiment['article_count']} recent articles actually took a directional stance "
            f"(the rest were routine, no-signal coverage and excluded from this read), recency-weighted, "
            f"average {news_sentiment['avg_impact_score']:+.0f} / median {news_sentiment['median_impact_score']:+.0f} "
            f"impact (scale -100 to +100){news_sentiment['coverage_note']} "
            f"— reads {news_direction} ({agreement_text})."
        )
    if agrees_with_model is False:
        summary += f" This runs counter to the model's own {ensemble_direction} read — worth a second look."
    elif agrees_with_model is True:
        summary += f" This agrees with the model's {ensemble_direction} read."

    return {**news_sentiment, "agrees_with_model": agrees_with_model, "summary": summary}


def get_news_context(ticker: str, ensemble_direction: str, news_service=None, fundamentals_service=None) -> dict:
    """Returns {available, article_count, avg_impact_score, news_direction,
    agrees_with_model, summary}. Never raises — a news fetch failure
    just means unavailable context, not a broken prediction.

    Kept as a single-call convenience wrapper around
    fetch_news_sentiment() + compare_news_to_ensemble(), for backward
    compatibility with existing callers/tests that don't need the
    ensemble-vote use case — behavior here is identical to before the
    split, just internally composed from the two smaller functions."""
    news_sentiment = fetch_news_sentiment(ticker, news_service=news_service, fundamentals_service=fundamentals_service)
    return compare_news_to_ensemble(news_sentiment, ensemble_direction)
