"""
News interpretation engine (spec sections 7 & 8).

Two things live here:
  1. Contextual sentiment — not just keyword counting. Headlines are
     split on contrast conjunctions ("but", "however", "despite"...)
     and the clause AFTER the conjunction is weighted much more
     heavily, because in financial headlines that's almost always the
     part that actually moves the stock (e.g. "revenue increased but
     guidance was reduced" is bearish, driven by the guidance cut).
  2. Category classification and an impact score that combines
     sentiment strength with how market-moving that category tends
     to be (guidance/earnings moves markets more than a generic
     macro mention, for instance).

This is a transparent, lexicon-and-rules system — not a trained model.
It's deliberately documented as such so its limits are clear: it will
miss subtlety a human (or an LLM) would catch, and it's tuned for
common financial-news phrasing, not general sentiment.
"""
from __future__ import annotations

import re

CONTRAST_MARKERS = [
    " but ", " however ", " despite ", " although ", " while ",
    " yet ", " even as ", " though ",
]

POSITIVE_WORDS = {
    "beat": 2, "beats": 2, "beating": 2, "surge": 2, "surged": 2, "surges": 2,
    "record": 1.5, "strong": 1, "stronger": 1, "growth": 1, "grows": 1, "grew": 1,
    "upgrade": 2, "upgraded": 2, "outperform": 1.5, "raises": 1.5, "raised": 1.5,
    "raise": 1.5, "expands": 1, "expansion": 1, "partnership": 1, "approval": 1.5,
    "approved": 1.5, "buyback": 1, "profit": 1, "profitable": 1, "gain": 1,
    "gains": 1, "gained": 1, "rally": 1.5, "rallies": 1.5, "rallied": 1.5,
    "soar": 2, "soars": 2, "soared": 2, "jump": 1.5, "jumps": 1.5, "jumped": 1.5,
    "boost": 1, "boosts": 1, "boosted": 1, "exceeds": 1.5, "exceeded": 1.5,
    "win": 1, "wins": 1, "won": 1, "launch": 0.5, "launches": 0.5, "launched": 0.5,
    "innovation": 0.5, "breakthrough": 1.5, "bullish": 2,
    "increase": 1, "increased": 1, "increases": 1, "rose": 1, "rises": 1, "rising": 1,
    "improve": 1, "improved": 1, "improves": 1, "improving": 1, "higher": 0.5,
    "top": 0.5, "topped": 1, "upbeat": 1.5, "positive": 1, "success": 1, "successful": 1,
    # Added — real, common financial-journalism vocabulary that was
    # previously missing entirely, found by testing realistic headline
    # samples and identifying genuine sentiment-bearing words the
    # engine failed to recognize at all (scoring a false 0.0 for a
    # headline that clearly says something positive).
    "tailwind": 1, "tailwinds": 1, "accelerate": 1, "accelerates": 1, "accelerating": 1,
    "acceleration": 1, "resilient": 1, "resilience": 1, "robust": 1, "solid": 0.5,
    "momentum": 1, "outpace": 1, "outpaces": 1, "outpaced": 1, "premium": 0.5,
    "optimistic": 1, "optimism": 1, "thrive": 1, "thriving": 1, "thrives": 1,
    "milestone": 1, "expand margins": 1, "margin expansion": 1.5, "beat and raise": 2,
    "tops estimates": 1.5, "beats consensus": 1.5, "raises dividend": 1.5,
    "expanded buyback": 1.5, "guides higher": 1.5, "reaffirms guidance": 0.5,
    "outshine": 1, "outshines": 1, "outshined": 1, "blowout": 2, "surpass": 1.5,
    "surpassed": 1.5, "surpasses": 1.5, "upside": 1, "recovering": 1, "recovery": 1,
    "rebound": 1.5, "rebounded": 1.5, "rebounds": 1.5, "advantage": 0.5, "advantageous": 0.5,
}

NEGATIVE_WORDS = {
    "miss": 2, "misses": 2, "missed": 2, "plunge": 2, "plunged": 2, "plunges": 2,
    "cut": 1.5, "cuts": 1.5, "cutting": 1.5, "lawsuit": 1.5, "sued": 1.5, "sues": 1.5,
    "investigation": 1.5, "investigated": 1.5, "downgrade": 2, "downgraded": 2,
    "recall": 1.5, "recalled": 1.5, "layoffs": 1.5, "layoff": 1.5, "bankruptcy": 2.5,
    "delay": 1, "delayed": 1, "delays": 1, "warns": 1.5, "warned": 1.5, "warning": 1.5,
    "weak": 1, "weaker": 1, "decline": 1.5, "declined": 1.5, "declines": 1.5,
    "loss": 1.5, "losses": 1.5, "drop": 1.5, "drops": 1.5, "dropped": 1.5,
    "fall": 1, "falls": 1, "fell": 1, "plummet": 2, "plummeted": 2, "plummets": 2,
    "slump": 1.5, "slumps": 1.5, "slumped": 1.5, "crash": 2.5, "crashed": 2.5,
    "reduced": 1.5, "reduces": 1.5, "reduce": 1.5, "shortfall": 1.5, "probe": 1.5,
    "fraud": 2.5, "scandal": 2, "resign": 1.5, "resigned": 1.5, "resigns": 1.5,
    "fired": 1.5, "fires": 1.5, "bearish": 2, "underperform": 1.5, "concerns": 1,
    "concern": 1, "risk": 0.5, "risks": 0.5, "struggling": 1.5, "struggles": 1.5,
    "decrease": 1, "decreased": 1, "decreases": 1, "lower": 0.5, "lowered": 1.5,
    "lowers": 1.5, "worse": 1, "worsened": 1.5, "disappointing": 1.5, "disappoints": 1.5,
    "negative": 1, "underwhelm": 1, "underwhelming": 1, "halt": 1.5, "halted": 1.5,
    "suspend": 1.5, "suspended": 1.5, "default": 2, "downturn": 1.5, "turmoil": 1.5,
    # Added — same rationale as the positive-word additions above: real,
    # common negative financial-journalism vocabulary previously
    # missing, verified by testing to have previously scored a false
    # 0.0 despite clearly expressing negative sentiment.
    "headwind": 1, "headwinds": 1, "decelerate": 1, "decelerates": 1, "decelerating": 1,
    "deceleration": 1, "slowdown": 1, "slowing": 1, "sluggish": 1, "soften": 1,
    "softening": 1, "softened": 1, "tumble": 1.5, "tumbled": 1.5, "tumbles": 1.5,
    "impairment": 1.5, "write-down": 1.5, "writedown": 1.5, "restructuring": 1,
    "restructure": 1, "short seller": 1.5, "short report": 1.5, "activist short": 1.5,
    "antitrust": 1, "margin compression": 1.5, "margin pressure": 1.5,
    "miss and cut": 2, "misses consensus": 1.5, "cuts dividend": 1.5,
    "suspends dividend": 2, "guides lower": 1.5, "slashed": 1.5, "slashes": 1.5,
    "slashing": 1.5, "contracting": 1, "contraction": 1, "erode": 1, "eroded": 1,
    "erodes": 1, "erosion": 1, "downside": 1, "setback": 1, "setbacks": 1,
    "shrinking": 1, "shrank": 1, "shrinks": 1, "exodus": 1.5, "outflow": 1, "outflows": 1,
}

# Category keyword map, checked in order — first match wins.
CATEGORY_KEYWORDS = [
    ("earnings", ["earnings", "eps", "quarterly results", "q1", "q2", "q3", "q4",
                  "revenue", "beats estimates", "misses estimates"]),
    ("guidance", ["guidance", "outlook", "forecast raised", "forecast cut", "raises forecast", "cuts forecast"]),
    ("analyst_rating", ["upgrade", "downgrade", "initiates coverage", "reiterates", "overweight", "underweight"]),
    ("price_target", ["price target"]),
    ("acquisition", ["acquire", "acquisition", "merger", "to buy", "buyout", "takeover"]),
    ("partnership", ["partnership", "partners with", "collaborat", "joint venture"]),
    ("product_launch", ["unveils", "launches", "launch of", "new product", "announces new"]),
    ("lawsuit", ["lawsuit", "sues", "sued", "litigation", "court ruling"]),
    ("regulation", ["regulator", "regulation", "sec probe", "antitrust", "compliance", "fine imposed"]),
    ("management_change", ["ceo", "cfo", "resigns", "resignation", "steps down", "appoints", "names new"]),
    ("insider_activity", ["insider", "buys shares", "sells shares", "form 4"]),
    ("supply_chain", ["supply chain", "shortage", "chip shortage", "factory", "production halt"]),
    ("ai_development", [" ai ", "artificial intelligence", "machine learning", "chatgpt", "llm"]),
    ("geopolitical", ["tariff", "sanctions", "trade war", "geopolitical", "war", "conflict"]),
    ("competition", ["competitor", "competition", "market share", "rival"]),
    ("macro", ["fed", "federal reserve", "interest rate", "inflation", "jobs report", "gdp", "cpi"]),
]

# How strongly each category tends to move a stock, and over what horizon.
# These are documented editorial judgments, not fitted parameters.
CATEGORY_WEIGHT = {
    "earnings": (1.4, "short_term"),
    "guidance": (1.5, "medium_term"),
    "analyst_rating": (0.9, "short_term"),
    "price_target": (0.7, "short_term"),
    "acquisition": (1.6, "long_term"),
    "partnership": (1.0, "medium_term"),
    "product_launch": (0.9, "medium_term"),
    "lawsuit": (1.1, "medium_term"),
    "regulation": (1.2, "long_term"),
    "management_change": (1.0, "medium_term"),
    "insider_activity": (0.6, "short_term"),
    "supply_chain": (1.1, "medium_term"),
    "ai_development": (0.8, "long_term"),
    "geopolitical": (0.9, "medium_term"),
    "competition": (0.8, "long_term"),
    "macro": (0.7, "medium_term"),
    "other": (0.6, "short_term"),
}

# Source-quality hierarchy (spec section 15): primary/official sources
# outrank major wire services and financial press, which outrank
# everything else. This is metadata only — it never filters out a
# headline, just labels how much weight it should carry.
SOURCE_TIERS = {
    1: ["sec.gov", "sec edgar", "investor relations", "8-k", "10-k", "10-q", "press release"],
    2: ["reuters", "bloomberg", "cnbc", "wall street journal", "wsj", "financial times",
        "associated press", " ap ", "marketwatch", "barron's", "barrons"],
}


def classify_source_tier(source: str) -> int:
    """Returns 1 (primary), 2 (major wire/financial press), or 3 (other).
    Case-insensitive substring match against a known-source list —
    unrecognized sources default to tier 3 rather than being guessed at."""
    if not source:
        return 3
    lower = f" {source.lower()} "
    for tier, names in SOURCE_TIERS.items():
        if any(name in lower for name in names):
            return tier
    return 3


def classify_category(headline: str) -> str:
    text = f" {headline.lower()} "
    for category, keywords in CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in text:
                return category
    return "other"


def _clause_sentiment(text: str) -> float:
    """Weighted lexicon score for one clause: sum of positive weights
    minus sum of negative weights, with basic negation handling for
    'not', "n't", and 'no'.

    Two passes, deliberately in this order: multi-word PHRASES first
    (matched via substring search against the raw text — real bug
    found via testing: a naive single-word tokenizer can NEVER match a
    dictionary key like "margin pressure", since it splits text into
    individual words before checking the dictionary at all, silently
    making every multi-word entry in POSITIVE_WORDS/NEGATIVE_WORDS dead
    weight that never fires), then single WORDS on whatever text is
    left after matched phrases are removed — so a phrase and its own
    constituent words are never double-counted.

    Known, documented limitation: negation handling only applies to
    single words in the second pass, not to phrases in the first
    ("did not see margin pressure" would still count "margin pressure"
    as negative) — a real gap, not silently ignored, but a smaller one
    than the phrases not firing at all."""
    lower_text = text.lower()
    score = 0.0
    remaining_text = lower_text

    phrase_entries = (
        [(phrase, weight, True) for phrase, weight in POSITIVE_WORDS.items() if " " in phrase]
        + [(phrase, weight, False) for phrase, weight in NEGATIVE_WORDS.items() if " " in phrase]
    )
    # Longest phrases first, so a longer, more specific phrase is
    # consumed before a shorter one that might be its substring.
    for phrase, weight, is_positive in sorted(phrase_entries, key=lambda p: -len(p[0])):
        if phrase in remaining_text:
            score += weight if is_positive else -weight
            remaining_text = remaining_text.replace(phrase, " ")

    words = re.findall(r"[a-z']+", remaining_text)
    for i, w in enumerate(words):
        prev_negator = i > 0 and (words[i - 1] in ("not", "no", "never") or words[i - 1].endswith("n't"))
        if w in POSITIVE_WORDS:
            weight = POSITIVE_WORDS[w]
        elif w in NEGATIVE_WORDS:
            weight = -NEGATIVE_WORDS[w]
        else:
            continue
        score += -weight if prev_negator else weight
    return score


def analyze_sentiment(headline: str) -> dict:
    """Returns a dict with a -100..100 sentiment score, a label, and
    a short trace of which clause(s) drove the result — so the
    reasoning is inspectable, not a black box."""
    # Pad consistently so index positions line up between the lowercased
    # matching string and the string we actually slice.
    padded = f" {headline} "
    lower = padded.lower()

    contrast_hit = next((m for m in CONTRAST_MARKERS if m in lower), None)

    if contrast_hit:
        idx = lower.index(contrast_hit)
        before = padded[:idx]
        after = padded[idx + len(contrast_hit):]
        before_score = _clause_sentiment(before)
        after_score = _clause_sentiment(after)
        # The clause after the contrast marker dominates (70/30 split),
        # matching how markets typically react to "beat but guidance cut"
        # style headlines.
        combined = 0.3 * before_score + 0.7 * after_score
        reasoning = (
            f"Contrast detected ('{contrast_hit.strip()}') — weighting the clause "
            f"after it more heavily: \"{after.strip()}\""
        )
    else:
        combined = _clause_sentiment(headline)
        reasoning = "No contrast conjunction detected; scored the full headline."

    # Squash into -100..100 with a soft cap so a couple of strong words
    # don't automatically max out the scale.
    scaled = max(-100, min(100, combined * 18))

    if scaled >= 40:
        label = "strong_positive"
    elif scaled >= 12:
        label = "moderate_positive"
    elif scaled <= -40:
        label = "strong_negative"
    elif scaled <= -12:
        label = "moderate_negative"
    else:
        label = "neutral"

    return {"score": round(scaled, 1), "label": label, "reasoning": reasoning}


def score_news_impact(headline: str) -> dict:
    """Combines sentiment with category weighting into a single -100..100
    impact score plus a horizon, per spec section 8."""
    sentiment = analyze_sentiment(headline)
    category = classify_category(headline)
    weight, horizon = CATEGORY_WEIGHT.get(category, CATEGORY_WEIGHT["other"])

    impact_score = max(-100, min(100, sentiment["score"] * weight))

    if impact_score >= 50:
        impact_label = "strong_positive"
    elif impact_score >= 15:
        impact_label = "moderate_positive"
    elif impact_score <= -50:
        impact_label = "strong_negative"
    elif impact_score <= -15:
        impact_label = "moderate_negative"
    else:
        impact_label = "neutral"

    direction = "improves" if impact_score > 0 else "weakens" if impact_score < 0 else "has limited effect on"
    explanation = (
        f"Classified as '{category.replace('_', ' ')}' news (typically a {horizon.replace('_', ' ')} driver). "
        f"{sentiment['reasoning']} Net effect: this {direction} the near-term outlook "
        f"(impact score {impact_score:+.0f})."
    )

    return {
        "category": category,
        "sentiment": sentiment,
        "impact_score": round(impact_score, 1),
        "impact_label": impact_label,
        "horizon": horizon,
        "explanation": explanation,
    }
