from backend.news.sentiment import analyze_sentiment, classify_category, score_news_impact


def test_contrast_conjunction_flips_to_bearish():
    """The exact example from the spec: positive-sounding word before
    'but', negative clause after it — must net out bearish."""
    result = analyze_sentiment("Company revenue increased but guidance was reduced")
    assert result["score"] < 0
    assert result["label"] in ("moderate_negative", "strong_negative")


def test_unambiguous_positive_headline():
    result = analyze_sentiment("Company beats earnings estimates, raises full-year guidance")
    assert result["score"] > 0
    assert result["label"] in ("moderate_positive", "strong_positive")


def test_unambiguous_negative_headline():
    result = analyze_sentiment("Stock plunges after CEO resigns amid fraud investigation")
    assert result["score"] < 0
    assert result["label"] == "strong_negative"


def test_double_negative_is_not_strongly_negative():
    result = analyze_sentiment("Regulators are not investigating the company")
    assert result["score"] >= 0


def test_neutral_headline_has_small_magnitude():
    result = analyze_sentiment("Company to present at investor conference next week")
    assert abs(result["score"]) < 20


def test_category_classification():
    assert classify_category("Company beats Q2 earnings estimates") == "earnings"
    assert classify_category("Analyst downgrades stock to underweight") == "analyst_rating"
    assert classify_category("Company to acquire smaller rival in $2B deal") == "acquisition"
    assert classify_category("Fed raises interest rates by 25bps") == "macro"


def test_impact_score_bounded():
    for headline in [
        "Company beats earnings estimates, raises full-year guidance",
        "Stock plunges after CEO resigns amid fraud investigation",
        "Company to present at investor conference next week",
    ]:
        impact = score_news_impact(headline)
        assert -100 <= impact["impact_score"] <= 100
        assert impact["horizon"] in ("short_term", "medium_term", "long_term")
        assert isinstance(impact["explanation"], str) and len(impact["explanation"]) > 10


def test_multiword_phrase_actually_fires():
    """Real, significant bug found via testing: the matcher tokenized
    text into single words BEFORE checking the dictionary, meaning a
    multi-word dictionary key like "margin pressure" could never match
    anything — it would silently sit in the dictionary as permanently
    dead weight. Direct proof this is now fixed: a sentence containing
    the phrase must score more negatively than the same sentence with
    the phrase removed."""
    from backend.news.sentiment import _clause_sentiment
    with_phrase = _clause_sentiment("analysts cite margin pressure as costs rise")
    without_phrase = _clause_sentiment("analysts cite costs rise")
    assert with_phrase < without_phrase


def test_phrase_and_its_own_words_are_not_double_counted():
    """A phrase like 'beat and raise' contains the words 'beat' and
    'raise', which are ALSO individually in the dictionary — the
    phrase must be scored once, not once as a phrase AND again as its
    separate constituent words."""
    from backend.news.sentiment import _clause_sentiment, POSITIVE_WORDS
    score = _clause_sentiment("the company had a beat and raise quarter")
    phrase_weight = POSITIVE_WORDS["beat and raise"]
    assert score == phrase_weight  # exactly the phrase weight, nothing added on top


def test_expanded_vocabulary_catches_previously_missed_real_sentiment():
    """Direct, concrete evidence the vocabulary expansion works: real,
    common financial phrases that scored a false 0.0 before this
    change (because the words genuinely weren't in the dictionary at
    all, not because the headline lacked real sentiment) must now
    score non-zero in the correct direction."""
    negative_headlines = [
        "Company faces headwinds as demand softens in key markets",
        "Stock tumbles after firm slashes full-year outlook",
    ]
    positive_headlines = [
        "Business shows robust momentum heading into next quarter",
        "Firm reports resilient results despite a challenging environment",
    ]
    for h in negative_headlines:
        assert score_news_impact(h)["impact_score"] < 0, f"Expected negative for: {h}"
    for h in positive_headlines:
        assert score_news_impact(h)["impact_score"] > 0, f"Expected positive for: {h}"


def test_genuinely_routine_headlines_still_correctly_score_zero():
    """The vocabulary expansion must not create FALSE signal where
    none exists — a genuinely routine, non-event headline should still
    score exactly 0.0, the same honest 'no signal' read as before."""
    routine_headlines = [
        "Company announces board appointment",
        "Company to present at upcoming investor conference",
        "Analysts maintain neutral rating on the stock ahead of earnings",
    ]
    for h in routine_headlines:
        assert score_news_impact(h)["impact_score"] == 0.0, f"Expected genuine no-signal for: {h}"


def test_longer_phrases_matched_before_shorter_overlapping_ones():
    """A defensive correctness check: when a longer phrase and a
    shorter one could both technically match overlapping text, the
    longer, more specific phrase should be the one that actually
    fires, not an accidental partial/shorter match."""
    from backend.news.sentiment import _clause_sentiment
    # "beat and raise" (longer, specific) contains "raise" (shorter,
    # already a separate dictionary entry) as a substring context.
    score = _clause_sentiment("company delivers a beat and raise quarter")
    from backend.news.sentiment import POSITIVE_WORDS
    assert score == POSITIVE_WORDS["beat and raise"]
