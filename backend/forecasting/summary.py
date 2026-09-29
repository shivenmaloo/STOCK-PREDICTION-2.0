"""
Plain-English prediction summary — one readable paragraph synthesizing
everything the app computed (ensemble read, technical trend, relative
strength, news context, risk sizing) into a single narrative, written
for someone with no ML or finance background. This is the answer to
"what does this even mean" — everything else in the prediction bundle
(model votes, walk-forward accuracy tables, feature importance) stays
available for anyone who wants to dig in, but this is the one thing
meant to be read first, in full, without translation.
"""
from __future__ import annotations

SIGNAL_PLAIN_LANGUAGE = {
    "STRONG_BULLISH_SETUP": "a strong bullish setup",
    "BULLISH_WATCH": "a moderate bullish lean",
    "NEUTRAL": "no clear direction",
    "BEARISH_WATCH": "a moderate bearish lean",
    "HIGH_RISK_POSSIBLE_EXIT": "a high-risk setup worth reviewing",
}


def build_plain_summary(ticker: str, horizon_days: int, signal: dict, ensemble: dict,
                         news_context: dict, position_sizing: dict, relaxed_mode: bool,
                         out_of_distribution: dict) -> str:
    plain_signal = SIGNAL_PLAIN_LANGUAGE.get(signal["signal"], signal["signal"])
    direction = ensemble["direction"]
    confidence_pct = round(ensemble["confidence"] * 100)
    expected_return = ensemble["expected_return_pct"]
    agreement = ensemble["model_agreement"]

    sentences = []

    # Opening: what the app thinks, in one plain, internally consistent
    # sentence. Three distinct cases, not two — a signal can land on
    # NEUTRAL either because the models genuinely conflict, or because
    # they loosely lean one way without clearing the bar to count as a
    # real signal. Conflating those produced a contradictory-sounding
    # summary ("reads as no clear direction... more likely to rise") —
    # found via testing, fixed by branching on the actual reason.
    if ensemble["low_confidence_disagreement"]:
        sentences.append(
            f"For {ticker} over the next {horizon_days} trading day{'s' if horizon_days != 1 else ''}, "
            f"the models genuinely disagree with each other, so this shows up as no clear direction "
            f"rather than a confident call one way or the other."
        )
    elif signal["signal"] == "NEUTRAL":
        lean = "bullish" if direction == "bullish" else "bearish" if direction == "bearish" else "flat"
        if lean == "flat":
            sentences.append(
                f"For {ticker} over the next {horizon_days} trading day{'s' if horizon_days != 1 else ''}, "
                f"the models don't see a meaningful move coming either way — expected change is close to flat."
            )
        else:
            sentences.append(
                f"For {ticker} over the next {horizon_days} trading day{'s' if horizon_days != 1 else ''}, "
                f"the models lean slightly {lean} ({agreement} agree, expected move about {expected_return:+.1f}%), "
                f"but not strongly enough to count as a real signal — treated as neutral rather than overstating a weak lean."
            )
    else:
        move = "rise" if direction == "bullish" else "fall"
        sentences.append(
            f"For {ticker} over the next {horizon_days} trading day{'s' if horizon_days != 1 else ''}, "
            f"this reads as {plain_signal}: {agreement} of the models used agree the price is more likely "
            f"to {move}, with an expected move of about {expected_return:+.1f}% and {confidence_pct}% confidence."
        )

    # Why: the reasoning behind the signal tier (technical/news agreement or lack thereof).
    sentences.append(signal["reason"])

    # News, if available — ALWAYS mentioned, not just when there's a
    # clean agree/disagree comparison to make. Real gap found via
    # testing: when the model itself lands on neutral, agrees_with_model
    # is None (there's no direction to compare against), and the news
    # sentence was being silently dropped entirely — even though real
    # news data existed and the person asked to see it as an
    # independent factor. Now it's always shown, phrased appropriately
    # for whichever comparison actually applies.
    if news_context.get("available"):
        headline_summary = news_context["summary"].split("—")[0].strip()
        if news_context.get("agrees_with_model") is True:
            sentences.append(f"Recent news coverage supports this — {headline_summary}.")
        elif news_context.get("agrees_with_model") is False:
            sentences.append(f"Worth noting: recent news actually points the other way — {headline_summary}.")
        else:
            sentences.append(f"For additional context, {headline_summary.lower()}.")
    else:
        sentences.append("No recent news was available to cross-check this against.")

    # Risk sizing, translated into plain terms.
    if position_sizing.get("suggested_position_pct") is not None:
        pct = position_sizing["suggested_position_pct"]
        if pct > 0:
            sentences.append(
                f"Based on how this model has actually performed on real, past predictions for this "
                f"ticker, a risk-sized position would be around {pct}% of a portfolio — this already "
                f"accounts for the model's real track record, not just this one prediction."
            )
        else:
            sentences.append("The model's own track record doesn't currently support sizing a position on this one.")
    else:
        sentences.append("There isn't yet enough validated trade history to suggest a position size for this ticker.")

    # Caveats, always last, always plain.
    caveats = []
    if relaxed_mode:
        caveats.append("this ticker has less trading history than usual, so treat it as lower-confidence than a well-established stock")
    if out_of_distribution.get("is_out_of_distribution"):
        caveats.append("current market conditions look unusual compared to what the model trained on, so extra caution is warranted")
    if caveats:
        sentences.append("One more thing: " + "; and ".join(caveats) + ".")

    sentences.append("This is a probabilistic estimate from historical patterns, not a guarantee — it's information to weigh, not instructions to follow.")

    return " ".join(sentences)
