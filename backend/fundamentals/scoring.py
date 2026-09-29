"""
Fundamental scoring (spec section 6).

A transparent, rules-based 0-100 score built from four sub-scores —
growth, profitability, valuation, financial health — each worth up to
25 points. Every point awarded is explainable; nothing here is a
black-box model.

This deliberately does NOT compare against live sector/industry
averages (that would require pulling fundamentals for every peer on
every request, which the flaky fundamentals provider can't support
reliably). Instead it scores against fixed, documented thresholds
that are reasonable for large/mid-cap US equities — the explanation
text says so explicitly rather than implying a peer comparison that
isn't happening.
"""
from __future__ import annotations

from typing import Optional


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _score_growth(data: dict) -> tuple[float, list[str]]:
    notes = []
    score = 0.0
    rev_g = data.get("revenue_growth")
    eps_g = data.get("earnings_growth")

    if rev_g is not None:
        # 0% growth -> 0 pts, 20%+ growth -> full 12.5 pts, negative growth -> 0
        pts = _clip(rev_g / 0.20, 0, 1) * 12.5
        score += pts
        notes.append(f"revenue growth of {rev_g * 100:.1f}%")
    if eps_g is not None:
        pts = _clip(eps_g / 0.20, 0, 1) * 12.5
        score += pts
        notes.append(f"earnings growth of {eps_g * 100:.1f}%")

    return score, notes


def _score_profitability(data: dict) -> tuple[float, list[str]]:
    notes = []
    score = 0.0
    net_margin = data.get("net_margin")
    roe = data.get("return_on_equity")

    if net_margin is not None:
        # 0% margin -> 0, 20%+ margin -> full 12.5
        pts = _clip(net_margin / 0.20, 0, 1) * 12.5
        score += pts
        notes.append(f"net margin of {net_margin * 100:.1f}%")
    if roe is not None:
        pts = _clip(roe / 0.25, 0, 1) * 12.5
        score += pts
        notes.append(f"return on equity of {roe * 100:.1f}%")

    return score, notes


def _score_valuation(data: dict) -> tuple[float, list[str]]:
    notes = []
    score = 0.0
    pe = data.get("trailing_pe")
    ps = data.get("price_to_sales")

    if pe is not None and pe > 0:
        # Lower P/E scores higher, full marks at P/E<=15, zero at P/E>=50.
        pts = _clip((50 - pe) / (50 - 15), 0, 1) * 12.5
        score += pts
        notes.append(f"trailing P/E of {pe:.1f}")
    if ps is not None and ps > 0:
        pts = _clip((10 - ps) / (10 - 1), 0, 1) * 12.5
        score += pts
        notes.append(f"price/sales of {ps:.1f}")

    return score, notes


def _score_financial_health(data: dict) -> tuple[float, list[str]]:
    notes = []
    score = 0.0
    d2e = data.get("debt_to_equity")
    current_ratio = data.get("current_ratio")

    if d2e is not None:
        # d2e is often reported as a percentage-like number (e.g. 45 = 0.45x)
        d2e_ratio = d2e / 100 if d2e > 5 else d2e
        pts = _clip((2.0 - d2e_ratio) / 2.0, 0, 1) * 12.5
        score += pts
        notes.append(f"debt/equity of {d2e_ratio:.2f}x")
    if current_ratio is not None:
        pts = _clip(current_ratio / 2.0, 0, 1) * 12.5
        score += pts
        notes.append(f"current ratio of {current_ratio:.2f}")

    return score, notes


def compute_fundamental_score(data: dict) -> dict:
    """Returns {score, max_possible, breakdown, explanation}. `score`
    is scaled to 0-100 based on however many sub-metrics were actually
    available — a ticker missing half its fields isn't unfairly
    punished with zeros for missing data."""

    growth_pts, growth_notes = _score_growth(data)
    profit_pts, profit_notes = _score_profitability(data)
    value_pts, value_notes = _score_valuation(data)
    health_pts, health_notes = _score_financial_health(data)

    raw_total = growth_pts + profit_pts + value_pts + health_pts

    # How many of the 8 individual metrics were actually available?
    available_metrics = sum([
        data.get("revenue_growth") is not None,
        data.get("earnings_growth") is not None,
        data.get("net_margin") is not None,
        data.get("return_on_equity") is not None,
        data.get("trailing_pe") is not None and (data.get("trailing_pe") or 0) > 0,
        data.get("price_to_sales") is not None and (data.get("price_to_sales") or 0) > 0,
        data.get("debt_to_equity") is not None,
        data.get("current_ratio") is not None,
    ])

    if available_metrics == 0:
        return {
            "score": None,
            "coverage": "0/8 metrics available",
            "breakdown": {"growth": None, "profitability": None, "valuation": None, "financial_health": None},
            "explanation": "Not enough fundamental data was available to compute a score for this ticker.",
        }

    max_possible = available_metrics * 12.5
    scaled_score = round((raw_total / max_possible) * 100, 1) if max_possible > 0 else None

    all_notes = growth_notes + profit_notes + value_notes + health_notes
    if scaled_score is None:
        explanation = "Not enough fundamental data was available to compute a score for this ticker."
    else:
        tier = (
            "strong" if scaled_score >= 70 else
            "middling" if scaled_score >= 40 else
            "weak"
        )
        detail = "; ".join(all_notes) if all_notes else "limited underlying data"
        explanation = (
            f"Fundamental score of {scaled_score}/100 ({tier}), based on {detail}. "
            f"Scored against fixed thresholds for large/mid-cap US equities "
            f"(not a live peer/sector comparison), using {available_metrics}/8 available metrics."
        )

    return {
        "score": scaled_score,
        "coverage": f"{available_metrics}/8 metrics available",
        "breakdown": {
            "growth": round(growth_pts, 1),
            "profitability": round(profit_pts, 1),
            "valuation": round(value_pts, 1),
            "financial_health": round(health_pts, 1),
        },
        "explanation": explanation,
    }
