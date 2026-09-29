"""
Extracts real financial-statement figures from SEC's XBRL companyfacts
payload and compares them across periods (spec section 1 & 6) —
"Revenue grew 21% YoY, but operating margin declined from 27% to 24%"
style analysis, generated from actual filed numbers, not estimated.

Every figure returned carries the accession number and form it came
from, so the frontend can link straight back to the source filing.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

# Different companies tag the same concept differently in XBRL — this
# tries each candidate tag in order and uses the first one with data.
METRIC_TAGS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "sbc": ["ShareBasedCompensation"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}

DOLLAR_METRICS = {"revenue", "gross_profit", "operating_income", "net_income", "cash",
                   "long_term_debt", "operating_cash_flow", "capex", "rd_expense", "sbc"}


def _extract_series(facts: dict, metric: str, forms: tuple[str, ...] = ("10-K", "10-Q")) -> list[dict]:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in METRIC_TAGS.get(metric, []):
        concept = us_gaap.get(tag)
        if not concept:
            continue
        units = concept.get("units", {})
        # Dollar metrics are tagged USD; per-share metrics USD/shares; share counts "shares".
        unit_key = next((u for u in units if u in ("USD", "USD/shares", "shares")), None)
        if unit_key is None:
            continue
        points = units[unit_key]
        filtered = [p for p in points if p.get("form") in forms and p.get("val") is not None]
        if filtered:
            # Dedupe by (end, val) — restatements can produce multiple
            # entries for the same period; keep the most recently filed.
            best_by_end: dict[str, dict] = {}
            for p in filtered:
                end = p.get("end")
                if end not in best_by_end or p.get("filed", "") > best_by_end[end].get("filed", ""):
                    best_by_end[end] = p
            series = sorted(best_by_end.values(), key=lambda p: p["end"])
            return series
    return []


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / abs(old) * 100, 1)


def _find_yoy_comparison(series: list[dict], latest: dict) -> Optional[dict]:
    """Finds the data point ~1 year before `latest`'s period end, for a
    same-quarter (or same-year, for annual figures) YoY comparison."""
    try:
        latest_end = datetime.fromisoformat(latest["end"]).date()
    except (ValueError, KeyError):
        return None
    target = latest_end.replace(year=latest_end.year - 1)
    best, best_diff = None, None
    for p in series:
        try:
            end = datetime.fromisoformat(p["end"]).date()
        except (ValueError, KeyError):
            continue
        diff = abs((end - target).days)
        if diff <= 20 and (best_diff is None or diff < best_diff):
            best, best_diff = p, diff
    return best


def build_financial_trend(facts: dict, cik: int) -> dict:
    """Returns per-metric series (annual+quarterly) plus derived margins,
    QoQ/YoY changes, and a plain-English narrative of what moved."""
    metrics: dict[str, list[dict]] = {}
    for metric in METRIC_TAGS:
        series = _extract_series(facts, metric)
        for p in series:
            accession_nodash = p.get("accn", "").replace("-", "")
            p["source_url"] = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/"
                if accession_nodash else None
            )
        metrics[metric] = series

    revenue_series = metrics.get("revenue", [])
    if not revenue_series:
        return {
            "metrics": metrics,
            "latest_period": None,
            "narrative": [],
            "note": "No revenue data found in SEC XBRL facts for this company — it may report under an uncommon tag, or SEC data may not be available for it.",
        }

    latest = revenue_series[-1]
    latest_end = latest["end"]

    def latest_value(metric: str) -> Optional[dict]:
        series = metrics.get(metric, [])
        return next((p for p in reversed(series) if p["end"] == latest_end), None)

    def yoy_value(metric: str) -> Optional[dict]:
        series = metrics.get(metric, [])
        current = latest_value(metric)
        if not current:
            return None
        return _find_yoy_comparison(series, current)

    narrative = []
    facts_summary = {}

    rev_latest = latest_value("revenue")
    rev_yoy = yoy_value("revenue")
    if rev_latest and rev_yoy:
        rev_growth = _pct_change(rev_latest["val"], rev_yoy["val"])
        facts_summary["revenue_yoy_growth_pct"] = rev_growth
        if rev_growth is not None:
            direction = "grew" if rev_growth >= 0 else "declined"
            narrative.append(f"Revenue {direction} {abs(rev_growth):.1f}% YoY to ${rev_latest['val']:,.0f}.")

    op_latest = latest_value("operating_income")
    op_yoy = yoy_value("operating_income")
    if op_latest and rev_latest:
        op_margin_now = round(op_latest["val"] / rev_latest["val"] * 100, 1) if rev_latest["val"] else None
        facts_summary["operating_margin_pct"] = op_margin_now
        if op_yoy and rev_yoy and rev_yoy["val"]:
            op_margin_prior = round(op_yoy["val"] / rev_yoy["val"] * 100, 1)
            facts_summary["operating_margin_pct_prior_year"] = op_margin_prior
            if op_margin_now is not None:
                delta = round(op_margin_now - op_margin_prior, 1)
                trend = "expanded" if delta > 0 else "declined" if delta < 0 else "held steady"
                narrative.append(
                    f"Operating margin {trend} from {op_margin_prior}% to {op_margin_now}% "
                    f"({delta:+.1f} percentage points YoY)."
                )

    net_latest = latest_value("net_income")
    net_yoy = yoy_value("net_income")
    if net_latest and net_yoy:
        net_growth = _pct_change(net_latest["val"], net_yoy["val"])
        if net_growth is not None:
            facts_summary["net_income_yoy_growth_pct"] = net_growth
            direction = "grew" if net_growth >= 0 else "declined"
            narrative.append(f"Net income {direction} {abs(net_growth):.1f}% YoY.")

    fcf_latest_ocf = latest_value("operating_cash_flow")
    fcf_latest_capex = latest_value("capex")
    if fcf_latest_ocf and fcf_latest_capex:
        fcf = fcf_latest_ocf["val"] - abs(fcf_latest_capex["val"])
        facts_summary["free_cash_flow"] = round(fcf, 0)
        narrative.append(f"Free cash flow (operating cash flow minus capex) was ${fcf:,.0f} for the period.")

    debt_latest = latest_value("long_term_debt")
    cash_latest = latest_value("cash")
    debt_yoy = yoy_value("long_term_debt")
    if debt_latest and debt_yoy:
        debt_change = _pct_change(debt_latest["val"], debt_yoy["val"])
        if debt_change is not None and abs(debt_change) >= 10:
            direction = "increased" if debt_change > 0 else "decreased"
            narrative.append(f"Long-term debt {direction} {abs(debt_change):.1f}% YoY.")

    sbc_latest = latest_value("sbc")
    sbc_yoy = yoy_value("sbc")
    if sbc_latest and sbc_yoy:
        sbc_change = _pct_change(sbc_latest["val"], sbc_yoy["val"])
        if sbc_change is not None and abs(sbc_change) >= 15:
            direction = "increased" if sbc_change > 0 else "decreased"
            narrative.append(f"Stock-based compensation {direction} {abs(sbc_change):.1f}% YoY — worth watching for share dilution.")

    return {
        "metrics": metrics,
        "latest_period": latest_end,
        "facts_summary": facts_summary,
        "narrative": narrative,
        "note": None,
    }
