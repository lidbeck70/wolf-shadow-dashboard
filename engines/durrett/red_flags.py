"""
engines/durrett/red_flags.py — RED FLAG ENGINE (SPEC §20).

Varje flagga: {flag, severity, reason, source, data, date}. Trösklarna
ligger i config.red_flag_thresholds. En flagga sätts bara när talet
finns — saknat tal ger ingen flagga (det syns i missing_data). Red Flag
Score = 100 − avdrag per flagga (config.RED_FLAG_PENALTY), golv 0.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from engines.durrett import config as dc
from engines.durrett._base import Ctx
from engines.durrett.models import RedFlag, Score


def _src(ctx: Ctx, key: str) -> tuple:
    p = ctx.c.get(key)
    if p is None:
        return "", None
    return p.source or "", p.pub_date or p.data_date


def detect(ctx: Ctx, company_type: str, ss: dict, res: dict, pm: dict, scores: dict, cases: dict,
           today: Optional[date] = None) -> list:
    t = ctx.cfg["red_flag_thresholds"]
    today = today or date.today()
    flags: list = []

    def add(flag, sev, reason, key="", data=""):
        src, when = _src(ctx, key) if key else ("beräknat", None)
        flags.append(RedFlag(flag, sev, reason, src, data, when))

    price = ctx.num("commodity_price", required=False)
    aisc = ctx.num("aisc", required=False)
    pre = company_type in (dc.DEVELOPER, dc.EXPLORER)

    # High AISC
    if price and aisc is not None and company_type in (dc.PRODUCER, dc.HYBRID):
        r = aisc / price
        if r >= t["aisc_to_price_high"]:
            add("High AISC", dc.HIGH, f"AISC/pris {r:.2f} ≥ {t['aisc_to_price_high']}", "aisc", f"AISC {aisc:g}, pris {price:g}")
        elif r >= t["aisc_to_price_medium"]:
            add("High AISC", dc.MEDIUM, f"AISC/pris {r:.2f} ≥ {t['aisc_to_price_medium']}", "aisc", f"AISC {aisc:g}, pris {price:g}")

    # Low IRR / Low NPV/CAPEX / Huge CAPEX
    irr = ctx.num("irr_pct", required=False)
    if irr is not None and pre:
        if irr < t["irr_critical_pct"]:
            add("Low IRR", dc.HIGH, f"IRR {irr:g} % < {t['irr_critical_pct']:g} %", "irr_pct", f"{irr:g} %")
        elif irr < t["irr_low_pct"]:
            add("Low IRR", dc.MEDIUM, f"IRR {irr:g} % < {t['irr_low_pct']:g} %", "irr_pct", f"{irr:g} %")
    nc = pm.get("npv_capex")
    if nc is not None:
        if nc < t["npv_capex_low"]:
            add("Low NPV/CAPEX", dc.HIGH, f"NPV/CapEx {nc:.2f} < {t['npv_capex_low']:g} (Durrett-heuristik)", "npv_musd", f"{nc:.2f}×")
        elif nc < t["npv_capex_ok"]:
            add("Low NPV/CAPEX", dc.MEDIUM, f"NPV/CapEx {nc:.2f} i 1–2-intervallet (Durrett-heuristik)", "npv_musd", f"{nc:.2f}×")
    capex = ctx.num("capex_musd", required=False)
    mcap = ss.get("mcap_usd")
    if capex is not None and mcap:
        r = capex / mcap
        if r >= t["capex_to_mcap_huge"]:
            add("Huge CAPEX", dc.HIGH, f"initial CapEx {capex:g} MUSD = {r:.1f}× börsvärdet", "capex_musd", f"{capex:g} MUSD")
        elif r >= t["capex_to_mcap_large"]:
            add("Huge CAPEX", dc.MEDIUM, f"initial CapEx {capex:g} MUSD = {r:.1f}× börsvärdet", "capex_musd", f"{capex:g} MUSD")

    # Weak balance sheet / Short cash runway / Financing cliff
    nde = ctx.num("net_debt_ebitda", required=False)
    if nde is not None:
        if nde >= t["net_debt_ebitda_high"]:
            add("Weak balance sheet", dc.HIGH, f"nettoskuld/EBITDA {nde:.1f}×", "net_debt_ebitda", f"{nde:.2f}×")
        elif nde >= t["net_debt_ebitda_medium"]:
            add("Weak balance sheet", dc.MEDIUM, f"nettoskuld/EBITDA {nde:.1f}×", "net_debt_ebitda", f"{nde:.2f}×")
    rw = (ctx.metrics.get("cash_runway_years") or {}).get("value")
    if rw is not None:
        if rw < t["runway_short_years"]:
            add("Short cash runway", dc.HIGH, f"runway {rw:.1f} år", "cash_musd", f"{rw:.1f} år")
        elif rw < t["runway_medium_years"]:
            add("Short cash runway", dc.MEDIUM, f"runway {rw:.1f} år", "cash_musd", f"{rw:.1f} år")
    gap = (ctx.metrics.get("funding_cliff_gap") or {}).get("value")
    if gap is not None and gap > 0:
        add("Financing cliff", dc.HIGH, f"{gap:g} MUSD saknas före {ctx.metrics['funding_cliff_gap'].get('note', 'nästa milstolpe')}",
            "milestone_cost_musd", f"{gap:g} MUSD")
    fin = ctx.truth("financing_committed", required=False)
    if pre and capex is not None and mcap and capex / mcap >= t["capex_to_mcap_large"] and fin is False:
        add("Financing required before construction", dc.HIGH, "stor CapEx utan åtagen finansiering", "financing_committed",
            f"CapEx {capex:g} MUSD, kassa {ss.get('cash_usd') or 0:g}")

    # Dilution
    d1, d3 = ss.get("dilution_1y"), ss.get("dilution_3y")
    if ss.get("serial_diluter"):
        add("Serial diluter", dc.CRITICAL, f"{d3:+.0f} % fler aktier på 3 år", "shares_3y_ago_m", f"{d3:+.0f} %")
    elif d3 is not None and d3 >= t["dilution_3y_high_pct"]:
        add("High dilution", dc.HIGH, f"{d3:+.0f} % på 3 år", "shares_3y_ago_m", f"{d3:+.0f} %")
    elif d1 is not None and d1 >= t["dilution_1y_high_pct"]:
        add("High dilution", dc.MEDIUM, f"{d1:+.0f} % senaste året", "shares_1y_ago_m", f"{d1:+.0f} %")

    # Ownership / management
    own = ctx.num("insider_ownership_pct", required=False)
    if own is not None and own < t["insider_ownership_low_pct"]:
        add("Low insider ownership", dc.MEDIUM, f"insynsägande {own:g} % < {t['insider_ownership_low_pct']:g} %", "insider_ownership_pct", f"{own:g} %")
    tv = ctx.text("mgmt_track_verified")
    ms = scores.get("management")
    if ms is not None and ms.value is not None and ms.value < 40:
        add("Weak management track record", dc.HIGH if ms.value < 25 else dc.MEDIUM, f"Management Score {ms.value:g}", "mgmt_track_verified", tv or "")
    elif tv == "claimed":
        add("Unverified management claims", dc.LOW, "meriter bara ur bolagets egen presentation", "mgmt_track_verified", "claimed")

    # Jurisdiction / permitting / infrastructure
    for key, flag in (("risk_permitting", "Permitting risk"), ("risk_infrastructure", "Infrastructure risk"),
                      ("risk_nationalization", "Jurisdiction risk"), ("risk_community", "Community risk")):
        v = ctx.num(key, required=False)
        if v is not None and v == 0:
            add(flag, dc.HIGH, f"{key} bedömd 0/2 (hög risk)", key, "0/2")
        elif v is not None and v == 1:
            add(flag, dc.LOW, f"{key} bedömd 1/2", key, "1/2")
    cs = (ctx.metrics.get("country_risk_score") or {}).get("value")
    if cs is not None and cs < 50:
        add("Jurisdiction risk", dc.HIGH, f"landrisk {cs:g}/100 i repots tabell", "country", f"{cs:g}")

    # Metallurgy / strip / grade / royalties / streams
    rec = ctx.num("recovery_pct", required=False)
    if rec is not None and rec < t["recovery_low_pct"]:
        add("Metallurgy risk", dc.MEDIUM, f"recovery {rec:g} % < {t['recovery_low_pct']:g} %", "recovery_pct", f"{rec:g} %")
    mc = ctx.num("metallurgy_confidence", required=False)
    if mc is not None and mc == 0:
        add("Metallurgy risk", dc.MEDIUM, "metallurgin bara testad i labb", "metallurgy_confidence", "0/2")
    sr = ctx.num("strip_ratio", required=False)
    if sr is not None and sr >= t["strip_ratio_high"]:
        add("High strip ratio", dc.MEDIUM, f"strip ratio {sr:g} ≥ {t['strip_ratio_high']:g}", "strip_ratio", f"{sr:g}")
    grade = ctx.num("grade", required=False)
    low = t["grade_low_by_unit"].get(ctx.price_unit)
    if grade is not None and low:
        if grade < low:
            add("Low grade", dc.MEDIUM, f"halt {grade:g} under {low:g}", "grade", f"{grade:g}")
    width = ctx.num("intercept_width_m", required=False)
    if width is not None and width < 2.0:
        add("Narrow veins", dc.MEDIUM, f"typisk bredd {width:g} m", "intercept_width_m", f"{width:g} m")
    roy = ctx.num("royalty_burden_pct", required=False)
    if roy is not None and roy >= t["royalty_burden_high_pct"]:
        add("High royalties", dc.MEDIUM, f"royaltybörda {roy:g} % ≥ {t['royalty_burden_high_pct']:g} %", "royalty_burden_pct", f"{roy:g} %")
    stream = ctx.num("streaming_burden_pct", required=False)
    if stream is not None and stream >= t["streaming_burden_high_pct"]:
        add("High streaming burden", dc.MEDIUM, f"stream {stream:g} % av produktionen", "streaming_burden_pct", f"{stream:g} %")

    # Path to production / assumptions / price sensitivity
    yr = ctx.num("first_cashflow_year", required=False)
    if pre and yr is not None and yr - today.year > t["path_to_production_long_years"]:
        add("Long path to production", dc.MEDIUM, f"{yr - today.year:g} år till kassaflöde", "first_cashflow_year", f"{int(yr)}")
    npv_px = ctx.num("npv_price_assumption", required=False)
    if npv_px is not None and price and npv_px > price * (1 + t["npv_price_above_spot_pct"] / 100):
        add("Unrealistic economic assumptions", dc.HIGH, f"NPV räknat på {npv_px:g} mot spot {price:g} (> +{t['npv_price_above_spot_pct']:g} %)",
            "npv_price_assumption", f"{npv_px:g}")
    npv = ctx.num("npv_musd", required=False)
    npv_s = ctx.num("npv_stress_price_musd", required=False)
    if npv and npv_s is not None:
        drop = 1 - npv_s / npv
        if drop >= t["price_sensitivity_high"]:
            add("Commodity price sensitivity", dc.HIGH, f"NPV faller {drop * 100:.0f} % vid −20 % pris", "npv_stress_price_musd", f"{npv_s:g} / {npv:g}")
    elif cases.get("bear") and cases.get("base") and cases["bear"].upside_multiple is not None and cases["base"].upside_multiple:
        if cases["bear"].upside_multiple <= 0 and cases["base"].upside_multiple > 0:
            add("Commodity price sensitivity", dc.HIGH, "Bear-scenariot ger negativt värde", "", "bear ≤ 0")
    if ctx.truth("record_price_dependent", required=False):
        add("Unrealistic economic assumptions", dc.CRITICAL, "projektet fungerar bara vid rekordpris", "record_price_dependent", "ja")

    flags.sort(key=lambda f: -dc.SEVERITY_RANK[f.severity])
    return flags


def score(flags: list) -> Score:
    s = Score("red_flags", "Red Flags", 100.0)
    pen = 0.0
    for f in flags:
        pen += dc.RED_FLAG_PENALTY[f.severity]
        s.negative.append(f"− {f.severity} {f.flag}: {f.reason}")
    s.value = round(max(0.0, 100.0 - pen), 1)
    if not flags:
        s.positive.append("+ inga red flags i det som är angivet")
    s.components["antal"] = (float(len(flags)), 0)
    return s
