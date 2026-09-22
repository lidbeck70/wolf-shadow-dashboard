"""
confidence/scoring/confidence.py — Confidence Score 0–100: hur säker är
Case Score-bedömningen? (SPEC)

Sju delar: Data Quality 20, Resource Certainty 20, Project Maturity 15,
Economic Certainty 15, Financing Certainty 10, Timeline Certainty 10,
Management Track Record 10. Summan får sedan kill-caps (config.KILL_CAPS):
ingen oberoende resurs → max 50, ingen finansieringsplan → 60, bara
rekordpris → 65, en osäker parameter → 70, kapitalförstöring → 75.

Confidence ändrar aldrig Case Score. Saknad data sänker Confidence (0 p
för delen + DATA_MISSING), aldrig ett gissat tal.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from confidence import config as cfg
from confidence.data.models import CompanyInput, ConfidenceScore, PillarScore
from confidence.data.provenance import freshness_points, source_points
from confidence.data.validation import usable
from confidence.scoring._steps import finish, note, read, read_bool, src, step_ge, table_lt


def band_for(total: float) -> str:
    for floor, label in cfg.CONFIDENCE_BANDS:
        if total >= floor:
            return label
    return cfg.CONFIDENCE_BANDS[-1][1]


def _part(key: str) -> PillarScore:
    label = next(l for k, l, _m in cfg.CONFIDENCE_PARTS if k == key)
    return PillarScore(key, label, 0.0, cfg.CONFIDENCE_MAX[key])


# ── Data Quality 20 ──────────────────────────────────────────────────────────
def data_quality(company: CompanyInput, today: Optional[date] = None) -> PillarScore:
    p = _part("data_quality")
    sub = cfg.DATA_QUALITY_SUB
    points = [company.get(k) for k in company.fields if usable(company, k)]
    if not points:
        note(p, "Källkvalitet", 0, sub["source_quality"], "inga giltiga fält (DATA_MISSING)")
        note(p, "Färskhet", 0, sub["freshness"], "inga giltiga fält (DATA_MISSING)")
    else:
        n = len(points)
        sq = sum(source_points(x) for x in points) / n
        fr = sum(freshness_points(x, today) for x in points) / n
        kinds: dict = {}
        for x in points:
            kinds[x.kind] = kinds.get(x.kind, 0) + 1
        undated = sum(1 for x in points if not (x.data_date or x.pub_date))
        unsourced = sum(1 for x in points if source_points(x) == 0)
        note(p, "Källkvalitet", round(sq, 1), sub["source_quality"],
             f"snitt över {n} fält; {unsourced} utan källtyp; typer "
             + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
        note(p, "Färskhet", round(fr, 1), sub["freshness"], f"snitt över {n} fält; {undated} utan datum")
    for key, name in (("independent_verification", "Oberoende verifiering"),
                      ("cross_source_consistency", "Källkonsistens")):
        v = read(company, key, p)
        note(p, name, 0 if v is None else min(v, float(sub[key])), sub[key],
             "DATA_MISSING (bedömning saknas)" if v is None else src(company, key))
    return finish(p)


# ── Resource Certainty 20 ────────────────────────────────────────────────────
def resource_certainty(company: CompanyInput) -> PillarScore:
    p = _part("resource_certainty")
    sub = cfg.RESOURCE_CERTAINTY_SUB
    cat = str(company.get("resource_category").value) if usable(company, "resource_category") else None
    if cat is None:
        p.missing.append("resource_category")
        note(p, "Resurskategori", 0, sub["resource_category"], "DATA_MISSING")
    else:
        w = cfg.RESOURCE_CATEGORY_WEIGHT[cat]
        note(p, "Resurskategori", w * sub["resource_category"], sub["resource_category"],
             f"{cat} (vikt {w:g}) · {src(company, 'resource_category')}")
    indep = read_bool(company, "independent_resource_estimate", p)
    note(p, "Oberoende uppskattning", sub["independent_resource_estimate"] if indep else 0,
         sub["independent_resource_estimate"],
         "DATA_MISSING" if indep is None else src(company, "independent_resource_estimate"))
    share = read(company, "resource_verified_share_pct", p)
    note(p, "Verifierad andel", 0 if share is None else min(share, 100) / 100 * sub["resource_verified_share_pct"],
         sub["resource_verified_share_pct"],
         "DATA_MISSING" if share is None else f"{share:g} % · {src(company, 'resource_verified_share_pct')}")
    for key in ("drilling_density", "resource_conversion_history", "grade_consistency", "metallurgy_confidence"):
        spec = cfg.FIELD_BY_KEY[key]
        v = read(company, key, p)
        pts = 0 if v is None else min(v, spec.max) / spec.max * sub[key]
        note(p, spec.label, pts, sub[key], "DATA_MISSING" if v is None else src(company, key))
    return finish(p)


# ── Project Maturity 15 ──────────────────────────────────────────────────────
def project_maturity(company: CompanyInput) -> PillarScore:
    p = _part("project_maturity")
    lo, hi = cfg.MATURITY_BASE.get(company.maturity, (0, 3))
    label = cfg.MATURITY_LABEL.get(company.maturity, company.maturity)
    if hi == lo:
        note(p, "Mognad", hi, p.max, f"{label} · {lo}–{hi} p")
        return finish(p)
    adjusters = [k for k in cfg.MATURITY_ADJUSTERS if usable(company, k) or
                 not cfg.FIELD_BY_KEY[k].stages or company.stage in cfg.FIELD_BY_KEY[k].stages]
    done = []
    for k in adjusters:
        b = read_bool(company, k, p)
        if b:
            done.append(cfg.FIELD_BY_KEY[k].label)
    step = (hi - lo) / len(adjusters) if adjusters else 0
    pts = lo + step * len(done)
    note(p, "Mognad", round(pts, 2), p.max,
         f"{label} · grund {lo} p + {len(done)}/{len(adjusters)} milstolpar × {step:.2f} p"
         + (f" ({', '.join(done)})" if done else ""))
    return finish(p)


# ── Economic Certainty 15 ────────────────────────────────────────────────────
def economic_certainty(company: CompanyInput) -> PillarScore:
    p = _part("economic_certainty")
    sub, mcap = cfg.ECON_CERTAINTY_SUB, cfg.ECON_CERTAINTY_MODELLED_CAP
    applicable = float(p.max)
    if company.stage in cfg.PRE_REVENUE:
        npv = read(company, "npv_musd", p)
        capex = read(company, "capex_musd", p)
        price = read(company, "commodity_price", p)
        be = read(company, "breakeven_price", p)
        # prisstress
        s = read(company, "npv_stress_price_musd", p)
        if s is not None and npv:
            note(p, "Prisstress", _survival(s, npv) * sub["price_stress"], sub["price_stress"],
                 f"NPV {s:g}/{npv:g} MUSD vid −20 % · {src(company, 'npv_stress_price_musd')}")
        elif price is not None and be is not None and be > 0:
            surv = max(0.0, min(1.0, (price * (1 + cfg.STRESS_PRICE_PCT / 100) - be) / (price - be))) \
                if price > be else 0.0
            note(p, "Prisstress", min(surv * sub["price_stress"], mcap["price_stress"]), sub["price_stress"],
                 f"MODELLED: marginal över break-even {surv * 100:.0f} % kvar vid −20 % (tak {mcap['price_stress']} p)")
        else:
            note(p, "Prisstress", 0, sub["price_stress"], "DATA_MISSING (FS-känslighet eller pris + break-even)")
        # capexstress
        s = read(company, "npv_stress_capex_musd", p)
        if s is not None and npv:
            note(p, "Capexstress", _survival(s, npv) * sub["capex_stress"], sub["capex_stress"],
                 f"NPV {s:g}/{npv:g} MUSD vid +20 % · {src(company, 'npv_stress_capex_musd')}")
        elif npv is not None and capex is not None and npv > 0:
            surv = max(0.0, min(1.0, (npv - cfg.STRESS_CAPEX_PCT / 100 * capex) / npv))
            note(p, "Capexstress", min(surv * sub["capex_stress"], mcap["capex_stress"]), sub["capex_stress"],
                 f"MODELLED: NPV − 0,2 × CapEx = {surv * 100:.0f} % av NPV (tak {mcap['capex_stress']} p)")
        else:
            note(p, "Capexstress", 0, sub["capex_stress"], "DATA_MISSING")
        # studiekvalitet
        sq = cfg.STUDY_QUALITY_BY_MATURITY.get(company.maturity, 0)
        note(p, "Studiekvalitet", sq, sub["study_quality"], cfg.MATURITY_LABEL.get(company.maturity, ""))
        # NPV-pris över spot
        npv_px = read(company, "npv_price_assumption", p) if company.has("npv_price_assumption") else None
        if npv_px is not None and price is not None and price > 0 and \
                npv_px > price * (1 + cfg.NPV_PRICE_ABOVE_SPOT_PCT / 100):
            p.components["NPV-pris över spot"] = -cfg.NPV_PRICE_ABOVE_SPOT_PENALTY
            p.notes.append(f"NPV-pris över spot −{cfg.NPV_PRICE_ABOVE_SPOT_PENALTY} p — NPV räknat på "
                           f"{npv_px:g} mot spot {price:g} (> +{cfg.NPV_PRICE_ABOVE_SPOT_PCT:g} %)")
    else:
        applicable -= sub["capex_stress"]
        p.notes.append("producent: capexstress gäller inte (sustaining capex ingår i AISC)")
        price = read(company, "commodity_price", p)
        if company.stage == "royalty":
            applicable -= sub["price_stress"]
            p.notes.append("royalty: prisstress ur AISC gäller inte")
        else:
            aisc = read(company, "aisc", p)
            if price is not None and aisc is not None and price > aisc:
                surv = max(0.0, (price * (1 + cfg.STRESS_PRICE_PCT / 100) - aisc) / (price - aisc))
                note(p, "Prisstress", min(1.0, surv) * sub["price_stress"], sub["price_stress"],
                     f"{surv * 100:.0f} % av AISC-marginalen kvar vid −20 %")
            else:
                note(p, "Prisstress", 0, sub["price_stress"],
                     "DATA_MISSING" if price is None or aisc is None else "AISC ≥ pris")
        fin = company.get("ebitda_margin_pct")
        sq = sub["study_quality"] if fin is not None and source_points(fin) >= 4 else \
            (sub["study_quality"] / 2 if fin is not None and not fin.missing else 0)
        note(p, "Studiekvalitet", sq, sub["study_quality"],
             "reviderade siffror" if sq == sub["study_quality"] else
             ("ej primär källa för resultatet" if sq else "DATA_MISSING"))
    if company.truth("record_price_dependent"):
        total = sum(p.components.values())
        p.components["Rekordprisberoende"] = -round(total * (1 - cfg.RECORD_PRICE_FACTOR), 2)
        p.notes.append(f"Rekordprisberoende × {cfg.RECORD_PRICE_FACTOR:g} — projektet fungerar bara vid "
                       f"nära rekordpris (SPEC: kraftigt reducerad)")
    return finish(p, applicable)


def _survival(stressed: float, base: float) -> float:
    return max(0.0, min(1.0, stressed / base)) if base > 0 else 0.0


# ── Financing Certainty 10 ───────────────────────────────────────────────────
def financing_certainty(company: CompanyInput) -> PillarScore:
    p = _part("financing_certainty")
    if company.stage in cfg.PRE_REVENUE:
        cash = read(company, "cash_musd", p)
        burn = read(company, "quarterly_burn_musd", p)
        rmax = cfg.FIN_RUNWAY_STEPS[0][1]
        smax = cfg.FIN_COMMITTED_SHARE_STEPS[0][1]
        applicable = float(p.max)
        if company.stage == "explorer":
            applicable -= smax
            p.notes.append("explorer: åtagen andel av CapEx gäller inte")
        else:
            capex = read(company, "capex_musd", p)
            committed = company.num("committed_financing_musd") if company.has("committed_financing_musd") else None
            if capex is not None and capex > 0 and committed is not None:
                share = committed / capex
                note(p, "Åtagen andel av CapEx", step_ge(share, cfg.FIN_COMMITTED_SHARE_STEPS), smax,
                     f"{share * 100:.0f} % · {src(company, 'committed_financing_musd')}")
            elif capex is not None and capex > 0:
                note(p, "Åtagen andel av CapEx", 0, smax, "ingen åtagen finansiering registrerad")
            else:
                note(p, "Åtagen andel av CapEx", 0, smax, "DATA_MISSING")
        if cash is not None and burn is not None and burn > 0:
            q = cash / burn
            note(p, "Runway", step_ge(q, cfg.FIN_RUNWAY_STEPS), rmax, f"{q:.1f} kvartal")
        else:
            note(p, "Runway", 0, rmax, "DATA_MISSING")
        return finish(p, applicable)
    nd = read(company, "net_debt_ebitda", p)
    note(p, "Nettoskuld/EBITDA", table_lt(nd, cfg.FIN_ND_EBITDA_STEPS, cfg.FIN_ND_EBITDA_BEYOND), p.max,
         "DATA_MISSING" if nd is None else f"{nd:.2f}× · {src(company, 'net_debt_ebitda')}")
    return finish(p)


# ── Timeline Certainty 10 ────────────────────────────────────────────────────
def timeline_certainty(company: CompanyInput) -> PillarScore:
    p = _part("timeline_certainty")
    delays = read(company, "historical_delays", p)
    dmax = cfg.FIELD_BY_KEY["historical_delays"].max
    if company.stage in cfg.PRE_REVENUE:
        doc = read_bool(company, "timeline_documented", p)
        note(p, "Dokumenterad plan", cfg.TIMELINE_DOCUMENTED_POINTS if doc else 0, cfg.TIMELINE_DOCUMENTED_POINTS,
             "DATA_MISSING" if doc is None else src(company, "timeline_documented"))
        note(p, "Förseningshistorik", 0 if delays is None else min(delays, dmax) * cfg.TIMELINE_DELAYS_FACTOR,
             dmax * cfg.TIMELINE_DELAYS_FACTOR, "DATA_MISSING" if delays is None else src(company, "historical_delays"))
        m = cfg.TIMELINE_MATURITY_POINTS.get(company.maturity, 0)
        note(p, "Mognad", m, max(cfg.TIMELINE_MATURITY_POINTS.values()),
             cfg.MATURITY_LABEL.get(company.maturity, company.maturity))
        return finish(p)
    note(p, "I produktion", cfg.TIMELINE_PRODUCER_BASE, cfg.TIMELINE_PRODUCER_BASE, "kassaflöde nu")
    note(p, "Förseningshistorik", 0 if delays is None else min(delays, dmax), dmax,
         "DATA_MISSING" if delays is None else src(company, "historical_delays"))
    return finish(p)


# ── Management Track Record 10 ───────────────────────────────────────────────
def management_track_record(company: CompanyInput) -> PillarScore:
    p = _part("management_track_record")
    for key, mx in cfg.MGMT_TRACK_SUB.items():
        spec = cfg.FIELD_BY_KEY[key]
        v = read(company, key, p)
        note(p, spec.label, 0 if v is None else min(v, spec.max) / spec.max * mx, mx,
             "DATA_MISSING" if v is None else src(company, key))
    return finish(p)


# ── summa + kill-caps ────────────────────────────────────────────────────────
def kill_caps(company: CompanyInput) -> list:
    """[(nyckel, tak, text)] för de kill switches som slår till. Ej bedömda
    switchar flaggas som TODO av confidence_score, de slår inte till."""
    out = []
    for key, limit, text in cfg.KILL_CAPS:
        if key == "no_independent_resource":
            hit = company.truth("independent_resource_estimate") is False
        else:
            hit = company.truth(key) is True
        if hit:
            out.append((key, limit, text))
    return out


def confidence_score(company: CompanyInput, today: Optional[date] = None) -> ConfidenceScore:
    parts = [
        data_quality(company, today),
        resource_certainty(company),
        project_maturity(company),
        economic_certainty(company),
        financing_certainty(company),
        timeline_certainty(company),
        management_track_record(company),
    ]
    assert [p.key for p in parts] == [k for k, _l, _m in cfg.CONFIDENCE_PARTS]
    raw = round(sum(p.points for p in parts), 1)
    caps = kill_caps(company)
    total = raw
    for _key, limit, _text in caps:
        total = min(total, float(limit))
    missing: list = []
    for p in parts:
        for k in p.missing:
            if k not in missing:
                missing.append(k)
    flags = [f"KILL: {text} → max {limit}" for _k, limit, text in caps]
    for key, _limit, text in cfg.KILL_CAPS:
        field = "independent_resource_estimate" if key == "no_independent_resource" else key
        if company.truth(field) is None and (not cfg.FIELD_BY_KEY[field].stages or
                                             company.stage in cfg.FIELD_BY_KEY[field].stages):
            flags.append(f"TODO: bedöm '{text}' ({field}) — ej bedömd kill switch")
    if missing:
        flags.append(f"{len(missing)} fält saknas — Confidence sänkt, inte gissad")
    return ConfidenceScore(total=round(total, 1), band=band_for(total), raw_total=raw, parts=parts,
                           caps_applied=caps, missing=missing, flags=flags)


__all__ = ["confidence_score", "band_for", "kill_caps"]
