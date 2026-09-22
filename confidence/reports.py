"""
confidence/reports.py — hela analysen för ett bolag, Thesis Killer,
rekommendation, Investment Card och rapport i klartext. (SPEC)

analyze() kör alla motorer i ordning och samlar resultatet i Analysis.
thesis_killer() ger minst fem risker med nivå LOW/MEDIUM/HIGH/CRITICAL,
alla härledda ur poängen — inga fria bedömningar. recommendation() följer
CLAUDE.md:s statusar (BUY CANDIDATE / WATCH / PASS / REJECT) och kräver
både Case Score och Confidence. Rena funktioner, ingen Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from confidence import commodities as com
from confidence import config as cfg
from confidence.data.models import CaseScore, CompanyInput, ConfidenceScore, to_jsonable
from confidence.data.validation import validate
from confidence.regional import RegionalScarcity, regional_scarcity
from confidence.scenarios import ScenarioSet, TimeToMoney, scenario_set, time_to_money
from confidence.scoring import case_score, confidence_score
from confidence.why_now import Signals, WhyNow, why_now

LOW, MEDIUM, HIGH, CRITICAL = "LOW", "MEDIUM", "HIGH", "CRITICAL"
_SEV = {CRITICAL: 3, HIGH: 2, MEDIUM: 1, LOW: 0}

BUY, WATCH, PASS, REJECT = "BUY CANDIDATE", "WATCH", "PASS", "REJECT"
BUY_CASE_MIN, BUY_CONF_MIN = 80, 70          # VAL: PICK + GOOD CONFIDENCE
WATCH_CASE_MIN, WATCH_CONF_MIN = 60, 50      # VAL: WATCHLIST + SPECULATIVE
MIN_RISKS = 5                                 # SPEC


@dataclass
class Risk:
    name: str
    level: str
    why: str


@dataclass
class Analysis:
    company: CompanyInput
    commodity: Optional[com.Commodity]
    case: CaseScore
    confidence: ConfidenceScore
    scenarios: ScenarioSet
    ttm: TimeToMoney
    why_now: WhyNow
    regional: RegionalScarcity
    risks: list
    recommendation: str
    recommendation_why: str
    issues: list = field(default_factory=list)
    today: str = ""


def analyze(company: CompanyInput, commodity_overrides: Optional[dict] = None,
            signals: Optional[Signals] = None, today: Optional[date] = None) -> Analysis:
    today = today or date.today()
    commodity = com.get(company.commodity, commodity_overrides)
    case = case_score(company, commodity, today)
    conf = confidence_score(company, today)
    scen = scenario_set(company)
    ttm = time_to_money(company, today)
    sig = signals or Signals()
    if sig.time_to_money_years is None:
        sig.time_to_money_years = ttm.years
    wn = why_now(commodity, sig)
    reg = regional_scarcity(company, commodity)
    risks = thesis_killer(company, case, conf, scen, ttm, wn, reg)
    rec, why = recommendation(case, conf, risks)
    return Analysis(company, commodity, case, conf, scen, ttm, wn, reg, risks, rec, why,
                    validate(company), today.isoformat())


# ── Thesis Killer ────────────────────────────────────────────────────────────
def thesis_killer(company: CompanyInput, case: CaseScore, conf: ConfidenceScore, scen: ScenarioSet,
                  ttm: TimeToMoney, wn: WhyNow, reg: RegionalScarcity) -> list:
    risks: list = []
    pre = company.stage in cfg.PRE_REVENUE
    econ = case.pillar("economics")

    # 1 råvarupris
    if econ and econ.caps:
        risks.append(Risk("Råvarupris", HIGH, "stressen pris −20 % / capex +20 % faller: " +
                          "; ".join(n for n in econ.notes if n.startswith("TAK"))))
    elif econ and any("ej möjlig" in n for n in econ.notes):
        risks.append(Risk("Råvarupris", HIGH, "stressen kunde inte göras (DATA_MISSING) — okänd känslighet"))
    else:
        risks.append(Risk("Råvarupris", MEDIUM, "klarar −20 % enligt stressen, men caset vilar på priset"))
    if company.truth("record_price_dependent"):
        risks.append(Risk("Rekordprisberoende", CRITICAL, "projektet fungerar bara vid nära rekordpris"))

    # 2 finansiering
    bal = case.pillar("balance_sheet")
    if company.truth("no_financing_plan"):
        risks.append(Risk("Finansiering", CRITICAL, "ingen realistisk finansieringsplan (kill switch)"))
    elif pre:
        gap = bal.components.get("Finansieringsgap") if bal else None
        run = bal.components.get("Runway") if bal else None
        if gap is None and company.stage == "developer":
            risks.append(Risk("Finansiering", HIGH, "finansieringsgapet kan inte beräknas (DATA_MISSING)"))
        elif (gap is not None and gap <= 1) or (run is not None and run == 0):
            risks.append(Risk("Finansiering", HIGH, "stort gap mot CapEx och/eller runway under två kvartal"))
        elif (gap is not None and gap <= 2) or (run is not None and run <= 1):
            risks.append(Risk("Finansiering", MEDIUM, "gap eller runway kräver kapitalanskaffning inom ett år"))
        else:
            risks.append(Risk("Finansiering", LOW, "gap litet och runway god"))
    else:
        nd = company.num("net_debt_ebitda")
        if nd is None:
            risks.append(Risk("Skuldsättning", MEDIUM, "nettoskuld/EBITDA saknas (DATA_MISSING)"))
        elif nd >= 3:
            risks.append(Risk("Skuldsättning", HIGH, f"nettoskuld/EBITDA {nd:.1f}× — tål inte ett prisfall"))
        elif nd >= 2:
            risks.append(Risk("Skuldsättning", MEDIUM, f"nettoskuld/EBITDA {nd:.1f}×"))
        else:
            risks.append(Risk("Skuldsättning", LOW, f"nettoskuld/EBITDA {nd:.1f}×"))

    # 3 utspädning
    ds = company.num("dilution_score")
    if company.truth("capital_destruction_history"):
        risks.append(Risk("Utspädning", CRITICAL, "dokumenterad kapitalförstöring (kill switch)"))
    elif ds is not None and ds >= 8:
        risks.append(Risk("Utspädning", HIGH, f"DS {ds:g}/10 — EXTREM enligt kontrollerna"))
    elif ds is not None and ds >= 6:
        risks.append(Risk("Utspädning", MEDIUM, f"DS {ds:g}/10 — köp låst utan finansieringskatalysator"))
    elif ds is not None:
        risks.append(Risk("Utspädning", LOW, f"DS {ds:g}/10"))
    elif pre:
        risks.append(Risk("Utspädning", MEDIUM, "DS ej bedömd — pre-revenue-bolag späder ut"))

    # 4 jurisdiktion
    j = reg.jurisdiction_score
    if j is None or "jurisdiction" in reg.missing:
        risks.append(Risk("Jurisdiktion", MEDIUM, "land/region okänd i repots tabell"))
    elif j < 50:
        risks.append(Risk("Jurisdiktion", HIGH, f"jurisdiktion {j:g}/100 — hög politisk risk"))
    elif j < cfg.SAFE_JURISDICTION_MIN:
        risks.append(Risk("Jurisdiktion", MEDIUM, f"jurisdiktion {j:g}/100"))
    else:
        risks.append(Risk("Jurisdiktion", LOW, f"jurisdiktion {j:g}/100"))

    # 5 tidsplan / genomförande
    if pre:
        lvl = {"låg": HIGH, "medel": MEDIUM, "hög": LOW}[ttm.confidence]
        why = f"{ttm.years:g} år till kassaflöde ({ttm.basis})" + (": " + "; ".join(ttm.flags) if ttm.flags else "")
        risks.append(Risk("Tidsplan & genomförande", lvl, why))

    # 6 resurs
    cat = str(company.get("resource_category").value) if company.has("resource_category") else None
    indep = company.truth("independent_resource_estimate")
    if indep is False:
        risks.append(Risk("Resurs", HIGH if company.stage == "explorer" else CRITICAL,
                          "ingen oberoende resursuppskattning (kill switch → Confidence max 50)"))
    elif cat in ("exploration_target", "inferred"):
        risks.append(Risk("Resurs", HIGH, f"högsta kategori {cat} — resursen kan krympa vid konvertering"))
    elif cat == "indicated":
        risks.append(Risk("Resurs", MEDIUM, "indicated — inte reserv ännu"))
    elif cat is None:
        risks.append(Risk("Resurs", MEDIUM, "resurskategori saknas (DATA_MISSING)"))
    else:
        risks.append(Risk("Resurs", LOW, f"{cat}"))

    # 7 datakvalitet
    dq = conf.part("data_quality")
    if dq and dq.points < 8:
        risks.append(Risk("Datakvalitet", HIGH, f"Data Quality {dq.points:g}/20 — svaga eller odaterade källor"))
    elif dq and dq.points < 13:
        risks.append(Risk("Datakvalitet", MEDIUM, f"Data Quality {dq.points:g}/20"))

    # 8 värdering
    val = case.pillar("valuation")
    if val and val.points == 0 and not any("DATA_MISSING" in n for n in val.notes):
        risks.append(Risk("Värdering", HIGH, "ingen värderingspoäng — priset räknar redan in caset"))
    elif val and "p_nav" in val.missing and pre:
        risks.append(Risk("Värdering", MEDIUM, "P/NAV saknas (DATA_MISSING)"))

    # 9 enskild parameter / cykel
    if company.truth("single_fragile_parameter"):
        risks.append(Risk("Enskild parameter", CRITICAL, "hela caset vilar på en extremt osäker parameter"))
    cyc = wn.components.get("Cykelläge")
    if cyc == 0 and "cycle_label" not in wn.missing:
        risks.append(Risk("Cykelläge", HIGH, "råvaran handlas i toppen av sitt 10-årsintervall"))
    elif cyc is not None and cyc <= cfg.WHY_NOW_CYCLE["SEN"]:
        risks.append(Risk("Cykelläge", MEDIUM, "sent i cykeln"))

    # 10 asymmetri
    a = scen.asymmetry
    if a is not None and a.band == "NEGATIV":
        risks.append(Risk("Asymmetri", HIGH, f"Bear {a.downside_pct:+.0f} % mot Bull {a.upside_pct:+.0f} % — mer att förlora än vinna"))
    elif a is None and scen.scenarios:
        risks.append(Risk("Asymmetri", MEDIUM, "scenarierna är ofullständiga (DATA_MISSING)"))

    if len(risks) < MIN_RISKS:
        risks.append(Risk("Kända okända", MEDIUM, f"{len(case.missing)} fält saknas — riskbilden är ofullständig"))
    risks.sort(key=lambda r: -_SEV[r.level])
    return risks


# ── Rekommendation ───────────────────────────────────────────────────────────
def recommendation(case: CaseScore, conf: ConfidenceScore, risks: list) -> tuple:
    critical = [r for r in risks if r.level == CRITICAL]
    if critical:
        return REJECT, "kill switch: " + "; ".join(r.name for r in critical)
    if case.total < 50:
        return PASS, f"Case Score {case.total:g} < 50 ({case.rating})"
    if case.total >= BUY_CASE_MIN and conf.total >= BUY_CONF_MIN:
        return BUY, f"Case {case.total:g} ({case.rating}) med Confidence {conf.total:g} ({conf.band})"
    if case.total >= BUY_CASE_MIN:
        return WATCH, f"Case {case.total:g} men Confidence bara {conf.total:g} ({conf.band}) — verifiera innan köp"
    if case.total >= WATCH_CASE_MIN and conf.total >= WATCH_CONF_MIN:
        return WATCH, f"Case {case.total:g} ({case.rating}), Confidence {conf.total:g}"
    return PASS, f"Case {case.total:g} ({case.rating}) med Confidence {conf.total:g} ({conf.band}) räcker inte"


# ── Investment Card ──────────────────────────────────────────────────────────
def investment_card(a: Analysis) -> dict:
    c, case, conf = a.company, a.case, a.confidence
    passed = [p.label for p in case.pillars if p.max and p.points / p.max >= 0.7]
    failed = [p.label for p in case.pillars if p.max and p.points / p.max < 0.4]
    base = a.scenarios.scenario("base")
    bull = a.scenarios.scenario("bull")
    bear = a.scenarios.scenario("bear")
    asym = a.scenarios.asymmetry
    return {
        "ticker": c.ticker, "name": c.name, "strategy": "Confidence score",
        "stage": c.stage, "maturity": c.maturity, "commodity": a.commodity.label if a.commodity else c.commodity,
        "sleeve": cfg.STAGE_SLEEVE.get(c.stage, ""),
        "regime": f"Why Now {a.why_now.score:g} ({a.why_now.band})",
        "sector_score": case.pillar("strategic_commodity").points,
        "setup_score": case.total, "rating": case.rating,
        "confidence": conf.total, "confidence_band": conf.band,
        "regional": f"{a.regional.score:g} ({a.regional.band})",
        "passed_rules": passed, "failed_rules": failed,
        "ai_comment": "",                                     # AI-lagret fyller i (v2)
        "entry_zone": "ej i v1 — köpgrind och entry sätts i strategiflikarna",
        "stop": None, "target": None,
        "upside_base_pct": base.upside_pct if base else None,
        "upside_bull_pct": bull.upside_pct if bull else None,
        "downside_bear_pct": bear.upside_pct if bear else None,
        "asymmetry": f"{asym.ratio:.1f}× ({asym.band})" if asym and asym.ratio not in (None, float("inf"))
        else (asym.band if asym else "DATA_MISSING"),
        "time_to_money": f"{a.ttm.years:g} år ({a.ttm.confidence})",
        "top_risks": [f"{r.level}: {r.name}" for r in a.risks[:3]],
        "recommendation": a.recommendation, "why": a.recommendation_why,
        "missing_fields": len(case.missing), "date": a.today,
    }


# ── Rapport ──────────────────────────────────────────────────────────────────
def report_markdown(a: Analysis) -> str:
    c = a.company
    L = [f"# {c.ticker} — {c.name}", f"*{c.stage} · {cfg.MATURITY_LABEL.get(c.maturity, c.maturity)} · "
         f"{a.commodity.label if a.commodity else c.commodity} · {c.jurisdiction or c.country} · {a.today}*", "",
         f"**Rekommendation: {a.recommendation}** — {a.recommendation_why}", "",
         f"## Case Score {a.case.total:g}/100 — {a.case.rating}"]
    for p in a.case.pillars:
        L.append(f"### {p.label} {p.points:g}/{p.max:g}")
        L += [f"- {n}" for n in p.notes]
    if a.case.discovery_option is not None:
        L.append(f"\nDiscovery Option (separat): {a.case.discovery_option:g}/{cfg.DISCOVERY_OPTION_MAX}")
    L += ["", f"## Confidence Score {a.confidence.total:g}/100 — {a.confidence.band} "
             f"(råpoäng {a.confidence.raw_total:g})"]
    for p in a.confidence.parts:
        L.append(f"### {p.label} {p.points:g}/{p.max:g}")
        L += [f"- {n}" for n in p.notes]
    L += [f"- {f}" for f in a.confidence.flags]
    L += ["", f"## Why Now {a.why_now.score:g}/100 — {a.why_now.band} (täckning {a.why_now.coverage * 100:.0f} %)"]
    L += [f"- {n}" for n in a.why_now.notes] + [f"- ⚠ {f}" for f in a.why_now.flags]
    L += ["", f"## Regional knapphet {a.regional.score:g}/100 — {a.regional.band}"]
    L += [f"- {n}" for n in a.regional.notes] + [f"- ⚠ {f}" for f in a.regional.flags]
    L += ["", f"## Time-to-money: {a.ttm.years:g} år ({a.ttm.confidence}, {a.ttm.basis})"]
    L += [f"- {n}" for n in a.ttm.notes] + [f"- ⚠ {f}" for f in a.ttm.flags]
    L += ["", "## Scenarier"]
    for s in a.scenarios.scenarios:
        up = f"{s.upside_pct:+.0f} %" if s.upside_pct is not None else "DATA_MISSING"
        L.append(f"### {s.label} (pris {s.price_change_pct:+g} %, capex {s.capex_change_pct:+g} %): {up}")
        L += [f"- {st}" for st in s.steps]
    if a.scenarios.asymmetry:
        x = a.scenarios.asymmetry
        ratio = "∞" if x.ratio == float("inf") else (f"{x.ratio:.1f}×" if x.ratio is not None else "—")
        L.append(f"\n**Asymmetri**: Bull {x.upside_pct:+.0f} % / Bear {x.downside_pct:+.0f} % = {ratio} ({x.band}); "
                 f"sannolikhetsvägd {x.expected_pct:+.0f} % ({', '.join(f'{k} {v:g} %' for k, v in x.probs.items())})")
    for p in a.scenarios.paths:
        L.append(f"\n**{p.target:g}× — vad måste hända**")
        L += [f"- {st}" for st in p.steps]
    L += ["", "**Antaganden**"] + [f"- {x.text()}" for x in a.scenarios.assumptions]
    L += ["", "## Thesis Killer"] + [f"- **{r.level}** {r.name}: {r.why}" for r in a.risks]
    if a.case.missing:
        L += ["", f"## DATA_MISSING ({len(a.case.missing)})", ", ".join(a.case.missing)]
    errs = [i for i in a.issues if i.level == "error"]
    if errs:
        L += ["", "## Fel i indata"] + [f"- {i.field}: {i.message}" for i in errs]
    return "\n".join(L)


def analysis_json(a: Analysis) -> dict:
    d = to_jsonable(a)
    d["card"] = investment_card(a)
    return d
