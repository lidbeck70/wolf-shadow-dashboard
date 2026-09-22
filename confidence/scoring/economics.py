"""
confidence/scoring/economics.py — Economics 15 p, stage-routad (SPEC).

Producenter/royalty: AISC-marginal 4, EBITDA-marginal 3, FCF-yield 3,
ROIC 3, break-even-täckning 2. Developers/explorers: NPV/CapEx 4, IRR 4,
payback 2, break-even-täckning 3, capex-intensitet 2.

Stress är obligatoriskt (SPEC: pris −20 %, capex +20 %). Ordning:
  1. FS-känslighetstabellens tal om de finns (npv_stress_*, irr_stress_*).
  2. Annars en konservativ modellerad bound, märkt MODELLED i notes:
       pris:  0,8 × pris ≤ break-even (eller AISC)  → negativ marginal
       capex: NPV − 0,2 × CapEx ≤ 0                  → NPV faller minst så mycket
  3. Går inget att stressa: flagga, inget tak (Confidence tar osäkerheten).
Ett negativt stressutfall tar pelaren till ECON_CAP_STRESS_NEGATIVE, en
svag IRR under stress till ECON_CAP_STRESS_WEAK.
"""

from __future__ import annotations

from confidence import config as cfg
from confidence.data.models import CompanyInput, PillarScore
from confidence.scoring._steps import cap, finish, note, read, src, step_ge, step_le

_PRICE_FACTOR = 1.0 + cfg.STRESS_PRICE_PCT / 100.0
_CAPEX_FACTOR = cfg.STRESS_CAPEX_PCT / 100.0


def score(company: CompanyInput) -> PillarScore:
    p = PillarScore("economics", "Economics", 0.0, cfg.PILLAR_MAX["economics"])
    if company.stage in cfg.PRE_REVENUE:
        return _developer(company, p)
    return _producer(company, p)


# ── producenter / royalty ────────────────────────────────────────────────────
def _producer(company: CompanyInput, p: PillarScore) -> PillarScore:
    sub = cfg.ECON_PRODUCER_SUB
    royalty = company.stage == "royalty"
    applicable = float(p.max)
    price = read(company, "commodity_price", p)

    if royalty:
        applicable -= sub["aisc"] + sub["breakeven"]
        p.notes.append("royalty: AISC och break-even gäller inte (ingen egen drift)")
    else:
        aisc = read(company, "aisc", p)
        if price is not None and aisc is not None and price > 0:
            margin = (price - aisc) / price * 100.0
            note(p, "AISC-marginal", step_ge(margin, cfg.AISC_MARGIN_STEPS) if margin > 0 else 0,
                 sub["aisc"], f"{margin:.0f} % · {src(company, 'aisc', 'AISC')}")
            stressed = price * _PRICE_FACTOR - aisc
            if stressed <= 0:
                cap(p, "pris −20 %", cfg.ECON_CAP_STRESS_NEGATIVE,
                    f"MODELLED: {price * _PRICE_FACTOR:g} − AISC {aisc:g} ≤ 0")
            else:
                p.notes.append(f"stress pris −20 % OK (MODELLED): marginal {stressed:g}/enhet")
        else:
            note(p, "AISC-marginal", 0, sub["aisc"], "DATA_MISSING")
            p.notes.append("stress pris −20 % ej möjlig: AISC eller pris saknas (DATA_MISSING)")

    for key, name, steps in (("ebitda_margin_pct", "EBITDA-marginal", cfg.EBITDA_MARGIN_STEPS),
                             ("fcf_yield_pct", "FCF-yield", cfg.FCF_YIELD_STEPS),
                             ("roic_pct", "ROIC", cfg.ROIC_STEPS)):
        v = read(company, key, p)
        note(p, name, step_ge(v, steps), sub[key[:-4]],       # "ebitda_margin_pct" → "ebitda_margin"
             "DATA_MISSING" if v is None else src(company, key))

    if not royalty:
        be = read(company, "breakeven_price", p)
        if price is not None and be is not None and be > 0:
            cover = price / be
            note(p, "Break-even-täckning", step_ge(cover, cfg.BREAKEVEN_COVER_STEPS_PROD),
                 sub["breakeven"], f"pris/break-even {cover:.2f}× · {src(company, 'breakeven_price')}")
        else:
            note(p, "Break-even-täckning", 0, sub["breakeven"], "DATA_MISSING")
    p.notes.append("capex +20 %: gäller inte producenter (sustaining capex ingår i AISC)")
    return finish(p, applicable)


# ── developers / explorers ───────────────────────────────────────────────────
def _developer(company: CompanyInput, p: PillarScore) -> PillarScore:
    sub = cfg.ECON_DEVELOPER_SUB
    npv = read(company, "npv_musd", p)
    capex = read(company, "capex_musd", p)
    irr = read(company, "irr_pct", p)
    payback = read(company, "payback_years", p)
    price = read(company, "commodity_price", p)
    be = read(company, "breakeven_price", p)

    if npv is not None and capex is not None and capex > 0:
        ratio = npv / capex
        note(p, "NPV/CapEx", step_ge(ratio, cfg.NPV_CAPEX_STEPS), sub["npv_capex"],
             f"{ratio:.2f}× · {src(company, 'npv_musd', 'NPV')} · {src(company, 'capex_musd', 'CapEx')}")
    else:
        note(p, "NPV/CapEx", 0, sub["npv_capex"], "DATA_MISSING")
    note(p, "IRR", step_ge(irr, cfg.IRR_STEPS), sub["irr"],
         "DATA_MISSING" if irr is None else src(company, "irr_pct"))
    note(p, "Payback", step_le(payback, cfg.PAYBACK_STEPS), sub["payback"],
         "DATA_MISSING" if payback is None else src(company, "payback_years"))
    if price is not None and be is not None and be > 0:
        cover = price / be
        note(p, "Break-even-täckning", step_ge(cover, cfg.BREAKEVEN_COVER_STEPS_DEV), sub["breakeven"],
             f"pris/break-even {cover:.2f}× · {src(company, 'breakeven_price')}")
    else:
        note(p, "Break-even-täckning", 0, sub["breakeven"], "DATA_MISSING")

    prod = read(company, "annual_production", p)
    if capex is not None and prod is not None and price is not None and prod > 0 and price > 0:
        revenue = prod * price / 1e6
        intensity = capex / revenue
        note(p, "Capex-intensitet", step_le(intensity, cfg.CAPEX_INTENSITY_STEPS), sub["capex_intensity"],
             f"CapEx/årsintäkt {intensity:.2f}× (intäkt {revenue:,.0f} MUSD; förutsätter samma enhet "
             f"för produktion och pris)")
    else:
        note(p, "Capex-intensitet", 0, sub["capex_intensity"], "DATA_MISSING")

    _stress_developer(company, p, npv, capex, price, be)
    return finish(p)


def _stress_developer(company, p, npv, capex, price, be) -> None:
    # pris −20 %
    npv_s = read(company, "npv_stress_price_musd", p)
    irr_s = read(company, "irr_stress_price_pct", p)
    if npv_s is not None:
        if npv_s <= 0:
            cap(p, "pris −20 %", cfg.ECON_CAP_STRESS_NEGATIVE, f"NPV {npv_s:g} MUSD ≤ 0 · "
                f"{src(company, 'npv_stress_price_musd')}")
        else:
            p.notes.append(f"stress pris −20 %: NPV {npv_s:g} MUSD · {src(company, 'npv_stress_price_musd')}")
    elif irr_s is not None:
        if irr_s < cfg.IRR_STRESS_MIN:
            cap(p, "pris −20 %", cfg.ECON_CAP_STRESS_WEAK,
                f"IRR {irr_s:g} % < {cfg.IRR_STRESS_MIN:g} % · {src(company, 'irr_stress_price_pct')}")
        else:
            p.notes.append(f"stress pris −20 %: IRR {irr_s:g} % · {src(company, 'irr_stress_price_pct')}")
    elif price is not None and be is not None:
        if price * _PRICE_FACTOR <= be:
            cap(p, "pris −20 %", cfg.ECON_CAP_STRESS_NEGATIVE,
                f"MODELLED: {price * _PRICE_FACTOR:g} ≤ break-even {be:g}")
        else:
            p.notes.append(f"stress pris −20 % OK (MODELLED): {price * _PRICE_FACTOR:g} > break-even {be:g}")
    else:
        p.notes.append("stress pris −20 % ej möjlig: DATA_MISSING (FS-känslighet, eller pris + break-even)")
    if irr_s is not None and npv_s is not None and irr_s < cfg.IRR_STRESS_MIN:
        cap(p, "IRR under stress", cfg.ECON_CAP_STRESS_WEAK, f"IRR {irr_s:g} % < {cfg.IRR_STRESS_MIN:g} %")

    # capex +20 %
    npv_c = read(company, "npv_stress_capex_musd", p)
    if npv_c is not None:
        if npv_c <= 0:
            cap(p, "capex +20 %", cfg.ECON_CAP_STRESS_NEGATIVE,
                f"NPV {npv_c:g} MUSD ≤ 0 · {src(company, 'npv_stress_capex_musd')}")
        else:
            p.notes.append(f"stress capex +20 %: NPV {npv_c:g} MUSD · {src(company, 'npv_stress_capex_musd')}")
    elif npv is not None and capex is not None:
        bound = npv - _CAPEX_FACTOR * capex
        if bound <= 0:
            cap(p, "capex +20 %", cfg.ECON_CAP_STRESS_NEGATIVE,
                f"MODELLED: NPV {npv:g} − 0,2 × CapEx {capex:g} = {bound:g} ≤ 0")
        else:
            p.notes.append(f"stress capex +20 % OK (MODELLED, konservativ): NPV ≥ {bound:g} MUSD")
    else:
        p.notes.append("stress capex +20 % ej möjlig: DATA_MISSING")
