"""
confidence/scenarios/engine.py — Bear/Base/Bull/Super Bull, asymmetri och
5×/10×-motorn. (SPEC: pris → produktion → intäkt → EBITDA → FCF → NAV/EV →
equity → aktiekurs, alla antaganden synliga, inga magiska multiplar.)

Producenter/royalty:  pris_s = pris × (1 + Δ); intäkt = produktion × pris_s;
    EBITDA ≈ produktion × (pris_s − AISC) [royalty: intäkt − fasta kostnader];
    FCF = EBITDA × (1 − skatt); EV = EBITDA × EV/EBITDA-antagande;
    equity = EV − skuld + kassa; aktie = equity / aktier.
Developers:  NAV_s = NPV + lutning × Δ % (lutning ur FS-känsligheten vid −20 %,
    annars annuitet av Δpris × produktion × (1 − skatt)); NAV_s −= ΔCapEx;
    equity = NAV_s × P/NAV-antagande + kassa − skuld.
Explorers utan NPV får inga scenarier — DATA_MISSING, inte en gissning.

Varje scenario listar sina steg i klartext och sina antaganden med kind
(ACTUAL/ESTIMATE/MODELLED/ASSUMPTION) och källa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from confidence import config as cfg
from confidence.data.models import CompanyInput
from confidence.data.provenance import describe
from confidence.data.validation import usable

LOW_CONF, MID_CONF, HIGH_CONF = "låg", "medel", "hög"


@dataclass
class Assumption:
    name: str
    value: float
    unit: str = ""
    kind: str = "ASSUMPTION"
    source: str = ""

    def text(self) -> str:
        return f"{self.name}: {self.value:,.4g}{(' ' + self.unit) if self.unit else ''} ({self.kind}" + \
               (f", {self.source})" if self.source else ")")


@dataclass
class Scenario:
    key: str
    label: str
    price_change_pct: float
    capex_change_pct: float
    price: Optional[float] = None
    revenue_musd: Optional[float] = None
    ebitda_musd: Optional[float] = None
    fcf_musd: Optional[float] = None
    value_musd: Optional[float] = None       # EV (producent) eller NAV (developer)
    equity_musd: Optional[float] = None
    share_price: Optional[float] = None
    upside_pct: Optional[float] = None       # mot börsvärde (eller kurs)
    steps: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.upside_pct is not None


@dataclass
class MultiplierPath:
    target: float                             # 5 eller 10
    required_price: Optional[float] = None
    price_change_pct: Optional[float] = None
    production_factor: Optional[float] = None   # produktion × faktor vid dagens pris
    required_p_nav: Optional[float] = None      # developers: enbart omvärdering
    steps: list = field(default_factory=list)


@dataclass
class Asymmetry:
    upside_pct: Optional[float]              # Bull
    downside_pct: Optional[float]            # Bear
    ratio: Optional[float]                   # upside / |downside|
    band: str
    expected_pct: Optional[float]            # sannolikhetsvägd
    probs: dict


@dataclass
class ScenarioSet:
    stage: str
    scenarios: list
    asymmetry: Optional[Asymmetry]
    paths: list                              # [MultiplierPath]
    assumptions: list                        # [Assumption] gemensamma
    missing: list = field(default_factory=list)
    flags: list = field(default_factory=list)

    def scenario(self, key: str) -> Optional[Scenario]:
        return next((s for s in self.scenarios if s.key == key), None)


# ── inläsning med bokföring ──────────────────────────────────────────────────
class _Inputs:
    def __init__(self, company: CompanyInput):
        self.c = company
        self.missing: list = []
        self.assumptions: list = []

    def num(self, key: str, required: bool = True) -> Optional[float]:
        if usable(self.c, key):
            v = self.c.num(key)
            p = self.c.get(key)
            self.assumptions.append(Assumption(cfg.FIELD_BY_KEY[key].label, v, p.unit or cfg.FIELD_BY_KEY[key].unit,
                                               p.kind, p.source))
            return v
        if required and key not in self.missing:
            self.missing.append(key)
        return None

    def default(self, key: str, fallback: float, label: str, unit: str = "") -> float:
        v = self.num(key, required=False)
        if v is not None:
            return v
        self.assumptions.append(Assumption(label, fallback, unit, "ASSUMPTION", "confidence.config"))
        return fallback


def _anchor(inp: _Inputs) -> tuple:
    """(börsvärde, aktier, kurs) — börsvärdet är ankaret för upside; kurs/aktier om de finns."""
    mcap = inp.num("market_cap_musd")
    shares = inp.num("shares_outstanding_m", required=False)
    price = inp.num("share_price", required=False)
    if mcap is None and shares and price:
        mcap = shares * price
        inp.assumptions.append(Assumption("Börsvärde", mcap, "MUSD", "MODELLED", "aktier × kurs"))
    return mcap, shares, price


# ── producent / royalty ──────────────────────────────────────────────────────
def _producer_scenarios(inp: _Inputs) -> tuple:
    c = inp.c
    price = inp.num("commodity_price")
    prod = inp.num("annual_production")
    unit = str(c.get("production_unit").value) if c.has("production_unit") else "enheter"
    tax = inp.default("tax_rate_pct", cfg.DEFAULT_TAX_RATE_PCT, "Skattesats", "%") / 100
    mult = inp.default("target_ev_ebitda", cfg.DEFAULT_TARGET_EV_EBITDA, "EV/EBITDA i scenariot", "×")
    debt = inp.default("debt_musd", 0.0, "Skuld", "MUSD")
    cash = inp.default("cash_musd", 0.0, "Kassa", "MUSD")
    mcap, shares, _px = _anchor(inp)
    royalty = c.stage == "royalty"
    aisc = None if royalty else inp.num("aisc")
    margin_pct = inp.num("ebitda_margin_pct") if royalty else None
    out = []
    for key, label, dp_, dc in cfg.SCENARIOS:
        s = Scenario(key, label, dp_, dc)
        if price is None or prod is None:
            s.steps.append("DATA_MISSING: råvarupris eller årsproduktion saknas")
            out.append(s)
            continue
        s.price = price * (1 + dp_ / 100)
        s.revenue_musd = prod * s.price / 1e6
        s.steps.append(f"pris {price:g} × (1 {dp_:+g} %) = {s.price:,.4g}")
        s.steps.append(f"intäkt = {prod:,.4g} {unit} × {s.price:,.4g} = {s.revenue_musd:,.0f} MUSD "
                       f"(förutsätter samma enhet för produktion och pris)")
        if royalty:
            if margin_pct is None:
                s.steps.append("DATA_MISSING: EBITDA-marginal saknas (royalty)")
                out.append(s)
                continue
            fixed = prod * price / 1e6 * (1 - margin_pct / 100)
            s.ebitda_musd = s.revenue_musd - fixed
            s.steps.append(f"EBITDA = intäkt − fasta kostnader {fixed:,.0f} MUSD (bas-intäkt × (1 − {margin_pct:g} %)) "
                           f"= {s.ebitda_musd:,.0f} MUSD")
        else:
            if aisc is None:
                s.steps.append("DATA_MISSING: AISC saknas")
                out.append(s)
                continue
            s.ebitda_musd = prod * (s.price - aisc) / 1e6
            s.steps.append(f"EBITDA ≈ {prod:,.4g} × ({s.price:,.4g} − AISC {aisc:g}) = {s.ebitda_musd:,.0f} MUSD "
                           f"(AISC-marginal; sustaining capex ligger i AISC)")
        s.fcf_musd = s.ebitda_musd * (1 - tax) if s.ebitda_musd > 0 else s.ebitda_musd
        s.steps.append(f"FCF = EBITDA × (1 − skatt {tax * 100:g} %) = {s.fcf_musd:,.0f} MUSD")
        s.value_musd = s.ebitda_musd * mult
        s.steps.append(f"EV = EBITDA × {mult:g} = {s.value_musd:,.0f} MUSD")
        s.equity_musd = s.value_musd - debt + cash
        s.steps.append(f"equity = EV − skuld {debt:g} + kassa {cash:g} = {s.equity_musd:,.0f} MUSD")
        _finish_equity(s, mcap, shares)
        out.append(s)
    paths = _producer_paths(inp, price, prod, aisc if not royalty else None, mult, debt, cash, mcap, royalty,
                            margin_pct)
    return out, paths


def _producer_paths(inp, price, prod, aisc, mult, debt, cash, mcap, royalty, margin_pct) -> list:
    paths = []
    for k in cfg.MULTIPLIER_TARGETS:
        mp = MultiplierPath(k)
        if mcap is None or price is None or prod is None or (aisc is None and not royalty):
            mp.steps.append("DATA_MISSING: börsvärde, pris, produktion eller AISC saknas")
            paths.append(mp)
            continue
        ev_req = k * mcap + debt - cash
        ebitda_req = ev_req / mult
        mp.steps.append(f"{k:g}× börsvärde {mcap:,.0f} → equity {k * mcap:,.0f} → EV {ev_req:,.0f} MUSD "
                        f"→ EBITDA {ebitda_req:,.0f} MUSD vid EV/EBITDA {mult:g}")
        if royalty:
            fixed = prod * price / 1e6 * (1 - (margin_pct or 0) / 100)
            rev_req = ebitda_req + fixed
            mp.required_price = rev_req * 1e6 / prod
        else:
            mp.required_price = aisc + ebitda_req * 1e6 / prod
        mp.price_change_pct = (mp.required_price / price - 1) * 100
        mp.steps.append(f"vid dagens produktion krävs pris {mp.required_price:,.4g} ({mp.price_change_pct:+.0f} %)")
        base_margin = None if royalty else (price - aisc)
        if base_margin and base_margin > 0:
            mp.production_factor = (ebitda_req * 1e6 / base_margin) / prod
            mp.steps.append(f"eller vid dagens pris: produktion × {mp.production_factor:.1f}")
        paths.append(mp)
    return paths


# ── developer ────────────────────────────────────────────────────────────────
def _developer_scenarios(inp: _Inputs) -> tuple:
    c = inp.c
    npv = inp.num("npv_musd")
    capex = inp.num("capex_musd")
    price = inp.num("commodity_price")
    prod = inp.num("annual_production", required=False)
    be = inp.num("breakeven_price", required=False)
    unit = str(c.get("production_unit").value) if c.has("production_unit") else "enheter"
    tax = inp.default("tax_rate_pct", cfg.DEFAULT_TAX_RATE_PCT, "Skattesats", "%") / 100
    pnav = inp.default("target_p_nav", cfg.DEFAULT_TARGET_P_NAV, "P/NAV vid omvärdering", "×")
    debt = inp.default("debt_musd", 0.0, "Skuld", "MUSD")
    cash = inp.default("cash_musd", 0.0, "Kassa", "MUSD")
    mcap, shares, _px = _anchor(inp)
    slope, slope_text = _nav_slope(inp, npv, prod, tax)
    out = []
    for key, label, dp_, dc in cfg.SCENARIOS:
        s = Scenario(key, label, dp_, dc)
        if npv is None or price is None:
            s.steps.append("DATA_MISSING: NPV eller råvarupris saknas — scenarier kräver minst en PEA")
            out.append(s)
            continue
        s.price = price * (1 + dp_ / 100)
        s.steps.append(f"pris {price:g} × (1 {dp_:+g} %) = {s.price:,.4g}")
        if prod is not None:
            s.revenue_musd = prod * s.price / 1e6
            s.steps.append(f"steady state-intäkt = {prod:,.4g} {unit} × {s.price:,.4g} = {s.revenue_musd:,.0f} MUSD")
            if be is not None:
                s.ebitda_musd = prod * (s.price - be) / 1e6
                s.fcf_musd = s.ebitda_musd * (1 - tax) if s.ebitda_musd > 0 else s.ebitda_musd
                s.steps.append(f"marginal ≈ {prod:,.4g} × ({s.price:,.4g} − break-even {be:g}) = "
                               f"{s.ebitda_musd:,.0f} MUSD; FCF × (1 − skatt {tax * 100:g} %) = {s.fcf_musd:,.0f} MUSD")
        if slope is None:
            s.steps.append(f"DATA_MISSING: kan inte flytta NAV med priset ({slope_text})")
            out.append(s)
            continue
        nav = npv + slope * dp_
        s.steps.append(f"NAV = NPV {npv:,.0f} + {slope:,.1f} MUSD per % × {dp_:+g} % = {nav:,.0f} MUSD ({slope_text})")
        if abs(dp_) > abs(cfg.STRESS_PRICE_PCT):
            s.steps.append("OBS: linjär extrapolation utanför FS-känslighetens intervall")
        if dc and capex is not None:
            nav -= capex * dc / 100
            s.steps.append(f"CapEx {dc:+g} % → NAV − {capex * dc / 100:,.0f} MUSD = {nav:,.0f} MUSD (odiskonterat, konservativt)")
        s.value_musd = nav
        s.equity_musd = max(nav, 0.0) * pnav + cash - debt
        s.steps.append(f"equity = max(NAV, 0) × P/NAV {pnav:g} + kassa {cash:g} − skuld {debt:g} = {s.equity_musd:,.0f} MUSD")
        _finish_equity(s, mcap, shares)
        out.append(s)
    paths = _developer_paths(inp, npv, price, slope, pnav, debt, cash, mcap)
    return out, paths


def _nav_slope(inp: _Inputs, npv, prod, tax) -> tuple:
    """MUSD NAV per % prisändring: ur FS-känsligheten, annars annuitet."""
    if npv is None:
        return None, "NPV saknas"
    s = inp.num("npv_stress_price_musd", required=False)
    if s is not None:
        slope = (npv - s) / abs(cfg.STRESS_PRICE_PCT)
        return slope, f"MODELLED linjärt ur FS-känsligheten: NPV {s:,.0f} vid {cfg.STRESS_PRICE_PCT:g} %"
    price = inp.num("commodity_price", required=False)
    life = inp.num("mine_life_years", required=False)
    if prod is None or price is None or life is None:
        inp.missing.append("npv_stress_price_musd")
        return None, "FS-känslighet saknas och produktion/pris/gruvliv räcker inte för annuitet"
    r = inp.default("npv_discount_pct", 8.0, "Diskonteringsränta", "%") / 100
    annuity = (1 - (1 + r) ** -life) / r if r > 0 else life
    slope = price / 100 * prod / 1e6 * (1 - tax) * annuity
    return slope, (f"MODELLED annuitet: 1 % pris = {price / 100:g} × {prod:,.4g} × (1 − skatt) × "
                   f"annuitet({r * 100:g} %, {life:g} år) = {slope:,.1f} MUSD")


def _developer_paths(inp, npv, price, slope, pnav, debt, cash, mcap) -> list:
    paths = []
    for k in cfg.MULTIPLIER_TARGETS:
        mp = MultiplierPath(k)
        if mcap is None or npv is None or npv <= 0:
            mp.steps.append("DATA_MISSING: börsvärde eller positivt NPV saknas")
            paths.append(mp)
            continue
        eq_req = k * mcap
        mp.required_p_nav = eq_req / npv
        mp.steps.append(f"{k:g}× börsvärde {mcap:,.0f} = {eq_req:,.0f} MUSD → enbart omvärdering kräver "
                        f"P/NAV {mp.required_p_nav:.2f}× på dagens NAV")
        nav_req = (eq_req - cash + debt) / pnav
        mp.steps.append(f"vid P/NAV {pnav:g}: NAV måste bli {nav_req:,.0f} MUSD")
        if slope and price:
            mp.price_change_pct = (nav_req - npv) / slope
            mp.required_price = price * (1 + mp.price_change_pct / 100)
            mp.steps.append(f"→ pris {mp.required_price:,.4g} ({mp.price_change_pct:+.0f} %) med dagens plan"
                            + (" — OBS långt utanför känslighetsintervallet" if abs(mp.price_change_pct) > 50 else ""))
        paths.append(mp)
    return paths


# ── gemensamt ────────────────────────────────────────────────────────────────
def _finish_equity(s: Scenario, mcap, shares) -> None:
    if shares:
        s.share_price = s.equity_musd / shares
        s.steps.append(f"aktie = {s.equity_musd:,.0f} / {shares:g} M = {s.share_price:,.3g}")
    if mcap:
        s.upside_pct = (s.equity_musd / mcap - 1) * 100
        s.steps.append(f"mot börsvärde {mcap:,.0f} MUSD: {s.upside_pct:+.0f} %")
    else:
        s.steps.append("DATA_MISSING: börsvärde saknas — ingen upside")
    for attr in ("price", "revenue_musd", "ebitda_musd", "fcf_musd", "value_musd", "equity_musd",
                 "share_price", "upside_pct"):
        v = getattr(s, attr)
        if v is not None:
            setattr(s, attr, round(v, 6))          # flyttalsbrus bort, siffrorna blir läsbara


def asymmetry(scenarios: list) -> Optional[Asymmetry]:
    by = {s.key: s for s in scenarios}
    bull, bear = by.get("bull"), by.get("bear")
    if not bull or not bear or not bull.complete or not bear.complete:
        return None
    up, down = bull.upside_pct, bear.upside_pct
    ratio = None
    if down < 0:
        ratio = up / abs(down)
    elif up > 0:
        ratio = float("inf")
    if ratio is None or ratio < 0:
        band = cfg.ASYMMETRY_BANDS[-1][1]
    else:
        band = next((lbl for floor, lbl in cfg.ASYMMETRY_BANDS if ratio >= floor), cfg.ASYMMETRY_BANDS[-1][1])
    exp = sum(cfg.SCENARIO_PROBS[k] / 100 * by[k].upside_pct for k in cfg.SCENARIO_PROBS
              if k in by and by[k].complete)
    return Asymmetry(up, down, ratio, band, exp, dict(cfg.SCENARIO_PROBS))


def scenario_set(company: CompanyInput) -> ScenarioSet:
    inp = _Inputs(company)
    if company.stage == "explorer" and not company.has("npv_musd"):
        flags = ["explorer utan NPV: inga scenarier (DATA_MISSING) — Discovery Option bär caset"]
        return ScenarioSet(company.stage, [], None, [], [], ["npv_musd"], flags)
    if company.stage in cfg.PRE_REVENUE:
        scen, paths = _developer_scenarios(inp)
    else:
        scen, paths = _producer_scenarios(inp)
    asym = asymmetry(scen)
    flags = []
    if not all(s.complete for s in scen):
        flags.append("ofullständiga scenarier — se DATA_MISSING i stegen")
    for a in inp.assumptions:
        if a.kind == "ASSUMPTION" and a.source == "confidence.config":
            flags.append(f"antagande ur config: {a.text()}")
    seen, uniq = set(), []
    for a in inp.assumptions:
        if a.name not in seen:
            seen.add(a.name)
            uniq.append(a)
    return ScenarioSet(company.stage, scen, asym, paths, uniq, inp.missing, flags)


def describe_assumptions(company: CompanyInput, keys: tuple) -> list:
    return [describe(company.get(k), cfg.FIELD_BY_KEY[k].label) for k in keys]
