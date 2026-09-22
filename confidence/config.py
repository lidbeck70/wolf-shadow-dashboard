"""
confidence/config.py — ALLA trösklar, poängtabeller och fältdefinitioner.

Inga magiska tal i scoringmodulerna: allt som avgör en poäng står här, med
kommentar om varifrån talet kommer. Tabeller markerade SPEC följer
kravspecen ordagrant; tabeller markerade VAL är val gjorda där specen
listar posterna utan tal (kommentaren säger hur valet gjordes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# Vokabulär
# ═══════════════════════════════════════════════════════════════════════════
STAGES = ("explorer", "developer", "producer", "royalty")
PRE_REVENUE = ("explorer", "developer")

# Projektmognad — finare än stage, styr Project Maturity i Confidence.
MATURITY = ("exploration", "pea", "pfs", "dfs", "fid", "construction", "production")
MATURITY_LABEL = {"exploration": "Prospektering", "pea": "PEA", "pfs": "PFS", "dfs": "DFS/BFS",
                  "fid": "FID / finansierat", "construction": "Byggnation",
                  "production": "Produktion"}

# Datatyper för proveniens (princip 19)
KINDS = ("ACTUAL", "ESTIMATE", "GUIDANCE", "MODELLED", "ASSUMPTION")

# Källtyper → Data Quality "Source quality" 0–5 (SPEC)
SOURCE_QUALITY = {
    "primary": 5,        # audited / official / primary source (43-101, JORC, årsredovisning)
    "independent": 4,    # high-quality independent source (QP-rapport, broker med egen modell)
    "secondary": 3,      # reputable secondary source
    "mixed": 2,
    "weak": 1,
    "unsupported": 0,
    "": 0,
}

# Färskhet i månader → 0–5 (SPEC)
FRESHNESS_TABLE = ((3, 5), (6, 4), (12, 3), (24, 2))      # < månader → poäng; äldre → 1; okänd → 0
FRESHNESS_OLD = 1
FRESHNESS_UNKNOWN = 0

# Resurskategorier (JORC/43-101) → vikt i Resource Certainty (VAL: proven=1.0 … target=0.1)
RESOURCE_CATEGORY_WEIGHT = {
    "proven": 1.0, "probable": 0.9, "measured": 0.75, "indicated": 0.6,
    "inferred": 0.3, "exploration_target": 0.1,
}

# Stress (SPEC)
STRESS_PRICE_PCT = -20.0
STRESS_CAPEX_PCT = 20.0

# ═══════════════════════════════════════════════════════════════════════════
# CASE SCORE — pelare (SPEC, summa 100)
# ═══════════════════════════════════════════════════════════════════════════
PILLARS = (
    ("strategic_commodity", "Strategic Commodity", 20),
    ("resource_quality", "Resource & Project Quality", 15),
    ("demand_scarcity", "Demand & Scarcity", 15),
    ("economics", "Economics", 15),
    ("production_growth", "Production & Growth", 10),
    ("balance_sheet", "Balance Sheet & Financing", 10),
    ("valuation", "Valuation", 10),
    ("management", "Management", 5),
)
PILLAR_MAX = {k: m for k, _l, m in PILLARS}

# Strategic Commodity 20 (SPEC)
STRATEGIC_SUB = {"strategic_significance": 10, "demand_growth": 5, "geopolitical_scarcity": 5}

# Resource & Project Quality 15 (SPEC)
RESOURCE_SUB = {"res_size": 2, "res_grade": 2, "res_metallurgy": 1, "res_mine_life": 1,
                "res_geology": 2, "res_infrastructure": 4, "res_expansion": 3}

# Demand & Scarcity 15 (SPEC). Balansen anges som underskott i % av
# efterfrågan: negativt = överskott. Tabellen läses uppifrån.
# Specen saknar raden underskott 0–5 %; den får 4 p (samma som överskott 0–5).
SUPPLY_BALANCE_TABLE = (
    (-10.0, 0),    # överskott > 10 %
    (-5.0, 2),     # överskott 5–10 %
    (0.0, 4),      # överskott 0–5 %
    (5.0, 4),      # underskott 0–5 %   (VAL — kontinuitet)
    (10.0, 6),     # underskott 5–10 %
    (20.0, 9),     # underskott 10–20 %
    (30.0, 12),    # underskott 20–30 %
)
SUPPLY_BALANCE_MAX_POINTS = 15    # underskott > 30 %
# Justeringar (SPEC listar dem, VAL ger varje −1 p; aldrig under 0)
SUPPLY_ADJUSTMENTS = ("adj_substitution", "adj_recycling", "adj_tech_change",
                      "adj_secondary_supply", "adj_project_delays")

# Economics 15 — producenter (SPEC: 4/3/3/3/2)
ECON_PRODUCER_SUB = {"aisc": 4, "ebitda_margin": 3, "fcf_yield": 3, "roic": 3, "breakeven": 2}
# Economics 15 — developers (SPEC listar posterna utan tal; VAL: 4/4/2/3/2)
ECON_DEVELOPER_SUB = {"npv_capex": 4, "irr": 4, "payback": 2, "breakeven": 3, "capex_intensity": 2}

# Trappor (VAL — Rick Rule-arkets 40/25 %-marginal återanvänds för AISC)
AISC_MARGIN_STEPS = ((40.0, 4), (25.0, 3), (15.0, 2), (0.0, 1))      # (pris−AISC)/pris % → p
EBITDA_MARGIN_STEPS = ((40.0, 3), (25.0, 2), (10.0, 1))
FCF_YIELD_STEPS = ((10.0, 3), (6.0, 2), (3.0, 1))
ROIC_STEPS = ((15.0, 3), (10.0, 2), (5.0, 1))
BREAKEVEN_COVER_STEPS_PROD = ((1.5, 2), (1.2, 1))                     # pris / breakeven
BREAKEVEN_COVER_STEPS_DEV = ((1.5, 3), (1.3, 2), (1.15, 1))
NPV_CAPEX_STEPS = ((2.0, 4), (1.5, 3), (1.0, 2), (0.7, 1))
IRR_STEPS = ((30.0, 4), (20.0, 3), (15.0, 2), (10.0, 1))
PAYBACK_STEPS = ((2.0, 2), (3.0, 1))                                   # ≤ år → p
CAPEX_INTENSITY_STEPS = ((1.0, 2), (2.0, 1))                           # capex / årsintäkt ≤ → p
# Stresstak (SPEC kräver stress; VAL: ekonomin får aldrig mer än så här om stressen faller)
ECON_CAP_STRESS_NEGATIVE = 5       # NPV ≤ 0 eller marginal ≤ 0 vid pris −20 % / capex +20 %
ECON_CAP_STRESS_WEAK = 8           # IRR under stress < IRR_STRESS_MIN
IRR_STRESS_MIN = 10.0

# Production & Growth 10 (SPEC)
TIME_TO_MONEY_TABLE = ((2.0, 10), (3.0, 9), (5.0, 8), (7.0, 6), (10.0, 4), (15.0, 2))   # < år → p
TIME_TO_MONEY_BEYOND = 1
DISCOVERY_OPTION_MAX = 5           # explorers, separat från huvudscoren

# Balance Sheet 10 — producenter (SPEC)
ND_EBITDA_TABLE = ((0.0, 10), (0.5, 9), (1.0, 8), (1.5, 7), (2.0, 5), (3.0, 3), (4.0, 1))  # < → p
ND_EBITDA_BEYOND = 0
# Balance Sheet 10 — developers (SPEC listar posterna; VAL: gap 4, runway 3, åtaganden 3)
FUNDING_GAP_STEPS = ((0.0, 4), (0.25, 3), (0.5, 2), (0.75, 1))       # gap / total capex ≤ → p
RUNWAY_STEPS = ((8, 3), (4, 2), (2, 1))                                # kvartal ≥ → p
COMMITMENT_POINTS = {"has_strategic_partner": 1, "has_offtake": 1, "committed_financing_ok": 1}
COMMITTED_FINANCING_MIN_SHARE = 0.25   # committed / capex ≥ 25 % räknas som åtagande
# Utspädning (SPEC listar "dilution" under Balance Sheet; VAL: DS ur controls.py —
# DS ≥ 6 låser köp där, ≥ 8 är EXTREM → −1 / −2 p, aldrig under 0)
DILUTION_PENALTY_STEPS = ((8, 2), (6, 1))                              # DS ≥ → avdrag

# Valuation 10 — producenter (SPEC listar måtten; VAL: 3/3/1/1/2)
EV_EBITDA_STEPS = ((4.0, 3), (6.0, 2), (8.0, 1))                       # ≤ → p
VAL_FCF_YIELD_STEPS = ((12.0, 3), (8.0, 2), (5.0, 1))
PE_STEPS = ((8.0, 1.0), (12.0, 0.5))
EV_EBIT_STEPS = ((6.0, 1.0), (9.0, 0.5))
NAV_PROD_STEPS = ((0.7, 2), (1.0, 1))                                  # P/NAV ≤ → p
# Valuation 10 — developers: P/NAV (SPEC)
P_NAV_TABLE = ((0.30, 10), (0.50, 9), (0.70, 8), (0.90, 7), (1.10, 5), (1.30, 3), (1.50, 1))  # < → p
P_NAV_BEYOND = 0

# Management 5 (SPEC listar sex bedömningar; VAL: fem delar à 1 p)
MANAGEMENT_SUB = {"mgmt_track_record": 1, "mgmt_capital_allocation": 1, "mgmt_dilution_history": 1,
                  "mgmt_insider_ownership": 1, "mgmt_alignment_delivery": 1}
INSIDER_OWNERSHIP_STEPS = ((10.0, 1.0), (3.0, 0.5))                    # % ≥ → p

# Betyg (SPEC)
RATING_BANDS = ((90, "ELITE"), (80, "PICK"), (70, "STRONG CANDIDATE"), (60, "WATCHLIST"),
                (50, "SPECULATIVE"), (0, "PASS"))

# ═══════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORE (SPEC, summa 100)
# ═══════════════════════════════════════════════════════════════════════════
CONFIDENCE_PARTS = (
    ("data_quality", "Data Quality", 20),
    ("resource_certainty", "Resource Certainty", 20),
    ("project_maturity", "Project Maturity", 15),
    ("economic_certainty", "Economic Certainty", 15),
    ("financing_certainty", "Financing Certainty", 10),
    ("timeline_certainty", "Timeline Certainty", 10),
    ("management_track_record", "Management Track Record", 10),
)
CONFIDENCE_MAX = {k: m for k, _l, m in CONFIDENCE_PARTS}

# Project Maturity 15 — grundnivå per mognad (SPEC), finjustering ±
MATURITY_BASE = {"exploration": (0, 3), "pea": (3, 6), "pfs": (6, 9), "dfs": (9, 12),
                 "fid": (12, 14), "construction": (14, 15), "production": (15, 15)}
MATURITY_ADJUSTERS = ("permits_granted", "engineering_done", "infrastructure_secured",
                      "financing_committed", "offtake_signed", "construction_contracts",
                      "procurement_started")

# Kill switches (SPEC) — appliceras EFTER summan
KILL_CAPS = (
    ("no_independent_resource", 50, "Ingen oberoende resursuppskattning"),
    ("no_financing_plan", 60, "Ingen realistisk finansieringsplan"),
    ("record_price_dependent", 65, "Projektet fungerar bara vid nära rekordhöga råvarupriser"),
    ("single_fragile_parameter", 70, "Hela caset vilar på en extremt osäker parameter"),
    ("capital_destruction_history", 75, "Dokumenterad historik av kraftig utspädning / kapitalförstöring"),
)

CONFIDENCE_BANDS = ((90, "VERIFIED"), (80, "HIGH CONFIDENCE"), (70, "GOOD CONFIDENCE"),
                    (60, "MODERATE"), (50, "SPECULATIVE"), (0, "LOW CONFIDENCE"))

# ═══════════════════════════════════════════════════════════════════════════
# Fältregister — allt en analys kan innehålla. Nyckel, etikett, enhet, typ,
# vilka stages det gäller, vilken pelare/del som läser det.
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    unit: str
    kind: str                     # number | int | bool | choice | text | date
    stages: tuple                 # () = alla
    pillar: str                   # case-pelare eller confidence-del som läser fältet
    max: Optional[float] = None   # för int-delpoäng
    choices: tuple = ()
    hint: str = ""


_ALL: tuple = ()
_PRE = PRE_REVENUE
_PROD = ("producer", "royalty")

FIELDS: tuple = (
    # ── Resource & Project Quality (manuella delpoäng 0–max) ─────────────
    FieldSpec("res_size", "Resursstorlek", "p", "int", _ALL, "resource_quality", 2,
              hint="0 liten · 1 medel · 2 tier 1-storlek för råvaran"),
    FieldSpec("res_grade", "Halt", "p", "int", _ALL, "resource_quality", 2,
              hint="0 under branschsnitt · 1 snitt · 2 över snitt (AQS kostnadsposition 2 → 2)"),
    FieldSpec("res_metallurgy", "Metallurgi / recovery", "p", "int", _ALL, "resource_quality", 1,
              hint="1 bevisad enkel · 0 problematisk eller obevisad"),
    FieldSpec("res_mine_life", "Gruvlivslängd", "p", "int", _ALL, "resource_quality", 1,
              hint="1 om > 10 år (AQS livslängd 2 → 1)"),
    FieldSpec("res_geology", "Geologi", "p", "int", _ALL, "resource_quality", 2,
              hint="0 komplex · 1 normal · 2 enkel, förutsägbar"),
    FieldSpec("res_infrastructure", "Infrastruktur / läge", "p", "int", _ALL, "resource_quality", 4,
              hint="0 saknas · 2 delvis · 4 väg, kraft, vatten, tillstånd (AQS 0/1/2 → 0/2/4)"),
    FieldSpec("res_expansion", "Expansionspotential", "p", "int", _ALL, "resource_quality", 3,
              hint="0 ingen · 1 viss · 3 tydlig, billig (AQS 0/1/2 → 0/1/3)"),
    FieldSpec("mine_life_years", "Gruvlivslängd", "år", "number", _ALL, "resource_quality"),
    # ── Economics — producenter ───────────────────────────────────────────
    FieldSpec("commodity_price", "Råvarupris nu", "USD/enhet", "number", _ALL, "economics"),
    FieldSpec("aisc", "AISC / opex per enhet", "USD/enhet", "number", _PROD, "economics"),
    FieldSpec("ebitda_margin_pct", "EBITDA-marginal", "%", "number", _PROD, "economics"),
    FieldSpec("fcf_yield_pct", "FCF-yield", "%", "number", _PROD, "economics"),
    FieldSpec("roic_pct", "ROIC", "%", "number", _PROD, "economics"),
    FieldSpec("breakeven_price", "Break-even råvarupris", "USD/enhet", "number", _ALL, "economics"),
    # ── Economics — developers ────────────────────────────────────────────
    FieldSpec("npv_musd", "NPV efter skatt", "MUSD", "number", _PRE, "economics"),
    FieldSpec("npv_discount_pct", "Diskonteringsränta i NPV", "%", "number", _PRE, "economics"),
    FieldSpec("npv_price_assumption", "Råvarupris i NPV", "USD/enhet", "number", _PRE, "economics"),
    FieldSpec("capex_musd", "Initial CapEx", "MUSD", "number", _PRE, "economics"),
    FieldSpec("irr_pct", "IRR efter skatt", "%", "number", _PRE, "economics"),
    FieldSpec("payback_years", "Payback", "år", "number", _PRE, "economics"),
    FieldSpec("annual_production", "Årsproduktion", "enheter/år", "number", _ALL, "economics"),
    FieldSpec("production_unit", "Produktionsenhet", "", "text", _ALL, "economics",
              hint="oz, lb, t, boe …"),
    FieldSpec("npv_stress_price_musd", "NPV vid pris −20 %", "MUSD", "number", _PRE, "economics",
              hint="Ur FS-känslighetstabellen. Saknas → modelleras i scenarier."),
    FieldSpec("npv_stress_capex_musd", "NPV vid CapEx +20 %", "MUSD", "number", _PRE, "economics"),
    FieldSpec("irr_stress_price_pct", "IRR vid pris −20 %", "%", "number", _PRE, "economics"),
    # ── Production & Growth ───────────────────────────────────────────────
    FieldSpec("first_cashflow_year", "Första kassaflöde (år)", "år", "int", _ALL, "production_growth",
              hint="Producenter: innevarande år."),
    FieldSpec("discovery_option", "Discovery Option", "p", "int", ("explorer",), "production_growth", 5,
              hint="0–5, hålls separat från huvudscoren"),
    # ── Balance Sheet & Financing ─────────────────────────────────────────
    FieldSpec("net_debt_ebitda", "Nettoskuld/EBITDA", "×", "number", _PROD, "balance_sheet"),
    FieldSpec("cash_musd", "Kassa", "MUSD", "number", _ALL, "balance_sheet"),
    FieldSpec("debt_musd", "Skuld", "MUSD", "number", _ALL, "balance_sheet"),
    FieldSpec("quarterly_burn_musd", "Burn per kvartal", "MUSD", "number", _PRE, "balance_sheet"),
    FieldSpec("committed_financing_musd", "Åtagen finansiering", "MUSD", "number", _PRE, "balance_sheet"),
    FieldSpec("has_strategic_partner", "Strategisk partner", "", "bool", _PRE, "balance_sheet"),
    FieldSpec("has_offtake", "Offtake-avtal", "", "bool", _PRE, "balance_sheet"),
    FieldSpec("dilution_score", "DS (utspädning 0–10)", "p", "int", _ALL, "balance_sheet", 10,
              hint="Från kontrollerna (controls.ds_total)"),
    # ── Valuation ─────────────────────────────────────────────────────────
    FieldSpec("market_cap_musd", "Börsvärde", "MUSD", "number", _ALL, "valuation"),
    FieldSpec("enterprise_value_musd", "Enterprise value", "MUSD", "number", _ALL, "valuation"),
    FieldSpec("ev_ebitda", "EV/EBITDA", "×", "number", _PROD, "valuation"),
    FieldSpec("pe", "P/E", "×", "number", _PROD, "valuation"),
    FieldSpec("ev_ebit", "EV/EBIT", "×", "number", _PROD, "valuation"),
    FieldSpec("nav_musd", "NAV (efter skatt)", "MUSD", "number", _ALL, "valuation"),
    FieldSpec("p_nav", "P/NAV", "×", "number", _ALL, "valuation",
              hint="Räknas som börsvärde / NAV om båda finns."),
    # ── Management (Case) ─────────────────────────────────────────────────
    FieldSpec("mgmt_track_record", "Byggmeriter", "p", "int", _ALL, "management", 1),
    FieldSpec("mgmt_capital_allocation", "Kapitalallokering", "p", "int", _ALL, "management", 1),
    FieldSpec("mgmt_dilution_history", "Utspädningshistorik", "p", "int", _ALL, "management", 1,
              hint="1 = disciplinerad"),
    FieldSpec("insider_ownership_pct", "Insynsägande", "%", "number", _ALL, "management"),
    FieldSpec("mgmt_alignment_delivery", "Alignment & leverans mot löften", "p", "int", _ALL,
              "management", 1),
    # ── Confidence: Resource Certainty ────────────────────────────────────
    FieldSpec("resource_category", "Högsta resurskategori", "", "choice", _ALL, "resource_certainty",
              choices=tuple(RESOURCE_CATEGORY_WEIGHT)),
    FieldSpec("resource_verified_share_pct", "Andel oberoende verifierad", "%", "number", _ALL,
              "resource_certainty"),
    FieldSpec("independent_resource_estimate", "Oberoende resursuppskattning finns", "", "bool", _ALL,
              "resource_certainty", hint="43-101/JORC av oberoende QP"),
    FieldSpec("drilling_density", "Borrtäthet", "p", "int", _ALL, "resource_certainty", 2,
              hint="0 gles · 1 normal · 2 tät"),
    FieldSpec("resource_conversion_history", "Konverteringshistorik", "p", "int", _ALL,
              "resource_certainty", 2, hint="0 nedskrivningar · 1 stabil · 2 uppgraderingar"),
    FieldSpec("grade_consistency", "Haltkonsistens", "p", "int", _ALL, "resource_certainty", 2),
    FieldSpec("metallurgy_confidence", "Metallurgisk säkerhet", "p", "int", _ALL, "resource_certainty", 2,
              hint="0 labb · 1 pilot · 2 kommersiell drift"),
    # ── Confidence: Project Maturity ──────────────────────────────────────
    FieldSpec("permits_granted", "Nyckeltillstånd beviljade", "", "bool", _ALL, "project_maturity"),
    FieldSpec("engineering_done", "Detaljprojektering klar", "", "bool", _PRE, "project_maturity"),
    FieldSpec("infrastructure_secured", "Infrastruktur säkrad", "", "bool", _PRE, "project_maturity"),
    FieldSpec("financing_committed", "Finansiering åtagen", "", "bool", _PRE, "project_maturity"),
    FieldSpec("offtake_signed", "Offtake signerat", "", "bool", _PRE, "project_maturity"),
    FieldSpec("construction_contracts", "Byggkontrakt tecknade", "", "bool", _PRE, "project_maturity"),
    FieldSpec("procurement_started", "Upphandling påbörjad", "", "bool", _PRE, "project_maturity"),
    # ── Confidence: Timeline ──────────────────────────────────────────────
    FieldSpec("timeline_documented", "Tidsplan dokumenterad och finansierad", "", "bool", _ALL,
              "timeline_certainty"),
    FieldSpec("historical_delays", "Tidigare förseningar", "p", "int", _ALL, "timeline_certainty", 2,
              hint="0 upprepade · 1 någon · 2 inga"),
    # ── Confidence: kill switches (manuella bedömningar med belägg) ───────
    FieldSpec("no_financing_plan", "Ingen realistisk finansieringsplan", "", "bool", _PRE, "kill"),
    FieldSpec("record_price_dependent", "Fungerar bara vid rekordpris", "", "bool", _ALL, "kill"),
    FieldSpec("single_fragile_parameter", "Caset vilar på en osäker parameter", "", "bool", _ALL, "kill"),
    FieldSpec("capital_destruction_history", "Dokumenterad kapitalförstöring", "", "bool", _ALL, "kill"),
)
FIELD_BY_KEY: dict = {f.key: f for f in FIELDS}


def fields_for(stage: str, pillar: Optional[str] = None) -> list:
    """Fälten som gäller ett stage (och ev. en pelare/del)."""
    out = []
    for f in FIELDS:
        if f.stages and stage not in f.stages:
            continue
        if pillar and f.pillar != pillar:
            continue
        out.append(f)
    return out


# Stage → allokeringssleeve i allocator.py (VAL: ingen ändring i allocator;
# mappning tills modellen bevisat sig)
STAGE_SLEEVE = {"producer": "producenter", "developer": "optionalitet",
                "explorer": "optionalitet", "royalty": "royalty"}
