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

# Data Quality 20 (SPEC: fyra delar à 0–5). Källkvalitet och färskhet räknas ur
# proveniensen (snitt över alla giltiga fält); oberoende verifiering och
# källkonsistens är bedömningar 0–5 (fält nedan) — de kan inte räknas ur en
# datapunkt per fält utan att gissa.
DATA_QUALITY_SUB = {"source_quality": 5, "freshness": 5, "independent_verification": 5,
                    "cross_source_consistency": 5}

# Resource Certainty 20 (SPEC listar faktorerna; VAL: 8/3/3/2/2/1/1)
RESOURCE_CERTAINTY_SUB = {"resource_category": 8, "independent_resource_estimate": 3,
                          "resource_verified_share_pct": 3, "drilling_density": 2,
                          "resource_conversion_history": 2, "grade_consistency": 1,
                          "metallurgy_confidence": 1}

# Project Maturity 15 — grundnivå per mognad (SPEC), finjustering ±
MATURITY_BASE = {"exploration": (0, 3), "pea": (3, 6), "pfs": (6, 9), "dfs": (9, 12),
                 "fid": (12, 14), "construction": (14, 15), "production": (15, 15)}
MATURITY_ADJUSTERS = ("permits_granted", "engineering_done", "infrastructure_secured",
                      "financing_committed", "offtake_signed", "construction_contracts",
                      "procurement_started")

# Economic Certainty 15 (SPEC: stress bas / pris −20 % / capex +20 %; rekordpris-
# projekt kraftigt reducerade). VAL: prisstress 6, capexstress 5, studiekvalitet 4.
# Överlevnad = stressat NPV / bas-NPV (0–1) × delpoäng. En MODELLED-bound är
# mindre säker än FS-känsligheten → lägre tak.
ECON_CERTAINTY_SUB = {"price_stress": 6, "capex_stress": 5, "study_quality": 4}
ECON_CERTAINTY_MODELLED_CAP = {"price_stress": 4, "capex_stress": 3}
STUDY_QUALITY_BY_MATURITY = {"exploration": 0, "pea": 1, "pfs": 3, "dfs": 4, "fid": 4,
                             "construction": 4, "production": 4}
NPV_PRICE_ABOVE_SPOT_PCT = 10.0        # NPV-pris > spot × 1,10 → avdrag
NPV_PRICE_ABOVE_SPOT_PENALTY = 3
RECORD_PRICE_FACTOR = 0.3              # record_price_dependent → Economic Certainty × 0,3

# Financing Certainty 10 (VAL). Developers: åtagen andel av CapEx 7 + runway 3.
# Explorers: runway (skalas). Producenter/royalty: nettoskuld/EBITDA.
FIN_COMMITTED_SHARE_STEPS = ((1.0, 7), (0.5, 5), (0.25, 3), (0.01, 1))   # ≥ andel → p
FIN_RUNWAY_STEPS = ((8, 3), (4, 2), (2, 1))                                # kvartal ≥ → p
FIN_ND_EBITDA_STEPS = ((1.0, 10), (2.0, 7), (3.0, 4))                      # < → p, annars 1
FIN_ND_EBITDA_BEYOND = 1

# Timeline Certainty 10 (VAL). Developers: dokumenterad & finansierad plan 4,
# förseningshistorik 0–2 → 0–4, mognad 0–2. Producenter: 8 + förseningar 0–2.
TIMELINE_DOCUMENTED_POINTS = 4
TIMELINE_DELAYS_FACTOR = 2             # historical_delays (0–2) × 2
TIMELINE_MATURITY_POINTS = {"fid": 1, "construction": 2, "production": 2}
TIMELINE_PRODUCER_BASE = 8

# Management Track Record 10 (VAL: samma bedömningar som Case-Management, viktade)
MGMT_TRACK_SUB = {"mgmt_track_record": 3, "mgmt_alignment_delivery": 3,
                  "mgmt_dilution_history": 2, "mgmt_capital_allocation": 2}

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
# SCENARIER, ASYMMETRI, 5×/10×, TIME-TO-MONEY (SPEC listar; talen är VAL och
# syns som ASSUMPTION i varje scenariorapport — inga dolda multiplar)
# ═══════════════════════════════════════════════════════════════════════════
# (nyckel, etikett, prisändring %, capexändring %)
SCENARIOS = (("bear", "Bear", -30.0, 20.0), ("base", "Base", 0.0, 0.0),
             ("bull", "Bull", 30.0, 0.0), ("super_bull", "Super Bull", 80.0, 0.0))
SCENARIO_PROBS = {"bear": 25.0, "base": 50.0, "bull": 20.0, "super_bull": 5.0}   # summa 100
DEFAULT_TAX_RATE_PCT = 25.0            # ASSUMPTION när tax_rate_pct saknas
DEFAULT_TARGET_EV_EBITDA = 5.0         # ASSUMPTION producenter: EV/EBITDA i scenariot
DEFAULT_TARGET_P_NAV = 0.7             # ASSUMPTION developers: P/NAV vid omvärdering
MULTIPLIER_TARGETS = (5.0, 10.0)       # "vad måste hända" för 5× och 10×
ASYMMETRY_BANDS = ((3.0, "STARK ASYMMETRI"), (2.0, "ASYMMETRI"), (1.0, "SYMMETRISK"), (0.0, "NEGATIV"))
# Time-to-money: typiska år kvar per mognadssteg (VAL, branschtypiskt) och
# konfidens per steg. Bolagets eget årtal jämförs med detta.
TTM_STAGE_YEARS = (("exploration", "Prospektering → PEA", 2.0, "låg"),
                   ("pea", "PEA → PFS", 1.5, "låg"),
                   ("pfs", "PFS → DFS", 1.5, "medel"),
                   ("dfs", "DFS → tillstånd + finansiering (FID)", 1.5, "medel"),
                   ("fid", "FID → byggstart", 0.5, "hög"),
                   ("construction", "Byggnation → första kassaflöde", 2.5, "hög"))
TTM_AGGRESSIVE_RATIO = 0.6             # bolagets plan < 60 % av typiskt → "aggressiv plan"

# ═══════════════════════════════════════════════════════════════════════════
# WHY NOW 0–100 (SPEC listar; VAL: sex signaler ur repots egna moduler)
# Signaler som saknas ger 0 och skalas pro rata; täckningen redovisas.
# ═══════════════════════════════════════════════════════════════════════════
WHY_NOW_SUB = {"cycle": 30, "triple_signal": 25, "ratio": 15, "complex": 10,
               "supply": 10, "time_to_money": 10}
WHY_NOW_CYCLE = {"TIDIG": 30, "MITTEN": 20, "SEN": 8, "TOPP": 0}           # blindspot 10y-percentil
WHY_NOW_RATIO = {"RUBBER_BAND_STRETCHED": 15, "TENSION_BUILDING": 10, "NEUTRAL": 5}
WHY_NOW_COMPLEX = {"PÅ": 10, "SELEKTIV": 5, "AV": 0}                       # ember-komplexets utlåtande
WHY_NOW_TTM_STEPS = ((1.0, 10), (2.0, 7), (3.0, 4))                        # år ≤ → p, annars 1
WHY_NOW_TTM_BEYOND = 1
WHY_NOW_MIN_COVERAGE = 0.5             # under 50 % av signalerna: ingen uppskalning, flagga
WHY_NOW_BANDS = ((75, "NU"), (55, "SNART"), (35, "BEVAKA"), (0, "INTE NU"))

# ═══════════════════════════════════════════════════════════════════════════
# REGIONAL KNAPPHET 0–100 (SPEC listar; VAL). Jurisdiktion ur repots
# tabell (contrarian_alpha.resource_scoring); koncentration och västligt
# utbud ur råvaruregistret (null tills sourcat).
# ═══════════════════════════════════════════════════════════════════════════
REGIONAL_SUB = {"jurisdiction": 50, "concentration": 30, "western_gap": 20}
CONCENTRATION_STEPS = ((70.0, 30), (50.0, 22), (30.0, 12), (0.0, 5))      # största lands andel % ≥ → p
SAFE_JURISDICTION_MIN = 75.0           # under detta halveras koncentrationspoängen (tillgången är inte "västlig")
WESTERN_GAP_STEPS = ((10.0, 20), (25.0, 14), (50.0, 7))                    # västlig andel % ≤ → p, annars 2
WESTERN_GAP_BEYOND = 2
REGIONAL_BANDS = ((75, "KRITISK KNAPPHET"), (55, "KNAPP"), (35, "NORMAL"), (0, "RIKLIG"))
REGIONAL_MIN_COVERAGE = 0.6            # jurisdiktion ensam (50 %) skalas inte upp — säger inget om knapphet

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
    # ── Scenarier (kedjan pris → produktion → EBITDA → FCF → EV/NAV → aktie) ──
    FieldSpec("shares_outstanding_m", "Antal aktier (fullt utspätt)", "M", "number", _ALL, "scenarios"),
    FieldSpec("share_price", "Aktiekurs", "valuta/aktie", "number", _ALL, "scenarios",
              hint="Samma valuta som börsvärdet delat med aktier"),
    FieldSpec("tax_rate_pct", "Skattesats", "%", "number", _ALL, "scenarios",
              hint=f"Saknas → {DEFAULT_TAX_RATE_PCT:g} % (ASSUMPTION)"),
    FieldSpec("target_ev_ebitda", "EV/EBITDA i scenariot", "×", "number", _PROD, "scenarios",
              hint=f"Saknas → {DEFAULT_TARGET_EV_EBITDA:g}× (ASSUMPTION)"),
    FieldSpec("target_p_nav", "P/NAV vid omvärdering", "×", "number", _PRE, "scenarios",
              hint=f"Saknas → {DEFAULT_TARGET_P_NAV:g}× (ASSUMPTION)"),
    # ── Management (Case) ─────────────────────────────────────────────────
    FieldSpec("mgmt_track_record", "Byggmeriter", "p", "int", _ALL, "management", 1),
    FieldSpec("mgmt_capital_allocation", "Kapitalallokering", "p", "int", _ALL, "management", 1),
    FieldSpec("mgmt_dilution_history", "Utspädningshistorik", "p", "int", _ALL, "management", 1,
              hint="1 = disciplinerad"),
    FieldSpec("insider_ownership_pct", "Insynsägande", "%", "number", _ALL, "management"),
    FieldSpec("mgmt_alignment_delivery", "Alignment & leverans mot löften", "p", "int", _ALL,
              "management", 1),
    # ══ Durrett-motorn (engines/durrett) — pillar "durrett_*" ═══════════════
    # ── Aktiestruktur (SPEC §7) ───────────────────────────────────────────
    FieldSpec("basic_shares_m", "Aktier (basic)", "M", "number", _ALL, "durrett_shares"),
    FieldSpec("options_m", "Optioner", "M", "number", _ALL, "durrett_shares"),
    FieldSpec("warrants_m", "Warranter", "M", "number", _ALL, "durrett_shares"),
    FieldSpec("convertible_shares_m", "Konvertibler (aktier vid konvertering)", "M", "number", _ALL, "durrett_shares"),
    FieldSpec("other_dilutive_m", "Övriga utspädande instrument (RSU m.m.)", "M", "number", _ALL, "durrett_shares"),
    FieldSpec("shares_1y_ago_m", "Aktier för 1 år sedan", "M", "number", _ALL, "durrett_shares"),
    FieldSpec("shares_3y_ago_m", "Aktier för 3 år sedan", "M", "number", _ALL, "durrett_shares"),
    FieldSpec("shares_5y_ago_m", "Aktier för 5 år sedan", "M", "number", _ALL, "durrett_shares"),
    FieldSpec("atm_program", "ATM-program aktivt", "", "bool", _ALL, "durrett_shares"),
    FieldSpec("recent_financing_musd", "Senaste finansiering (12 mån)", "MUSD", "number", _ALL, "durrett_shares"),
    FieldSpec("expected_financing_musd", "Väntad finansiering (12 mån)", "MUSD", "number", _ALL, "durrett_shares"),
    FieldSpec("market_currency", "Valuta för kurs/börsvärde", "", "choice", _ALL, "durrett_shares",
              choices=("USD", "CAD", "AUD", "GBP", "EUR", "SEK", "NOK")),
    FieldSpec("fx_to_usd", "Växelkurs → USD", "USD per enhet", "number", _ALL, "durrett_shares",
              hint="1 CAD = 0,73 USD → 0.73. Krävs när valutan inte är USD; ingen dold FX."),
    # ── Resurser & reserver per kategori (SPEC §5) — i råvarans enhet ─────
    FieldSpec("reserve_proven", "Reserv — Proven", "enheter", "number", _ALL, "durrett_resources"),
    FieldSpec("reserve_probable", "Reserv — Probable", "enheter", "number", _ALL, "durrett_resources"),
    FieldSpec("resource_measured", "Resurs — Measured", "enheter", "number", _ALL, "durrett_resources"),
    FieldSpec("resource_indicated", "Resurs — Indicated", "enheter", "number", _ALL, "durrett_resources"),
    FieldSpec("resource_inferred", "Resurs — Inferred", "enheter", "number", _ALL, "durrett_resources"),
    FieldSpec("resource_unit", "Enhet för resurser/produktion", "", "choice", _ALL, "durrett_resources",
              choices=("oz", "koz", "Moz", "lb", "Mlb", "t", "kt", "Mt"),
              hint="Samma enhet för reserv, resurs och produktion — motorn räknar om till bas-enheten"),
    FieldSpec("ownership_pct", "Ägarandel i projektet", "%", "number", _ALL, "durrett_resources"),
    FieldSpec("recovery_pct", "Metallurgisk recovery", "%", "number", _ALL, "durrett_resources"),
    FieldSpec("grade", "Halt", "", "number", _ALL, "durrett_resources", hint="g/t för guld/silver, % för basmetaller"),
    FieldSpec("grade_unit", "Haltenhet", "", "choice", _ALL, "durrett_resources", choices=("g/t", "%", "ppm", "lb/t")),
    FieldSpec("strip_ratio", "Strip ratio", "×", "number", _ALL, "durrett_resources"),
    FieldSpec("mining_method", "Brytningsmetod", "", "text", _ALL, "durrett_resources"),
    FieldSpec("processing_method", "Processmetod", "", "text", _ALL, "durrett_resources"),
    FieldSpec("land_package_km2", "Landpaket", "km²", "number", _ALL, "durrett_resources"),
    # ── Produktion & kostnader (SPEC §9, §11) ─────────────────────────────
    FieldSpec("production_current", "Produktion nu", "enheter/år", "number", _ALL, "durrett_production",
              hint="Samma enhet som resource_unit. Producenter: senaste år."),
    FieldSpec("production_future", "Produktion i framtiden (plan)", "enheter/år", "number", _ALL, "durrett_production"),
    FieldSpec("production_future_year", "År för framtida produktion", "år", "int", _ALL, "durrett_production"),
    FieldSpec("production_3y_ago", "Produktion för 3 år sedan", "enheter/år", "number", _PROD, "durrett_production"),
    FieldSpec("resource_3y_ago", "Total resurs för 3 år sedan", "enheter", "number", _ALL, "durrett_production"),
    FieldSpec("reserve_3y_ago", "Total reserv för 3 år sedan", "enheter", "number", _ALL, "durrett_production"),
    FieldSpec("cash_cost", "Cash cost / C1", "USD/enhet", "number", _PROD, "durrett_production"),
    FieldSpec("sustaining_capex_musd", "Sustaining CapEx", "MUSD/år", "number", _PROD, "durrett_production"),
    FieldSpec("expansion_capex_musd", "Expansions-CapEx", "MUSD", "number", _ALL, "durrett_production"),
    FieldSpec("royalty_burden_pct", "Royaltybörda (NSR/GRR-summa)", "%", "number", _ALL, "durrett_production"),
    FieldSpec("streaming_burden_pct", "Streamingbörda (andel av produktionen)", "%", "number", _ALL, "durrett_production"),
    FieldSpec("royalty_revenue_share_pct", "Andel av intäkten från royalty/stream", "%", "number", _ALL,
              "durrett_production", hint="Klassificering: ≥ 40 % + egen drift → HYBRID"),
    FieldSpec("revenue_musd", "Omsättning (senaste 12 mån)", "MUSD", "number", _ALL, "durrett_production",
              hint="Klassificering: kommersiell produktion"),
    # ── Kassa & skuld (SPEC §12) ──────────────────────────────────────────
    FieldSpec("working_capital_musd", "Rörelsekapital", "MUSD", "number", _ALL, "durrett_balance"),
    FieldSpec("operating_cash_flow_musd", "Operativt kassaflöde (12 mån)", "MUSD", "number", _ALL, "durrett_balance"),
    FieldSpec("free_cash_flow_musd", "Fritt kassaflöde (12 mån)", "MUSD", "number", _ALL, "durrett_balance"),
    FieldSpec("interest_expense_musd", "Räntekostnad (12 mån)", "MUSD", "number", _ALL, "durrett_balance"),
    FieldSpec("debt_maturity_year", "Skuldens förfalloår", "år", "int", _ALL, "durrett_balance"),
    FieldSpec("next_milestone", "Nästa stora milstolpe", "", "text", _PRE, "durrett_balance"),
    FieldSpec("next_milestone_year", "År för nästa milstolpe", "år", "number", _PRE, "durrett_balance",
              hint="Decimal ok (2027.5). Funding cliff om finansiering krävs före den."),
    FieldSpec("milestone_cost_musd", "Kostnad fram till milstolpen", "MUSD", "number", _PRE, "durrett_balance"),
    # ── Management (SPEC §6) ──────────────────────────────────────────────
    FieldSpec("mgmt_track_verified", "Track record", "", "choice", _ALL, "durrett_management",
              choices=("verified", "claimed", "unknown"),
              hint="verified = källa som bekräftar (rapport, börsdata); claimed = bolagets egen presentation"),
    FieldSpec("ceo_mines_built", "Gruvor CEO byggt", "st", "int", _ALL, "durrett_management"),
    FieldSpec("team_mines_built", "Gruvor teamet byggt", "st", "int", _ALL, "durrett_management"),
    FieldSpec("team_mines_financed", "Gruvor teamet finansierat", "st", "int", _ALL, "durrett_management"),
    FieldSpec("team_mines_operated", "Gruvor teamet drivit", "st", "int", _ALL, "durrett_management"),
    FieldSpec("team_discoveries", "Fyndigheter teamet upptäckt", "st", "int", _ALL, "durrett_management"),
    FieldSpec("team_exits", "Tidigare exits (uppköp)", "st", "int", _ALL, "durrett_management"),
    FieldSpec("prior_shareholder_returns", "Tidigare aktieägaravkastning", "p", "int", _ALL, "durrett_management", 2,
              hint="0 förstört värde · 1 blandat · 2 skapat värde (med källa)"),
    FieldSpec("board_mining_experience", "Styrelsens gruverfarenhet", "p", "int", _ALL, "durrett_management", 2),
    FieldSpec("insider_buying_12m_musd", "Insiderköp (12 mån)", "MUSD", "number", _ALL, "durrett_management"),
    FieldSpec("insider_selling_12m_musd", "Insiderförsäljning (12 mån)", "MUSD", "number", _ALL, "durrett_management"),
    FieldSpec("mgmt_compensation_musd", "Ledningens ersättning (år)", "MUSD", "number", _ALL, "durrett_management"),
    FieldSpec("related_party_issues", "Närståendetransaktioner / problem", "", "bool", _ALL, "durrett_management"),
    # ── Jurisdiktion — projektrisk (SPEC §8), 0 hög risk · 1 måttlig · 2 låg ──
    FieldSpec("risk_permitting", "Tillståndsrisk", "p", "int", _ALL, "durrett_jurisdiction", 2),
    FieldSpec("risk_infrastructure", "Infrastruktur (väg, hamn)", "p", "int", _ALL, "durrett_jurisdiction", 2),
    FieldSpec("risk_power_water", "Kraft och vatten", "p", "int", _ALL, "durrett_jurisdiction", 2),
    FieldSpec("risk_community", "Lokalsamhälle / urfolk", "p", "int", _ALL, "durrett_jurisdiction", 2),
    FieldSpec("risk_security", "Säkerhet", "p", "int", _ALL, "durrett_jurisdiction", 2),
    FieldSpec("risk_environment", "Miljörestriktioner", "p", "int", _ALL, "durrett_jurisdiction", 2),
    FieldSpec("risk_nationalization", "Nationaliserings-/ägarrestriktioner", "p", "int", _ALL, "durrett_jurisdiction", 2),
    FieldSpec("risk_currency", "Valutarisk", "p", "int", _ALL, "durrett_jurisdiction", 2),
    FieldSpec("government_take_pct", "Skatt + royalty till staten", "%", "number", _ALL, "durrett_jurisdiction"),
    # ── Explorer (SPEC §17–19) ────────────────────────────────────────────
    FieldSpec("lassonde_stage", "Lassonde-position", "", "choice", _ALL, "durrett_explorer",
              choices=("exploration", "discovery", "resource_definition", "pea", "pfs", "fs",
                       "construction", "production", "expansion")),
    FieldSpec("drill_holes", "Antal borrhål", "st", "int", _ALL, "durrett_explorer"),
    FieldSpec("best_intercept_gram_m", "Bästa intercept (halt × meter)", "g/t·m eller %·m", "number", _ALL,
              "durrett_explorer"),
    FieldSpec("intercept_width_m", "Typisk mineraliserad bredd", "m", "number", _ALL, "durrett_explorer"),
    FieldSpec("continuity", "Kontinuitet", "p", "int", _ALL, "durrett_explorer", 2),
    FieldSpec("multiple_zones", "Flera zoner", "", "bool", _ALL, "durrett_explorer"),
    FieldSpec("step_out_success", "Step-out-framgång", "p", "int", _ALL, "durrett_explorer", 2),
    FieldSpec("historical_drilling", "Historisk borrning finns", "", "bool", _ALL, "durrett_explorer"),
    FieldSpec("geological_model", "Geologisk modell", "p", "int", _ALL, "durrett_explorer", 2,
              hint="0 oklar · 1 arbetshypotes · 2 väl förstådd"),
    FieldSpec("implied_value_per_unit", "Implicit värde per enhet i marken", "USD/enhet", "number", _ALL,
              "durrett_explorer", hint="Vad marknaden betalar för jämförbara uns/lb — ange källa"),
    # ── Momentum (SPEC §10) — hämtas i fliken, matas in som data ──────────
    FieldSpec("price_vs_ma200_pct", "Kurs mot MA200", "%", "number", _ALL, "durrett_momentum"),
    FieldSpec("momentum_6m_pct", "Kursutveckling 6 mån", "%", "number", _ALL, "durrett_momentum"),
    FieldSpec("rs_rank", "RS-rank (Börsdata 0–100)", "", "number", _ALL, "durrett_momentum"),
    FieldSpec("volume_trend", "Volymtrend", "p", "int", _ALL, "durrett_momentum", 2),
    FieldSpec("sector_momentum", "Sektormomentum", "p", "int", _ALL, "durrett_momentum", 2),
    FieldSpec("commodity_momentum", "Råvarumomentum", "p", "int", _ALL, "durrett_momentum", 2),
    FieldSpec("news_flow", "Nyhetsflöde", "p", "int", _ALL, "durrett_momentum", 2),
    # ── Confidence: Data Quality (bedömningar; källkvalitet/färskhet räknas) ──
    FieldSpec("independent_verification", "Oberoende verifiering", "p", "int", _ALL, "data_quality", 5,
              hint="0 bara bolagets ord · 3 nyckeltal bekräftade av QP/revisor · 5 alla nyckeltal oberoende"),
    FieldSpec("cross_source_consistency", "Källkonsistens", "p", "int", _ALL, "data_quality", 5,
              hint="0 motstridiga eller en enda källa · 3 huvudsak överens · 5 flera källor säger samma"),
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
