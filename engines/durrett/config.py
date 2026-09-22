"""
engines/durrett/config.py — DURRETT_CONFIG: ALLA vikter, multiplar,
red-flag-trösklar och scenario-defaults. Ingen tröskel i affärslogiken.

SPEC = ur masterprompten. VAL = eget val där prompten listar posten utan
tal; kommentaren säger hur valet gjordes. Metallpriser per scenario är
VAL-platshållare — de ska ses och ändras i fliken, aldrig dolt.
"""

from __future__ import annotations

# ── Bolagstyper (SPEC §4) ────────────────────────────────────────────────────
PRODUCER, DEVELOPER, EXPLORER, ROYALTY, HYBRID, UNKNOWN = (
    "PRODUCER", "DEVELOPER", "EXPLORER", "ROYALTY_STREAMER", "HYBRID", "UNKNOWN")
COMPANY_TYPES = (PRODUCER, DEVELOPER, EXPLORER, ROYALTY, HYBRID, UNKNOWN)

# ── Lassonde-kurvan (SPEC §19) ───────────────────────────────────────────────
LASSONDE_STAGES = ("exploration", "discovery", "resource_definition", "pea", "pfs", "fs",
                   "construction", "production", "expansion")
LASSONDE_LABEL = {"exploration": "Exploration", "discovery": "Discovery", "resource_definition": "Resource Definition",
                  "pea": "PEA", "pfs": "PFS", "fs": "FS", "construction": "Construction",
                  "production": "Production", "expansion": "Expansion"}
# confidence.config.MATURITY → Lassonde (bakåtkompatibelt med Confidence score)
MATURITY_TO_LASSONDE = {"exploration": "exploration", "pea": "pea", "pfs": "pfs", "dfs": "fs",
                        "fid": "fs", "construction": "construction", "production": "production"}

DURRETT_CONFIG: dict = {
    # ── Quality Score-vikter (SPEC §25, summa 100). Momentum ingår INTE. ──
    "weights": {
        "properties": 15, "management": 10, "dilution": 10, "jurisdiction": 10, "growth": 10,
        "costs": 10, "financing": 8, "balance_sheet": 8, "valuation": 12, "upside": 7,
    },
    # Risk Score (100 = lägst risk) väger delpoängen som mäter risk (VAL)
    "risk_weights": {"dilution": 20, "jurisdiction": 20, "financing": 20, "balance_sheet": 20, "costs": 20},
    "risk_levels": ((80, "LOW"), (60, "MODERATE"), (40, "ELEVATED"), (0, "HIGH")),   # Risk Score ≥ → nivå

    # ── Värderingsmultiplar (SPEC §13) ────────────────────────────────────
    "valuation_multiples": {"default": 5.0, "large_low_risk": 10.0},
    # Framtida vinst-multipel (Durretts köpregel, scoring.py) för upside
    "mcap_future_earnings_buy_max": 10.0,

    # ── Scenario-defaults (SPEC §29–30). Priser i råvarans enhet. ─────────
    # VAL-platshållare — visas alltid som ASSUMPTION och kan ändras i fliken.
    "scenario_defaults": {
        "price_change_pct": {"bear": -20.0, "base": 0.0, "bull": 25.0},   # när ingen pristabell finns
        "capex_change_pct": {"bear": 20.0, "base": 0.0, "bull": 0.0},
        "aisc_change_pct": {"bear": 10.0, "base": 0.0, "bull": -5.0},
        "production_change_pct": {"bear": -10.0, "base": 0.0, "bull": 10.0},
        "recovery_change_pct": {"bear": -5.0, "base": 0.0, "bull": 0.0},
        "mine_life_change_pct": {"bear": -10.0, "base": 0.0, "bull": 10.0},
        "multiple": {"bear": 4.0, "base": 5.0, "bull": 7.0},
        "fx_change_pct": {"bear": 0.0, "base": 0.0, "bull": 0.0},
        "commodity_prices": {                       # USD, ASSUMPTION — TODO: sätt egna
            "gold": {"bear": 2500.0, "base": 3000.0, "bull": 3500.0, "unit": "USD/oz"},
            "silver": {"bear": 28.0, "base": 35.0, "bull": 45.0, "unit": "USD/oz"},
            "copper": {"bear": 3.6, "base": 4.5, "bull": 5.5, "unit": "USD/lb"},
            "uranium": {"bear": 60.0, "base": 80.0, "bull": 110.0, "unit": "USD/lb U3O8"},
        },
    },
    "scenario_probabilities": {"bear": 25.0, "base": 50.0, "bull": 25.0},
    "default_tax_rate_pct": 25.0,                   # ASSUMPTION när tax_rate_pct saknas

    # ── Red flag-trösklar (SPEC §20; talen VAL) ───────────────────────────
    "red_flag_thresholds": {
        "aisc_to_price_high": 0.85,                 # AISC/pris ≥ → High AISC (HIGH); ≥ 0.70 MEDIUM
        "aisc_to_price_medium": 0.70,
        "irr_low_pct": 15.0,                        # IRR < → Low IRR (HIGH < 10, annars MEDIUM)
        "irr_critical_pct": 10.0,
        "npv_capex_low": 1.0,                       # NPV/CAPEX < 1 HIGH, 1–2 MEDIUM (SPEC §14)
        "npv_capex_ok": 2.0,
        "capex_to_mcap_huge": 3.0,                  # CapEx ≥ 3 × börsvärde → Huge CAPEX (HIGH); ≥ 1,5 MEDIUM
        "capex_to_mcap_large": 1.5,
        "net_debt_ebitda_high": 3.0,                # Weak balance sheet
        "net_debt_ebitda_medium": 2.0,
        "runway_short_years": 1.0,                  # Short cash runway (HIGH); < 2 MEDIUM
        "runway_medium_years": 2.0,
        "dilution_1y_high_pct": 20.0,               # High dilution
        "dilution_3y_high_pct": 50.0,
        "serial_diluter_3y_pct": 75.0,              # Serial diluter (CRITICAL) — eller ≥ 3 år i rad > 15 %
        "serial_diluter_yearly_pct": 15.0,
        "insider_ownership_low_pct": 3.0,           # Low ownership
        "strip_ratio_high": 8.0,                    # High strip ratio (öppet dagbrott)
        "royalty_burden_high_pct": 5.0,             # High royalties (NSR/GRR-summa)
        "streaming_burden_high_pct": 15.0,          # High streaming burden (andel av produktionen)
        "path_to_production_long_years": 5.0,       # Long path to production
        "npv_price_above_spot_pct": 10.0,           # Unrealistic assumptions: NPV-pris > spot +10 %
        "price_sensitivity_high": 0.5,              # NPV-fall > 50 % vid −20 % pris
        "grade_low_by_unit": {"USD/oz": 0.8, "USD/lb": 0.4, "USD/t": 0.0},   # g/t Au, % Cu — TODO per råvara
        "recovery_low_pct": 75.0,
    },

    # ── Delpoängens trösklar (VAL — varje rad förklaras i notes) ──────────
    "score_steps": {
        "npv_capex": ((3.0, 100), (2.0, 80), (1.5, 65), (1.0, 50), (0.5, 30), (0.0, 10)),     # ≥ → p
        "irr_pct": ((40.0, 100), (30.0, 85), (20.0, 70), (15.0, 55), (10.0, 35), (0.0, 10)),
        "aisc_margin_pct": ((50.0, 100), (40.0, 85), (30.0, 70), (20.0, 55), (10.0, 35), (0.0, 15)),
        "mine_life_years": ((15.0, 100), (10.0, 80), (7.0, 60), (5.0, 40), (0.0, 20)),
        "recovery_pct": ((92.0, 100), (85.0, 80), (75.0, 60), (0.0, 30)),
        "runway_years": ((3.0, 100), (2.0, 80), (1.5, 60), (1.0, 40), (0.5, 20), (0.0, 5)),
        "net_debt_ebitda": ((0.0, 100), (0.5, 90), (1.0, 80), (2.0, 60), (3.0, 40), (4.0, 20)),  # < → p, annars 5
        "dilution_3y_pct": ((0.0, 100), (10.0, 90), (25.0, 75), (50.0, 50), (75.0, 30), (100.0, 15)),  # ≤ → p, annars 5
        "insider_ownership_pct": ((20.0, 100), (10.0, 85), (5.0, 65), (3.0, 45), (1.0, 25), (0.0, 10)),
        "growth_multiple": ((3.0, 100), (2.0, 85), (1.5, 70), (1.2, 55), (1.0, 40), (0.0, 20)),
        "production_cagr_pct": ((25.0, 100), (15.0, 85), (8.0, 70), (3.0, 55), (0.0, 40)),
        "resource_growth_pct": ((50.0, 100), (25.0, 80), (10.0, 65), (0.0, 50)),
        "upside_multiple": ((10.0, 100), (5.0, 85), (3.0, 70), (2.0, 55), (1.5, 40), (1.0, 25), (0.0, 5)),
        "ev_per_oz_usd": ((25.0, 100), (50.0, 85), (100.0, 70), (150.0, 55), (250.0, 40), (400.0, 20)),  # ≤ → p (guld)
        "ev_npv": ((0.3, 100), (0.5, 85), (0.7, 70), (0.9, 55), (1.1, 40), (1.5, 20)),               # ≤ → p
        "mcap_future_earnings": ((3.0, 100), (5.0, 85), (7.0, 70), (10.0, 55), (15.0, 35), (25.0, 15)),
        "capex_to_mcap": ((0.5, 100), (1.0, 80), (1.5, 60), (3.0, 40), (5.0, 20)),                   # ≤ → p
        "funding_coverage": ((1.0, 100), (0.75, 80), (0.5, 60), (0.25, 40), (0.0, 20)),               # (kassa+åtagen)/capex ≥
        "momentum_pct": ((30.0, 100), (15.0, 80), (5.0, 65), (0.0, 50), (-10.0, 35), (-25.0, 20)),
    },
    "score_beyond": {"net_debt_ebitda": 5, "dilution_3y_pct": 5, "ev_per_oz_usd": 10, "ev_npv": 10,
                     "mcap_future_earnings": 5, "capex_to_mcap": 10, "momentum_pct": 10},

    # Delvikter inom varje område (VAL). Nycklarna är komponentnamn i notes.
    "sub_weights": {
        "properties": {"npv_capex": 25, "irr": 15, "mine_life": 10, "recovery": 10, "grade": 10,
                       "infrastructure": 10, "resource_size": 10, "exploration_upside": 10},
        "management": {"track_record": 40, "insider_ownership": 25, "insider_activity": 15,
                       "board": 10, "alignment": 10},
        "jurisdiction": {"country": 40, "project": 60},
        "growth": {"production": 40, "resource": 30, "reserve": 15, "pipeline": 15},
        "costs": {"margin": 50, "cost_position": 30, "sustaining": 20},
        "financing": {"coverage": 50, "requirement": 30, "partners": 20},
        "balance_sheet": {"net_debt": 40, "runway": 40, "liquidity": 20},
        "valuation": {"resource_value": 25, "reserve_value": 20, "ev_npv": 25, "future_earnings": 30},
        "momentum": {"price_trend": 35, "relative_strength": 25, "commodity": 20, "sector": 10, "news": 10},
    },

    # Developer-checklistan (SPEC §16) — sex kriterier, trösklar VAL
    "developer_checklist": {
        "strong_project_npv_capex_min": 1.5, "strong_project_irr_min": 20.0,
        "high_upside_multiple_min": 3.0, "good_location_min_score": 65.0,
        "strong_management_min_score": 65.0, "path_to_production_max_years": 5.0,
        "strong_insiders_min_pct": 5.0,
    },
    # Explorer (SPEC §17–19)
    "explorer": {
        "optionality_ev_per_unit_cheap": {"USD/oz": 20.0, "USD/lb": 0.02, "USD/t": 50.0},  # EV/enhet i marken ≤ → 100
        "optionality_ev_per_unit_expensive": {"USD/oz": 150.0, "USD/lb": 0.10, "USD/t": 400.0},
        "discovery_play_min_intercept": 50.0,        # gram×meter (eller %×m) för "discovery play"
    },
    # Klassificering (SPEC §4) — trösklar VAL
    "classification": {
        "producer_min_revenue_musd": 5.0,            # kommersiell produktion
        "hybrid_royalty_share_pct": 40.0,            # royaltyintäkt ≥ 40 % men även gruvdrift → HYBRID
        "developer_min_stage": "pea",                # PEA eller senare = developer
    },
    # Confidence (SPEC §28): Model Confidence = data confidence viktad med scenariorobusthet
    "confidence": {"data_weight": 0.7, "model_weight": 0.3, "robustness_bear_floor_pct": -50.0},
    # Momentum-viktens tak i Quality (SPEC §10: dominerar aldrig) — 0 = utanför
    "momentum_in_quality": 0,
}

# ── Valutor (SPEC §44). Intern modellvaluta USD. Kurserna är None tills du
#    anger dem (fältet fx_to_usd per bolag, med datum) — ingen dold FX. ──
CURRENCIES = ("USD", "CAD", "AUD", "GBP", "EUR", "SEK", "NOK")

# ── Metallnormalisering (SPEC §45) — enhet per råvara ur confidence.commodities;
#    här bara vilka kostnadsmått som gäller. ──
COST_METRIC_BY_UNIT = {"USD/oz": "AISC", "USD/lb": "C1/AISC", "USD/t": "AISC/opex", "USD/t LCE": "opex",
                       "USD/lb U3O8": "AISC", "USD/boe": "opex"}

# ── Red flag-allvar ─────────────────────────────────────────────────────────
LOW, MEDIUM, HIGH, CRITICAL = "LOW", "MEDIUM", "HIGH", "CRITICAL"
SEVERITY_RANK = {CRITICAL: 3, HIGH: 2, MEDIUM: 1, LOW: 0}
# Red Flag Score 0–100: 100 − avdrag per flagga (VAL), golv 0
RED_FLAG_PENALTY = {CRITICAL: 30, HIGH: 15, MEDIUM: 7, LOW: 3}

# Katalysator-typer (SPEC §36)
CATALYST_TYPES = ("drill_results", "resource_update", "pea", "pfs", "fs", "permit", "financing",
                  "construction", "production", "expansion", "m_and_a", "strategic_investment",
                  "offtake", "debt_reduction", "commodity_price")
CATALYST_IMPORTANCE = ("LOW", "MEDIUM", "HIGH")
