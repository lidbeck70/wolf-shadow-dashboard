"""
asymmetry/fields.py — de fält de tre poängen faktiskt läser. Inget annat
visas i Wolf Asymmetrys ark.

  🚀 Commodity Leverage och 🛡️ Margin of Safety läser ekonomiblocket och
     Börsdata-blocket (asymmetry/model.py, confidence/scenarios/engine.py).
  🎯 Confidence läser Confidence-blocket (confidence/scoring/confidence.py):
     kärnan är de tungt vägda bedömningarna, finliret de lätta.

Nycklarna är confidence.config.FIELDS. Ordningen är visningsordningen.
"""

from __future__ import annotations

from typing import Optional

from confidence import config as cfg

# Börsdata fyller (sifferuppdateringen) — visas, kan ändras, matas sällan in
AUTO: tuple = ("market_cap_musd", "share_price", "basic_shares_m", "cash_musd", "debt_musd")

_STUDY = ("commodity_price", "npv_musd", "npv_price_assumption", "capex_musd", "breakeven_price",
          "annual_production", "production_unit", "irr_pct", "npv_stress_price_musd", "npv_stress_capex_musd")
ECONOMY: dict = {
    "producer":  ("commodity_price", "production_current", "production_unit", "aisc"),
    "royalty":   ("commodity_price", "production_current", "production_unit", "ebitda_margin_pct"),
    "developer": _STUDY,
    "explorer":  _STUDY,
}

# Confidence — kärnan (tungt vägda: resurs 20, datakvalitet 20, tidplan 10, ledning 10, kill-caps)
_CORE_ALL = ("resource_category", "independent_resource_estimate", "independent_verification",
             "cross_source_consistency", "historical_delays", "timeline_documented",
             "mgmt_track_record", "mgmt_dilution_history", "record_price_dependent", "capital_destruction_history")
CONFIDENCE_CORE: dict = {
    "producer":  _CORE_ALL + ("net_debt_ebitda",),
    "royalty":   _CORE_ALL + ("net_debt_ebitda",),
    "developer": _CORE_ALL + ("quarterly_burn_musd", "committed_financing_musd", "financing_committed",
                              "permits_granted", "no_financing_plan"),
    "explorer":  _CORE_ALL + ("quarterly_burn_musd", "no_financing_plan"),
}
# Confidence — finliret (lätt vägda delpoäng)
_MORE_ALL = ("resource_verified_share_pct", "drilling_density", "grade_consistency", "metallurgy_confidence",
             "resource_conversion_history", "mgmt_alignment_delivery", "mgmt_capital_allocation",
             "single_fragile_parameter")
CONFIDENCE_MORE: dict = {
    "producer":  _MORE_ALL,
    "royalty":   _MORE_ALL,
    "developer": _MORE_ALL + ("engineering_done", "infrastructure_secured", "offtake_signed"),
    "explorer":  _MORE_ALL + ("engineering_done", "infrastructure_secured", "offtake_signed"),
}

# bolagets råvarunyckel → terminsnamnet i commodity_prices.TICKERS
FUTURES_NAME: dict = {"gold": "guld", "silver": "silver", "platinum": "platina", "palladium": "palladium",
                      "copper": "koppar", "oil_gas": "olja", "natural_gas": "gas"}


def _stage(stage: str) -> str:
    return stage if stage in ECONOMY else "developer"


def economy(stage: str) -> list:
    return [cfg.FIELD_BY_KEY[k] for k in ECONOMY[_stage(stage)]]


def confidence_core(stage: str) -> list:
    return [cfg.FIELD_BY_KEY[k] for k in CONFIDENCE_CORE[_stage(stage)]]


def confidence_more(stage: str) -> list:
    return [cfg.FIELD_BY_KEY[k] for k in CONFIDENCE_MORE[_stage(stage)]]


def auto() -> list:
    return [cfg.FIELD_BY_KEY[k] for k in AUTO]


def all_keys(stage: str) -> list:
    return list(ECONOMY[_stage(stage)]) + list(AUTO) + list(CONFIDENCE_CORE[_stage(stage)]) \
        + list(CONFIDENCE_MORE[_stage(stage)])


def futures_name(commodity: str) -> Optional[str]:
    return FUTURES_NAME.get(str(commodity or "").strip().lower())
