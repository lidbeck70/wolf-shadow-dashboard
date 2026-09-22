"""
confidence/data/validation.py — indata granskas innan något räknas.

Fel (error) stoppar beräkningen av det fältet: fel typ, utanför intervall,
okänt val. Saknat (missing) är inte ett fel — det är DATA_MISSING och
hanteras av scoringen med noll poäng och lägre Confidence. Validatorn
returnerar en lista av Issue så UI:t kan visa dem vid fältet.
"""

from __future__ import annotations

from dataclasses import dataclass

from confidence.config import FIELD_BY_KEY, KINDS, MATURITY, STAGES, fields_for
from confidence.data.models import CompanyInput
from confidence.data.provenance import num, truth

ERROR, WARN, MISSING = "error", "warn", "missing"

# Fält som får vara negativa (allt annat ≥ 0)
_MAY_BE_NEGATIVE = {"net_debt_ebitda", "npv_musd", "npv_stress_price_musd", "npv_stress_capex_musd",
                    "irr_stress_price_pct", "roic_pct", "fcf_yield_pct", "ebitda_margin_pct"}

# Fält utan vilka en stage inte kan få poäng i sin pelare (rapporteras som MISSING)
_CORE_BY_STAGE = {
    "producer": ("commodity_price", "aisc", "ebitda_margin_pct", "fcf_yield_pct", "roic_pct",
                 "net_debt_ebitda", "ev_ebitda", "market_cap_musd"),
    "royalty": ("ebitda_margin_pct", "fcf_yield_pct", "net_debt_ebitda", "ev_ebitda", "market_cap_musd"),
    "developer": ("npv_musd", "capex_musd", "irr_pct", "commodity_price", "breakeven_price",
                  "cash_musd", "quarterly_burn_musd", "market_cap_musd", "nav_musd",
                  "first_cashflow_year", "resource_category"),
    "explorer": ("cash_musd", "quarterly_burn_musd", "market_cap_musd", "resource_category"),
}


@dataclass(frozen=True)
class Issue:
    level: str        # error | warn | missing
    field: str
    message: str


def validate(company: CompanyInput) -> list:
    issues: list = []
    if company.stage not in STAGES:
        issues.append(Issue(ERROR, "stage", f"okänt stage {company.stage!r} — giltiga: {STAGES}"))
    if company.maturity not in MATURITY:
        issues.append(Issue(ERROR, "maturity", f"okänd mognad {company.maturity!r} — giltiga: {MATURITY}"))
    if not company.ticker:
        issues.append(Issue(ERROR, "ticker", "ticker saknas"))
    if not company.commodity:
        issues.append(Issue(MISSING, "commodity", "råvara saknas — Strategic Commodity och Demand & Scarcity ger 0"))

    for key, raw in company.fields.items():
        spec = FIELD_BY_KEY.get(key)
        if spec is None:
            issues.append(Issue(WARN, key, "okänt fält — ignoreras"))
            continue
        p = company.get(key)
        if p is None or p.missing:
            continue
        if p.kind not in KINDS:
            issues.append(Issue(ERROR, key, f"okänd datatyp {p.kind!r}"))
        if spec.kind in ("number", "int"):
            v = num(p)
            if v is None:
                issues.append(Issue(ERROR, key, f"{spec.label}: inte ett tal ({p.value!r})"))
                continue
            if v < 0 and key not in _MAY_BE_NEGATIVE:
                issues.append(Issue(ERROR, key, f"{spec.label}: får inte vara negativt ({v:g})"))
            if spec.kind == "int":
                if abs(v - round(v)) > 1e-9:
                    issues.append(Issue(ERROR, key, f"{spec.label}: heltal krävs ({v:g})"))
                if spec.max is not None and v > spec.max:
                    issues.append(Issue(ERROR, key, f"{spec.label}: max {spec.max:g} ({v:g})"))
            if key.endswith("_pct") and v > 1000:
                issues.append(Issue(WARN, key, f"{spec.label}: {v:g} % — fel enhet?"))
        elif spec.kind == "bool":
            if truth(p) is None:
                issues.append(Issue(ERROR, key, f"{spec.label}: ja/nej krävs ({p.value!r})"))
        elif spec.kind == "choice":
            if str(p.value) not in spec.choices:
                issues.append(Issue(ERROR, key, f"{spec.label}: ogiltigt val {p.value!r} — {spec.choices}"))
        if spec.stages and company.stage not in spec.stages:
            issues.append(Issue(WARN, key, f"{spec.label} gäller inte stage {company.stage} — ignoreras"))
        if p.kind == "ASSUMPTION" and not p.note:
            issues.append(Issue(WARN, key, f"{spec.label}: ASSUMPTION utan motivering"))

    for key in _CORE_BY_STAGE.get(company.stage, ()):
        if not company.has(key):
            label = FIELD_BY_KEY[key].label if key in FIELD_BY_KEY else key
            issues.append(Issue(MISSING, key, f"{label} saknas (DATA_MISSING)"))
    return issues


def errors(issues: list) -> list:
    return [i for i in issues if i.level == ERROR]


def missing_keys(issues: list) -> list:
    return [i.field for i in issues if i.level == MISSING]


def usable(company: CompanyInput, key: str) -> bool:
    """Fältet finns, är giltigt och gäller stage — det scoringen får läsa."""
    spec = FIELD_BY_KEY.get(key)
    if spec is None or not company.has(key):
        return False
    if spec.stages and company.stage not in spec.stages:
        return False
    p = company.get(key)
    if spec.kind in ("number", "int"):
        v = num(p)
        if v is None or (v < 0 and key not in _MAY_BE_NEGATIVE):
            return False
        if spec.kind == "int" and spec.max is not None and v > spec.max:
            return False
        return True
    if spec.kind == "bool":
        return truth(p) is not None
    if spec.kind == "choice":
        return str(p.value) in spec.choices
    return True


def stage_fields(stage: str) -> list:
    return fields_for(stage)
