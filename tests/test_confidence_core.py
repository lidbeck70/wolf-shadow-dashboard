"""
confidence — grunden: config summerar rätt, proveniensen bär källa och
ålder, valideringen skiljer fel från saknat, råvaruregistret mappar repots
fyra vokabulärer och hittar aldrig på tal.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from confidence import config as cfg
from confidence import commodities as com
from confidence.data import provenance as prov
from confidence.data.models import CompanyInput, to_jsonable
from confidence.data import validation as val


# ── config ───────────────────────────────────────────────────────────────────
def test_pillars_and_confidence_parts_sum_to_100():
    assert sum(m for _k, _l, m in cfg.PILLARS) == 100
    assert sum(m for _k, _l, m in cfg.CONFIDENCE_PARTS) == 100
    assert sum(cfg.STRATEGIC_SUB.values()) == 20
    assert sum(cfg.RESOURCE_SUB.values()) == 15
    assert sum(cfg.ECON_PRODUCER_SUB.values()) == 15
    assert sum(cfg.ECON_DEVELOPER_SUB.values()) == 15
    assert sum(cfg.MANAGEMENT_SUB.values()) == 5
    assert cfg.SUPPLY_BALANCE_MAX_POINTS == 15 and cfg.TIME_TO_MONEY_TABLE[0] == (2.0, 10)
    assert cfg.P_NAV_TABLE[0] == (0.30, 10) and cfg.ND_EBITDA_TABLE[0] == (0.0, 10)
    assert [c[1] for c in cfg.KILL_CAPS] == [50, 60, 65, 70, 75]


def test_every_field_spec_is_consistent():
    keys = [f.key for f in cfg.FIELDS]
    assert len(keys) == len(set(keys))
    for f in cfg.FIELDS:
        assert f.kind in ("number", "int", "bool", "choice", "text", "date")
        if f.kind == "int" and f.key != "first_cashflow_year":      # ett årtal har inget max
            assert f.max is not None
        if f.kind == "choice":
            assert f.choices
        for s in f.stages:
            assert s in cfg.STAGES
    # varje Resource-delpoäng har ett fält med samma max
    for k, m in cfg.RESOURCE_SUB.items():
        assert cfg.FIELD_BY_KEY[k].max == m
    assert {f.key for f in cfg.fields_for("producer", "economics")} >= {"aisc", "roic_pct"}
    assert "npv_musd" not in {f.key for f in cfg.fields_for("producer")}


# ── proveniens ───────────────────────────────────────────────────────────────
def test_datapoint_carries_provenance_and_never_invents():
    p = prov.dp(180_000, kind="guidance", source="Årsredovisning", source_type="primary",
                pub_date="2026-03-15", unit="t")
    assert p.kind == "GUIDANCE" and prov.num(p) == 180_000.0
    assert prov.source_points(p) == 5
    assert prov.freshness_points(p, today=date(2026, 5, 1)) == 5
    assert prov.freshness_points(p, today=date(2026, 8, 1)) == 4       # 4,5 mån → 3–6
    assert prov.freshness_points(p, today=date(2026, 12, 1)) == 3      # 8,5 mån → 6–12
    assert prov.freshness_points(p, today=date(2027, 9, 1)) == 2       # 17,5 mån → 1–2 år
    assert prov.freshness_points(p, today=date(2029, 1, 1)) == 1
    assert prov.freshness_points(prov.dp(1)) == 0
    assert prov.num(prov.dp(None)) is None and prov.dp(None).missing
    assert prov.truth(prov.dp("ja")) is True and prov.truth(prov.dp("nej")) is False
    assert prov.truth(prov.dp("kanske")) is None
    with pytest.raises(ValueError):
        prov.dp(1, kind="gissning")
    assert "GUIDANCE" in prov.describe(p, "Kopparproduktion") and "180,000 t" in prov.describe(p)
    assert prov.describe(None, "AISC") == "AISC: DATA_MISSING"
    # rått värde utan proveniens blir ASSUMPTION — syns i Confidence
    assert prov.from_dict(42).kind == "ASSUMPTION"
    assert prov.from_dict({"value": 3, "kind": "actual", "source": "FS"}).kind == "ACTUAL"


# ── modell + validering ──────────────────────────────────────────────────────
def _dev():
    c = CompanyInput(ticker="KDK", name="Kodiak Copper", commodity="copper", country="CA",
                     jurisdiction="British Columbia", stage="developer", maturity="pea")
    c.set("npv_musd", prov.dp(650, kind="ACTUAL", source="PEA 2025", source_type="independent",
                              pub_date="2025-11-01", unit="MUSD"))
    c.set("capex_musd", prov.dp(400, kind="ACTUAL", source="PEA 2025", source_type="independent"))
    c.set("res_infrastructure", prov.dp(2, kind="ESTIMATE", source="Presentation"))
    return c


def test_company_roundtrip_and_accessors():
    c = _dev()
    d = c.as_dict()
    assert d["fields"]["npv_musd"]["source"] == "PEA 2025"
    back = CompanyInput.from_dict(d)
    assert back.num("npv_musd") == 650.0 and back.get("capex_musd").kind == "ACTUAL"
    assert back.has("npv_musd") and not back.has("irr_pct") and back.num("irr_pct") is None
    with pytest.raises(KeyError):
        c.set("magic_number", prov.dp(1))
    assert isinstance(to_jsonable(c), dict)


def test_validation_separates_errors_from_missing():
    c = _dev()
    issues = val.validate(c)
    assert not val.errors(issues)
    assert "irr_pct" in val.missing_keys(issues) and "cash_musd" in val.missing_keys(issues)
    assert "npv_musd" not in val.missing_keys(issues)

    c.set("res_infrastructure", prov.dp(7))                       # max 4
    c.set("irr_pct", prov.dp("hög"))                              # inte ett tal
    c.set("capex_musd", prov.dp(-5))                              # negativ
    c.set("resource_category", prov.dp("rykte"))                  # ogiltigt val
    c.set("permits_granted", prov.dp("kanske"))                   # inte ja/nej
    errs = {i.field for i in val.errors(val.validate(c))}
    assert errs == {"res_infrastructure", "irr_pct", "capex_musd", "resource_category", "permits_granted"}
    assert not val.usable(c, "irr_pct") and not val.usable(c, "capex_msud")
    assert val.usable(c, "npv_musd")
    # fält för fel stage läses inte
    c.set("aisc", prov.dp(1200))
    assert not val.usable(c, "aisc")
    assert any(i.level == val.WARN and i.field == "aisc" for i in val.validate(c))


# ── råvaruregister ───────────────────────────────────────────────────────────
def test_registry_crosswalks_the_four_vocabularies():
    assert com.resolve_key("Koppar") == "copper" == com.resolve_key("copper") == com.resolve_key("koppar")
    assert com.resolve_key("oil_gas") == "oil_gas" == com.resolve_key("Olja") == com.resolve_key("Oil & Gas")
    assert com.resolve_key("rare_earth") == "rare_earth" == com.resolve_key("sallsynta")
    assert com.resolve_key("naturgas") == "natural_gas" == com.resolve_key("gas")
    assert com.resolve_key("Gruv - Guld & Silver") == "gold"        # första träffen i namnet
    assert com.resolve_key("Betting") is None and com.resolve_key("") is None
    cw = com.crosswalk("copper")
    assert cw == {"rotation": "koppar", "theme": "koppar", "ratio_exposure": "copper",
                  "ember_complex": "basmetaller", "proxies": ["COPX", "CPER"]}
    # varje rotation- och tema-nyckel i repot finns i registret
    from rotation import COMMODITIES
    from blindspot.theme_board import _THEMES
    for c in COMMODITIES:
        if c.key != "royalty":
            assert com.resolve_key(c.key) is not None, c.key
    for t in _THEMES:
        assert com.resolve_key(t.key) is not None, t.key


def test_registry_seeds_only_significance_and_marks_it_estimate():
    u = com.get("uranium")
    assert u.strategic_significance.value == 9.5
    assert u.strategic_significance.kind == "ESTIMATE" and "repo-tabell" in u.strategic_significance.source
    assert u.demand_growth.missing and u.geopolitical_scarcity.missing and u.supply_balance_pct.missing
    # sourcad överlagring tar över, med proveniens
    ov = {"uranium": {"supply_balance_pct": {"value": 18.0, "kind": "ESTIMATE",
                                             "source": "WNA Nuclear Fuel Report 2025",
                                             "source_type": "independent", "pub_date": "2025-09-01"},
                      "demand_growth": {"value": 4, "kind": "ESTIMATE", "source": "WNA",
                                        "source_type": "independent"},
                      "adjustments": ["adj_secondary_supply"]}}
    u2 = com.get("uran", ov)
    assert u2.supply_balance_pct.value == 18.0 and u2.supply_balance_pct.source.startswith("WNA")
    assert u2.demand_growth.value == 4 and u2.adjustments == ("adj_secondary_supply",)
    assert com.get("uranium").supply_balance_pct.missing            # basregistret orört
