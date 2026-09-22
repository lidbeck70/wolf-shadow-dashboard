"""
Syntetiska testbolag för Confidence score-motorn (specens tio fall).
Talen är påhittade testdata — inte verkliga bolag. Alla fält får full
proveniens så Data Quality kan testas separat.
"""
from confidence.data.models import CompanyInput
from confidence.data.provenance import dp

COPPER_OVERRIDES = {"copper": {
    "supply_balance_pct": {"value": 12, "kind": "ESTIMATE", "source": "ICSG 2026", "source_type": "independent",
                           "pub_date": "2026-06-01"},
    "demand_growth": {"value": 4, "kind": "ESTIMATE", "source": "IEA CMO 2026", "source_type": "independent",
                      "pub_date": "2026-05-01"},
    "geopolitical_scarcity": {"value": 3, "kind": "ESTIMATE", "source": "USGS 2026", "source_type": "primary",
                              "pub_date": "2026-01-31"},
}}

_PRIMARY = dict(kind="ACTUAL", source="Årsredovisning 2025", source_type="primary", pub_date="2026-02-15")
_STUDY = dict(kind="ESTIMATE", source="DFS (Ausenco) 2025", source_type="independent", pub_date="2025-10-01")
_MKT = dict(kind="ACTUAL", source="Börsdata", source_type="secondary", pub_date="2026-09-01")


def fill(c: CompanyInput, values: dict, **meta) -> CompanyInput:
    for k, v in values.items():
        c.set(k, dp(v, **meta))
    return c


_RESOURCE_GOOD = dict(res_size=2, res_grade=2, res_metallurgy=1, mine_life_years=18, res_geology=2,
                      res_infrastructure=4, res_expansion=3)
_MGMT_GOOD = dict(mgmt_track_record=1, mgmt_capital_allocation=1, mgmt_dilution_history=1,
                  insider_ownership_pct=12, mgmt_alignment_delivery=1)
_CERTAINTY_GOOD = dict(resource_category="proven", resource_verified_share_pct=90, independent_resource_estimate=True,
                       drilling_density=2, resource_conversion_history=2, grade_consistency=2,
                       metallurgy_confidence=2, permits_granted=True, timeline_documented=True, historical_delays=2)


def producer_high_quality() -> CompanyInput:
    c = CompanyInput(ticker="PRD", name="Test Producer", commodity="copper", country="CA",
                     jurisdiction="Quebec", stage="producer", maturity="production")
    fill(c, dict(commodity_price=4.5, aisc=2.3, ebitda_margin_pct=48, fcf_yield_pct=11, roic_pct=18,
                 breakeven_price=2.7, net_debt_ebitda=-0.2, **_RESOURCE_GOOD, **_MGMT_GOOD), **_PRIMARY)
    fill(c, dict(ev_ebitda=3.8, pe=7.5, ev_ebit=5.5, market_cap_musd=5000, nav_musd=8000), **_MKT)
    fill(c, _CERTAINTY_GOOD, **_STUDY)
    return c


def producer_leveraged() -> CompanyInput:
    c = producer_high_quality()
    c.ticker, c.name = "LEV", "Test Leveraged Producer"
    fill(c, dict(net_debt_ebitda=4.5, fcf_yield_pct=1, aisc=3.9, breakeven_price=4.2), **_PRIMARY)
    return c


def developer_high_quality() -> CompanyInput:
    c = CompanyInput(ticker="DEV", name="Test Developer", commodity="copper", country="CA",
                     jurisdiction="British Columbia", stage="developer", maturity="dfs")
    fill(c, dict(npv_musd=1800, npv_discount_pct=8, npv_price_assumption=4.0, capex_musd=800, irr_pct=32,
                 payback_years=1.9, annual_production=180e6, production_unit="lb", breakeven_price=2.6,
                 npv_stress_price_musd=900, npv_stress_capex_musd=1550, irr_stress_price_pct=21,
                 first_cashflow_year=2029, **_RESOURCE_GOOD), **_STUDY)
    fill(c, dict(commodity_price=4.5, market_cap_musd=700, nav_musd=1800, cash_musd=250, quarterly_burn_musd=12,
                 committed_financing_musd=400, has_strategic_partner=True, has_offtake=True), **_MKT)
    fill(c, dict(**_MGMT_GOOD, **{**_CERTAINTY_GOOD, "resource_category": "probable"}, engineering_done=True,
                 infrastructure_secured=True, financing_committed=True, offtake_signed=True,
                 construction_contracts=False, procurement_started=True), **_PRIMARY)
    return c


def developer_overvalued() -> CompanyInput:
    c = developer_high_quality()
    c.ticker, c.name = "OVR", "Test Overvalued Developer"
    fill(c, dict(market_cap_musd=3000), **_MKT)                       # P/NAV 1,67
    return c


def developer_cheap_but_poor() -> CompanyInput:
    c = CompanyInput(ticker="CHP", name="Test Cheap Poor Project", commodity="copper", country="AR",
                     jurisdiction="Salta", stage="developer", maturity="pea")
    fill(c, dict(npv_musd=200, capex_musd=900, irr_pct=9, payback_years=6, annual_production=60e6,
                 production_unit="lb", breakeven_price=4.3, first_cashflow_year=2034, res_size=1, res_grade=0,
                 res_metallurgy=0, mine_life_years=8, res_geology=0, res_infrastructure=0, res_expansion=1),
         **_STUDY)
    fill(c, dict(commodity_price=4.5, market_cap_musd=40, nav_musd=200, cash_musd=5, quarterly_burn_musd=3,
                 has_strategic_partner=False, has_offtake=False), **_MKT)
    fill(c, dict(mgmt_track_record=0, mgmt_capital_allocation=0, mgmt_dilution_history=0,
                 insider_ownership_pct=1, mgmt_alignment_delivery=0, dilution_score=8,
                 resource_category="inferred", independent_resource_estimate=True, drilling_density=0,
                 resource_conversion_history=0, grade_consistency=1, metallurgy_confidence=0,
                 permits_granted=False, historical_delays=0, timeline_documented=False), **_PRIMARY)
    return c


def developer_price_stressed() -> CompanyInput:
    c = developer_high_quality()
    c.ticker, c.name = "PST", "Test Price-Stressed Developer"
    fill(c, dict(npv_stress_price_musd=-50, irr_stress_price_pct=4), **_STUDY)
    return c


def developer_capex_stressed_modelled() -> CompanyInput:
    """Ingen FS-känslighet → modellerad bound: NPV 150 − 0,2 × 900 < 0."""
    c = developer_high_quality()
    c.ticker, c.name = "CST", "Test Capex-Stressed Developer"
    fill(c, dict(npv_musd=150, capex_musd=900), **_STUDY)
    for k in ("npv_stress_price_musd", "npv_stress_capex_musd", "irr_stress_price_pct"):
        c.fields.pop(k, None)
    return c


def explorer_early() -> CompanyInput:
    c = CompanyInput(ticker="EXP", name="Test Early Explorer", commodity="lithium", country="CA",
                     jurisdiction="Ontario", stage="explorer", maturity="exploration")
    fill(c, dict(cash_musd=18, quarterly_burn_musd=2.5, market_cap_musd=60, discovery_option=4,
                 res_size=1, res_grade=1, res_geology=1, res_infrastructure=2, res_expansion=3,
                 has_strategic_partner=False, has_offtake=False, resource_category="exploration_target",
                 independent_resource_estimate=False), **_MKT)
    fill(c, dict(mgmt_track_record=1, mgmt_capital_allocation=1, mgmt_dilution_history=1,
                 insider_ownership_pct=15, mgmt_alignment_delivery=1), **_PRIMARY)
    return c


def developer_excellent_low_confidence() -> CompanyInput:
    """Samma tal som DEV men allt är bolagets egna antaganden utan datum,
    ingen oberoende resurs, ingen finansiering."""
    c = developer_high_quality()
    c.ticker, c.name = "LOC", "Test Excellent But Unverified"
    for k, p in list(c.fields.items()):
        c.fields[k] = dp(p.value, kind="ASSUMPTION", source="Bolagspresentation", source_type="weak")
    fill(c, dict(independent_resource_estimate=False, resource_category="inferred", resource_verified_share_pct=0,
                 financing_committed=False, no_financing_plan=True, drilling_density=0,
                 resource_conversion_history=0), kind="ASSUMPTION", source="Bolagspresentation", source_type="weak")
    c.fields.pop("committed_financing_musd", None)
    return c


def missing_everything() -> CompanyInput:
    return CompanyInput(ticker="NUL", name="Test Missing Data", commodity="copper", stage="developer",
                        maturity="pea")


def royalty() -> CompanyInput:
    c = CompanyInput(ticker="ROY", name="Test Royalty", commodity="gold", stage="royalty", maturity="production")
    fill(c, dict(ebitda_margin_pct=78, fcf_yield_pct=4, roic_pct=12, net_debt_ebitda=0.8, ev_ebitda=15, pe=25,
                 ev_ebit=18, market_cap_musd=9000, nav_musd=7000, **_MGMT_GOOD), **_PRIMARY)
    return c


ALL = {
    "producer_high_quality": producer_high_quality, "producer_leveraged": producer_leveraged,
    "developer_high_quality": developer_high_quality, "developer_overvalued": developer_overvalued,
    "developer_cheap_but_poor": developer_cheap_but_poor, "developer_price_stressed": developer_price_stressed,
    "developer_capex_stressed_modelled": developer_capex_stressed_modelled, "explorer_early": explorer_early,
    "developer_excellent_low_confidence": developer_excellent_low_confidence,
    "missing_everything": missing_everything, "royalty": royalty,
}
