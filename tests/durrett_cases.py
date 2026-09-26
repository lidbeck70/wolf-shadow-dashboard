"""
Syntetiska testbolag för Durrett-motorn. Påhittade tal — inte verkliga
bolag. Alla fält får proveniens så Data Confidence kan räknas.
"""
from confidence.data.models import CompanyInput
from confidence.data.provenance import dp

_FS = dict(kind="ACTUAL", source="Årsredovisning 2025", source_type="primary", pub_date="2026-02-15")
_TR = dict(kind="ESTIMATE", source="NI 43-101 FS (Ausenco) 2025", source_type="independent", pub_date="2025-10-01")
_MKT = dict(kind="ACTUAL", source="Börsdata", source_type="secondary", pub_date="2026-09-01")
_PRES = dict(kind="ESTIMATE", source="Bolagspresentation 2026-08", source_type="weak", pub_date="2026-08-15")


def fill(c, values, **meta):
    for k, v in values.items():
        c.set(k, dp(v, **meta))
    return c


def gold_producer() -> CompanyInput:
    """Mid-tier guldproducent, 200 koz, AISC 1 450, nettokassa, disciplinerad."""
    c = CompanyInput(ticker="GPR", name="Test Gold Producer", commodity="gold", country="CA", jurisdiction="Ontario",
                     exchange="TSX", stage="producer", maturity="production")
    fill(c, dict(revenue_musd=600, production_current=200_000, production_future=300_000, production_future_year=2029,
                 production_3y_ago=150_000, resource_unit="oz", aisc=1450, cash_cost=1050, sustaining_capex_musd=40,
                 cash_musd=120, debt_musd=40, net_debt_ebitda=-0.3, free_cash_flow_musd=110, operating_cash_flow_musd=180,
                 working_capital_musd=90, interest_expense_musd=3, basic_shares_m=250, options_m=8, warrants_m=2,
                 shares_1y_ago_m=245, shares_3y_ago_m=230, shares_5y_ago_m=200, insider_ownership_pct=8,
                 insider_buying_12m_musd=1.2, insider_selling_12m_musd=0.2, mgmt_compensation_musd=6,
                 related_party_issues=False, mgmt_dilution_history=1, mgmt_track_verified="verified",
                 ceo_mines_built=2, team_mines_built=3, team_mines_operated=4, team_discoveries=1,
                 prior_shareholder_returns=2, board_mining_experience=2, reserve_proven=1_200_000,
                 reserve_probable=1_800_000, resource_measured=800_000, resource_indicated=1_500_000,
                 resource_inferred=900_000, resource_3y_ago=2_600_000, reserve_3y_ago=2_400_000, ownership_pct=100,
                 recovery_pct=91, grade=2.1, grade_unit="g/t", strip_ratio=4, mine_life_years=14, royalty_burden_pct=2,
                 streaming_burden_pct=0, res_size=1, res_infrastructure=4, res_expansion=2, risk_permitting=2,
                 risk_infrastructure=2, risk_power_water=2, risk_community=1, risk_security=2, risk_environment=1,
                 risk_nationalization=2, risk_currency=1, government_take_pct=35, tax_rate_pct=27), **_FS)
    fill(c, dict(commodity_price=3000, market_cap_musd=1500, share_price=6.0, market_currency="USD",
                 price_vs_ma200_pct=8, momentum_6m_pct=22, rs_rank=71, volume_trend=1, sector_momentum=2,
                 commodity_momentum=2, news_flow=1), **_MKT)
    fill(c, dict(resource_category="proven", independent_resource_estimate=True, drilling_density=2,
                 resource_conversion_history=2, grade_consistency=2, metallurgy_confidence=2, permits_granted=True,
                 timeline_documented=True, historical_delays=2, independent_verification=4,
                 cross_source_consistency=4), **_TR)
    c.catalysts = [{"name": "Expansion till 300 koz — byggbeslut", "type": "expansion", "expected": "2027-Q2",
                    "importance": "HIGH", "impact": "produktion, upside", "confidence": "medium", "source": "Q2-rapport"},
                   {"name": "Resursuppdatering", "type": "resource_update", "expected": "2026-Q4",
                    "importance": "MEDIUM", "impact": "resurs, värdering", "confidence": "high", "source": "Q2-rapport"}]
    return c


def leveraged_producer() -> CompanyInput:
    c = gold_producer()
    c.ticker, c.name = "LVP", "Test Leveraged Producer"
    fill(c, dict(aisc=2650, cash_cost=2100, net_debt_ebitda=3.4, debt_musd=600, cash_musd=30, free_cash_flow_musd=-20,
                 operating_cash_flow_musd=15, interest_expense_musd=40, debt_maturity_year=2027,
                 shares_3y_ago_m=110, shares_1y_ago_m=190, shares_5y_ago_m=60), **_FS)
    return c


def copper_developer() -> CompanyInput:
    """DFS-klar kopparutvecklare: NPV 1 800, CapEx 800, IRR 32 %, delvis finansierad."""
    c = CompanyInput(ticker="CDV", name="Test Copper Developer", commodity="copper", country="CA",
                     jurisdiction="British Columbia", exchange="TSXV", stage="developer", maturity="dfs")
    fill(c, dict(npv_musd=1800, npv_discount_pct=8, npv_price_assumption=4.0, capex_musd=800, irr_pct=32, payback_years=1.9,
                 annual_production=180e6, production_unit="lb", production_future=180e6, production_future_year=2029,
                 resource_unit="lb", breakeven_price=2.6, npv_stress_price_musd=1200, npv_stress_capex_musd=1550,
                 irr_stress_price_pct=24, first_cashflow_year=2029, mine_life_years=18, recovery_pct=88, grade=0.55,
                 grade_unit="%", strip_ratio=2.5, reserve_proven=900e6, reserve_probable=1_600e6, resource_measured=600e6,
                 resource_indicated=1_400e6, resource_inferred=2_000e6, ownership_pct=100, res_size=2, res_infrastructure=4,
                 res_expansion=3, resource_category="probable", independent_resource_estimate=True, drilling_density=2,
                 resource_conversion_history=2, grade_consistency=2, metallurgy_confidence=1, permits_granted=True,
                 engineering_done=True, infrastructure_secured=True, financing_committed=False, offtake_signed=True,
                 timeline_documented=True, historical_delays=2, next_milestone="FID", next_milestone_year=2027,
                 milestone_cost_musd=60, independent_verification=4, cross_source_consistency=3), **_TR)
    fill(c, dict(commodity_price=4.5, market_cap_musd=700, share_price=3.5, market_currency="USD", cash_musd=250,
                 debt_musd=0, quarterly_burn_musd=12, committed_financing_musd=400, has_strategic_partner=True,
                 has_offtake=True, basic_shares_m=200, options_m=12, warrants_m=10, shares_1y_ago_m=190,
                 shares_3y_ago_m=160, shares_5y_ago_m=120, expected_financing_musd=0, atm_program=False), **_MKT)
    fill(c, dict(insider_ownership_pct=12, insider_buying_12m_musd=0.8, mgmt_track_verified="verified",
                 ceo_mines_built=1, team_mines_built=2, team_mines_financed=2, team_discoveries=1, team_exits=1,
                 prior_shareholder_returns=2, board_mining_experience=2, mgmt_dilution_history=1,
                 related_party_issues=False, risk_permitting=2, risk_infrastructure=2, risk_power_water=1,
                 risk_community=1, risk_security=2, risk_environment=1, risk_nationalization=2, risk_currency=1,
                 government_take_pct=38, tax_rate_pct=25), **_FS)
    c.catalysts = [{"name": "FID", "type": "financing", "expected": "2027-Q1", "importance": "HIGH",
                    "impact": "finansiering, risk", "confidence": "medium", "source": "DFS"}]
    return c


def developer_huge_capex_small_company() -> CompanyInput:
    """SPEC §11: starkt projekt + enorm CapEx + litet bolag = hög finansieringsrisk."""
    c = copper_developer()
    c.ticker, c.name = "HCX", "Test Huge Capex Developer"
    fill(c, dict(capex_musd=2400, npv_musd=2600, market_cap_musd=300, cash_musd=40, committed_financing_musd=0,
                 has_strategic_partner=False, financing_committed=False, milestone_cost_musd=90), **_TR)
    return c


def serial_diluter() -> CompanyInput:
    c = copper_developer()
    c.ticker, c.name = "SDL", "Test Serial Diluter"
    fill(c, dict(basic_shares_m=400, shares_1y_ago_m=310, shares_3y_ago_m=160, shares_5y_ago_m=90, atm_program=True,
                 expected_financing_musd=60, mgmt_dilution_history=0), **_MKT)
    return c


def lithium_explorer() -> CompanyInput:
    c = CompanyInput(ticker="LEX", name="Test Lithium Explorer", commodity="lithium", country="CA",
                     jurisdiction="Quebec", exchange="TSXV", stage="explorer", maturity="exploration")
    fill(c, dict(cash_musd=18, quarterly_burn_musd=2.5, market_cap_musd=60, share_price=0.6, market_currency="CAD",
                 fx_to_usd=0.73, basic_shares_m=100, options_m=6, warrants_m=9, shares_1y_ago_m=85, shares_3y_ago_m=60,
                 has_strategic_partner=False, has_offtake=False), **_MKT)
    fill(c, dict(lassonde_stage="discovery", drill_holes=42, best_intercept_gram_m=180, intercept_width_m=25,
                 continuity=1, multiple_zones=True, step_out_success=2, historical_drilling=False, geological_model=1,
                 land_package_km2=120, resource_inferred=1_200_000, resource_unit="t", discovery_option=4,
                 res_size=1, res_grade=1, res_geology=1, res_infrastructure=2, res_expansion=3,
                 resource_category="inferred", independent_resource_estimate=False, insider_ownership_pct=15,
                 mgmt_track_verified="claimed", team_discoveries=2, ceo_mines_built=0, board_mining_experience=1,
                 risk_permitting=1, risk_infrastructure=2, risk_community=1, risk_nationalization=2), **_PRES)
    return c


def royalty_company() -> CompanyInput:
    c = CompanyInput(ticker="ROY", name="Test Royalty Co", commodity="gold", country="CA", exchange="TSX",
                     stage="royalty", maturity="production")
    fill(c, dict(revenue_musd=400, royalty_revenue_share_pct=100, ebitda_margin_pct=80, fcf_yield_pct=4, roic_pct=12,
                 net_debt_ebitda=0.6, free_cash_flow_musd=300, operating_cash_flow_musd=320, cash_musd=200, debt_musd=350,
                 basic_shares_m=190, options_m=2, shares_3y_ago_m=185, insider_ownership_pct=2,
                 mgmt_track_verified="verified", team_exits=2, prior_shareholder_returns=2, board_mining_experience=2,
                 production_current=200_000, resource_unit="oz", npv_musd=9000), **_FS)
    fill(c, dict(commodity_price=3000, market_cap_musd=9000, share_price=47.4, market_currency="USD", ev_ebitda=22,
                 pe=30), **_MKT)
    return c


def overvalued_producer() -> CompanyInput:
    """Samma bolag som GPR men börsvärde 6 000 i stället för 1 500: bra bolag, dåligt pris."""
    c = gold_producer()
    c.ticker, c.name = "OVP", "Test Overvalued Producer"
    fill(c, dict(market_cap_musd=6000, share_price=24.0), **_MKT)
    return c


def missing_everything() -> CompanyInput:
    return CompanyInput(ticker="NUL", name="Test Missing", commodity="gold", stage="developer", maturity="pea")


def currency_mismatch() -> CompanyInput:
    """Kurs i CAD utan växelkurs → börsvärde N/A, inte fel tal."""
    c = copper_developer()
    c.ticker = "FXM"
    fill(c, dict(market_currency="CAD"), **_MKT)
    c.fields.pop("fx_to_usd", None)
    return c


ALL = {"gold_producer": gold_producer, "leveraged_producer": leveraged_producer, "copper_developer": copper_developer,
       "developer_huge_capex_small_company": developer_huge_capex_small_company, "serial_diluter": serial_diluter,
       "lithium_explorer": lithium_explorer, "royalty_company": royalty_company,
       "overvalued_producer": overvalued_producer,
       "missing_everything": missing_everything, "currency_mismatch": currency_mismatch}
