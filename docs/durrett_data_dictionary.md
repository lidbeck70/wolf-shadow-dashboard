# Durrett data dictionary

Alla fält i `confidence.config.FIELDS` som Durrett-motorn (och Confidence score) läser. Varje fält lagras som en
Datapoint: `value`, `kind` (ACTUAL/ESTIMATE/GUIDANCE/MODELLED/ASSUMPTION), `source`, `source_type`
(primary/independent/secondary/mixed/weak/unsupported), `pub_date`, `data_date`, `confidence`, `unit`, `note`.
Saknat fält = N/A i beräkningen och `missing_data` i svaret — aldrig 0. *Stage* säger vilka bolagstyper fältet gäller
(tom = alla). *Max* gäller heltalsbedömningar.

## Aktiestruktur & valuta (steg 3)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `basic_shares_m` | Aktier (basic) | M | number | alla |  |  |
| `options_m` | Optioner | M | number | alla |  |  |
| `warrants_m` | Warranter | M | number | alla |  |  |
| `convertible_shares_m` | Konvertibler (aktier vid konvertering) | M | number | alla |  |  |
| `other_dilutive_m` | Övriga utspädande instrument (RSU m.m.) | M | number | alla |  |  |
| `shares_1y_ago_m` | Aktier för 1 år sedan | M | number | alla |  |  |
| `shares_3y_ago_m` | Aktier för 3 år sedan | M | number | alla |  |  |
| `shares_5y_ago_m` | Aktier för 5 år sedan | M | number | alla |  |  |
| `atm_program` | ATM-program aktivt | — | bool | alla |  |  |
| `recent_financing_musd` | Senaste finansiering (12 mån) | MUSD | number | alla |  |  |
| `expected_financing_musd` | Väntad finansiering (12 mån) | MUSD | number | alla |  |  |
| `market_currency` | Valuta för kurs/börsvärde | — | choice | alla |  |  Val: USD, CAD, AUD, GBP, EUR, SEK, NOK |
| `fx_to_usd` | Växelkurs → USD | USD per enhet | number | alla |  | 1 CAD = 0,73 USD → 0.73. Krävs när valutan inte är USD; ingen dold FX. |

## Reserver & resurser (steg 1)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `reserve_proven` | Reserv — Proven | enheter | number | alla |  |  |
| `reserve_probable` | Reserv — Probable | enheter | number | alla |  |  |
| `resource_measured` | Resurs — Measured | enheter | number | alla |  |  |
| `resource_indicated` | Resurs — Indicated | enheter | number | alla |  |  |
| `resource_inferred` | Resurs — Inferred | enheter | number | alla |  |  |
| `resource_unit` | Enhet för resurser/produktion | — | choice | alla |  | Samma enhet för reserv, resurs och produktion — motorn räknar om till bas-enheten Val: oz, koz, Moz, lb, Mlb, t, kt, Mt |
| `ownership_pct` | Ägarandel i projektet | % | number | alla |  |  |
| `recovery_pct` | Metallurgisk recovery | % | number | alla |  |  |
| `grade` | Halt | — | number | alla |  | g/t för guld/silver, % för basmetaller |
| `grade_unit` | Haltenhet | — | choice | alla |  |  Val: g/t, %, ppm, lb/t |
| `strip_ratio` | Strip ratio | × | number | alla |  |  |
| `mining_method` | Brytningsmetod | — | text | alla |  |  |
| `processing_method` | Processmetod | — | text | alla |  |  |
| `land_package_km2` | Landpaket | km² | number | alla |  |  |

## Produktion & kostnader (steg 5, 7)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `production_current` | Produktion nu | enheter/år | number | alla |  | Samma enhet som resource_unit. Producenter: senaste år. |
| `production_future` | Produktion i framtiden (plan) | enheter/år | number | alla |  |  |
| `production_future_year` | År för framtida produktion | år | int | alla |  |  |
| `production_3y_ago` | Produktion för 3 år sedan | enheter/år | number | producer, royalty |  |  |
| `resource_3y_ago` | Total resurs för 3 år sedan | enheter | number | alla |  |  |
| `reserve_3y_ago` | Total reserv för 3 år sedan | enheter | number | alla |  |  |
| `cash_cost` | Cash cost / C1 | USD/enhet | number | producer, royalty |  |  |
| `sustaining_capex_musd` | Sustaining CapEx | MUSD/år | number | producer, royalty |  |  |
| `expansion_capex_musd` | Expansions-CapEx | MUSD | number | alla |  |  |
| `royalty_burden_pct` | Royaltybörda (NSR/GRR-summa) | % | number | alla |  |  |
| `streaming_burden_pct` | Streamingbörda (andel av produktionen) | % | number | alla |  |  |
| `royalty_revenue_share_pct` | Andel av intäkten från royalty/stream | % | number | alla |  | Klassificering: ≥ 40 % + egen drift → HYBRID |
| `revenue_musd` | Omsättning (senaste 12 mån) | MUSD | number | alla |  | Klassificering: kommersiell produktion |

## Kassa & skuld (steg 8)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `working_capital_musd` | Rörelsekapital | MUSD | number | alla |  |  |
| `operating_cash_flow_musd` | Operativt kassaflöde (12 mån) | MUSD | number | alla |  |  |
| `free_cash_flow_musd` | Fritt kassaflöde (12 mån) | MUSD | number | alla |  |  |
| `interest_expense_musd` | Räntekostnad (12 mån) | MUSD | number | alla |  |  |
| `debt_maturity_year` | Skuldens förfalloår | år | int | alla |  |  |
| `next_milestone` | Nästa stora milstolpe | — | text | explorer, developer |  |  |
| `next_milestone_year` | År för nästa milstolpe | år | number | explorer, developer |  | Decimal ok (2027.5). Funding cliff om finansiering krävs före den. |
| `milestone_cost_musd` | Kostnad fram till milstolpen | MUSD | number | explorer, developer |  |  |

## Management (steg 2)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `mgmt_track_verified` | Track record | — | choice | alla |  | verified = källa som bekräftar (rapport, börsdata); claimed = bolagets egen presentation Val: verified, claimed, unknown |
| `ceo_mines_built` | Gruvor CEO byggt | st | int | alla |  |  |
| `team_mines_built` | Gruvor teamet byggt | st | int | alla |  |  |
| `team_mines_financed` | Gruvor teamet finansierat | st | int | alla |  |  |
| `team_mines_operated` | Gruvor teamet drivit | st | int | alla |  |  |
| `team_discoveries` | Fyndigheter teamet upptäckt | st | int | alla |  |  |
| `team_exits` | Tidigare exits (uppköp) | st | int | alla |  |  |
| `prior_shareholder_returns` | Tidigare aktieägaravkastning | p | int | alla | 2 | 0 förstört värde · 1 blandat · 2 skapat värde (med källa) |
| `board_mining_experience` | Styrelsens gruverfarenhet | p | int | alla | 2 |  |
| `insider_buying_12m_musd` | Insiderköp (12 mån) | MUSD | number | alla |  |  |
| `insider_selling_12m_musd` | Insiderförsäljning (12 mån) | MUSD | number | alla |  |  |
| `mgmt_compensation_musd` | Ledningens ersättning (år) | MUSD | number | alla |  |  |
| `related_party_issues` | Närståendetransaktioner / problem | — | bool | alla |  |  |

## Projektrisk (steg 4)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `risk_permitting` | Tillståndsrisk | p | int | alla | 2 |  |
| `risk_infrastructure` | Infrastruktur (väg, hamn) | p | int | alla | 2 |  |
| `risk_power_water` | Kraft och vatten | p | int | alla | 2 |  |
| `risk_community` | Lokalsamhälle / urfolk | p | int | alla | 2 |  |
| `risk_security` | Säkerhet | p | int | alla | 2 |  |
| `risk_environment` | Miljörestriktioner | p | int | alla | 2 |  |
| `risk_nationalization` | Nationaliserings-/ägarrestriktioner | p | int | alla | 2 |  |
| `risk_currency` | Valutarisk | p | int | alla | 2 |  |
| `government_take_pct` | Skatt + royalty till staten | % | number | alla |  |  |

## Explorer (Lassonde, discovery, optionality)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `lassonde_stage` | Lassonde-position | — | choice | alla |  |  Val: exploration, discovery, resource_definition, pea, pfs, fs, construction, production, expansion |
| `drill_holes` | Antal borrhål | st | int | alla |  |  |
| `best_intercept_gram_m` | Bästa intercept (halt × meter) | g/t·m eller %·m | number | alla |  |  |
| `intercept_width_m` | Typisk mineraliserad bredd | m | number | alla |  |  |
| `continuity` | Kontinuitet | p | int | alla | 2 |  |
| `multiple_zones` | Flera zoner | — | bool | alla |  |  |
| `step_out_success` | Step-out-framgång | p | int | alla | 2 |  |
| `historical_drilling` | Historisk borrning finns | — | bool | alla |  |  |
| `geological_model` | Geologisk modell | p | int | alla | 2 | 0 oklar · 1 arbetshypotes · 2 väl förstådd |
| `implied_value_per_unit` | Implicit värde per enhet i marken | USD/enhet | number | alla |  | Vad marknaden betalar för jämförbara uns/lb — ange källa |

## Momentum (steg 6)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `price_vs_ma200_pct` | Kurs mot MA200 | % | number | alla |  |  |
| `momentum_6m_pct` | Kursutveckling 6 mån | % | number | alla |  |  |
| `rs_rank` | RS-rank (Börsdata 0–100) | — | number | alla |  |  |
| `volume_trend` | Volymtrend | p | int | alla | 2 |  |
| `sector_momentum` | Sektormomentum | p | int | alla | 2 |  |
| `commodity_momentum` | Råvarumomentum | p | int | alla | 2 |  |
| `news_flow` | Nyhetsflöde | p | int | alla | 2 |  |

## Ekonomi (delas med Confidence score)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `commodity_price` | Råvarupris nu | USD/enhet | number | alla |  |  |
| `aisc` | AISC / opex per enhet | USD/enhet | number | producer, royalty |  |  |
| `ebitda_margin_pct` | EBITDA-marginal | % | number | producer, royalty |  |  |
| `fcf_yield_pct` | FCF-yield | % | number | producer, royalty |  |  |
| `roic_pct` | ROIC | % | number | producer, royalty |  |  |
| `breakeven_price` | Break-even råvarupris | USD/enhet | number | alla |  |  |
| `npv_musd` | NPV efter skatt | MUSD | number | explorer, developer |  |  |
| `npv_discount_pct` | Diskonteringsränta i NPV | % | number | explorer, developer |  |  |
| `npv_price_assumption` | Råvarupris i NPV | USD/enhet | number | explorer, developer |  |  |
| `capex_musd` | Initial CapEx | MUSD | number | explorer, developer |  |  |
| `irr_pct` | IRR efter skatt | % | number | explorer, developer |  |  |
| `payback_years` | Payback | år | number | explorer, developer |  |  |
| `annual_production` | Årsproduktion | enheter/år | number | alla |  |  |
| `production_unit` | Produktionsenhet | — | text | alla |  | oz, lb, t, boe … |
| `npv_stress_price_musd` | NPV vid pris −20 % | MUSD | number | explorer, developer |  | Ur FS-känslighetstabellen. Saknas → modelleras i scenarier. |
| `npv_stress_capex_musd` | NPV vid CapEx +20 % | MUSD | number | explorer, developer |  |  |
| `irr_stress_price_pct` | IRR vid pris −20 % | % | number | explorer, developer |  |  |

## Resurskvalitet (delas)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `res_size` | Resursstorlek | p | int | alla | 2 | 0 liten · 1 medel · 2 tier 1-storlek för råvaran |
| `res_grade` | Halt | p | int | alla | 2 | 0 under branschsnitt · 1 snitt · 2 över snitt (AQS kostnadsposition 2 → 2) |
| `res_metallurgy` | Metallurgi / recovery | p | int | alla | 1 | 1 bevisad enkel · 0 problematisk eller obevisad |
| `res_mine_life` | Gruvlivslängd | p | int | alla | 1 | 1 om > 10 år (AQS livslängd 2 → 1) |
| `res_geology` | Geologi | p | int | alla | 2 | 0 komplex · 1 normal · 2 enkel, förutsägbar |
| `res_infrastructure` | Infrastruktur / läge | p | int | alla | 4 | 0 saknas · 2 delvis · 4 väg, kraft, vatten, tillstånd (AQS 0/1/2 → 0/2/4) |
| `res_expansion` | Expansionspotential | p | int | alla | 3 | 0 ingen · 1 viss · 3 tydlig, billig (AQS 0/1/2 → 0/1/3) |
| `mine_life_years` | Gruvlivslängd | år | number | alla |  |  |

## Produktion & tillväxt (delas)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `first_cashflow_year` | Första kassaflöde (år) | år | int | alla |  | Producenter: innevarande år. |
| `discovery_option` | Discovery Option | p | int | explorer | 5 | 0–5, hålls separat från huvudscoren |

## Balansräkning (delas)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `net_debt_ebitda` | Nettoskuld/EBITDA | × | number | producer, royalty |  |  |
| `cash_musd` | Kassa | MUSD | number | alla |  |  |
| `debt_musd` | Skuld | MUSD | number | alla |  |  |
| `quarterly_burn_musd` | Burn per kvartal | MUSD | number | explorer, developer |  |  |
| `committed_financing_musd` | Åtagen finansiering | MUSD | number | explorer, developer |  |  |
| `has_strategic_partner` | Strategisk partner | — | bool | explorer, developer |  |  |
| `has_offtake` | Offtake-avtal | — | bool | explorer, developer |  |  |
| `dilution_score` | DS (utspädning 0–10) | p | int | alla | 10 | Från kontrollerna (controls.ds_total) |

## Värdering (delas)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `market_cap_musd` | Börsvärde | MUSD | number | alla |  |  |
| `enterprise_value_musd` | Enterprise value | MUSD | number | alla |  |  |
| `ev_ebitda` | EV/EBITDA | × | number | producer, royalty |  |  |
| `pe` | P/E | × | number | producer, royalty |  |  |
| `ev_ebit` | EV/EBIT | × | number | producer, royalty |  |  |
| `nav_musd` | NAV (efter skatt) | MUSD | number | alla |  |  |
| `p_nav` | P/NAV | × | number | alla |  | Räknas som börsvärde / NAV om båda finns. |

## Scenarier (delas)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `shares_outstanding_m` | Antal aktier (fullt utspätt) | M | number | alla |  |  |
| `share_price` | Aktiekurs | valuta/aktie | number | alla |  | Samma valuta som börsvärdet delat med aktier |
| `tax_rate_pct` | Skattesats | % | number | alla |  | Saknas → 25 % (ASSUMPTION) |
| `target_ev_ebitda` | EV/EBITDA i scenariot | × | number | producer, royalty |  | Saknas → 5× (ASSUMPTION) |
| `target_p_nav` | P/NAV vid omvärdering | × | number | explorer, developer |  | Saknas → 0.7× (ASSUMPTION) |

## Management-bedömningar (delas)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `mgmt_track_record` | Byggmeriter | p | int | alla | 1 |  |
| `mgmt_capital_allocation` | Kapitalallokering | p | int | alla | 1 |  |
| `mgmt_dilution_history` | Utspädningshistorik | p | int | alla | 1 | 1 = disciplinerad |
| `insider_ownership_pct` | Insynsägande | % | number | alla |  |  |
| `mgmt_alignment_delivery` | Alignment & leverans mot löften | p | int | alla | 1 |  |

## Resurssäkerhet (Confidence)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `resource_category` | Högsta resurskategori | — | choice | alla |  |  Val: proven, probable, measured, indicated, inferred, exploration_target |
| `resource_verified_share_pct` | Andel oberoende verifierad | % | number | alla |  |  |
| `independent_resource_estimate` | Oberoende resursuppskattning finns | — | bool | alla |  | 43-101/JORC av oberoende QP |
| `drilling_density` | Borrtäthet | p | int | alla | 2 | 0 gles · 1 normal · 2 tät |
| `resource_conversion_history` | Konverteringshistorik | p | int | alla | 2 | 0 nedskrivningar · 1 stabil · 2 uppgraderingar |
| `grade_consistency` | Haltkonsistens | p | int | alla | 2 |  |
| `metallurgy_confidence` | Metallurgisk säkerhet | p | int | alla | 2 | 0 labb · 1 pilot · 2 kommersiell drift |

## Projektmognad (Confidence)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `permits_granted` | Nyckeltillstånd beviljade | — | bool | alla |  |  |
| `engineering_done` | Detaljprojektering klar | — | bool | explorer, developer |  |  |
| `infrastructure_secured` | Infrastruktur säkrad | — | bool | explorer, developer |  |  |
| `financing_committed` | Finansiering åtagen | — | bool | explorer, developer |  |  |
| `offtake_signed` | Offtake signerat | — | bool | explorer, developer |  |  |
| `construction_contracts` | Byggkontrakt tecknade | — | bool | explorer, developer |  |  |
| `procurement_started` | Upphandling påbörjad | — | bool | explorer, developer |  |  |

## Tidsplan (Confidence)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `timeline_documented` | Tidsplan dokumenterad och finansierad | — | bool | alla |  |  |
| `historical_delays` | Tidigare förseningar | p | int | alla | 2 | 0 upprepade · 1 någon · 2 inga |

## Datakvalitet (Confidence)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `independent_verification` | Oberoende verifiering | p | int | alla | 5 | 0 bara bolagets ord · 3 nyckeltal bekräftade av QP/revisor · 5 alla nyckeltal oberoende |
| `cross_source_consistency` | Källkonsistens | p | int | alla | 5 | 0 motstridiga eller en enda källa · 3 huvudsak överens · 5 flera källor säger samma |

## Kill switches (Confidence)

| Nyckel | Etikett | Enhet | Typ | Stage | Max | Beskrivning |
|---|---|---|---|---|---|---|
| `no_financing_plan` | Ingen realistisk finansieringsplan | — | bool | explorer, developer |  |  |
| `record_price_dependent` | Fungerar bara vid rekordpris | — | bool | alla |  |  |
| `single_fragile_parameter` | Caset vilar på en osäker parameter | — | bool | alla |  |  |
| `capital_destruction_history` | Dokumenterad kapitalförstöring | — | bool | alla |  |  |

## Beräknade nyckeltal (`DurrettAnalysis.metrics`)

| Nyckel | Formel |
|---|---|
| `fd_m` | basic + optioner + warranter + konvertibler + övrigt |
| `dilution_1y/3y/5y` | aktier nu / aktier då − 1 |
| `mcap_usd`, `fd_mcap_usd` | börsvärde (× fx_to_usd), FD = kurs × FD-aktier |
| `ev_usd`, `fd_ev_usd` | börsvärde + nettoskuld |
| `reserve_total`, `resource_total`, `mi_total` | P+P, M+I+I, M+I (attributable) med kategorinot |
| `mcap_per_*_unit`, `ev_per_*_unit` | MUSD × 1e6 / enheter |
| `npv_capex` | NPV / initial CapEx |
| `ev_npv` | EV / NPV |
| `operating_margin_per_unit`, `aisc_to_price` | pris − AISC; AISC / pris |
| `production_growth_multiple`, `production_cagr` | framtid / nu; CAGR över åren dit |
| `cash_runway_years` | kassa / årsburn (kvartalsburn × 4 eller −OCF) |
| `funding_coverage` | (kassa + åtagen) / CapEx |
| `funding_cliff_gap` | kostnad till nästa milstolpe − kassa − åtagen |
| `future_operating_cf`, `future_fcf`, `future_ev` | produktion × (pris − AISC); × (1 − skatt); × multipel |
| `mcap_future_earnings` | börsvärde / framtida FCF (Durretts 10×-regel) |
| `upside_multiple`, `upside_pct` | framtida börsvärde / FD-börsvärde |

## Konfiguration

Vikter, riskvikter, multiplar, scenario-defaults, red-flag-trösklar, delpoängstrappor, developer-checklista,
explorer-trösklar och klassificering ligger i `engines/durrett/config.py::DURRETT_CONFIG`.
