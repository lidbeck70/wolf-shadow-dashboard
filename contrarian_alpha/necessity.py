"""
Necessity Score — sektor-map för Contrarian Alpha Screener.

Mäter hur fysiskt/ekonomiskt nödvändig en sektor är i den verkliga ekonomin.
Höga poäng → hårda råvaror, energi, kritisk infrastruktur (contrarian value-vänliga)
Låga poäng  → mjukvara, krypto, spekulativ growth (contrarian-olämpliga)

Threshold pipeline: NECESSITY_THRESHOLD = 60
Källa: Börsdata GICS-klassificering (sektor → industrigrupp → industri → sub-industri)
"""

from __future__ import annotations
from dataclasses import dataclass

# ─── Pipeline-tröskel ────────────────────────────────────────────────────────

NECESSITY_THRESHOLD = 60  # Necessity Score >= 60 krävs för att gå vidare

# ─── Datastruktur ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NecessityEntry:
    score: int          # 0–100
    label: str          # Läsbar etikett
    rationale: str      # Kortfattad motivering


# ─── GICS Sub-industri-map (8-siffrig kod → NecessityEntry) ─────────────────
#
# Prioritetsordning vid lookup:
#   1. sub_industry (8 siffror)  — mest specifik
#   2. industry     (6 siffror)
#   3. industry_group (4 siffror)
#   4. sector       (2 siffror)
#   5. FALLBACK_SCORE             — okänd sektor
#
# Börsdata fällt: branch_id (sektor), sector_id (industri), etc.
# Se borsdata_api.py get_instruments() för faktiska fältnamn.

NECESSITY_MAP: dict[int, NecessityEntry] = {

    # ═══════════════════════════════════════════════════════════════
    # SEKTOR 10 — ENERGY
    # ═══════════════════════════════════════════════════════════════
    10: NecessityEntry(88, "Energi", "Fossil energi, kritisk global infrastruktur"),

    # Industri 1010 — Energy
    1010: NecessityEntry(88, "Energi", "Olja, gas, kol — ekonomins ryggrad"),

    # Sub-industrier
    10101010: NecessityEntry(87, "Olja & Gas Borrning", "Uppströmsvärdekedja, cyklisk men nödvändig"),
    10101020: NecessityEntry(84, "Olja & Gas Utrustning", "Servicebolag, nödvändig stödinfrastruktur"),
    10102010: NecessityEntry(92, "Integrerat Olja & Gas", "Vertikalt integrerade majors, stabil kassaflödesmaskin"),
    10102020: NecessityEntry(90, "Olja & Gas E&P",       "Utvinning av fysisk råvara, ingen digital substitut"),
    10102030: NecessityEntry(86, "Olja & Gas Raffinering","Förädlingsled, bränsle till transport & industri"),
    10102040: NecessityEntry(88, "Olja & Gas Transport",  "Pipelines & tankers, kritisk energiinfrastruktur"),
    10102050: NecessityEntry(82, "Kol & Bränslen",        "Kol används fortfarande i kraft & stål globalt"),

    # ═══════════════════════════════════════════════════════════════
    # SEKTOR 15 — MATERIALS
    # ═══════════════════════════════════════════════════════════════
    15: NecessityEntry(80, "Material", "Råvaror och halvfabrikat för realekonomin"),

    # Industrigrupp 1510 — Materials
    1510: NecessityEntry(80, "Material", "Metaller, kemi, skog — industrins byggstenar"),

    # Kemikalier (151010)
    15101010: NecessityEntry(72, "Diversifierade Kemikalier",  "Insatsvara i jordbruk, industri, medicin"),
    15101020: NecessityEntry(68, "Fertilitets- & Jordbrukskemi","Gödsel & bekämpningsmedel, global matproduktion"),
    15101030: NecessityEntry(65, "Industrikemikalier",          "Processindustri, plast, halvfabrikat"),
    15101040: NecessityEntry(60, "Specialkemikalier",           "Nischad men funktionskritisk för tillverkning"),

    # Konstruktionsmaterial (151020)
    15102010: NecessityEntry(74, "Konstruktionsmaterial",  "Cement, glas, ballast — fysisk infrastruktur"),

    # Förpackningar (151030)
    15103010: NecessityEntry(62, "Metall-, Glas- & Plastförpackning", "Matförsörjning, läkemedel — nödvändig logistik"),
    15103020: NecessityEntry(58, "Papper- & Plastförpackning",        "Viktig men lättare substituerbar"),

    # Metaller & Gruvor (151040) — KÄRNA i Contrarian Alpha
    15104010: NecessityEntry(82, "Aluminium",                 "Lättvikt, fordon, flyg, förpackning"),
    15104020: NecessityEntry(86, "Diversifierade Metaller",   "Koppar, zink, bly — elektrifiering & industri"),
    15104025: NecessityEntry(92, "Koppar",                    "Elektricitetsledare, EV, förnybart — ingen substitut"),
    15104030: NecessityEntry(90, "Guld",                      "Monetärt värde, säkerhetsvaluta, industriell elek"),
    15104035: NecessityEntry(88, "Uran & Kärnbränsle",        "Basenergi med exponentiellt ökad strategisk efterfrågan"),
    15104040: NecessityEntry(86, "Ädelmetaller & Mineral",    "Silver, PGM — industri + monetary hedge"),
    15104045: NecessityEntry(85, "Silver",                    "Solar panels, elektronik, monetärt värde"),
    15104050: NecessityEntry(80, "Stål",                      "Konstruktion, maskineri, infrastruktur"),
    15104060: NecessityEntry(85, "Litium & Batterimineral",   "EV-revolution, energilagring, strategisk råvara"),
    15104065: NecessityEntry(83, "Nickel & Kobolt",           "Batterianoder, rostfritt stål, superlegeringar"),
    15104070: NecessityEntry(79, "Järnmalm",                  "Stålproduktion, global infrastruktur"),

    # Skog & Papper (151050)
    15105010: NecessityEntry(62, "Skogsprodukter",    "Trä, byggmaterial, papper"),
    15105020: NecessityEntry(52, "Pappersprodukter",  "Avtagande digitaliserar bort viss efterfrågan"),

    # ═══════════════════════════════════════════════════════════════
    # SEKTOR 20 — INDUSTRIALS
    # ═══════════════════════════════════════════════════════════════
    20: NecessityEntry(62, "Industri", "Kapitalvaror, logistik, professionella tjänster"),

    2010: NecessityEntry(66, "Kapitalvaror",             "Maskiner, försvar, byggindustri"),
    2020: NecessityEntry(55, "Kommersiella Tjänster",    "Outsourcing, personaluthyrning, administration"),
    2030: NecessityEntry(70, "Transport",                "Sjöfart, järnväg, flyg — kritisk logistik"),

    20101010: NecessityEntry(76, "Flyg & Försvar",     "Nationell säkerhet, strategisk industri"),
    20101020: NecessityEntry(64, "Byggprodukter",      "Material för bostäder & infrastruktur"),
    20101030: NecessityEntry(68, "Byggentreprenad",    "Infrastrukturbyggnad, sjukhus, skolor"),
    20101040: NecessityEntry(70, "Elektroteknik",      "Kraftdistribution, automation"),
    20101050: NecessityEntry(62, "Industrikonglomerat", "Diversifierat, indirekta nödvändigheter"),
    20101060: NecessityEntry(64, "Maskiner",            "Produktionsutrustning, jordbruksmaskiner"),
    20101070: NecessityEntry(58, "Handelsföretag",      "Distribution & trading, indirekt"),
    20102010: NecessityEntry(55, "Kommersiella Tryck",  "Tryckerier, avtagande sektor"),
    20102020: NecessityEntry(52, "Miljötjänster",       "Avfallshantering, vattenrening"),
    20102030: NecessityEntry(54, "Kontorservice",       "Stödtjänster, ej kärnverksamhet"),
    20103010: NecessityEntry(72, "Flygtransport",       "Global varurörlighet, expressfrakt"),
    20103020: NecessityEntry(68, "Flygplatser",         "Kritisk infrastruktur"),
    20103030: NecessityEntry(66, "Lastbilstransport",   "Sista-mil logistik"),
    20103040: NecessityEntry(74, "Järnväg",             "Bulkgods, energieffektiv transport"),
    20103050: NecessityEntry(72, "Sjöfart",             "90% av globalt handelsvolym"),
    20103060: NecessityEntry(64, "Transportinfrastruktur","Hamnar, broar, motorvägar"),

    # ═══════════════════════════════════════════════════════════════
    # SEKTOR 25 — CONSUMER DISCRETIONARY
    # ═══════════════════════════════════════════════════════════════
    25: NecessityEntry(30, "Konsumtion (Diskretionär)", "Lyxkonsumtion, ej existentiellt nödvändig"),

    2510: NecessityEntry(45, "Bilar & Komponenter",      "Transport är nödvändig, men ej lyxbil"),
    2520: NecessityEntry(28, "Kapitalvaror & Mode",      "Kläder, möbler — icke-nödvändig nivå"),
    2530: NecessityEntry(25, "Konsumenttjänster",        "Restauranger, hotell, fritid"),
    2550: NecessityEntry(22, "Detaljhandel (Diskret)",   "E-handel, specialbutiker — valbara"),
    2560: NecessityEntry(18, "E-handel (Lyx)",           "Online lyx/mode, starkt cyklisk"),

    25101010: NecessityEntry(48, "Bildelar",        "Underhåll är nödvändigt"),
    25101020: NecessityEntry(40, "Personbilar",     "Transport behov, men lyxpremium ej nödvändig"),
    25102010: NecessityEntry(22, "Hushållsapparater","Vita varor — viss nödvändighet"),
    25102020: NecessityEntry(18, "Heminredning",    "Ej nödvändig"),
    25102030: NecessityEntry(15, "Leksaker & Fritid","Underhållning, ej nödvändig"),
    25201010: NecessityEntry(30, "Klädtillverkning", "Kläder nödvändigt, men varumärkespremie ej"),
    25301010: NecessityEntry(28, "Hotell & Resorter","Turism, ej nödvändig"),
    25301020: NecessityEntry(22, "Fritid & Nöje",   "Ej nödvändig"),
    25301030: NecessityEntry(32, "Restauranger",    "Mat nödvändig, men restaurang ej"),
    25501010: NecessityEntry(25, "Varuhus",         "Ersättningsbar shoppingkanal"),
    25502020: NecessityEntry(12, "Internet-detaljhandel","E-handel, ej fysisk nödvändighet"),

    # ═══════════════════════════════════════════════════════════════
    # SEKTOR 30 — CONSUMER STAPLES
    # ═══════════════════════════════════════════════════════════════
    30: NecessityEntry(80, "Dagligvaror", "Mat, dryck, hygien — grundläggande konsumtion"),

    3010: NecessityEntry(78, "Dagligvaruhandel",       "ICA, Coop — matdistribution"),
    3020: NecessityEntry(82, "Mat, Dryck & Tobak",     "Baslivsmedel, hög defensiv karaktär"),
    3030: NecessityEntry(72, "Hushåll & Personvård",   "Hygienartiklar, rengöring"),

    30101010: NecessityEntry(80, "Dagligvarubutiker",    "Daglig mat & hushållsvaror"),
    30101020: NecessityEntry(72, "Hypermarkets",         "Mathandel med bred varukorg"),
    30201010: NecessityEntry(85, "Drycker (Alkoholfria)","Vatten, mjölk, juice — nödvändig"),
    30201020: NecessityEntry(70, "Drycker (Alkohol)",    "Hög priskänslighet, inte livsviktigt"),
    30201030: NecessityEntry(68, "Tobak",                "Beroendeprodukt, defensiv men avtagande"),
    30202010: NecessityEntry(87, "Jordbruksprodukter",   "Vete, soja, majs — global matförsörjning"),
    30202020: NecessityEntry(82, "Livsmedel & Kött",     "Bröd, kött, mejeriprodukter"),
    30202030: NecessityEntry(78, "Förpackade Livsmedel", "Konserver, frysvaror, snabbmat"),
    30301010: NecessityEntry(74, "Hushållsprodukter",    "Rengöring, tvättmedel"),
    30301020: NecessityEntry(70, "Personvårdsprodukter", "Hygien, hög lojalitet"),

    # ═══════════════════════════════════════════════════════════════
    # SEKTOR 35 — HEALTH CARE
    # ═══════════════════════════════════════════════════════════════
    35: NecessityEntry(82, "Hälsovård", "Läkemedel, sjukvård — existentiellt nödvändig"),

    3510: NecessityEntry(80, "Sjukvårdsutrustning",      "MRI, kirurgi, diagnostik"),
    3520: NecessityEntry(85, "Läkemedel & Biotech",      "Läkemedel, vacciner, livsavgörande"),

    35101010: NecessityEntry(80, "Sjukvårds-IT",         "Journalsystem, administrativ digitalisering"),
    35101020: NecessityEntry(84, "Sjukvårdstjänster",    "Sjukhus, kliniker, akutvård"),
    35101030: NecessityEntry(82, "Medicinsk Utrustning", "Operationsrobotar, diagnostikinstrument"),
    35101040: NecessityEntry(78, "Medicinsk Teknologi",  "Implantat, proteser, testprodukter"),
    35102010: NecessityEntry(82, "Bioteknik",            "Ny medicinsk forskning, högrisk men nödvändig"),
    35102020: NecessityEntry(88, "Läkemedel",            "Etablerade läkemedel, stabil efterfrågan"),
    35102030: NecessityEntry(72, "Livsvetenskaplig CRO", "Forskningstjänster — viktig men B2B"),

    # ═══════════════════════════════════════════════════════════════
    # SEKTOR 40 — FINANCIALS
    # ═══════════════════════════════════════════════════════════════
    40: NecessityEntry(65, "Finans", "Banker & försäkring är infrastruktur, men abstrakt"),

    4010: NecessityEntry(70, "Banker",              "Kreditgivning, betalningssystem, systemviktiga"),
    4020: NecessityEntry(55, "Diversifierad Finans", "Fondbolag, investmentbanker"),
    4030: NecessityEntry(68, "Försäkring",           "Riskdelning, pensioner, kritisk funktion"),

    40101010: NecessityEntry(72, "Diversifierade Banker",   "Storbanker med systemvikt"),
    40101015: NecessityEntry(68, "Regionala Banker",        "Lokal kreditgivning"),
    40101020: NecessityEntry(65, "Thrift & Hypoteksbanker", "Bolåneinstitut"),
    40102010: NecessityEntry(50, "Finansieringsbolag",      "Leasing, faktoring"),
    40203010: NecessityEntry(38, "Kapitalmarknader",        "Investment banking, trading"),
    40203020: NecessityEntry(30, "Investmentbolag",         "Fonder, PE — kapitalallokering"),
    40203030: NecessityEntry(20, "Krypto & Digitala Tillgångar", "Spekulativa, ej nödvändiga"),
    40301010: NecessityEntry(72, "Försäkring (Liv & Hälsa)","Pensionssparande, socialt viktigt"),
    40301020: NecessityEntry(70, "Försäkring (Sak)",        "Fastighet, fordon, skadeersättning"),
    40301030: NecessityEntry(66, "Återförsäkring",          "Systemstabilisator för försäkringsmarknaden"),

    # ═══════════════════════════════════════════════════════════════
    # SEKTOR 45 — INFORMATION TECHNOLOGY
    # ═══════════════════════════════════════════════════════════════
    45: NecessityEntry(8, "IT", "Mjukvara, hårdvara, IT-tjänster — ej fysiska nödvändigheter"),

    4510: NecessityEntry(5, "Mjukvara & IT-tjänster", "SaaS, konsulting — bytbara och spekulativa"),
    4520: NecessityEntry(12, "Hårdvara & Utrustning",  "Servers, mobiltelefoner — nödvändig klass"),
    4530: NecessityEntry(18, "Halvledare",              "Kritisk komponent, men cyklisk & spekulativ"),

    45101010: NecessityEntry(5,  "IT-konsulting",          "Outsourcad IT, ej fysisk nödvändighet"),
    45101020: NecessityEntry(3,  "Datatjänster",            "Molntjänster, ej fysisk nödvändighet"),
    45102010: NecessityEntry(4,  "Internet-mjukvara & SaaS","SaaS — noll fysisk nödvändighet"),
    45102020: NecessityEntry(6,  "Systemapplikationer",     "ERP, CRM — viktig men ej nödvändig"),
    45103010: NecessityEntry(14, "Elektronisk utrustning",  "Sensorer, kameror, industri-elektronik"),
    45103020: NecessityEntry(10, "Elektronikkomponenter",   "PCBs, kablar"),
    45103030: NecessityEntry(20, "Elektronisk tillverkning","Kontraktstillverkning av hårdvara"),
    45103040: NecessityEntry(8,  "Teknologidistribution",   "IT-återförsäkring, grossister"),
    45201010: NecessityEntry(12, "Kommunikationsutrustning","Routrar, switchar — infrastrukturell"),
    45201020: NecessityEntry(15, "Datalagrings- & Perifera","Lagringshårdvara"),
    45202010: NecessityEntry(12, "Smartphones & Datorer",   "Konsumentelektronik, ej nödvändig"),
    45301010: NecessityEntry(20, "Halvledare",              "Kiselbaserade chips — industriellt kritisk"),
    45301020: NecessityEntry(16, "Halvledarutrustning",     "ASML m.fl., B2B specialutrustning"),

    # ═══════════════════════════════════════════════════════════════
    # SEKTOR 50 — COMMUNICATION SERVICES
    # ═══════════════════════════════════════════════════════════════
    50: NecessityEntry(28, "Kommunikationstjänster", "Telekom nödvändig, media/social ej"),

    5010: NecessityEntry(58, "Telekommunikation",   "Mobilnät, bredband — infrastruktur"),
    5020: NecessityEntry(12, "Media & Underhållning","Streaming, sociala medier — ej nödvändig"),

    50101010: NecessityEntry(60, "Alternativa Carriers",  "Fiber, satellit, 5G-infrastruktur"),
    50101020: NecessityEntry(62, "Integrerad Telekom",    "Telia, Telenor — digital infrastruktur"),
    50102010: NecessityEntry(8,  "Interaktiv Media",       "Sociala medier — noll fysisk nödvändighet"),
    50102020: NecessityEntry(10, "Interaktivt Hemspel",    "Spelindustri — underhållning"),
    50202010: NecessityEntry(15, "Kabelnätverk",           "TV-distribution, vikande relevans"),
    50202020: NecessityEntry(12, "Film & TV",              "Underhållning, ej nödvändig"),
    50203010: NecessityEntry(6,  "Sociala Medier-plattformar","Annonsdriven, spekulativ"),
    50203020: NecessityEntry(14, "Reklamtjänster",         "B2B, men beroende av konjunktur"),

    # ═══════════════════════════════════════════════════════════════
    # SEKTOR 55 — UTILITIES
    # ═══════════════════════════════════════════════════════════════
    55: NecessityEntry(90, "Allmännyttiga Tjänster", "El, vatten, gas — absolut nödvändig infrastruktur"),

    5510: NecessityEntry(90, "Utilities", "El, vatten, gas"),

    55101010: NecessityEntry(94, "Vatten & Avlopp",      "Vatten = liv, absolut monopol"),
    55102010: NecessityEntry(92, "Elproduktion (Diversif.)","Elbolag, stabil reglerad intäkt"),
    55102020: NecessityEntry(96, "Kärnkraft",             "Baslast, koldioxidfri, strategisk"),
    55102030: NecessityEntry(90, "Förnybar Elproduktion", "Sol, vind — framtidens basenergi"),
    55102040: NecessityEntry(88, "Fossil Elproduktion",   "Gas/kol kraftverk, transitional"),
    55103010: NecessityEntry(88, "Gasinfrastruktur",      "Distribution av naturgas, uppvärmning"),
    55104010: NecessityEntry(85, "Oberoende Kraft",       "Privata elproducenter"),
    55105010: NecessityEntry(86, "Energihandel (Diversif.)","Multi-utility, stabil kontantflöde"),

    # ═══════════════════════════════════════════════════════════════
    # SEKTOR 60 — REAL ESTATE
    # ═══════════════════════════════════════════════════════════════
    60: NecessityEntry(62, "Fastigheter", "Bostad & lokal är nödvändig, REIT-strukturen ger stabilitet"),

    6010: NecessityEntry(62, "Fastigheter", "REITs och fastighetsbolag"),

    60101010: NecessityEntry(70, "Bostadsfastigheter",     "Bostäder är nödvändiga, stabilt kassaflöde"),
    60101020: NecessityEntry(65, "Kontor & Lokaler",       "Affärslokaler, vikande post-covid"),
    60101030: NecessityEntry(68, "Industrifastigheter",    "Lager, logistikhubbar — e-handelstillväxt"),
    60101040: NecessityEntry(64, "Köpcentrum",             "Detaljhandelsfastigheter, strukturellt utmanade"),
    60101050: NecessityEntry(66, "Hälsovårdsfastigheter",  "Sjukhus, äldreboenden, demographics-driven"),
    60101060: NecessityEntry(62, "Hotell & Resortfastigheter","Cyklisk, ej nödvändig"),
    60101070: NecessityEntry(60, "Diversifierade REIT",    "Bred exponering"),
    60102010: NecessityEntry(55, "Fastighetsutveckling",   "Projektbolag, mer cykliska"),
    60102020: NecessityEntry(52, "Fastighetstjänster",     "Mäkleri, förvaltning"),

    # ═══════════════════════════════════════════════════════════════
    # SPECIALKLASSER — ej standardiserade GICS-koder
    # Används som override via namn/tag-matching i get_necessity_score()
    # ═══════════════════════════════════════════════════════════════
    # (Lagras som negativa nycklar för att undvika GICS-kollision)
    -1:  NecessityEntry(98, "Uran / Kärnbränsle",    "Strategisk energiråvara, noll digital substitut"),
    -2:  NecessityEntry(95, "Kritiska Mineraler",     "Litium, kobolt, REE — geopolitisk förstahandsresurs"),
    -3:  NecessityEntry(93, "Råoljeproduktion",       "Global energibas"),
    -4:  NecessityEntry(92, "Koppargruvor",           "Elektrifiering, EV, elnät — absolut nödvändig"),
    -5:  NecessityEntry(90, "Guldgruvor",             "Monetary hedge, industriell efterfrågan"),
    -6:  NecessityEntry(88, "Silvergruvor",           "Solar + elektronik + monetary"),
    -7:  NecessityEntry(0,  "Kryptovaluta",           "Ingen fysisk nytta, ren spekulation"),
    -8:  NecessityEntry(2,  "SaaS / Cloudtjänst",     "Mjukvara som tjänst — ej fysisk nödvändighet"),
    -9:  NecessityEntry(4,  "Sociala Medier",         "Annonsbaserad plattform"),
    -10: NecessityEntry(6,  "Fintech / BNPL",         "Finansiell mjukvara, spekulativ"),
}

# Bekväm lookup: GICS-textsträng → score (Börsdata returnerar ibland sektorsnamn, ej kod)
SECTOR_NAME_MAP: dict[str, int] = {
    # Energi
    "energy":                   10,
    "energi":                   10,
    "oil gas":                  10102010,
    "oil & gas":                10102010,
    "olja gas":                 10102010,
    "uranium":                  -1,
    "uran":                     -1,
    "nuclear":                  55102020,
    "kärnkraft":                55102020,
    # Material
    "materials":                15,
    "material":                 15,
    "metals mining":            15104020,
    "gold":                     15104030,
    "guld":                     15104030,
    "copper":                   15104025,
    "koppar":                   15104025,
    "silver":                   15104045,
    "lithium":                  15104060,
    "litium":                   15104060,
    "steel":                    15104050,
    "stål":                     15104050,
    "aluminum":                 15104010,
    "aluminium":                15104010,
    # Utilities
    "utilities":                55,
    "allmännyttigt":            55,
    "water":                    55101010,
    "vatten":                   55101010,
    # Industri
    "industrials":              20,
    "industri":                 20,
    "defense":                  20101010,
    "försvar":                  20101010,
    # Hälsovård
    "health care":              35,
    "healthcare":               35,
    "hälsovård":                35,
    "pharmaceuticals":          35102020,
    "läkemedel":                35102020,
    # Finans
    "financials":               40,
    "finans":                   40,
    "banks":                    4010,
    "banker":                   4010,
    "insurance":                4030,
    "försäkring":               4030,
    "crypto":                   -7,
    "krypto":                   -7,
    # Dagligvaror
    "consumer staples":         30,
    "dagligvaror":              30,
    "food":                     30202010,
    "mat":                      30202010,
    # Fastigheter
    "real estate":              60,
    "fastigheter":              60,
    # Lågpoängs-sektorer
    "information technology":   45,
    "it":                       45,
    "technology":               45,
    "teknik":                   45,
    "software":                 45102010,
    "mjukvara":                 45102010,
    "saas":                     -8,
    "semiconductors":           45301010,
    "halvledare":               45301010,
    "communication services":   50,
    "kommunikation":            50,
    "media":                    5020,
    "social media":             -9,
    "sociala medier":           -9,
    "consumer discretionary":   25,
    "konsumtion":               25,
    "fintech":                  -10,
}


# ─── Fallback ────────────────────────────────────────────────────────────────

FALLBACK_SCORE = NecessityEntry(40, "Okänd Sektor", "Ingen GICS-matchning — neutral poäng")


# ─── Lookup-funktioner ───────────────────────────────────────────────────────

def get_necessity_score(
    gics_sub_industry: int | None = None,
    gics_industry: int | None = None,
    gics_industry_group: int | None = None,
    gics_sector: int | None = None,
    sector_name: str | None = None,
) -> NecessityEntry:
    """
    Returnerar NecessityEntry för ett instrument.

    Prioritetsordning: sub_industry → industry → industry_group → sector → namn → fallback.
    Accepterar GICS-koder (int) eller sektorsnamn (str).

    Börsdata-fältmapping (från get_instruments()):
        gics_sub_industry  ← instrument['branchId']      (om 8-siffrig finns)
        gics_industry      ← instrument['sectorId']
        gics_industry_group← instrument['marketId']      (approximation)
        gics_sector        ← instrument['branchId']       (2-siffrig sektor)
        sector_name        ← instrument['sector']         (textsträng)
    """
    for code in (gics_sub_industry, gics_industry, gics_industry_group, gics_sector):
        if code is not None and code in NECESSITY_MAP:
            return NECESSITY_MAP[code]

    if sector_name:
        key = sector_name.strip().lower()
        if key in SECTOR_NAME_MAP:
            mapped_code = SECTOR_NAME_MAP[key]
            if mapped_code in NECESSITY_MAP:
                return NECESSITY_MAP[mapped_code]
        # Partiell matchning
        for name_key, code in SECTOR_NAME_MAP.items():
            if name_key in key or key in name_key:
                if code in NECESSITY_MAP:
                    return NECESSITY_MAP[code]

    return FALLBACK_SCORE


def passes_threshold(entry: NecessityEntry) -> bool:
    """Returnerar True om Necessity Score >= NECESSITY_THRESHOLD (60)."""
    return entry.score >= NECESSITY_THRESHOLD


def score_from_gics(
    gics_sub_industry: int | None = None,
    gics_industry: int | None = None,
    gics_industry_group: int | None = None,
    gics_sector: int | None = None,
    sector_name: str | None = None,
) -> int:
    """Enkel wrapper som returnerar bare poängen (int) istället för NecessityEntry."""
    return get_necessity_score(
        gics_sub_industry, gics_industry, gics_industry_group, gics_sector, sector_name
    ).score


# ─── CLI-diagnostik ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print(f"\n{'─'*60}")
    print(f"  NECESSITY SCORE MAP  |  Tröskel: {NECESSITY_THRESHOLD}")
    print(f"{'─'*60}")

    test_cases = [
        # (beskrivning, kwargs)
        ("Uran (override)",        dict(gics_sub_industry=15104035)),
        ("Koppar (25-kod)",        dict(gics_sub_industry=15104025)),
        ("Guld (30-kod)",          dict(gics_sub_industry=15104030)),
        ("Silver (45-kod)",        dict(gics_sub_industry=15104045)),
        ("Olja & Gas Integrated",  dict(gics_sub_industry=10102010)),
        ("Kärnkraft (utility)",    dict(gics_sub_industry=55102020)),
        ("Vatten",                 dict(gics_sub_industry=55101010)),
        ("Läkemedel",              dict(gics_sub_industry=35102020)),
        ("Storbanker",             dict(gics_sub_industry=40101010)),
        ("Halvledare",             dict(gics_sub_industry=45301010)),
        ("SaaS (namnlookup)",      dict(sector_name="saas")),
        ("Krypto (namnlookup)",    dict(sector_name="crypto")),
        ("IT-sektor (fallback)",   dict(gics_sector=45)),
        ("Dagligvaror",            dict(gics_sector=30)),
        ("Okänd sektor",           dict()),
    ]

    for label, kwargs in test_cases:
        entry = get_necessity_score(**kwargs)
        flag = "✓ PASS" if passes_threshold(entry) else "✗ FAIL"
        print(f"  {entry.score:>3}  {flag}  {label:<35} [{entry.label}]")

    print(f"{'─'*60}\n")
