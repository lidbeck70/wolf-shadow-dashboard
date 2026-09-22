# Durrett Engine

Don Durretts 10-stegsmetod som fristående, deterministisk och förklarbar
analysmotor i Wolfpanel. Paket: `engines/durrett/`. Flik: GRANSKNING →
**🐺 Durrett**. Kontrakt: `engines/contract.py` (samma format som framtida
Rick Rule-, Graham-, Buffett/Munger-, Lynch- och Lukacs-motorer).

## Metod

| Steg | Modul | Vad | Poäng |
|---|---|---|---|
| 1 Properties | `properties.py` | Reserver (P+P) och resurser (M+I+I) hålls isär med kategori; NPV/CAPEX, MCap/EV per enhet, $/oz i marken; halt, recovery, gruvliv, infrastruktur, storlek, prospekteringsuppsida | Properties |
| 2 Management | `management.py` | Byggt/finansierat/drivit gruvor, upptäckter, exits, aktieägaravkastning (VERIFIED ×1, CLAIMED ×0,5, UNKNOWN = N/A), insynsägande, insideraffärer, styrelse, ersättning/närstående | Management |
| 3 Share structure | `dilution.py` | FD-aktier = basic + optioner + warranter + konvertibler + övrigt; utspädning 1/3/5 år; FD-börsvärde och FD-EV i USD; SERIAL DILUTER | Dilution |
| 4 Location | `jurisdiction.py` | Landrisk ur repots tabell (`contrarian_alpha.resource_scoring`, 40 %) + projektrisk ur åtta 0–2-bedömningar och statens andel (60 %) | Jurisdiction |
| 5 Growth | `growth.py` | Produktionsmultipel (framtid/nu), CAGR, resurs-/reservtillväxt 3 år, pipeline | Growth |
| 6 Market buzz | `momentum.py` | Kurs mot MA200, 6 mån, RS-rank, volym, sektor, råvara, nyheter — utanför Quality | Momentum |
| 7 Costs | `costs.py` | Producenter: pris − AISC, AISC/pris, cash cost, sustaining. Developers: IRR, NPV/CAPEX, payback, CapEx mot börsvärde | Costs |
| 7b Financing | `financing.py` | (kassa + åtagen)/CapEx, CapEx mot börsvärde, partner/offtake, funding cliff före nästa milstolpe | Financing |
| 8 Cash/debt | `balance_sheet.py` | Nettoskuld, ND/EBITDA, runway = kassa/årsburn, rörelsekapital, räntetäckning, förfall | Balance Sheet |
| 9 Valuation | `valuation.py` | A resursvärde, B reservvärde, C framtida kassaflöde = produktion × (pris − AISC) × multipel (input, default 5×, 10× alternativ), EV/NPV, MCap/framtida vinst (10×-regeln) | Valuation |
| 10 Upside | `upside.py` | Upside Multiple = framtida börsvärde / nuvarande FD-börsvärde | Upside |
| Red flags | `red_flags.py` | 25 flaggor med {flag, severity, reason, source, data, date} | Red Flag Score = 100 − avdrag |

Typprofiler: `classifier.py` (PRODUCER / DEVELOPER / EXPLORER / ROYALTY_STREAMER /
HYBRID / UNKNOWN med skäl), `developer.py` (DURRETT DEVELOPER FIT 0–6),
`explorer.py` (OPTIONALITY PLAY / DISCOVERY PLAY, Lassonde-position).
Royaltybolag tvingas inte genom kostnadsstrukturen (Costs = N/A, vikten 0).

## Poäng

Alla poäng 0–100: 0 extremt svagt, 50 neutralt, 100 extremt starkt.
**None = N/A** — aldrig 0 för okänt. `Score` bär drivrutiner (+ / − / ?) och
`explain_score(key)` skriver ut dem.

- **Quality** = viktat snitt av tio områden (`DURRETT_CONFIG["weights"]`,
  summa 100; Properties 15, Management 10, Share Structure 10, Jurisdiction 10,
  Growth 10, Costs 10, Financing 8, Balance Sheet 8, Valuation 12, Upside 7).
  Momentum ingår inte (`momentum_in_quality = 0`).
- **Risk Score** (100 = lägst risk) = viktat snitt av Dilution, Jurisdiction,
  Financing, Balance Sheet, Costs, nedjusterat av Red Flag Score. **Risk Level**
  = LOW / MODERATE / ELEVATED / HIGH ur `risk_levels`.
- Komponenter som saknas får vikten omfördelad; är under 50 % av vikten känd
  blir poängen N/A med skäl.

## Confidence

`confidence.py` integrerar med Wolfpanels Confidence score:
**Data Confidence** = `confidence.scoring.confidence_score` (komplett-
het, färskhet, källkvalitet, verifiering, konsistens, kill-caps).
**Model Confidence** = 0,7 × Data Confidence + 0,3 × scenariorobusthet
(Bear/Base-multipel) − upp till 25 för andel N/A-poäng. Confidence är
inte attraktivitet: "Durrett 88 · Upside 14× · Confidence 61" = stor
potential, betydande osäkerhet.

## Scenarier

`scenarios.py`: Bear / Base / Bull. Per scenario kan råvarupris,
produktion, AISC, CapEx, FX, multipel, recovery och gruvliv justeras
(fliken → Scenarier). Defaults i `scenario_defaults`: metallpris ur en
synlig tabell per råvara (guld 2 500/3 000/3 500, silver, koppar, uran —
platshållare att ändra), annars spot ± % och det sägs i antagandet.
Kedjan skrivs ut: rörelsekassaflöde → FCF (skatt) → EV (multipel) →
börsvärde (− nettoskuld − ofinansierad CapEx för developers) → fair
value/aktie (FD) → upside. Dagens spot används aldrig tyst som
långsiktig prognos.

## Red flags

Trösklar i `red_flag_thresholds`. En flagga sätts bara när talet finns —
saknat tal syns i `missing_data`. Flaggor: High AISC, Low IRR, Low
NPV/CAPEX, Huge CAPEX, Weak balance sheet, Short cash runway, Financing
cliff, Financing required before construction, High dilution, Serial
diluter (CRITICAL), Low insider ownership, Weak/Unverified management,
Permitting/Infrastructure/Jurisdiction/Community risk, Metallurgy risk,
High strip ratio, Low grade, Narrow veins, High royalties, High streaming
burden, Long path to production, Unrealistic economic assumptions,
Commodity price sensitivity.

## Datakällor

Inmatningar lagras i `data/confidence.json` (delat med Confidence score)
via 💾 Spara. Varje datapunkt har value, kind (ACTUAL/ESTIMATE/GUIDANCE/
MODELLED/ASSUMPTION), source, source_type, pub_date/data_date,
confidence. Motorn rör aldrig nätverket. Tre vägar fyller fälten som
FÖRSLAG (inget skrivs utan Använd):

- **Håven** (`screens_scan.py`, Durrett-screenern i Börsdata) lägger in
  bolaget med `ins_id` och börsvärde.
- **Sifferuppdateringen** (`sheets_refresh.py`, GitHub Actions) läser
  arket, hämtar färska tal ur Börsdata för rader med `ins_id` (eller
  ticker som går att slå upp) och skriver dem till Gisten under
  `confidence:<TICKER>`: kurs, valuta, börsvärde, EV, EV/EBITDA,
  ND/EBITDA, P/E, omsättning, FCF, OCF, RS-rank, EBITDA-marginal och
  jobbets FX-kurs (fast tabell, märkt ASSUMPTION). Arket visar dem via
  `engines/durrett/refresh.py` med Använd per tal eller alla; källa
  "Börsdata (sifferuppdatering ÅÅÅÅ-MM-DD)". Jobbet räknar också
  händelsen `durrett_engine_buy_rule` (MCap/framtida vinst korsar 10× med
  färskt börsvärde) med motorn själv — larmbenet "sheets" plockar upp den.
- **Copilot-extraktion** ur PDF (ark `confidence` i `ai/extract_prompt.py`)
  för tekniska rapporter; annars manuell inmatning i arket. Källhierarkin (filings > tekniska rapporter >
43-101/JORC > FS > årsrapport > … > marknadsföring) uttrycks genom
`source_type` (primary/independent/secondary/mixed/weak/unsupported).

## Valuta och enheter

Intern valuta USD. `market_currency` ≠ USD kräver `fx_to_usd` (med datum)
— annars N/A, ingen dold kurs. Mängder i `resource_unit` (oz/koz/Moz,
lb/Mlb, t/kt/Mt) räknas om till basenheten som råvarans prisenhet ger
(USD/oz → oz, USD/lb → lb, USD/t → t); enhetskonflikt loggas och ger N/A.
Kostnadsmått per råvara: AISC för guld/silver/uran, C1/AISC för koppar,
opex för litium (LCE).

## Begränsningar

- Peer-median för EV/oz finns inte i repot: trapporna är absoluta (VAL)
  och bara för USD/oz; koppar/litium/uran får N/A i resurs-/reservvärdet
  tills en trappa läggs in (TODO i config).
- Halttrösklar per råvara är platshållare (`grade_low_by_unit`).
- Historisk utspädning kräver aktieantal 1/3/5 år tillbaka.
- Momentum: fliken hämtar kurs mot MA200, 6-månadersutveckling och
  volymtrend ur yfinance på knapp (Indata → "Hämta momentum",
  `momentum_fetch.py`; volymtrend är MODELLED enligt regeln i noten).
  RS-rank, sektor-/råvarumomentum och nyhetsflöde är bedömningar.
- Model Confidence mäter inte "model agreement" mot andra motorer förrän
  multi-model-vyn finns.
- Ingen händelsedriven omräkning i bakgrunden: analysen räknas om varje
  gång fliken ritas (ny resursuppskattning → uppdatera fälten → allt
  räknas om).

## Exempel

`tests/durrett_cases.py::gold_producer` (syntetiskt): 200 koz, AISC
1 450, guld 3 000, 3,0 Moz reserv, nettokassa 80 MUSD, FD 260 M aktier,
börsvärde 1 500 MUSD.

```
DURRETT ENGINE · GPR · PRODUCER (high)
Properties 76 · Management 84 · Dilution 88 · Jurisdiction 80 · Growth 69
Momentum 75 · Costs 96 · Financing 80 · Balance Sheet 94 · Valuation 48
Upside 25 · Red Flags 97
Quality 74.7 · Risk 87.8 (LOW) · Upside 1.14× · Confidence 65.6 % (data 75.4)
Base: 300 000 oz × (3 000 − 1 450) = 465 MUSD → FCF 339 → EV × 5 = 1 697
      → börsvärde 1 777 → 6.84 USD/aktie → +14 %
Bear (2 500, AISC +10 %, 4×): −51 %   Bull (3 500, 7×): +135 %
```

Se `tests/test_durrett_engine.py` för alla fall, inklusive kantfallen.
