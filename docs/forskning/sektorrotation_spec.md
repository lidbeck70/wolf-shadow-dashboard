# Sektorrotation — förregistrering (version 1, 2026-10-25)

Underlag: "Sector Rotation Engine — Nordic Alpha" (ägarens dokument). Den här sidan låser dess startmodell
för ett historiskt test **innan testet körs första gången**. Ändras något efter första körningen blir det
version 2 med en ny rad i ändringsloggen, och version 1:s resultat redovisas bredvid — aldrig ersatt.

Testet är forskning. Det ändrar inga regler. Om modellen blir sektorgrind för swing är ägarens beslut efter
utfallet.

## Universum och data

* De 11 SPDR-sektorfonderna i `valhall/segments.py` (XLE, XLB, XLI, XLF, XLRE, XLK, XLV, XLC, XLY, XLP, XLU).
  En fond är med från den månad den har hela underlaget (12 månaders historik och 10-månaders snitt);
  XLRE (2015) och XLC (2018) kommer alltså in sent.
* Jämförelseindex: SPY.
* Dagskurser via `market_prices` (Yahoo, utdelningsjusterade = ungefär totalavkastning).
* Nordiska branschindex är **uteslutna** ur testet: Valhalls index byggs ur dagens bolag (överlevnadsbias).
  De visas i fliken som information.

## Modellen (dokumentets startmodell, låst)

Beräknas vid varje månadsskifte på månadens sista stängning — bara data fram till den dagen.

| Del | Regel |
|---|---|
| Relativ styrka | RS_k = (1 + sektorns avkastning k månader) / (1 + SPY:s avkastning k månader) − 1, k = 3, 6, 12. Senaste månaden **ingår** (branschmomentum saknar den kortsiktiga rekyl som enskilda aktier har). |
| Rankning | Percentil 0–100 per horisont bland de fonder som är med den månaden. |
| Poäng | S = 0,40 × R12 + 0,35 × R6 + 0,25 × R3 |
| Trendfilter 1 | Stängningen över 10-månaders medelvärdet (månadsstängningar). |
| Trendfilter 2 | 10-månaders medelvärdet högre än för en månad sedan. |
| Trendfilter 3 | Relativa linjen (sektor / SPY) över sitt 6-månaders medelvärde. |

Faser (för fliken; avgör inte testet), prövas i den här ordningen:

| Fas | Villkor |
|---|---|
| 4 Förlorat ledarskap | filter 1 och filter 3 brutna |
| 3 Mognande ledare | S ≥ 60 och (R3 ≤ 40 eller tappat minst 3 placeringar på 3 månader) |
| 2 Bekräftad ledare | S ≥ 60 och alla tre filter |
| 1 Tidig rotation | (R3 ≥ 60 och R12 ≤ 40) eller (klättrat minst 3 placeringar på 3 månader och R3 ≥ 50) |
| Neutral | övrigt |

## Strategin som prövas

**Rotation topp 3:** vid varje ombalansering ägs de 3 fonder med högst S **bland dem som klarar alla tre
trendfilter**, lika viktade. Klarar färre än 3 filtren ligger resten i kassa (0 % avkastning).

* Signal på månadens sista stängning; handel på **stängningen nästa handelsdag**.
* Kostnad: 45 bp per sida av omsättningen (`backtest_costs`: utländskt papper, 90 bp tur och retur
  inklusive växling) — samma för alla jämförelser.

## Jämförelser (dokumentets krav)

| Namn | Regel |
|---|---|
| SPY | köp och behåll |
| Likaviktade sektorer | alla fonder som är med, lika viktade, ombalanseras månadsvis |
| Ren 12-månaders momentum | topp 3 efter RS_12, inga filter |
| Rent trendfilter | lika vikt i alla fonder över sitt 10-månaders medelvärde, resten kassa |

## Perioder

Från första månaden med minst 8 fonder och hela underlaget (omkring 2000) till i dag. Uppdelning:
**2000–2014** och **2015–** (efter dokumentets tillkomst finns ingen anpassad del — båda perioderna är
utanför urvalet, men den senare är den moderna marknaden).

## Mått

Årstakt (CAGR) netto, maximal nedgång (månadsvis), volatilitet, Sharpe (årsbasis, räntefri = 0), omsättning per
år, andel tid investerad. Aktiv avkastning = strategins månadsavkastning − SPY:s; dess årssnitt med
**blockbootstrap** (block om 12 månader, cirkulär, 5 000 dragningar, frö 20261025) ger ett 95 %-intervall.

## Godkännande

* **R1 Slår marknaden:** CAGR netto högre än både SPY och likaviktade sektorer, i **båda** perioderna.
* **R2 Bättre risk:** Sharpe högre än SPY i båda perioderna, och maximal nedgång högst SPY:s över hela tiden.
* **R3 Värd sin komplexitet:** CAGR netto minst lika hög som ren 12-månaders momentum och som rent trendfilter
  i perioden 2015–.
* **S Statistiskt säkerställd:** 95 %-intervallets nedre gräns för årssnittet av aktiv avkastning mot SPY > 0
  över hela tiden.

Utfall:

* **R1 och R2 godkända** → modellen föreslås som sektorgrind för swing (fas 2 = handla, fas 4 = inte).
  Ägarens beslut.
* **Bara R2** → modellen föreslås som riskfilter (minska vid fas 4), inte som avkastningsmotor.
* **Annars** → fliken visas som information, märkt "ej validerad". Ingen grind.
* R3 och S redovisas alltid; S krävs inte (med elva fonder och 25 år är intervallen breda), men ett utfall
  utan S kallas aldrig "bevisat".

Robusthet (beskrivande, avgör inget): lika vikter (1/3 var), 12–1 (senaste månaden utesluten), topp 2 och
topp 4, 200-dagars medelvärde i stället för 10 månader.

## Kända begränsningar

* Bara 11 fonder (8–9 de första 15 åren) — rankningen är grov.
* Kassa ger 0 %; i perioder med hög ränta underskattas strategier som ofta ligger i kassa.
* Yahoos utdelningsjustering kan avvika marginellt från verklig totalavkastning.
* Kostnaden antar handel i USA-noterade fonder från ett svenskt konto (växling ingår). UCITS-fonder kostar
  annorlunda.

## Körning

`sector_rotation_backtest.py` i Actions (`sector-rotation-backtest.yml`, manuell). Resultatet i loggen och i
Gisten (`sector_rotation_backtest.json`).

## Ändringslogg

| Version | Datum | Ändring |
|---|---|---|
| 1 | 2026-10-25 | Första versionen, före första körningen. |
