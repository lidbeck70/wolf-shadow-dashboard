# Hatets varaktighet — förregistrering (version 1, 2026-10-25)

Den här sidan skrevs och checkades in **innan testet kördes första gången**. Definitioner, parametrar och
kriterier är låsta. Ändras något efter första körningen blir det version 2 med en ny rad i ändringsloggen,
och version 1:s resultat redovisas bredvid — aldrig ersatt.

Testet är forskning. Det ändrar inga regler.

## Frågorna

1. **D1** Hur länge är sektorns storbolag hatade (och undervärderade), per sektor i Norden och för råvarorna?
2. **D2** Lönar det sig att vänta: köp vid hatets start, efter 6 eller efter 12 månader?
3. **D3** Är andelen hatade storbolag i en nordisk sektor (hat-bredden) en användbar rotationssignal?

## Urval

| Grupp | Serier | Jämförelse ("marknaden") |
|---|---|---|
| **Norden** | Per Valhall-segment de **10 största** nordiska bolagen efter föregående års börsvärde (Börsdata KPI 50, årssnitt, i kronor), ett aktieslag per bolag (`valhall.picks.company_key`). Variant: de 5 största. Kurser från Börsdata (högst 20 år). | **Nordens storbolag**: likaviktat index av alla sektorers topp 10 det året (dagliga avkastningar, kedjade). |
| **Råvaror** | De 15 producent-ETF:erna (som långsidans test) och råvarudrivarna (`pick_driver`, en symbol en gång). | SPY |

Ett nordiskt bolags hatperiod räknas bara om bolaget hör till sektorns topp 10 (resp. 5) det år perioden
börjar. Händelser och perioder från 2008-01-01.

Överlevnadsbias: Börsdatas lista har bara dagens noterade bolag (avnoteringskontrollen 2026-10-25: 0 av 1 600).
Bolagen väljs utan facit bland överlevarna, och jämförelseindexet byggs ur samma urval, så biasen tar delvis ut
sig själv. Siffrorna för Norden är ändå en övre gräns. Råvarugruppen har ingen sådan bias.

## Definitioner

| Begrepp | Definition |
|---|---|
| **Hatad** (dag för dag) | Cykelmotorn (`valhall/cycle.py`, oförändrad) på bolagets kurs mot marknaden: tioårspercentil ≤ 30 och hatpoäng ≥ 60 — samma villkor som långsidans E1. Nordiska bolag mäts mot Nordens storbolag (percentil och relativ styrka); producenter mot SPY; råvaror absolut (relativ styrka mot SPY). |
| **Undervärderad** (år för år) | EV/EBIT (KPI 10, bara positiva värden) eller P/B (KPI 4) bland de 30 % lägsta i bolagets egen historik fram till året (minst 5 år). En årssiffra räknas som känd 4 månader efter årets slut. Bara Norden. |
| **Hatperiod** | Börjar första hatade dagen; slutar när bolaget inte varit hatat på 21 handelsdagar i rad (slutdag = sista hatade dagen). Pågår den vid datans slut är den **censurerad**. Perioder som börjar första dagen villkoret alls kan räknas stryks (okänd start). |
| **Längd** | Handelsdagar från start till slut; månad = 21 handelsdagar. |

## D1 — varaktighet (beskrivande)

Per nordisk sektor och för råvarugrupperna: antal perioder (avslutade och pågående), **Kaplan–Meier-median**
och kvartiler för längden (tar hänsyn till pågående perioder), andel som fortfarande är hatade efter 12 och
24 månader, och andel perioder som var undervärderade vid starten. Inget godkännande — det här är kartan.

## D2 — lönar det sig att vänta?

För varje hatperiod (start t0): köp på stängningen dagen efter t0 + fördröjning, fördröjning 0, 126 och
252 handelsdagar (0, 6, 12 månader) **oavsett om hatet finns kvar** — annars väljer urvalet förlorarna.
Varje köp hålls 504 handelsdagar (2 år). Relativ avkastning = bolagets avkastning − marknadens samma dagar.
Bara perioder där alla tre köpen har hel horisont räknas.

Parvisa skillnader per period: **V6 = rel(6 mån) − rel(0)** och **V12 = rel(12 mån) − rel(0)**.
Intervall: klusterbootstrap per startår (5 000 dragningar, frö 20261025).

Utfall per grupp (Norden topp 10; råvaror) och skillnad:

* **"Vänta lönar sig"**: minst 30 perioder från minst 5 startår, 95 %-intervallets nedre gräns > 0 och median > 0.
* **"Köp direkt"**: intervallets övre gräns < 0 och median < 0.
* Annars **"ingen skillnad bevisad"**.

Undergruppen "hatad och undervärderad vid starten" redovisas beskrivande.

## D3 — hat-bredd som rotationssignal (Norden)

Hat-bredd = andelen av sektorns topp 10 (det årets) som är hatade, när minst 5 har ett läge.

* **B1 massivt hatad**: första dagen bredden ≥ 50 % efter en dag under 50 %. Rensning: ingen ny B1 inom 252 dagar.
* **B2 hatet släpper**: första dagen bredden ≤ 30 % efter en B1, inom 756 dagar från den B1:an (högst en per B1).

Mått: sektorkorgen (likaviktad topp 10 det året, `valhall.long_study.basket_forward`) mot alla sektorers
storbolag, 252 och 504 dagar från stängningen dagen efter. Överavkastning = den relativa avkastningen minus
samma serie snitt över alla dagar sedan 2008 (= mot en slumpdag). **504 dagar avgör.**

B1 respektive B2 är **godkänd** om: minst 15 händelser från minst 5 år, klusterbootstrapens (per år) nedre gräns
> 0 och median > 0.

## Variant

Allt räknas också med topp 5 (beskrivande; avgör inget).

## Vad utfallet leder till

* D1 ger kolumnen "hatad sedan X mån · typiskt Y mån för sektorn" i Sektorrotation-fliken och på Kartan
  (information).
* D2 "vänta" eller "köp direkt" godkänd → förslag på när långsidans första köp läggs. Ägarens beslut.
* B1 eller B2 godkänd → förslag på en nordisk sektorsignal (bevaka vid B1, köp vid B2 …). Ägarens beslut.
* Inget godkänt → varaktigheten visas som information; ingen tajmingsregel.

## Kända begränsningar

* Överlevnadsbias i Norden (se ovan).
* Börsdatas kurser och nyckeltal räcker högst 20 år; percentilen kräver två års historik, så Nordens perioder
  börjar runt 2008–2010.
* Bolag i flera valutor jämförs i lokal valuta; marknadsindexet blandar valutor på samma sätt.
* Årsvärdering med fyra månaders fördröjning är grov — "undervärderad" byter läge en gång per år.
* Perioderna klumpar sig i tiden (2008, 2015–16, 2020, 2022); därför klusterbootstrap per år.

## Körning

`hate_duration_study.py` i Actions (`hate-duration-study.yml`, manuell). Resultatet i loggen och i Gisten
(`hate_duration.json`), med dagens hatade storbolag och råvaror (sedan när) för fliken.

## Ändringslogg

| Version | Datum | Ändring |
|---|---|---|
| 1 | 2026-10-25 | Första versionen, före första körningen. |
