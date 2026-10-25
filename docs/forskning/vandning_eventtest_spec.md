# Eventtest nivå 1 — förregistrering (version 1, 2026-10-25)

Den här sidan skrevs och checkades in **innan testet kördes första gången**. Hypoteser, parametrar och
godkännandekriterier är låsta. Ändras något efter första körningen blir det version 2 med en ny rad i
ändringsloggen längst ned, och resultatet från version 1 ska redovisas bredvid — aldrig ersättas.

Testet är forskning. Det ändrar inga regler, inga papperskonton och ingen strategi. Utfallet avgör bara om
nivå 2 (aktier per komplex) körs.

## Frågan

Ger en **bekräftad cykelvändning** i Valhalls cykelmotor (`valhall/cycle.py`, oförändrad) högre
avkastning framåt på sektor- och råvaruserien än en slumpvis vald dag på samma serie och i samma period?
Och ger den mer än en enkel trendregel?

## Serier

| Grupp | Serier | Vändning |
|---|---|---|
| Segment | De 11 SPDR-fonderna i `valhall/segments.py` (XLE, XLB, XLI, XLF, XLRE, XLK, XLV, XLC, XLY, XLP, XLU). Cykelläget mäts mot SPY, som i kartan. | RÖD → GRÖN som hållit `TURN_CONFIRM_DAYS` = 10 handelsdagar. Händelsedag = den tionde GRÖNA dagen (= `turn_state` → `confirmed_since`). |
| Råvara | Råvarutemana under segmenten med en drivare (`berserk.backtest.pick_driver`, start 2008-01-01 — samma drivare som fas 3). Teman som delar drivarsymbol räknas en gång. | Första dagen med `confirmed` i varje period där fasen är BAISSE · VÄNDER (= `turn_state` → `confirmed_since`). |
| Komplex | De sex komplexkorgarna (`valhall/complexes.basket`) | Som råvara. **Bara beskrivande** — korgarna består av samma drivare som råvarugruppen. |

Uteslutna på nivå 1: nordiska sektorindex och producentkorgar. Båda byggs ur dagens universum
(överlevnadsbias), så de hör hemma på nivå 2 där den biasen redovisas.

## Jämförelsesignal (trendregel)

Första dagen stängningen ligger över både EMA50 och SMA200 efter en dag då den inte gjorde det. Samma
serier, samma rensning och samma mått som vändningen.

## Låsta parametrar

| Parameter | Värde |
|---|---|
| Horisonter | 21, 63 och 126 handelsdagar. **63 är huvudhorisonten** — bara den avgör. |
| Inträde | Stängningen dagen efter händelsedagen (signalen syns först vid stängning). |
| Avkastning | `close[t+1+h] / close[t+1] − 1`. Händelser utan hel horisont räknas inte för den horisonten. |
| Rensning | En händelse stryks om samma signal gav en händelse på samma serie inom de 63 föregående handelsdagarna. |
| Perioder | 2008-01-01–2020-12-31 och 2021-01-01–idag, efter händelsedag. |
| Baslinje | Medelavkastningen över alla dagar i samma serie och period (samma inträde och horisont). Överavkastning = händelsens avkastning − baslinjen. |
| Konfidensintervall | Bootstrap över händelserna, 5 000 dragningar, 95 % (percentil). |
| Slumptest | Per serie och period dras lika många slumpdagar som det finns händelser (utan återläggning). 2 000 omgångar. p = (1 + antal omgångar med minst lika hög sammanvägd överavkastning) / (1 + 2 000). |
| Slumpfrö | 20261025 |
| Kostnader | Inga på nivå 1. En tur och retur i en sektor-ETF kostar runt 0,1–0,3 %, litet mot en 63-dagarshorisont. |

## Hypoteser och godkännande

Varje hypotes prövas på den sammanvägda gruppen (alla serier i gruppen, varje händelse väger lika).

* **H1 — segmentvändning.** Snitt-överavkastning på 63 dagar efter segmentvändning > 0.
* **H2 — råvaruvändning.** Snitt-överavkastning på 63 dagar efter bekräftad råvaruvändning > 0.
* **H3 — vändning slår trend.** Inom samma grupp: snitt-överavkastningen efter vändning minus snittet efter
  trendregeln > 0 (bootstrap av skillnaden, oberoende dragning i de två grupperna).

En hypotes är **godkänd** bara om allt detta gäller i **båda** perioderna:

1. minst 30 händelser i gruppen och perioden,
2. 95 %-intervallets nedre gräns > 0,
3. slumptestets p < 0,05 (gäller H1 och H2; H3 har bara intervallet).

Allt annat är **ej godkänt**: "ingen bevisad kant". Det finns ingen mellannivå och inget "nästan".

Siffror per serie, för 21 och 126 dagar och för komplexen redovisas, men de **avgör ingenting**: med elva
segment och en handfull vändningar per serie är enskilda rutor brus.

## Vad utfallet leder till

* H1 eller H2 godkänd → nivå 2 körs för den gruppen: Viking Nine-backtestet per komplex
  (`complex_backtest.py`, redan i fas 3b) med kostnadsmodellen. Fortfarande ingen regeländring utan ägarens beslut.
* Varken H1 eller H2 godkänd → nivå 2 körs inte. Vändningen behålls som information i kartan, men
  ingen ny regel eller vikt byggs på den.
* H3 avgör om vändningen är värd mer än en trendregel. Ej godkänd H3 med godkänd H1/H2 betyder: kanten finns,
  men en enklare regel ger den lika bra.

## Kända begränsningar

* Få cykler: en sektor vänder kanske 3–8 gånger på 18 år. Därför sammanvägning och 63 dagars rensning.
* Händelser inom samma grupp är inte oberoende (sektorer vänder ofta samtidigt). Bootstrap över händelser
  underskattar då osäkerheten något; slumptestet drar per serie och tar delvis hänsyn till det.
* Råvaruterminerna hos Yahoo (`=F`) är rullade utan justering. Rullhopp kan ge falska rörelser runt
  förfallodagar.
* XLRE (2015) och XLC (2018) har kort historik och bidrar mest i perioden 2021–.
* Utdelningar ingår inte i stängningskursen från alla källor. Det sänker nivån men slår lika på händelse och
  baslinje.

## Körning

`valhall_event_study.py` i Actions (`valhall-event-study.yml`, manuell). Resultatet i loggen och i Gisten
(`valhall_event_study.json`), med specens version, parametrarna och datum.

## Ändringslogg

| Version | Datum | Ändring |
|---|---|---|
| 1 | 2026-10-25 | Första versionen, före första körningen. |
