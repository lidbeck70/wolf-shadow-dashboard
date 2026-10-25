# Långsidans cykeltest — förregistrering (version 1, 2026-10-25)

Den här sidan skrevs och checkades in **innan testet kördes första gången**. Inträden, serier, parametrar
och godkännandekriterier är låsta. Ändras något efter första körningen blir det version 2 med en ny rad i
ändringsloggen, och version 1:s resultat redovisas bredvid — aldrig ersatt.

Bakgrund: eventtestet nivå 1 (`vandning_eventtest_spec.md`) mätte 21–126 dagar efter bekräftad vändning
och fann ingen kant. Det mätte swinghorisonten, inte långsidans tes. Det här testet prövar långsidans tes
med långsidans horisont: köp en sektor nära botten, hatad, och håll i ett till tre år.

Testet är forskning. Det ändrar inga regler, papperskonton eller strategier.

## Frågan

Ger ett inträde nära cykelbotten i en hatad sektor högre avkastning över 1–3 år än en slumpvis dag på samma
serie? Och spelar det roll om man väntar på en liten trend upp eller på dagens bekräftade vändning?

## Tre inträden

Alla räknas dag för dag ur cykelmotorn (`valhall/cycle.py`, oförändrad) — bara data fram till dagen.

| Inträde | Villkor | Händelsedag |
|---|---|---|
| **E1 billig och hatad** | tioårspercentilen ≤ 30 (cykelläge TIDIG) och hatpoängen ≥ 60 (Blindspots gräns för hatad) | första dagen villkoret gäller efter en dag det inte gjorde |
| **E2 billig och hatad + liten trend upp** | E1:s villkor har gällt någon dag de senaste 126 handelsdagarna, och drivarens vändning (`berserk.signals.driver_frame` → `turn`: EMA20 stigande och över EMA50) gäller i dag | första dagen `turn` gäller efter en dag den inte gjorde |
| **E3 bekräftad vändning** | dagens regel, som i nivå 1: sektorfonder RÖD → GRÖN i 10 dagar; producenter och råvaror första bekräftade dagen i BAISSE · VÄNDER | som i nivå 1 |

Rensning: en händelse stryks om samma inträde gav en händelse på samma serie inom de 252 föregående
handelsdagarna.

## Serier och grupper

| Grupp | Serier | Roll |
|---|---|---|
| **producent** | Producent-ETF:er (Yahoo): XOP, XES, FCG, XME, GDX, GDXJ, SIL, COPX, URA, PICK, SLX, WOOD, MOO, REMX, LIT. Cykelläget mäts mot SPY. | **avgör** |
| **sektor** | De 11 SPDR-fonderna i `valhall/segments.py`, mot SPY som i kartan. | **avgör** |
| **bolag** | Per sektor de **5 största nordiska bolagen efter börsvärde föregående år** (Börsdata KPI 50, årssnitt, omräknat till kronor), likaviktade. Händelserna är sektorfondens (samma sektor). Ett bolag som avnoteras under horisonten räknas med sin sista kurs. | avgör **bara** om Börsdata har med avnoterade bolag (se nedan); annars beskrivande |
| **råvara** | Råvarudrivarna som i nivå 1 (terminer/ETC). | beskrivande |

ETF:erna ägde de bolag som fanns vid varje tidpunkt — ingen överlevnadsbias. Bolagsgruppen väljs också
utan facit (föregående års börsvärde), men om Börsdatas lista bara innehåller dagens noterade bolag väljs
de 5 största bara bland överlevarna. Regeln, bestämd i förväg: **bolagsgruppen avgör om minst 20 nordiska
instrument i Börsdatas lista har senaste kurs äldre än 365 dagar** (= avnoterade finns med). Annars märks
den "överlevnadsbias — beskrivande".

## Låsta parametrar

| Parameter | Värde |
|---|---|
| Horisonter | 252, 504 och 756 handelsdagar (1, 2, 3 år). **504 är huvudhorisonten.** |
| Inträde | Stängningen dagen efter händelsedagen. |
| Avkastning | `close[t+1+h] / close[t+1] − 1`. Händelser utan hel horisont räknas inte för den horisonten. |
| Period | Händelser från 2008-01-01. Uppdelningen 2008–2014 / 2015– redovisas men avgör inte (för få hela horisonter efter 2021). |
| Baslinje | Medelavkastningen över alla dagar i samma serie och period (samma inträde och horisont). Överavkastning = händelsens avkastning − baslinjen. |
| Konfidensintervall | **Klusterbootstrap per kalenderår för händelsen**, 5 000 dragningar, 95 %. Bottnar sammanfaller mellan serier (2009, 2016, 2020) — att dra hela år i stället för enskilda händelser tar hänsyn till det. |
| Slumpfrö | 20261025 |
| Kostnader | Inga. En tur och retur i en ETF eller storbolag är under 1 % mot en tvåårshorisont. |

Extra mått (beskrivande): median, andel händelser med mer än +50 % och +100 % (mot samma andel bland alla
dagar), och medianen av största nedgången från inträdet inom horisonten (hur djupt det föll först).

## Hypoteser

Per avgörande grupp (producent, sektor och — om villkoret ovan är uppfyllt — bolag):

* **L1** E1 billig och hatad ger överavkastning på 504 dagar.
* **L2** E2 billig och hatad + liten trend ger överavkastning på 504 dagar.
* **L3** E3 bekräftad vändning ger överavkastning på 504 dagar.
* **L4** E2 slår E1 (att vänta på trenden lönar sig).
* **L5** E2 slår E3 (den lilla trenden är bättre än dagens bekräftelse).

L4 och L5 är parvisa skillnader i snitt-överavkastning, klusterbootstrap med samma dragna år för båda.

## Godkännande

L1–L3 är **godkända** bara om allt detta gäller:

1. minst 20 händelser, från minst 5 olika kalenderår,
2. 95 %-intervallets nedre gräns för snitt-överavkastningen på 504 dagar > 0,
3. medianens överavkastning på 504 dagar > 0,
4. snitt-överavkastningen på 756 dagar > 0 (samma riktning på längre sikt).

L4–L5 är godkända om intervallets nedre gräns för skillnaden > 0 och båda inträdena har minst 20 händelser.

Allt annat är **ej godkänt**: "ingen bevisad kant". Med upp till 15 hypoteser kan en enstaka godkänd vara
tur; ett utfall räknas som **starkt** först när samma inträde godkänns i minst två avgörande grupper.

## Vad utfallet leder till

* Ett inträde starkt godkänt → det inträdet föreslås som cykelgrind för långsidan (Deep Contrarian,
  Contrarian Alpha, Long Screener) och prövas i backtest på bolagsnivå. Ingen regel ändras utan ägarens beslut.
* L4/L5 godkänd → vänta på den lilla trenden i stället för att köpa direkt / i stället för bekräftelsen.
* Inget godkänt → långsidans tes är inte bevisad med de här serierna. Metoden behålls som hypotes;
  inga vikter byggs på cykelmotorn.

## Kända begränsningar

* Få oberoende cykler: 18 år och tvåårshorisont ger 1–3 cykler per serie. Klusterbootstrapen gör
  intervallen breda — testet kan visa riktning men sällan bevisa.
* Producent-ETF:erna startar 2006–2010; tioårspercentilen kräver två års historik, så de första
  händelserna kommer 2008–2012.
* Börsdatas kurshistorik är högst 20 år; börsvärdet per år högst 20 år.
* Utdelningar ingår inte alltid i kurserna; det slår lika på händelse och baslinje.
* Bolagsgruppen mäter nordiska bolag mot den globala sektorns cykel (SPDR), inte mot ett nordiskt sektorindex
  (de byggs ur dagens universum).

## Körning

`valhall_long_study.py` i Actions (`valhall-long-study.yml`, manuell). Resultatet i loggen och i Gisten
(`valhall_long_study.json`), med specens version, parametrarna, datum och utfallet av avnoteringskontrollen.

## Ändringslogg

| Version | Datum | Ändring |
|---|---|---|
| 1 | 2026-10-25 | Första versionen, före första körningen. |
