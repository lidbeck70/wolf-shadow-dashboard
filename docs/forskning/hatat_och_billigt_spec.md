# Hatat och billigt — förregistrering (version 1, 2026-10-25)

Den här sidan skrevs och checkades in **innan testet kördes första gången**. Ändras något efter första körningen
blir det version 2 med en ny rad i ändringsloggen, och version 1:s resultat redovisas bredvid — aldrig ersatt.

Testet är forskning. Det ändrar inga regler.

## Bakgrund och varför ett nytt urval

Studien "Hatets varaktighet" (`hatets_varaktighet_spec.md`, körd 2026-10-25) visade beskrivande, på Nordens
topp 10 per sektor: köp vid hatets start gav −7,8 % mot marknaden på två år, men bara −0,8 % när bolaget
**också var undervärderat** — ungefär 7 %-enheter bättre. Det fyndet kom ur datan; att pröva det igen på
samma bolag bevisar ingenting. Därför prövas det här på **bolag som studien aldrig rörde**: rank 11–30 per
sektor. Topp 10 redovisas bredvid, märkt "redan sett".

## Urval

* **Prövning (ej sedd)**: per Valhall-segment bolagen på **rank 11–30** efter föregående års börsvärde
  (Börsdata KPI 50, årssnitt, i kronor), ett aktieslag per bolag (`valhall.picks.company_key`).
  Marknaden = likaviktat index av alla sektorers rank 11–30 det året.
* **Redan sett**: topp 10 med marknaden från förra studien (beskrivande).
* Kurser från Börsdata; perioder som börjar från 2008-01-01. Ett bolags hatperiod räknas bara om bolaget hör
  till urvalet det år perioden börjar.

## Definitioner (oförändrade från förra studien)

* **Hatad**: tioårspercentil ≤ 30 och hatpoäng ≥ 60 mot urvalets marknad (`valhall/cycle.py`).
* **Hatperiod**: börjar första hatade dagen, slutar efter 21 handelsdagar utan hat.
* **Undervärderad vid starten**: EV/EBIT (bara positiva) eller P/B bland de 30 % lägsta i bolagets egen historik
  (minst 5 år), årssiffran känd 4 månader efter årets slut. Perioder utan värdering räknas inte.
* **Mått**: köp på stängningen dagen efter starten, håll 504 handelsdagar; relativ avkastning = bolagets −
  marknadens samma dagar. Bara perioder med hel horisont.

## Hypoteser (på rank 11–30)

* **V1 Billigt slår dyrt bland de hatade**: snittet för "hatad och undervärderad" minus snittet för "hatad men
  inte undervärderad" > 0.
* **V2 Hatat och billigt slår marknaden**: snittet för "hatad och undervärderad" > 0.

Intervall: klusterbootstrap per startår, 5 000 dragningar, frö 20261025. För V1 dras samma år för båda grupperna.

## Godkännande

* **V1 godkänd**: minst 30 perioder i varje grupp, från minst 5 startår; 95 %-intervallets nedre gräns > 0;
  och medianen för "undervärderad" högre än medianen för "inte undervärderad".
* **V2 godkänd**: minst 30 perioder från minst 5 startår; nedre gräns > 0; median > 0.

## Vad utfallet leder till

* **V1 och V2** → förslag: långsidans köp kräver både hat och undervärdering (som Rick Rule), och kan läggas vid
  hatets start. Ägarens beslut.
* **Bara V1** → förslag: värderingen som filter (undvik hatat men dyrt), men ingen köpsignal. Ägarens beslut.
* **Inget** → fyndet i topp 10 räknas som slump eller som något som bara gäller storbolagen.

## Kända begränsningar

* Överlevnadsbias: Börsdatas lista har bara dagens noterade bolag. Mellanstora bolag avnoteras oftare än stora,
  så biasen är större här — men den slår åt samma håll för båda grupperna i V1, och marknadsindexet byggs ur samma
  urval.
* Rank 11–30 är tunt i små sektorer (färre bolag med värderingshistorik).
* Årsvärdering med fyra månaders fördröjning.

## Körning

`hate_value_study.py` i Actions (`hate-value-study.yml`, manuell). Resultatet i loggen och i Gisten
(`hate_value.json`).

## Ändringslogg

| Version | Datum | Ändring |
|---|---|---|
| 1 | 2026-10-25 | Första versionen, före första körningen. |
