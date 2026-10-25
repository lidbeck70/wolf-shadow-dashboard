# Rule-köp i Norden — förregistrering (version 1, 2026-10-25)

Den här sidan skrevs och checkades in **innan testet kördes första gången**. Ändras något efter första körningen
blir det version 2 med en ny rad i ändringsloggen, och version 1:s resultat redovisas bredvid — aldrig ersatt.

Testet är forskning. Det ändrar inga regler.

## Bakgrund

Två tester (`hatets_varaktighet_spec.md`, `hatat_och_billigt_spec.md`) visade att nordiska aktier köpta för att
de är pris-hatade — även när de är undervärderade — i snitt gått sämre än marknaden på två år. Rick Rule kräver
mer än hat: **kvalitet, en stark balansräkning och en katalysator**. Det är också vad Contrarian Alphas Deep
Contrarian räknar på. Här prövas om de delarna gör skillnad. Hypotesen är ny (de villkoren har inte prövats på
datan), så hela urvalet används.

## Urval

* Per Valhall-segment de **30 största** nordiska bolagen efter föregående års börsvärde (Börsdata KPI 50, i
  kronor), ett aktieslag per bolag. Marknaden = likaviktat index av alla sektorers topp 30 det året.
* Kurser och nyckeltal från Börsdata; händelser från 2008-01-01. Ett bolag räknas bara det år det hör till urvalet.

## Definitioner

| Del | Definition (allt känt på dagen; årssiffror räknas som kända 4 månader efter årets slut) |
|---|---|
| **Hat** | Hatperiod som i varaktighetsstudien: tioårspercentil ≤ 30 och hatpoäng ≥ 60 mot marknaden; slutar efter 21 dagar utan hat. |
| **Katalysator** | Contrarian Alphas tekniska vändning: stängningen över SMA50 **och** SMA50 högre än för 10 handelsdagar sedan. Händelsedag = första dagen villkoret slår om till sant, från hatperiodens start till 63 handelsdagar efter dess slut. Högst en per hatperiod. |
| **Kvalitet** (Deep Contrarians grind) | Medianen av de senast kända upp till 4 årens ROIC (KPI 37, %) ≥ 8 (minst 2 år). |
| **Balansräkning** (Contrarian Alphas grind) | Senast kända år: FCF > 0 (KPI 63), skuldsättningsgrad < 0,6 (KPI 40, kvot), EBITDA > 0 (KPI 54) och eget kapital > 0 (KPI 58). |
| **Rule-köp** | Katalysatordag där både kvalitet och balansräkning är godkända. |
| **Utan kvalitet** | Katalysatordag där kvalitet eller balansräkning är underkänd (båda kända). |
| **Mått** | Köp på stängningen dagen efter, håll 504 handelsdagar; relativ avkastning = bolagets − marknadens samma dagar. Bara hel horisont. Saknas nyckeltal räknas händelsen inte. |

## Hypoteser

* **K1 Rule-köp slår marknaden**: snittet för Rule-köp > 0.
* **K2 Kvaliteten gör skillnad**: snittet för Rule-köp minus snittet för "utan kvalitet" > 0.
* **K3 (beskrivande)**: för samma Rule-perioder, köp vid katalysatorn mot köp vid hatets start.

Intervall: klusterbootstrap per år (5 000 dragningar, frö 20261025); för K2 samma dragna år för båda grupperna.

## Godkännande

* **K1**: minst 30 händelser från minst 5 år; 95 %-intervallets nedre gräns > 0; median > 0.
* **K2**: minst 30 händelser i varje grupp från minst 5 år; nedre gräns > 0; Rule-köpens median högre än "utan
  kvalitet".

## Vad utfallet leder till

* **K1 och K2** → förslag: långsidans nordiska köp kräver hat + kvalitet + balansräkning + katalysator (Deep
  Contrarians grindar med pris-hatet som urval). Papperskonto först. Ägarens beslut.
* **Bara K2** → kvalitetsgrindarna behålls som filter, men ingen köpsignal.
* **Inget** → Rule-metoden ger inte kanten på nordiska aktier med de här reglerna; fokus på råvaror globalt.

## Kända begränsningar

* Överlevnadsbias (Börsdatas lista har bara noterade bolag) — slår åt samma håll i K2:s båda grupper.
* Insiderköp ingår inte i katalysatorn (ingen historik i licensen).
* Årsnyckeltal med fyra månaders fördröjning; finansbolag faller ofta på skuldsättningsgraden.

## Körning

`rule_kop_study.py` i Actions (`rule-kop-study.yml`, manuell). Resultatet i loggen och i Gisten (`rule_kop.json`).

## Ändringslogg

| Version | Datum | Ändring |
|---|---|---|
| 1 | 2026-10-25 | Första versionen, före första körningen. |
