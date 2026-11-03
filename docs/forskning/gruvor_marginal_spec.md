# Gruvorna mot marginalen — förregistrering (version 1, 2026-11-03)

Den här sidan skrevs och checkades in **innan testet kördes första gången**. Ändras något efter första körningen
blir det version 2 med en ny rad i ändringsloggen, och version 1:s resultat redovisas bredvid — aldrig ersatt.

Testet är forskning. Det ändrar inga regler, trösklar, grindar eller positionsstorlekar. Marginalmätaren i panelen
är fortfarande **bara kontext** och når ingen signal, räkning eller bekräftelse.

## Bakgrund — ägarens fråga

Gummisnodden GLD/GDX (och GLD/GDXJ) står ofta sträckt: gruvorna ser historiskt billiga ut mot guldet. Ändå räknas
det inte som köpstöd när GLD självt står nära toppen av sitt eget prisspann. Då blir drivaren
`NUMERATOR_EXPENSIVE` ("kvoten är hög för att guldet är dyrt, inte för att gruvorna är billiga"). Bara
`DENOMINATOR_CHEAP` eller `BOTH`, alltså gruv-ETF:ns eget prisläge vid 20:e percentilen eller lägre, ger 🟢 KÖP-STÖD
i "Gummisnoddar: …" och motorns +1 ACCUMULATE.

Ägaren frågar: **är ett marginalbaserat mått en bättre bekräftelse?** Gruvorna tjänar (guldpris − AISC) per uns.
När guldet stiger mer än kostnaderna växer marginalen, och då kan GDX vara billigt per dollar marginal fast kursen
står högt i sitt eget spann. Mätaren "Gruvorna mot marginalen" (`alpha_regime/miner_margin.py`) visar

    M = gruv-ETF:ns kurs / (guldpris − branschens AISC)

som kontext. Här prövas om M hade varit en bättre bekräftelse än drivarens prisvillkor.

## Data

* Dagsstängningar för **GLD, GDX, GDXJ och GC=F** (guldterminen) via `market_prices` med period `"max"`, alltså den
  längsta historik som finns. I Actions kommer de från Yahoo (utdelningsjusterade, ungefär totalavkastning).
  `market_prices` provar Börsdata först när en ticker har ett Börsdata-id. Historiken börjar för GLD 2004, GDX 2006,
  GDXJ i november 2009 och GC=F 2000.
* **Branschens AISC** (USD/uns) ur `AISC_TABLE` i `alpha_regime/miner_margin.py` via
  `aisc_series(point_in_time=True)`. Ett värde används först från sitt publiceringsdatum, och dessförinnan gäller det
  senaste värde som var känt. Tabellen håller första publicerade värdet, utan revisioner. Det första värdet som var
  känt kom 2015-11-12.
* Kalender: dagar där både GLD och gruv-ETF:n har en stängning, samma inre koppling som kvotkortet. Guldet kopplas
  till gruv-ETF:n på samma sätt som mätaren gör. En dag utan guldkurs saknar M.

## Definitioner (point-in-time)

Varje handelsdag t räknas med **bara data till och med t**.

* **Fönster**: de senaste tio åren till och med t, (t − 10 år, t], som kvotkorten (tio års dagsdata).
* **Percentil**: `_percentile_of` ur `alpha_regime/commodity_ratios.py`, det vill säga andelen värden i fönstret som
  är ≤ dagens värde, med dagens värde medräknat.
* **Krav**: fönstret måste börja minst **3 år** före t. För M gäller att första giltiga M-värdet i fönstret ska ligga
  minst 3 år före t. Annars kan signalen inte räknas den dagen.
* **P_kvot**: percentilen för GLD/GDX. **P_GLD** och **P_GDX**: varje bens egen kurs i samma fönster, alltså benens
  prisläge.
* **M på dag s** = `margin_ratio(GDX_s, GC=F_s, AISC_pit(s))`, där AISC_pit(s) är den AISC som var publicerad på dag s.
  En marginal på 50 USD/uns eller mindre ger inget M. **P_M** är percentilen för M(t) bland giltiga M i fönstret.

### Signaler

Definitionerna gäller GDX. För GDXJ används GLD/GDXJ och GDXJ:s egen M.

| Signal | Villkor |
|---|---|
| **S0 nuvarande regel** | `_classify(P_kvot, "high")` = `RUBBER_BAND_STRETCHED` (P_kvot ≥ 90) **och** `_classify_driver(P_GLD, P_GDX, "high")` är `DENOMINATOR_CHEAP` eller `BOTH`, det vill säga P_GDX ≤ 20. Det är panelens egna funktioner och motorns +1. |
| **S1 bara kvoten** | P_kvot ≥ 90. Drivaren ignoreras. |
| **S2 bara marginalen** | P_M ≤ 20 |
| **S3 föreslagen ersättare** | P_kvot ≥ 90 **och** P_M ≤ 30 |

Signalerna räknas från första dagen med tre års historik. En dag där signalen inte kan räknas räknas som "ingen
signal". Utfall räknas bara för episoder som börjar **2012-01-01 eller senare**. Dagarna före det avgör bara om en
signal redan pågick.

**Täckning**: S0 och S1 behöver kvotens fönster, S2 och S3 behöver dessutom M. Eftersom den första AISC-siffran
publicerades 2015-11-12 kan S2 och S3 räknas först från omkring 2018-11-12. Det kallas *marginalperioden*.
*Kvotperioden* börjar 2012-01-01 för GDX och omkring 2012-11-11 för GDXJ, som noterades i november 2009.

## Episoder och utfall

* **Episod**: börjar första signaldagen efter minst **63 handelsdagar** utan den signalen.
* **Köp** på stängningen nästa handelsdag.
* **Utfall per episod**: gruv-ETF:ns totalavkastning och avkastningen **mot GLD** (gruv-ETF:ns avkastning minus
  GLD:s, samma dagar) över **63, 126 och 252 handelsdagar** efter köpet. Bara hela horisonter räknas. Dessutom den
  **värsta nedgången** (topp till botten) för gruv-ETF:n inom 252 dagar efter köpet, också bara med hel horisont.
* **Baslinje**: samma horisonter med köp **varje handelsdag** (ovillkorat) och **var 63:e handelsdag**. Den räknas
  för kvotperioden (S0/S1:s täckning) och för marginalperioden (S2:s täckning, där M finns), båda från och med
  2012-01-01.

## Hypoteser

Huvudprövningen är **GDX**. GDXJ prövas på samma sätt som en andra prövning, eftersom branschens AISC bara är en
approximation för juniorerna. Varje hypotes avgörs för sig per gruv-ETF.

* **H1 S3 är en bättre bekräftelse än S0**: S3 hittar fler köpstödsepisoder än S0, och snittet för S3:s 252-dagars
  avkastning mot GLD är > 0 och ≥ S0:s. S0 och S3 jämförs över S3:s täckning, alltså episoder som börjar i
  marginalperioden. Har S0 ingen episod med hel 252-dagarshorisont räknas det som "ingen signal", och då räcker
  det att S3:s snitt är > 0.
* **H2 S2 ensam slår baslinjen**: snittet för S2:s 252-dagars avkastning mot GLD är > baslinjens snitt, och
  95 %-intervallet från en **moving-block-bootstrap med cirkulära block** (block om 63 handelsdagar, 5 000
  dragningar, frö 20261103) har en nedre gräns > baslinjens snitt. Baslinjen är köp varje handelsdag i S2:s
  täckning med hel 252-dagarshorisont.
* **H3 S1 utan drivarens nedgradering**: snittet för S1:s 252-dagars avkastning mot GLD är > 0, med minst 3
  episoder.

**Bootstrapen** (H2) använder de N dagarna i S2:s täckning som har hel 252-dagarshorisont. De dras i block om 63
dagar i följd. Varje block startar likformigt bland alla N dagar och fortsätter från periodens början när det når
slutet (cirkulära block, som sektorrotationens test). Blocken läggs efter varandra tills N dagar är fyllda, och det
sista blocket kapas. Med cirkulära block har varje dag samma chans att dras. Med raka block drogs en episod på
periodens första eller sista dag nästan aldrig, och S2:s första episod kan mycket väl ligga på täckningens första
dag. I varje dragning räknas snittet av utfallet över de episodstarter som kom med. Dragningar utan någon
episodstart räknas inte, men antalet redovisas. Baslinjens snitt hålls fast.

## Godkännande

* **Golv**: alla tre hypoteserna kräver **minst 3 episoder med hel 252-dagarshorisont**. För H3 står golvet i själva
  hypotesen. För H1 och H2 gäller samma golv, eftersom en enda episod annars kan avgöra (med en episod blir
  bootstrapintervallet en punkt).
* **H1 godkänd**: S3 har fler episoder än S0 i marginalperioden, alla episoder medräknade. S3 har minst 3 hela
  episoder. S3:s snitt är > 0. Och S3:s snitt är ≥ S0:s, eller så har S0 ingen hel episod.
* **H2 godkänd**: S2 har minst 3 hela episoder. S2:s snitt är > baslinjens snitt, och 95 %-intervallets nedre gräns
  är > baslinjens snitt.
* **H3 godkänd**: S1 har minst 3 hela episoder från 2012-01-01, och S1:s snitt är > 0.

**Det här testet kan i bästa fall stödja ett förslag.** Det handlar om en enda tillgång (guld och guldgruvor; GDX
och GDXJ är två närbesläktade fonder) och några få oberoende episoder. Ett godkänt utfall bevisar ingenting.
Varje regeländring är ägarens beslut.

## Vad utfallet leder till (ägarens beslut)

Förslagen gäller per kvot: GLD/GDX följer GDX:s utfall och GLD/GDXJ följer GDXJ:s.

* **H1 håller** → förslag: för guld/gruv-kvoten ersätts drivarens prisvillkor (gruv-ETF:ns eget prisläge ≤ 20) med
  marginalvillkoret (P_M ≤ 30) för KÖP-STÖD och motorns +1.
* **H3 håller men inte H1** → förslag: drivarens nedgradering tas bort för guld/gruv-kvoten, så att kvoten ≥ 90
  räcker.
* **H1 och H3 håller** → H1:s förslag gäller, eftersom det behåller ett filter. H3 redovisas som alternativ.
* **H2 håller** → förslag: marginalmätaren blir en egen bekräftelse, och den prövas först på papper.
* **Inget håller** → regeln behålls, och marginalmätaren är kvar som kontext.
* **Bara GDXJ håller** → förslaget gäller bara GLD/GDXJ och är svagare (AISC är en approximation där).

## Kända begränsningar

* **En tillgång.** Guld och guldgruvor. GDX och GDXJ överlappar kraftigt, så de två prövningarna är inte oberoende.
* **Få oberoende episoder.** 63-dagarsgapet ger högst några episoder per år, och 252-dagarsutfallen överlappar.
* **Kort marginalhistorik.** Point-in-time finns ingen AISC före 2015-11-12. S2 och S3 kan därför räknas först från
  omkring 2018-11, och bara omkring sju år har hel 252-dagarshorisont. Under de första åren är marginalen tunn
  (guld runt 1 100 mot AISC runt 900), så M rör sig mycket.
* **AISC-revisioner och framåtblick.** Tabellen håller första tryck, men publiceringsdatumen är ungefärliga (ofta
  månadens sista dag). Ett datum som är satt för tidigt ger lite framåtblick. Några år kommer från andra serier med
  annan metod: tidiga WGC-siffror med okänd leverantör, S&P Global 2016, GFMS 2018 och Metals Focus helår 2019.
* **Panelens mätare räknar annorlunda.** Den använder varje kvartals eget värde över hela historiken, och det värdet
  blev känt först senare. Testet använder det värde som var känt då, så nivåerna skiljer sig något.
* **Överlevnadsbias i ETF:ernas sammansättning.** GDX och GDXJ byter innehav, och dagens fonder består inte av samma
  bolag som 2012. Indexreglerna har ändrats; GDXJ breddades till exempel 2017.
* **Branschens AISC som approximation för juniorerna (GDXJ).** Juniorbolagens kostnader är ofta högre, och deras
  egna siffror finns inte i tabellen.
* **Justerade kurser.** Yahoos utdelningsjusterade kurser räknas om bakåt vid varje utdelning, så nivåerna och
  percentilerna bakåt ändras lite.
* **GC=F** är närmaste terminen (den rullas), inte spotpriset. Skillnaden är liten mot marginalen.
* **Trösklarna är valda i förväg.** 20 och 30 är satta före körningen och inte optimerade. Fler varianter prövas
  inte.

## Körning

`miner_margin_study.py` i Actions (`miner-margin-study.yml`, manuell). Resultatet hamnar i loggen och i Gisten
(`miner_margin_study.json`). Skriptet avslutas med 0 även när hypoteserna faller. Det avslutas med annat än 0 bara
när kurser saknas, och då sparas inget.

## Ändringslogg

| Version | Datum | Ändring |
|---|---|---|
| 1 | 2026-11-03 | Första versionen, före första körningen. |
