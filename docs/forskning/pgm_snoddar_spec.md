# PGM-snoddarna — förregistrering (version 1, 2026-11-04)

Den här sidan skrevs och checkades in **innan testet kördes första gången**. Ändras något efter första körningen
blir det version 2 med en ny rad i ändringsloggen, och version 1:s resultat redovisas bredvid — aldrig ersatt.

Testet är forskning. Det ändrar inga regler, trösklar, grindar eller positionsstorlekar. Snoddarna Gold/Platinum och
Gold/Palladium går genom samma regel som i dag tills ägaren beslutar något annat.

## Bakgrund — ägarens fråga

Två nya gummisnoddar kom in i #300:

* **Gold/Platinum** (`gold_platinum`): GC=F/PL=F, med GLD/PPLT som reserv.
* **Gold/Palladium** (`gold_palladium`): GC=F/PA=F, med GLD/PALL som reserv.

Båda har riktningen `"high"`: en hög kvot betyder att metallen är billig mot guldet. De går genom den generiska
drivarregeln i `alpha_regime/buy_support.py`, samma som Gold/Silver. Köpstöd, alltså KÖP-STÖD i "Gummisnoddar: …"
och motorns +1 ACCUMULATE, kräver två saker. Kvoten ska vara `RUBBER_BAND_STRETCHED`, det vill säga på 90:e
percentilen eller högre. Och drivaren ska vara `DENOMINATOR_CHEAP` eller `BOTH` (`ENGINE_DRIVERS`), det vill säga
metallterminens eget prisläge ska stå på 20:e percentilen eller lägre.

Ägaren frågar: **hade köpstödet fungerat?** Platinan föll och stod sedan länge och stampade, ungefär 2015–2024,
medan guldet steg. Kvoten guld/platina stod därför högt under lång tid. Risken är att köpstödet slår till för
tidigt, medan metallen fortfarande faller: en fallande kniv. Palladium steg till rekordnivåer kring 2021–2022 och
föll sedan kraftigt, så samma fråga gäller där.

Därför prövas också en variant med ett trendfilter. Där räcker det inte att kvoten är sträckt. Metallterminen ska
också stå över sitt eget 200-dagars glidande medelvärde, alltså ha vänt upp.

## Data

* Dagsstängningar via `market_prices` med period `"max"`, alltså den längsta historik som finns. I Actions kommer de
  från Yahoo (utdelningsjusterade, ungefär totalavkastning). `market_prices` provar Börsdata först när en ticker har
  ett Börsdata-id.
  * **Signaler**: GC=F (guld), PL=F (platina) och PA=F (palladium). Det är terminerna, samma primära par som
    panelens kvoter.
  * **Utfall**: GLD, PPLT och PALL, fysiskt backade ETF:er. PPLT och PALL noterades i januari 2010.
  * **Sekundärt, bara beskrivande**: PGM-gruvorna IMPUY (Impala Platinum, ADR) och SBSW (Sibanye-Stillwater, från
    omkring 2017).
* **Krav**: GC=F, PL=F, PA=F, GLD, PPLT och PALL ska ha minst tre års dagskurser (756 dagar). Saknas någon av dem
  avslutas studien med annat än 0, och inget sparas. IMPUY och SBSW krävs inte. Saknas de står det i rapporten.
* **Signalkalendern** är dagar där både GC=F och metallterminen har en stängning, samma inre koppling som
  kvotkortet. **Utfallskalendern** är dagar där både metall-ETF:en och GLD har en stängning. För gruvorna gäller
  dagar där både gruvan och GLD har en stängning.

## Definitioner (point-in-time)

Varje handelsdag t räknas med **bara data till och med t**.

* **Fönster**: de senaste tio åren till och med t, (t − 10 år, t], som kvotkorten (tio års dagsdata).
* **Percentil**: `_percentile_of` ur `alpha_regime/commodity_ratios.py`, det vill säga andelen värden i fönstret som
  är ≤ dagens värde, med dagens värde medräknat.
* **Krav**: fönstret måste börja minst **3 år** före t. Annars kan signalen inte räknas den dagen.
* **P_kvot**: percentilen för GC=F / metallterminen (PL=F eller PA=F). **P_GC** och **P_metall** är varje bens
  egen kurs i samma fönster, alltså benens prisläge.
* **MA200**: det enkla medelvärdet av metallterminens 200 senaste stängningar till och med t, räknat på terminens
  egna handelsdagar. **Över MA200** betyder att stängningen på t är högre än MA200 på t. Med färre än 200
  stängningar kan MA200 inte räknas.

### Signaler

Definitionerna gäller platina (GC=F/PL=F). För palladium används GC=F/PA=F.

| Signal | Villkor |
|---|---|
| **S0 nuvarande regel** | `_classify(P_kvot, "high")` = `RUBBER_BAND_STRETCHED` **och** `_classify_driver(P_GC, P_metall, "high")` finns i `buy_support.ENGINE_DRIVERS` (`DENOMINATOR_CHEAP` eller `BOTH`, det vill säga P_metall ≤ 20). Det är exakt vad motorns +1 gör i dag för de två nycklarna, räknat med panelens egna funktioner. |
| **S1 bara kvoten** | P_kvot ≥ 90. Drivaren ignoreras. |
| **S2 kvoten och trenden** | P_kvot ≥ 90 **och** metallterminens stängning är högre än MA200 på dag t. Trenden har vänt upp, vilket ska skydda mot den fallande kniven. |

Signalerna räknas från första dagen med tre års historik. En dag där signalen inte kan räknas räknas som "ingen
signal". Utfall räknas bara för episoder som börjar **2012-01-01 eller senare**. Dagarna före det avgör bara om en
signal redan pågick.

**Täckning**: alla tre signalerna behöver kvotens fönster. S2 behöver dessutom MA200, och terminerna har mer än 200
stängningar långt före 2012. Täckningen börjar därför i praktiken 2012-01-01 för alla tre signalerna och båda
metallerna. Skulle S2:s täckning ändå börja senare räknas S2:s episoder från den dagen. Rapporten skriver ut varje
signals täckning.

## Episoder och utfall

* **Episod**: börjar första signaldagen efter minst **63 handelsdagar** utan den signalen, räknat i
  signalkalendern.
* **Köp** på stängningen nästa handelsdag, alltså första dagen efter t i utfallskalendern.
* **Utfall per episod**: metall-ETF:ens totalavkastning (PPLT för platina, PALL för palladium) och avkastningen
  **mot GLD** (ETF:ens avkastning minus GLD:s, samma dagar) över **63, 126 och 252 handelsdagar** efter köpet. Bara
  hela horisonter räknas. Dessutom räknas den **värsta nedgången** (topp till botten) för metall-ETF:en inom 252
  dagar efter köpet, också bara med hel horisont.
* **Baslinje**: samma horisonter med köp **varje handelsdag** (ovillkorat) och **var 63:e handelsdag** i samma
  täckning, från första dagen 2012-01-01 eller senare då S1 kan räknas. Bara hela horisonter räknas.
* **Sekundärt, bara beskrivande och utan dom**:
  * **Gruvorna**: samma episoders 252-dagarsavkastning mot GLD för IMPUY och SBSW, där kurser finns. Köpet sker
    första dagen efter t då både gruvan och GLD har en stängning. Det kräver att de båda också har en stängning
    på t eller tidigare. En episod som börjar före gruvans första notering får därför inget utfall, och köpet
    flyttas inte fram till noteringsdagen. Samma sak gäller ETF:erna, i huvudutfallen och korsvis. Gruvans
    baslinje redovisas bredvid.
  * **Korsvis**: palladiums utfall (PALL) på Gold/Platinum-episoderna och platinans utfall (PPLT) på
    Gold/Palladium-episoderna. Panelens exponeringslistor (`EXPOSURE_TO_RATIO`) visar nämligen också den andra
    metallens snodd.

## Hypoteser

Platina och palladium prövas var för sig, och varje hypotes avgörs för sig per metall. Utfallet är
252-dagarsavkastningen mot GLD. Baslinjens snitt avser köp varje handelsdag i täckningen med hel
252-dagarshorisont.

* **H1 den nuvarande regeln fungerar**: snittet för S0 är > 0 och > baslinjens snitt.
* **H2 kvoten ensam fungerar**: snittet för S1 är > 0 och > baslinjens snitt.
* **H3 trendfiltret är bättre**: snittet för S2 är > 0, > baslinjens snitt och ≥ S0:s snitt. Har S0 ingen episod
  med hel 252-dagarshorisont räcker de två första villkoren.

**Bootstrapen** är beskrivande och inget villkor för godkännande. Den ger ett 95 %-intervall för varje signals
snitt jämfört med baslinjen, med en **moving-block-bootstrap med cirkulära block** (block om 63 handelsdagar,
5 000 dragningar, frö 20261104). Metoden är densamma som i gruvornas test (`block_bootstrap` i
`miner_margin_study.py`). Den använder de N dagarna i täckningen som har hel 252-dagarshorisont. De dras i block om
63 dagar i följd. Varje block startar likformigt bland alla N dagar och fortsätter från periodens början när det
når slutet. Blocken läggs efter varandra tills N dagar är fyllda, och det sista blocket kapas. I varje dragning
räknas snittet av utfallet över de episodstarter som kom med. Dragningar utan någon episodstart räknas inte, men
antalet redovisas. Baslinjens snitt hålls fast och redovisas bredvid intervallet.

## Godkännande

* **Golv**: varje dom kräver **minst 3 episoder med hel 252-dagarshorisont** för signalen som prövas, alltså S0
  för H1, S1 för H2 och S2 för H3. Når signalen inte golvet blir domen **för få episoder**. Den är då varken
  godkänd eller underkänd.
* **H1 godkänd**: S0 har minst 3 hela episoder. S0:s snitt är > 0 och > baslinjens snitt.
* **H2 godkänd**: S1 har minst 3 hela episoder. S1:s snitt är > 0 och > baslinjens snitt.
* **H3 godkänd**: S2 har minst 3 hela episoder. S2:s snitt är > 0 och > baslinjens snitt. Och S2:s snitt är ≥
  S0:s, eller så har S0 ingen hel episod. S0:s snitt räknas här över alla S0:s hela episoder, också när de är färre
  än 3.
* Bootstrapintervallen, gruvorna och korsutfallen avgör ingenting.

**Det här testet kan i bästa fall stödja ett förslag.** Det handlar om en enda tillgångsklass (ädelmetaller, där
platina och palladium följs åt) och några få oberoende episoder. Ett godkänt utfall bevisar ingenting. Varje
regeländring är ägarens beslut.

## Vad utfallet leder till (ägarens beslut)

Studien ändrar inga regler. Förslagen gäller per snodd: Gold/Platinum följer platinans domar och Gold/Palladium
palladiums. Fallen prövas i den här ordningen, och det första som stämmer gäller.

1. **H1 håller** → förslag: behåll regeln för snodden.
2. **H1 håller inte (ej godkänd eller för få episoder), men H3 håller** → förslag: ersätt drivarvillkoret med
   trendfiltret för snodden, alltså kvoten ≥ 90 och metallterminen över MA200.
3. **H1 och H3 håller inte, men H2 håller** → förslag: ta bort drivarens nedgradering för snodden, så att kvoten
   ≥ 90 räcker.
4. **Inget håller, och alla tre domarna nådde golvet** → förslag: snodden blir bara kontext, utan +1 och utan
   KÖP-STÖD-räkning, tills bättre belägg finns.
5. **Inget håller, och minst en dom blev "för få episoder"** → **för få episoder**: inget förslag utöver att
   snodden behålls som den är, med en varning.

## Kända begränsningar

* **En tillgångsklass.** Guld, platina och palladium är alla ädelmetaller, och guldet är täljare i båda kvoterna.
* **Få oberoende episoder.** 63-dagarsgapet ger högst några episoder per år, och 252-dagarsutfallen överlappar.
  Kvoten guld/platina stod högt under långa perioder, så en enda lång signalperiod kan bli en enda episod.
* **Terminsrullning i signalserien.** GC=F, PL=F och PA=F är närmaste terminen, som rullas. Den sammanfogade
  serien kan hoppa vid rullningen, och det påverkar percentilerna och MA200 något. Utfallen räknas därför på
  ETF:erna, inte på terminerna.
* **ETF-historiken börjar 2010.** PPLT och PALL noterades i januari 2010. Episoder räknas från 2012-01-01, och
  bara knappt 14 år har hel 252-dagarshorisont.
* **Platina och palladium är korrelerade.** De två prövningarna är inte oberoende.
* **Gruvorna.** Överlevnadsbias: bara bolag som finns kvar i dag är med. IMPUY är en ADR med tunnare handel.
  SBSW finns först från omkring 2017, efter Sibanyes köp av Stillwater. Därför är gruvorna bara beskrivande.
* **Justerade kurser.** Yahoos utdelningsjusterade kurser räknas om bakåt vid varje utdelning. Det berör gruvorna.
  ETF:erna delar inte ut.
* **Reservparet prövas inte.** Panelen faller tillbaka på GLD/PPLT och GLD/PALL när terminerna saknas. Studien
  räknar bara det primära paret.
* **Trösklarna är valda i förväg.** 90, 20 och MA200 är satta före körningen och inte optimerade. Fler varianter
  prövas inte.

## Körning

`pgm_band_study.py` i Actions (`pgm-band-study.yml`, manuell). Resultatet hamnar i loggen och i Gisten
(`pgm_band_study.json`). Skriptet avslutas med 0 även när hypoteserna faller. Det avslutas med annat än 0 bara när
kurser som krävs saknas, och då sparas inget.

## Ändringslogg

| Version | Datum | Ändring |
|---|---|---|
| 1 | 2026-11-04 | Första versionen, före första körningen. |
| 1 | 2026-11-04 | Förtydligat före första körningen, efter verifieringen. Under *Sekundärt* står nu att en episod som börjar före gruvans (eller ETF:ens) första notering inte får något utfall. Den gamla meningen kunde läsas som att köpet flyttas fram till noteringsdagen, kanske flera år efter t. Det stämde inte med den förregistrerade designen, som mäter samma episoders utfall där kurser finns. Studien räknade redan så, så beräkningarna är oförändrade. |
