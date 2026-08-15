"""
strategy_rules_masterguide.py — Masterguiden's strategies that the panel does
not implement natively.

Familj 1 (kontrarisk råvara): Rule · Sprott · Durrett · Tiggre · Royalty —
together they cover the whole Lassonde curve, from drill hole to dividend.
Familj 2 (nordisk aktiv): Insider — the second uncorrelated engine alongside
Momentum-swing, which the panel already has.

They are defined here rather than in strategy_rules.py only to keep the files
readable — strategy_rules.PLAYBOOKS imports them, so there is still one registry
and one source of truth.

Each playbook states its `support` honestly, and the honest answer has changed:
all six now have a tab. Rule has the Rick Rule sheet (the guide calls it
"Producenter A"), Sprott and Durrett the Poängmodell, Tiggre the Lobo sheet,
Royalty the Royalty C sheet, Insider the Insiderbevakare. What is still manual in every case is the Börsdata screener
itself — the panel fetches no fundamentals — and each support_note says so.
Naming the gap remains the point of this module; the gap is just smaller.
"""

from __future__ import annotations

from strategy_rules import (
    Playbook, RiskModel, _rules,
    AMBER, CYAN, EMBER, GREEN, PURPLE,
    LEVEL_MEDIUM, LEVEL_ADVANCED,
    SUPPORT_PARTIAL,
)

_SRC = "Masterguiden Del 4"


# ═════════════════════════════════════════════════════════════════════════════
#  RULE — Överlevarna (producenter i hatad sektor)
# ═════════════════════════════════════════════════════════════════════════════

RULE = Playbook(
    key="rule",
    name="Rule — Överlevarna",
    tagline="Köp de finansiellt starkaste bolagen i en hatad sektor och vänta ut cykeln",
    color=GREEN,
    level=LEVEL_MEDIUM,
    horizon="2–5 år",
    universe="Kanada, Australien, USA, Norden · Metals & Mining + Oil & Gas",
    where="GRANSKNING → Rick Rule (guidens 'Producenter A'). "
          "Screenern körs i Börsdata; SCREENING → Contrarian Alpha ger "
          "hat-rankningen.",
    idea=(
        "Rick Rule: 'bear markets are the authors of bull markets'. I en hatad "
        "sektor prissätts alla bolag som döende — även de med nettokassa och "
        "positivt kassaflöde. Marknaden skiljer inte på friska och sjuka i panik, "
        "och det är din edge. Det enda som avgör vem som finns kvar när cykeln "
        "vänder är balansräkningen, inte projektkvalitet eller story. Äg dem som "
        "ligger längst ner på kostnadskurvan, helst med utdelning som betalar "
        "väntetiden."
    ),
    risk=RiskModel(
        risk_per_trade="Ingen mekanisk stop — tesen är cykelbaserad. Du måste kunna "
                       "ha fel länge, därför krävs balansräkningen",
        position_size="Max 4 % per bolag av totala portföljen — taket gäller. "
                      "Strategikartans 5–10 % är conviction-spannet inom "
                      "Producent-ramen, inte en tillåten positionsstorlek",
        max_positions="Producent-ramen 10–25 % av portföljen (mål 15 %)",
        stop="Ingen stop. Sälj sker i kärleken, inte i hatet",
        targets="Trappa ur i tredjedelar över 6–12 månader när sektorn blir älskad",
    ),
    entry=_rules([
        ("Screener: balansräkningen först",
         "Nettoskuld/EBITDA < 0,5 · soliditet > 50 % · FCF-marginal > 0. Bolaget ska "
         "FINANSIERA sin väntan, inte förbränna den.",
         "Börsdata → sparad screener 'Rule'. Justermån: ND/EBITDA < 1,0 för olja/gas, "
         "soliditet > 40 % i djup baisse.", True),
        ("Screener: värderingen ska visa att sektorn är hatad",
         "EV/EBITDA < 6 och P/B < 1,5. Sortera EV/EBITDA stigande — billigast överst.",
         "Börsdata: lägg direktavkastning, land och börsvärde som KOLUMNER (inte "
         "filter) så du ser varför ett bolag kvalar."),
        ("Landrisk (2 min)",
         "Korsa tillgångarnas läge mot Fraser-rankingen. Politisk risk prissätts, "
         "ignoreras inte — toppjurisdiktion får kosta 30–50 % mer.",
         "Bolagets Översikt-flik i Börsdata + Fraser Institute Mining Survey."),
        ("Kostnadsposition (5 min)",
         "Gruvor: AISC mot dagens metallpris (AISC $1 400 vid guld $2 600 = 46 % "
         "marginal = lågkostnad). Olja: corporate breakeven mot WTI.",
         "Bolagspresentationens cost-sida. 2 poäng i Rick Rule-arket "
         "(guidens Producenter A)."),
        ("Ledning (5 min)",
         "Insynsägande i KRONOR (inte procent), och har någon i teamet drivit ett "
         "bolag genom en hel cykel förut?",
         "Börsdata Ägare-flik + presentationens management-sida."),
        ("Kapitaldisciplin (3 min)",
         "Utdelning eller återköp i svackan = ledningen prioriterar ägarna. Stora "
         "förvärv i svackan kräver dubbel granskning av skulden efteråt.",
         "Börsdata Utdelnings-flik + senaste rapportens kapitalallokeringsavsnitt."),
    ]),
    exit=_rules([
        ("Sälj i kärleken — aldrig i hatet",
         "När sektorns EV/EBITDA-snitt passerar ~10, screenern ger en handfull "
         "träffar, förvärvsvågen rullar och tillväxtcapex är tillbaka i vartannat "
         "pressmeddelande.",
         "Kör Rule-screenern månadsvis och notera antalet träffar — listlängden ÄR "
         "signalen.", True),
        ("Trappa ur i tredjedelar över 6–12 månader",
         "Att sälja för tidigt i en råvarubull är regel, inte undantag. "
         "Tredjedelarna löser det psykologiskt.",
         "Logga varje tredjedel i journalen."),
        ("Passa/sälj vid döende tillgång",
         "Låg multipel förklaras ibland av gruvlivslängd < 5 år eller R/P < 8 år. "
         "Då är priset lågt av ett skäl.",
         "Står i bolagspresentationen (Reserves & Resources).", True),
    ]),
    workflow=(
        "Månadsvis: kör Rule-screenern i Börsdata och notera antal träffar.",
        "Under ~15 träffar = sektorn är dyr, vila. Över ~100 = kapitulation, "
        "nu görs grovjobbet.",
        "Gå igenom topp 15–20 på EV/EBITDA.",
        "Stryk bolag i länder du inte vill äga och bolag med döende tillgång.",
        "Kör de fyra granskningsstegen på resten (≈15 min per bolag).",
        "5/5 i Rick Rule-arket = full position.",
    ),
    cheatsheet=(
        ("ND/EBITDA", "< 0,5 (olja/gas < 1,0)"),
        ("Soliditet", "> 50 % (djup baisse > 40 %)"),
        ("EV/EBITDA", "< 6 (kapitulation < 4)"),
        ("P/B", "< 1,5 (kriscase < 1,0)"),
        ("FCF-marginal", "> 0"),
        ("Listlängd", "< 15 = dyrt · > 100 = kapitulation"),
        ("Position", "Max 4 % av total (taket vinner)"),
        ("Sälj", "Sektorns EV/EBITDA > ~10, i tredjedelar"),
    ),
    pitfalls=(
        "Köpa lågt EV/EBITDA utan att kolla gruvlivslängd — billigt av ett skäl.",
        "Sälja i hatet när det gör som mest ont, i stället för i kärleken.",
        "Ignorera listlängden. När screenern plötsligt ger dubbelt så många träffar "
        "har något hänt med sektorn — det är information, inte brus.",
        "Acceptera skuld 'för att bolaget är så bra'. Skulder dödar i väntan.",
    ),
    support=SUPPORT_PARTIAL,
    support_note="GRANSKNING → Rick Rule ÄR arket guiden "
                 "kallar Producenter A: marginalen mot kostnadskurvan, de tre "
                 "disciplinfrågorna och strykregeln för döende tillgång "
                 "(gruvlivslängd < 5 år, R/P < 8 år). Contrarian "
                 "Alpha-screenern ger hat-rankning som överlappar Rules idé, "
                 "men kör INTE Rules filter (ND/EBITDA, soliditet, P/B, "
                 "FCF-marginal) — de fyra granskningsstegen görs i Börsdata "
                 "och matas in i arket.",
    source=_SRC,
)

# Masterguiden states 5–10 % in the strategy map but caps Rule producers at 4 %
# of the total portfolio. The guide resolves it itself — "de två taken
# (viktigare än målen)" and "taket bryts aldrig" — so the 4 % cap wins. It also
# checks out arithmetically: the 15 % producer sleeve holds ~4 companies at 4 %,
# which is the diversification a survive-the-wait strategy needs.


# ═════════════════════════════════════════════════════════════════════════════
#  SPROTT — Optionalitet (prospektering, lottsedlar)
# ═════════════════════════════════════════════════════════════════════════════

SPROTT = Playbook(
    key="sprott",
    name="Sprott — Optionalitet",
    tagline="En korg lottsedlar där förlusten är begränsad men vinsten inte är det",
    color=PURPLE,
    level=LEVEL_ADVANCED,
    horizon="1–5 år",
    universe="Kanada, Australien (TSX-V och ASX) · Metals & Mining, sub-industri per metall",
    where="GRANSKNING → Poängmodell (Sprott). Screenern körs i Börsdata.",
    idea=(
        "Eric Sprott: i prospekteringsbolag är förlusten begränsad till insatsen men "
        "vinsten obegränsad — spelet handlar inte om att gissa rätt utan om att "
        "överleva tills någon lott vinner. Räkna på korgen: 15 bolag à 1 %. Typiskt "
        "utfall: 10 faller 80 % (−8 %), 3 står stilla, 1 tredubblas (+2 %), 1 gör 15x "
        "(+14 %) → netto +8 % trots att 2 av 3 misslyckades. Matematiken kräver många "
        "bett, små bett, och att inget bolag tvingas till nyemission på botten."
    ),
    risk=RiskModel(
        risk_per_trade="Hela insatsen kan gå förlorad per bolag — det är designen",
        position_size="1–2 % per bolag · positionstak 1,5 % av totala portföljen",
        max_positions="10–15 bolag (aldrig 3–4 — då fungerar inte matematiken)",
        stop="Ingen stop. Runway < 18 månader = sälj/avstå oavsett projekt",
        targets="Låt vinnarna löpa — en enda 30x gör korgen",
    ),
    entry=_rules([
        ("Screener: endast balans, storlek och pris",
         "Prospekteringsbolag har inga vinster — vinstbaserade filter (EV/EBITDA, "
         "P/E, marginaler) raderar hela universumet.",
         "Börsdata → sparad screener 'Sprott'. Filter: Kanada/Australien, "
         "Metals & Mining, börsvärde < 200 MUSD (gärna > 20).", True),
        ("Nettoskuld < 0 — det ENDA hårda kvalitetsfiltret",
         "Skuld + ingen intäkt = död. Kompromissa aldrig här.",
         "Börsdata-filter: Nettoskuld < 0 (nettokassa).", True),
        ("P/B < 1 — prisad under kassa och tillgångar = glömd",
         "< 0,7 när listan är i hundratal.",
         "Börsdata-filter. Sortera P/B stigande och arbeta uppifrån."),
        ("Runway ≥ 18 månader (viktigast, 5 min)",
         "Kassa ÷ årsburn. Under 18 månader = stopp oavsett projekt — nyemissionen "
         "som kommer äter din uppsida.",
         "Senaste kvartalsrapporten: 'Cash and cash equivalents' och 'Cash flow from "
         "operating activities' × 4. Matas i Sprott-fliken som räknar runway.", True),
        ("Poäng ≥ 8 i Poängmodellen",
         "Projektstadium, runway, människor, jurisdiktion, metall enligt rotationen.",
         "Poängmodellen (Sprott-fliken)."),
        ("Många bett, små bett",
         "10–15 bolag à 1–2 %. Aldrig 'jag tror extra på denna' — det förstör "
         "asymmetrin.",
         "Portföljallokeraren: Optionalitets-ramen 0–12 % (delas med Tiggre).", True),
    ]),
    exit=_rules([
        ("Runway faller under 18 månader",
         "Utspädningen tar din andel av vinstlotten innan den dras.",
         "Kontrollera runway varje kvartalsrapport.", True),
        ("Nyemission på botten",
         "Det enda som kan förstöra asymmetrin. Omvärdera positionen från noll.",
         "Bolagets IR-sida / pressmeddelanden.", True),
        ("Låt vinnarna löpa",
         "Korgens resultat kommer från 1–2 bolag. Trimma inte bort dem tidigt.",
         "Journalen: notera varför du säljer — otålighet är inte ett skäl."),
    ]),
    workflow=(
        "Månadsvis: kör Sprott-screenern (200–400 träffar är normalt).",
        "Sortera P/B stigande, grovsålla på metall enligt rotationens AGERA-råvaror.",
        "Målet är 20–30 bolag till manuell granskning per månad — inte alla.",
        "Räkna runway på varje kandidat (5 min). Under 18 mån = bort direkt.",
        "Poängsätt resten i Poängmodellen. ≥ 8 = kandidat.",
        "Köp 1–2 % per bolag tills korgen är 10–15 st.",
    ),
    cheatsheet=(
        ("Börsvärde", "< 200 MUSD (gärna > 20)"),
        ("Nettoskuld", "< 0 — hårt krav, kompromissa aldrig"),
        ("P/B", "< 1 (< 0,7 vid hundratals träffar)"),
        ("Runway", "≥ 18 månader — annars stopp"),
        ("Poäng", "≥ 8"),
        ("Position", "1–2 % · tak 1,5 %"),
        ("Korgstorlek", "10–15 bolag"),
        ("Förväntat utfall", "10 faller 80 % · 1 gör 15x · netto +8 %"),
    ),
    pitfalls=(
        "Ta 3–4 stora positioner i stället för 10–15 små. Då är det inte längre "
        "optionalitet utan en satsning på att du gissar rätt.",
        "Köpa ett bolag med under 18 månaders runway för att projektet är spännande.",
        "Sälja vinnaren vid +100 % — korgens hela resultat sitter i den.",
        "Kompromissa med nettoskuldsfiltret.",
    ),
    support=SUPPORT_PARTIAL,
    support_note="GRANSKNING → Poängmodell räknar runwayen ur kassa och burn, "
                 "poängsätter de fem faktorerna och ger bedömningen. Screenern "
                 "körs fortfarande i Börsdata — panelen hämtar inga "
                 "kvartalssiffror, de matas in för hand.",
    source=_SRC,
)


# ═════════════════════════════════════════════════════════════════════════════
#  DURRETT — Guld/silver-hävstång
# ═════════════════════════════════════════════════════════════════════════════

DURRETT = Playbook(
    key="durrett",
    name="Durrett — Guld/silver-hävstång",
    tagline="Mid-tier producenter där högre kostnad ger större hävstång mot metallpriset",
    color=AMBER,
    level=LEVEL_ADVANCED,
    horizon="1–3 år",
    universe="Kanada, Australien, USA · Metals & Mining → Gold / Silver",
    where="GRANSKNING → Poängmodell (Durrett). Screenern körs i Börsdata.",
    idea=(
        "Don Durrett värderar mot en enda fråga: vad tjänar bolaget vid nästa "
        "guldpris-platå, och vad betalar jag för den vinsten idag? Motorn är operativ "
        "hävstång. Producent A (AISC $1 400, guld $2 600) har marginal $1 200/uns; "
        "guld +50 % → marginal $2 500 = vinst +108 %. Producent B med AISC $1 900: "
        "+186 %. Högre kostnad = större hävstång — därför är mid-tier och juniora "
        "producenter jaktmarken, och därför kräver de balansräkning: samma hävstång "
        "verkar nedåt när guldet faller."
    ),
    risk=RiskModel(
        risk_per_trade="Hävstången verkar åt båda håll — balansräkningen är skyddet",
        position_size="3–5 % per bolag · positionstak 3 % av totala portföljen",
        max_positions="Durrett-ramen 0–15 % av portföljen (mål 8 %)",
        stop="Ingen mekanisk stop — tesen är metallpriscykeln",
        targets="Multipel OCH vinst expanderar samtidigt — där föds 5–10-baggers",
    ),
    entry=_rules([
        ("Screener: hävstångszonen",
         "Börsvärde 50–500 MUSD. Majors har för liten hävstång; upp till 1 000 MUSD "
         "i tidig cykel.",
         "Börsdata → sparad screener 'Durrett'. Sub-industri Gold respektive Silver.", True),
        ("P/S < 2 — billig produktion när vinsten ännu är nedtryckt",
         "< 3 sent i cykeln.",
         "Börsdata-filter. Sortera P/S stigande. Förvänta 15–40 träffar."),
        ("Bruttomarginal > 20 % — överlever DAGENS priser",
         "Hävstång utan konkursrisk. > 30 % skärper.",
         "Börsdata-filter.", True),
        ("Nettoskuld/EBITDA < 2",
         "Durrett tål mer skuld än Rule OM produktionen växer. < 1 om guldtesen "
         "känns osäker.",
         "Börsdata-filter."),
        ("Omsättningstillväxt 1 år > 0 %",
         "Växande produktion = hävstång på hävstången.",
         "Börsdata-filter."),
        ("Börsvärde per uns + poäng ≥ 8, under 10x",
         "Summera reserver (P&P) + Measured & Indicated ur presentationens "
         "'Reserves & Resources'-tabell och räkna börsvärde per uns.",
         "Poängmodellen (Durrett-fliken)."),
        ("Silver-varianten: aktivera bara när guld/silver-kvoten > 85",
         "Verifiera manuellt > 50 % silverintäkter i presentationens "
         "'Revenue by metal'.",
         "Guld/silver-kvoten följs månadsvis i Råvarurotationen.", True),
    ]),
    exit=_rules([
        ("Multipeln har expanderat klart",
         "När både vinst och multipel gått upp är hävstången uttagen.",
         "Poängmodellen: räkna om börsvärde per uns mot nya priset."),
        ("Balansräkningen försämras",
         "ND/EBITDA över 2 utan att produktionen växer = hävstången pekar nedåt.",
         "Kvartalsrapporten.", True),
        ("Guldtesen bryts",
         "Metallpriscykeln är hela tesen — bryts den finns ingen anledning att "
         "sitta kvar i ett högkostnadsbolag.",
         "Råvarurotationen: guldets läge."),
    ]),
    workflow=(
        "Månadsvis: kontrollera guld/silver-kvoten (> 85 aktiverar silver-varianten).",
        "Kör Durrett-screenern i Börsdata (15–40 träffar väntat).",
        "Räkna börsvärde per uns på topp-kandidaterna.",
        "Poängsätt i Poängmodellen — krav ≥ 8 OCH under 10x.",
        "Köp 3–5 %, håll positionstaket 3 % av totalen i minnet.",
    ),
    cheatsheet=(
        ("Börsvärde", "50–500 MUSD (tidig cykel upp till 1 000)"),
        ("P/S", "< 2 (sent i cykeln < 3)"),
        ("Bruttomarginal", "> 20 % (skärpt > 30 %)"),
        ("ND/EBITDA", "< 2 (osäker tes < 1)"),
        ("Tillväxt 1 år", "> 0 %"),
        ("Krav", "Poäng ≥ 8 och under 10x"),
        ("Silver", "Endast när guld/silver-kvot > 85"),
        ("Position", "3–5 % · tak 3 %"),
    ),
    pitfalls=(
        "Köpa majors för 'trygghetens' skull — de har för liten hävstång, hela "
        "poängen försvinner.",
        "Ignorera bruttomarginalen. Hävstång utan marginal är konkursrisk.",
        "Köpa silverbolag när guld/silver-kvoten är låg — då är silvertesen redan spelad.",
        "Glömma att hävstången verkar lika brutalt nedåt.",
    ),
    support=SUPPORT_PARTIAL,
    support_note="GRANSKNING → Poängmodell räknar MCap/uns, MCap/framtida vinst "
                 "och köpregeln under 10x, med hjälpräknaren produktion × "
                 "(målpris − AISC). Guld/silver-kvoten syns i EMBER-regimen "
                 "(Ädelmetaller). Screenern körs i Börsdata.",
    source=_SRC,
)


# ═════════════════════════════════════════════════════════════════════════════
#  TIGGRE — Sweet spot (byggklara utvecklare)
# ═════════════════════════════════════════════════════════════════════════════

TIGGRE = Playbook(
    key="tiggre",
    name="Tiggre — Sweet spot",
    tagline="Byggklara utvecklare vars omvärdering drivs av kalenderhändelser, inte metallpriset",
    color=EMBER,
    level=LEVEL_ADVANCED,
    horizon="6–24 månader",
    universe="Kanada, Australien, USA · Metals & Mining · börsvärde 50–1 000 MUSD · omsättning ~0",
    where="GRANSKNING → Tiggre (Lobo-arket). Håven körs i Börsdata.",
    idea=(
        "Lobo Tiggre skiljer knivskarpt på investering och spekulation: en spekulation "
        "är ett tidsbestämt köp av en SPECIFIK omvärdering, med händelsen definierad "
        "före köpet. Kartan är Lassonde-kurvan. Efter upptäcktsrusningen säljer "
        "spekulanterna men inga nya köpare kommer — fonder köper inte bolag fem år "
        "från kassaflöde. Kursen stagnerar i åratal medan bolaget objektivt blir "
        "mindre riskabelt för varje studie och tillstånd: värde och pris går åt olika "
        "håll. Köpzonen är öknens slut — allt bevisat, allt tillåtet, finansieringen "
        "löst — men kursen står kvar på 0,2–0,4× NAV för att ingen tittat på fem år."
    ),
    risk=RiskModel(
        risk_per_trade="Nedsidan bedöms per case (t.ex. −40 % vid utspädning eller "
                       "byggförsening) och måste ge U/N ≥ 3",
        position_size="2–4 % per bolag · positionstak 2–4 % av totala portföljen",
        max_positions="4–6 bolag · räknas mot Optionalitets-ramen (0–12 %) tills produktion",
        stop="Ingen kursstop — men fyra hårda sälj-allt-triggers (se exit)",
        targets="Free ride vid +100 % (sälj halva) · resten i etapper vid 0,8–1,0× NAV "
                "eller produktionsstart",
    ),
    entry=_rules([
        ("Håven: rätt storlek och rätt balansräkning",
         "Kanada/Australien/USA · Metals & Mining · börsvärde 50–1 000 MUSD (under 50 "
         "= för tidigt, över 1 000 = omvärderingen ofta redan gjord) · nettoskuld < 0 "
         "ELLER skuld som ÄR byggkrediten · omsättning ~0. Inga andra filter — "
         "vinstmått raderar universumet.",
         "Börsdata → sparad screener 'Tiggre'. Screenern är bara håven; urvalet sker "
         "i presentationerna.", True),
        ("Grovsållning: 2 av 3 nyckelfraser (2 min per bolag)",
         "Öppna presentationens tidslinje-sida och leta: 'Feasibility Study complete' "
         "(DFS/BFS) · 'permits received/granted' · 'fully funded' eller 'financing "
         "package'. Två av tre = vidare. Noll av tre = tillbaka i havet.",
         "PEA-bolag sparas i en 'för tidigt'-lista med datum — de blir kandidater "
         "om 1–2 år.", True),
        ("NAV = feasibility-studiens NPV after tax",
         "Alltid after-tax, aldrig pre-tax — skillnaden är 30–40 %.",
         "Utvecklare skyltar med siffran; googla annars '[bolag] feasibility NPV'.", True),
        ("Uppsida räknas till 0,8× NAV",
         "Exempel: börsvärde $200M, NAV $650M → 0,31× NAV → uppsida +160 %.",
         "Lobo-arket räknar automatiskt."),
        ("U/N-kvoten måste vara ≥ 3",
         "Sätt nedsidan själv: värsta rimliga scenariot (30 % utspädning om "
         "finansieringen brister? ett års byggförsening?). Exempel: 160/40 = 4:1 = "
         "godkänt. Ofinansierat med 40 % utspädning framför sig: 85/50 = 1,7:1 → "
         "vänta på finansieringsbeskedet, beskedet ÄR katalysatorn.",
         "Lobo-arket: U/N-kalkylen.", True),
        ("Fem faktorer 0–2, krav ≥ 8",
         "Stadium, finansiering, människor, jurisdiktion, U/N. Människofaktorn är "
         "Tiggres hårdaste: byggmeriter i teamet, insyn > 5–10 %, långsam "
         "utspädningstakt.",
         "Lobo-arket. 'Lifestyle companies' — bolag vars affärsidé är att betala "
         "ledningens löner via emissioner — känns igen på aktieantal som dubblats "
         "utan att projektet flyttat sig.", True),
        ("Katalysatorkalender: minst 2 namngivna, tidsatta händelser inom 12 månader",
         "Kedjan: miljötillstånd → finansieringsbesked → FID → byggstart → 50 % "
         "färdigt → first pour → kommersiell drift. Kan du inte namnge dem är "
         "innehavet en förhoppning, inte en spekulation — passa.",
         "Lobo-arket: katalysatorfliken. Logga utfallet — din träffsäkerhet på "
         "katalysatorbedömningar är strategins viktigaste lärdata.", True),
        ("Timing: köp svaghet, aldrig nyheter",
         "Lägg köp i tystnaden MELLAN katalysatorer när kursen driver ner av "
         "ointresse. Bästa fönstret är tax-loss season (november–december): "
         "kanadensiska och australiska investerare realiserar förluster före "
         "årsskiftet, vilket tvingar fram försäljningar OAVSETT kvalitet — och ger "
         "en statistiskt dokumenterad januaristuds.",
         "Lägg limitordrar under marknaden i november och låt säsongen fylla dem."),
    ]),
    exit=_rules([
        ("Free ride vid +100 % — sälj halva",
         "Insatsen är uttagen och resten åker på husets pengar (eget kapital i risk "
         "= 0 kr). Psykologiskt avgörande: du kan hålla genom byggfasens volatilitet "
         "utan rädsla.",
         "Lobo-arket larmar vid +100 % och visar eget kapital i risk.", True),
        ("Resten säljs vid 0,8–1,0× NAV eller produktionsstart",
         "Eller behålls om bolaget växlar till Rule-kvalitet (kassaflöde + utdelning) "
         "— då omklassas det till Producenter.",
         "Lobo-arket: P/NAV mot dagens kurs."),
        ("Sälj allt samma vecka — fyra triggers",
         "Tillstånd nekas · FS-ekonomin försämras väsentligt · nyckelperson lämnar · "
         "katalysator försenad ANDRA gången utan god förklaring.",
         "Bolagets pressmeddelanden. Ingen tolkning — samma vecka.", True),
        ("−40 % utan att tesen ändrats",
         "Du har missat något. Omvärdera från noll innan du snittar ner — oftast blir "
         "svaret sälj.",
         "Lobo-arket: jämför mot din ursprungliga U/N-kalkyl.", True),
    ]),
    workflow=(
        "Kör håven i Börsdata (screenern är bara håven — urvalet sker i "
        "presentationerna).",
        "Grovsålla 100 bolag → ~10 på en kväll med 2-av-3-testet (2 min per bolag).",
        "För varje överlevare: hämta NAV (NPV after tax) och räkna U/N i Lobo-arket.",
        "U/N < 3 → vänta. Ofta är finansieringsbeskedet självt katalysatorn.",
        "Poängsätt fem faktorer (krav ≥ 8) och namnge minst två katalysatorer "
        "inom 12 månader.",
        "Lägg limitordrar i tystnaden mellan katalysatorer — helst november.",
        "Vid +100 %: sälj halva samma dag. Logga katalysatorutfallen i arket.",
    ),
    cheatsheet=(
        ("Håv", "50–1 000 MUSD · CA/AU/US · omsättning ~0"),
        ("Grovsållning", "2 av 3: FS klar · tillstånd · finansierad"),
        ("NAV", "FS:ens NPV AFTER TAX (aldrig pre-tax)"),
        ("Köpzon", "0,2–0,4× NAV"),
        ("Uppsida", "Räknas till 0,8× NAV"),
        ("U/N-krav", "≥ 3:1"),
        ("Poängkrav", "≥ 8 av 10 (fem faktorer 0–2)"),
        ("Katalysatorer", "≥ 2 namngivna, tidsatta, inom 12 mån"),
        ("Timing", "Köp i tystnaden · tax-loss season nov–dec"),
        ("Free ride", "+100 % → sälj halva"),
        ("Slutsälj", "0,8–1,0× NAV eller produktionsstart"),
        ("Position", "2–4 % · 4–6 bolag"),
    ),
    pitfalls=(
        "Köpa på nyheter i stället för i tystnaden mellan katalysatorer — då betalar "
        "du för omvärderingen i stället för att äga den.",
        "Acceptera en katalysatorkalender du inte kan namnge. Då är det en "
        "förhoppning, inte en spekulation.",
        "Använda pre-tax NPV som NAV — 30–40 % för högt, och hela U/N-kalkylen blir fel.",
        "Snitta ner efter −40 % utan att omvärdera från noll.",
        "Missa 'lifestyle companies': aktieantalet dubblas utan att projektet rör sig.",
    ),
    note="Tiggre fyller gapet mellan Sprotts lotter och Rules producenter — "
         "tillsammans täcker de tre hela Lassonde-kurvan.",
    support=SUPPORT_PARTIAL,
    support_note="GRANSKNING → Tiggre kör Lobo-arket: grovsållning, U/N-kalkyl, "
                 "femfaktorpoäng, katalysatorkalender, free ride-larm och de fyra "
                 "sälj-allt-triggarna. KÖP är låst tills alla fyra grindar passerar. "
                 "Själva håven körs fortfarande i Börsdata, och NAV matas manuellt "
                 "från feasibility-studien.",
    source=_SRC,
)


# ═════════════════════════════════════════════════════════════════════════════
#  ROYALTY — den smarta exponeringen
# ═════════════════════════════════════════════════════════════════════════════

ROYALTY = Playbook(
    key="royalty",
    name="Royalty — Kärnan",
    tagline="Evig andel av gruvors intäkter — betalas oavsett vad gruvan kostar",
    color=GREEN,
    level=LEVEL_MEDIUM,
    horizon="Evig kärna — ägs genom cykeln",
    universe="Kanada, USA, Australien · klassas som Metals & Mining (ingen egen branschkod)",
    where="GRANSKNING → Royalty C. Screenern körs i Börsdata.",
    idea=(
        "En royalty är en evig procentandel av en gruvas intäkter (typiskt 1–3 % NSR) "
        "köpt för en engångssumma; en stream är rätten att köpa produktion till fast "
        "lågt pris. Båda betalas oavsett gruvans kostnader — som låtskrivaren som får "
        "betalt varje gång låten spelas, oavsett vad turnén kostar. Följden: ingen "
        "kostnadsinflation, inga capex-överdrag, gratis optionalitet när gruvan "
        "expanderar, extrem diversifiering (10–300 tillgångar) och 80–95 % "
        "bruttomarginal. Nedsidan i björnmarknad är historiskt −30/−50 % mot "
        "gruvornas −60/−90 %."
    ),
    risk=RiskModel(
        risk_per_trade="Lägst risk i hela systemet — stabilisatorn",
        position_size="Upp till 10 % per bolag (royaltykärnan får glida till 12 %)",
        max_positions="Royalty-ramen 15–30 % av portföljen (mål 20 %)",
        stop="Ingen — ägs genom cykeln",
        targets="Ingen slutförsäljning; trimmas bara mot positionstaket",
    ),
    entry=_rules([
        ("Screener: marginalfiltret avslöjar affärsmodellen",
         "Bruttomarginal > 70 % OCH EBIT-marginal > 40 % är fysiskt omöjligt för en "
         "operatör som betalar diesel, löner och sprängmedel — filtren sållar därför "
         "fram exakt royaltymodellen ur tusentals gruvbolag, automatiskt.",
         "Börsdata → sparad screener 'Royalty'. Ger 20–40 bolag = HELA världens "
         "investerbara royaltyuniversum.", True),
        ("Nettoskuld/EBITDA < 1,5",
         "Belånade royaltybolag i björnmarknad är sektorns enda verkliga risk.",
         "Börsdata-filter.", True),
        ("Köp vid P/NAV-botten medan GEO växer",
         "GEO (guldekvivalenta uns) som växer visar att portföljen av royalties "
         "expanderar.",
         "Royalty C-arket: P/NAV-historik + GEO-utveckling."),
        ("Jämför multipeln mot bolagets EGEN historik",
         "EV/EBITDA 12–25 är NORMALT och motiverat. Nybörjarfelet är att kalla det "
         "dyrt med gruvglasögon — jämför aldrig mot gruvor.",
         "Royalty C-arket.", True),
    ]),
    exit=_rules([
        ("Trimma mot positionstaket",
         "Royaltykärnan får glida till 12 % innan den trimmas — annars 10 %.",
         "Portföljallokeraren: positionstak per bolag."),
        ("Skuldsättningen stiger över 1,5× EBITDA",
         "Den enda verkliga risken i sektorn.",
         "Kvartalsrapporten.", True),
        ("Sälj i princip aldrig annars",
         "Detta är stabilisatorn som ägs genom hela cykeln.",
         "Portföljallokeraren: royaltykärnan är nivå 1."),
    ]),
    workflow=(
        "Kör Royalty-screenern EN gång och spara listan — den ändras med något "
        "bolag per år.",
        "Följ P/NAV mot bolagets egen historik i Royalty C-arket.",
        "Köp när P/NAV är i botten av sitt historiska spann och GEO växer.",
        "Kvartalsvis: kontrollera skuldsättning och trimma mot positionstaket.",
    ),
    cheatsheet=(
        ("Bruttomarginal", "> 70 % (avslöjar modellen)"),
        ("EBIT-marginal", "> 40 %"),
        ("ND/EBITDA", "< 1,5"),
        ("Universum", "20–40 bolag globalt — kör screenern en gång"),
        ("Normal multipel", "EV/EBITDA 12–25 (inte dyrt)"),
        ("Köpsignal", "P/NAV-botten + GEO växer"),
        ("Position", "Upp till 10 % (kärnan får glida till 12 %)"),
        ("Roll", "Stabilisatorn — ägs genom cykeln"),
    ),
    pitfalls=(
        "Kalla EV/EBITDA 12–25 för dyrt. Det är normalt för modellen — jämför mot "
        "bolagets egen historik, inte mot gruvor.",
        "Köpa belånade royaltybolag. Skulden är sektorns enda verkliga risk.",
        "Sälja kärnan i en nedgång. Den finns just för att bära dig genom den.",
    ),
    support=SUPPORT_PARTIAL,
    support_note="GRANSKNING → Royalty C räknar rabatten mot "
                 "egen botten, mot egen median och GEO-tillväxten per aktie, "
                 "och ger köpsignalen. Screenern körs en gång i Börsdata, "
                 "och siffrorna matas in för hand ur presentationerna.",
    source=_SRC,
)


# ═════════════════════════════════════════════════════════════════════════════
#  INSIDER (Norden) — trestegsraketen
# ═════════════════════════════════════════════════════════════════════════════

INSIDER = Playbook(
    key="insider",
    name="Insider — Trestegsraketen",
    tagline="Följ dem som vet mest — men köp först när marknaden börjar hålla med",
    color=CYAN,
    level=LEVEL_MEDIUM,
    horizon="6–18 månader",
    universe="Norden · börsvärde > 300 MSEK",
    where="GRANSKNING → Insider (bevakaren). Insynsflödet läses i Börsdata.",
    idea=(
        "Att insiderköp förutsäger överavkastning är dokumenterat sedan 1960-talet, "
        "men edgen sitter på specifika ställen. Lakonishok & Lee: kluster slår "
        "ensamköp — en person kan köpa av tusen privata skäl, tre samtidigt har en "
        "gemensam åsikt. Cohen, Malloy & Pomorski: opportunistiska köpare "
        "(oregelbundna, informationsdrivna) är en stark signal, rutinköpare (samma "
        "månad varje år) ingen alls. Effekten är störst i småbolag med tunn "
        "analytikertäckning — vilket beskriver Norden perfekt. Asymmetrin som gör "
        "allt: insiders SÄLJER av tusen skäl (hus, skilsmässa, skatt) men KÖPER av "
        "exakt ett — de tror att aktien är för billig."
    ),
    risk=RiskModel(
        risk_per_trade="Stop −15 % under klustersnittet — läggs som order hos mäklaren",
        position_size="8–15 % av strategidelen · positionstaket är 4 % av totala "
                      "portföljen",
        max_positions="6–10 innehav · max 1–2 nya per månad",
        stop="−15 % under insiderklustrets snittkurs",
        targets="+50–100 % när värderingen kommit ikapp sektorn · 18 månaders tidsstopp",
    ),
    entry=_rules([
        ("Steg 1: hitta klustret — och rensa bruset först",
         "Räkna ENDAST riktiga marknadsköp. Optionslösen, aktieprogram, arv och "
         "interna omflyttningar är inte signaler och ska bort innan du poängsätter.",
         "Börsdata → Insider/Insynshandel: skanna 1–2 ggr/vecka, filtrera på KÖP, "
         "sortera på belopp. Transaktionstypen visar vad som är ett riktigt köp.", True),
        ("Poängsätt signalen — krav ≥ 7 av 10",
         "Kluster (3+ = 3 p, 2 = 2 p) · högsta roll (VD/CFO 2 p — de ser HELA bilden, "
         "styrelse 1 p) · belopp > 1 MSEK 2 p · relativ storlek (ökar innehavet "
         "> 25 % = 1 p) · kontext (efter fall > 20 % i friskt bolag = 1 p) · "
         "återkommande köpare 1 p. 7–10 = stark signal · 5–6 = bevaka · < 5 = brus.",
         "Insiderbevakaren. Relativ storlek är underskattad: 200 tkr kan vara enormt "
         "för en CFO med litet innehav och ingenting för en miljardär.", True),
        ("Steg 2: kvalitetsgrinden",
         "Börsvärde > 300 MSEK (under det äter spreadarna edgen) · F-score ≥ 5 · "
         "nettoskuld/EBITDA < 2 eller nettokassa · positivt FCF eller tydlig väg dit. "
         "Insiders fångar också fallande knivar — grinden sorterar bort bolagen där "
         "även VD:n har fel.",
         "Börsdata → sparad screener 'Insider – grind'. Fallande omsättning i "
         "strukturell nedgång = varning även om allt annat kvalar.", True),
        ("Steg 3: vänta på den tekniska triggern",
         "Insiders är notoriskt 3–6 månader tidiga — du förlorar inget på att vänta "
         "på att marknaden börjar hålla med. A) stängning över MA20 som planat ut "
         "eller vänt upp · B) positiv 1-månadsutveckling OCH kurs över klustrets "
         "snittkurs · C) första rapporten efter köpen bekräftar (köp på rapportdagen).",
         "Insiderbevakaren driver statusflödet: Ignorera → Bevaka → Kör grinden! → "
         "Väntar på trigger → KÖP.", True),
        ("Klustersnittkursen är ankaret — max 30 % över",
         "Insiders betalade den. Köper du nära eller under har du samma ingång som "
         "de som vet mest. Mer än 30 % över = edgen förbrukad, passa.",
         "Insiderbevakaren räknar snittet automatiskt.", True),
        ("Max 1–2 nya per månad, 6–10 innehav",
         "Detta är inte swing — edgen realiseras via rapporter, uppköp och "
         "omvärdering över 6–18 månader.",
         "PORTFOLIO → Allokering: Insider-ramen 10–30 % (mål 20 %), positionstak "
         "4 % per bolag."),
    ]),
    exit=_rules([
        ("Säljregel 1 — säljKLUSTER",
         "2+ insiders säljer (inte småposter). Signalen som tog dig in har vänt.",
         "Börsdata → Insynshandel: samma flöde, men filtrerat på SÄLJ.", True),
        ("Säljregel 2 — −15 % under klustersnittet",
         "Läggs som order hos mäklaren direkt vid köp, inte som en påminnelse.",
         "Insiderbevakaren räknar stoppnivån ur klustersnittet.", True),
        ("Säljregel 3 — grinden bryts",
         "F-score < 4 eller skulden drar iväg. Kvalitetsgrinden gäller även efter köp.",
         "Börsdata: kontrollera F-score och nettoskuld vid varje rapport.", True),
        ("Säljregel 4 — +50–100 % och värderingen ikapp sektorn",
         "Omvärderingen är gjord. Det var hela tesen.",
         "Jämför multipeln mot sektorn i Börsdata."),
        ("Säljregel 5 — 18 månaders tidsstopp",
         "Kapitalet jobbar bättre i nästa signal. Har inget hänt på 18 månader hade "
         "insidern fel om timingen även om hen hade rätt om bolaget.",
         "Insiderbevakaren: köpdatum + 18 månader.", True),
    ]),
    workflow=(
        "1–2 ggr/vecka: skanna Börsdatas insynsflöde, filtrera på KÖP, sortera på "
        "belopp. Notera bolag med flera transaktioner.",
        "Rensa bort optionslösen, aktieprogram, arv och interna omflyttningar.",
        "Poängsätt kandidaterna i Insiderbevakaren — under 5 poäng är brus.",
        "Kör 7+-kandidaterna mot kvalitetsgrinden (F-score, skuld, FCF).",
        "Vänta på trigger A, B eller C. Köp aldrig kniven — köp bekräftelsen.",
        "Kontrollera att kursen är max 30 % över klustersnittet.",
        "Köp, lägg stop −15 % under snittet som order, logga i journalen.",
    ),
    cheatsheet=(
        ("Skanning", "1–2 ggr/vecka, endast riktiga marknadsköp"),
        ("Kluster", "3+ köpare = 3 p · 2 = 2 p"),
        ("Roll", "VD/CFO 2 p · styrelse 1 p"),
        ("Belopp", "> 1 MSEK = 2 p"),
        ("Poängkrav", "≥ 7 av 10 (5–6 = bevaka, < 5 = brus)"),
        ("Grinden", "> 300 MSEK · F-score ≥ 5 · ND/EBITDA < 2 · FCF+"),
        ("Trigger", "A) över MA20 · B) 1 mån+ över snittet · C) rapport"),
        ("Ankare", "Max 30 % över klustersnittet"),
        ("Stop", "−15 % under klustersnittet"),
        ("Position", "8–15 % av delen · tak 4 % av total"),
        ("Antal", "6–10 innehav · max 1–2 nya/månad"),
        ("Tidsstopp", "18 månader"),
    ),
    pitfalls=(
        "Räkna optionslösen och aktieprogram som insiderköp. De är ersättning, inte "
        "en åsikt om kursen — och de dominerar flödet om du inte rensar.",
        "Köpa kniven direkt på signalen. Insiders är 3–6 månader tidiga; triggern "
        "finns för att du inte ska sitta och blöda under tiden.",
        "Köpa mer än 30 % över klustersnittet — då har du inte längre samma ingång "
        "som de som vet mest.",
        "Tolka en enstaka försäljning som en säljsignal. Insiders säljer av tusen "
        "skäl; först ett säljKLUSTER betyder något.",
        "Sitta kvar förbi 18-månadersstoppet i hopp om att tesen ska mogna.",
    ),
    note="Synergin gör hela systemet skarpare: insiderkluster i ett bolag som redan "
         "ligger i momentum- eller Överlevar-listan = dubbel bekräftelse (position i "
         "övre intervallet, aldrig över taket). Säljkluster i ett swinginnehav = dra "
         "åt stoppen. Klusterköp i en hatad råvarusektor = extra vikt i "
         "rotationsbetyget.",
    support=SUPPORT_PARTIAL,
    support_note="GRANSKNING → Insider kör bevakaren: poängen 0–10, "
                 "kvalitetsgrinden, den tekniska triggern, statusflödet, "
                 "vs-kluster och stoppen. Själva insynsflödet läses fortfarande "
                 "manuellt i Börsdatas Insynshandel — panelen hämtar inga "
                 "transaktioner. PORTFOLIO → Allokering håller Insider-ramen "
                 "(mål 20 %) och positionstaket 4 %.",
    source=_SRC,
)


MASTERGUIDE_PLAYBOOKS = {
    "rule": RULE,
    "royalty": ROYALTY,
    "durrett": DURRETT,
    "sprott": SPROTT,
    "tiggre": TIGGRE,
    "insider": INSIDER,
}
