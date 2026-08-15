"""
commodity_book.py — Råvarukartboken (Masterguiden Del 5).

The depth behind rotation.py's one-liners. The rotation table says *that* uran
is hated when spot trades under the incentive price; this says why the
contract cycle produces that, which basket to buy it through, and where the
number lives.

Read while grading — the hat-poäng is a judgement call, and this is the
material the judgement is supposed to rest on.

Keyed by rotation.COMMODITIES, so a commodity can never have a chapter the
rotation does not know about (test_commodity_book asserts both directions).
Royalty has no chapter here: it is a strategy, not a commodity cycle, and the
guide covers it in Del 4 — the Royalty playbook in strategy_rules_masterguide
owns it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    """One commodity, four questions: what is it, how do I play it, when, and
    what kills people here."""
    key: str
    subtitle: str
    market: str
    play: str
    timing: str
    sources: tuple[str, ...] = ()
    pitfall: str = ""          # tom = guiden namnger ingen för den här
    role: str = ""             # särskild portföljroll, om någon


CHAPTERS: tuple[Chapter, ...] = (
    Chapter(
        key="guld",
        subtitle="den monetära metallen",
        market="Inte en råvara i vanlig mening — pengar utan motpartsrisk. "
               "Årsproduktionen är ~1,5 % av ovanjordslagret, så priset styrs "
               "inte av utbudet utan av realräntor (negativ realränta = guld "
               "stiger), centralbanksköp (strukturellt sedan 2022) och "
               "förtroendet för statsfinanserna. Det kontrariska läget sitter "
               "i GRUVORNA: rekordmarginaler, låga multiplar och frånvarande "
               "generalister — FCF-yield > 10 % och EV/EBITDA < 5 för "
               "seniorer trots stigande guldpris.",
        play="Tre verktyg du redan har: Durrett (hävstångsproducenter), "
             "Tiggre (utvecklare), Royalty (kärnan). I korrektioner köps det i "
             "tre trancher: −10 %, −15/20 %, reserv. Trimma gruvorna när "
             "FCF-yielden komprimerats under ~5 %.",
        timing="ETF-utflöden samtidigt som centralbankerna köper = väst säljer, "
               "öst köper. Det är det kontrariska läget.",
        sources=("Trading Economics — guld + \"US 10Y TIPS yield\"",
                 "World Gold Council — \"Gold Demand Trends\", kvartalsvis och gratis"),
        role="Enda råvaran som stiger i riskaversion. Guldbenet plus "
             "royaltykärnan är hedgen mot att alla kontrariska ben "
             "korrelerar i en krasch.",
    ),
    Chapter(
        key="silver",
        subtitle="högbeta-hybriden",
        market="Hälften monetär (följer guld), hälften industriell (solceller "
               "~30 % och växande). Liten marknad där utbudet mest är biprodukt "
               "från bly, zink och koppar — det svarar knappt på silverpriset. "
               "Därför rör sig silver 2–3× guld åt båda hållen och exploderar "
               "sent i varje guldcykel.",
        play="Durrett-screenern med sub-industri Silver. Kräv > 50 % "
             "silverintäkter — kolla intäktsmixen, de flesta \"silverbolag\" är "
             "guld- eller zinkbolag. AISC < ~$15/uns är starkt. Positioner "
             "3–5 %, men swing-mentalitet på toppen.",
        timing="Guld/silver-kvoten: över 85–90 = ackumulera, under 50 = "
               "sencykliskt, trimma. Trögrörlig och förlåtande — tänk zoner, "
               "inte dagar.",
        sources=("Trading Economics — guld/silver-kvoten",),
        pitfall="Hävstången funkar åt båda hållen: i guldkorrektioner faller "
                "silverbolagen 2–3× mer. Köp bara vid hög kvot, aldrig som "
                "momentumjakt. Ägs aldrig genom en mani-topp.",
    ),
    Chapter(
        key="platina",
        subtitle="den äkta kontrariska PGM:en",
        market="Sydafrika står för ~70 % av utbudet — elkris, djupa gruvor och "
               "strejker gör att det krymper strukturellt. Efterfrågan kommer "
               "från katalysatorer med substitution IN från palladium "
               "(billigare), smycken, och vätgas/bränsleceller som en gratis "
               "option. Platina långt under guldpriset är den historiska "
               "anomali-signalen.",
        play="Få rena bolag finns: Överlevar-screenern plus manuell "
             "PGM-filtrering. Kräv lägsta tredjedelen av kostnadskurvan och "
             "prisa in SA-risken med extra rabatt — hela sektorn ÄR Sydafrika. "
             "Fysiskt uppbackad ETC är ett legitimt alternativ här, eftersom "
             "gruvrisken är ovanligt hög relativt metallcaset. Ägbar 1–3 år.",
        timing="Ihållande underskott + pris under SA-gruvornas AISC + "
               "schaktstängningar = köp. Sälj när substitutionsstoryn är "
               "konsensus och priset närmar sig guldets.",
        sources=("WPIC — \"Platinum Quarterly\", gratis",),
    ),
    Chapter(
        key="palladium",
        subtitle="den falska frestelsen",
        market="Bensinkatalysatorer, med utbud från Ryssland och Sydafrika. "
               "EV-övergången äter efterfrågan strukturellt och substitutionen "
               "går UT mot platina. Billig av strukturella skäl, inte cykliska "
               "— sektorns tydligaste exempel på att \"billigt\" och "
               "\"köpvärt\" är olika saker.",
        play="ENDAST trade. Efter krascher på −60 % eller mer kan "
             "utbudsstörningar (sanktioner, SA-elkris) ge våldsamma studsar. "
             "Definierad katalysator, definierad exit, max 1–2 %, tidsstopp "
             "12 månader. Aldrig buy and hold.",
        timing="Trading Economics plus nyhetsflödet. Regeln står inpräntad i "
               "rotationstabellen: Palladium = trade.",
        sources=("Trading Economics — palladium",),
        pitfall="Den strukturella motvinden vinner alltid till slut. Som "
                "litium ska palladium aldrig ägas genom en topp — och till "
                "skillnad från litium inte ens genom en cykel.",
    ),
    Chapter(
        key="uran",
        subtitle="kontraktscykelns metall",
        market="Kärnkraftverk köper via 5–10-årskontrakt, inte spot. "
               "Fullkontrakterade verk gör att sektorn sover i åratal — sedan "
               "MÅSTE de tillbaka, och utbudet finns inte. Extremt långa, "
               "extremt kraftiga cykler; mellan dem faller sektorn 80–90 % och "
               "kan ligga död i 5+ år.",
        play="Optionalitets-screenern (Kanada/Australien, sub-industri "
             "Uranium, nettokassa) plus poängmodellen. Börsvärde ÷ Mlbs U3O8 "
             "under ~$5/lb är billigt för tillståndsgivna utvecklare — och "
             "tillstånden är värda mer än fyndigheten. 1–2 % per bolag, 5–8 "
             "bolag, endast kapital som tål 5 års inlåsning. Sälj i etapper "
             "när uran är förstasidesstoff.",
        timing="Spot under incitamentspriset ~$80–90/lb plus tystnad i media "
               "= hat-poäng 4–5.",
        sources=("Cameco.com — \"Uranium prices\", gratis",),
    ),
    Chapter(
        key="olja",
        subtitle="kassaflödesmaskinen",
        market="Källor sinar 5–8 % om året av sig själva, så "
               "capex-nedskärningar garanterar nästa uppgång 2–4 år senare. "
               "Marknaden belönar numera utdelning och straffar tillväxt, "
               "vilket håller utbudet strukturellt lågt.",
        play="Överlevar-screenern med skuld/EBITDA < 1,0, EV/EBITDA < 4, "
             "FCF-marginal > 10 % och utdelning. Granska sedan: breakeven "
             "under $45 WTI, R/P-kvot över 10 år, mer än 50 % av kassaflödet "
             "till ägarna. Position 5–10 % — njut av utdelningarna.",
        timing="Riggantal nära flerårslägsta plus breda capex-nedskärningar = "
               "köpläge. Andra kvittot: energisektorns S&P-vikt under ~4 %. "
               "Sälj när branschen byter till förvärv och tillväxtcapex — "
               "alltid toppen-beteende.",
        sources=("Baker Hughes — rig count, fredagar och gratis",
                 "Trading Economics — WTI"),
    ),
    Chapter(
        key="gas",
        subtitle="vädrets och exportens marknad",
        market="Regional prissättning (Henry Hub i USA, TTF i Europa): väder "
               "på kort sikt, LNG-exportkapacitet på lång. Bottnar när priset "
               "understiger torrgas-breakeven runt $2,5/MMBtu och producenter "
               "stänger in volymer.",
        play="Som olja, men EV/EBITDA < 5 är OK. Kräv hedgebok > 40 % av "
             "nästa års produktion (presentationens Hedging-avsnitt) och en "
             "lågkostnadsbassäng — Appalachia eller Haynesville. Katalysator: "
             "ny LNG-exportkapacitet i drift.",
        timing="Flerårslägsta pris plus rekordfulla lager = botten bekräftad.",
        sources=("Trading Economics — Henry Hub",
                 "EIA — Weekly Natural Gas Storage Report"),
        pitfall="Sälj väderdrivna spikar. De faller alltid tillbaka.",
    ),
    Chapter(
        key="kol",
        subtitle="den permanent hatade utdelningsmaskinen",
        market="ESG-exkludering betyder inga nya gruvor och permanent låga "
               "multiplar — men befintliga gruvor är kassaflödesmaskiner på "
               "2–4× vinsten. Avkastningen kommer via utdelning och återköp, "
               "INTE via multipelexpansion. Met-kol (stål) före termiskt (el).",
        play="FCF-yield > 15 %, nettokassa eller minimal skuld, pågående "
             "återföring till ägarna. Viktigast av allt: normalisera vinsten "
             "mot 10-årssnittpriset — P/E 3 på topppris kan vara P/E 10 på "
             "normalpris. Behåll så länge återföringen består; sälj först när "
             "utdelningen sänks eller ledningen förvärvar utanför kärnan.",
        timing="FCF-yield över 20 % efter ett prisras är köpläget.",
        sources=("Börsdata — FCF-yield",
                 "Trading Economics — Coal / Coking Coal"),
        pitfall="Lågt P/E är kolens normalläge, inte ett köpargument.",
    ),
    Chapter(
        key="koppar",
        subtitle="elektrifieringens flaskhals",
        market="Elbilar, elnät och datacenter möter en historiskt tunn "
               "pipeline — det tar 10–15 år till en ny gruva. Starkaste "
               "strukturella caset bland basmetallerna, men vägen dit går via "
               "Kina-svackor som ger återkommande köplägen.",
        play="Dubbelspår. Korg A (producenter): Överlevarna, C1-kostnad i "
             "lägsta halvan, jurisdiktionsrabatt för Chile och Peru — kan ägas "
             "genom cykeln. Korg B (utvecklare): Optionalitet eller Tiggre "
             "med krav på FS och tillstånd — säljs i eufori, den rör sig 3–5× "
             "producenterna.",
        timing="Pris under incitament (~$4,5/lb) plus stigande lager plus "
               "fryst projektfinansiering.",
        sources=("Trading Economics — koppar", "LME.com — \"LME stocks\""),
    ),
    Chapter(
        key="zink",
        subtitle="den korta, snabba cykeln",
        market="Renaste utbudscykeln av alla, 2–4 år. Termometern är "
               "TC-avgifterna — smältverkens förädlingsavgift. Höga TC betyder "
               "koncentratöverskott och att botten är nära; kollapsande TC "
               "betyder brist och att uppgången pågår.",
        play="Överlevarna på Metals & Mining plus manuell filtrering på > 50 % "
             "zinkintäkter. Kräv NETTOKASSA — skuldsatta zinkbolag dör i "
             "svackorna — AISC i lägsta tredjedelen och gruvlivslängd över 5 "
             "år. Silver- och blybiprodukter är plus. Sätt måltavlan redan vid "
             "köpet (+50–80 %).",
        timing="TC-topp plus annonserade gruvstängningar = köp. Kolla "
               "kvartalsvis.",
        sources=("Branschpress — \"zinc treatment charges benchmark\"",),
        pitfall="Sälj snabbare här än i någon annan metall — cykeln är kort "
                "och vänder utan förvarning.",
    ),
    Chapter(
        key="jarnmalm",
        subtitle="Kina-barometern",
        market="I praktiken en enda fråga: Kinas stål- och fastighetssektor. "
               "Fyra majors med $20–40/t i kostnad sätter golvet — under ~$80 "
               "slås högkostnadsproducenterna ut och utbudet krymper av sig "
               "självt. Sektorns mest mekaniska cykel.",
        play="ENDAST lågkostnadsproducenter med C1 under ~$50/t — högkostnad "
             "är trading, inte investering. EV/EBITDA < 4 på normaliserat pris "
             "(~$90–100/t), nettokassa, utdelningshistorik. Fe-grade över 65 % "
             "ger premie. Ta utdelningarna — 8–12 % hos majors i svackor — och "
             "sälj i stimulans-eufori.",
        timing="Pris $70–80/t plus total Kina-pessimism = köpläge.",
        sources=("Trading Economics — \"Iron Ore 62% Fe\"",),
    ),
    Chapter(
        key="litium",
        subtitle="bubblornas metall",
        market="EV-tillväxt möter utbud i vågor, vilket ger extrema bubblor "
               "följda av 80–90 %-krascher — sedan pausas allt och bristen "
               "byggs upp igen. Ung marknad utan färdigt cykelmönster.",
        play="Köp BARA efter kraschen (−70 % eller mer) och bara kvalitet: "
             "producenter med AISC i lägsta kvartilen (sydamerikansk brine "
             "eller bästa hard-rock), nettokassa, P/B < 1,5. Utvecklare endast "
             "med runway över 24 månader och offtake-avtal. Lera och DLE är "
             "obevisad teknik — extra rabatt.",
        timing="Prisras på −70 %+ och breda projektpauser öppnar köpfönstret.",
        sources=("Trading Economics — litiumkarbonat", "Branschpress"),
        pitfall="SÄLJ ALLTID I EUFORIN. Minst buy-and-hold-vänliga metallen "
                "som finns.",
    ),
)

CHAPTER_BY_KEY = {c.key: c for c in CHAPTERS}

# Royalty is the one rotation row without a chapter — see the module docstring.
NO_CHAPTER = ("royalty",)


def chapter(key: str) -> Chapter | None:
    """The kartbok entry for a rotation key, or None (royalty)."""
    return CHAPTER_BY_KEY.get(key)


def has_chapter(key: str) -> bool:
    return key in CHAPTER_BY_KEY
