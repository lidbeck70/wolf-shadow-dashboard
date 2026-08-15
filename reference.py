"""
reference.py — Snabbreferensen (Masterguiden Del 7).

Four lookup tables that answer the questions you get mid-trade, when you do
not want to read a playbook:

  · Vilket filter hade den där screenern?
  · Vad var säljregeln för det här benet?
  · Var hittar jag AISC / TC-avgifter / realräntan?
  · Vad betyder P/NAV?

Nothing here is new policy. The sell rules are the *same* rules the playbooks
enforce, collected on one page — test_reference asserts that every strategy
with a playbook has a matching sell row, so this page can never quietly say
something the engines do not do.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Screener:
    name: str
    key: str            # playbook-nyckel, tom om screenern inte har en
    filters: str
    where: str = "Börsdata"


@dataclass(frozen=True)
class SellRule:
    strategy: str
    key: str            # playbook-nyckel, tom för portfölj-/råvaruregler
    rule: str


@dataclass(frozen=True)
class Source:
    number: str         # siffran du är ute efter
    source: str
    where: str          # var/sökord


@dataclass(frozen=True)
class Term:
    term: str
    meaning: str


# ── Alla screeners (sida 32) ─────────────────────────────────────────────────
# Spara dem i Börsdata med exakt de här namnen, så matchar guide och panel.

SCREENERS: tuple[Screener, ...] = (
    Screener("Överlevarna (Rule)", "rule",
             "Råvarubransch · skuld/EBITDA < 0,5 (olja < 1,0) · soliditet > 50 % · "
             "EV/EBITDA < 6 · P/B < 1,5 · FCF > 0"),
    Screener("Optionalitet (Sprott)", "sprott",
             "Kanada/Australien · Metals & Mining · MCap < 200 MUSD · "
             "nettokassa · P/B < 1"),
    Screener("Durrett", "durrett",
             "Guld/silver · MCap 50–500 MUSD · P/S < 2 · bruttomarginal > 20 % · "
             "skuld/EBITDA < 2 · omsättningstillväxt > 0"),
    Screener("Tiggre (sweet spot)", "tiggre",
             "Kanada/Australien/USA · MCap 50–1 000 MUSD · nettokassa eller "
             "byggkredit · omsättning ~0 → manuell FS- och tillståndssållning"),
    Screener("Royalty", "royalty",
             "Kanada/USA/Australien · bruttomarginal > 70 % · EBIT-marginal > 40 % · "
             "skuld/EBITDA < 1,5"),
    Screener("Swing – universum", "momentum",
             "Norden · MCap > 1 000 MSEK · kurs > MA200 · 3 mån > 0 · "
             "6 mån > 10 % · F-score ≥ 5",
             "Börsdata · panelen räknar rankingen"),
    Screener("Insider – grind", "insider",
             "Norden · MCap > 300 MSEK · F-score ≥ 5 · skuld/EBITDA < 2 · "
             "FCF positivt eller på väg"),
)


# ── Alla säljregler på en sida (sida 32) ─────────────────────────────────────
# Först inträffad gäller.

SELL_FIRST_WINS = "Först inträffad gäller."

SELL_RULES: tuple[SellRule, ...] = (
    SellRule("Rule", "rule",
             "Sektorn älskad: EV/EBITDA > ~10, tom screener, förvärvsvåg "
             "→ trappa ur"),
    SellRule("Sprott", "sprott",
             "Runway < 12 mån utan besked · caset ändrat · etappvis efter 10x"),
    SellRule("Durrett", "durrett",
             "MCap/framtida vinst närmar sig 10x · produktion sviker "
             "2 kvartal · cykeltopp"),
    SellRule("Tiggre", "tiggre",
             "Tes bruten = allt · +100 % = halva (free ride) · katalysator "
             "försenad 2:a gången = allt · 0,8–1,0x NAV = resten · "
             "−40 % = omvärdera"),
    SellRule("Royalty", "royalty",
             "Nivå 1: i princip aldrig · Nivå 2: GEO/aktie stagnerar 2 år · "
             "Nivå 3: uppköp eller tes klar"),
    SellRule("Swing", "momentum",
             "Stängning under MA50 · −10 % stop · ur topp 40 · "
             "(+20 % = sälj halva, stop till entry)"),
    SellRule("Insider", "insider",
             "Säljkluster · −15 % under klustersnitt · grinden bryts · "
             "+50–100 % · 18 mån tidsstopp"),
    SellRule("Palladium / Litium / Silver", "",
             "ALLTID i euforin — ägs aldrig genom en mani-topp"),
    SellRule("Portföljnivå", "",
             "Strömbrytaren: −10 % skärpt · −20 % halverad risk"),
)


# ── Alla datakällor (sida 33) ────────────────────────────────────────────────

SOURCES: tuple[Source, ...] = (
    Source("Alla nyckeltal, F-score, MA, historik", "Börsdata",
           "Screener / Nyckeltal / Graf"),
    Source("Insynstransaktioner", "Börsdata (Holdings)",
           "Insynsflödet + bolagens insynsflik"),
    Source("Kassa / burn rate", "Kvartalsrapport",
           "\"Cash and cash equivalents\" / op. kassaflöde × 4"),
    Source("AISC, C1, breakeven, hedgebok", "Bolagspresentation",
           "Cost- och Hedging-sidorna, sida 8–15"),
    Source("Reserver, uns, pounds, NAV, GEO", "Presentation / FS",
           "\"Reserves & Resources\", \"NPV after tax\""),
    Source("Insynsägande, institutioner", "Presentation / Börsdata",
           "\"Share Structure\" / Ägare-fliken"),
    Source("Råvarupriser + guld/silver-kvot", "Trading Economics",
           "Gratis, alla råvaror"),
    Source("Realränta (guld)", "Trading Economics", "\"US 10Y TIPS yield\""),
    Source("Centralbanksköp, ETF-flöden", "World Gold Council",
           "\"Gold Demand Trends\""),
    Source("PGM-balanser", "WPIC", "\"WPIC Platinum Quarterly\""),
    Source("Riggantal / gaslager / metallager", "Baker Hughes / EIA / LME",
           "Fredagar / veckorapport / \"LME stocks\""),
    Source("Uranspot / TC-avgifter zink", "Cameco.com / branschpress",
           "\"Uranium prices\" / \"zinc TC benchmark\""),
    Source("Jurisdiktionsrisk", "Fraser Institute",
           "\"Annual Survey of Mining Companies\""),
)


# ── Ordlistan (sida 33–34) ───────────────────────────────────────────────────

GLOSSARY: tuple[Term, ...] = (
    Term("EV/EBITDA", "Hur många årsvinster kostar bolaget, inklusive skuld? "
                      "Lägre = billigare."),
    Term("P/B · P/S", "Pris mot bokfört värde / mot omsättning."),
    Term("FCF-yield", "Fritt kassaflöde ÷ börsvärde — bolagets \"ränta\"."),
    Term("F-score", "Kvalitetsbetyg 0–9 på räkenskaperna. Under 5 = varning."),
    Term("Nettoskuld/EBITDA", "Antal årsvinster för att betala av skulden. "
                              "Negativ = nettokassa."),
    Term("AISC / C1 / breakeven", "Total / kontant produktionskostnad per "
                                  "enhet / bolagets smärtgräns."),
    Term("Runway", "Kassa ÷ årsburn = år till pengarna är slut. Nyemission "
                   "= utspädning."),
    Term("NAV / P/NAV", "Nuvärdet av projektets kassaflöden / priset mot det."),
    Term("Lassonde-kurvan", "Upptäcktsrusning → öknenvandring → omvärdering "
                            "mot produktion."),
    Term("PEA → PFS → FS → FID", "Studietrappan: idé → trolig → byggklar "
                                 "→ byggbeslut."),
    Term("GEO", "Gold Equivalent Ounces — royaltybolagens produktionsmått."),
    Term("Free ride", "Sälj halva vid +100 % — resten åker på husets pengar."),
    Term("Incitamentspris", "Priset som krävs för att ny produktion ska byggas."),
    Term("Kostnadskurva", "Producenter billigast → dyrast. De dyraste dör "
                          "först i prisfall."),
    Term("Guld/silver-kvoten", "Guld ÷ silver. Över 85 = silver billigt, "
                               "under 50 = dyrt."),
    Term("Realränta", "Ränta minus inflation — guldets huvudmotor."),
    Term("TC-avgifter", "Smältverkens förädlingsavgift — zinkcykelns "
                        "termometer."),
    Term("Offtake-avtal", "Kund förhandsköper produktion — utvecklarens "
                          "överlevnadsgaranti."),
    Term("MA200 / MA50 / RSI", "Trendlinjer (snittkurs 200/50 dagar) / "
                               "styrkemätare 0–100."),
    Term("R-multipel", "Affärens resultat mätt i risk-enheter (entry till "
                       "stop) — ärligaste måttet."),
)


# ── Riskdoktrinen (Masterguiden 4.0, Del 2) ──────────────────────────────────
# De tre förlusttyperna. Poängen är att bara en av dem är dödlig, och att alla
# kontrollsystem finns för att förhindra just den.

@dataclass(frozen=True)
class LossType:
    name: str
    what: str
    response: str


LOSS_TYPES: tuple[LossType, ...] = (
    LossType("Marknadsförlust",
             "Priset faller men tesen är intakt.",
             "Uthärdas — eller stoppas mekaniskt i swing."),
    LossType("Modellförlust",
             "Antagandet var fel.",
             "Omvärdera från noll. Oftast sälj."),
    LossType("Permanent kapitalförlust",
             "Skuld, utspädning eller tillgångsförstörelse gör att kapitalet "
             "aldrig kommer tillbaka.",
             "Den enda förlust systemet inte tål — och den alla kontroll"
             "system finns för att förhindra."),
)

# Två regler som gäller över alla strategier.
AVERAGING_RULE = (
    "Snittningsregeln: snitta aldrig ner bara för att priset fallit. "
    "Snittning kräver att ursprungstesen är giltig OCH att kvalitets- och "
    "utspädningskontrollerna inte försämrats.")

TOOL_RULE = (
    "Verktygsregeln: inget enskilt verktyg får ensamt ge ett automatiskt köp. "
    "Kandidaten måste klara strategi, kvalitet/värdering och relevant trigger.")

# Kontrollsignalerna — vad ett dåligt utfall i varje kontroll ska leda till.
CONTROL_SIGNALS: tuple[tuple[str, str], ...] = (
    ("AQS svag", "Kräv rabatt — eller passa."),
    ("DS hög", "Mindre position, eller vänta på finansieringsbeskedet."),
    ("CSM Bear-katastrof", "Passa."),
    ("Kontrollerna försämrade", "Snitta aldrig ner."),
)

# De nio frågorna varje större investering ska kunna svara på (Del 7).
NINE_QUESTIONS: tuple[str, ...] = (
    "Varför är sektorn attraktiv just nu?",
    "Är själva tillgången bra?",
    "Vad händer om råvarupriset går åt fel håll?",
    "Hur mycket kapital krävs till nästa katalysator?",
    "Hur mycket kan jag bli utspädd?",
    "Vilken konkret händelse ger omvärdering?",
    "När säljer jag om jag har rätt?",
    "När säljer jag om jag har fel?",
    "Hur stor får positionen vara innan den hotar hela systemet?",
)


# ── Uppslag ──────────────────────────────────────────────────────────────────
def sell_rule(key: str) -> SellRule | None:
    """The sell row for a playbook key, or None."""
    for s in SELL_RULES:
        if s.key and s.key == key:
            return s
    return None


def screener(key: str) -> Screener | None:
    for s in SCREENERS:
        if s.key and s.key == key:
            return s
    return None


def find_terms(query: str) -> list[Term]:
    """Case-insensitive substring search over term and meaning."""
    q = (query or "").strip().lower()
    if not q:
        return list(GLOSSARY)
    return [t for t in GLOSSARY if q in t.term.lower() or q in t.meaning.lower()]
