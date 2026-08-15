"""
routines.py — Årshjulet och rutinerna (Masterguiden Del 6).

The layer that says WHEN. The playbooks say what a good trade looks like, the
allocator says how much, the rotation says where — this says which day you sit
down and do it, and for how long.

  "Härifrån skapas avkastningen inte av fler dokument — utan av exekvering.
   Kör rutinerna. Logga affärerna. Läs journalen. Vänta på dina lägen."

The five cadences, the quarterly ritual, the journal rules and the 10-week
onboarding order are the guide's, verbatim where it publishes numbers.

NOTE on the due-date logic: the guide names the cadences ("varje söndag",
"första helgen i månaden", "kvartalsvis — fast datum") but never fixes the
calendar anchors, because the quarterly date is the reader's to choose.
`due_on()` and `next_due()` below are therefore this module's construction:
Sunday for the weekly, the first weekend of the month for the monthly, the
first weekend of Jan/Apr/Jul/Oct for the quarterly, and January for the yearly.
Change the anchors here if your fixed date is another one — they are not the
guide's numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

# ── Kadenser ─────────────────────────────────────────────────────────────────
WEEKLY = "weekly"
MONTHLY = "monthly"
QUARTERLY = "quarterly"
YEARLY = "yearly"
ON_EVENT = "on_event"          # inte kalenderstyrd — utlöses av nytt kapital

CADENCE_ORDER = (WEEKLY, MONTHLY, QUARTERLY, YEARLY, ON_EVENT)

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, CYAN, GOLD = "#2d8a4e", "#d4943a", "#c44545", "#00E5FF", "#c9a84c"

QUARTER_MONTHS = (1, 4, 7, 10)
YEARLY_MONTH = 1

# Statuslägen för dagens vy
DUE, SOON, VILANDE = "DAGS", "Snart", "Vilande"
STATUS_COLOR = {DUE: GREEN, SOON: AMBER, VILANDE: DIM}
SOON_DAYS = 7                  # så många dagar i förväg vi flaggar "Snart"


@dataclass(frozen=True)
class Step:
    """One line of work inside a routine."""
    text: str
    panel: str = ""            # var i panelen den görs — tom = utanför panelen


@dataclass(frozen=True)
class Routine:
    key: str
    cadence: str
    when: str                  # guidens formulering, ordagrant
    title: str
    minutes: str
    tools: str
    color: str
    steps: tuple[Step, ...] = field(default_factory=tuple)
    note: str = ""


# ── Årshjulet ────────────────────────────────────────────────────────────────
# Tabellen på sida 30, rad för rad. `steps` bryter ut vad raden faktiskt
# innebär, med panelvägen där panelen gör jobbet åt dig.

ROUTINES: tuple[Routine, ...] = (
    Routine(
        key="weekly",
        cadence=WEEKLY,
        when="Varje söndag",
        title="Swing-rutinen + insider-skanningen",
        minutes="45–60 min",
        tools="Screeners, bevakaren, journalen, panelen",
        color=GREEN,
        steps=(
            Step("Kolla marknadsfiltret först — rött ljus låser alla köp, "
                 "då är söndagen kort.",
                 "REGIME → Swing Regime"),
            Step("Kör momentum-screenern och gå igenom topp 40.",
                 "SCREENING → Swing Screener"),
            Step("Uppdatera bevakningslistan: skicka nya kandidater dit, "
                 "rensa dem som tappat setupen.",
                 "PORTFOLIO → Swing"),
            Step("Gå igenom öppna positioner mot säljreglerna: stängning under "
                 "MA50, −10 % stop, ur topp 40, +20 % = sälj halva.",
                 "PORTFOLIO → Swing"),
            Step("Skanna veckans insynsköp — kluster, inte enstaka köp.",
                 ""),
            Step("Logga veckans affärer om något inte redan är loggat.",
                 "PORTFOLIO → Trade Journal"),
        ),
        note="Veckans enda fasta möte med marknaden. Klarar du bara en rutin "
             "är det den här.",
    ),
    Routine(
        key="monthly",
        cadence=MONTHLY,
        when="Första helgen i månaden",
        title="Rotationen: Triple Signal + kontroller på seriösa kandidater",
        minutes="1–2 h",
        tools="Rotation-fliken, poängmodeller",
        color=GOLD,
        steps=(
            Step("Triple Signal per råvara: hat räknas ur checklistan, "
                 "fundamenta och katalysator bedöms 1–5, case intakt Ja/Nej.",
                 "REGIME → Råvarurotation"),
            Step("Läs av vilka som hamnar på AGERA — kapitalet går till de 2–3 "
                 "mest hatade med intakta case.",
                 "REGIME → Råvarurotation"),
            Step("Kör screenern för de råvarorna och poängsätt kandidaterna.",
                 ""),
            Step("Kör AQS, DS och CSM på de seriösa kandidaterna — i den "
                 "omfattning proportionalitetsregeln kräver.",
                 "GRANSKNING → 🎯 Scorecard"),
            Step("Guld och royaltybenet ligger kvar oavsett betyg — de "
                 "roterar inte.",
                 "REGIME → Råvarurotation"),
        ),
        note="Inget på AGERA är ett giltigt utfall. Då är rutinen klar på "
             "20 minuter och du gör ingenting.",
    ),
    Routine(
        key="quarterly",
        cadence=QUARTERLY,
        when="Kvartalsvis (fast datum)",
        title="Portföljgenomgången + rapportuppdateringar",
        minutes="2–3 h",
        tools="Portföljallokeraren, alla granskningsark",
        color=CYAN,
        steps=(
            Step("Kvartalsritualen — de fyra stegen nedan.",
                 "PORTFOLIO → Allokering"),
            Step("Rapportuppdatering: kassa och burn rate ur kvartalsrapporten.",
                 ""),
            Step("Rapportuppdatering: produktion och AISC ur "
                 "bolagspresentationen.",
                 ""),
            Step("Rapportuppdatering: katalysatorerna — har någon flyttats?",
                 ""),
            Step("Kör Tiggre-screenern och uppdatera Lobo-arket.",
                 "GRANSKNING → Tiggre"),
        ),
        note="Fast datum, satt i förväg. Kvartalsgenomgången är det enda "
             "tillfället då ramarna får ändras.",
    ),
    Routine(
        key="yearly",
        cadence=YEARLY,
        when="Årligen",
        title="Prisriktmärken, Fraser-rankingen, skattesetup, backtest-omkörning",
        minutes="En kväll",
        tools="Handboken, backtestern",
        color=AMBER,
        steps=(
            Step("Uppdatera prisriktmärkena för varje råvara — "
                 "incitamentspriser flyttar sig.",
                 "REGIME → Råvarurotation"),
            Step("Läs årets Fraser-ranking och stryk jurisdiktioner som "
                 "tappat.",
                 ""),
            Step("Se över skattesetupen inför nästa år.",
                 ""),
            Step("Kör om backtestet på swing-reglerna med ett år mer data.",
                 "PORTFOLIO → Backtest"),
        ),
    ),
    Routine(
        key="new_capital",
        cadence=ON_EVENT,
        when="Vid nytt kapital",
        title="Alltid till strategin längst under målet",
        minutes="5 min",
        tools="Allokering-fliken",
        color=DIM,
        steps=(
            Step("Läs av vilket ben som ligger längst under sin målprocent.",
                 "PORTFOLIO → Allokering"),
            Step("Pengarna dit — inte till det som gått bäst senast.",
                 "PORTFOLIO → Allokering"),
        ),
        note="Den enda rutinen utan datum. Den utlöses av lön, utdelning "
             "eller en försäljning — inte av kalendern.",
    ),
)

ROUTINES_BY_KEY = {r.key: r for r in ROUTINES}


# ── Kvartalsritualen (sida 30) ───────────────────────────────────────────────

@dataclass(frozen=True)
class RitualStep:
    number: int
    title: str
    minutes: int
    body: str


QUARTERLY_RITUAL: tuple[RitualStep, ...] = (
    RitualStep(1, "Siffrorna", 20,
               "Uppdatera Portföljallokeraren och ombalansera mot ramarna."),
    RitualStep(2, "Beteendet", 30,
               "Läs journalens lärdomskolumn. Leta DITT mönster: sena köp? "
               "brutna säljregler? för stora favoritpositioner? Skriv "
               "kvartalets viktigaste lärdom — en mening räcker."),
    RitualStep(3, "Strategihälsan", 20,
               "En strategi som följt reglerna men gått dåligt ETT kvartal "
               "ändras inte — cyklerna är långa. Dåligt i 4+ kvartal trots "
               "följda regler → pausa och backtesta om."),
    RitualStep(4, "Ramarna", 10,
               "Livssituationen ändrad → justera målprocenten. Tillåtet "
               "kvartalsvis — aldrig mitt i ett kvartal, aldrig som reaktion "
               "på en kursrörelse."),
)

RITUAL_TOTAL_MIN = 60
RITUAL_TOTAL_MAX = 90


# ── Journalen och statistiken (sida 30) ──────────────────────────────────────

MIN_TRADES_FOR_STATS = 15      # "dra inga slutsatser under 15–20 affärer"

JOURNAL_RULES: tuple[str, ...] = (
    "Varje affär loggas <b>samma dag</b> — inte i efterhand, inte i klump.",
    "Journalen räknar R-multipeln: resultatet i risk-enheter. Det ärligaste "
    "måttet, eftersom det mäter mot vad du faktiskt riskerade.",
    f"Dra inga slutsatser av statistiken under {MIN_TRADES_FOR_STATS}–20 "
    "affärer — vinstandel och payoff-kvot är brus dessförinnan.",
    "I början är lärdomskolumnen värd mer än siffrorna.",
)

BACKTEST_RULES: tuple[str, ...] = (
    "Kör parameter-svepet och kräv <b>platta berg</b> i parameterrymden. "
    "Spetsiga toppar = kurvanpassning, inte edge.",
    "Granska 10 slumpade affärer manuellt innan du litar på summorna.",
    "Dra mentalt 1–3 procentenheter för survivorship bias.",
)


# ── Kom igång-ordningen (sida 30) ────────────────────────────────────────────

ONBOARDING: tuple[tuple[str, str], ...] = (
    ("Vecka 1", "Sätt upp screeners och mallar."),
    ("Vecka 2–9", "Pappershandla eller granska i lugn takt."),
    ("Vecka 10", "Börja smått med riktiga pengar — i det som känns mest "
                 "begripligt."),
)

CLOSING_LINE = ("Systemet ska kännas tråkigt och repetitivt — det är exakt så "
                "det ska kännas när det fungerar.")


# ── Kalenderlogiken ──────────────────────────────────────────────────────────
# Se modulens docstring: ankarpunkterna är den här modulens konstruktion.

def _is_first_weekend(day: date) -> bool:
    """First Saturday or Sunday of the month."""
    return day.weekday() in (5, 6) and day.day <= 7


def due_on(key: str, day: date) -> bool:
    """True if the routine falls on `day`.

    Event-driven routines (new capital) are never calendar-due.
    """
    r = ROUTINES_BY_KEY.get(key)
    if r is None:
        return False
    if r.cadence == WEEKLY:
        return day.weekday() == 6
    if r.cadence == MONTHLY:
        return _is_first_weekend(day)
    if r.cadence == QUARTERLY:
        return day.month in QUARTER_MONTHS and _is_first_weekend(day)
    if r.cadence == YEARLY:
        return day.month == YEARLY_MONTH and _is_first_weekend(day)
    return False


def next_due(key: str, day: date) -> date | None:
    """The next date this routine is due, counting `day` itself.

    Returns None for event-driven routines. Searches at most ~15 months, which
    covers every cadence here.
    """
    r = ROUTINES_BY_KEY.get(key)
    if r is None or r.cadence == ON_EVENT:
        return None
    for offset in range(0, 460):
        d = day + timedelta(days=offset)
        if due_on(key, d):
            return d
    return None


def days_until(key: str, day: date) -> int | None:
    """Days until the routine is next due. 0 = today. None if event-driven."""
    nxt = next_due(key, day)
    return None if nxt is None else (nxt - day).days


def status(key: str, day: date) -> str:
    """DAGS / Snart / Vilande for `day`."""
    left = days_until(key, day)
    if left is None:
        return VILANDE
    if left == 0:
        return DUE
    return SOON if left <= SOON_DAYS else VILANDE


def agenda(day: date) -> list[dict]:
    """Every routine with its status for `day`, most urgent first."""
    rows = []
    for r in ROUTINES:
        left = days_until(r.key, day)
        rows.append({
            "routine": r,
            "status": status(r.key, day),
            "next_due": next_due(r.key, day),
            "days_until": left,
        })
    # Event-driven last, then soonest first.
    rows.sort(key=lambda x: (x["days_until"] is None,
                             x["days_until"] if x["days_until"] is not None else 0))
    return rows


def due_today(day: date) -> list[Routine]:
    """The routines to actually do on `day`."""
    return [r for r in ROUTINES if due_on(r.key, day)]


# ── Svenska datum ────────────────────────────────────────────────────────────
# strftime follows the server locale, which on Streamlit Cloud is C — that
# would put "Sunday 4 October" in the middle of a Swedish panel.

WEEKDAYS_SV = ("måndag", "tisdag", "onsdag", "torsdag", "fredag",
               "lördag", "söndag")
MONTHS_SV = ("januari", "februari", "mars", "april", "maj", "juni", "juli",
             "augusti", "september", "oktober", "november", "december")


def fmt_date(day: date, with_year: bool = False) -> str:
    """'4 oktober' — or '4 oktober 2026'."""
    out = f"{day.day} {MONTHS_SV[day.month - 1]}"
    return f"{out} {day.year}" if with_year else out


def fmt_weekday(day: date) -> str:
    """'söndag 4 oktober'."""
    return f"{WEEKDAYS_SV[day.weekday()]} {fmt_date(day)}"
