"""
rotation.py — Råvarurotationen (Masterguiden Del 3).

The layer that decides WHERE the contrarian strategies hunt. Rule, Sprott and
Durrett all need a hated sector to work in; this is what picks it.

  "Poängen med att bevaka 14 råvaror är inte att äga alla — det är att alltid
   ha någon sektor som är hatad."

Each month every commodity is graded on the Triple Signal (Masterguiden 4.0):
hatred, fundamentals and catalyst 1–5 each, plus case intakt Ja/Nej as a hard
gate. The sum 3–15 decides: 13–15 AGERA, 10–12 Bevaka, 9 or below Vila. Capital
goes to the 2–3 highest with intact cases. Gold and the royalty leg stay put
regardless of grade — gold is the only commodity that rises in risk aversion.

The point of three axes rather than one: hatred alone cannot tell a bottom from
a value trap. A sector can be despised because nobody wants it yet (buy) or
because the case is gone (never buy), and the fundamentals axis is what
separates them. Both failure modes raise their own warning badge.

Upgraded from the 3.x model (hat 1–5 + timing Ja/Delvis/Nej); saved grades are
migrated on read by migrate_grade().

NOTE: the 4.0 thresholds here are the spec's. The 3.x priority formula was this
module's own construction and has been replaced by the published one.
"""

from __future__ import annotations

import streamlit as st
from dataclasses import dataclass
from datetime import date
from html import escape as _esc

# Del 5 — the deep-dive behind each row. Kept in its own module so this one
# stays the grading engine; commodity_book is keyed by COMMODITIES.
import commodity_book

import csv_export

try:
    from gist_storage import load_blob as _blob_load, save_blob as _blob_save
    _HAS_GIST = True
except Exception:
    _HAS_GIST = False

_STORE_FILE = "rotation_data.json"
_CACHE_KEY = "rotation_data"

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, CYAN, GOLD = "#2d8a4e", "#d4943a", "#c44545", "#00E5FF", "#c9a84c"
BORDER = "#2a2a38"

# ── Statuslägen ──────────────────────────────────────────────────────────────
AGERA, BEVAKA, VILA = "AGERA", "Bevaka", "Vila"
STATUS_COLOR = {AGERA: GREEN, BEVAKA: AMBER, VILA: DIM}

# ── Triple Signal (Masterguiden 4.0) ─────────────────────────────────────────
# Tre axlar 1–5 i stället för hat + timing. Summan 3–15 avgör status. Till
# skillnad från 3.x-modellen är trösklarna guidens, inte panelens.
SIGNALS = (
    ("hatred", "Hat",
     "Hur avskydd är sektorn? Femkryss-checklistan nedan."),
    ("fundamentals", "Fundamenta",
     "Utbuds- och efterfrågecasets styrka. Finns ett skäl att priset ska upp?"),
    ("catalyst", "Katalysator",
     "Konkret, tidsatt mekanism — LNG-våg, kontraktscykel, kapacitetsstängning."),
)
SIGNAL_MIN, SIGNAL_MAX = 1, 5
SUM_MIN, SUM_MAX = 3, 15

AGERA_MIN = 13           # 13–15 AGERA
BEVAKA_MIN = 10          # 10–12 Bevaka, <= 9 Vila
CAPITAL_SLOTS = 3        # "de 2–3 mest hatade med intakta case"

# Hat-checklistan. Guiden 4.0: "Antal Ja ger poängen" — hatred är alltså inte
# en bedömning utan en räkning, och kryssen är fältet.
HATRED_CHECKLIST = (
    ("pris", "Priset ligger under incitamentspriset"),
    ("utbud", "Utbudet krymper av nöd — stängningar, inte planer"),
    ("kapital", "Kapitalet är stängt: inga emissioner, inga nya projekt"),
    ("lista", "Screener-listan är lång"),
    ("media", "Media är tyst eller negativ"),
)


def hatred_from_checklist(checks: dict) -> int:
    """Antal Ja, klämt till skalan 1–5.

    Noll kryss ger 1, inte 0: guiden anger dimensionerna som 1–5, så en helt
    okryssad checklista är skalans botten och inte ett eget nolläge.
    """
    n = sum(1 for key, _text in HATRED_CHECKLIST if (checks or {}).get(key))
    return max(SIGNAL_MIN, min(SIGNAL_MAX, n))

# ── Varningsbadges (Masterguiden 4.0) ────────────────────────────────────────
# Två sätt att ha hög summa av fel skäl.
WARN_VALUE_TRAP = "VÄRDEFÄLLA — hög hat utan case"
WARN_NOT_CONTRARIAN = "Ej kontrarisk — ägs via momentum/kvalitet"
WARN_COLOR = {WARN_VALUE_TRAP: RED, WARN_NOT_CONTRARIAN: AMBER}

# Den gamla timing-skalan behålls enbart för att kunna migrera sparad data.
TIMING_YES, TIMING_PARTLY, TIMING_NO = "Ja", "Delvis", "Nej"
LEGACY_TIMING_TO_CATALYST = {TIMING_YES: 3, TIMING_PARTLY: 2, TIMING_NO: 1}
LEGACY_FUNDAMENTALS_DEFAULT = 3    # sätts manuellt efter migreringen


@dataclass(frozen=True)
class Commodity:
    key: str
    name: str
    engine: str          # cykelmotor
    buy_signal: str      # hatad när… (köpsignal)
    anchor: bool = False # ligger fast som stabilisator oavsett betyg


# ── Master-tabellen (Masterguiden Del 3, sida 7) ─────────────────────────────
COMMODITIES: tuple[Commodity, ...] = (
    Commodity("guld", "Guld",
              "Realräntor, centralbanksköp, valutaförsvagning",
              "ETF-utflöden; gruvor på FCF-yield > 10 % trots högt guld", True),
    Commodity("silver", "Silver",
              "Guldets högbeta + industri (sol ~30 %)",
              "Guld/silver-kvot > 85–90"),
    Commodity("platina", "Platina",
              "SA ~70 % av utbud; substitution IN; vätgas-option",
              "Pris under SA-gruvornas AISC, schakt stängs"),
    Commodity("palladium", "Palladium",
              "Bensinkatalysatorer; EV = strukturell motvind",
              "ENDAST trade: −60 %+ och utbudskatalysator"),
    Commodity("uran", "Uran",
              "Kärnkraftverkens kontraktscykel",
              "Spot under incitamentspris (~$80–90/lb)"),
    Commodity("olja", "Olja",
              "OPEC + skifferdisciplin; capex-cykler",
              "Riggantal kollapsat, capex nedskuret"),
    Commodity("gas", "Gas",
              "Väder + LNG-export; regionalt pris",
              "Under torrgas-breakeven (~$2,5 HH), volymer stängs in"),
    Commodity("kol", "Kol",
              "ESG = permanent kapitalbrist",
              "Alltid hatad — köp vid FCF-yield > 20 %"),
    Commodity("koppar", "Koppar",
              "Elektrifiering möter tom pipeline",
              "Under incitament (~$4,5/lb), stigande LME-lager"),
    Commodity("zink", "Zink",
              "Smältverk vs gruvutbud; 2–4-årscykel",
              "TC-avgifter toppar, gruvor stänger"),
    Commodity("jarnmalm", "Järnmalm",
              "Kinas stål/fastigheter",
              "Total Kina-pessimism, pris ~$70–80/t"),
    Commodity("litium", "Litium",
              "EV-tillväxt möter utbudsvågor; bubblor",
              "−70 %+ från topp, projekt pausas"),
    Commodity("royalty", "Royalty",
              "Följer metallsentimentet, bottnar först",
              "Nedre kvartilen av egen P/NAV-historik", True),
)

COMMODITY_BY_KEY = {c.key: c for c in COMMODITIES}

# The guide's text says "fjorton råvaror" in four places, but the master table
# on page 7 lists thirteen. Nothing was lost extracting it — the rows are all
# there. Flagged rather than invented: add the missing one here if you have it.
DOCUMENTED_COUNT = 13
GUIDE_CLAIMS = 14

NOTE_PRECIOUS = ("Ädelmetallerna skiljer sig fundamentalt: guld och silver "
                 "konsumeras inte — deras cykler är monetära (räntor, valutor, "
                 "rädsla), inte lager och fabriksefterfrågan. Platina och "
                 "palladium är hybrider: ädelmetallpris, industrimetallcykel.")


# ── Rena beräkningar ─────────────────────────────────────────────────────────
def _num(value, default=None):
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _signal(value) -> int:
    """En axel klämd till 1–5. Osatt räknas som 1 — inte som noll."""
    v = _num(value)
    if v is None:
        return SIGNAL_MIN
    return max(SIGNAL_MIN, min(SIGNAL_MAX, int(v)))


def migrate_grade(grade: dict) -> dict:
    """3.x-betyg (hat + timing) -> 4.0 Triple Signal.

    hatred behåller hat-poängen, catalyst kommer ur timingen (Ja/Delvis/Nej ->
    3/2/1) och fundamentals sätts till 3. Trean är en platshållare, inte en
    bedömning: den gamla modellen frågade aldrig om caset, så den siffran finns
    inte att migrera. Raden märks som migrerad tills den rörts för hand.
    """
    g = dict(grade or {})
    if "hatred" in g:
        return g
    if "hat" not in g and "timing" not in g:
        return g
    g["hatred"] = _signal(g.get("hat"))
    g["fundamentals"] = LEGACY_FUNDAMENTALS_DEFAULT
    g["catalyst"] = LEGACY_TIMING_TO_CATALYST.get(g.get("timing"), 1)
    g["migrated"] = True
    return g


def migrate_grades(grades: dict) -> dict:
    return {k: migrate_grade(v) for k, v in (grades or {}).items()}


def signal_sum(grade: dict) -> int:
    """Triple Signal-summan 3–15."""
    g = grade or {}
    return sum(_signal(g.get(key)) for key, _label, _help in SIGNALS)


def priority(grade: dict) -> float:
    """Prioritet = Triple Signal-summan. Brutet case ger alltid 0."""
    if not bool((grade or {}).get("case_intact", True)):
        return 0.0
    return float(signal_sum(grade))


def status(grade: dict) -> tuple[str, str]:
    """(status, motivering) — AGERA kör screenern, Bevaka väntar, Vila avstår.

    Case intakt är en hård grind före summan: ett brutet case kan inte köpas
    hur hatat, välmotiverat eller tidsatt det än är.
    """
    g = grade or {}
    if not bool(g.get("case_intact", True)):
        return VILA, "Caset är brutet — ingen screener, oavsett summa"
    total = signal_sum(g)
    if total >= AGERA_MIN:
        return AGERA, f"Summa {total}/{SUM_MAX} — hat, case och katalysator drar åt samma håll"
    if total >= BEVAKA_MIN:
        return BEVAKA, f"Summa {total}/{SUM_MAX} — en av de tre saknas ännu"
    return VILA, f"Summa {total}/{SUM_MAX} — vila"


def warnings(grade: dict) -> list:
    """Två sätt att ha en hög summa av fel skäl."""
    g = grade or {}
    h, f = _signal(g.get("hatred")), _signal(g.get("fundamentals"))
    out = []
    if h >= 4 and f <= 2:
        out.append(WARN_VALUE_TRAP)
    if f >= 4 and h <= 2:
        out.append(WARN_NOT_CONTRARIAN)
    return out


def ranked(grades: dict) -> list:
    """Alla råvaror sorterade på prioritet, högst först."""
    out = []
    graded = migrate_grades(grades)
    for c in COMMODITIES:
        g = graded.get(c.key, {}) or {}
        intact = bool(g.get("case_intact", True))
        st_, why = status(g)
        row = {"commodity": c, "case_intact": intact, "grade": g,
               "sum": signal_sum(g), "priority": priority(g),
               "status": st_, "why": why, "warnings": warnings(g),
               "screener_hits": _num(g.get("screener_hits")),
               "migrated": bool(g.get("migrated"))}
        for key, _label, _help in SIGNALS:
            row[key] = _signal(g.get(key))
        out.append(row)
    out.sort(key=lambda r: (-r["priority"], r["commodity"].name))
    return out


def capital_targets(grades: dict, slots: int = CAPITAL_SLOTS) -> list:
    """De 2–3 mest hatade med intakta case — dit kapitalet går."""
    agera = [r for r in ranked(grades) if r["status"] == AGERA]
    return agera[:slots]


def anchors() -> list:
    """Guld- och royaltybenet ligger fast oavsett betyg."""
    return [c for c in COMMODITIES if c.anchor]


# ── Lagring ──────────────────────────────────────────────────────────────────
def _month() -> str:
    return date.today().strftime("%Y-%m")


def _default() -> dict:
    return {"month": _month(), "grades": {}, "history": []}


def _load() -> dict:
    if _CACHE_KEY in st.session_state:
        return st.session_state[_CACHE_KEY]
    data = _blob_load(_STORE_FILE, None) if _HAS_GIST else None
    if not isinstance(data, dict):
        data = _default()
    for k, v in _default().items():
        data.setdefault(k, v)
    if not isinstance(data.get("grades"), dict):
        data["grades"] = {}
    if not isinstance(data.get("history"), list):
        data["history"] = []
    st.session_state[_CACHE_KEY] = data
    return data


def _save(data: dict) -> None:
    st.session_state[_CACHE_KEY] = data
    if _HAS_GIST:
        _blob_save(_STORE_FILE, data)


# ── UI ───────────────────────────────────────────────────────────────────────
def render_rotation_page() -> None:
    """Huvud-entry point för Råvarurotationen."""
    try:
        data = _load()
        st.markdown(
            f"<h1 style='color:{TEXT};margin:0;letter-spacing:0.06em;'>"
            f"Råvarurotationen <span style='color:{GOLD};'>· vart kapitalet ska</span></h1>"
            f"<p style='color:{DIM};font-size:0.8rem;margin:6px 0 12px;'>"
            f"Poängen är inte att äga alla — det är att alltid ha någon sektor som är "
            f"hatad. Betygsätts en gång i månaden; kapitalet går till de 2–3 mest "
            f"hatade med intakta case.</p>", unsafe_allow_html=True)

        _export(data)
        _month_header(data)
        _targets(data)
        _grid(data)
        _reference()
    except Exception as e:
        st.error(f"Råvarurotationen kunde inte renderas: {e}")


ROT_CSV = [("_name", "Råvara"), ("hatred", "Hat (1–5)"),
           ("fundamentals", "Fundamenta (1–5)"), ("catalyst", "Katalysator (1–5)"),
           ("sum", "Summa (3–15)"), ("case_intact", "Case intakt"),
           ("_status", "Status"), ("_why", "Motivering"),
           ("_warnings", "Varningar"), ("screener_hits", "Screener-träffar"),
           ("_engine", "Cykelmotor"), ("_signal", "Hatad när")]


def _export(data: dict) -> None:
    rows = []
    for r in ranked(data.get("grades", {})):
        c = r["commodity"]
        rows.append({"_name": c.name, "hatred": r["hatred"],
                     "fundamentals": r["fundamentals"], "catalyst": r["catalyst"],
                     "sum": r["sum"], "case_intact": r["case_intact"],
                     "_status": r["status"], "_why": r["why"],
                     "_warnings": " · ".join(r["warnings"]),
                     "screener_hits": r["screener_hits"],
                     "_engine": c.engine, "_signal": c.buy_signal})
    csv_export.download_button(rows, ROT_CSV, "ravarurotationen",
                               key="csv_rotation")


def _month_header(data: dict) -> None:
    cur = data.get("month", _month())
    c1, c2 = st.columns([1, 3])
    if cur != _month():
        with c1:
            if st.button(f"➜ Starta {_month()}", key="rot_new_month"):
                data["history"].append({"month": cur, "grades": dict(data["grades"])})
                data["history"] = data["history"][-24:]
                data["month"] = _month()
                _save(data)
                st.rerun()
        c2.warning(f"Betygen är från {cur}. Rotationen görs första helgen varje "
                   f"månad — starta {_month()} när du gör om den.")
    else:
        c1.markdown(f"<div style='color:{DIM};font-size:0.8rem;padding-top:6px;'>"
                    f"Betygsatt för <b style='color:{TEXT};'>{cur}</b></div>",
                    unsafe_allow_html=True)


def _targets(data: dict) -> None:
    targets = capital_targets(data.get("grades", {}))
    if targets:
        chips = "".join(
            f"<span style='background:{GREEN}18;border:1px solid {GREEN}66;"
            f"border-radius:5px;padding:4px 11px;margin-right:6px;'>"
            f"<b style='color:{GREEN};'>{t['commodity'].name}</b>"
            f"<span style='color:{DIM};font-size:0.72rem;'> "
            f"{t['sum']}/{SUM_MAX} · hat {t['hatred']} · fund {t['fundamentals']} "
            f"· kat {t['catalyst']}</span></span>" for t in targets)
        st.markdown(
            f"<div style='margin:8px 0 12px;'>"
            f"<div style='color:{DIM};font-size:0.68rem;letter-spacing:0.1em;"
            f"margin-bottom:5px;'>KAPITALET HIT — KÖR DESSA SCREENERS</div>"
            f"{chips}</div>", unsafe_allow_html=True)
    else:
        st.info("Ingen råvara på AGERA. Inget att jaga — bevaka månadsvis och låt "
                "kassan växa. Det är ett giltigt läge, inte ett misslyckande.")

    a = anchors()
    st.markdown(
        f"<div style='color:{DIM};font-size:0.72rem;margin-bottom:10px;'>"
        f"⚓ {' och '.join(c.name for c in a)}-benet ligger fast som stabilisator "
        f"oavsett betyg — guld är den enda råvaran som stiger i riskaversion.</div>",
        unsafe_allow_html=True)


def _book_section(title: str, body: str, color: str) -> str:
    return (f"<div style='margin-top:12px;'>"
            f"<div style='color:{color};font-size:0.63rem;font-weight:700;"
            f"letter-spacing:0.12em;text-transform:uppercase;'>{title}</div>"
            f"<div style='color:{TEXT};font-size:0.82rem;line-height:1.6;"
            f"margin-top:3px;'>{_esc(body)}</div></div>")


def chapter_html(ch) -> str:
    """Råvarukartboken (Del 5) — the case behind the one-liner above.

    The prose is escaped, not trusted as markup: it is full of threshold
    comparisons like "EV/EBITDA < 5 ... hedgebok > 40 %", and an unescaped
    run of those reads as an HTML tag and swallows everything between them.
    """
    html = (f"<div style='color:{DIM};font-size:0.72rem;font-style:italic;"
            f"margin-top:8px;'>{_esc(ch.subtitle)}</div>")
    html += _book_section("Marknaden", ch.market, CYAN)
    html += _book_section("Spelet", ch.play, GREEN)
    html += _book_section("Timing", ch.timing, AMBER)
    if ch.role:
        html += _book_section("Portföljroll", ch.role, GOLD)
    if ch.pitfall:
        html += (f"<div style='border-left:3px solid {RED};background:{RED}0d;"
                 f"border-radius:0 6px 6px 0;padding:8px 12px;margin-top:12px;"
                 f"color:{TEXT};font-size:0.8rem;line-height:1.55;'>"
                 f"⚠️ <b style='color:{RED};'>Fallgropen:</b> "
                 f"{_esc(ch.pitfall)}</div>")
    if ch.sources:
        items = "".join(f"<li style='margin-bottom:2px;'>{_esc(s)}</li>"
                        for s in ch.sources)
        html += (f"<div style='margin-top:12px;'>"
                 f"<div style='color:{DIM};font-size:0.63rem;font-weight:700;"
                 f"letter-spacing:0.12em;'>DÄR SIFFRAN FINNS</div>"
                 f"<ul style='color:{DIM};font-size:0.76rem;margin:4px 0 0 16px;"
                 f"padding:0;'>{items}</ul></div>")
    return html


def _chapter_block(ch) -> None:
    st.markdown(chapter_html(ch), unsafe_allow_html=True)


def _grid(data: dict) -> None:
    grades = data.setdefault("grades", {})
    rows = ranked(grades)
    changed = False

    st.markdown(f"<div style='color:{DIM};font-size:0.68rem;letter-spacing:0.1em;"
                f"margin:6px 0 4px;'>MÅNADENS BETYG — HAT · FUNDAMENTA · "
                f"KATALYSATOR (1–5 var) · CASE INTAKT</div>",
                unsafe_allow_html=True)

    st.caption("Hat räknas, det bedöms inte: kryssa checklistan under varje "
               "råvara så sätts siffran av antalet Ja. Fundamenta och "
               "katalysator är dina bedömningar.")

    for r in rows:
        c = r["commodity"]
        g = grades.setdefault(c.key, {})

        # Migrera raden på plats första gången den läses.
        migrated = migrate_grade(g)
        if migrated is not g and migrated != g:
            g.update(migrated)
            changed = True

        h1, h2, h3, h4, h5, h6, h7 = st.columns([1.4, .8, .8, .8, .7, .8, 2.0])
        anchor_mark = (f"<span style='color:{GOLD};font-size:0.7rem;'>⚓</span>"
                       if c.anchor else "")
        engine_short = c.engine[:38]
        h1.markdown(
            f"<div style='padding-top:6px;'>"
            f"<b style='color:{TEXT};'>{c.name}</b> {anchor_mark}"
            f"<div style='color:{DIM};font-size:0.66rem;'>{engine_short}</div></div>",
            unsafe_allow_html=True)

        opts = list(range(SIGNAL_MIN, SIGNAL_MAX + 1))
        # Hat räknas ur checklistan (i expandern nedan), inte ur en dropdown.
        hat_now = hatred_from_checklist(g.get("hatred_checks", {}))
        if g.get("hatred") != hat_now and g.get("hatred_checks") is not None:
            g["hatred"] = hat_now
            changed = True
        h2.markdown(
            f"<div style='padding-top:6px;text-align:center;'>"
            f"<span style='color:{TEXT};font-size:1.05rem;font-weight:700;'>"
            f"{_signal(g.get('hatred'))}</span>"
            f"<div style='color:{DIM};font-size:0.6rem;'>hat</div></div>",
            unsafe_allow_html=True)

        for (skey, slabel, shelp), col in zip(SIGNALS[1:], (h3, h4)):
            cur = _signal(g.get(skey))
            v = col.selectbox(slabel, opts, index=opts.index(cur),
                              key=f"rot_{skey}_{c.key}", help=shelp,
                              label_visibility="collapsed")
            if v != g.get(skey):
                g[skey] = v
                changed = True

        intact = h5.checkbox("Case", value=bool(g.get("case_intact", True)),
                             key=f"rot_case_{c.key}",
                             help="Hård grind: nej = Vila oavsett summa.")
        if intact != g.get("case_intact"):
            g["case_intact"] = intact
            changed = True

        hits = h6.number_input("Träffar", min_value=0, step=1,
                               value=int(_num(g.get("screener_hits"), 0) or 0),
                               key=f"rot_hits_{c.key}",
                               label_visibility="collapsed",
                               help="Screener-träffar denna månad. En lång "
                                    "lista är i sig ett hat-tecken.")
        if hits != g.get("screener_hits"):
            g["screener_hits"] = hits
            g.setdefault("hits_history", [])
            g["hits_history"] = (g["hits_history"] + [hits])[-24:]
            changed = True

        st_now, why_now = status(g)
        sc_now = STATUS_COLOR.get(st_now, DIM)
        warn_html = "".join(
            f"<div style='color:{WARN_COLOR.get(w, DIM)};font-size:0.66rem;"
            f"margin-top:2px;'>⚠️ {w}</div>" for w in warnings(g))
        mig_html = (f"<div style='color:{AMBER};font-size:0.62rem;margin-top:2px;'>"
                    f"Migrerad från 3.x — fundamenta är en platshållare (3), "
                    f"sätt den själv</div>" if g.get("migrated") else "")
        h7.markdown(
            f"<div style='padding-top:4px;'>"
            f"<span style='background:{sc_now}22;border:1px solid {sc_now};"
            f"color:{sc_now};font-size:0.68rem;font-weight:700;padding:2px 9px;"
            f"border-radius:10px;'>{st_now} {signal_sum(g)}/{SUM_MAX}</span>"
            f"<div style='color:{DIM};font-size:0.66rem;margin-top:2px;'>{why_now}</div>"
            f"{warn_html}{mig_html}</div>", unsafe_allow_html=True)

        hist = g.get("hits_history") or []
        if len(hist) > 1:
            h6.line_chart(hist, height=40)

        with st.expander(f"Hat-checklistan — {c.name} "
                         f"({_signal(g.get('hatred'))}/5)", expanded=False):
            checks = g.setdefault("hatred_checks", {})
            for hkey, htext in HATRED_CHECKLIST:
                v = st.checkbox(htext, value=bool(checks.get(hkey)),
                                key=f"rot_hc_{c.key}_{hkey}")
                if v != bool(checks.get(hkey)):
                    checks[hkey] = v
                    g["hatred"] = hatred_from_checklist(checks)
                    changed = True
            st.caption("Antal Ja ger poängen. Noll kryss är skalans botten "
                       "(1), inte ett eget nolläge.")

        ch = commodity_book.chapter(c.key)
        label = (f"📖 Kartboken — {c.name}" if ch
                 else f"Köpsignal — {c.name}")
        with st.expander(label, expanded=False):
            st.markdown(f"<div style='color:{TEXT};font-size:0.82rem;'>"
                        f"<b>Cykelmotor:</b> {c.engine}<br>"
                        f"<b>Hatad när:</b> {c.buy_signal}</div>",
                        unsafe_allow_html=True)
            if ch:
                _chapter_block(ch)

    if changed:
        _save(data)


def _reference() -> None:
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.caption(NOTE_PRECIOUS)
    if DOCUMENTED_COUNT != GUIDE_CLAIMS:
        st.caption(f"⚠️ Masterguidens text säger {GUIDE_CLAIMS} råvaror, men "
                   f"master-tabellen (Del 3, sida 7) listar {DOCUMENTED_COUNT}. "
                   f"Här finns de {DOCUMENTED_COUNT} som står i tabellen — lägg till "
                   f"den saknade i rotation.py om du vet vilken den är.")
    st.caption(f"Triple Signal (Masterguiden 4.0): hat + fundamenta + "
               f"katalysator, {SIGNAL_MIN}–{SIGNAL_MAX} var. "
               f"{AGERA_MIN}–{SUM_MAX} AGERA · {BEVAKA_MIN}–{AGERA_MIN - 1} "
               f"Bevaka · {BEVAKA_MIN - 1} och under Vila. Brutet case är en "
               f"hård grind före summan.")
