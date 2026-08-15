"""
rotation.py — Råvarurotationen (Masterguiden Del 3).

The layer that decides WHERE the contrarian strategies hunt. Rule, Sprott and
Durrett all need a hated sector to work in; this is what picks it.

  "Poängen med att bevaka 14 råvaror är inte att äga alla — det är att alltid
   ha någon sektor som är hatad."

Each month every commodity is graded: hat-poäng 1–5, case intakt Ja/Nej,
timing-signal Ja/Delvis/Nej. Capital goes to the 2–3 most hated with intact
cases. Gold and the royalty leg stay put regardless of grade — gold is the only
commodity that rises in risk aversion.

NOTE on the priority formula: the guide says "arket räknar prioritet" but never
publishes how. PRIORITY and the AGERA/Bevaka/Vila thresholds below are therefore
this module's construction, chosen to match the behaviour the guide describes
(a broken case can never be AGERA; high hate plus some timing confirmation is
what triggers a screener run). Adjust them here if your sheet differs — they are
not the guide's numbers.
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

# Timing-signalen
TIMING_YES, TIMING_PARTLY, TIMING_NO = "Ja", "Delvis", "Nej"
TIMING_BONUS = {TIMING_YES: 2, TIMING_PARTLY: 1, TIMING_NO: 0}

# Thresholds — this module's construction, not the guide's (see docstring).
HAT_AGERA_MIN = 4        # hat-poäng som krävs för AGERA
HAT_BEVAKA_MIN = 3
CAPITAL_SLOTS = 3        # "de 2–3 mest hatade med intakta case"


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


def priority(hat, timing: str, case_intact: bool) -> float:
    """Prioritet = hat-poäng + timing-bonus. Brutet case ger alltid 0.

    Not the guide's formula — see the module docstring.
    """
    if not case_intact:
        return 0.0
    h = _num(hat, 0) or 0
    h = max(1, min(5, int(h))) if h else 0
    return float(h) + TIMING_BONUS.get(timing, 0)


def status(hat, timing: str, case_intact: bool) -> tuple[str, str]:
    """(status, motivering) — AGERA kör screenern, Bevaka väntar, Vila avstår."""
    if not case_intact:
        return VILA, "Caset är brutet — ingen screener, oavsett hur hatad sektorn är"
    h = int(_num(hat, 0) or 0)
    if h >= HAT_AGERA_MIN and timing in (TIMING_YES, TIMING_PARTLY):
        return AGERA, "Hatad OCH timing bekräftar — kör screenern"
    if h >= HAT_AGERA_MIN:
        return BEVAKA, "Hatad men timing saknas — vänta på signalen"
    if h >= HAT_BEVAKA_MIN or timing == TIMING_YES:
        return BEVAKA, "På väg — bevaka månadsvis"
    return VILA, "Varken hatad nog eller timing — vila"


def ranked(grades: dict) -> list:
    """Alla råvaror sorterade på prioritet, högst först."""
    out = []
    for c in COMMODITIES:
        g = grades.get(c.key, {}) or {}
        hat = g.get("hat", 0)
        timing = g.get("timing", TIMING_NO)
        intact = bool(g.get("case_intact", True))
        st_, why = status(hat, timing, intact)
        out.append({
            "commodity": c, "hat": int(_num(hat, 0) or 0), "timing": timing,
            "case_intact": intact, "priority": priority(hat, timing, intact),
            "status": st_, "why": why,
        })
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


ROT_CSV = [("_name", "Råvara"), ("hat", "Hat-poäng"), ("timing", "Timing"),
           ("case_intact", "Case intakt"), ("_priority", "Prioritet"),
           ("_status", "Status"), ("_why", "Motivering"),
           ("_engine", "Cykelmotor"), ("_signal", "Hatad när")]


def _export(data: dict) -> None:
    rows = []
    for r in ranked(data.get("grades", {})):
        c = r["commodity"]
        rows.append({"_name": c.name, "hat": r["hat"], "timing": r["timing"],
                     "case_intact": r["case_intact"], "_priority": r["priority"],
                     "_status": r["status"], "_why": r["why"],
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
            f"<span style='color:{DIM};font-size:0.72rem;'> hat {t['hat']}/5 · "
            f"timing {t['timing']}</span></span>" for t in targets)
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
                f"margin:6px 0 4px;'>MÅNADENS BETYG — HAT 1–5 · CASE INTAKT · "
                f"TIMING</div>", unsafe_allow_html=True)

    for r in rows:
        c = r["commodity"]
        g = grades.setdefault(c.key, {})

        h1, h2, h3, h4, h5 = st.columns([1.5, 1, 1, 1.2, 2.2])
        anchor_mark = (f"<span style='color:{GOLD};font-size:0.7rem;'>⚓</span>"
                       if c.anchor else "")
        engine_short = c.engine[:38]
        h1.markdown(
            f"<div style='padding-top:6px;'>"
            f"<b style='color:{TEXT};'>{c.name}</b> {anchor_mark}"
            f"<div style='color:{DIM};font-size:0.66rem;'>{engine_short}</div></div>",
            unsafe_allow_html=True)
        hat = h2.selectbox("Hat", [0, 1, 2, 3, 4, 5],
                           index=int(max(0, min(5, int(_num(g.get("hat"), 0) or 0)))),
                           key=f"rot_hat_{c.key}", label_visibility="collapsed")
        intact = h3.checkbox("Case", value=bool(g.get("case_intact", True)),
                             key=f"rot_case_{c.key}")
        timing = h4.selectbox("Timing", [TIMING_NO, TIMING_PARTLY, TIMING_YES],
                              index=[TIMING_NO, TIMING_PARTLY, TIMING_YES].index(
                                  g.get("timing", TIMING_NO)
                                  if g.get("timing") in (TIMING_NO, TIMING_PARTLY, TIMING_YES)
                                  else TIMING_NO),
                              key=f"rot_tim_{c.key}", label_visibility="collapsed")
        if (hat != g.get("hat") or intact != g.get("case_intact")
                or timing != g.get("timing")):
            g["hat"], g["case_intact"], g["timing"] = hat, intact, timing
            changed = True

        st_now, why_now = status(hat, timing, intact)
        sc_now = STATUS_COLOR.get(st_now, DIM)
        h5.markdown(
            f"<div style='padding-top:4px;'>"
            f"<span style='background:{sc_now}22;border:1px solid {sc_now};"
            f"color:{sc_now};font-size:0.68rem;font-weight:700;padding:2px 9px;"
            f"border-radius:10px;'>{st_now}</span>"
            f"<div style='color:{DIM};font-size:0.66rem;margin-top:2px;'>{why_now}</div>"
            f"</div>", unsafe_allow_html=True)

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
    st.caption("Prioritetsformeln och AGERA/Bevaka-gränserna är panelens "
               "konstruktion — guiden säger att arket räknar prioritet men "
               "publicerar aldrig hur. Justeras i rotation.py.")
