"""
tiggre.py — Lobo-arket som Streamlit-flik (Masterguiden Del 4, Strategi 4).

Tiggre är en spekulation, inte en investering: ett tidsbestämt köp av en SPECIFIK
omvärdering, med händelsen definierad före köpet. Den här fliken gör arket:

  1. Håven        — screener-kriterierna att köra i Börsdata
  2. Grovsållning — 2 av 3 nyckelfraser i presentationen, annars tillbaka i havet
  3. Lobo-arket   — NAV (after tax) -> P/NAV -> uppsida till 0,8× NAV,
                    din nedsidebedömning -> U/N-kvot (krav >= 3),
                    fem faktorer 0-2 (krav >= 8)
  4. Katalysatorer— minst 2 namngivna, tidsatta händelser inom 12 månader
  5. Positioner   — free ride-larm vid +100 %, eget kapital i risk,
                    de fyra sälj-allt-triggarna, P/NAV mot 0,8-1,0

Grindarna är hårda: KÖP är låst tills 2-av-3, U/N >= 3, poäng >= 8 och två
katalysatorer alla passerar. Panelen ska göra det svårt att bryta reglerna, inte
bara påminna om dem.

Lagring: samma modell som holdings/swing — Gist ("tiggre_data.json") med lokal
fallback, cachad per session.
"""

from __future__ import annotations

import streamlit as st
from datetime import date
from typing import Optional

import csv_export

try:
    from gist_storage import load_blob as _blob_load, save_blob as _blob_save
    _HAS_GIST = True
except Exception:
    _HAS_GIST = False

_STORE_FILE = "tiggre_data.json"
_CACHE_KEY = "tiggre_data"

# ── Krav ur Masterguiden (ändras här, syns överallt) ─────────────────────────
UN_MIN = 3.0              # U/N-kvot: krav >= 3
SCORE_MIN = 8             # fem faktorer 0-2, krav >= 8
CATALYSTS_MIN = 2         # minst två namngivna, tidsatta inom 12 månader
SCREEN_HITS_MIN = 2       # 2 av 3 nyckelfraser
NAV_TARGET = 0.8          # uppsidan räknas till 0,8× NAV
MAX_POSITIONS = 6         # 4–6 bolag
POS_MIN_PCT, POS_MAX_PCT = 2.0, 4.0
FREE_RIDE_PCT = 100.0     # +100 % -> sälj halva

TEXT, DIM = "#e8e4dc", "#8a8578"
EMBER, GREEN, AMBER, RED = "#FF6B3D", "#2d8a4e", "#d4943a", "#c44545"
BG_CARD, BG_ALT, BORDER = "#14141e", "#1a1f25", "#2a2a38"

FACTORS = [
    ("stadium", "Stadium", "Hur långt är projektet? FS klar, tillstånd, byggklart."),
    ("finansiering", "Finansiering", "Löst = 2. Delvis = 1. Ofinansierat = 0."),
    ("manniskor", "Människor", "Byggmeriter i teamet, insyn > 5–10 %, långsam "
                               "utspädning. Tiggres hårdaste faktor."),
    ("jurisdiktion", "Jurisdiktion", "Fraser-rankingen. Toppjurisdiktion = 2."),
    ("un", "U/N", "Räknas automatiskt ur kvoten: ≥ 3 = 2 poäng, 2–3 = 1, < 2 = 0."),
]

# The U/N factor is not entered — the sheet computes it from the ratio
# (Lobo-arket, kolumn N: >=3 -> 2p, >=2 -> 1p, annars 0).
UN_POINTS = ((3.0, 2), (2.0, 1))

# Katalysatorkalenderns lägen. "Försenad 2:a ggn" är inte en status bland
# andra — det ÄR säljregeln, och utan den går den inte att registrera.
CAT_WAITING = "Väntar"
CAT_DELIVERED = "Levererad"
CAT_LATE_1 = "Försenad 1:a ggn"
CAT_LATE_2 = "Försenad 2:a ggn — SÄLJREGEL"
CAT_MISSED = "Utebliven"
CAT_STATUSES = (CAT_WAITING, CAT_DELIVERED, CAT_LATE_1, CAT_LATE_2, CAT_MISSED)
CAT_STATUS_COLOR = {CAT_WAITING: "#8a8578", CAT_DELIVERED: "#2d8a4e",
                    CAT_LATE_1: "#d4943a", CAT_LATE_2: "#c44545",
                    CAT_MISSED: "#c44545"}

SCREEN_PHRASES = [
    ("fs", "Feasibility Study complete (DFS/BFS)"),
    ("permits", "Permits received / granted"),
    ("funded", "Fully funded / financing package"),
]

CATALYST_CHAIN = ("miljötillstånd", "finansieringsbesked", "FID", "byggstart",
                  "50 % färdigt", "first pour", "kommersiell drift")

SELL_ALL_TRIGGERS = [
    ("permit_denied", "Tillstånd nekas"),
    ("fs_worse", "FS-ekonomin försämras väsentligt"),
    ("key_person", "Nyckelperson lämnar"),
    ("delayed_twice", "Katalysator försenad ANDRA gången utan god förklaring"),
]


# ── Hjälpare ─────────────────────────────────────────────────────────────────
def _today() -> str:
    return date.today().isoformat()


def _uid() -> str:
    import time
    return f"{int(time.time() * 1000):x}"


def _num(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return f


def _fmt(n, dec: int = 2) -> str:
    v = _num(n)
    return "–" if v is None else f"{v:,.{dec}f}".replace(",", " ")


def _pct(n, dec: int = 0) -> str:
    v = _num(n)
    return "–" if v is None else f"{v:+.{dec}f} %"


# ── Kärnberäkningar (rena — enhetstestade) ───────────────────────────────────
def upside_pct(mcap: float, nav: float, target: float = NAV_TARGET) -> Optional[float]:
    """Uppsida i procent om kursen når target × NAV. Masterguidens exempel:
    MCap 200, NAV 650 -> 0,31× NAV -> +160 %."""
    m, n = _num(mcap), _num(nav)
    if not m or not n or m <= 0 or n <= 0:
        return None
    return (target * n / m - 1) * 100


def p_nav(mcap: float, nav: float) -> Optional[float]:
    m, n = _num(mcap), _num(nav)
    if not m or not n or n <= 0:
        return None
    return m / n


def un_ratio(up: Optional[float], down: Optional[float]) -> Optional[float]:
    """U/N-kvoten. Nedsidan anges som positivt eller negativt tal — beloppet gäller."""
    u, d = _num(up), _num(down)
    if u is None or d is None or abs(d) < 1e-9:
        return None
    return u / abs(d)


def un_points(un: Optional[float]) -> int:
    """U/N-faktorns poäng, räknad ur kvoten precis som Lobo-arket gör."""
    v = _num(un)
    if v is None:
        return 0
    for threshold, points in UN_POINTS:
        if v >= threshold:
            return points
    return 0


def factor_score(factors: dict, un: Optional[float] = None) -> int:
    """Summa av de fem faktorerna, var och en klämd till 0–2.

    U/N-faktorn matas inte in: skickas `un` med räknas den ur kvoten, så att
    poängen inte kan bli en annan än den arket ger.
    """
    total = 0
    for key, _label, _help in FACTORS:
        if key == "un" and un is not None:
            total += un_points(un)
            continue
        v = _num(factors.get(key), 0) or 0
        total += max(0, min(2, int(v)))
    return total


def catalyst_sell_signal(catalysts: list) -> Optional[dict]:
    """Den första katalysatorn som utlöst säljregeln, om någon.

    "Katalysator försenad andra gången utan god förklaring = sälj allt."
    """
    for c in catalysts or []:
        if (c or {}).get("status") == CAT_LATE_2:
            return c
    return None


def screen_hits(flags: dict) -> int:
    return sum(1 for k, _label in SCREEN_PHRASES if flags.get(k))


def buy_gates(cand: dict) -> list[tuple[str, bool, str]]:
    """The hard gates, in order. Returns (label, passed, detail)."""
    hits = screen_hits(cand.get("screen", {}))
    up = upside_pct(cand.get("mcap"), cand.get("nav"))
    un = un_ratio(up, cand.get("downside"))
    score = factor_score(cand.get("factors", {}), un)
    cats = [c for c in cand.get("catalysts", []) if c.get("name") and c.get("date")]
    return [
        (f"Grovsållning {SCREEN_HITS_MIN} av 3", hits >= SCREEN_HITS_MIN,
         f"{hits}/3 nyckelfraser"),
        (f"U/N ≥ {UN_MIN:g}", un is not None and un >= UN_MIN,
         f"{un:.1f}:1" if un is not None else "saknar NAV/nedsida"),
        (f"Poäng ≥ {SCORE_MIN}", score >= SCORE_MIN, f"{score}/10"),
        (f"≥ {CATALYSTS_MIN} katalysatorer", len(cats) >= CATALYSTS_MIN,
         f"{len(cats)} namngivna och tidsatta"),
    ]


def free_ride_reached(entry: float, current: float) -> bool:
    e, c = _num(entry), _num(current)
    return bool(e and c and e > 0 and (c / e - 1) * 100 >= FREE_RIDE_PCT)


def equity_at_risk(entry: float, current: float, shares: float,
                   half_sold: bool) -> Optional[float]:
    """Eget kapital i risk. Efter free ride (halva sålt vid +100 %) är insatsen
    uttagen — kvarvarande position åker på husets pengar, alltså 0 kr i risk."""
    e, c, s = _num(entry), _num(current), _num(shares)
    if e is None or s is None:
        return None
    if half_sold and free_ride_reached(e, c or 0):
        return 0.0
    return e * s


# ── Lagring ──────────────────────────────────────────────────────────────────
def _default() -> dict:
    return {"candidates": [], "positions": [], "closed": [], "parked": []}


def _normalize(data: dict) -> dict:
    for k, v in _default().items():
        if not isinstance(data.get(k), list):
            data[k] = v
    return data


def _load() -> dict:
    if _CACHE_KEY in st.session_state:
        return st.session_state[_CACHE_KEY]
    data = _blob_load(_STORE_FILE, None) if _HAS_GIST else None
    if not isinstance(data, dict):
        data = _default()
    _normalize(data)
    st.session_state[_CACHE_KEY] = data
    return data


def _save(data: dict) -> None:
    st.session_state[_CACHE_KEY] = data
    if _HAS_GIST:
        _blob_save(_STORE_FILE, data)


# ── Entry point ──────────────────────────────────────────────────────────────
def render_tiggre_page() -> None:
    """Huvud-entry point för Tiggre-fliken."""
    try:
        data = _load()
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:baseline;flex-wrap:wrap;gap:8px;'>"
            f"<h1 style='color:{TEXT};margin:0;letter-spacing:0.06em;'>"
            f"Tiggre <span style='color:{EMBER};'>· sweet spot</span></h1>"
            f"<span style='color:{DIM};font-size:0.85rem;'>Lobo-arket · {_today()}</span>"
            f"</div>"
            f"<p style='color:{DIM};font-size:0.8rem;margin:6px 0 14px;'>"
            f"En spekulation är ett tidsbestämt köp av en <b>specifik</b> omvärdering, "
            f"med händelsen definierad före köpet. Kan du inte namnge katalysatorerna "
            f"är det en förhoppning — passa.</p>",
            unsafe_allow_html=True)

        _export(data)
        _screener_card()
        _positions(data)
        _candidates(data)
        _parked(data)
    except Exception as e:
        st.error(f"Tiggre-fliken kunde inte renderas: {e}")


# ── 1. Håven ─────────────────────────────────────────────────────────────────
POS_CSV = [("date", "Köpdatum"), ("ticker", "Ticker"), ("name", "Bolag"),
           ("shares", "Antal"), ("entry", "Entry"), ("current", "Kurs nu"),
           ("_ret", "Utveckling %"), ("_free_ride", "+100 %-nivå"),
           ("half_sold", "Halva såld"), ("_risk", "Kapital i risk"),
           ("_pnav", "P/NAV nu"), ("_sell", "Säljregel utlöst")]

CAT_CSV = [("_ticker", "Bolag"), ("name", "Katalysator"), ("date", "Förväntad"),
           ("status", "Status"), ("actual", "Faktiskt datum"),
           ("reaction", "Kursreaktion %"), ("lesson", "Utfall/lärdom")]


def _export(data: dict) -> None:
    """Positionerna och katalysatorkalendern — arkets två viktigaste blad."""
    pos_rows = []
    for p in data.get("positions", []):
        entry = _num(p.get("entry"), 0.0) or 0.0
        cur = _num(p.get("current"), 0.0) or 0.0
        ret = (cur / entry - 1) * 100 if entry > 0 and cur > 0 else None
        pn = p_nav(p.get("mcap"), p.get("nav"))
        sell = catalyst_sell_signal(p.get("catalysts", []))
        pos_rows.append({
            **p, "_ret": None if ret is None else round(ret, 1),
            "_free_ride": round(entry * 2, 2) if entry else None,
            "_risk": equity_at_risk(entry, cur, p.get("shares", 0),
                                    p.get("half_sold")),
            "_pnav": None if pn is None else round(pn, 2),
            "_sell": sell.get("name") if sell else None,
        })

    cat_rows = []
    for p in data.get("positions", []) + data.get("candidates", []):
        for c in p.get("catalysts", []) or []:
            cat_rows.append({**c, "_ticker": p.get("ticker", "?")})

    c1, c2 = st.columns(2)
    with c1:
        csv_export.download_button(pos_rows, POS_CSV, "lobo_positioner",
                                   label="⬇ Positioner (CSV)",
                                   key="csv_tiggre_pos")
    with c2:
        csv_export.download_button(cat_rows, CAT_CSV, "lobo_katalysatorer",
                                   label="⬇ Katalysatorer (CSV)",
                                   key="csv_tiggre_cat")


def _screener_card() -> None:
    with st.expander("🕸️ Håven — screener-kriterier att köra i Börsdata", expanded=False):
        st.markdown(
            f"<div style='color:{TEXT};font-size:0.84rem;line-height:1.8;'>"
            f"<b>Kanada / Australien / USA</b> · Metals & Mining<br>"
            f"<b>Börsvärde 50–1 000 MUSD</b> — under 50 = för tidigt, "
            f"över 1 000 = omvärderingen ofta redan gjord<br>"
            f"<b>Nettoskuld &lt; 0</b> ELLER skuld som ÄR byggkrediten<br>"
            f"<b>Omsättning ~0</b><br>"
            f"<span style='color:{DIM};'>Inga andra filter — vinstmått raderar "
            f"universumet. Screenern är bara håven; urvalet sker i "
            f"presentationerna.</span></div>", unsafe_allow_html=True)


# ── 2+3. Kandidater: grovsållning + Lobo-arket ───────────────────────────────
def _candidates(data: dict) -> None:
    st.markdown(f"<div style='font-weight:700;color:{TEXT};margin:18px 0 6px;'>"
                f"Kandidater — grovsållning och granskning</div>",
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 2, 0.8])
    tkr = c1.text_input("Ticker", key="tg_add_ticker")
    nm = c2.text_input("Bolagsnamn", key="tg_add_name")
    if c3.button("Lägg till", key="tg_add_btn"):
        if tkr.strip():
            data["candidates"].append({
                "id": _uid(), "ticker": tkr.upper().strip(), "name": nm.strip(),
                "added": _today(), "screen": {}, "mcap": 0.0, "nav": 0.0,
                "downside": -40.0, "factors": {}, "catalysts": [],
            })
            _save(data)
            st.rerun()

    if not data["candidates"]:
        st.caption("Tom — kör håven i Börsdata och lägg in bolagen du vill grovsålla.")
        return

    for cand in list(data["candidates"]):
        _candidate_card(data, cand)


def _candidate_card(data: dict, cand: dict) -> None:
    hits = screen_hits(cand.get("screen", {}))
    up = upside_pct(cand.get("mcap"), cand.get("nav"))
    pn = p_nav(cand.get("mcap"), cand.get("nav"))
    un = un_ratio(up, cand.get("downside"))
    score = factor_score(cand.get("factors", {}), un)
    gates = buy_gates(cand)
    all_pass = all(g[1] for g in gates)

    head = (f"{'✅' if all_pass else '🔍'}  {cand.get('ticker','?')}"
            f"  ·  {hits}/3 fraser  ·  U/N {un:.1f}:1" if un is not None
            else f"{'✅' if all_pass else '🔍'}  {cand.get('ticker','?')}  ·  {hits}/3 fraser")
    with st.expander(f"{head}  ·  poäng {score}/10", expanded=False):
        # Grovsållning
        st.markdown(f"<b style='color:{TEXT};'>Grovsållning</b> "
                    f"<span style='color:{DIM};font-size:0.78rem;'>"
                    f"— {SCREEN_HITS_MIN} av 3 krävs. Noll av tre = tillbaka i havet."
                    f"</span>", unsafe_allow_html=True)
        sc = cand.setdefault("screen", {})
        changed = False
        for key, label in SCREEN_PHRASES:
            v = st.checkbox(label, value=bool(sc.get(key)), key=f"tg_sc_{cand['id']}_{key}")
            if v != bool(sc.get(key)):
                sc[key] = v
                changed = True

        # Lobo-arket
        st.markdown(f"<div style='height:8px;'></div><b style='color:{TEXT};'>"
                    f"Lobo-arket</b>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        mcap = m1.number_input("Börsvärde (MUSD)", min_value=0.0,
                               value=float(_num(cand.get("mcap"), 0.0) or 0.0),
                               step=10.0, key=f"tg_mcap_{cand['id']}")
        nav = m2.number_input("NAV = NPV after tax (MUSD)", min_value=0.0,
                              value=float(_num(cand.get("nav"), 0.0) or 0.0),
                              step=10.0, key=f"tg_nav_{cand['id']}",
                              help="Alltid after-tax — pre-tax är 30–40 % för högt.")
        down = m3.number_input("Nedsida (%)", value=float(_num(cand.get("downside"), -40.0) or -40.0),
                               step=5.0, key=f"tg_down_{cand['id']}",
                               help="Värsta rimliga scenariot: utspädning, byggförsening.")
        if (mcap != cand.get("mcap") or nav != cand.get("nav")
                or down != cand.get("downside")):
            cand["mcap"], cand["nav"], cand["downside"] = mcap, nav, down
            changed = True

        k1, k2, k3 = st.columns(3)
        k1.metric("P/NAV", f"{pn:.2f}×" if pn is not None else "–",
                  help="Köpzonen är 0,2–0,4× NAV.")
        k2.metric(f"Uppsida till {NAV_TARGET:g}× NAV", _pct(up))
        k3.metric("U/N-kvot", f"{un:.1f}:1" if un is not None else "–",
                  help=f"Krav ≥ {UN_MIN:g}. Under det: vänta — ofta ÄR "
                       f"finansieringsbeskedet katalysatorn.")

        # Fem faktorer
        st.markdown(f"<div style='height:6px;'></div><b style='color:{TEXT};'>"
                    f"Fem faktorer (0–2)</b> <span style='color:{DIM};font-size:0.78rem;'>"
                    f"— krav ≥ {SCORE_MIN}</span>", unsafe_allow_html=True)
        fac = cand.setdefault("factors", {})
        fcols = st.columns(5)
        for (fkey, flabel, fhelp), col in zip(FACTORS, fcols):
            if fkey == "un":
                # Räknas ur kvoten, precis som arket — inte en bedömning.
                col.markdown(
                    f"<div style='font-size:0.8rem;color:{DIM};'>{flabel}</div>"
                    f"<div style='color:{TEXT};font-size:1.1rem;font-weight:700;"
                    f"padding-top:4px;'>{un_points(un)}"
                    f"<span style='color:{DIM};font-size:0.72rem;'> auto</span></div>",
                    unsafe_allow_html=True)
                continue
            v = col.selectbox(flabel, [0, 1, 2],
                              index=int(max(0, min(2, int(_num(fac.get(fkey), 0) or 0)))),
                              key=f"tg_f_{cand['id']}_{fkey}", help=fhelp)
            if v != fac.get(fkey):
                fac[fkey] = v
                changed = True

        # Katalysatorer
        _catalysts(data, cand)

        # Grindar
        st.markdown(f"<div style='height:8px;'></div>", unsafe_allow_html=True)
        gate_html = ""
        for label, passed, detail in gates:
            c = GREEN if passed else RED
            gate_html += (f"<div style='color:{TEXT};font-size:0.82rem;padding:2px 0;'>"
                          f"<span style='color:{c};font-weight:700;'>"
                          f"{'✓' if passed else '✕'}</span> {label} "
                          f"<span style='color:{DIM};'>— {detail}</span></div>")
        st.markdown(
            f"<div style='border:1px solid {(GREEN if all_pass else RED)}55;"
            f"background:{(GREEN if all_pass else RED)}0d;border-radius:8px;"
            f"padding:10px 14px;'>{gate_html}</div>", unsafe_allow_html=True)

        b1, b2, b3 = st.columns([1.4, 1, 1])
        n_pos = len(data["positions"])
        room = n_pos < MAX_POSITIONS
        entry = b1.number_input("Entry-kurs", min_value=0.0, value=0.0, step=0.5,
                                key=f"tg_entry_{cand['id']}")
        if b2.button("KÖP →", key=f"tg_buy_{cand['id']}",
                     disabled=not (all_pass and room)):
            _promote(data, cand, _num(entry))
            st.rerun()
        if b3.button("Parkera (för tidigt)", key=f"tg_park_{cand['id']}"):
            data["parked"].append({**cand, "parked": _today()})
            data["candidates"] = [c for c in data["candidates"] if c["id"] != cand["id"]]
            _save(data)
            st.rerun()

        if not room:
            st.warning(f"Max {MAX_POSITIONS} positioner — stäng en först.")
        elif not all_pass:
            st.caption("KÖP är låst tills alla fyra grindar passerar.")

        if changed:
            _save(data)


def _catalysts(data: dict, cand: dict) -> None:
    st.markdown(
        f"<div style='height:6px;'></div><b style='color:{TEXT};'>Katalysatorkalender</b> "
        f"<span style='color:{DIM};font-size:0.78rem;'>— minst {CATALYSTS_MIN} "
        f"namngivna och tidsatta inom 12 månader. Kedjan: "
        f"{' → '.join(CATALYST_CHAIN)}</span>", unsafe_allow_html=True)

    cats = cand.setdefault("catalysts", [])
    changed = False
    for cat in list(cats):
        cc1, cc2, cc3, cc4 = st.columns([2.0, 1.0, 1.6, 0.4])
        cc1.markdown(f"<span style='color:{TEXT};font-size:0.84rem;'>"
                     f"{cat.get('name','?')}</span>", unsafe_allow_html=True)
        cc2.markdown(f"<span style='color:{DIM};font-size:0.8rem;'>"
                     f"{cat.get('date','?')}</span>", unsafe_allow_html=True)
        cur = cat.get("status") if cat.get("status") in CAT_STATUSES else CAT_WAITING
        new = cc3.selectbox("Status", list(CAT_STATUSES),
                            index=CAT_STATUSES.index(cur),
                            key=f"tg_catst_{cand['id']}_{cat.get('id','')}",
                            label_visibility="collapsed")
        if new != cat.get("status"):
            cat["status"] = new
            changed = True
        if cc4.button("✕", key=f"tg_delcat_{cand['id']}_{cat.get('id','')}"):
            cand["catalysts"] = [c for c in cats if c.get("id") != cat.get("id")]
            _save(data)
            st.rerun()

        # Utfallet är strategins viktigaste lärdata — arket samlar det per
        # händelse, inte per bolag.
        if new in (CAT_DELIVERED, CAT_LATE_1, CAT_LATE_2, CAT_MISSED):
            o1, o2, o3 = st.columns([1.0, 1.0, 3.0])
            act = o1.text_input("Faktiskt datum", value=cat.get("actual", ""),
                                key=f"tg_catact_{cand['id']}_{cat.get('id','')}",
                                placeholder="ÅÅÅÅ-MM-DD",
                                label_visibility="collapsed")
            rea = o2.number_input("Kursreaktion %",
                                  value=float(_num(cat.get("reaction"), 0.0) or 0.0),
                                  step=1.0,
                                  key=f"tg_catrea_{cand['id']}_{cat.get('id','')}",
                                  label_visibility="collapsed")
            les = o3.text_input("Lärdom", value=cat.get("lesson", ""),
                                key=f"tg_catles_{cand['id']}_{cat.get('id','')}",
                                placeholder="Utfall / lärdom — vad lärde den dig?",
                                label_visibility="collapsed")
            if (act != cat.get("actual", "") or rea != cat.get("reaction")
                    or les != cat.get("lesson", "")):
                cat["actual"], cat["reaction"], cat["lesson"] = act, rea, les
                changed = True

    sell = catalyst_sell_signal(cats)
    if sell:
        st.markdown(
            f"<div style='border:1px solid {RED};background:{RED}1a;"
            f"border-radius:8px;padding:10px 14px;margin:8px 0;'>"
            f"<b style='color:{RED};'>SÄLJREGEL UTLÖST</b> "
            f"<span style='color:{TEXT};font-size:0.84rem;'>— "
            f"{sell.get('name','katalysatorn')} är försenad andra gången. "
            f"Utan god förklaring säljs hela positionen.</span></div>",
            unsafe_allow_html=True)

    if changed:
        _save(data)

    a1, a2, a3 = st.columns([2.4, 1.2, 0.5])
    cname = a1.text_input("Händelse", key=f"tg_cat_name_{cand['id']}",
                          label_visibility="collapsed",
                          placeholder="t.ex. Finansieringsbesked")
    cdate = a2.text_input("Datum", key=f"tg_cat_date_{cand['id']}",
                          label_visibility="collapsed", placeholder="ÅÅÅÅ-MM")
    if a3.button("+", key=f"tg_addcat_{cand['id']}"):
        if cname.strip() and cdate.strip():
            cats.append({"id": _uid(), "name": cname.strip(),
                         "date": cdate.strip(), "status": CAT_WAITING})
            _save(data)
            st.rerun()


def _promote(data: dict, cand: dict, entry: Optional[float]) -> None:
    if entry is None or entry <= 0:
        st.warning("Ange en entry-kurs > 0.")
        return
    data["positions"].append({
        "id": _uid(), "ticker": cand.get("ticker", "?"), "name": cand.get("name", ""),
        "entry": entry, "current": entry, "shares": 0.0,
        "mcap": cand.get("mcap"), "nav": cand.get("nav"),
        "date": _today(), "half_sold": False,
        "triggers": {}, "catalysts": cand.get("catalysts", []),
    })
    data["candidates"] = [c for c in data["candidates"] if c["id"] != cand["id"]]
    _save(data)


# ── 5. Positioner ────────────────────────────────────────────────────────────
def _positions(data: dict) -> None:
    pos = data["positions"]
    st.markdown(
        f"<div style='font-weight:700;color:{TEXT};margin-bottom:4px;'>"
        f"Positioner ({len(pos)}/4–6)</div>"
        f"<div style='color:{DIM};font-size:0.75rem;margin-bottom:8px;'>"
        f"{POS_MIN_PCT:g}–{POS_MAX_PCT:g} % per bolag · räknas mot "
        f"Optionalitets-ramen (0–12 %) tills produktion</div>",
        unsafe_allow_html=True)

    if not pos:
        st.caption("Inga öppna positioner.")
        return

    for p in list(pos):
        entry = _num(p.get("entry"), 0.0) or 0.0
        cur = _num(p.get("current"), 0.0) or 0.0
        ret = (cur / entry - 1) * 100 if entry > 0 and cur > 0 else None
        free_ride = free_ride_reached(entry, cur) and not p.get("half_sold")
        # Kalendern styr sin egen trigger: en katalysator satt till "Försenad
        # 2:a ggn" ÄR säljregeln, så den ska inte behöva kryssas i för hand.
        cat_sell = catalyst_sell_signal(p.get("catalysts", []))
        if cat_sell:
            p.setdefault("triggers", {})["delayed_twice"] = True
        fired = [lbl for k, lbl in SELL_ALL_TRIGGERS if p.get("triggers", {}).get(k)]
        pn_now = p_nav(p.get("mcap"), p.get("nav"))

        if fired:
            bd, badge = RED, (f"<span style='background:{RED};color:#fff;font-size:0.7rem;"
                              f"font-weight:700;padding:2px 8px;border-radius:4px;'>"
                              f"SÄLJ ALLT — {fired[0]}</span>")
        elif free_ride:
            bd, badge = AMBER, (f"<span style='background:{AMBER};color:#000;font-size:0.7rem;"
                                f"font-weight:700;padding:2px 8px;border-radius:4px;'>"
                                f"+100 % — FREE RIDE: sälj halva</span>")
        else:
            bd, badge = BORDER, ""

        st.markdown(
            f"<div style='border:1px solid {bd};background:{BG_ALT};border-radius:10px;"
            f"padding:10px 12px 4px;margin-bottom:4px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"flex-wrap:wrap;gap:6px;'>"
            f"<span style='font-weight:700;color:{TEXT};'>{p.get('ticker','?')}"
            f"<span style='color:{DIM};font-size:0.72rem;margin-left:8px;'>"
            f"{p.get('name','')} · {p.get('date','')}</span></span>{badge}</div></div>",
            unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        new_entry = c1.number_input("Entry", min_value=0.0, value=float(entry),
                                    step=0.5, key=f"tg_p_entry_{p['id']}")
        new_cur = c2.number_input("Kurs nu", min_value=0.0, value=float(cur),
                                  step=0.5, key=f"tg_p_cur_{p['id']}")
        c3.metric("Avkastning", _pct(ret, 1) if ret is not None else "–")
        c4.metric("P/NAV nu", f"{pn_now:.2f}×" if pn_now is not None else "–",
                  help="Slutsälj i etapper vid 0,8–1,0× NAV eller produktionsstart.")
        eq = equity_at_risk(new_entry, new_cur, p.get("shares", 0), p.get("half_sold"))
        c5.metric("Kapital i risk", "0 kr" if eq == 0 else (_fmt(eq, 0) if eq else "–"),
                  help="Efter free ride är insatsen uttagen — resten åker på husets pengar.")
        if new_entry != entry or new_cur != cur:
            p["entry"], p["current"] = new_entry, new_cur
            _save(data)

        t_cols = st.columns(len(SELL_ALL_TRIGGERS) + 2)
        trig = p.setdefault("triggers", {})
        for (tkey, tlabel), col in zip(SELL_ALL_TRIGGERS, t_cols):
            v = col.checkbox(tlabel, value=bool(trig.get(tkey)),
                             key=f"tg_t_{p['id']}_{tkey}")
            if v != bool(trig.get(tkey)):
                trig[tkey] = v
                _save(data)
        hs = t_cols[-2].checkbox("Halva såld (+100 %)", value=bool(p.get("half_sold")),
                                 key=f"tg_half_{p['id']}")
        if hs != bool(p.get("half_sold")):
            p["half_sold"] = hs
            _save(data)
        if t_cols[-1].button("Stäng", key=f"tg_close_{p['id']}"):
            data["closed"].append({
                "id": p["id"], "ticker": p.get("ticker", "?"),
                "entry": new_entry, "exit": new_cur, "date": _today(),
                "ret": (new_cur / new_entry - 1) if new_entry > 0 else 0.0,
                "reason": fired[0] if fired else "manuell",
                "catalysts": p.get("catalysts", []),
            })
            data["positions"] = [x for x in data["positions"] if x["id"] != p["id"]]
            _save(data)
            st.rerun()

        if fired:
            st.error("Sälj allt samma vecka: " + " · ".join(fired))
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)


def _parked(data: dict) -> None:
    parked = data.get("parked", [])
    if not parked:
        return
    with st.expander(f"🅿️ För tidigt-listan ({len(parked)}) — PEA-bolag att "
                     f"återbesöka om 1–2 år", expanded=False):
        for p in parked:
            st.markdown(
                f"<div style='color:{TEXT};font-size:0.82rem;'>"
                f"<b>{p.get('ticker','?')}</b> "
                f"<span style='color:{DIM};'>{p.get('name','')} · parkerad "
                f"{p.get('parked','')}</span></div>", unsafe_allow_html=True)
