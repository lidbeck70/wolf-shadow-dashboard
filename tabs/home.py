"""
tabs/home.py — Mission Control.

Läget just nu ur det som faktiskt finns sparat (wolf_regime.json, EMBER-
resultaten, håvarna, sifferuppdateringen och larmtillståndet), zonerna ur
navigationsträdet (ui/nav.py) och de senaste larmhändelserna ur Gisten.

Förut lästes tre session_state-nycklar som ingen satte, så pulsen sa alltid
UNKNOWN, och "Recent alerts" var en processlista som var tom efter varje
omstart och aldrig såg de schemalagda larmen.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from ui import nav
from ui.components import card, kpi
from ui.theme import section_title
from ui.tokens import ACCENT, AMBER as AMBER_C, DIM, EMBER, GOLD as GOLD_C, GREEN, GREY, PURPLE, TEXT, regime_color

_ZONE_COLOR = {"odin": PURPLE, "thor": ACCENT, "review": "#c9a84c",
               "valhall": EMBER, "portfolio": GREEN, "more": GREY}
_WEEKDAYS = ("måndag", "tisdag", "onsdag", "torsdag", "fredag", "lördag", "söndag")
_MONTHS = ("januari", "februari", "mars", "april", "maj", "juni", "juli",
           "augusti", "september", "oktober", "november", "december")


def swedish_date(d: datetime) -> str:
    return f"{_WEEKDAYS[d.weekday()]} {d.day} {_MONTHS[d.month - 1]} {d.year}"


# ── Läget ur det sparade ─────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def _load_status() -> dict:
    """Ren läsning av det som redan ligger sparat. Saknas en källa blir den
    'okänd' med orsak — aldrig ett gissat läge."""
    out = {}
    try:
        from gist_storage import load_wolf_json
        d = load_wolf_json("wolf_regime.json") or {}
        out["swing"] = {"label": str(d.get("regime") or "OKÄND"),
                        "asof": str(d.get("generated") or "")[:10]}
    except Exception:
        out["swing"] = {"label": "OKÄND", "asof": ""}
    try:
        from ember.cache import load_ember_results
        e = load_ember_results() or {}
        n = len(e.get("eligible") or [])
        out["ember"] = {"label": f"{n} setup{'s' if n != 1 else ''}",
                        "asof": str(e.get("timestamp") or "")[:10]}
    except Exception:
        out["ember"] = {"label": "–", "asof": ""}
    try:
        from gist_storage import load_blob
        sc = load_blob("screens.json", None) or {}
        hits = sum(len((v or {}).get("rows") or []) for v in (sc.get("screens") or {}).values())
        new = sum(len((v or {}).get("new") or []) for v in (sc.get("screens") or {}).values())
        out["screens"] = {"label": f"{hits} träffar · {new} nya" if sc else "–",
                          "asof": str(sc.get("generated") or "")[:10]}
        sr = load_blob("sheets_refresh.json", None) or {}
        ev = sr.get("events") or []
        out["events"] = {"label": f"{len(ev)} händelser" if sr else "–",
                         "asof": str(sr.get("generated") or "")[:10],
                         "rows": ev[:8]}
    except Exception:
        out["screens"] = {"label": "–", "asof": ""}
        out["events"] = {"label": "–", "asof": "", "rows": []}
    return out


def _next_step(status: dict) -> str:
    swing = status.get("swing", {}).get("label", "OKÄND").upper()
    if swing == "RÖD":
        return "⛔ Swing-regimen är RÖD — inga nya köp, bara exits. Gå igenom innehaven mot säljreglerna."
    if swing == "GUL":
        return "🟡 GUL regim — halv positionsstorlek. Kör screenern, men var selektiv."
    if swing == "GRÖN":
        return "🟢 GRÖN regim — välj Momentum Swing i THOR och kontrollera setup A/B."
    return "🔍 Ingen regimdata ännu — kör wolf_data.py eller vänta på nästa schemalagda körning."


def _render_pulse(status: dict) -> None:
    section_title("Läget just nu", "📡")
    sw, em, sc, ev = (status.get(k, {}) for k in ("swing", "ember", "screens", "events"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Swing-regim", sw.get("label", "OKÄND"),
                        regime_color(sw.get("label")), sw.get("asof", "")), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("EMBER-setups", em.get("label", "–"), EMBER, em.get("asof", "")),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Håvarna", sc.get("label", "–"), ACCENT, sc.get("asof", "")),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Arkens händelser", ev.get("label", "–"), PURPLE, ev.get("asof", "")),
                    unsafe_allow_html=True)
    st.markdown(
        f"<div style='background:#1A1F25;border:1px solid rgba(255,255,255,0.06);"
        f"border-left:3px solid {ACCENT};border-radius:8px;padding:12px 18px;margin:12px 0 4px;"
        f"font-size:12px;color:{TEXT};'>{_next_step(status)}</div>", unsafe_allow_html=True)


# ── Zonerna ur navigationsträdet ─────────────────────────────────────────────
def _render_zones() -> None:
    for key, title, cards in nav.HOME_ZONES:
        color = _ZONE_COLOR.get(key, GREY)
        st.markdown(
            f"<div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;"
            f"color:{color};border-left:2px solid {color};padding-left:8px;margin:18px 0 12px;'>"
            f"{nav.TOP_LABEL[key]} · {title}</div>", unsafe_allow_html=True)
        cols = st.columns(len(cards))
        for col, (name, desc) in zip(cols, cards):
            with col:
                st.markdown(card(name, desc, color), unsafe_allow_html=True)
                # Kortet öppnar sin flik. Första namnet på ett kort med flera
                # ("Rick Rule · Royalty C") är målet; okänt namn → toppfliken.
                first = name.split(" · ")[0]
                segs = [key, first] if first in nav.options(key) else [key]
                target = nav.slug(segs)
                if st.button("Öppna →", key=f"home_go_{target}", width="stretch"):
                    st.session_state["nav_goto"] = target
                    st.rerun()


# ── Senaste händelserna ur arken ─────────────────────────────────────────────
def _render_recent_events(status: dict) -> None:
    section_title("Senaste händelserna ur arken", "🔔")
    rows = status.get("events", {}).get("rows") or []
    if not rows:
        st.markdown(
            f"<div style='background:#1A1F25;border:1px solid rgba(255,255,255,0.06);"
            f"border-radius:8px;padding:16px;color:{DIM};font-size:0.76rem;text-align:center;'>"
            f"Inga händelser i senaste sifferuppdateringen. Larmen skickas till Discord av "
            f"de schemalagda körningarna — se ALERTS.</div>", unsafe_allow_html=True)
        return
    for e in rows:
        st.markdown(
            f"<div style='background:#1A1F25;border:1px solid rgba(255,255,255,0.06);"
            f"border-radius:8px;padding:10px 14px;margin-bottom:6px;'>"
            f"<div style='color:{TEXT};font-size:0.8rem;font-weight:700;'>{e.get('title', '')}</div>"
            f"<div style='color:{DIM};font-size:0.74rem;'>{e.get('body', '')}</div></div>",
            unsafe_allow_html=True)


# ── 🏠 I DAG (designförslaget PR 8) ──────────────────────────────────────────
def _block(title: str, color: str = ACCENT) -> None:
    st.markdown(f"<div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;color:{color};"
                f"border-left:2px solid {color};padding-left:8px;margin:16px 0 8px;'>{title}</div>",
                unsafe_allow_html=True)


def _line(text: str, color: str = TEXT, size: str = "0.86rem") -> None:
    st.markdown(f"<div style='color:{color};font-size:{size};line-height:1.45;margin:2px 0;'>{text}</div>",
                unsafe_allow_html=True)


def _open(label: str, key: str, target: str, ticker: str = "", strategy: str = "") -> None:
    from ui.links import go
    st.button(label, key=key, on_click=go, args=(target, ticker, strategy), width="stretch")


def _today_data() -> dict:
    from tabs import home_today as ht
    if st.session_state.get("_home_refresh") or "home_today" not in st.session_state:
        st.session_state.pop("_home_refresh", None)
        with st.spinner("Läser det sparade …"):
            st.session_state["home_today"] = ht.load()
    return st.session_state["home_today"]


def _render_today() -> None:
    from html import escape

    from odin import laget
    from portfolio import mot_malet as mm
    from tabs import home_today as ht
    from thor import registry as reg
    from ui import settings as us
    d = _today_data()
    board = d.get("board")

    # 0 · VÄNDNING — KÖP I DAG
    cards_ = ht.turn_cards(board)
    if cards_:
        for i, c in enumerate(cards_):
            st.markdown(f"<div style='border:2px solid {EMBER};background:{EMBER}14;border-radius:10px;padding:12px 14px;"
                        f"margin:6px 0;'><div style='color:{EMBER};font-size:1.15rem;font-weight:900;'>"
                        f"{escape(ht.turn_headline(c))}</div><div style='color:{DIM};font-size:0.74rem;'>"
                        f"{escape(c['where'])} · {escape(str(c.get('kind') or ''))} bekräftad</div></div>",
                        unsafe_allow_html=True)
            for kind, tickers, strat in (("Deep Contrarian", c["deep"], "deep"), ("Lång", c["long"], "long")):
                for t in tickers:
                    _open(f"{kind}: {t} → Thor", f"home_turn_{i}_{strat}_{t}", "thor", t, strat)
            e = c.get("etf") or {}
            if e.get("ticker"):
                _line(f"ETF: {escape(e['ticker'])} · {escape(str(e.get('exposure') or ''))}"
                      + ("" if e.get("verified") else " · verifiera linjen hos Nordnet"), DIM, "0.78rem")
    else:
        _line(escape(ht.no_turn_line(board)), DIM, "0.8rem")

    # 1 · Kapitalet
    _block("Kapitalet", PURPLE)
    items = laget.capital_row(d.get("alloc"), d.get("regime"), d.get("risk"), board)
    _line(escape(laget.row_text(items)))
    _open("Läget i Odin →", "home_laget", "laget")

    # 2 · Kvalar  ·  3 · Storlek
    _block("Kvalar och storlek", ACCENT)
    n_buy, n_watch, top = ht.qualifiers((d.get("thor") or {}).get("rows"))
    _line(f"<b>{n_buy} KÖP · {n_watch} BEVAKA</b>")
    capital = us.capital() or 100000.0
    for c in top:
        new = " · NY" if c.new else ""
        _line(escape(ht.size_line(c, capital)) + f"<span style='color:{DIM};'> · {escape(c.status)}{new}</span>",
              TEXT, "0.8rem")
        _open(f"{c.ticker} → Thor ({reg.label(c.strategy)})", f"home_buy_{c.strategy}_{c.ticker}", "thor",
              c.ticker, c.strategy)
    if not top:
        _line("Inga KÖP i dag.", DIM, "0.8rem")
    _open("Hela listan i Thor →", "home_thor", "thor")

    # 4 · Sälj och bevaka
    _block("Sälj och bevaka", GREEN)
    lines = ht.holdings_lines(d.get("alert_state"))
    for t, w, why in lines:
        _line(f"<b>{escape(t)}</b> · {w} — {escape(why)}", TEXT, "0.82rem")
    if not lines:
        _line(escape(ht.all_hold_line(d.get("n_open"))), DIM, "0.8rem")
    _open("Positioner →", "home_holdings", "holdings")

    # 5 · Mot målet
    _block("Mot målet", GOLD_C)
    import journal_stats as js
    _line(escape(mm.today_line(js.realized_cagr(d.get("trades") or []), d.get("paper"), d.get("trades"))))
    _open("Mot målet →", "home_goal", "goal")


def _render_alert_log(d: dict) -> None:
    from html import escape

    from alerts import history
    section_title("Senaste larmen", "🔔")
    rows = history.latest(d.get("alert_log"), 8)
    if not rows:
        _line("Inga sparade larm än — loggen fylls av nästa schemalagda körning som skickar ett.", DIM, "0.78rem")
    for i, r in enumerate(rows):
        _line(f"<b>{escape(r.get('title', ''))}</b> <span style='color:{DIM};'>· {escape(r.get('time', ''))}</span>",
              TEXT, "0.8rem")
        if r.get("target"):
            _open("Öppna →", f"home_alert_{i}", r["target"], r.get("ticker") or "", r.get("strategy") or "")


def _render_due() -> None:
    from datetime import date

    try:
        import routines
        due = routines.due_today(date.today())
    except Exception:
        due = []
    if due:
        section_title("I dag förfaller", "🗓")
        st.markdown(" ".join(f"<span style='display:inline-block;border:1px solid {AMBER_C};border-radius:4px;"
                             f"padding:1px 6px;margin:2px;color:{AMBER_C};font-size:0.74rem;'>{r.title} · {r.minutes}</span>"
                             for r in due), unsafe_allow_html=True)


# ── Sidan ────────────────────────────────────────────────────────────────────
def tab_home() -> None:
    section_title("I dag", "🏠")
    c1, c2 = st.columns([3, 1])
    c1.markdown(f'<p style="color:{DIM};font-size:0.8rem;margin:-8px 0 8px;">{swedish_date(datetime.now())} · '
                f'vändning → kapital → kvalar → storlek → sälj → mål</p>', unsafe_allow_html=True)
    if c2.button("📡 Uppdatera", key="home_refresh"):
        st.session_state["_home_refresh"] = True
        st.rerun()
    _render_today()
    st.markdown("<br>", unsafe_allow_html=True)
    _render_alert_log(st.session_state.get("home_today") or {})
    _render_due()
    with st.expander("🗺 Flikguiden — var allt ligger", expanded=False):
        _render_zones()
        st.caption("Adressfältet följer fliken — kopiera URL:en (t.ex. `?p=odin/marknad/"
                   "swing-regime`) för en direktlänk. Alla länkar står i "
                   "MER → Regler & Guider → 🗺 FLIKGUIDE.")
