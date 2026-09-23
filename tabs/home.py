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
from ui.tokens import ACCENT, DIM, EMBER, GREEN, GREY, PURPLE, TEXT, regime_color

_ZONE_COLOR = {"regime": PURPLE, "screening": ACCENT, "review": "#c9a84c",
               "intel": EMBER, "portfolio": GREEN}
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
        return "🟢 GRÖN regim — kör SCREENING → Swing Screener och kontrollera setup A/B."
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


# ── Sidan ────────────────────────────────────────────────────────────────────
def tab_home() -> None:
    section_title("Mission Control", "🔱")
    st.markdown(
        f'<p style="color:{DIM};font-size:0.8rem;margin:-8px 0 20px;">'
        f'Nordic Arc Systems — regim → screening → granskning → köp.  '
        f'<span style="font-size:0.72rem;">{swedish_date(datetime.now())}</span></p>',
        unsafe_allow_html=True)

    status = _load_status()
    _render_pulse(status)
    _render_zones()
    st.caption("Adressfältet följer fliken — kopiera URL:en (t.ex. `?p=regime/marknad/"
               "arc-regime/wolf-regime`) för en direktlänk. Alla länkar står i "
               "RULES → Regler & Guider → 🗺 FLIKGUIDE.")
    st.markdown("<br>", unsafe_allow_html=True)
    _render_recent_events(status)
