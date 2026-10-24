"""
odin/laget.py — 👁 ODIN → Läget (designförslaget PR 5): kapitalraden överst, Swing-läget under och
Marknadsrisken på begäran.

Kapitalraden visar fyra ljus, vart och ett med sitt eget ord och ur sin egen källa:
  Allokering    strömbrytaren (allocator.breaker_state ur data/allocator.json: topp och nuvarande värde)
  Swing         Swing-läget (wolf_regime.json)
  Marknadsrisk  den sparade nivån för SPY och OMXS30 (market_risk.json, sparad av alert_scan)
  Norden        hur många av Valhalls nordiska segment som är GRÖNA (valhall_board.json)

Raden drar ingen egen slutsats och slår inte ihop ljusen — det vore en ny regel. Allt läses ur det som är
sparat; inget räknas live förrän du själv slår på Marknadsrisken.
"""
from __future__ import annotations

from typing import Optional

RISK_BLOB = "market_risk.json"
SAKNAS = "Saknas"


def _color(word: str) -> str:
    from ui.tokens import AMBER, DIM, GREEN, RED
    return {"NORMAL": GREEN, "GRÖN": GREEN, "LÅG": GREEN, "SKÄRPT": AMBER, "GUL": AMBER, "FÖRHÖJD": AMBER,
            "HALVERAD RISK": RED, "RÖD": RED, "HÖG": RED}.get(str(word or "").upper(), DIM)


def allocation_light(alloc: Optional[dict]) -> tuple:
    """('NORMAL'|'SKÄRPT'|'HALVERAD RISK'|'Saknas', förklaring) ur allokerarens topp och nuvarande värde."""
    import allocator as al
    dd = al.drawdown_pct((alloc or {}).get("peak"), (alloc or {}).get("current"))
    if dd is None:
        return SAKNAS, "ange portföljens topp och värde i Allokering"
    word, _col, _act = al.breaker_state(dd)
    return word, f"{dd:.1f} % från toppen"


def swing_light(regime: Optional[dict]) -> tuple:
    word = str((regime or {}).get("regime") or "").upper()
    return (word, str((regime or {}).get("generated") or "")[:10]) if word in ("GRÖN", "GUL", "RÖD") else \
        (SAKNAS, "wolf_regime.json saknas")


def risk_light(blob: Optional[dict]) -> tuple:
    """('SPY LÅG · OMXS30 FÖRHÖJD', datum) ur den sparade nivån, en marknad i taget."""
    levels = (blob or {}).get("levels") or {}
    parts = [f"{(v or {}).get('label') or m} {(v or {}).get('level')}" for m, v in levels.items()
             if isinstance(v, dict) and v.get("level")]
    if not parts:
        return SAKNAS, "ingen sparad nivå än"
    return " · ".join(parts), str((blob or {}).get("generated") or "")[:16]


def risk_worst(blob: Optional[dict]) -> str:
    order = {"LÅG": 0, "FÖRHÖJD": 1, "HÖG": 2}
    lv = [v.get("level") for v in ((blob or {}).get("levels") or {}).values() if isinstance(v, dict)]
    lv = [x for x in lv if x in order]
    return max(lv, key=order.get) if lv else SAKNAS


def nordic_light(board: Optional[dict]) -> tuple:
    """('5/9 GRÖN', ...) — antal nordiska segment som är GRÖNA av dem som har ett läge."""
    st = [((r or {}).get("nordic") or {}).get("status") for r in (board or {}).get("segments") or []]
    st = [s for s in st if s in ("GRÖN", "GUL", "RÖD")]
    if not st:
        return SAKNAS, "valhall_board.json saknas"
    g = sum(1 for s in st if s == "GRÖN")
    word = "GRÖN" if g * 2 > len(st) else "GUL" if g else "RÖD"
    return f"{g}/{len(st)} GRÖN", word


def capital_row(alloc: Optional[dict], regime: Optional[dict], risk: Optional[dict], board: Optional[dict]) -> list:
    """[(rubrik, ord, under, färg)] i ordningen Allokering · Swing · Marknadsrisk · Norden."""
    a, a_sub = allocation_light(alloc)
    s, s_sub = swing_light(regime)
    r, r_sub = risk_light(risk)
    n, n_word = nordic_light(board)
    return [("Allokering", a, a_sub, _color(a)), ("Swing", s, s_sub, _color(s)),
            ("Marknadsrisk", r, r_sub, _color(risk_worst(risk))),
            ("Norden", n, "Valhalls nordiska segment", _color(n_word))]


def row_text(items: list) -> str:
    """'Allokering NORMAL · Swing GRÖN · Marknadsrisk SPY LÅG · Norden 5/9 GRÖN' — startsidans rad."""
    return " · ".join(f"{t} {w}" for t, w, _s, _c in items)


def load() -> dict:
    """Det sparade: allokeraren ur repot, resten ur Gisten. Aldrig ett fel."""
    out = {}
    try:
        import storage
        out["alloc"] = storage.load_json("allocator", None)
    except Exception:
        out["alloc"] = None
    try:
        from gist_storage import load_blob, load_wolf_json
        out["regime"] = load_wolf_json("wolf_regime.json")
        out["risk"] = load_blob(RISK_BLOB, None)
        out["board"] = load_blob("valhall_board.json", None)
    except Exception:
        out.setdefault("regime", None)
        out.setdefault("risk", None)
        out.setdefault("board", None)
    return out


def render_laget_page() -> None:
    import streamlit as st

    from thor import cards
    from ui.components import note, page_header
    page_header("👁 Läget", "Var kapitalet ska ligga i dag: fyra ljus ur fyra källor, vart och ett med sitt eget ord.")
    if st.button("📡 Uppdatera", key="laget_refresh") or "laget_data" not in st.session_state:
        st.session_state["laget_data"] = load()
    d = st.session_state["laget_data"]
    items = capital_row(d.get("alloc"), d.get("regime"), d.get("risk"), d.get("board"))
    st.markdown(cards.kpi_row([(t, w, c, s) for t, w, s, c in items]), unsafe_allow_html=True)
    note("Raden slår inte ihop ljusen — varje strategi läser sitt eget enligt sina regler.")
    st.markdown("---")
    try:
        from wolf_regime_ui import render_wolf_regime_page
        render_wolf_regime_page()
    except Exception as exc:
        note(f"Swing Regime kunde inte visas: {exc}")
    st.markdown("---")
    if st.toggle("🌩️ Visa Marknadsrisken (räknas live, tar en stund)", key="laget_risk"):
        try:
            from market_risk_ui import render_market_risk_page
            render_market_risk_page()
        except Exception as exc:
            note(f"Marknadsrisken kunde inte visas: {exc}")
