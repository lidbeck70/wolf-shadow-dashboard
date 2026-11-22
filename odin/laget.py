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
    levels = (blob.get("levels") if isinstance(blob, dict) else None) or {}
    levels = levels if isinstance(levels, dict) else {}
    parts = [f"{(v or {}).get('label') or m} {(v or {}).get('level')}" for m, v in levels.items()
             if isinstance(v, dict) and v.get("level")]
    if not parts:
        return SAKNAS, "ingen sparad nivå än"
    return " · ".join(parts), str((blob or {}).get("generated") or "")[:16]


def risk_worst(blob: Optional[dict]) -> str:
    order = {"LÅG": 0, "FÖRHÖJD": 1, "HÖG": 2}
    levels = (blob.get("levels") if isinstance(blob, dict) else None) or {}
    lv = [v.get("level") for v in (levels.values() if isinstance(levels, dict) else []) if isinstance(v, dict)]
    lv = [x for x in lv if x in order]
    return max(lv, key=order.get) if lv else SAKNAS


def nordic_light(board: Optional[dict]) -> tuple:
    """('5/9 GRÖN', ...) — antal nordiska segment som är GRÖNA av dem som har ett läge."""
    from thor import candidates as tc
    raw = board.get("segments") if isinstance(board, dict) else None
    segs = tc.board_segments(board) or [r for r in (raw if isinstance(raw, list) else []) if isinstance(r, dict)]
    st = [((r.get("nordic") if isinstance(r.get("nordic"), dict) else None) or {}).get("status")
          for r in segs if isinstance(r, dict)]
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


SUBTITLE = "Var kapitalet ska ligga i dag: fyra ljus ur fyra källor, vart och ett med sitt eget ord."
NOTE_TEXT = "Raden slår inte ihop ljusen — varje strategi läser sitt eget enligt sina regler."
RISK_LABEL = "🌩️ Visa Marknadsrisken (räknas live, tar en stund)"

# Sidans egna regler (LIDSKJALV §4.4), i en dold stilbehållare: Marknadsriskens reglagerad på 56 px som ett kort,
# "Kolla en ticker"-fältet på 48 px med 16 px text (iOS zoomar under 16 px) och hopfällbara rutor (Historik,
# Marknadsriskens status) med en tryckyta på 44 px
_ROOT = '.st-key-swreg_ticker [data-testid="stTextInputRootElement"]'
_CSS = f"""
.st-key-laget_riskrow {{ min-height: 56px; justify-content: center; padding: 0 14px; margin: 12px 0 4px;
    background: #14141e; border: 1px solid rgba(255,255,255,.08); border-radius: 12px; box-sizing: border-box; }}
.st-key-laget_riskrow label p {{ font-size: 15px; line-height: 1.35; }}
/* Hela kortet är tryckytan (§2.4): reglagets etikett fyller raden, inte bara texten */
.st-key-laget_riskrow [data-testid="stElementContainer"], .st-key-laget_riskrow [data-testid="stCheckbox"],
.st-key-laget_riskrow label {{ width: 100% !important; max-width: none; }}
.st-key-laget_riskrow label {{ min-height: 54px; align-items: center; cursor: pointer; }}
{_ROOT} {{ height: 48px; border-radius: 10px; overflow: hidden; }}
{_ROOT} input {{ height: 100%; font-size: 16px !important; font-family: 'JetBrains Mono', monospace !important; }}
.st-key-laget_page [data-testid="stExpander"] summary {{ min-height: 44px; box-sizing: border-box; }}
/* Huvudraden: "👁 LÄGET" är kort — rubriken behöver inte golvet på 180 px, så 📡 och pillret ryms på samma rad */
.st-key-head_laget > [data-testid="stElementContainer"]:has(.wolf-head) {{ min-width: 0; }}
"""


def refresh() -> dict:
    """📡 på Läget — EN knapp läser om båda (beslut 11): de fyra ljusens källor och Swing-lägets wolf_regime.json
    (wolf_regime_ui._get_data(force=True), samma läsning som Swing-sidans egen knapp gjorde)."""
    import streamlit as st
    st.session_state["laget_data"] = load()
    try:
        import wolf_regime_ui
        wolf_regime_ui._get_data(force=True)
    except Exception:                                # Swing-läget visar själv att det saknas — ljusen står kvar
        pass
    return st.session_state["laget_data"]


def render_laget_page() -> None:
    """👁 Läget (§4.4): huvudraden (titel · ETT 📡 · Morgonrundans pill), de fyra ljusen som rutnät (2 × 2, fyra i rad
    från 1024 px), Swing-läget inbäddat (regimkortet och bevisen), Marknadsrisken på begäran, Kolla en ticker och
    Morgonrundans fot. Ljusen läses ur det sparade; inget räknas live förrän du slår på Marknadsrisken."""
    import streamlit as st

    from ui import routine
    from ui.components import head_row, light_grid, note, page_header
    from ui.shell import page_style
    with st.container(key="laget_page"):
        page_style(_CSS, "laget")
        with head_row("laget", subtitle=SUBTITLE):
            page_header("👁 Läget")
            clicked = st.button("📡 Uppdatera", key="laget_refresh")
            routine.pill("laget")
        if clicked:
            refresh()
        elif "laget_data" not in st.session_state:
            st.session_state["laget_data"] = load()
        d = st.session_state["laget_data"]
        items = capital_row(d.get("alloc"), d.get("regime"), d.get("risk"), d.get("board"))
        st.markdown(light_grid(items), unsafe_allow_html=True)
        note(NOTE_TEXT)
        regime = None
        try:
            import wolf_regime_ui
            regime = wolf_regime_ui.render_wolf_regime_page(embedded=True)
        except Exception as exc:
            note(f"Swing Regime kunde inte visas: {exc}")
        with st.container(key="laget_riskrow"):
            risk_on = st.toggle(RISK_LABEL, key="laget_risk")
        if risk_on:
            with st.status("Räknar marknadsrisken …", expanded=True) as box:
                try:
                    from market_risk_ui import render_market_risk_page
                    render_market_risk_page()
                    box.update(label="🌩️ Marknadsrisken — räknad live", state="complete", expanded=True)
                except Exception as exc:
                    note(f"Marknadsrisken kunde inte visas: {exc}")
                    box.update(label="🌩️ Marknadsrisken kunde inte räknas", state="error", expanded=True)
        if isinstance(regime, dict):
            try:
                wolf_regime_ui._ticker_check(regime, embedded=True)
            except Exception as exc:
                note(f"Tickerkollen kunde inte visas: {exc}")
        routine.footer("laget")
