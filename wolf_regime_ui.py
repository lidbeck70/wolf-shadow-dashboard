"""
wolf_regime_ui.py — Swing-marknadsregim (Streamlit-port av RegimeTab).

Läser wolf_regime.json (genererad av wolf_data.py) via Gist/lokal fallback och
visar trafikljus + regelverk, index vs MA200, marknadsbredd (dör före index),
antal kvalande bolag och 26-veckors historik.
"""

from __future__ import annotations

import streamlit as st

try:
    from gist_storage import load_wolf_json as _load_wolf
except Exception:
    _load_wolf = None

_FILE = "wolf_regime.json"
_CACHE = "_wolf_regime_cache"

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, BLUE, GREY = "#2d8a4e", "#d4943a", "#c44545", "#3b82f6", "#6b7280"

_STYLE = {
    "GRÖN":  {"bg": "#12351f", "bd": GREEN, "dot": GREEN, "label": "GRÖN — full gas enligt reglerna"},
    "GUL":   {"bg": "#3a2a10", "bd": AMBER, "dot": AMBER, "label": "GUL — selektiv, halv storlek"},
    "RÖD":   {"bg": "#3a1414", "bd": RED,   "dot": RED,   "label": "RÖD — inga nya köp"},
    "OKÄND": {"bg": "#1a1f25", "bd": GREY,  "dot": GREY,  "label": "OKÄND — kontrollera datakällan"},
}


def _pct(n, dec: int = 1) -> str:
    try:
        return f"{float(n) * 100:.{dec}f}%"
    except (TypeError, ValueError):
        return "–"


def _num(n, default=None):
    try:
        f = float(n)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def _get_data(force: bool = False):
    if force:
        st.session_state.pop(_CACHE, None)
    if _CACHE not in st.session_state:
        st.session_state[_CACHE] = _load_wolf(_FILE) if _load_wolf else None
    return st.session_state[_CACHE]


def render_wolf_regime_page() -> None:
    try:
        c_head, c_btn = st.columns([4, 1])
        with c_head:
            st.markdown(f"<h1 style='color:{TEXT};margin:0;'>Regim</h1>",
                        unsafe_allow_html=True)
        with c_btn:
            if st.button("🔄 Uppdatera data", key="wolf_reg_refresh"):
                _get_data(force=True)
                st.rerun()

        d = _get_data()
        if not d or not isinstance(d, dict):
            st.warning("Ingen regimdata hittad ännu.")
            st.caption("Kör `python wolf_data.py` så att **wolf_regime.json** hamnar "
                       "i Gist:en eller panelens public-/data-mapp.")
            return

        st.caption(f"Uppdaterad {d.get('generated','?')}")

        stl = _STYLE.get(d.get("regime"), _STYLE["OKÄND"])
        rules_html = "".join(f"<li>• {r}</li>" for r in d.get("rules", []))
        st.markdown(
            f"<div style='background:{stl['bg']};border:1px solid {stl['bd']};"
            f"border-radius:12px;padding:16px;margin-bottom:12px;'>"
            f"<div style='display:flex;align-items:center;gap:10px;'>"
            f"<span style='width:16px;height:16px;border-radius:50%;background:{stl['dot']};"
            f"display:inline-block;'></span>"
            f"<span style='font-size:1.25rem;font-weight:800;color:{TEXT};'>{stl['label']}</span></div>"
            f"<ul style='margin:10px 0 0 0;padding-left:18px;color:{TEXT};font-size:0.88rem;'>"
            f"{rules_html}</ul></div>",
            unsafe_allow_html=True,
        )

        idx = d.get("index")
        breadth = _num(d.get("breadth"), 0.0)
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(f"<div style='color:{DIM};font-size:0.7rem;text-transform:uppercase;'>"
                        f"OMXSPI vs MA200</div>", unsafe_allow_html=True)
            if idx:
                above = idx.get("above")
                col = GREEN if above else RED
                st.markdown(
                    f"<div style='font-size:1.5rem;font-weight:800;color:{TEXT};'>"
                    f"{_pct(idx.get('dist'))} "
                    f"<span style='font-size:0.85rem;color:{col};'>"
                    f"{'över' if above else 'UNDER'}</span></div>",
                    unsafe_allow_html=True)
                spark = idx.get("spark") or []
                if len(spark) >= 2:
                    st.line_chart(spark, height=70)
                st.caption(f"{idx.get('close','?')} vs MA200 {idx.get('ma200','?')} · 30 dagar")
            else:
                st.markdown(f"<span style='color:{DIM};'>saknas</span>", unsafe_allow_html=True)

        with m2:
            st.markdown(f"<div style='color:{DIM};font-size:0.7rem;text-transform:uppercase;'>"
                        f"Marknadsbredd (andel över MA200)</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.5rem;font-weight:800;color:{TEXT};'>"
                        f"{_pct(breadth)}</div>", unsafe_allow_html=True)
            bcol = GREEN if breadth > 0.55 else AMBER if breadth > 0.45 else RED
            st.markdown(
                f"<div style='background:#2a2a38;border-radius:5px;height:8px;margin-top:4px;'>"
                f"<div style='width:{min(breadth*100,100):.0f}%;background:{bcol};height:8px;"
                f"border-radius:5px;'></div></div>",
                unsafe_allow_html=True)
            st.caption("> 55 % friskt · 45–55 % smalnar · < 45 % bredden dör — den dör före index.")

        with m3:
            st.markdown(f"<div style='color:{DIM};font-size:0.7rem;text-transform:uppercase;'>"
                        f"Kvalande bolag</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:1.5rem;font-weight:800;color:{TEXT};'>"
                f"{d.get('qualifying','?')} <span style='font-size:0.85rem;color:{DIM};'>"
                f"av {d.get('universe','?')}</span></div>", unsafe_allow_html=True)
            st.caption("Krympande lista vecka för vecka = momentum smalnar.")

        _history(d.get("history", []))
    except Exception as e:
        st.error(f"Regim-fliken kunde inte renderas: {e}")


def _history(hist: list) -> None:
    if not hist:
        return
    st.markdown(f"<div style='color:{DIM};font-size:0.7rem;text-transform:uppercase;"
                f"margin:14px 0 6px;'>Regim-historik — senaste {len(hist)} handelsdagarna "
                f"(regim · bredd)</div>",
                unsafe_allow_html=True)
    bar_bg = {"GRÖN": "#12351f", "GUL": "#3a2a10", "RÖD": "#3a1414"}
    cells = ""
    for h in hist:
        bg = bar_bg.get(h.get("regime"), "#1a1f25")
        fill = min(_num(h.get("breadth"), 0.0) * 100, 100)
        tip = f"{h.get('date','')}: {h.get('qualifying','?')} kvalar, bredd {fill:.0f}%"
        cells += (
            f"<div title=\"{tip}\" style='width:22px;height:40px;background:{bg};"
            f"border-radius:3px;display:flex;flex-direction:column;justify-content:flex-end;'>"
            f"<div style='height:{fill:.0f}%;background:{BLUE};opacity:0.75;"
            f"border-radius:0 0 3px 3px;'></div></div>")
    st.markdown(f"<div style='display:flex;gap:3px;flex-wrap:wrap;'>{cells}</div>",
                unsafe_allow_html=True)
    st.caption("Stapelfärg = regim den veckan · fyllnad = bredd. "
               "GRÖN med fallande fyllnad är hur toppar ser ut.")
