"""
wolf_screener_ui.py — Momentum-screener (Streamlit-port av ScreenerTab).

Läser wolf_screener.json (genererad av wolf_data.py) via Gist/lokal fallback
och visar den färdigrankade momentum-listan med setup-flaggor. "→ Bevakning"
skriver kandidaten till Swing-flikens bevakningslista (samma lager).
"""

from __future__ import annotations

import streamlit as st

try:
    from gist_storage import load_wolf_json as _load_wolf
except Exception:
    _load_wolf = None

try:
    from swing import add_to_watchlist as _add_to_watchlist
except Exception:
    _add_to_watchlist = None

_FILE = "wolf_screener.json"
_CACHE = "_wolf_screener_cache"

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, BLUE, AMBER = "#2d8a4e", "#3b82f6", "#d4943a"


def _pct(n, dec: int = 1) -> str:
    try:
        return f"{float(n) * 100:.{dec}f}%"
    except (TypeError, ValueError):
        return "–"


def _get_data(force: bool = False):
    if force:
        st.session_state.pop(_CACHE, None)
    if _CACHE not in st.session_state:
        st.session_state[_CACHE] = _load_wolf(_FILE) if _load_wolf else None
    return st.session_state[_CACHE]


def render_wolf_screener_page() -> None:
    try:
        c_head, c_btn = st.columns([4, 1])
        with c_head:
            st.markdown(
                f"<h1 style='color:{TEXT};margin:0;'>Screener "
                f"<span style='color:{GREEN};'>· momentum-ranking</span></h1>",
                unsafe_allow_html=True,
            )
        with c_btn:
            if st.button("🔄 Uppdatera data", key="wolf_scr_refresh"):
                _get_data(force=True)
                st.rerun()

        data = _get_data()
        if not data or not isinstance(data, dict) or not data.get("top"):
            st.warning("Ingen screenerdata hittad ännu.")
            st.caption("Kör `python wolf_data.py` så att **wolf_screener.json** hamnar "
                       "i Gist:en eller panelens public-/data-mapp. Flikarna läser den sedan.")
            return

        st.caption(f"Genererad {data.get('generated','?')} · "
                   f"{data.get('qualifying','?')} av {data.get('universe_size','?')} kvalar")

        only_setup = st.checkbox("Visa endast setup-kandidater (A / B?)",
                                 key="wolf_scr_only_setup")
        st.markdown(
            f"<div style='color:{DIM};font-size:0.78rem;margin-bottom:6px;'>"
            f"Rankingen förutsätter att universumet redan passerat Börsdata-screenerns "
            f"kvalitetsfilter (börsvärde, F-score). Flaggor: "
            f"<b style='color:{GREEN};'>A</b> = pullback till MA20/50, RSI 35–55 · "
            f"<b style='color:{BLUE};'>B?</b> = inom 3 % av 52v-högsta.</div>",
            unsafe_allow_html=True,
        )

        rows = [r for r in data["top"]
                if not only_setup or r.get("setupA") or r.get("nearHigh")]
        if not rows:
            st.info("Inga rader matchar filtret.")
            return

        # Rubrikrad
        h = st.columns([0.5, 1.6, 1, 1, 1, 0.8, 1, 1.3, 1])
        for col, lbl in zip(h, ["#", "Ticker", "Score", "3 mån", "6 mån",
                                 "RSI", "vs MA20", "Setup", ""]):
            col.markdown(f"<span style='color:{DIM};font-size:0.72rem;'>{lbl}</span>",
                         unsafe_allow_html=True)

        for r in rows:
            rank = r.get("rank", "")
            dim = "opacity:0.5;" if isinstance(rank, (int, float)) and rank > 20 else ""
            c = st.columns([0.5, 1.6, 1, 1, 1, 0.8, 1, 1.3, 1])
            c[0].markdown(f"<span style='{dim}'>{rank}</span>", unsafe_allow_html=True)
            c[1].markdown(
                f"<span style='{dim}'><b>{r.get('ticker','?')}</b>"
                f"<br><span style='color:{DIM};font-size:0.65rem;'>"
                f"{(r.get('name') or '')[:22]}</span></span>", unsafe_allow_html=True)
            c[2].markdown(f"<span style='color:{GREEN};font-weight:700;{dim}'>"
                          f"{_pct(r.get('score'))}</span>", unsafe_allow_html=True)
            c[3].markdown(f"<span style='{dim}'>{_pct(r.get('mom3'))}</span>",
                          unsafe_allow_html=True)
            c[4].markdown(f"<span style='{dim}'>{_pct(r.get('mom6'))}</span>",
                          unsafe_allow_html=True)
            _rsi = r.get("rsi")
            rsi_c = AMBER if isinstance(_rsi, (int, float)) and _rsi > 70 else TEXT
            c[5].markdown(f"<span style='color:{rsi_c};{dim}'>{_rsi if _rsi is not None else '–'}</span>",
                          unsafe_allow_html=True)
            c[6].markdown(f"<span style='{dim}'>{_pct(r.get('dist_ma20'))}</span>",
                          unsafe_allow_html=True)
            flags = ""
            if r.get("setupA"):
                flags += f"<span style='background:{GREEN};color:#fff;font-size:0.62rem;font-weight:700;padding:1px 6px;border-radius:3px;margin-right:3px;'>A</span>"
            if r.get("nearHigh"):
                flags += f"<span style='background:{BLUE};color:#fff;font-size:0.62rem;font-weight:700;padding:1px 6px;border-radius:3px;'>B?</span>"
            if not flags and r.get("nearMA"):
                flags = f"<span style='color:{DIM};font-size:0.65rem;'>nära MA</span>"
            c[7].markdown(f"<span style='{dim}'>{flags}</span>", unsafe_allow_html=True)
            if c[8].button("→ Bevakning", key=f"wolf_scr_add_{r.get('ticker','?')}"):
                _promote_to_swing(r)

        st.markdown(
            f"<p style='color:{DIM};font-size:0.72rem;margin-top:8px;'>"
            f"Rad 1–20 = köpbara (topp 20). Rad 21–40 dämpade: rank-exit-gränsen "
            f"för befintliga innehav (regel 3), inte köpkandidater.</p>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Screener-fliken kunde inte renderas: {e}")


def _promote_to_swing(r: dict) -> None:
    if _add_to_watchlist is None:
        st.warning("Swing-fliken (bevakningslistan) är inte tillgänglig.")
        return
    mom6 = r.get("mom6")
    mom6_disp = round(float(mom6) * 100) if isinstance(mom6, (int, float)) else ""
    note = ("Setup A-kandidat (från screenern)" if r.get("setupA")
            else "Nära 52v-högsta (setup B-kandidat)" if r.get("nearHigh") else "")
    added = _add_to_watchlist(r.get("ticker", ""), mom6_disp, note)
    if added:
        st.success(f"{r.get('ticker','?')} → Swing-bevakningen.")
    else:
        st.info(f"{r.get('ticker','?')} finns redan i bevakningen (eller saknar ticker).")
