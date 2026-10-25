"""
odin/sektorrotation.py — 👁 ODIN → Sektorrotation: var ledarskapet finns i dag (ägarens "Sector Rotation
Engine", låst i docs/forskning/sektorrotation_spec.md).

Sidan visar för de 11 SPDR-sektorerna: poängen 12–6–3, de tre trendfiltren, fasen och hur rankningen ändrats —
bredvid Valhalls cykelstatus för samma sektor, så att "ledare" (momentum) och "billig och hatad" (contrarian)
syns på ett ställe. Råvaruproducenterna (GDX, SIL, COPX …) rankas för sig, som dokumentet kräver.

Sidan räknar live på Yahoo-kurser (cache sex timmar) och visar backtestets utfall (sector_rotation_backtest.json).
Tills backtestet godkänt modellen står den som "ej validerad". Ingen regel och ingen grind byggs här.
"""
from __future__ import annotations

from typing import Callable, Optional

BACKTEST_BLOB = "sector_rotation_backtest.json"
PERIOD = "5y"
FILTER_MARK = {True: "✓", False: "✗"}
PHASE_COLOR_KEY = {"2 Bekräftad ledare": "green", "1 Tidig rotation": "cyan", "3 Mognande ledare": "gold",
                   "4 Förlorat ledarskap": "red", "Neutral": "dim"}


def _closes(getter: Callable, symbols, period: str = PERIOD):
    import pandas as pd
    from valhall_event_study import _close
    cols = {}
    for s in symbols:
        try:
            c = _close(getter(s, period))
        except Exception:
            c = None
        if c is not None:
            cols[s] = c
    return pd.DataFrame(cols)


def compute_live(getter: Callable) -> dict:
    """{sectors: [rad], producers: [rad], climb_1m, climb_3m, date} ur dagens kurser. Aldrig ett fel uppåt."""
    from odin import rotation_engine as re_
    from valhall import long_paper as vlp
    from valhall import segments as sgm
    from valhall_event_study import _close
    spy = _close(getter("SPY", PERIOD))
    if spy is None:
        return {"error": "SPY saknas"}
    labels = {s.spdr: s.label for s in sgm.SEGMENTS}
    out = {}
    for key, syms, lab in (("sectors", list(labels), labels),
                           ("producers", list(vlp.PRODUCER_SEGMENT), {s: s for s in vlp.PRODUCER_SEGMENT})):
        daily = _closes(getter, syms).reindex(spy.index).ffill(limit=5)
        if daily.shape[1] < 2:
            out[key] = []
            continue
        sig = re_.compute(daily, spy)
        out[key] = re_.table(sig, labels=lab)
        if key == "sectors":
            out["climb_1m"], out["climb_3m"] = re_.climbers(sig, 1), re_.climbers(sig, 3)
            out["date"] = str(sig["S"].index[-1].date())
    return out


def valhall_status(board: Optional[dict]) -> dict:
    """{SPDR: global status} ur Valhalls karta (valhall_board.json)."""
    from valhall import segments as sgm
    spdr = {s.key: s.spdr for s in sgm.SEGMENTS}
    segs = board.get("segments") if isinstance(board, dict) else None
    out = {}
    for r in segs if isinstance(segs, list) else []:
        if isinstance(r, dict) and r.get("key") in spdr:
            g = r.get("global") if isinstance(r.get("global"), dict) else {}
            out[spdr[r["key"]]] = g.get("status")
    return out


def validation(blob: Optional[dict]) -> tuple:
    """(rubrik, färgnyckel, förklaring) ur backtestets utfall — 'ej testad' innan första körningen."""
    v = (blob or {}).get("verdict") if isinstance(blob, dict) else None
    if not isinstance(v, dict):
        return "EJ TESTAD", "dim", "Backtestet har inte körts än — siffrorna är en hypotes."
    if v.get("R1") and v.get("R2"):
        return "GODKÄND", "green", f"R1 och R2 godkända{' och statistiskt säkerställd' if v.get('S') else ''} — {v.get('use')}"
    if v.get("R2"):
        return "RISKFILTER", "gold", f"Bara R2 godkänd — {v.get('use')}"
    return "EJ VALIDERAD", "red", "Backtestet godkände inte modellen — visas som information, ingen grind."


def ui_rows(rows: list, status: Optional[dict] = None) -> list:
    """Tabellraderna för sidan."""
    out = []
    for r in rows or []:
        out.append({"Sektor": r.get("label") or r["ticker"], "Fond": r["ticker"], "Fas": r.get("phase"),
                    "Poäng S": r.get("S"), "R12": r.get("R12"), "R6": r.get("R6"), "R3": r.get("R3"),
                    "Filter": "".join(FILTER_MARK[bool(r.get(k))] for k in ("f1", "f2", "f3")),
                    "Klättring 3 mån": r.get("climb"), "Valhall": (status or {}).get(r["ticker"]) or "—"})
    return out


def climb_text(pairs: list, n: int = 3) -> str:
    """'XLE +4 · XLB +2 … / XLK −3' — de som klättrat mest och fallit mest."""
    up = [f"{t} +{v}" for t, v in pairs if v > 0][:n]
    down = [f"{t} {v}" for t, v in reversed(pairs) if v < 0][:n]
    return f"upp: {' · '.join(up) or '—'}   ·   ned: {' · '.join(down) or '—'}"


def render_sektorrotation_page() -> None:
    import pandas as pd
    import streamlit as st

    from thor import cards
    from ui.components import note, page_header
    from ui.tokens import CYAN, DIM, GOLD, GREEN, RED
    page_header("Sektorrotation", "Var ledarskapet finns: relativ styrka 12–6–3 månader, trendfilter och fas per "
                "sektor — bredvid Valhalls cykelstatus.")

    @st.cache_data(ttl=6 * 3600, show_spinner="Räknar sektorrotationen …")
    def _live() -> dict:
        from market_prices import ohlcv
        try:
            return compute_live(ohlcv)
        except Exception as exc:
            return {"error": str(exc)}

    @st.cache_data(ttl=600, show_spinner=False)
    def _saved() -> tuple:
        try:
            from gist_storage import load_blob
            return load_blob(BACKTEST_BLOB, None), load_blob("valhall_board.json", None)
        except Exception:
            return None, None

    if st.button("📡 Uppdatera", key="rotation_refresh"):
        _live.clear()
        _saved.clear()
    data = _live()
    blob, board = _saved()
    word, ck, text = validation(blob)
    col = {"green": GREEN, "gold": GOLD, "red": RED, "dim": DIM}[ck]
    by = ((blob or {}).get("by_period") or {}).get("Hela") or {}
    rot, spy = by.get("Rotation topp 3") or {}, by.get("SPY") or {}
    st.markdown(cards.kpi_row([
        ("Backtestet", word, col, text[:80]),
        ("Rotation topp 3", f"{rot.get('cagr')} %/år" if rot.get("cagr") is not None else "—", CYAN,
         f"max DD {rot.get('max_dd')} %" if rot else "efter körning"),
        ("SPY", f"{spy.get('cagr')} %/år" if spy.get("cagr") is not None else "—", DIM,
         f"max DD {spy.get('max_dd')} %" if spy else "efter körning")]), unsafe_allow_html=True)
    note(text)
    if data.get("error"):
        note(f"Kurserna kunde inte hämtas: {data['error']}")
        return
    st.markdown(f"#### Sektorerna · månadsskiftet {data.get('date') or '—'}")
    rows = ui_rows(data.get("sectors"), valhall_status(board))
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    note("Filter: över 10-månaders snittet · snittet stiger · relativa linjen mot SPY över sitt 6-månaders snitt. "
         "Fas 2 = bekräftad ledare (momentum); Valhalls GRÖN/RÖD är cykeln (contrarian) — två olika frågor.")
    st.markdown(f"**Rankningen 1 mån:** {climb_text(data.get('climb_1m') or [])}")
    st.markdown(f"**Rankningen 3 mån:** {climb_text(data.get('climb_3m') or [])}")
    st.markdown("#### Undergrupper: råvaruproducenter")
    prod = ui_rows(data.get("producers"))
    if prod:
        st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if k not in ("Sektor", "Valhall")} for r in prod]),
                     hide_index=True, width="stretch")
    note("Producenterna rankas mot varandra och mot SPY — gruvbolag har andra drivkrafter än materialsektorn. "
         "Makrolagret (räntor, krediter, dollar) finns under ODIN → Makro. Nordiska branschindex visas inte: de byggs "
         "ur dagens bolag.")
