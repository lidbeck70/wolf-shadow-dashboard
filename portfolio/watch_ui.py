"""
portfolio/watch_ui.py — 💼 PORTFÖLJ → 👀 Övervakning: aktierna du bevakar med strategin, och de som är nära köp nu.

Inga nya regler och ingen ny skanning. Läget per (aktie, strategi) läses ur Thors rader (watchlist.status_of) och
avståndet ur körningens egna fält (thor.near). Sidan delar Thors data i sessionen (st.session_state["thor_data"]),
så Thor och Övervakningen läser Gisten en gång; 📡 Uppdatera läser om den för båda. Listan sparas i
data/watchlist.json (bara inmatningar — läget skrivs aldrig vid ritning).

  Min bevakning      Aktie · Strategi · Läge · Strategins ord · Avstånd · Entry/Stopp · Körd · Sedan · Vid tillägg,
                     och per rad → Thor (strategin förvald) och ✕ Ta bort. 💲 Hämta kurser lägger kursen bredvid
                     (bara i sessionen). Egna nivåer och en anteckning per rad.
  Nära köp nu        Thors BEVAKA-rader — strategins eget "nästan" — de första per strategi i listans ordning,
                     med 👀 för att bevaka. Svarar på frågan även innan något är bevakat.
"""
from __future__ import annotations

from typing import Optional

AUTO_TOP = 5                                # Nära köp nu: rader per strategi
_PRICES = "_watch_prices"                   # {ticker: (kurs, datum)} — bara sessionen, sparas aldrig


# ── Rena funktioner ─────────────────────────────────────────────────────────
def _num(x) -> str:
    from thor import cards
    return cards._num(x)


def _pct(x: Optional[float]) -> str:
    return f"{x:+.1f} %" if x is not None else "—"


def levels_text(s: dict) -> str:
    """'110.00 / 104.00' (entry / stopp) — '(egna)' när nivåerna är dina, inte strategins."""
    if s.get("entry") is None and s.get("stop") is None:
        return "—"
    return f"{_num(s.get('entry'))} / {_num(s.get('stop'))}" + (" (egna)" if s.get("own") else "")


def at_add_text(item: dict) -> str:
    """Thors läge när paret lades till: 'BEVAKA · rad 5' — 'ingen rad' när strategin saknade aktien då."""
    a = item.get("at_add") or {}
    return " · ".join(x for x in (a.get("class"), a.get("status")) if x) or "ingen rad"


def lage(s: dict) -> str:
    """Läget i tabellen: strategins klass, 'GAMMAL' för en körning äldre än gränsen, '—' utan rad."""
    if not s.get("in_list"):
        return "—"
    return f"{s['status_class']} · GAMMAL" if s.get("stale") else s["status_class"]


def lage_color(value) -> str:
    """Lägets färg som Thors kort (thor.cards.CLASS_COLOR); gammalt och saknat i grått."""
    from thor import cards
    from ui.tokens import GREY
    v = str(value or "")
    col = GREY if "GAMMAL" in v else cards.CLASS_COLOR.get(v, GREY)
    return f"color: {col}; font-weight: 700"


def missing_text(strategy: str, data: dict) -> str:
    """Varför paret saknar rad: Gisten oläslig, strategins körning fallerade — eller bara inte i dagens lista."""
    import watchlist as wl
    from thor import live
    if "gist" in (data.get("errors") or {}):
        return "körningarna gick inte att läsa"
    failed = live.blob_errors(data).get(strategy)
    if failed:
        return f"körningen fallerade ({failed[0]})"
    return wl.NOT_IN_RUN


def table_rows(items: list, rows: list, data: dict, prices: Optional[dict] = None) -> list:
    """Min bevakning som tabellrader. prices = {ticker: (kurs, datum)} ger Kurs, Till entry och Över stopp."""
    import watchlist as wl
    from thor import near
    from thor import registry as reg
    out = []
    for it in items:
        s = wl.status_of(it, rows)
        word = (" · ".join(x for x in (s["status"], s["score_text"]) if x) if s["in_list"]
                else missing_text(it["strategy"], data))
        row = {"Aktie": it["ticker"], "Strategi": reg.label(it["strategy"]), "Läge": lage(s),
               "Strategins ord": word, "Avstånd": near.near_text(s["candidate"], data) if s["in_list"] else "—",
               "Entry/Stopp": levels_text(s), "Körd": s["asof"] or "—", "Sedan": it["added"] or "—",
               "Vid tillägg": at_add_text(it)}
        if prices is not None:
            px = (prices.get(it["ticker"]) or (None, ""))[0]
            row.update({"Kurs": _num(px) if px is not None else "—", "Till entry": _pct(wl.distance_pct(px, s["entry"])),
                        "Över stopp": _pct(wl.distance_pct(px, s["stop"]))})
        if it.get("note"):
            row["Anteckning"] = it["note"]
        out.append(row)
    return out


def near_buy(rows: list, top: int = AUTO_TOP) -> list:
    """[(strategi, [rader])]: Thors BEVAKA-rader per strategi, de första i listans ordning (tc.sort_key), i
    registrets ordning. BEVAKA är strategins eget 'nästan' — inget annat räknas in."""
    from thor import candidates as tc
    from thor import registry as reg
    by: dict = {}
    for c in sorted(rows or [], key=tc.sort_key):
        if c.status_class == tc.BEVAKA and reg.get(c.strategy):
            by.setdefault(c.strategy, []).append(c)
    return [(k, by[k][:top]) for k in reg.ORDER if k in by]


def last_prices(tickers: list, closes=None) -> dict:
    """{ticker: (senaste stängning, datum)} via market_prices.closes (Börsdata först, Yahoo som reserv)."""
    if closes is None:
        from market_prices import closes
    out = {}
    for t, s in (closes(list(tickers), "1mo") or {}).items():
        try:
            out[str(t).upper()] = (float(s.iloc[-1]), str(s.index[-1])[:10])
        except Exception:
            continue
    return out


# ── Callbacks (körs före omritningen) ───────────────────────────────────────
def _toast(msg: str) -> None:
    import streamlit as st
    st.toast(msg, icon="👀")


def _remove(item_id: str) -> None:
    import watchlist as wl
    _toast(wl.remove(item_id)[1])


def _pin(ticker: str, name: str, strategy: str, at_add: dict) -> None:
    import watchlist as wl
    _toast(wl.add(ticker, strategy, name=name, source="auto", at_add=at_add)[1])


# ── Sidan ───────────────────────────────────────────────────────────────────
def _data(st, refresh: bool = False) -> dict:
    """Thors data ur sessionen — läses (och sparas där för Thor) först när den saknas eller vid 📡 Uppdatera."""
    from thor import ui as tui
    data = st.session_state.get(tui._STATE)
    if refresh or not data:
        with st.spinner("Läser körningarna ur Gisten …"):
            data = tui.rebuild(tui.load())
        st.session_state[tui._STATE] = data
    return data


def render_watch_page() -> None:
    import pandas as pd
    import streamlit as st

    import storage_ui
    import watchlist as wl
    from ui.components import note, page_header
    from ui.tokens import DIM
    page_header("👀 Övervakning", "Aktier nära köpläge, med strategin — läget ur strategins egen körning, inga nya "
                "regler. Lägg till med 👀 i Thor.")
    store = wl.load()
    storage_ui.save_bar(wl.STORE, "Övervakning")
    c1, c2 = st.columns([1, 3])
    data = _data(st, refresh=c1.button("📡 Uppdatera", key="watch_refresh"))
    rows = data.get("rows") or []
    c2.markdown(f"<div style='color:{DIM};font-size:0.78rem;padding-top:8px;'>körningarna hämtade {data.get('loaded')} · "
                f"{len(rows)} rader i Thor</div>", unsafe_allow_html=True)
    if "gist" in (data.get("errors") or {}):
        note("Gisten gick inte att läsa — läget nedan kan inte visas. Tryck 📡 Uppdatera om en stund.")
    _mine(st, pd, store["items"], rows, data)
    _auto(st, rows, data, wl.keys(store))
    note("Läget är strategins eget verdikt ur senaste körningen, översatt rakt av som i Thor — GAMMAL betyder att "
         "körningen är äldre än gränsen. 'Inte i dagens körning' betyder att strategins lista inte har aktien i dag "
         "(listorna bär bara sina bästa rader); → Thor bedömer den med strategins motor. Inga order läggs här.")
    if not store["items"]:
        return
    with st.expander("✏️ Egna nivåer och anteckning", expanded=False):
        _levels(st, store["items"])


def _mine(st, pd, items: list, rows: list, data: dict) -> None:
    """Min bevakning: tabellen och knapparna per rad."""
    from thor import registry as reg
    from ui.components import empty_state, section
    from ui.links import go
    st.markdown(section("Min bevakning"), unsafe_allow_html=True)
    if not items:
        empty_state("Inget bevakas än — tryck 👀 Lägg i Övervakning på en aktie i Thor, eller 👀 på en rad under "
                    "Nära köp nu.")
        return
    if st.button("💲 Hämta kurser", key="watch_prices", help="Senaste stängning (Börsdata, annars Yahoo) — bara i "
                 "sessionen, sparas inte"):
        with st.spinner("Hämtar kurser …"):
            try:
                st.session_state[_PRICES] = last_prices([i["ticker"] for i in items])
            except Exception as exc:
                st.warning(f"Kurserna gick inte att hämta: {exc}")
    df = pd.DataFrame(table_rows(items, rows, data, st.session_state.get(_PRICES))).fillna("")
    st.dataframe(df.style.map(lage_color, subset=["Läge"]), hide_index=True, width="stretch")
    for it in items:
        a, b = st.columns([3, 1])
        a.button(f"{it['ticker']} → Thor ({reg.label(it['strategy'])})", key=f"watch_open_{it['id']}", on_click=go,
                 args=("thor", it["ticker"], it["strategy"]), width="stretch")
        b.button("✕ Ta bort", key=f"watch_del_{it['id']}", on_click=_remove, args=(it["id"],), width="stretch")


def _auto(st, rows: list, data: dict, watched: set) -> None:
    """Nära köp nu: strategiernas BEVAKA-rader med 👀 för att bevaka dem."""
    import watchlist as wl
    from thor import cards
    from thor import near
    from thor import registry as reg
    from ui.components import note, section
    from ui.tokens import AMBER
    groups = near_buy(rows)
    st.markdown(section(f"Nära köp nu (automatiskt) · {sum(len(g) for _k, g in groups)} BEVAKA"), unsafe_allow_html=True)
    if not groups:
        note("Ingen strategi har en BEVAKA-rad i dag (eller så har körningarna inte lästs) — se Thor.")
        return
    note(f"Strategins egen BEVAKA ur senaste körningen, de {AUTO_TOP} första per strategi i Thors ordning. "
         "👀 lägger aktien i din bevakning med strategin.")
    for key, group in groups:
        with st.expander(f"{reg.label(key)} · {len(group)} BEVAKA", expanded=False):
            for c in group:
                a, b = st.columns([5, 1])
                a.markdown(cards.candidate_card(c) + f"<div style='color:{AMBER};font-size:0.74rem;margin:0 0 6px 4px;'>"
                           f"→ {cards._e(near.near_text(c, data))}</div>", unsafe_allow_html=True)
                done = (c.ticker, c.strategy) in watched
                b.button("✓" if done else "👀", key=f"watch_pin_{c.strategy}_{c.ticker}", disabled=done,
                         help="Bevakas redan" if done else f"Lägg {c.ticker} i Övervakning ({reg.label(key)})",
                         on_click=_pin, args=(c.ticker, c.name, c.strategy, wl.snapshot(c)), width="stretch")


def _levels(st, items: list) -> None:
    """Egna nivåer och anteckning för en rad — dina inmatningar; strategins nivåer går före när de finns."""
    import watchlist as wl
    from positions import UNSAVED
    from thor import registry as reg
    from ui.components import note
    ids = [i["id"] for i in items]
    by = {i["id"]: i for i in items}
    if st.session_state.get("watch_lv_pick") not in ids:          # vald rad borttagen: börja om på första
        st.session_state.pop("watch_lv_pick", None)
    pick = st.selectbox("Rad", ids, key="watch_lv_pick",
                        format_func=lambda k: f"{by[k]['ticker']} · {reg.label(by[k]['strategy'])}" if k in by else k)
    it = by.get(pick)
    if it is None:
        return
    lv = it.get("levels") or {}
    c1, c2, c3 = st.columns(3)
    entry = c1.number_input("Entry", min_value=0.0, value=lv.get("entry"), key=f"watch_lv_entry_{pick}")
    stop = c2.number_input("Stopp", min_value=0.0, value=lv.get("stop"), key=f"watch_lv_stop_{pick}")
    target = c3.number_input("Mål", min_value=0.0, value=lv.get("target"), key=f"watch_lv_target_{pick}")
    text = st.text_input("Anteckning", value=it.get("note") or "", key=f"watch_lv_note_{pick}")
    note("Egna nivåer visas när strategins körning inte har några (märkta '(egna)'). De ändrar inga regler.")
    if st.button("Spara nivåer", key=f"watch_lv_save_{pick}"):
        ok, msg = wl.set_levels(pick, entry, stop, target, note=text)
        (st.success if ok and UNSAVED not in msg else st.warning)(msg)
