"""
ui/settings.py — ⚙️ Inställningar (designförslaget PR 4, avsnitt 6): kapitalet, reglerna, länkadressen och
datakällorna på ett ställe.

Sidan samlar och visar. Den ändrar inga standardvärden av sig själv: kapitalet sparas i data/settings.json, och
vilka sidor som läser det är beslut 5 (eget steg). Reglerna visas låsta ur strategy_rules.py och allokeraren —
den enda sanningen; här går de inte att ändra. Gamla Viking läser aldrig den här filen.
"""
from __future__ import annotations

from typing import Optional

STORE = "settings"                       # data/settings.json via storage (samma väg som arken)
DEFAULTS: dict = {"capital": None}
BENCHMARKS = (("Wolf Regime", "benchmark-väljaren på sidan (SPY/OMXS30)", "XOM"),
              ("Viking Regime", "egen väljare — gamla Viking, rörs inte", "VOLV-B.ST"),
              ("Viking Nine Regime", "gamla Viking, rörs inte", "SPY"),
              ("Long Trend", "benchmark-väljaren på sidan", "ATCO-A.ST"))


def capital_of(data) -> Optional[float]:
    """Det sparade kapitalet, eller None när inget är ifyllt (sidorna använder då sina egna standardvärden)."""
    try:
        v = float((data or {}).get("capital"))
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def capital(state=None) -> Optional[float]:
    """Kapitalet ur sessionen (läses en gång per session ur data/settings.json). Aldrig ett fel — None vid problem."""
    try:
        if state is None:
            import storage
            return capital_of(storage.session_load(STORE, dict(DEFAULTS)))
        return capital_of(state.get(STORE))
    except Exception:
        return None


def capital_or(default: float) -> float:
    """Standardvärdet för ett kapitalfält (beslut 5): ditt kapital ur Inställningar, annars sidans eget värde.
    Fältet finns kvar på sidan för tillfälliga räkningar. Gamla Viking anropar aldrig den här."""
    return float(capital() or default)


def rules_rows() -> list:
    """Den låsta regeltabellen: risk per affär, positionsstorlek, max positioner och stopp per strategi ur
    strategy_rules.PLAYBOOKS, plus allokerarens hårda tak (% av hela portföljen)."""
    import strategy_rules as sr
    try:
        import allocator as al
        cap = {k: al.rule_for_strategy(k) for k in sr.PLAYBOOKS}
    except Exception:
        cap = {}
    rows = []
    for key, p in sr.PLAYBOOKS.items():
        r = cap.get(key)
        rows.append({"Strategi": p.name, "Risk per affär": p.risk.risk_per_trade, "Position": p.risk.position_size,
                     "Max positioner": p.risk.max_positions, "Stopp": p.risk.stop,
                     "Hårt tak (allokeraren)": f"{r.hard_cap:g} % av portföljen" if r else "—"})
    return rows


def render_settings_page() -> None:
    import pandas as pd
    import streamlit as st

    import storage
    import storage_ui
    from ui.components import note, page_header
    page_header("⚙️ Inställningar", "Kapital, regler, länkadress och datakällor på ett ställe. Sidan visar och "
                "sparar — den ändrar inga regler.")
    data = storage.session_load(STORE, dict(DEFAULTS))
    if not isinstance(data, dict):
        data = dict(DEFAULTS)
        st.session_state[STORE] = data
    storage_ui.save_bar(STORE, "Inställningar")

    st.markdown("#### Kapital")
    cur = capital_of(data) or 0.0
    val = st.number_input("Ditt kapital (kr) — 0 = inte ifyllt", min_value=0.0, value=float(cur), step=5000.0,
                          key="settings_capital")
    if float(val or 0) != float(cur):
        data["capital"] = float(val) if val else None
        st.session_state[STORE] = data
    note("Kapitalet blir standardvärde i Wolf Regime, Thor, Position Sizing och journalens startkapital. "
         "Sidornas egna fält finns kvar för tillfälliga räkningar. Gamla Viking läser det aldrig.")

    st.markdown("#### Reglerna (låsta)")
    st.dataframe(pd.DataFrame(rules_rows()), hide_index=True, use_container_width=True)
    note("Ur strategy_rules.py och allokeraren — den enda sanningen. Ändras bara där, med ditt beslut.")

    st.markdown("#### Benchmark och standardbolag")
    st.dataframe(pd.DataFrame([{"Sida": a, "Benchmark": b, "Standardbolag": c} for a, b, c in BENCHMARKS]),
                 hide_index=True, use_container_width=True)
    note("Ändras inte härifrån: benchmark är indata till grindarna för relativ styrka.")

    st.markdown("#### Länkadress i larmen")
    note("Discord-larmen får en länk till rätt vy när hemligheten PANEL_URL finns i GitHub (Settings → Secrets → "
         "Actions) med panelens adress. Utan den skickas larmen som förut, utan länk.")

    with st.expander("Positionskalkylator", expanded=False):
        try:
            from position_sizing import render_position_sizing
            render_position_sizing()
        except Exception as exc:
            note(f"Kalkylatorn kunde inte visas: {exc}")

    with st.expander("Datakällor", expanded=False):
        try:
            from data_health import render_data_health
            render_data_health()
        except Exception as exc:
            note(f"Datakällorna kunde inte visas: {exc}")
