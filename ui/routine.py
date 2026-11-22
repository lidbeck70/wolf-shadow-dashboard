"""
ui/routine.py — Morgonrundan (LIDSKJALV §3.5.20, beslut 7): I DAG → Läget → Thor → 🎯 Scorecard → Positioner
på fyra tryck.

Bara navigering. Rundan minns vilka steg som tagits i dag — st.session_state["morning"] = {"date", "seen"} — och
aldrig data eller en regel. Den startar när du trycker pillret eller "Nästa" på I DAG; då visar de fyra andra
sidorna pillret och foten tills rundan är klar ("✓ Morgonrundan klar"). En vanlig visning av de sidorna lämnas
ren. Ny dag = ny runda. Inget skrivs när en sida öppnas — bara när du trycker.

  pill(step)    tertiär "1/5 ›" i sidans huvudrad (morning_top_<steg>)
  footer(step)  prickarna ●○○○○ och en primär "Nästa: Läget →" över hela bredden (morning_next_<steg>)

Stegens nycklar är mål i ui/nav.TARGETS; knapparna öppnar nästa sida med ui/links.go.
"""
from __future__ import annotations

from typing import Optional

from ui import links
from ui.tokens import ACCENT, FONT_MONO, FONT_UI, MUTED

STATE_KEY = "morning"
STEPS = (("home", "I DAG"), ("laget", "Läget"), ("thor", "Thor"), ("scorecard", "🎯 Scorecard"),
         ("holdings", "Positioner"))
KEYS = tuple(k for k, _label in STEPS)
LABEL = dict(STEPS)
DONE = "✓ Morgonrundan klar · Till I DAG ↑"

# Pillret (tertiärt, 44 px, cyan 13 px), foten (prickar och en primär knapp på 48 px). Nycklarna väljs — inga
# etiketter. Knapparnas cyan fyllning är temats primärfärg (PR 16). Pillret har en tooltip (help): då ligger knappen
# ett omslag längre ned än .stButton > button — därför ättlingsselektorn.
PILL = '[class*="st-key-morning_top_"] button[kind="tertiary"]'
CSS = f"""
{PILL} {{ min-height: 44px; min-width: 44px; padding: 0 8px; border-radius: 10px; color: {ACCENT};
    background: transparent; font: 600 13px/1 {FONT_UI}; letter-spacing: .04em; text-transform: none; }}
{PILL} p, {PILL} [data-testid="stMarkdownContainer"] {{ font: inherit; color: inherit; letter-spacing: inherit; }}
{PILL}:hover, {PILL}:focus-visible {{ color: {ACCENT}; background: rgba(0,229,255,.08); }}
[class*="st-key-morning_foot_"] {{ gap: 8px; margin-top: 24px; }}
[class*="st-key-morning_next_"] .stButton > button[kind="primary"] {{ min-height: 48px; }}
.wolf-dots {{ display: flex; align-items: center; gap: 8px; font: 600 12px/1.2 {FONT_UI}; color: {MUTED};
    letter-spacing: .12em; text-transform: uppercase; }}
.wolf-dots b {{ font: 400 14px/1 {FONT_MONO}; letter-spacing: 4px; }}
"""


def _today() -> str:
    from tabs.home_today import now_se          # svensk klocka: rundan byter dag vid svensk midnatt
    return now_se().date().isoformat()


def _state(state):
    if state is None:
        import streamlit as st
        return st.session_state
    return state


def index(step: str) -> int:
    return KEYS.index(step)


def next_step(step: str) -> str:
    """Nästa stegs nyckel; efter Positioner tillbaka till I DAG."""
    i = index(step)
    return KEYS[i + 1] if i + 1 < len(KEYS) else KEYS[0]


def seen(state=None, today: Optional[str] = None) -> list:
    """Stegen som tagits i dag ([] en ny dag eller utan runda)."""
    m = _state(state).get(STATE_KEY)
    day = today or _today()
    return list(m.get("seen") or []) if isinstance(m, dict) and m.get("date") == day else []


def mark(step: str, state=None, today: Optional[str] = None) -> dict:
    """Steget är taget i dag. I DAG startar (om) rundan; en ny dag nollställer den. Bara navigering."""
    if step not in KEYS:
        raise ValueError(f"okänt steg: {step}")
    state, day = _state(state), today or _today()
    done = [] if step == KEYS[0] else seen(state, day)
    if step not in done:
        done.append(step)
    state[STATE_KEY] = {"date": day, "seen": done}
    return state[STATE_KEY]


def active(state=None, today: Optional[str] = None) -> bool:
    """Dagens runda är startad från I DAG och inte klar."""
    s = seen(state, today)
    return KEYS[0] in s and KEYS[-1] not in s


def shows(step: str, state=None, today: Optional[str] = None) -> bool:
    """På I DAG alltid; på de andra fyra sidorna bara medan rundan pågår."""
    return step == KEYS[0] or active(state, today)


def advance(step: str, state=None) -> None:
    """Knapparnas callback: steget är taget, nästa sida öppnas (ui/links.go → nav_goto)."""
    mark(step, state)
    links.go(next_step(step), state=state)


def next_label(step: str) -> str:
    """'Nästa: Läget →' — på sista steget 'Morgonrundan klar · Till I DAG ↑'."""
    return DONE if next_step(step) == KEYS[0] else f"Nästa: {LABEL[next_step(step)]} →"


def dots_html(step: str) -> str:
    """●●○○○ — cyan till och med sidans steg, MUTED för dem som återstår."""
    i = index(step)
    marks = "".join(f'<span style="color:{ACCENT if j <= i else MUTED};">{"●" if j <= i else "○"}</span>'
                    for j in range(len(KEYS)))
    return (f'<div class="wolf-dots" role="img" aria-label="Morgonrundan: steg {i + 1} av {len(KEYS)}">'
            f'<span>Morgonrundan</span><b>{marks}</b>'
            f'<span style="font-family:{FONT_MONO};letter-spacing:0;">{i + 1}/{len(KEYS)}</span></div>')


def _css() -> None:
    """Pillrets och fotens CSS, en gång per körning (dold stilbehållare pcss_morning)."""
    from streamlit.errors import StreamlitDuplicateElementKey

    from ui.shell import page_style
    try:
        page_style(CSS, "morning")
    except StreamlitDuplicateElementKey:
        pass


def pill(step: str) -> bool:
    """'1/5 ›' i huvudraden — nästa steg med ett tryck. Ritas bara där rundan visas (shows)."""
    import streamlit as st
    if not shows(step):
        return False
    _css()
    nxt = next_step(step)
    return st.button(f"{index(step) + 1}/{len(KEYS)} ›", key=f"morning_top_{step}", type="tertiary",
                     help=f"Morgonrundan — nästa: {LABEL[nxt]}", on_click=advance, args=(step,))


def footer(step: str) -> bool:
    """Foten: prickarna och 'Nästa: … →' över hela bredden. Ritas bara där rundan visas (shows)."""
    import streamlit as st
    if not shows(step):
        return False
    _css()
    with st.container(key=f"morning_foot_{step}"):
        st.markdown(dots_html(step), unsafe_allow_html=True)
        return st.button(next_label(step), key=f"morning_next_{step}", type="primary", width="stretch",
                         on_click=advance, args=(step,))
