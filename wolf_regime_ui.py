"""
wolf_regime_ui.py — Swing-marknadsregim (Streamlit-port av RegimeTab).

Läser wolf_regime.json (genererad av wolf_data.py) via Gist/lokal fallback och
visar trafikljus + regelverk, index vs MA200, marknadsbredd (dör före index),
antal kvalande bolag och historiken (en punkt per handelsdag, högst 60 ≈ tre månader):
överblick, stapeldiagram (höjd = bredd, färg = regim, streckat vid 55 % och 45 %) och perioderna.
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
    if st.session_state.get(_CACHE) is None:                 # None = gick inte att läsa: försök igen nästa gång
        st.session_state[_CACHE] = _load_wolf(_FILE) if _load_wolf else None
    return st.session_state[_CACHE]


def render_wolf_regime_page() -> None:
    try:
        c_head, c_btn = st.columns([4, 1])
        with c_head:
            st.markdown(f"<h1 style='color:{TEXT};margin:0;'>Regim</h1>",
                        unsafe_allow_html=True)
        with c_btn:
            if st.button("📡 Uppdatera", key="wolf_reg_refresh"):
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
        _ticker_check(d)
    except Exception as e:
        st.error(f"Regim-fliken kunde inte renderas: {e}")


def _ticker_check(regime_data: dict) -> None:
    """Slå en ticker mot momentum-reglerna: köp, bevaka, håll eller sälj.

    Samma dom som Copiloten ger — swing_verdict.verdict är EN funktion som
    båda flikarna anropar, så de kan inte säga olika saker. Underlaget är
    regimljuset ovan, screenerrankingen och dina positioner i Swing-fliken.
    """
    import storage
    import swing_verdict
    import wolf_screener_ui

    st.markdown(f"<h3 style='color:{TEXT};margin:18px 0 4px;'>Kolla en "
                f"ticker</h3>", unsafe_allow_html=True)
    ticker = st.text_input("Ticker", key="swreg_ticker",
                           placeholder="t.ex. ANOT, ERIC B eller ERIC-B.ST",
                           label_visibility="collapsed").strip()
    if not ticker:
        st.caption("Skriv en ticker så prövas den mot regimljuset, "
                   "screenerrankingen och säljreglerna för dina innehav.")
        return

    screener = wolf_screener_ui._get_data() or {}
    import swing as _swing
    swing_data = _swing.rules_data()        # positionerna ur registret (Holdings)
    v = swing_verdict.verdict(ticker, screener, regime_data, swing_data)

    colors = {swing_verdict.BUY: GREEN, swing_verdict.PARTIAL: "#d4943a",
              swing_verdict.HOLD: GREEN, swing_verdict.WATCH: "#d4943a",
              swing_verdict.SELL: RED, swing_verdict.ABSTAIN: RED,
              swing_verdict.UNKNOWN: DIM}
    c = colors.get(v["verdict"], DIM)
    tag = " · INNEHAV" if v["held"] else ""
    reasons = "".join(f"<li style='margin:2px 0;'>• {r}</li>"
                      for r in v["reasons"])
    st.markdown(
        f"<div style='background:{c}14;border:1px solid {c};"
        f"border-radius:12px;padding:14px;margin-top:6px;'>"
        f"<div style='font-size:1.15rem;font-weight:800;color:{c};'>"
        f"{ticker.upper()}{tag}: {v['verdict']}</div>"
        f"<ul style='list-style:none;padding:6px 0 0;margin:0;color:{TEXT};"
        f"font-size:0.85rem;'>{reasons}</ul></div>",
        unsafe_allow_html=True)


# ── Regim-historiken ─────────────────────────────────────────────────────────
# Bara visning: datan (wolf_regime.json → history, äldst först, en punkt per handelsdag)
# och trösklarna (55 % / 45 %, samma som bredd-rutan ovan) är oförändrade.

REGIMES = ("GRÖN", "GUL", "RÖD")
UNKNOWN = "OKÄND"
REGIME_COLOR = {"GRÖN": GREEN, "GUL": AMBER, "RÖD": RED, UNKNOWN: GREY}
HEALTHY, DYING = 55, 45                  # bredd i %, samma gränser som texten i bredd-rutan
CHART_KEY = "wolf_regime_history_chart"
NBSP = "\u00a0"
_SV_MONTHS = ("jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug", "sep", "okt", "nov", "dec")
_CHIP_BD, _CHIP_BG = "#2a2f37", "#11161d"
_PLOT_BG, _GRID = "#0D1117", "rgba(138,133,120,0.08)"
HISTORY_CAPTION = (f"Stapelns höjd = bredd (andel över MA200) · färg = regim den dagen · "
                   f"streckat: {HEALTHY}{NBSP}% friskt, {DYING}{NBSP}% bredden dör. "
                   f"GRÖN med fallande staplar är hur toppar ser ut.")


def _parse_date(v):
    """'YYYY-MM-DD' (även med klockslag efter) eller ett date-objekt → date, annars None."""
    import datetime as _dt
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, _dt.date):
        return v
    if not isinstance(v, str):
        return None
    try:
        return _dt.date.fromisoformat(v.strip()[:10])
    except ValueError:
        return None


def _sv_date(iso, year: bool = False) -> str:
    """'2026-10-08' → '8 okt' (eller '8 okt 2026')."""
    d = _parse_date(iso)
    if d is None:
        return str(iso or "–")
    s = f"{d.day} {_SV_MONTHS[d.month - 1]}"
    return f"{s} {d.year}" if year else s


def _sv_pct(x, dec: int = 0) -> str:
    """0.523 → '52 %' / '52,3 %' (decimalkomma, hårt mellanslag före %)."""
    if x is None:
        return "saknas"
    return f"{x * 100:.{dec}f}".replace(".", ",") + f"{NBSP}%"


def _sv_int(n) -> str:
    return "saknas" if n is None else f"{n:,}".replace(",", NBSP)


def _days(n: int) -> str:
    return f"{n} handelsdag" if n == 1 else f"{n} handelsdagar"


def clean_history(hist) -> list:
    """Historiken städad: en rad per datum (senaste posten vinner, som i wolf_data), äldst först.

    Rader utan giltigt datum hoppas över. Okänd regim → OKÄND. Bredd utanför 0..1 eller
    saknad → None, kvalande bolag saknas/negativt → None. Tål en tom lista och fel typ.
    """
    by_date = {}
    for h in hist if isinstance(hist, (list, tuple)) else []:
        if not isinstance(h, dict):
            continue
        d = _parse_date(h.get("date"))
        if d is None:
            continue
        reg = h.get("regime")
        reg = reg.strip().upper() if isinstance(reg, str) else ""
        b = _num(h.get("breadth"))
        q = _num(h.get("qualifying"))
        by_date[d.isoformat()] = {
            "date": d.isoformat(),
            "regime": reg if reg in REGIMES else UNKNOWN,
            "breadth": b if b is not None and 0.0 <= b <= 1.0 else None,
            "qualifying": int(round(q)) if q is not None and q >= 0 else None,
        }
    return [by_date[k] for k in sorted(by_date)]


def regime_periods(hist) -> list:
    """Sammanhängande perioder med samma regim, äldst först: [{regime, start, end, days}]."""
    out = []
    for r in clean_history(hist):
        if out and out[-1]["regime"] == r["regime"]:
            out[-1]["end"] = r["date"]
            out[-1]["days"] += 1
        else:
            out.append({"regime": r["regime"], "start": r["date"], "end": r["date"], "days": 1})
    return out


def history_summary(hist) -> dict:
    """Överblicken: regim nu och sedan när, dagar per regim i fönstret, bredd och kvalande först → sist."""
    rows = clean_history(hist)
    periods = regime_periods(rows)
    counts = {k: 0 for k in REGIMES}
    for r in rows:
        counts[r["regime"]] = counts.get(r["regime"], 0) + 1
    br = [r["breadth"] for r in rows if r["breadth"] is not None]
    qu = [r["qualifying"] for r in rows if r["qualifying"] is not None]
    cur = periods[-1] if periods else {}
    return {
        "n": len(rows),
        "first": rows[0]["date"] if rows else None,
        "last": rows[-1]["date"] if rows else None,
        "regime": cur.get("regime"),
        "since": cur.get("start"),
        "days": cur.get("days", 0),
        "since_window_start": len(periods) == 1,      # perioden kan ha börjat före fönstret
        "counts": counts,
        "breadth_first": br[0] if br else None,
        "breadth_last": br[-1] if br else None,
        "breadth_points": len(br),
        "qualifying_first": qu[0] if qu else None,
        "qualifying_last": qu[-1] if qu else None,
        "qualifying_points": len(qu),
    }


def _count_items(s: dict) -> list:
    """[(regim, dagar)] — alltid GRÖN, GUL, RÖD; OKÄND bara när den förekommer."""
    return [(k, v) for k, v in s["counts"].items() if k in REGIMES or v]


def _first_last(first, last, points: int, fmt) -> str:
    return f"{fmt(first)} → {fmt(last)}" if points > 1 else fmt(last)


def overview_parts(s: dict) -> list:
    """Överblicken som klartext, en bit per chip (tom lista utan historik)."""
    if not s or not s.get("n"):
        return []
    if s["since_window_start"]:
        now = f"Nu: {s['regime']} sedan minst {_sv_date(s['since'])} ({s['days']}+ handelsdagar)"
    else:
        now = f"Nu: {s['regime']} sedan {_sv_date(s['since'])} ({_days(s['days'])})"
    window = f"Senaste {s['n']} dagar" if s["n"] > 1 else "Senaste dagen"
    parts = [now, f"{window}: " + " · ".join(f"{k} {v}" for k, v in _count_items(s))]
    if s["breadth_points"]:
        parts.append("Bredd " + _first_last(s["breadth_first"], s["breadth_last"], s["breadth_points"], _sv_pct))
    if s["qualifying_points"]:
        parts.append("Kvalande " + _first_last(s["qualifying_first"], s["qualifying_last"],
                                               s["qualifying_points"], _sv_int))
    return parts


def period_parts(periods: list) -> list:
    """['GRÖN 12 sep–2 okt (15 d)', 'GUL 3 okt–nu (23 d)'] — sista perioden slutar 'nu'."""
    out = []
    for i, p in enumerate(periods):
        last = i == len(periods) - 1
        start = _sv_date(p["start"])
        if last:
            span = f"{start}–nu"
        elif p["start"] == p["end"]:
            span = start
        elif p["start"][:7] == p["end"][:7]:                 # samma månad: '2–3 sep'
            span = f"{_parse_date(p['start']).day}–{_sv_date(p['end'])}"
        else:
            span = f"{start}–{_sv_date(p['end'])}"
        out.append(f"{p['regime']} {span} ({p['days']}{NBSP}d)")
    return out


def _swatch(color: str) -> str:
    return (f"<span style='display:inline-block;width:9px;height:9px;border-radius:2px;"
            f"background:{color};margin:0 5px 0 0;vertical-align:-1px;'></span>")


def _chip(inner: str, border: str = _CHIP_BD, bg: str = _CHIP_BG) -> str:
    return (f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
            f"border:1px solid {border};background:{bg};color:{TEXT};font-size:0.82rem;"
            f"line-height:1.35;'>{inner}</span>")


def _overview_html(s: dict) -> str:
    parts = overview_parts(s)
    if not parts:
        return ""
    col = REGIME_COLOR.get(s["regime"], GREY)
    now = parts[0].replace(f" {s['regime']} ", f" <b>{s['regime']}</b> ", 1)
    chips = [_chip(f"{_swatch(col)}{now}", border=col, bg=f"{col}22")]
    window = parts[1].split(": ", 1)[0]
    counts = " · ".join(f"{_swatch(REGIME_COLOR[k])}{k} {v}" for k, v in _count_items(s))
    chips.append(_chip(f"{window}: {counts}"))
    chips += [_chip(p) for p in parts[2:]]
    return (f"<div style='display:flex;flex-wrap:wrap;gap:6px;margin:0 0 4px;'>"
            f"{''.join(chips)}</div>")


def _periods_html(periods: list) -> str:
    parts = period_parts(periods)
    if not parts:
        return ""
    items = [f"<span style='white-space:nowrap;'>{_swatch(REGIME_COLOR.get(p['regime'], GREY))}{t}</span>"
             for p, t in zip(periods, parts)]
    return (f"<div style='color:{TEXT};font-size:0.8rem;line-height:1.7;margin:2px 0 4px;'>"
            f"<span style='color:{DIM};'>Perioder:</span> {' → '.join(items)}</div>")


def _hover(r: dict) -> str:
    return (f"{_sv_date(r['date'], year=True)}<br>Regim: {r['regime']}<br>"
            f"Bredd: {_sv_pct(r['breadth'], 1)}<br>Kvalande bolag: {_sv_int(r['qualifying'])}")


def _tick_indices(n: int, max_ticks: int = 5) -> list:
    """Jämnt fördelade datumetiketter, förankrade i senaste dagen (läsbart även på 390 px)."""
    if n <= 0:
        return []
    step = max(1, -(-n // max_ticks))
    return list(range(n - 1, -1, -step))[::-1]


def history_figure(hist, go=None):
    """Stapeldiagram: en stapel per handelsdag, höjd = bredd i %, färg = regimen den dagen."""
    if go is None:
        import plotly.graph_objects as go
    rows = clean_history(hist)
    x = [r["date"] for r in rows]
    y = [None if r["breadth"] is None else round(r["breadth"] * 100, 1) for r in rows]
    fig = go.Figure(go.Bar(
        x=x, y=y, name="Bredd",
        marker=dict(color=[REGIME_COLOR.get(r["regime"], GREY) for r in rows], line=dict(width=0)),
        hovertext=[_hover(r) for r in rows],
        hoverinfo="text",                                    # bara svensk text, ingen ISO-etikett på x-axeln
    ))
    for level, label, pos in ((HEALTHY, f"friskt {HEALTHY}{NBSP}%", "top left"),
                              (DYING, f"dör {DYING}{NBSP}%", "bottom left")):
        fig.add_hline(y=level, line_dash="dash", line_width=1, line_color="rgba(232,228,220,0.55)",
                      annotation_text=label, annotation_position=pos,
                      annotation_font=dict(size=10, color=TEXT),
                      annotation_bgcolor="rgba(13,17,23,0.75)")
    top = max([v for v in y if v is not None], default=0)
    top = int(min(100, max(70, -(-(top + 8) // 10) * 10)))  # nollbaslinje, båda linjerna syns
    ticks = _tick_indices(len(x))
    fig.update_xaxes(type="category", tickmode="array", tickvals=[x[i] for i in ticks],
                     ticktext=[_sv_date(x[i]) for i in ticks], tickangle=0, showgrid=False,
                     fixedrange=True, automargin=False, title="")
    fig.update_yaxes(range=[0, top], dtick=25, ticksuffix=f"{NBSP}%", gridcolor=_GRID,
                     zeroline=False, fixedrange=True, automargin=False, ticklabelstandoff=4, title="")
    fig.update_layout(                                       # fasta marginaler: staplarna får bredden på 390 px
        height=220, margin=dict(l=44, r=16, t=8, b=24, pad=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=_PLOT_BG,
        font=dict(color=DIM, size=10), showlegend=False,
        bargap=0.25, barcornerradius=2, hovermode="x", dragmode=False,
        hoverlabel=dict(bgcolor="#161b22", bordercolor=_CHIP_BD, font=dict(color=TEXT, size=12)),
    )
    return fig


def _plotly_go():
    try:
        import plotly.graph_objects as go
        return go
    except Exception:
        return None


def _box_grid_html(rows: list) -> str:
    """Reserv utan plotly: samma kodning som diagrammet (höjd = bredd, färg = regim) i HTML."""
    cells = ""
    for r in rows:
        col = REGIME_COLOR.get(r["regime"], GREY)
        fill = 0 if r["breadth"] is None else r["breadth"] * 100
        tip = (f"{_sv_date(r['date'], year=True)}: {r['regime']}, bredd {_sv_pct(r['breadth'], 1)}, "
               f"{_sv_int(r['qualifying'])} kvalande")
        cells += (f"<div title=\"{tip}\" style='flex:1 1 0;min-width:2px;height:100%;background:#1a1f25;"
                  f"border-radius:2px;display:flex;flex-direction:column;justify-content:flex-end;'>"
                  f"<div style='height:{fill:.0f}%;background:{col};border-radius:2px 2px 0 0;'></div></div>")
    return f"<div style='display:flex;gap:2px;height:64px;margin:6px 0;'>{cells}</div>"


def _history(hist: list) -> None:
    rows = clean_history(hist)
    if not rows:
        return
    st.markdown(f"<div style='color:{DIM};font-size:0.7rem;text-transform:uppercase;"
                f"margin:14px 0 6px;'>Regim-historik — senaste {len(rows)} handelsdagarna "
                f"(regim · bredd)</div>",
                unsafe_allow_html=True)
    st.markdown(_overview_html(history_summary(rows)), unsafe_allow_html=True)
    go = _plotly_go()
    if go is not None:
        st.plotly_chart(history_figure(rows, go=go), width="stretch", key=CHART_KEY,
                        config={"displayModeBar": False})
    else:
        st.markdown(_box_grid_html(rows), unsafe_allow_html=True)
    st.markdown(_periods_html(regime_periods(rows)), unsafe_allow_html=True)
    st.caption(HISTORY_CAPTION)
