"""
valhall/ui.py — ⚡ VALHALL: cykelkartan över hela marknaden, korten, papperskontona och backtesterna.

Sidan räknar ingenting själv — den läser de schemalagda körningarnas resultat ur Gisten:
  valhall_board.json   kartan, swing, Wolf, köp, Long Screener och papperskontonas sammanställning (vardagar)
  valhall_fas3.json    fas 3a — cykelstatusen och swing (Viking Nine, BERSERK)
  valhall_fas4b.json   fas 4b — cykeln och de långsiktiga köpen (segment och kvalitetsaktier)
  valhall_fas4c.json   fas 4c — Long Screener punkt för punkt
  valhall_nordic_probe.json  Norden-proben — är hela Norden en bättre mätpunkt än OMXS30?

Designen följer 🪓 BERSERK Regime: en ruta per segment med färgad kant efter status, chips för val och
dämpade detaljrader. Korten är rena funktioner (HTML-strängar) så att de går att testa utan Streamlit.
"""

from __future__ import annotations

from html import escape
from typing import Optional

import pandas as pd

from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, PURPLE, RED, TEXT, regime_color

BLOBS = {"board": "valhall_board.json", "fas3": "valhall_fas3.json", "fas4b": "valhall_fas4b.json",
         "fas4c": "valhall_fas4c.json", "nordic_probe": "valhall_nordic_probe.json"}
_STATE = "valhall_data"
STRATEGY_LABEL = {"viking": "Viking Nine", "insider": "Insider Norden", "wolf": "Momentum Swing", "wolf_arc": "Wolf",
                  "berserk": "BERSERK",
                  "quality": "Quality Contrarian", "rule": "Rick Rule", "durrett": "Durrett", "tiggre": "Tiggre",
                  "long": "Long Screener", "long_ingen": "Long Screener (utan cykel)"}


def load(loader=None) -> dict:
    """Alla Valhall-resultat ur Gisten (None för det som saknas)."""
    if loader is None:
        from gist_storage import load_blob as loader
    out = {}
    for k, name in BLOBS.items():
        try:
            out[k] = loader(name, None)
        except Exception:
            out[k] = None
    out["loaded"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return out


# ── Byggstenar ──────────────────────────────────────────────────────────────
def _status(st: Optional[dict]) -> str:
    return (st or {}).get("status") or "DATA SAKNAS"


def status_chip(label: str, st: Optional[dict]) -> str:
    s = _status(st)
    col = regime_color(s) if s in ("GRÖN", "GUL", "RÖD") else GREY
    extra = " · ".join(x for x in ((st or {}).get("phase"), (st or {}).get("cykel")) if x and x != s)
    start = turn_text((st or {}).get("turn")) or (" ⚡ CYKELSTART" if (st or {}).get("cycle_start") else "")
    return (f"<span style='border:1px solid {col};border-radius:3px;padding:0 5px;color:{col};font-size:0.68rem;"
            f"font-weight:700;white-space:nowrap;'>{escape(label)} {s}</span>"
            + (f"<span style='color:{DIM};font-size:0.66rem;'> {escape(extra)}{start}</span>" if extra or start else ""))


def turn_text(turn: Optional[dict]) -> str:
    """' ⚡ VÄNDER dag 7 ✓' ur cycle.turn_state — tomt när ingen vändning pågår. ✓ = bekräftad."""
    t = turn or {}
    if not t.get("kind"):
        return ""
    return f" ⚡ {t['kind']} dag {t.get('days') or '?'}" + (" ✓" if t.get("confirmed") else "")


def complex_card(c: dict, swing_c: Optional[dict] = None) -> str:
    """Komplexrutan: komplexdrivarens status, vändningen, hur många teman som vänder/är starka, och Viking Nine
    per komplex (profilen) med drivargrinden."""
    st = c.get("state") or {}
    sw = swing_c or {}
    drv = sw.get("driver") or {}
    drv_txt = f" · drivare {escape(str(drv.get('detail')))}" if drv.get("detail") else ""
    col = regime_color(_status(st)) if _status(st) in ("GRÖN", "GUL", "RÖD") else GREY
    chips = ""
    for t in c.get("themes") or []:
        ts = _status(t.get("state"))
        tcol = regime_color(ts) if ts in ("GRÖN", "GUL", "RÖD") else GREY
        chips += (f"<span style='border:1px solid {tcol};border-radius:3px;padding:0 4px;margin:1px 2px 1px 0;"
                  f"font-size:0.62rem;color:{TEXT};display:inline-block;'>{escape(t.get('label') or t.get('theme') or '')}"
                  f"{escape(turn_text((t.get('state') or {}).get('turn')))}</span>")
    return (f"<div style='border:1px solid {col};border-left:4px solid {col};border-radius:4px;padding:6px 8px;"
            f"min-width:0;'><div style='display:flex;justify-content:space-between;gap:6px;flex-wrap:wrap;'>"
            f"<b style='color:{TEXT};font-size:0.84rem;'>{escape(c.get('label') or c.get('key') or '')}</b>"
            f"<span>{status_chip('', st)}</span></div>"
            f"<div style='color:{DIM};font-size:0.62rem;'>{escape(c.get('driver') or 'ingen drivare')} · "
            f"{c.get('themes_turning') or 0}/{c.get('n_themes') or 0} teman vänder eller är starka</div>"
            f"<div style='margin-top:3px;'>{chips}</div>"
            + (_line(f"Swing ({escape(sw.get('strategy') or 'Viking Nine · komplex')})",
                     pick_chips(sw.get("picks") or []) + f"<span style='color:{DIM};font-size:0.62rem;'>"
                     f" {sw.get('universe') or 0} bolag{drv_txt}</span>") if sw else "")
            + "</div>")


def turn_notice(turns: list) -> str:
    """'VÄNDNING — bekräftad: … · på väg: …' ur board["turns"]; tom sträng utan vändningar."""
    ts = [t for t in turns or [] if isinstance(t, dict) and t.get("kind")]
    if not ts:
        return ""

    def one(t):
        where = {"komplex": "komplex", "tema": "tema"}.get(t.get("level"), t.get("where") or "segment")
        return f"{t.get('label') or t.get('key')} ({where}, {t.get('kind')} dag {t.get('days') or '?'})"
    conf = [one(t) for t in ts if t.get("confirmed")]
    soon = [one(t) for t in ts if not t.get("confirmed")]
    parts = [p for p in (("bekräftad: " + ", ".join(conf)) if conf else "", ("på väg: " + ", ".join(soon)) if soon else "") if p]
    return "⚡ VÄNDNING — " + " · ".join(parts)


def pick_chips(picks: list, color: str = CYAN, blocked: bool = False) -> str:
    if not picks:
        return f"<span style='color:{DIM};font-size:0.66rem;'>—</span>"
    col = GREY if blocked else color
    out = []
    for p in picks:
        tag = p.get("tag") or p.get("category") or p.get("setup") or p.get("signal") or p.get("verdict") or ""
        if p.get("bagger"):
            tag = "15-BAGGER"
        txt = escape(str(p.get("ticker") or "")) + (f" <span style='opacity:0.7;'>{escape(str(tag))}</span>" if tag else "")
        out.append(f"<span style='border:1px solid {col};border-radius:3px;padding:0 4px;color:{col};"
                   f"font-size:0.66rem;white-space:nowrap;{'text-decoration:line-through;' if blocked else ''}'>"
                   f"{txt}</span>")
    return " ".join(out)


def _line(label: str, body: str) -> str:
    return (f"<div style='font-size:0.7rem;line-height:1.7;'><span style='color:{DIM};'>{label}</span> "
            f"{body}</div>")


def segment_card(row: dict, swing: dict, buys: dict, long: dict, niche: dict, wolf: Optional[dict] = None) -> str:
    """Sektorrutan: global (SPDR → UCITS) och nordisk status, swing med segmentets strategi, Wolf (Arc-Wolf —
    ENTRY när köpregeln är uppfylld) och Momentum (topp 20 med setup A), kvalitetsköp (grind: ej RÖD),
    Long Screener och nischsignaler."""
    nordic = row.get("nordic") or {}
    col = regime_color(_status(nordic)) if _status(nordic) in ("GRÖN", "GUL", "RÖD") else GREY
    sw = swing or {}
    gate = (buys or {}).get("gate") or {}
    blocked = gate.get("open") is False
    q = pick_chips((buys or {}).get("picks") or [], GOLD, blocked)
    if blocked:
        q += f" <span style='color:{RED};font-size:0.64rem;'>SPÄRRAD — sektorn är RÖD</span>"
    ucits = ", ".join(row.get("ucits") or []) or "—"
    nb = len((niche or {}).get("berserk") or [])
    ni = len((niche or {}).get("insider") or [])
    return (f"<div style='border:1px solid {col};border-left:4px solid {col};border-radius:4px;padding:6px 8px;"
            f"min-width:0;'><div style='display:flex;justify-content:space-between;gap:6px;flex-wrap:wrap;'>"
            f"<b style='color:{TEXT};font-size:0.84rem;'>{escape(row.get('label') or row.get('key') or '')}</b>"
            f"<span>{status_chip('Norden', nordic)}</span></div>"
            f"<div style='margin:2px 0;'>{status_chip('Globalt', row.get('global'))}"
            f"<span style='color:{DIM};font-size:0.64rem;'> · {escape(row.get('spdr') or '')} → {escape(ucits)}</span></div>"
            + _line(f"Swing ({escape(sw.get('strategy') or '—')})", pick_chips(sw.get("picks") or []))
            + _line("Wolf", pick_chips((wolf or {}).get("arc") or [], PURPLE))
            + _line("Momentum", pick_chips((wolf or {}).get("momentum") or [], TEXT))
            + _line("Kvalitet", q)
            + _line("Long Screener", pick_chips((long or {}).get("picks") or [], GREEN))
            + (_line("Nisch", f"<span style='color:{TEXT};'>BERSERK {nb} · Insider {ni}</span>") if nb or ni else "")
            + "</div>")


def theme_card(t: dict, swing_t: dict, buys_t: dict) -> str:
    """Råvarurutan: temats status, BERSERK-swing, Rick Rule-köp och hävstång (Durrett/Tiggre)."""
    st = t.get("state") or {}
    col = regime_color(_status(st)) if _status(st) in ("GRÖN", "GUL", "RÖD") else GREY
    lv = (buys_t or {}).get("leverage") or {}
    return (f"<div style='border:1px solid {col}88;border-left:3px solid {col};border-radius:4px;padding:5px 7px;"
            f"min-width:0;'><div style='display:flex;justify-content:space-between;gap:6px;flex-wrap:wrap;'>"
            f"<b style='color:{TEXT};font-size:0.78rem;'>{escape(t.get('label') or t.get('theme') or '')}</b>"
            f"<span>{status_chip('', st)}</span></div>"
            f"<div style='color:{DIM};font-size:0.62rem;'>{escape(t.get('driver') or 'ingen drivare')}</div>"
            + _line("Swing (BERSERK)", pick_chips((swing_t or {}).get("picks") or []))
            + _line("Rick Rule", pick_chips(((buys_t or {}).get("quality") or {}).get("picks") or [], GOLD))
            + _line(f"Hävstång ({escape(lv.get('strategy') or '—')})", pick_chips(lv.get("picks") or [], AMBER))
            + "</div>")


def paper_table(rows: list, keys: tuple) -> pd.DataFrame:
    """Papperskontonas sammanställning som tabell (svenska rubriker)."""
    names = {"strategy": "Strategi", "segment": "Segment", "status_in": "Status vid köp", "cykel_in": "Cykel vid köp",
             "n": "Affärer", "open": "Öppna", "closed": "Stängda", "avg_ret": "Snitt %", "win": "Vinst %",
             "avg_r": "Snitt R", "sum_r": "Summa R", "avg_excess": "Mot riktmärke %", "avg_red_ret": "Sälj på RÖD %",
             "red": "Blev RÖD"}
    cols = list(keys) + ["n", "open", "closed", "avg_ret", "win", "avg_r", "avg_excess", "avg_red_ret"]
    out = []
    for r in rows or []:
        d = {names.get(c, c): r.get(c) for c in cols}
        if "Strategi" in d:
            d["Strategi"] = STRATEGY_LABEL.get(d["Strategi"], d["Strategi"])
        out.append(d)
    return pd.DataFrame(out)


def portfolio_table(rows: list, key: str) -> pd.DataFrame:
    """Backtestens portföljrader (CAGR, DD m.m.) som tabell."""
    keep = [r for r in rows or [] if r.get("cagr") is not None]
    return pd.DataFrame([{"Variant": r.get(key), "Period": r.get("period"), "CAGR %": r.get("cagr"),
                          "Max DD %": r.get("dd"), "Snitt innehav": r.get("avg_n"),
                          "Norden %": r.get("bench_cagr")} for r in keep])


def probe_tables(payload: Optional[dict]) -> dict:
    """Norden-probens resultat som tabeller (svenska rubriker): samvariation, SMA200-korsningar, sektorernas
    tidighet och grindvarianterna. Tomma tabeller hoppas över."""
    p = payload or {}
    out = {}
    rows = [{"Par": key, "Fönster": w.get("window"), "Dagar": w.get("days"), "Korr dag": w.get("corr_daily"),
             "Korr vecka": w.get("corr_weekly"), "Samma sida SMA200 %": w.get("same_sma200_pct"),
             "Samma 20-dagarsriktning %": w.get("same_direction_pct")}
            for key, ws in (p.get("comovement") or {}).items() for w in ws or []]
    if rows:
        out["Samvariation"] = pd.DataFrame(rows)
    rows = [{"Par": key, "Uppkorsningar": c.get("n"), "Matchade": c.get("matched"), "Före": c.get("earlier"),
             "Efter": c.get("later"), "Samma dag": c.get("same_day"), "Median dagar": c.get("median_days"),
             "Snitt dagar": c.get("mean_days")}
            for key, c in (p.get("crossing_lead") or {}).items() if c]
    if rows:
        out["SMA200-korsningar"] = pd.DataFrame(rows)
    rows = []
    for market, segs in (p.get("earliness") or {}).items():
        for seg, e in (segs or {}).items():
            s = (e or {}).get("summary") or {}
            if not s.get("turns"):
                continue
            rows.append({"Marknad": market, "Segment": seg, "Gröna vändningar": s.get("turns"),
                         "Nära grinden": s.get("near_gate"), "Före grinden": s.get("earlier"),
                         "Median dagar": s.get("median_days"), "63 d %": s.get("avg_ret_63"),
                         "Träff 63 d %": s.get("hit_63"), "126 d %": s.get("avg_ret_126"),
                         "Träff 126 d %": s.get("hit_126")})
    if rows:
        out["Sektorernas tidighet"] = pd.DataFrame(rows)
    rows = [{"Motor": engine, "Grind": r.get("variant"), "Affärer": r.get("trades"), "Träff %": r.get("win_rate"),
             "Förväntan R": r.get("expectancy"), "PF": r.get("profit_factor"), "Max DD R": r.get("max_drawdown_r"),
             "Summa R": r.get("total_r"), "Källa / fel": r.get("error") or r.get("source")}
            for engine, rs in (p.get("gates") or {}).items() for r in rs or []]
    if rows:
        out["Grindvarianterna"] = pd.DataFrame(rows)
    return out


def render_nordic_probe(payload: Optional[dict]) -> None:
    """NORDEN-PROBEN i Backtester: klartextraderna, ledningen per par och tabellerna."""
    import streamlit as st
    if not payload:
        return
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:10px;'>NORDEN-PROBEN "
                f"— ÄR HELA NORDEN EN BÄTTRE MÄTPUNKT ÄN OMXS30? "
                f"<span style='color:{DIM};letter-spacing:0;'>körd {escape(str(payload.get('generated') or '—')[:16])}"
                f" · från {payload.get('start_year') or '—'}</span></div>", unsafe_allow_html=True)
    lines = [escape(str(x)) for x in (payload.get("summary") or [])]
    if lines:
        st.markdown(f"<div style='border-left:3px solid {GOLD};padding:2px 8px;margin:4px 0;color:{DIM};"
                    f"font-size:0.72rem;line-height:1.6;'>" + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
    for title, table in probe_tables(payload).items():
        st.markdown(f"<div style='color:{TEXT};font-size:0.74rem;font-weight:700;margin-top:6px;'>{escape(title)}</div>",
                    unsafe_allow_html=True)
        st.dataframe(table, hide_index=True, use_container_width=True)
    src = payload.get("sources") or {}
    if src:
        st.markdown(f"<div style='color:{DIM};font-size:0.66rem;line-height:1.6;margin-top:4px;'>" + "<br>".join(
            f"<b>{escape(str(k))}</b>: {escape(str(v.get('source')))} · {escape(str(v.get('breadth_source')))} · "
            f"{v.get('bars')} dagar {escape(str(v.get('from')))}–{escape(str(v.get('to')))}" for k, v in src.items())
            + "</div>", unsafe_allow_html=True)
    notes = payload.get("notes") or []
    if notes:
        st.markdown(f"<div style='color:{DIM};font-size:0.66rem;line-height:1.6;'>" + " · ".join(escape(str(n)) for n in notes)
                    + "</div>", unsafe_allow_html=True)


DECISIONS = (
    ("Swing per segment (fas 3a)", "Cykelstatusen spärrar inte swing — gröna grinden hjälpte 2008–2020 och skadade "
     "2021–. Viking Nine i Teknik och Finans, BERSERK i Energi, Material och råvarorna, ingen swing i "
     "Kraftförsörjning, övriga prövas i papperet."),
    ("Kvalitet: ej RÖD sektor (fas 4b)", "Kvalitetsköp bara när sektorn inte är RÖD: 17,9 % CAGR / −26 % DD mot "
     "13,2 % / −34 % för alla, i båda perioderna. Sälj INTE på RÖD — det kostade ≈ 20 procentenheter på 24 mån."),
    ("Råvaror: contrarian (fas 4b)", "Cykel TIDIG gav +46 % på 36 mån i båda perioderna — Rick Rule-tanken håller "
     "för råvarorna (men inte för nordiska sektorer, där GRÖN slog RÖD)."),
    ("Long Screener: cykel \"gron\" (fas 4c)", "Köp bara i GRÖN sektor, sälj när hemmamarknaden är RÖD: 12,3 % CAGR / "
     "−18 % DD mot 8,7 % / −31 % för första versionen. Utan cykel följs parallellt i papperet."),
)


# ── Sidan ───────────────────────────────────────────────────────────────────
def render_valhall_page(sub: str = "Kartan") -> None:
    import streamlit as st
    from ui.components import page_header
    page_header("⚡ Valhall", "Cykelkartan över hela marknaden — segment, råvaror, val per ruta och vad som "
                "fungerar enligt backtesterna och papperskontona.")
    c1, c2 = st.columns([1, 3])
    if c1.button("📡 Hämta senaste körningen", key="valhall_refresh") or _STATE not in st.session_state:
        with st.spinner("Läser Valhalls körningar ur Gisten …"):
            st.session_state[_STATE] = load()
    data = st.session_state[_STATE]
    board = data.get("board") or {}
    c2.markdown(f"<div style='color:{DIM};font-size:0.78rem;padding-top:8px;'>Körd {escape(str(board.get('generated') or '—')[:16])}"
                f" · hämtad {data.get('loaded')}</div>", unsafe_allow_html=True)
    if sub == "Papperskonton":
        render_paper(board.get("paper"))
    elif sub == "Backtester":
        render_backtests(data)
    else:
        render_map(board)


def render_map(board: dict) -> None:
    import streamlit as st
    from ui.components import note
    segs = ((board or {}).get("board") or {}).get("segments") or []
    if not segs:
        note("Kartan har inte körts ännu. Den byggs av arbetsflödet Valhall Scan vardagar efter nordisk stängning "
             "(16:47 UTC) — kör det manuellt i GitHub Actions för att fylla sidan direkt.")
        return
    tn = turn_notice(board.get("turns") or [])
    if tn:
        note(escape(tn) + " — bekräftad = råvara: VÄNDER och över SMA50 med positiv 63-dagarsavkastning; "
             "aktiesegment: GRÖN tio dagar efter RÖD.")
    from valhall import turns as vt
    plan_lines = vt.lines(board.get("turn_plan"))
    if plan_lines:
        st.markdown(f"<div style='color:{GOLD};font-size:0.72rem;letter-spacing:0.1em;margin-top:6px;'>VÄNDNING — KÖP I DAG "
                    f"(köpfönster {(board.get('turn_plan') or {}).get('window_days', vt.WINDOW_DAYS)} dagar efter bekräftelsen)</div>"
                    + "".join(f"<div style='font-size:0.74rem;line-height:1.6;color:{TEXT};'>• {escape(x)}</div>" for x in plan_lines),
                    unsafe_allow_html=True)
    starts = board.get("cycle_starts") or []
    if starts and not tn:
        note("⚡ Cykelstart nu (BAISSE · VÄNDER): " + ", ".join(f"{escape(s['segment'])} ({s['where']})" for s in starts))
    sw, buys, long, niche = (board.get("swing") or {}), (board.get("buys") or {}), (board.get("long") or {}), \
        (board.get("niche") or {})
    wolf = (board.get("wolf") or {}).get("segments") or {}
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:6px;'>SEKTORERNA</div>",
                unsafe_allow_html=True)
    st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px;"
                "margin-top:4px;'>" + "".join(
                    segment_card(r, (sw.get("segments") or {}).get(r["key"]), (buys.get("segments") or {}).get(r["key"]),
                                 (long.get("segments") or {}).get(r["key"]), niche.get(r["key"]), wolf.get(r["key"]))
                    for r in segs) + "</div>", unsafe_allow_html=True)
    cxs = ((board or {}).get("board") or {}).get("complexes") or []
    csw = ((board or {}).get("complex_swing") or {}).get("complexes") or {}
    if cxs:
        st.markdown(f"<div style='color:{GOLD};font-size:0.72rem;letter-spacing:0.1em;margin-top:12px;'>"
                    f"RÅVARUKOMPLEX — komplexdrivaren är en likaviktad korg av temadrivarna</div>", unsafe_allow_html=True)
        st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px;"
                    "margin-top:4px;'>" + "".join(complex_card(c, csw.get(c["key"])) for c in cxs) + "</div>",
                    unsafe_allow_html=True)
    for r in segs:
        if not r.get("themes"):
            continue
        st.markdown(f"<div style='color:{GOLD};font-size:0.72rem;letter-spacing:0.1em;margin-top:12px;'>"
                    f"RÅVAROR · {escape(r.get('label') or '')}</div>", unsafe_allow_html=True)
        st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;"
                    "margin-top:4px;'>" + "".join(
                        theme_card(t, (sw.get("themes") or {}).get(t["theme"]), (buys.get("themes") or {}).get(t["theme"]))
                        for t in r["themes"]) + "</div>", unsafe_allow_html=True)
    note("Status: GRÖN = fasen stark/upptrend och cykeln inte sen · GUL = sen cykel i upptrend · RÖD = baisse/svag "
         "eller cykeltopp. Norden = sektorindexet mot hela Norden, globalt = SPDR-fonden mot S&P 500 (handel i "
         "UCITS-ETF hos Nordnet), råvaror absolut. Överstrukna kvalitetsköp är spärrade av grinden ej RÖD. "
         "Wolf = Arc-Wolfs regimpoäng (ENTRY när strategins köpregel är uppfylld) · Momentum = Swing Screenerns "
         "topp 20 med setup A. Riktiga order läggs alltid manuellt.")


def render_paper(res: Optional[dict]) -> None:
    import streamlit as st
    from ui.components import note
    if not res or not res.get("by_strategy"):
        note("Papperskontona har inga affärer ännu. De stegas fram av Valhall Scan varje vardag — köp fylls på "
             "nästa öppning, så de första affärerna syns dagen efter första körningen.")
        return
    note(f"Sedan {escape(str(res.get('since')))} · {res.get('positions')} öppna · {res.get('orders')} order. "
         "En enhet per signal (ingen kassa), öppna värderade på senaste stängning. Mot riktmärke = segmentets "
         "index/SPDR/drivare under samma tid. Sälj på RÖD = skuggexit första dagen segmentet blev RÖD.")
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;'>PER STRATEGI</div>",
                unsafe_allow_html=True)
    st.dataframe(paper_table(res["by_strategy"], ("strategy",)), hide_index=True, use_container_width=True)
    view = st.radio("Dela upp på", ["Segment", "Status vid köp", "Cykel vid köp"], horizontal=True, key="valhall_paper_by")
    key, rows = {"Segment": ("segment", "by_segment"), "Status vid köp": ("status_in", "by_status"),
                 "Cykel vid köp": ("cykel_in", "by_cykel")}[view]
    st.dataframe(paper_table(res.get(rows) or [], ("strategy", key)), hide_index=True, use_container_width=True)


def render_backtests(data: dict) -> None:
    import streamlit as st
    from ui.components import note
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;'>BESLUTEN</div>",
                unsafe_allow_html=True)
    for title, text in DECISIONS:
        st.markdown(f"<div style='border-left:3px solid {GOLD};padding:2px 8px;margin:4px 0;'>"
                    f"<b style='color:{TEXT};font-size:0.8rem;'>{escape(title)}</b>"
                    f"<div style='color:{DIM};font-size:0.72rem;line-height:1.5;'>{escape(text)}</div></div>",
                    unsafe_allow_html=True)
    f4c = data.get("fas4c") or {}
    if f4c.get("portfolios"):
        st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:10px;'>LONG SCREENER "
                    f"— PORTFÖLJEN PER CYKELVARIANT (fas 4c)</div>", unsafe_allow_html=True)
        st.dataframe(portfolio_table(f4c["portfolios"], "variant"), hide_index=True, use_container_width=True)
    f4b = data.get("fas4b") or {}
    if (f4b.get("stocks") or {}).get("portfolios"):
        st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:10px;'>KVALITETSAKTIER "
                    f"— KVARTALSPORTFÖLJ PER FILTER (fas 4b)</div>", unsafe_allow_html=True)
        st.dataframe(portfolio_table(f4b["stocks"]["portfolios"], "filter"), hide_index=True, use_container_width=True)
    f3 = data.get("fas3") or {}
    for name, label in (("viking", "VIKING NINE"), ("berserk", "BERSERK")):
        rows = ((f3.get(name) or {}).get("portfolio")) or []
        if rows:
            st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:10px;'>{label} "
                        f"— PORTFÖLJ PER CYKELFILTER (fas 3a)</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{"Filter": r.get("filter"), "Affärer": r.get("trades"), "CAGR %": r.get("cagr"),
                                        "Max DD %": r.get("dd"), "Snitt R": r.get("exp")} for r in rows]),
                         hide_index=True, use_container_width=True)
    probe = data.get("nordic_probe") or {}
    render_nordic_probe(probe)
    if not (f4c or f4b or f3 or probe):
        note("Inga backtester sparade i Gisten än — kör Valhall backtest (fas 3), Valhall långsiktigt backtest "
             "(fas 4b), Valhall Long Screener-backtest (fas 4c) och Norden-proben i GitHub Actions.")
    note("Alla backtester är punkt för punkt (ingen framtidsdata) men med dagens noterade bolag (överlevnadsbias) — "
         "jämför varianterna med varandra, inte nivåerna med verkligheten.")
