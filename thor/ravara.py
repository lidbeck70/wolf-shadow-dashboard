"""
thor/ravara.py — Råvara-fliken i Thors analysblock: råvaran bakom aktien, i de motorer som redan finns.

Fyra block, alla för just det här temat (ett uppslag: thor.ticker.theme):
  RÅVARUCYKELN   BERSERK-temats drivare och fas (berserk.live), Valhall-temats status/cykel (dagens karta),
                 rotationsbetyget Triple Signal (rotation: månadens gradering) med kartbokens kapitel
                 (commodity_book) och Odins blindspot (blindspot.theme_board: cykel, hat, nödvändighet, percentil)
  HÄVSTÅNG       resultathävstång och kursbeta mot råvaran (commodity_leverage — samma kort som Råvaruhävstång)
  DEBASEMENT     fiat-miljön (fiat_debasement.environment) och råvaran nominellt, i fast penningvärde och mätt i guld
  KVOTER         guld/silver, guld/råvaran (gold_ratios) och gummibanden (alpha_regime.commodity_ratios)
Temat heter som i BERSERK (koppar, guld, uran …); varje motor har sitt eget ordförråd och mappas här. Ett block
som inte finns för temat säger det; ett block vars motor fallerar blir en notering, aldrig ett fel på sidan.
Inga regler ändras — allt är samma tal som på Odins visdom-sidorna.
"""

from __future__ import annotations

from html import escape
from typing import Callable, Optional

from thor import cards
from thor.ticker import Ticker
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, RED, TEXT

ROTATION_ALIAS = {"naturgas": "gas", "naturgas_eu": "gas", "zink_nickel": "zink"}
BLINDSPOT_ALIAS = {"naturgas_eu": "naturgas", "majs": "agri", "soja": "agri", "socker": "agri", "agri_korg": "agri",
                   "godsel": "agri"}
FIAT_ASSET = {"guld": "gold", "silver": "silver", "koppar": "copper", "olja": "oil"}
EXPOSURE = {"guld": "gold_miner", "silver": "silver", "olja": "oil", "koppar": "copper", "uran": "uranium",
            "litium": "lithium", "naturgas": "gas", "naturgas_eu": "gas"}
PRECIOUS = ("guld", "silver")
BASE_YEAR = 2010                   # fast penningvärde — samma standard som 🐺 Fiat Debasement
CYKEL_COLOR = {"TIDIG": GREEN, "MITTEN": CYAN, "SEN": AMBER, "TOPP": RED}
RATIO_COLOR = {"RUBBER_BAND_STRETCHED": GREEN, "TENSION_BUILDING": AMBER, "NEUTRAL": GREY}


# ── Ordförråden ─────────────────────────────────────────────────────────────
def label(theme: str) -> str:
    try:
        from berserk import themes as th
        lbl = th.label(theme)
        return lbl if lbl and lbl != theme else (theme or "").title()
    except Exception:
        return (theme or "").title()


def rotation_key(theme: str) -> Optional[str]:
    """Rotationsflikens nyckel (guld, uran, gas …) för BERSERK-temat, eller None."""
    import rotation
    key = ROTATION_ALIAS.get(theme, theme)
    return key if key in rotation.COMMODITY_BY_KEY else None


def blindspot_key(theme: str) -> Optional[str]:
    from blindspot.theme_board import _THEMES
    key = BLINDSPOT_ALIAS.get(theme, theme)
    return key if any(t.key == key for t in _THEMES) else None


def leverage_key(theme: str) -> Optional[str]:
    from commodity_leverage.config import COMMODITY_LABELS
    return theme if theme in COMMODITY_LABELS else None


def fiat_asset(theme: str) -> Optional[str]:
    return FIAT_ASSET.get(theme)


def gold_pair(theme: str) -> Optional[dict]:
    from gold_ratios import config as rc
    return next((p for p in rc.PAIRS if p.get("theme") == theme), None)


def ratio_keys(theme: str) -> list:
    from alpha_regime.commodity_ratios import EXPOSURE_TO_RATIO
    return list(EXPOSURE_TO_RATIO.get(EXPOSURE.get(theme, ""), []))


def coverage(theme: str) -> dict:
    """Vilka motorer som känner temat — ren funktion för chipsen och testerna."""
    pair = gold_pair(theme)
    return {"rotation": rotation_key(theme), "blindspot": blindspot_key(theme), "leverage": leverage_key(theme),
            "fiat": fiat_asset(theme), "gold_pair": pair["key"] if pair else None, "ratios": ratio_keys(theme),
            "gold_silver": theme in PRECIOUS}


def rotation_state(theme: str, rotation_data: Optional[dict]) -> Optional[dict]:
    """Månadens Triple Signal-läge (cycle.cycle_state) — None när råvaran inte är betygsatt."""
    import cycle as cy
    key = rotation_key(theme)
    return cy.cycle_state(key, rotation_data or {}) if key else None


def valhall_theme(board_blob: Optional[dict], theme: str) -> Optional[tuple]:
    """(temaraden, swing-temat, köp-temat) ur dagens karta — det theme_card vill ha — eller None."""
    blob = board_blob or {}
    inner = blob.get("board") if isinstance(blob.get("board"), dict) else {}
    for r in (inner or {}).get("segments") or []:
        for t in r.get("themes") or []:
            if isinstance(t, dict) and t.get("theme") == theme:
                return (t, ((blob.get("swing") or {}).get("themes") or {}).get(theme),
                        ((blob.get("buys") or {}).get("themes") or {}).get(theme))
    return None


# ── Fliken ──────────────────────────────────────────────────────────────────
def _section(st, title: str) -> None:
    st.markdown(f"<div style='color:{GOLD};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;"
                f"margin:12px 0 6px;'>{escape(title)}</div>", unsafe_allow_html=True)


def _block(name: str, fn: Callable, note: Callable) -> None:
    """Ett block i taget: motorn som fallerar blir en notering, resten av fliken ritas."""
    try:
        fn()
    except Exception as exc:
        note(f"{name} kunde inte läsas: {exc}")


def render(st, ident: Ticker, data: dict, rotation_data: Optional[dict] = None, prices: Optional[Callable] = None) -> None:
    from ui.components import note
    theme = ident.theme or ""
    if not theme:
        note(f"{ident.yf} är inte knuten till någon råvara — fliken visas bara för råvarudrivna aktier.")
        return
    cov = coverage(theme)
    chips = [cards.chip(f"{ident.theme_label or label(theme)} · källa: {ident.theme_source}", GOLD, filled=True)]
    for key, text in (("rotation", "Rotationen"), ("blindspot", "Blindspot"), ("leverage", "Hävstång"), ("fiat", "Debasement"),
                      ("gold_pair", "Guld/råvara"), ("gold_silver", "Guld/silver"), ("ratios", "Gummiband")):
        chips.append(cards.chip(text, CYAN if cov.get(key) else GREY))
    st.markdown("<div style='display:flex;flex-wrap:wrap;gap:4px;margin:4px 0;'>" + "".join(chips) + "</div>",
                unsafe_allow_html=True)
    _section(st, "Råvarucykeln")
    _block("BERSERK", lambda: _berserk(st, theme, prices, note), note)
    _block("Valhall", lambda: _valhall(st, data, theme, note), note)
    _block("Rotationen", lambda: _rotation(st, theme, rotation_data, note), note)
    _block("Odins blindspot", lambda: _blindspot(st, theme, note), note)
    _section(st, "Hävstång mot råvaran")
    _block("Hävstången", lambda: _leverage(st, ident, theme, note), note)
    _section(st, "Debasement")
    _block("Debasement", lambda: _debasement(st, theme, note), note)
    _section(st, "Kvoter")
    _block("Kvoterna", lambda: _ratios(st, theme, note), note)
    note("Samma motorer som Odins visdom: BERSERK Regime, Valhall-kartan, Råvarurotationen (Triple Signal), Odins "
         "blindspot, Råvaruhävstång, Fiat Debasement, Guld/Silver, Guldkvoter och Alpha Regimes gummiband — för just "
         "den här råvaran. Grått chip = motorn täcker inte temat.")


def _berserk(st, theme: str, prices, note) -> None:
    from berserk import live
    from berserk.regime_ui import theme_card
    drivers = live.load_drivers([theme], getter=prices)
    rows = [r for r in live.theme_states(drivers) if r.get("theme") == theme]
    if rows:
        st.markdown(theme_card(rows[0]), unsafe_allow_html=True)
    else:
        note(f"BERSERK har ingen drivare (prisserie) för {label(theme)}.")


def _valhall(st, data: dict, theme: str, note) -> None:
    found = valhall_theme((data or {}).get("board"), theme)
    if not found:
        note(f"{label(theme)} finns inte som tema på Valhall-kartan.")
        return
    from valhall.ui import theme_card
    st.markdown(theme_card(*found), unsafe_allow_html=True)


def _rotation(st, theme: str, rotation_data, note) -> None:
    import commodity_book
    import rotation
    key = rotation_key(theme)
    if not key:
        note(f"{label(theme)} finns inte i rotationsflikens tabell — inget Triple Signal-betyg att läsa.")
        return
    name = rotation.COMMODITY_BY_KEY[key].name
    state = rotation_state(theme, rotation_data)
    if state is None:
        st.markdown(cards.verdict_card("INTE BETYGSATT", f"{name} saknar månadens Triple Signal-betyg.",
                                       "Sätt betyget i Odins visdom → Råvarurotation — obetygsatt är inte godkänt.",
                                       GREY, label="ROTATIONEN · TRIPLE SIGNAL"), unsafe_allow_html=True)
    else:
        col = GREEN if state["status"] == rotation.AGERA else AMBER if state["status"] == rotation.BEVAKA else GREY
        parts = [f"{state['sum']}/{state['max']}", state.get("month") or "okänd månad"]
        parts += [f"VARNING: {w}" for w in state.get("warnings") or []]
        st.markdown(cards.verdict_card(str(state["status"]).upper(), state["why"], " · ".join(parts), col,
                                       label=f"ROTATIONEN · TRIPLE SIGNAL · {name.upper()}"), unsafe_allow_html=True)
    ch = commodity_book.chapter(key)
    if ch is not None:
        with st.expander(f"Kartboken: {name}", expanded=False):
            st.markdown(rotation.chapter_html(ch), unsafe_allow_html=True)


def _blindspot(st, theme: str, note) -> None:
    from blindspot import theme_board as tb
    key = blindspot_key(theme)
    if not key:
        note(f"Odins blindspot har inget tema för {label(theme)}.")
        return
    with st.spinner("Odins blindspot …"):
        themes = tb.build_theme_board() or []
    r = next((t for t in themes if t.key == key), None)
    if r is None or r.error:
        note(f"Blindspot: {label(theme)} — {r.error if r else 'ingen rad i temabrädet'}")
        return
    rank = themes.index(r) + 1
    pct = "—" if r.percentile_10y is None else f"P{r.percentile_10y:.0f}"
    slope = "" if r.ma200w_slope_pct is None else f"MA200w {r.ma200w_slope_pct:+.1f} %"
    st.markdown(cards.kpi_row([("Cykel", r.cykel_label, CYKEL_COLOR.get(r.cykel_label, GREY), " · ".join(x for x in (f"10 år {pct}", slope) if x)),
                               ("Hat", f"{r.hat_score:.0f}/100", AMBER, "avstånd från topp · volym · RS"),
                               ("Nödvändighet", f"{r.necessity}/100", CYAN, "proxy" if r.proxy_flag else ""),
                               ("Blindspot", f"{r.blindspot_score:.1f}", GOLD, f"plats {rank} av {len(themes)}")]),
                unsafe_allow_html=True)
    st.markdown(f"<div style='color:{TEXT};font-size:0.8rem;line-height:1.5;'>{escape(tb.theme_verdict_text(r))}</div>",
                unsafe_allow_html=True)


def _leverage(st, ident: Ticker, theme: str, note) -> None:
    key = leverage_key(theme)
    if not key:
        from commodity_leverage.config import COMMODITY_LABELS
        note(f"Hävstångsmotorn täcker {', '.join(v.lower() for v in COMMODITY_LABELS.values())} — inte {label(theme).lower()}.")
        return
    from commodity_leverage import ui as clu
    with st.spinner(f"Hävstången {ident.yf} mot {label(key).lower()} …"):
        r = clu._get(ident.yf, key, False)
    clu._detail(r)


def _debasement(st, theme: str, note) -> None:
    from fiat_debasement import environment as env
    st.markdown(env.badge_html(env.current("USD")), unsafe_allow_html=True)
    asset = fiat_asset(theme)
    if not asset:
        note(f"Realprisvyn finns för guld, silver, koppar och olja — inte {label(theme).lower()}.")
        return
    import plotly.graph_objects as go
    from fiat_debasement import engine as fe
    from fiat_debasement import ui as fu
    r = fu.real_commodity(asset, BASE_YEAR)
    nom, real, rel = r.get("nominal"), r.get("real"), r.get("vs_gold")
    if nom is None or not len(nom):
        note(f"{r.get('label') or asset}: {fu.NA}")
        return
    years = (nom.index[-1] - r["base"]).days / 365.25
    d_nom = fe.change_pct(nom, years) if nom.index[0] <= r["base"] else None
    d_real = fe.change_pct(real, years) if real is not None and len(real) and real.index[0] <= r["base"] else None
    items = [(f"{r['label']} nominellt", f"{float(nom.iloc[-1]):,.2f}", GOLD, f"{r.get('unit') or 'USD'} · {nom.index[-1].date()}"),
             (f"I {BASE_YEAR} års dollar", fu.NA if real is None or not len(real) else f"{float(real.iloc[-1]):,.2f}", CYAN,
              "KPI-justerat (CPI-U)"),
             (f"Sedan {BASE_YEAR}", fu.fmt(d_nom), AMBER, f"nominellt · realt {fu.fmt(d_real)}")]
    if rel is not None and len(rel):
        items.append(("Mätt i guld", f"{float(rel.iloc[-1]):.4g}", GREY, "pris / guldpris (uns)"))
    st.markdown(cards.kpi_row(items), unsafe_allow_html=True)
    fig = go.Figure(go.Scatter(x=nom.index, y=nom.values, name="Nominellt (USD)", line=dict(color=GOLD, width=1.6)))
    if real is not None and len(real):
        fig.add_trace(go.Scatter(x=real.index, y=real.values, name=f"Fast penningvärde ({BASE_YEAR} USD)",
                                 line=dict(color=CYAN, width=1.6)))
    if rel is not None and len(rel):
        fig.add_trace(go.Scatter(x=rel.index, y=rel.values, name="Relativt guld", yaxis="y2", line=dict(color=GREY, width=1.3, dash="dot")))
    lay = fu._layout(f"{r['label'].upper()}: NOMINELLT, REALT OCH MOT GULD", height=300, ytitle=r.get("unit") or "")
    lay["yaxis2"] = dict(overlaying="y", side="right", showgrid=False, title="mot guld")
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="thor_fd_chart")
    for n in r.get("notes") or []:
        note(n)


def _ratios(st, theme: str, note) -> None:
    items = []
    if theme in PRECIOUS:
        from fiat_debasement import ui as fu
        from gold_silver import engine as ge
        series, rows = fu.gs_ratio()
        if series is not None and len(series):
            cur = float(series.iloc[-1])
            row = next((x for x in rows if str(x.get("Period", "")).startswith("10")), rows[-1] if rows else {})
            med, pct = row.get("Median"), row.get("Nu mot perioden (percentil)")
            sub = " · ".join(x for x in (f"median {med:.0f}" if med is not None else "", f"P{pct:.0f}" if pct is not None else "",
                                         ge.guide_zone(cur) or "") if x)
            items.append(("Guld/silver", f"{cur:.1f}", GOLD, sub))
    pair = gold_pair(theme)
    if pair:
        from gold_ratios import engine as gre
        from gold_ratios import ui as gu
        d = (gu._load() or {}).get(pair["key"]) or {}
        row = gre.position_row(pair, d.get("series"))
        if row.get("current") is not None:
            sub = " · ".join(x for x in (f"median 10 år {row['median10']:,.4g}" if row.get("median10") is not None else "",
                                         f"{row['diff_pct']:+.0f} %" if row.get("diff_pct") is not None else "",
                                         f"P{row['pctl10']:.0f}" if row.get("pctl10") is not None else "",
                                         row.get("position") or "") if x)
            items.append((f"Guld/{pair['label'].lower()}", f"{row['current']:,.4g}", CYAN, sub))
    keys = ratio_keys(theme)
    if keys:
        from alpha_regime.commodity_ratios import fetch_all_ratios
        allr = fetch_all_ratios() or {}
        for k in keys:
            rr = allr.get(k)
            if rr is None or rr.error:
                continue
            items.append((rr.label, f"{rr.current:,.4g}", RATIO_COLOR.get(rr.status, GREY),
                          f"P{rr.percentile:.0f} · {rr.status.replace('_', ' ').lower()} · {rr.driver.replace('_', ' ').lower()}"))
    if not items:
        note(f"Inga kvoter för {label(theme).lower()}.")
        return
    st.markdown(cards.kpi_row(items), unsafe_allow_html=True)
    st.markdown(f"<div style='color:{DIM};font-size:0.72rem;'>Guld/silver och guld/råvara: hög kvot = ett uns guld "
                f"köper mycket av råvaran. Gummibanden: percentil i tio års historik — stretched = nämnaren billig "
                f"relativt täljaren. Neutralt underlag, ingen signal.</div>", unsafe_allow_html=True)
