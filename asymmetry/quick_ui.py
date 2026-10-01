"""
asymmetry/quick_ui.py — Snabbkollen i Wolf Asymmetry: skriv en ticker, få
Survival, Margin of Safety och Confidence (auto) som mätare med kort under,
ett poängkort i Viking Regime-stil och fyra grafer. Helt automatiskt —
inget matas in, inget sparas förrän du trycker "Lägg till i arket".
"""

from __future__ import annotations

import time
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from ui.charts import PLOTLY_LAYOUT, build_gauge
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, RED, TEXT

from asymmetry import quick, quick_data
from asymmetry import quick_config as qc

CACHE_KEY = "asym_quick_cache"
CACHE_TTL_S = 3600
_STATUS_COLOR = {"GREEN": GREEN, "AMBER": AMBER, "RED": RED, "DATA_GAP": GREY}
_STATUS_ICON = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴", "DATA_GAP": "⚪"}
_VERDICT_COLOR = {"GRÖN": GREEN, "GUL": AMBER, "RÖD": RED, "DATA_GAP": GREY}
_VERDICT_LABEL = {"GRÖN": "STARK ASYMMETRI", "GUL": "BLANDAD BILD", "RÖD": "SVAG ASYMMETRI",
                  "DATA_GAP": "FÖR LITE DATA"}


def _cached(ticker: str, force: bool = False) -> dict:
    cache = st.session_state.setdefault(CACHE_KEY, {})
    hit = cache.get(ticker)
    if hit and not force and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    data = quick_data.fetch(ticker)
    cache[ticker] = (time.time(), data)
    return data


def render_quick(add_to_sheet=None) -> None:
    """add_to_sheet(ticker, name) -> bool: kopplas av Wolf Asymmetry-sidan."""
    with st.form("asym_quick_form", border=False):
        c1, c2, c3 = st.columns([3, 1, 1])
        ticker = c1.text_input("Ticker", key="asym_quick_ticker", placeholder="T.ex. BOL.ST, LUG.ST, NEM, AEM.TO, IPT.V",
                               label_visibility="collapsed")
        go_btn = c2.form_submit_button("🔍 Analysera", use_container_width=True)
        refresh = c3.form_submit_button("🔄 Uppdatera", use_container_width=True)
    t = str(ticker or "").strip().upper()
    if not t:
        _note("Skriv en ticker. Börsdata först (nordiskt, sedan globalt), Yahoo som reserv. "
                   "Survival · Margin of Safety · Confidence räknas helt automatiskt.")
        return
    with st.spinner(f"Hämtar {t} …"):
        data = _cached(t, force=bool(refresh))
    if not data.get("prices") and data.get("source") in ("—", "", None):
        st.warning(f"Hittade inget för {t} i Börsdata eller Yahoo. Kontrollera tickern (t.ex. suffix .ST, .OL, .TO).")
        return
    res = quick.score(data)
    _score_card(res, data)
    _gauges(res)
    _volatility(res)
    lev = _leverage(data)
    _engine(data, lev)
    _charts(data)
    extra = []
    if data.get("filled_yahoo"):
        extra.append("Ur Yahoo (Börsdata tomt): " + ", ".join(data["filled_yahoo"]))
    if extra:
        _note(" · ".join(extra))
    if add_to_sheet is not None:
        if st.button("➕ Lägg till i arket (Analys/Ark)", key=f"asym_quick_add_{t}"):
            ok = add_to_sheet(data.get("yf_ticker") or t, data.get("name") or t)
            (st.success if ok else st.info)(
                f"{t} ligger i arket — öppna Ark för råvara och stage." if ok else f"{t} finns redan i arket.")


# Förklaringstext och stora kort delas med andra flikar (ui/components).
from ui.components import big_card as _metric_card  # noqa: E402
from ui.components import note as _note  # noqa: E402


# ── Poängkortet (Viking Regime-stil) ────────────────────────────────────────
def _score_card(res: quick.QuickResult, data: dict) -> None:
    col = _VERDICT_COLOR.get(res.verdict, GREY)
    total = res.total
    total_txt = f"{total:.0f}" if total is not None else "—"
    measured = sum(len(g.measured) for g in res.groups)
    n_cards = sum(len(g.scored) for g in res.groups)
    st.markdown(
        f"""<div style="background:linear-gradient(135deg, rgba(0,229,255,0.06) 0%, rgba(139,115,64,0.03) 100%);
        border:1px solid {col};border-radius:12px;padding:28px 16px;text-align:center;position:relative;
        overflow:hidden;margin-bottom:10px;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{CYAN},#00A8BF);"></div>
        <div style="font-size:11px;letter-spacing:4px;color:rgba(0,229,255,0.5);margin-bottom:8px;">WOLF ASYMMETRY SCORE</div>
        <div style="font-size:76px;font-weight:900;line-height:1;color:{col};font-family:'Courier New',monospace;">{total_txt}</div>
        <div style="font-size:11px;letter-spacing:3px;color:rgba(0,229,255,0.4);margin-top:4px;">/ 300 MAX</div>
        <div style="margin-top:14px;"><span style="background:{col}1a;border:1px solid {col};border-radius:20px;
        color:{col};font-size:11px;font-weight:700;letter-spacing:3px;padding:5px 16px;">
        {res.verdict} · {_VERDICT_LABEL.get(res.verdict, "")}</span></div>
        <div style="margin-top:12px;font-size:0.8rem;color:{TEXT};">{" · ".join(res.reasons)}</div>
        <div style="margin-top:14px;font-size:10px;color:rgba(0,229,255,0.35);letter-spacing:2px;">
        {res.ticker} · {res.name} · {res.source} · {measured}/{n_cards} kort mätta · {data.get("fetched", "")}</div>
        </div>""",
        unsafe_allow_html=True)


# ── Mätare + kort ────────────────────────────────────────────────────────────
def _gauges(res: quick.QuickResult) -> None:
    cols = st.columns(3)
    for col, g in zip(cols, res.groups):
        with col:
            if g.score is None:
                st.markdown(f"<div style='text-align:center;color:{DIM};font-family:Courier New;"
                            f"letter-spacing:2px;margin:40px 0 8px;'>{g.label.upper()}<br>"
                            f"<span style='font-size:1.6rem;color:{GREY};'>DATA_GAP</span></div>",
                            unsafe_allow_html=True)
            else:
                st.plotly_chart(build_gauge(round(g.score), 100, g.label.upper(), color_cyan=True),
                                use_container_width=True, config={"displayModeBar": False},
                                key=f"asym_quick_gauge_{g.key}")
            _note(f"{g.coverage}")
            for p in g.pillars:
                st.markdown(_card(p), unsafe_allow_html=True)


def _card(p) -> str:
    c = _STATUS_COLOR.get(p.status, GREY)
    return (f"<div style='border-left:3px solid {c};background:#14141e;border-radius:6px;"
            f"padding:7px 10px;margin-bottom:6px;'>"
            f"<div style='display:flex;justify-content:space-between;gap:8px;'>"
            f"<span style='color:{TEXT};font-size:0.82rem;'>{_STATUS_ICON.get(p.status, '')} {p.label}</span>"
            f"<span style='color:{c};font-size:0.82rem;font-weight:700;'>{p.value}</span></div>"
            f"<div style='color:{DIM};font-size:0.7rem;margin-top:2px;'>{p.why}</div></div>")


def _volatility(res: quick.QuickResult) -> None:
    """Cykelvolatilitet — egen dimension under mätarna, räknas inte i 300."""
    g = res.volatility
    if g is None or not g.pillars:
        return
    st.markdown(f"<div style='color:{CYAN};font-family:Courier New;letter-spacing:2px;font-size:0.8rem;"
                f"margin:6px 0 4px;'>📉 CYKELVOLATILITET <span style='color:{DIM};letter-spacing:0;'>"
                f"— info, räknas inte i 300. Svängande resultat och FCF är cykeln, inte sämre data.</span></div>",
                unsafe_allow_html=True)
    cols = st.columns(len(g.pillars))
    for col, p in zip(cols, g.pillars):
        col.markdown(_card(p), unsafe_allow_html=True)


def _leverage(data: dict):
    """🚀 Commodity Leverage och 💥 break-even — skattade ur historiken, utanför 300."""
    from asymmetry import quick_leverage as ql
    lev = ql.from_data(data)
    st.markdown(f"<div style='color:{CYAN};font-family:Courier New;letter-spacing:2px;font-size:0.8rem;"
                f"margin:10px 0 4px;'>🚀 RÅVARUHÄVSTÅNG <span style='color:{DIM};letter-spacing:0;'>"
                f"— egen dimension, räknas inte i 300</span></div>", unsafe_allow_html=True)
    name = (lev.commodity or "råvara").capitalize()
    c1, c2 = st.columns(2)
    if lev.score is None:
        c1.markdown(_metric_card("COMMODITY LEVERAGE", "—", f"DATA_GAP: {lev.error}", GREY), unsafe_allow_html=True)
        c2.markdown(_metric_card("BREAK-EVEN-MARGINAL", "—", "kräver ett användbart samband", GREY),
                    unsafe_allow_html=True)
    else:
        lc = GREEN if lev.downside_label == "STARK" else AMBER if lev.downside_label == "MÅTTLIG" else RED
        c1.markdown(_metric_card("COMMODITY LEVERAGE", f"{lev.score}/10",
                                 f"{name} +20 % → {lev.basis} {lev.response_pct:+.0f} % · {lev.flag}", lc),
                    unsafe_allow_html=True)
        if lev.break_even_margin_pct is None:
            c2.markdown(_metric_card("BREAK-EVEN-MARGINAL", "—", f"{lev.band} · {lev.break_even_note}", GREY),
                        unsafe_allow_html=True)
        else:
            bc = GREEN if lev.band in ("UTMÄRKT", "STARK") else AMBER if lev.band in ("MÅTTLIG", "SVAG") else RED
            c2.markdown(_metric_card("BREAK-EVEN-MARGINAL", f"{lev.break_even_margin_pct:.0f} %",
                                     f"{lev.band} · {name} nu {ql.fmt_price(lev.price_now)} mot break-even "
                                     f"{ql.fmt_price(lev.break_even_price)} {lev.unit}", bc), unsafe_allow_html=True)
    if not lev.sensitivity:
        return lev
    with st.expander(f"Känslighet mot {name.lower()} ({lev.ticker}) — hur det räknas"):
        _note(
            "Skattat ur bolagets egen historik: årlig intäkt, EBITDA och FCF (Börsdata, rapportvalutan) "
            f"mot {name.lower()}-priset som årssnitt omräknat till samma valuta. Rak linje (OLS) per mått; "
            f"break-even = priset där linjen når noll. Samband med färre än {qc.LEV_MIN_YEARS} år eller "
            f"R² under {qc.LEV_MIN_R2:g} poängsätts inte. En råvara räknas — multi-metallbolag får "
            f"huvudråvaran som proxy.")
        fits = " · ".join(f"{k}: R² {f.r2:.2f} ({f.n} år)" for k, f in lev.fits.items())
        if fits:
            _note("Samband — " + fits)
        if lev.downside:
            _note(f"Nedsida ({lev.downside_basis}): " + " · ".join(
                f"pris {pct:+.0f} % → {v:,.0f} M" for pct, v in lev.downside.items()))

        def _f(v, fmt="{:,.0f}"):
            return "—" if v is None else fmt.format(v)

        def _step(pct):
            return "BAS" if pct == 0 else f"{pct:+.0f} %"
        bold = f"font-weight:700;color:{CYAN};"
        rows = "".join(
            f"<tr style='{bold if r['pct'] == 0 else ''}'><td>{_step(r['pct'])}</td>"
            f"<td>{_f(r['price'], '{:,.2f}')}</td><td>{_f(r['revenue'])}</td><td>{_f(r['ebitda'])}</td>"
            f"<td>{_f(r['fcf'])}</td><td>{_f(r['ebitda_margin'], '{:.0f} %')}</td>"
            f"<td>{_f(r['fcf_margin'], '{:.0f} %')}</td></tr>" for r in lev.sensitivity)
        st.markdown(
            f"<table style='width:100%;font-size:0.78rem;color:{TEXT};text-align:right;'>"
            f"<tr style='color:{DIM};'><th>Pris</th><th>{name} ({lev.unit})</th><th>Intäkt M</th>"
            f"<th>EBITDA M</th><th>FCF M</th><th>EBITDA-marg.</th><th>FCF-marg.</th></tr>{rows}</table>",
            unsafe_allow_html=True)
        fig = leverage_chart(data, lev)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                            key=f"asym_quick_lev_{data.get('ticker', '')}")
    return lev


_FIVE_X_COLOR = {"JA": GREEN, "VILLKORAT": AMBER, "NEJ": RED}


def _ratio_color(r: float) -> str:
    return GREEN if r >= 2 else CYAN if r >= 1 else AMBER if r >= 0.7 else RED


def _engine(data: dict, lev) -> None:
    """⚡ 5×-motorn — scenarier, krav för 2×/3×/5×/10×, stressmatris och thesis
    killers ur samma linje som Commodity Leverage. Utanför 300."""
    from asymmetry import quick_scenarios as qs
    eng = qs.run(data, lev)
    st.markdown(f"<div style='color:{CYAN};font-family:Courier New;letter-spacing:2px;font-size:0.8rem;"
                f"margin:10px 0 4px;'>⚡ 5×-MOTORN <span style='color:{DIM};letter-spacing:0;'>"
                f"— automatisk, egen historik, räknas inte i 300</span></div>", unsafe_allow_html=True)
    if eng.error:
        st.markdown(_metric_card("5× POTENTIAL", "—", f"DATA_GAP: {eng.error}", GREY), unsafe_allow_html=True)
        return
    c1, c2 = st.columns(2)
    c1.markdown(_metric_card("5× POTENTIAL", eng.five_x or "—", f"kräver {eng.five_x_text}",
                             _FIVE_X_COLOR.get(eng.five_x, GREY)), unsafe_allow_html=True)
    base = next((x for x in eng.scenarios if x.name == "BASE"), None)
    if base:
        top = (f" · ⚠ råvaran nära 10-årstopp ({eng.price_pctl:.0f}:e percentilen) — troligen generöst"
               if eng.cycle_top else "")
        c2.markdown(_metric_card("BASE MOT BÖRSVÄRDET", f"{base.ratio:.2f}×",
                                 f"dagens pris, egen median {base.multiple:g}× EV/EBITDA{top}",
                                 AMBER if eng.cycle_top and base.ratio >= 1 else _ratio_color(base.ratio)),
                    unsafe_allow_html=True)
    ccy = eng.price_ccy
    with st.expander("5×-motorn — scenarier, krav, stressmatris och thesis killers"):
        m = eng.multiples
        _note(
            f"Kedjan: {eng.commodity}-pris → EBITDA (egen linje, R² {lev.fits['EBITDA'].r2:.2f}) → EV med egen "
            f"EV/EBITDA (lägsta {m['min']:g}× · 25:e percentil {m['low']:g}× · median {m['median']:g}× · "
            f"75:e {m['high']:g}×) → "
            f"− nettoskuld {eng.net_debt:,.0f} M → mot börsvärdet {eng.mcap:,.0f} M {eng.report_ccy} "
            f"({eng.mcap_note}) → kurs. Antalet aktier hålls fast (ingen utspädning).")
        head = (f"<tr style='color:{DIM};'><th style='text-align:left;'>Scenario</th><th>{eng.commodity.capitalize()}"
                f"</th><th>Pris ({eng.unit})</th><th>Multipel</th><th>EBITDA M</th><th>Eget kapital M</th>"
                f"<th>× börsvärde</th><th>Kurs {ccy}</th></tr>")
        rows = "".join(
            f"<tr><td style='text-align:left;'>{x.name}{' ⚠' if x.outside_history else ''}</td>"
            f"<td>{x.price_pct:+.0f} %</td><td>{x.price:,.2f}</td><td>{x.multiple:g}×</td>"
            f"<td>{x.ebitda:,.0f}</td><td>{x.equity:,.0f}</td>"
            f"<td style='color:{_ratio_color(x.ratio)};font-weight:700;'>{x.ratio:.2f}×</td>"
            f"<td>{'—' if x.share_price is None else f'{x.share_price:,.2f}'}</td></tr>" for x in eng.scenarios)
        st.markdown(f"<table style='width:100%;font-size:0.78rem;color:{TEXT};text-align:right;'>{head}{rows}</table>",
                    unsafe_allow_html=True)
        if any(x.outside_history for x in eng.scenarios):
            _note("⚠ = råvarupriset ligger utanför de senaste tio årens årssnitt — linjen extrapoleras.")
        req = "".join(
            f"<tr><td style='text-align:left;'>{r.multiple}×</td>"
            f"<td>{'—' if r.price_pct is None else f'{r.price_pct:+.0f} %'}</td>"
            f"<td>{'—' if r.price is None else f'{r.price:,.2f}'}</td>"
            f"<td style='color:{_FIVE_X_COLOR.get(r.verdict, GREY)};font-weight:700;'>{r.verdict}</td></tr>"
            for r in eng.requirements)
        st.markdown(
            f"<div style='color:{CYAN};font-size:0.8rem;margin-top:8px;'>Vad krävs? (egen median "
            f"{m['median']:g}× EV/EBITDA{f', högsta årssnitt 10 år {eng.price_high:,.2f}' if eng.price_high else ''})"
            f"</div><table style='width:100%;font-size:0.78rem;color:{TEXT};text-align:right;'>"
            f"<tr style='color:{DIM};'><th style='text-align:left;'>Mål</th><th>{eng.commodity.capitalize()}</th>"
            f"<th>Pris ({eng.unit})</th><th>Bedömning</th></tr>{req}</table>", unsafe_allow_html=True)
        _note(f"JA = priset har redan varit där (högsta årssnitt 10 år) · VILLKORAT = upp till "
                   f"{qc.FIVE_X_CONDITIONAL_FACTOR:g}× det · NEJ = längre bort.")
        mk = qc.STRESS_MULTIPLES
        srows = "".join(
            f"<tr><td style='text-align:left;'>{pct:+.0f} %</td>" + "".join(
                f"<td style='color:{_ratio_color(row[k][0])};'>{row[k][0]:.2f}×"
                f"{'' if row[k][1] is None else f' · {row[k][1]:,.0f}'}</td>" for k in mk) + "</tr>"
            for pct, row in eng.stress)
        st.markdown(
            f"<div style='color:{CYAN};font-size:0.8rem;margin-top:8px;'>Stressmatris — råvarupris mot multipel "
            f"(× börsvärdet · kurs {ccy})</div><table style='width:100%;font-size:0.78rem;color:{TEXT};"
            f"text-align:right;'><tr style='color:{DIM};'><th style='text-align:left;'>{eng.commodity.capitalize()}"
            f"</th>" + "".join(f"<th>{k} {m[k]:g}×</th>" for k in mk) + f"</tr>{srows}</table>",
            unsafe_allow_html=True)
        st.markdown(f"<div style='color:{RED};font-size:0.8rem;margin-top:10px;'>☠️ VAD DÖDAR CASET?</div>",
                    unsafe_allow_html=True)
        for k in eng.killers:
            tag = "" if k.measured else f" <span style='color:{DIM};'>(mäts inte automatiskt)</span>"
            st.markdown(f"<div style='font-size:0.78rem;color:{TEXT};'>• <b>{k.label}</b> — {k.detail}{tag}</div>",
                        unsafe_allow_html=True)
        _note("Sannolikhet: UNKNOWN — killers listas ur uppmätta tal, ingen sannolikhet hittas på.")


def leverage_chart(data: dict, lev) -> Optional[go.Figure]:
    """EBITDA och FCF mot råvarupriset per år, med den skattade linjen."""
    cp = data.get("commodity_px") or {}
    prices, fx = cp.get("prices") or {}, cp.get("fx_now") or 1.0
    fig = go.Figure()
    for key, series, color in (("EBITDA", data.get("ebitda_series"), CYAN), ("FCF", data.get("fcf_series"), GOLD)):
        pts = [(prices[y], v, y) for y, v in (series or []) if y in prices and v is not None]
        if len(pts) < 2:
            continue
        fig.add_trace(go.Scatter(x=[p / fx for p, _, _ in pts], y=[v for _, v, _ in pts], mode="markers+text",
                                 text=[str(y) for _, _, y in pts], textposition="top center", name=key,
                                 marker=dict(color=color, size=8), textfont=dict(color=DIM, size=9)))
        f = lev.fits.get(key)
        if f:
            xs = sorted(p for p, _, _ in pts)
            fig.add_trace(go.Scatter(x=[xs[0] / fx, xs[-1] / fx], y=[f.at(xs[0]), f.at(xs[-1])], mode="lines",
                                     name=f"{key}-linje (R² {f.r2:.2f})", line=dict(color=color, dash="dot")))
    if not fig.data:
        return None
    fig.update_layout(**_layout(f"{(lev.commodity or 'RÅVARA').upper()}-PRIS MOT EBITDA OCH FCF (PER ÅR)", 300))
    return fig


# ── Grafer ───────────────────────────────────────────────────────────────────
def _layout(title: str, height: int = 260) -> dict:
    base = {k: v for k, v in PLOTLY_LAYOUT.items()}
    base.update(title=dict(text=title, font=dict(size=12, color=CYAN)), height=height,
                legend=dict(orientation="h", y=-0.25, font=dict(color=DIM, size=10)))
    return base


def price_chart(data: dict) -> Optional[go.Figure]:
    px = data.get("prices") or []
    if len(px) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d for d, _ in px], y=[v for _, v in px], name="Kurs",
                             line=dict(color=CYAN, width=1.8)))
    sma = [(d, v) for d, v in (data.get("sma200") or []) if v is not None]
    if sma:
        fig.add_trace(go.Scatter(x=[d for d, _ in sma], y=[v for _, v in sma], name="SMA200",
                                 line=dict(color=GOLD, width=1.4, dash="dot")))
    fig.update_layout(**_layout("KURS 2 ÅR MOT 200-DAGARS SNITT"))
    return fig


def ev_ebitda_chart(data: dict) -> Optional[go.Figure]:
    s = [(y, v) for y, v in (data.get("ev_ebitda_series") or []) if v is not None and 0 < v < 100]
    if len(s) < 2:
        return None
    from statistics import median
    med = median(v for _, v in s)
    fig = go.Figure(go.Scatter(x=[y for y, _ in s], y=[v for _, v in s], mode="lines+markers", name="EV/EBITDA",
                               line=dict(color=CYAN, width=2), marker=dict(size=7)))
    fig.add_hline(y=med, line=dict(color=GOLD, width=1, dash="dot"), annotation_text=f"median {med:.1f}×",
                  annotation_font=dict(color=GOLD, size=10))
    fig.update_layout(**_layout("EV/EBITDA MOT EGEN MEDIAN"))
    return fig


def _bars(series, title: str, name: str, signed: bool = False) -> Optional[go.Figure]:
    s = [(y, v) for y, v in (series or []) if v is not None]
    if len(s) < 2:
        return None
    colors = [(GREEN if v >= 0 else RED) if signed else CYAN for _, v in s]
    fig = go.Figure(go.Bar(x=[y for y, _ in s], y=[v for _, v in s], name=name, marker=dict(color=colors)))
    fig.update_layout(**_layout(title), showlegend=False)
    return fig


def _charts(data: dict) -> None:
    figs = [(price_chart(data), "price"), (ev_ebitda_chart(data), "ev"),
            (_bars(data.get("shares_series"), "ANTAL AKTIER (M) — UTSPÄDNING", "Aktier"), "shares"),
            (_bars(data.get("fcf_series"), "FRITT KASSAFLÖDE PER ÅR (M)", "FCF", signed=True), "fcf")]
    for row in (figs[:2], figs[2:]):
        cols = st.columns(2)
        for col, (fig, key) in zip(cols, row):
            with col:
                if fig is None:
                    _note("Grafen kan inte ritas: för lite historik.")
                else:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                                    key=f"asym_quick_chart_{key}")
