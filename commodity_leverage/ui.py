"""
commodity_leverage/ui.py — GRANSKNING → 🚀 Råvaruhävstång.

Skriv ett eller flera bolag (upp till åtta) och se hävstången mot råvaran:
resultathävstång (Snabbkollens motor), kurshävstång (beta, upp/ned) och
5×-potential i en jämförelsetabell, sedan detaljen för ett valt bolag med
fritt råvarupris genom samma kedja som 5×-motorn.
"""

from __future__ import annotations

import time
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from commodity_leverage import beta as cb
from commodity_leverage import config as cc
from commodity_leverage import engine as ce
from ui.charts import PLOTLY_LAYOUT
from ui.components import big_card, note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, RED, TEXT

_CACHE = "cl_cache"
_TTL_S = 3600
_FLAG_COLOR = {"STARK": GREEN, "MÅTTLIG": AMBER, "SKÖR": RED}


def _section(title: str, sub: str = "") -> None:
    st.markdown(f"<div style='color:{CYAN};font-family:Courier New;letter-spacing:2px;font-size:0.85rem;"
                f"margin:18px 0 6px;'>{title}" + (f" <span style='color:{DIM};letter-spacing:0;'>— {sub}</span>"
                                                  if sub else "") + "</div>", unsafe_allow_html=True)


def _get(ticker: str, commodity: Optional[str], force: bool) -> ce.CompanyLeverage:
    cache = st.session_state.setdefault(_CACHE, {})
    key = f"{ticker}|{commodity or ''}"
    hit = cache.get(key)
    if force or not hit or time.time() - hit["t"] > _TTL_S:
        cache[key] = {"t": time.time(), "res": ce.analyze(ticker, commodity)}
    return cache[key]["res"]


def render_commodity_leverage_page() -> None:
    page_header("🚀 Råvaruhävstång", "Hur mycket rör sig bolagets resultat och aktie när råvaran rör sig? "
                                    "Resultathävstång, kurshävstång och vad som krävs för 5×.")
    labels = [cc.AUTO] + [cc.COMMODITY_LABELS[k] for k in cc.COMMODITY_LABELS]
    with st.form("cl_form"):
        raw = st.text_input(f"Bolag (upp till {cc.MAX_TICKERS}, kommaseparerade)", key="cl_tickers",
                            placeholder="t.ex. FCX, SCCO, TECK, LUN.TO, BOL.ST")
        c1, c2 = st.columns([2, 1])
        pick = c1.selectbox("Råvara", labels, key="cl_commodity",
                            help="Auto = bolagets tema (temakartan, arket, Holdings, Yahoos bransch). "
                                 "Välj för att låsa — t.ex. silver för ett silverbolag som Yahoo kallar guld.")
        run = c2.form_submit_button("🚀 Mät")
    tickers = ce.parse_tickers(raw)
    if not tickers:
        note("Skriv ett eller flera bolag. Resultathävstången kräver årsrapporter (Börsdata) — kurshävstången "
             "bara kurser, så den fungerar även för juniorer utan resultat.")
        return
    commodity = next((k for k, v in cc.COMMODITY_LABELS.items() if v == pick), None)
    with st.spinner("Mäter hävstången …"):
        rows = sorted((_get(t, commodity, run) for t in tickers), key=ce.rank_key)
    _comparison(rows)
    names = [f"{r.ticker} · {r.name}" for r in rows]
    choice = st.selectbox("Detalj för", names, key="cl_detail") if len(rows) > 1 else names[0]
    _detail(rows[names.index(choice)])


# ── Jämförelsen ──────────────────────────────────────────────────────────────
def _comparison(rows: list) -> None:
    _section("📋 JÄMFÖRELSE", "rankad efter resultathävstång, sedan kursbeta")

    def lev_cell(r):
        lev = r.lev
        if lev is None or lev.score is None:
            return f"<td style='color:{GREY};'>DATA_GAP</td><td>—</td><td>—</td>"
        be = "EJ MÄTBAR" if lev.break_even_margin_pct is None else f"{lev.break_even_margin_pct:.0f} %"
        c = _FLAG_COLOR.get(lev.downside_label, GREY)
        return (f"<td style='font-weight:700;color:{CYAN};'>{lev.score}/10<br><span style='color:{DIM};"
                f"font-size:0.68rem;font-weight:400;'>{lev.basis} {lev.response_pct:+.0f} %</span></td>"
                f"<td>{be}</td><td style='color:{c};'>{lev.downside_label}</td>")

    def beta_cell(r):
        b = r.beta
        if b is None:
            return f"<td style='color:{GREY};'>DATA_GAP</td><td>—</td>"
        updn = ("—" if b.up_beta is None or b.down_beta is None else
                f"{b.up_beta:.1f}× / {b.down_beta:.1f}×{' ⬆' if b.asymmetric else ''}")
        return (f"<td style='color:{AMBER if b.weak else TEXT};'>{b.beta:.2f}×<br><span style='color:{DIM};"
                f"font-size:0.68rem;'>R² {b.r2:.2f}</span></td><td>{updn}</td>")

    def five_cell(r):
        e = r.eng
        v = getattr(e, "five_x", "") if e is not None and not e.error else ""
        c = {"JA": GREEN, "VILLKORAT": AMBER, "NEJ": RED}.get(v, GREY)
        return f"<td style='color:{c};font-weight:700;'>{v or '—'}</td>"

    body = "".join(
        f"<tr><td style='text-align:left;'><b>{r.ticker}</b><br><span style='color:{DIM};font-size:0.68rem;'>"
        f"{(r.name or '')[:22]}</span></td><td>{(r.commodity or '—').capitalize()}{' 🔒' if r.locked else ''}</td>"
        + lev_cell(r) + beta_cell(r) + five_cell(r) + "</tr>" for r in rows)
    head = "".join(f"<th>{h}</th>" for h in ("Råvara", "Resultat-<br>hävstång", "Break-even", "Nedsida",
                                             "Kursbeta", "Upp / ned", "5×"))
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.78rem;color:{TEXT};"
                f"text-align:right;'><tr style='color:{DIM};'><th style='text-align:left;'>Bolag</th>{head}</tr>"
                f"{body}</table></div>", unsafe_allow_html=True)
    note("Resultathävstång = hur mycket FCF (annars EBITDA) rör sig när råvaran stiger 20 %, ur bolagets egna "
         "år (Snabbkollens motor). Kursbeta = hur mycket aktien rört sig per 1 % i råvaran, veckovis, "
         f"{cc.BETA_YEARS} år. Upp/ned = beta i veckor då råvaran steg respektive föll; ⬆ = följer med mer upp "
         f"än ned (asymmetri). Gul beta = svagt samband (R² < {cc.BETA_WEAK_R2:g}). 🔒 = råvaran vald manuellt.")


# ── Detaljen ─────────────────────────────────────────────────────────────────
def _detail(r: ce.CompanyLeverage) -> None:
    from asymmetry import quick_leverage as ql
    from asymmetry import quick_scenarios as qs
    _section(f"🔎 {r.ticker} · {r.name}", (r.commodity or "råvara okänd").capitalize())
    lev, b = r.lev, r.beta
    c1, c2, c3 = st.columns(3)
    if lev is None or lev.score is None:
        c1.markdown(big_card("RESULTATHÄVSTÅNG", "—", f"DATA_GAP: {getattr(lev, 'error', '') or 'saknas'}", GREY),
                    unsafe_allow_html=True)
        c2.markdown(big_card("BREAK-EVEN", "—", "kräver resultathävstång", GREY), unsafe_allow_html=True)
    else:
        c1.markdown(big_card("RESULTATHÄVSTÅNG", f"{lev.score}/10",
                             f"{r.commodity} +20 % → {lev.basis} {lev.response_pct:+.0f} % · {lev.flag}",
                             _FLAG_COLOR.get(lev.downside_label, GREY)), unsafe_allow_html=True)
        if lev.break_even_margin_pct is None:
            c2.markdown(big_card("BREAK-EVEN", "—", f"{lev.band} · {lev.break_even_note}", GREY),
                        unsafe_allow_html=True)
        else:
            c2.markdown(big_card("BREAK-EVEN", f"{lev.break_even_margin_pct:.0f} %",
                                 f"{lev.band} · nu {ql.fmt_price(lev.price_now)} mot "
                                 f"{ql.fmt_price(lev.break_even_price)} {lev.unit}", GREEN
                                 if lev.band in ("UTMÄRKT", "STARK") else AMBER), unsafe_allow_html=True)
    if b is None:
        c3.markdown(big_card("KURSBETA", "—", f"DATA_GAP: {r.beta_error}", GREY), unsafe_allow_html=True)
    else:
        sub = (f"upp {b.up_beta:.1f}× · ned {b.down_beta:.1f}×" if b.up_beta is not None and b.down_beta is not None
               else "för få upp-/nedveckor") + f" · R² {b.r2:.2f} · {b.weeks} veckor"
        c3.markdown(big_card("KURSBETA", f"{b.beta:.2f}×", sub + (" · ⬆ asymmetri" if b.asymmetric else ""),
                             AMBER if b.weak else CYAN), unsafe_allow_html=True)

    _custom_price(r, qs, ql)
    fig = beta_chart(r)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"cl_beta_{r.ticker}")
    if lev is not None and lev.sensitivity:
        try:
            from asymmetry.quick_ui import leverage_chart
            lf = leverage_chart(r.data, lev)
            if lf is not None:
                st.plotly_chart(lf, use_container_width=True, config={"displayModeBar": False},
                                key=f"cl_lev_{r.ticker}")
        except Exception:
            pass
    note("Båda måtten är historiska: de visar hur bolaget och aktien har följt råvaran, inte hur de kommer att "
         "göra det. Aktien mäts i sin handelsvaluta mot råvaran i USD — valutan ingår i betat.")


def _custom_price(r: ce.CompanyLeverage, qs, ql) -> None:
    cp = (r.data or {}).get("commodity_px") or {}
    p0, fx = cp.get("p0"), cp.get("fx_now") or 1.0
    if not p0 or r.eng is None or r.eng.error:
        note(f"Fritt råvarupris kräver bolagets värderingskedja: {getattr(r.eng, 'error', '') or 'saknas'}.")
        return
    now_usd = p0 / fx
    lo, hi = cc.CUSTOM_PCT_RANGE
    pct = st.slider(f"{r.commodity.capitalize()}-pris mot i dag (%)", lo, hi, 0, 5, key=f"cl_pct_{r.ticker}")
    usd = now_usd * (1 + pct / 100)
    steps = [("Valt pris", usd)] + [(f"{p:+.0f} %", now_usd * (1 + p / 100)) for p in (-30, -20, 0, 20, 50)
                                    if p != pct]
    _res, pts = qs.at_prices(r.data, steps, r.lev)
    if not pts:
        return
    ccy = r.eng.price_ccy
    unit = r.lev.unit if r.lev is not None else ""
    body = "".join(
        f"<tr style='{'font-weight:700;color:' + CYAN + ';' if p.name == 'Valt pris' else ''}'>"
        f"<td style='text-align:left;'>{p.name}{' ⚠' if p.outside_history else ''}</td>"
        f"<td>{ql.fmt_price(p.price)} ({p.price_pct:+.0f} %)</td>"
        f"<td>{'—' if p.revenue is None else f'{p.revenue:,.0f}'}</td><td>{p.ebitda:,.0f}</td>"
        f"<td>{'—' if p.fcf is None else f'{p.fcf:,.0f}'}</td><td>{p.equity:,.0f}</td>"
        f"<td style='color:{GREEN if p.ratio >= 1 else AMBER if p.ratio >= 0.7 else RED};'>{p.ratio:.2f}×</td>"
        f"<td>{'—' if p.share_price is None else f'{p.share_price:,.2f}'}</td></tr>" for p in pts)
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};"
                f"text-align:right;'><tr style='color:{DIM};'><th style='text-align:left;'></th>"
                f"<th>{r.commodity.capitalize()} ({unit})</th><th>Intäkt M</th><th>EBITDA M</th><th>FCF M</th>"
                f"<th>Eget kapital M</th><th>× börsvärde</th><th>Kurs {ccy}</th></tr>{body}</table></div>",
                unsafe_allow_html=True)
    note(f"Samma kedja som 5×-motorn: pris → bolagets egen linje → EV med egen median "
         f"{r.eng.multiples['median']:g}× EV/EBITDA → − nettoskuld → mot börsvärdet → kurs. ⚠ = utanför tio års "
         f"årssnitt (linjen extrapoleras).")


def beta_chart(r: ce.CompanyLeverage) -> Optional[go.Figure]:
    """Veckoavkastning aktie mot råvara, med beta-linjen."""
    if r.beta is None:
        return None
    try:
        stock = ce._series_default(r.data.get("yf_ticker") or r.ticker, cc.BETA_PERIOD)
        ctick = (r.data.get("commodity_px") or {}).get("ticker")
        com = ce._series_default(ctick, cc.BETA_PERIOD) if ctick else None
    except Exception:
        return None
    w = cb.weekly_returns(stock, com)
    if len(w) < 2:
        return None
    xs, ys = [float(v) * 100 for v in w["c"]], [float(v) * 100 for v in w["s"]]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", name="veckor",
                               marker=dict(color=CYAN, size=5, opacity=0.6)))
    lo, hi = min(xs), max(xs)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[r.beta.beta * lo, r.beta.beta * hi], mode="lines",
                             name=f"beta {r.beta.beta:.2f}×", line=dict(color=GOLD, dash="dot")))
    layout = {k: v for k, v in PLOTLY_LAYOUT.items()}
    layout.update(title=dict(text=f"AKTIEN MOT {r.commodity.upper()} — VECKOAVKASTNING %",
                             font=dict(size=12, color=CYAN)), height=320,
                  xaxis=dict(title=f"{r.commodity} %"), yaxis=dict(title="aktie %"),
                  legend=dict(orientation="h", y=-0.25, font=dict(color=DIM, size=10)))
    fig.update_layout(**layout)
    return fig
