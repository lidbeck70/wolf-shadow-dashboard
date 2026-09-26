"""
asymmetry/charts.py — de fem diagrammen i fliken 🐺 Wolf Asymmetry.

Alla tar AsymmetryResult och returnerar en plotly-figur, eller None när
underlaget saknas (DATA_MISSING ritas inte som noll). Husstilen ur
ui/charts.PLOTLY_LAYOUT; färgerna ur ui/tokens: cyan/guld för serier,
grön/gul/röd bara för status (poängband, tecken) och alltid med etikett.
"""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go

from ui.charts import PLOTLY_LAYOUT
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, RED

_GRID = "rgba(0,229,255,0.08)"
_DIVERGING = [[0.0, RED], [0.5, "#1a1f25"], [1.0, GREEN]]     # röd · neutral yta · grön, mitten = 0


def _layout(title: str, height: int = 300, **kw) -> dict:
    base = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    base.update(title=dict(text=title, font=dict(size=13, color=CYAN)), height=height,
                xaxis=dict(**PLOTLY_LAYOUT["xaxis"]), yaxis=dict(**PLOTLY_LAYOUT["yaxis"]),
                hoverlabel=dict(bgcolor="#14141e", font=dict(family="Courier New, monospace", size=11)))
    for k, v in kw.items():
        if k in ("xaxis", "yaxis"):
            base[k].update(v)
        else:
            base[k] = v
    return base


def _status(points: Optional[float], maximum: float) -> str:
    share = points / maximum if maximum else 0
    return GREEN if share >= 0.7 else AMBER if share >= 0.4 else RED


# ── 1. prisgrid: FCF och EBITDA mot råvarupriset ─────────────────────────────
def price_grid_chart(r) -> Optional[go.Figure]:
    pts = [p for p in r.leverage.grid if p.ebitda_musd is not None]
    if len(pts) < 2:
        return None
    x = [p.price_pct for p in pts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[p.ebitda_musd for p in pts], name="EBITDA", mode="lines+markers",
                             line=dict(color=GOLD, width=2), marker=dict(size=8),
                             hovertemplate="pris %{x:+g} %<br>EBITDA %{y:,.0f} MUSD<extra></extra>"))
    if any(p.fcf_musd is not None for p in pts):
        fig.add_trace(go.Scatter(x=x, y=[p.fcf_musd for p in pts], name="FCF", mode="lines+markers",
                                 line=dict(color=CYAN, width=2), marker=dict(size=8),
                                 hovertemplate="pris %{x:+g} %<br>FCF %{y:,.0f} MUSD<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"))
    probe, down = r.leverage.probe_pct, -r.leverage.probe_pct
    for v, label in ((probe, f"poäng mäts här ({r.leverage.label})"), (down, "nedsideskontroll")):
        fig.add_vline(x=v, line=dict(color="rgba(0,229,255,0.25)", width=1, dash="dash"),
                      annotation_text=label, annotation_font=dict(color=DIM, size=10), annotation_position="top")
    fig.update_layout(**_layout("PRISGRID — EBITDA OCH FCF MOT RÅVARUPRISET", 320,
                                xaxis=dict(ticksuffix=" %", tickvals=x), yaxis=dict(title="MUSD"),
                                legend=dict(orientation="h", y=-0.2, font=dict(color=DIM))))
    return fig


# ── 2. margin of safety: fem delar om 0–2 ────────────────────────────────────
def safety_chart(r) -> Optional[go.Figure]:
    comps = [c for c in r.safety.components if not c.not_applicable]
    if not comps or all(c.points is None for c in comps):
        return None
    labels = [c.label for c in comps][::-1]
    vals = [c.points if c.points is not None else 0 for c in comps][::-1]
    colors = [_status(c.points, c.max) if c.points is not None else DIM for c in comps][::-1]
    text = [(f"{c.points:g}/{c.max:g}" if c.points is not None else "DATA_MISSING") for c in comps][::-1]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker=dict(color=colors, line=dict(width=0)),
                           text=text, textposition="outside", textfont=dict(color="#e8e4dc", size=11),
                           hovertemplate="%{y}<br>%{text}<extra></extra>", width=0.55))
    fig.update_layout(**_layout(f"MARGIN OF SAFETY — {r.safety.label}", 60 + 48 * len(comps),
                                xaxis=dict(range=[0, comps[0].max * 1.35], tickvals=[0, 0.5, 1, 1.5, 2]),
                                yaxis=dict(tickfont=dict(color="#e8e4dc")), showlegend=False,
                                margin=dict(l=10, r=20, t=50, b=30)))
    return fig


# ── 3. scenarier: uppsida per scenario ───────────────────────────────────────
def scenario_chart(r) -> Optional[go.Figure]:
    scen = [s for s in r.scenarios if s.upside_pct is not None]
    if not scen:
        return None
    fig = go.Figure(go.Bar(
        x=[s.label for s in scen], y=[s.upside_pct for s in scen],
        marker=dict(color=[GREEN if s.upside_pct >= 0 else RED for s in scen], line=dict(width=0)),
        text=[f"{s.upside_pct:+.0f} %" for s in scen], textposition="outside",
        textfont=dict(color="#e8e4dc", size=11), width=0.5,
        customdata=[[s.price_change_pct, s.capex_change_pct, s.equity_musd or 0] for s in scen],
        hovertemplate="%{x}<br>pris %{customdata[0]:+g} % · capex %{customdata[1]:+g} %<br>"
                      "equity %{customdata[2]:,.0f} MUSD<br>uppsida %{y:+.0f} %<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1))
    sub = ""
    if r.asymmetry and r.asymmetry.ratio not in (None, float("inf")):
        sub = f" — BULL/|BEAR| {r.asymmetry.ratio:.1f}× ({r.asymmetry.band})"
    fig.update_layout(**_layout("SCENARIER — UPPSIDA MOT BÖRSVÄRDE" + sub, 300,
                                yaxis=dict(ticksuffix=" %"), showlegend=False))
    return fig


# ── 4. stressmatris: pris × capex ────────────────────────────────────────────
def matrix_chart(r) -> Optional[go.Figure]:
    z, text = [], []
    for pp in r.matrix.price_pct:
        row, trow = [], []
        for cp in r.matrix.capex_pct:
            p = r.matrix.cell(pp, cp)
            v = p.upside_pct if p else None
            row.append(v)
            trow.append("–" if v is None else f"{v:+.0f} %")
        z.append(row)
        text.append(trow)
    if all(v is None for row in z for v in row):
        return None
    lim = max(abs(v) for row in z for v in row if v is not None) or 1
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"capex {c:+g} %" for c in r.matrix.capex_pct], y=[f"pris {p:+g} %" for p in r.matrix.price_pct],
        text=text, texttemplate="%{text}", textfont=dict(size=12, family="Courier New"),
        colorscale=_DIVERGING, zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
        colorbar=dict(ticksuffix=" %", tickfont=dict(color=DIM, size=10), outlinewidth=0),
        hovertemplate="%{y} · %{x}<br>uppsida %{z:+.0f} %<extra></extra>"))
    fig.update_layout(**_layout("STRESSMATRIS — UPPSIDA VID PRIS × CAPEX", 80 + 60 * len(z),
                                yaxis=dict(autorange="reversed", tickfont=dict(color="#e8e4dc")),
                                xaxis=dict(side="bottom", tickfont=dict(color="#e8e4dc"))))
    return fig


# ── 5. justerad uppsida mot confidence ───────────────────────────────────────
def adjusted_upside_chart(r) -> Optional[go.Figure]:
    from asymmetry.stress import adjusted_upside
    if r.base_upside_pct is None:
        return None
    xs = list(range(0, 101, 5))
    ys = [adjusted_upside(r.base_upside_pct, c) for c in xs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="uppsida × confidence/100",
                             line=dict(color=CYAN, width=2),
                             hovertemplate="confidence %{x}<br>justerad uppsida %{y:+.0f} %<extra></extra>"))
    fig.add_hline(y=r.base_upside_pct, line=dict(color=GOLD, width=1, dash="dot"),
                  annotation_text=f"Base {r.base_upside_pct:+.0f} %", annotation_font=dict(color=GOLD, size=10),
                  annotation_position="top left")
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1))
    if r.confidence is not None and r.adjusted_upside_pct is not None:
        fig.add_trace(go.Scatter(x=[r.confidence], y=[r.adjusted_upside_pct], mode="markers+text",
                                 name="detta bolag", marker=dict(color=GOLD, size=12, line=dict(color="#0c0c12", width=2)),
                                 text=[f"{r.adjusted_upside_pct:+.0f} % vid {r.confidence:g}"], textposition="top center",
                                 textfont=dict(color="#e8e4dc", size=11),
                                 hovertemplate="confidence %{x:g}<br>justerad uppsida %{y:+.0f} %<extra></extra>"))
    fig.update_layout(**_layout("JUSTERAD UPPSIDA — VAD CONFIDENCE GÖR MED BASE", 280,
                                xaxis=dict(title="Confidence 0–100", range=[0, 100]), yaxis=dict(ticksuffix=" %"),
                                legend=dict(orientation="h", y=-0.3, font=dict(color=DIM))))
    return fig


ALL = (price_grid_chart, safety_chart, scenario_chart, matrix_chart, adjusted_upside_chart)
