"""
engines/durrett/momentum.py — STEP 6 market buzz / chart (SPEC §10).

Sekundär komponent, hålls utanför Quality Score (config.momentum_in_quality
= 0). Kurs mot MA200, 6-månadersutveckling, RS-rank, volym-, sektor-,
råvarumomentum och nyhetsflöde — allt matas in som data (fliken hämtar).
"""

from __future__ import annotations

from engines.durrett._base import Ctx, scale_0_2, score_from, steps_ge
from engines.durrett.models import Score


def score(ctx: Ctx) -> Score:
    st = ctx.cfg["score_steps"]
    comps: dict = {}
    ma = ctx.num("price_vs_ma200_pct", required=False)
    m6 = ctx.num("momentum_6m_pct", required=False)
    if ma is not None or m6 is not None:
        vals, why = [], []
        if ma is not None:
            vals.append(steps_ge(ma, st["momentum_pct"], ctx.cfg["score_beyond"]["momentum_pct"]))
            why.append(f"kurs {ma:+.0f} % mot MA200")
        if m6 is not None:
            vals.append(steps_ge(m6, st["momentum_pct"], ctx.cfg["score_beyond"]["momentum_pct"]))
            why.append(f"6 mån {m6:+.0f} %")
        comps["price_trend"] = (round(sum(vals) / len(vals), 1), ", ".join(why))
    else:
        comps["price_trend"] = (None, "kurs mot MA200 / 6 mån saknas")
    rs = ctx.num("rs_rank", required=False)
    comps["relative_strength"] = (None if rs is None else max(0.0, min(100.0, rs)),
                                  f"RS-rank {rs:g}" if rs is not None else "RS-rank saknas")
    for key, name in (("commodity_momentum", "commodity"), ("sector_momentum", "sector"), ("news_flow", "news")):
        v = ctx.num(key, required=False)
        comps[name] = (scale_0_2(v), f"{name} {v:g}/2" if v is not None else f"{name} okänt")
    vol = ctx.num("volume_trend", required=False)
    if vol is not None and comps["price_trend"][0] is not None:
        adj = {0: -5.0, 1: 0.0, 2: 5.0}.get(int(vol), 0.0)
        comps["price_trend"] = (round(max(0.0, min(100.0, comps["price_trend"][0] + adj)), 1),
                                comps["price_trend"][1] + f", volymtrend {int(vol)}/2")
    return score_from("momentum", "Momentum", comps, ctx.cfg["sub_weights"]["momentum"], ctx, min_coverage=0.35)
