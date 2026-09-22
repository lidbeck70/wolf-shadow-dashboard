"""
engines/durrett/management.py — STEP 2 management (SPEC §6).

Extra vikt på faktisk erfarenhet: byggt, finansierat, drivit gruvor,
upptäckt fyndigheter, skapat aktieägarvärde. CLAIMED räknas till hälften
av VERIFIED; UNKNOWN ger N/A för komponenten — inte 0.
"""

from __future__ import annotations

from engines.durrett._base import Ctx, scale_0_2, score_from, steps_ge
from engines.durrett.models import Score

_TRACK_FACTOR = {"verified": 1.0, "claimed": 0.5}


def score(ctx: Ctx, ss: dict) -> Score:
    st = ctx.cfg["score_steps"]
    comps: dict = {}

    # track record — byggt/finansierat/drivit/upptäckt/exits/avkastning
    verified = ctx.text("mgmt_track_verified") or "unknown"
    built = ctx.num("team_mines_built", required=False)
    ceo_built = ctx.num("ceo_mines_built", required=False)
    financed = ctx.num("team_mines_financed", required=False)
    operated = ctx.num("team_mines_operated", required=False)
    disc = ctx.num("team_discoveries", required=False)
    exits = ctx.num("team_exits", required=False)
    returns = ctx.num("prior_shareholder_returns", required=False)
    known = [v for v in (built, ceo_built, financed, operated, disc, exits, returns) if v is not None]
    if verified == "unknown" or not known:
        comps["track_record"] = (None, "track record ej verifierat — UNKNOWN, inte 0")
    else:
        pts = 0.0
        why = []
        for label, v, per, cap in (("gruvor byggda", built, 15, 45), ("CEO byggt", ceo_built, 10, 20),
                                   ("finansierade", financed, 8, 16), ("drivna", operated, 6, 12),
                                   ("upptäckter", disc, 8, 16), ("exits", exits, 6, 12)):
            if v is not None and v > 0:
                pts += min(v * per, cap)
                why.append(f"{label} {int(v)}")
        if returns is not None:
            pts += {0: -15, 1: 5, 2: 15}.get(int(returns), 0)
            why.append(f"aktieägaravkastning {int(returns)}/2")
        raw = min(100.0, 20.0 + pts)
        f = _TRACK_FACTOR[verified]
        val = raw * f if verified == "claimed" else raw
        comps["track_record"] = (round(val, 1), f"{verified.upper()}: " + ", ".join(why) +
                                 (" (CLAIMED räknas ×0,5)" if verified == "claimed" else ""))

    own = ctx.num("insider_ownership_pct", required=False)
    comps["insider_ownership"] = (steps_ge(own, st["insider_ownership_pct"]),
                                  f"insynsägande {own:g} %" if own is not None else "insynsägande saknas")

    buy = ctx.num("insider_buying_12m_musd", required=False)
    sell = ctx.num("insider_selling_12m_musd", required=False)
    if buy is None and sell is None:
        comps["insider_activity"] = (None, "insideraffärer okända")
    else:
        net = (buy or 0) - (sell or 0)
        mcap = ss.get("mcap_usd")
        rel = (net / mcap * 100) if mcap else None
        if rel is not None:
            val = steps_ge(rel, ((1.0, 100), (0.25, 80), (0.0, 60), (-0.25, 40), (-1.0, 20)), 5)
            comps["insider_activity"] = (val, f"netto {net:+.2f} MUSD = {rel:+.2f} % av börsvärdet")
        else:
            comps["insider_activity"] = (70.0 if net > 0 else (50.0 if net == 0 else 30.0), f"netto {net:+.2f} MUSD")

    board = ctx.num("board_mining_experience", required=False)
    comps["board"] = (scale_0_2(board), f"styrelsens gruverfarenhet {board:g}/2" if board is not None else "styrelse okänd")

    comp = ctx.num("mgmt_compensation_musd", required=False)
    related = ctx.truth("related_party_issues", required=False)
    align_parts = []
    align_val = None
    mcap = ss.get("mcap_usd")
    if comp is not None and mcap:
        share = comp / mcap * 100
        align_val = steps_ge(-share, ((-0.5, 100), (-1.0, 80), (-2.0, 60), (-4.0, 40)), 20)
        align_parts.append(f"ersättning {share:.1f} % av börsvärdet")
    if related is not None:
        adj = -30.0 if related else 0.0
        align_val = (align_val if align_val is not None else 70.0) + adj
        align_parts.append("närståendeproblem" if related else "inga närståendeproblem")
    dil = ctx.num("mgmt_dilution_history", required=False)
    if dil is not None:
        align_val = (align_val if align_val is not None else 50.0) + (15.0 if dil >= 1 else -15.0)
        align_parts.append("disciplinerad utspädning" if dil >= 1 else "odisciplinerad utspädning")
    comps["alignment"] = (None if align_val is None else round(max(0.0, min(100.0, align_val)), 1),
                          ", ".join(align_parts) or "ersättning/närstående okänt")
    return score_from("management", "Management", comps, ctx.cfg["sub_weights"]["management"], ctx)
