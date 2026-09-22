"""
engines/durrett/dilution.py — STEP 3 aktiestruktur (SPEC §7).

FD Shares = basic + optioner + warranter + konvertibler + övrigt.
Dilution 1/3/5 år = (aktier nu / aktier då − 1). FD Market Cap och FD EV
i USD (valutanormaliserat via fx_to_usd). SERIAL DILUTER identifieras
när 3-årsutspädningen ≥ tröskel eller årlig ≥ tröskel tre år i rad.

share_structure() används även av valuation/upside — en beräkning, en plats.
"""

from __future__ import annotations

from typing import Optional

from engines.durrett._base import Ctx, score_from, steps_le
from engines.durrett.models import Score


def fd_shares(basic, options=None, warrants=None, convertibles=None, other=None) -> Optional[float]:
    """Fullt utspätt antal aktier (M). None om basic saknas."""
    if basic is None:
        return None
    return basic + (options or 0.0) + (warrants or 0.0) + (convertibles or 0.0) + (other or 0.0)


def dilution_pct(now: Optional[float], then: Optional[float]) -> Optional[float]:
    if now is None or then is None or then <= 0:
        return None
    return (now / then - 1.0) * 100.0


def share_structure(ctx: Ctx) -> dict:
    """{basic_m, fd_m, dilutive_m, dilution_1y/3y/5y, price_usd, mcap_usd, fd_mcap_usd, ev_usd, fd_ev_usd,
       net_debt_usd, serial_diluter, notes}. Värden None när de inte går att räkna."""
    basic = ctx.num("basic_shares_m", required=False)
    if basic is None:
        basic = ctx.num("shares_outstanding_m", required=False)
        if basic is not None:
            ctx.log.append("basic_shares_m saknas — shares_outstanding_m används som basic (kan redan vara FD)")
        else:
            ctx.missing.append("basic_shares_m")
    opts = ctx.num("options_m", required=False)
    war = ctx.num("warrants_m", required=False)
    conv = ctx.num("convertible_shares_m", required=False)
    other = ctx.num("other_dilutive_m", required=False)
    fd = fd_shares(basic, opts, war, conv, other)
    dilutive = (opts or 0) + (war or 0) + (conv or 0) + (other or 0) if basic is not None else None
    d1 = dilution_pct(basic, ctx.num("shares_1y_ago_m", required=False))
    d3 = dilution_pct(basic, ctx.num("shares_3y_ago_m", required=False))
    d5 = dilution_pct(basic, ctx.num("shares_5y_ago_m", required=False))

    price = ctx.usd("share_price", required=False)
    mcap = ctx.usd("market_cap_musd", required=False)
    if mcap is None and price is not None and basic is not None:
        mcap = price * basic
        ctx.log.append("börsvärde = kurs × basic-aktier (MODELLED)")
    fd_mcap = price * fd if price is not None and fd is not None else None
    if fd_mcap is None and mcap is not None and basic and fd:
        fd_mcap = mcap * fd / basic
    cash = ctx.usd("cash_musd", required=False)
    debt = ctx.usd("debt_musd", required=False)
    net_debt = (debt or 0.0) - (cash or 0.0) if (cash is not None or debt is not None) else None
    ev = ctx.usd("enterprise_value_musd", required=False)
    if ev is None and mcap is not None and net_debt is not None:
        ev = mcap + net_debt
    fd_ev = fd_mcap + net_debt if fd_mcap is not None and net_debt is not None else None

    thr = ctx.cfg["red_flag_thresholds"]
    serial = None
    if d3 is not None:
        serial = d3 >= thr["serial_diluter_3y_pct"]
        if not serial and d1 is not None and d5 is not None:
            yearly = [d1, ((1 + d3 / 100) / (1 + d1 / 100) - 1) * 100 / 2, ((1 + d5 / 100) / (1 + d3 / 100) - 1) * 100 / 2]
            serial = all(y >= thr["serial_diluter_yearly_pct"] for y in yearly)
    out = {"basic_m": basic, "fd_m": fd, "dilutive_m": dilutive, "dilution_1y": d1, "dilution_3y": d3,
           "dilution_5y": d5, "price_usd": price, "mcap_usd": mcap, "fd_mcap_usd": fd_mcap, "ev_usd": ev,
           "fd_ev_usd": fd_ev, "net_debt_usd": net_debt, "cash_usd": cash, "debt_usd": debt,
           "serial_diluter": serial}
    for k, unit in (("fd_m", "M"), ("dilution_1y", "%"), ("dilution_3y", "%"), ("dilution_5y", "%"),
                    ("mcap_usd", "MUSD"), ("fd_mcap_usd", "MUSD"), ("ev_usd", "MUSD"), ("fd_ev_usd", "MUSD"),
                    ("net_debt_usd", "MUSD")):
        ctx.metric(k, out[k], unit)
    return out


def score(ctx: Ctx, ss: Optional[dict] = None) -> Score:
    ss = ss or share_structure(ctx)
    st = ctx.cfg["score_steps"]
    comps: dict = {}
    d3, d1 = ss["dilution_3y"], ss["dilution_1y"]
    if d3 is not None:
        comps["historisk utspädning 3 år"] = (steps_le(d3, st["dilution_3y_pct"], ctx.cfg["score_beyond"]["dilution_3y_pct"]),
                                             f"{d3:+.0f} % fler aktier på 3 år")
    elif d1 is not None:
        comps["historisk utspädning 3 år"] = (steps_le(d1 * 3, st["dilution_3y_pct"], 5), f"bara 1 år känt: {d1:+.0f} % (×3 som proxy)")
    else:
        comps["historisk utspädning 3 år"] = (None, "aktiehistorik saknas")
    if ss["basic_m"] and ss["fd_m"]:
        overhang = (ss["fd_m"] / ss["basic_m"] - 1) * 100
        comps["utestående instrument"] = (steps_le(overhang, ((2.0, 100), (5.0, 85), (10.0, 65), (20.0, 45), (35.0, 25)), 10),
                                          f"optioner/warranter/konvertibler = {overhang:.0f} % av basic")
    else:
        comps["utestående instrument"] = (None, "basic-aktier saknas")
    atm = ctx.truth("atm_program", required=False)
    exp_fin = ctx.num("expected_financing_musd", required=False)
    mcap = ss["mcap_usd"]
    if exp_fin is not None and mcap:
        share = exp_fin / mcap * 100
        comps["väntad finansiering"] = (steps_le(share, ((0.0, 100), (5.0, 80), (15.0, 55), (30.0, 30)), 10),
                                        f"{exp_fin:g} MUSD väntas = {share:.0f} % av börsvärdet" + (" · ATM aktivt" if atm else ""))
    elif atm is not None:
        comps["väntad finansiering"] = (30.0 if atm else 80.0, "ATM-program aktivt" if atm else "inget ATM-program")
    else:
        comps["väntad finansiering"] = (None, "väntad finansiering/ATM okänt")
    s = score_from("dilution", "Dilution", comps,
                   {"historisk utspädning 3 år": 50, "utestående instrument": 25, "väntad finansiering": 25}, ctx)
    if ss["serial_diluter"]:
        s.negative.append("− SERIAL DILUTER (red flag)")
        if s.value is not None:
            s.value = round(min(s.value, 25.0), 1)
    return s
