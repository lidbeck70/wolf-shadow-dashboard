"""
engines/durrett/balance_sheet.py — STEP 8 cash / debt (SPEC §12).

Nettoskuld, ND/EBITDA (producenter), cash runway = kassa / årsburn,
likviditet (rörelsekapital, räntetäckning, förfall). Runway räknas bara
när burn är känd — annars N/A, aldrig 0.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from engines.durrett import config as dc
from engines.durrett._base import Ctx, score_from, steps_ge, steps_le, steps_lt
from engines.durrett.models import Score


def runway_years(cash: Optional[float], annual_burn: Optional[float]) -> Optional[float]:
    if cash is None or annual_burn is None or annual_burn <= 0:
        return None
    return max(0.0, cash) / annual_burn


def annual_burn(ctx: Ctx) -> Optional[float]:
    """Årsburn i MUSD: quarterly_burn × 4, annars −operativt kassaflöde om negativt."""
    q = ctx.num("quarterly_burn_musd", required=False)
    if q is not None:
        return q * 4.0
    ocf = ctx.usd("operating_cash_flow_musd", required=False)
    if ocf is not None and ocf < 0:
        return -ocf
    return None


def score(ctx: Ctx, company_type: str, ss: dict, today: Optional[date] = None) -> Score:
    st = ctx.cfg["score_steps"]
    today = today or date.today()
    comps: dict = {}
    cash, debt, nd = ss.get("cash_usd"), ss.get("debt_usd"), ss.get("net_debt_usd")
    nde = ctx.num("net_debt_ebitda", required=False)
    if company_type in (dc.PRODUCER, dc.ROYALTY, dc.HYBRID):
        if nde is not None:
            comps["net_debt"] = (steps_lt(nde, st["net_debt_ebitda"], ctx.cfg["score_beyond"]["net_debt_ebitda"]),
                                 f"nettoskuld/EBITDA {nde:.2f}×")
        elif nd is not None:
            comps["net_debt"] = (85.0 if nd <= 0 else None, f"nettokassa {-nd:g} MUSD" if nd <= 0 else "nettoskuld utan EBITDA — kvoten okänd")
        else:
            comps["net_debt"] = (None, "nettoskuld/EBITDA saknas")
    else:
        if nd is not None:
            mcap = ss.get("mcap_usd")
            if nd <= 0:
                comps["net_debt"] = (90.0, f"nettokassa {-nd:g} MUSD")
            elif mcap:
                comps["net_debt"] = (steps_le(nd / mcap * 100, ((5.0, 70), (15.0, 50), (30.0, 30)), 10),
                                     f"nettoskuld {nd:g} MUSD = {nd / mcap * 100:.0f} % av börsvärdet")
            else:
                comps["net_debt"] = (40.0, f"nettoskuld {nd:g} MUSD")
        else:
            comps["net_debt"] = (None, "kassa/skuld saknas")

    burn = annual_burn(ctx)
    rw = runway_years(cash, burn)
    if rw is not None:
        ctx.metric("cash_runway_years", rw, "år", f"kassa {cash:g} / årsburn {burn:g}")
        comps["runway"] = (steps_ge(rw, st["runway_years"]), f"runway {rw:.1f} år (kassa {cash:g} / burn {burn:g} MUSD/år)")
    elif company_type in (dc.PRODUCER, dc.ROYALTY, dc.HYBRID):
        fcf = ctx.usd("free_cash_flow_musd", required=False)
        if fcf is not None and fcf >= 0:
            comps["runway"] = (90.0, f"FCF-positiv ({fcf:+.0f} MUSD) — ingen burn")
        elif cash is not None and burn is None:
            comps["runway"] = (None, "burn/FCF okänd")
        else:
            comps["runway"] = (None, "kassa saknas")
    else:
        comps["runway"] = (None, "kassa eller burn saknas — runway N/A")

    parts, vals = [], []
    wc = ctx.usd("working_capital_musd", required=False)
    if wc is not None:
        vals.append(80.0 if wc > 0 else 25.0)
        parts.append(f"rörelsekapital {wc:+.0f} MUSD")
    ie = ctx.usd("interest_expense_musd", required=False)
    ocf = ctx.usd("operating_cash_flow_musd", required=False)
    if ie is not None and ie > 0 and ocf is not None:
        cover = ocf / ie
        vals.append(steps_ge(cover, ((5.0, 100), (3.0, 75), (1.5, 50), (0.0, 25)), 10))
        parts.append(f"räntetäckning {cover:.1f}×")
    mat = ctx.num("debt_maturity_year", required=False)
    if mat is not None and debt:
        yrs = mat - today.year
        vals.append(steps_ge(yrs, ((4.0, 90), (2.0, 65), (1.0, 40), (0.0, 20))))
        parts.append(f"skuld förfaller {int(mat)} ({yrs:g} år)")
    comps["liquidity"] = (round(sum(vals) / len(vals), 1) if vals else None, ", ".join(parts) or "likviditetsdata saknas")
    return score_from("balance_sheet", "Balance Sheet", comps, ctx.cfg["sub_weights"]["balance_sheet"], ctx, min_coverage=0.4)
