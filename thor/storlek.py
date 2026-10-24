"""
thor/storlek.py — Storlek-raden i Thor (designförslaget PR 6): hur mycket som köps, med formeln synlig.

Två tal visas sida vid sida och slås aldrig ihop (beslut 6 = nej till "köp högst" — det vore en ny regel):
  riskstorlek  kapital × strategins risk per affär ÷ (entry − stopp)  — bara där strategin har en fast risk-%
  tak          kapital × allokerarens hårda tak (% av portföljen) ÷ entry
Mål = entry + 3 × (entry − stopp). Risk-procenten läses ur strategimodulerna (samma tal som strategy_rules visar),
taket ur allocator.POSITION_RULES. Rena funktioner — inget Streamlit, inget nätverk.

"Logga köp" lägger positionen i registret (positions.add, beslut 7). Journalraden skrivs när positionen
stängs (positions.close → journal_bridge), för journalen håller stängda affärer. Arkstrategierna går via
🎯 Scorecard: köpgrinden KLAR FÖR KÖP ligger kvar i kedjan.
"""
from __future__ import annotations

import math
from typing import Optional

# Thor-nyckel → positionsregistrets strategitagg (positions.STRATEGIES)
POSITION_TAG = {"momentum": "Momentum", "wolf": "Wolf", "viking": "Viking", "viking_nine": "Viking Nine",
                "berserk": "BERSERK", "ember": "Ember", "quality": "Quality", "deep": "Deep Contrarian",
                "long": "Alpha", "insider": "Insider", "rule": "Rule", "durrett": "Durrett", "tiggre": "Tiggre",
                "royalty": "Royalty", "hav": "Untagged"}
# Strategierna där köpet går via arket och köpgrinden (🎯 Scorecard) — Thor visar storleken men loggar inte köpet
SHEET_STRATEGIES = frozenset({"rule", "durrett", "tiggre", "royalty", "hav"})


def risk_pct(strategy: str) -> Optional[float]:
    """Strategins fasta risk per affär i procent, eller None där storleken inte är riskbaserad
    (Momentum är positionsbaserad, arken bedömer per case)."""
    try:
        if strategy == "wolf":
            from strategies.wolf import DEFAULT_PARAMS as p
            return float(p["risk_pct"]) * 100
        if strategy in ("viking", "viking_nine"):
            if strategy == "viking_nine":
                import strategy_rules as sr
                return float(sr._VN_RISK)
            from strategies.viking import DEFAULT_PARAMS as p
            return float(p["risk_pct"]) * 100
        if strategy == "ember":
            from ember.config import RISK_PCT
            return float(RISK_PCT) * 100
        if strategy in ("long", "quality", "deep"):
            from strategies.alpha import DEFAULT_PARAMS as p
            return float(p["risk_pct"]) * 100
    except Exception:
        return None
    return None


def cap_pct(strategy: str) -> Optional[float]:
    """Allokerarens hårda tak för strategin (% av hela portföljen), eller None."""
    import positions
    import allocator as al
    tag = POSITION_TAG.get(strategy)
    key = positions.ALLOCATOR_RULE.get(tag) if tag else None
    r = al.rule_for_strategy(key) if key else None
    return r.hard_cap if r else None


def size(entry, stop, capital, risk: Optional[float] = None, cap: Optional[float] = None) -> Optional[dict]:
    """{entry, stop, target_3r, risk_amt, risk_shares, cap_shares, cap_amt, formula} — None utan giltig entry/stopp."""
    try:
        e, s, k = float(entry), float(stop), float(capital)
    except (TypeError, ValueError):
        return None
    if not (e > 0 and 0 < s < e and k > 0):
        return None
    dist = e - s
    out = {"entry": e, "stop": s, "target_3r": round(e + 3 * dist, 2), "risk_amt": None, "risk_shares": None,
           "cap_shares": None, "cap_amt": None}
    parts = []
    if risk:
        out["risk_amt"] = round(k * risk / 100, 0)
        out["risk_shares"] = int(math.floor(out["risk_amt"] / dist))
        parts.append(f"riskstorlek {out['risk_shares']} st ({risk:g} % av {k:,.0f} = {out['risk_amt']:,.0f} kr "
                     f"÷ {dist:.2f} per aktie)".replace(",", " "))
    if cap:
        out["cap_amt"] = round(k * cap / 100, 0)
        out["cap_shares"] = int(math.floor(out["cap_amt"] / e))
        parts.append(f"tak ur Allokering {out['cap_shares']} st ({cap:g} % = {out['cap_amt']:,.0f} kr)".replace(",", " "))
    out["formula"] = " · ".join(parts)
    return out


def row_text(ticker: str, label: str, sz: dict) -> str:
    """'BOL.ST · 🛡 Viking · köp 100,00 · stopp 92,00 · mål 3R 124,00 · riskstorlek … · tak …'"""
    head = f"{ticker} · {label} · köp {sz['entry']:.2f} · stopp {sz['stop']:.2f} · mål 3R {sz['target_3r']:.2f}"
    return head + (f" · {sz['formula']}" if sz.get("formula") else "")


def position_args(ticker: str, name: str, strategy: str, sz: dict, shares: int) -> dict:
    """Argumenten till positions.add för 'Logga köp'."""
    return {"ticker": ticker, "strategy": POSITION_TAG.get(strategy, "Untagged"), "entry_price": sz["entry"],
            "shares": int(shares), "name": name, "stop": sz["stop"], "target": sz["target_3r"], "source": "thor"}
