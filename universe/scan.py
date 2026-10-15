"""
universe/scan.py — skannrarnas väg in i universum-registret.

  members_for(strategy, …)  registrets medlemmar för strategins marknader (universe.registry.markets_for), eller
                            None när registret inte finns i Gisten — då tar skannern sin gamla väg.
  Huvudlistorna             bolag som bara finns i Börsdata räknas på huvudlistorna (Large/Mid/Small Cap, Oslo Børs,
                            Köpenhamn, Helsingfors, NYSE, Nasdaq, Toronto, London — MAIN_BOARD_IDS ur markets.FALLBACK),
                            inte First North, TSX Venture eller CSE; bolag ur panelens kurerade listor räknas alltid.
  Taket (max_members)       skyddar jobbens körtid: kurerade listor först, sedan huvudlistorna marknad för marknad
                            (Norden, USA, Kanada, London, Australien) i bokstavsordning.
  describe(members)         en rad för loggen: antal per marknad och källa.
Rena funktioner; registret skickas in eller läses med den loader som ges (gist_storage.load_blob som förval).
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from universe import registry as ur

MAIN_BOARD_IDS = frozenset({1, 2, 3, 9, 14, 20, 32, 33, 35, 38})
MARKET_ORDER = {m.key: i for i, m in enumerate(ur.MARKETS)}


def is_curated(member: ur.Member) -> bool:
    """Kommer bolaget ur någon av panelens egna listor (inte bara Börsdata)?"""
    return any(not str(s).startswith("Börsdata") for s in member.sources or ())


def _priority(member: ur.Member) -> tuple:
    return (0 if is_curated(member) else 1, MARKET_ORDER.get(member.market, 9), member.ticker)


def select(universe: ur.Universe, markets: Iterable[str], themes: Optional[Iterable[str]] = None,
           main_boards: bool = True, kinds: Iterable[str] = (ur.STOCK,), commodity_only: bool = False,
           max_members: Optional[int] = None) -> list:
    """Medlemmarna som uppfyller filtren, i prioritetsordning (kurerade först), högst max_members."""
    ms, ks = set(markets or ()), set(kinds or ())
    ts = set(themes) if themes else None
    out = []
    for m in universe.members.values():
        if m.market not in ms or (ks and m.kind not in ks):
            continue
        if commodity_only and not m.theme:
            continue
        if ts is not None and m.theme not in ts:
            continue
        if main_boards and not is_curated(m) and m.market_id not in MAIN_BOARD_IDS:
            continue
        out.append(m)
    out.sort(key=_priority)
    return out[:max_members] if max_members else out


def members_for(strategy: str, universe: Optional[ur.Universe] = None, loader: Optional[Callable] = None,
                markets: Optional[Iterable[str]] = None, themes: Optional[Iterable[str]] = None,
                main_boards: bool = True, kinds: Iterable[str] = (ur.STOCK,), commodity_only: Optional[bool] = None,
                max_members: Optional[int] = None) -> Optional[list]:
    """Registrets medlemmar för strategin — None när registret saknas (skannern tar sin gamla väg)."""
    ms = tuple(markets) if markets else ur.markets_for(strategy)
    if universe is None:
        universe = ur.load(ms, loader)
    if universe is None or not universe.members:
        return None
    if commodity_only is None:
        commodity_only = tuple(ms) == ur.COMMODITY_MARKETS
    return select(universe, ms, themes, main_boards, kinds, commodity_only, max_members)


def tickers_for(strategy: str, **kw) -> Optional[list]:
    members = members_for(strategy, **kw)
    return None if members is None else [m.ticker for m in members]


def as_instruments(members: Iterable[ur.Member]) -> list:
    """Registrets Börsdata-bolag som instrumentrader i Börsdatas form (insId, ticker, name, marketId, sectorId,
    branchId, stockPriceCurrency) — för skannrar som förut läste api.get_instruments(). Listornas bolag utan
    insId hoppas över."""
    out = []
    for m in members or []:
        if m.ins_id is None or m.market_id is None:
            continue
        out.append({"insId": m.ins_id, "ticker": m.bd_ticker or m.ticker, "name": m.name, "marketId": m.market_id,
                    "sectorId": m.sector_id, "branchId": m.branch_id, "stockPriceCurrency": m.currency,
                    "market": m.market})
    return out


def describe(members: Iterable[ur.Member]) -> str:
    """'412 bolag · Norden 120 · USA 240 · Kanada 52 · kurerade 230 · bara Börsdata 182'."""
    rows = list(members or [])
    per: dict = {}
    for m in rows:
        per[m.market] = per.get(m.market, 0) + 1
    cur = sum(1 for m in rows if is_curated(m))
    keys = sorted(per, key=lambda k: MARKET_ORDER.get(k, 9))
    parts = [f"{len(rows)} bolag"] + [f"{ur.label(k)} {per[k]}" for k in keys]
    return " · ".join(parts + [f"kurerade {cur}", f"bara Börsdata {len(rows) - cur}"])
