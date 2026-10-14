"""
thor/dossier.py — allt panelen vet om en aktie, samlat: identiteten, dagens kandidatrader, ögonblicksbilden,
cykeln (aktiens fas, hemmamarknadens, Valhall-segmentets läge, playbookens grind) och arken (bara läsning).

Rena funktioner över data som skickas in: kandidatlistan (thor.candidates), kartan (valhall_board.json),
arken (storage.session_load-objekten), kursdata och en fasfunktion. Motorerna per strategi (regimlayouterna),
råvaran och backtestet kommer i fas 4–6 och läggs ovanpå det här. Inga regler ändras.
"""

from __future__ import annotations

from typing import Callable, Optional

from thor import candidates as tc
from thor import registry as reg
from thor.ticker import Ticker, base, forms, identify

NO_PLAYBOOK = "Strategin har ingen playbook med cykelfaser — fasen visas, men ingen grind."


def candidates_for(rows: list, ticker) -> list:
    """Dagens Thor-rader för aktien (alla strategier), i listans ordning. 'BOL' hittar 'BOL.ST'."""
    want = base(ticker)
    return [c for c in rows or [] if want and base(c.ticker) == want]


def default_strategy(ident: Ticker, cands: list) -> str:
    """Strategin dossiern bedömer som: den bästa kandidatradens, annars gissad ur identiteten
    (råvarudriven → BERSERK, nordisk → Viking Nine, annars Wolf)."""
    if cands:
        return cands[0].strategy
    if ident.theme:
        return "berserk"
    return "viking_nine" if ident.nordic else "wolf"


def snapshot(df, ticker) -> tuple:
    """(ögonblicksbilden som dict | None, fel) ur en OHLCV-ram — market_data.build, inget nätverk."""
    try:
        from market_data import build
        return build(df, ticker).as_dict(), ""
    except Exception as exc:
        return None, str(exc)


def _safe(fn: Optional[Callable], arg) -> tuple:
    if fn is None or not arg:
        return None, ""
    try:
        out = fn(arg)
    except Exception as exc:
        return None, str(exc)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], str(out[1] or "")
    return out, ""


def cycle(ident: Ticker, strategy: str, board: Optional[dict] = None, phase_fn: Optional[Callable] = None) -> dict:
    """Aktiens egen psykologifas (14 faser), hemmamarknadens, Valhall-segmentets (eller temats) läge och
    playbookens köp/håll/sälj-faser som PASS/MANUAL/FAIL. phase_fn(ticker) → ({phase, confidence}, fel);
    standard är cycle.market_phase (Market Cycle Engine, cachad en timme)."""
    import cycle as cy
    from valhall import cycle as vcy
    if phase_fn is None:
        phase_fn = cy.market_phase
    s = reg.get(strategy)
    pb = (s.playbook if s else "") or ""
    own, err = _safe(phase_fn, ident.yf)
    bench, berr = _safe(phase_fn, ident.home_index)
    seg: dict = {}
    inner = (board or {}).get("board") if isinstance((board or {}).get("board"), dict) else board
    if inner and (ident.segment or ident.theme):
        from valhall.paper import cycle_tag
        tag = cycle_tag(inner, ident.segment, ident.theme)
        g = "t" if ident.theme and tag.get("t_status") not in (None, vcy.NODATA) else "n" if ident.nordic else "g"
        seg = {"view": {"t": "temat", "n": "Norden", "g": "globalt"}[g], "status": tag.get(f"{g}_status"),
               "phase": tag.get(f"{g}_phase"), "cykel": tag.get(f"{g}_cykel"), "all": tag}
    if pb:
        sets = cy.playbook_phases(pb)
        status, note = cy.gate_from_market_phase(own, pb)
    else:
        sets, (status, note) = {"buy": set(), "hold": set(), "sell": set()}, ("MANUAL", NO_PLAYBOOK)
    return {"phase": own, "error": err, "benchmark": {"index": ident.home_index, "phase": bench, "error": berr},
            "segment": seg, "playbook": pb, "phases": {k: sorted(v) for k, v in sets.items()},
            "gate": {"status": status, "note": note}}


def sheets_for(ticker, stores: Optional[dict]) -> dict:
    """Vad arken säger om bolaget, bara läsning: granskningarnas utfall (review_link, PASS/MANUAL/FAIL),
    Durrett/Wolf Asymmetry-raden (confidence), scorecardets kort, registrets positioner (öppna och
    stängda), journalens affärer och Copilotens anteckningar. stores = {arknamn: data} ur
    storage.session_load — aldrig modulernas _load()."""
    import review_link
    from positions import BUCKET_KEYS, normalize_row
    stores = stores or {}
    fs = forms(ticker)
    fset = set(fs)
    full = {name: stores.get(name) or review_link.STORE_DEFAULTS[name] for name in review_link.STORE_DEFAULTS}
    reviews: dict = {}
    for key in review_link.SHEET:
        for f in fs:
            rev = review_link.review(key, f, full)
            if rev and rev.get("found"):
                reviews[key] = rev
                break
    comps = (stores.get("confidence") or {}).get("companies") or {}
    conf = next((dict(comps[f]) for f in fs if isinstance(comps.get(f), dict)), None)
    cards = {k: v for k, v in ((stores.get("scorecard") or {}).get("cards") or {}).items()
             if any(str(k).endswith("|" + f) for f in fs)}

    def mine(r) -> bool:
        return isinstance(r, dict) and str(r.get("ticker") or "").strip().upper() in fset

    hold = stores.get("holdings") or {}
    open_rows = [normalize_row(r, b) for b in BUCKET_KEYS for r in (hold.get(b) or []) if mine(r)]
    closed = [normalize_row(r) for r in (hold.get("closed") or []) if mine(r)]
    journal = [t for t in (stores.get("journal") or []) if mine(t)]
    copilot = [e for e in ((stores.get("copilot_journal") or {}).get("entries") or []) if mine(e)]
    return {"reviews": reviews, "confidence": conf, "scorecard": cards, "positions": open_rows, "closed": closed,
            "journal": journal, "copilot": copilot, "found": bool(reviews or conf or cards or open_rows)}


def assemble(raw, rows: Optional[list] = None, board: Optional[dict] = None, stores: Optional[dict] = None,
             df=None, api=None, sheets: Optional[dict] = None, phase_fn: Optional[Callable] = None,
             strategy: Optional[str] = None, table: Optional[dict] = None) -> dict:
    """Dossierns data i ett dict: ticker (identiteten), strategy, candidates, snapshot, cycle, sheets.
    Allt valfritt — det som inte skickas in blir None/tomt, aldrig ett fel."""
    sh = sheets if sheets is not None else stores
    ident = identify(raw, api, sh, table)
    cands = candidates_for(rows or [], ident.yf)
    if "." not in ident.yf and cands and "." in cands[0].ticker:      # 'eric b' → listan vet att det är ERIC-B.ST
        ident = identify(cands[0].ticker, api, sh, table)
    strategy = strategy or default_strategy(ident, cands)
    snap, snap_err = snapshot(df, ident.yf) if df is not None else (None, "")
    return {"ticker": ident.as_dict(), "strategy": strategy, "strategy_label": reg.label(strategy),
            "candidates": [c.as_dict() for c in cands], "snapshot": snap, "snapshot_error": snap_err,
            "cycle": cycle(ident, strategy, board, phase_fn) if (phase_fn is not None or board) else None,
            "sheets": sheets_for(ident.yf, stores)}
