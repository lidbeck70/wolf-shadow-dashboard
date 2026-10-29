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
    """Dagens Thor-rader för aktien (alla strategier), i listans ordning. 'BOL' hittar 'BOL.ST'; med suffix krävs
    samma notering — EQNR.OL får inte EQNR-raden från New York (granskning 2, F10) — och samma marknad: L'Oréal
    OR.PA får inte Osiskos OR.TO (granskning 3)."""
    from universe import registry as ur
    want = base(ticker)
    t = str(ticker or "").strip().upper()
    if "." in t:
        exact = [c for c in rows or [] if str(c.ticker).upper() == t]
        if exact:
            return exact
        mkt = ur.market_of(t)
        return [c for c in rows or [] if want and base(c.ticker) == want and "." in str(c.ticker)
                and ur.market_of(c.ticker) == mkt]
    return [c for c in rows or [] if want and base(c.ticker) == want]


def bare_rows(cands: list, ticker) -> list:
    """Raderna med exakt samma bara ticker: 'TEL' i listan är TE Connectivity och vinner över Telenor TEL.OL
    (granskning 3, T1). Tom lista för en noterad ticker eller när ingen bar rad finns."""
    t = str(ticker or "").strip().upper()
    if not t or "." in t:
        return []
    return [c for c in cands or [] if str(c.ticker).upper() == t]


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


def valhall_tag(inner: Optional[dict], ident) -> dict:
    """Valhalls läge för aktien (valhall.paper.cycle_tag): Norden och globalt ur aktiens eget segment, temat ur
    temats egen segmentrad. Kartan har temat bara under sitt segment, och Börsdatas sektor kan vara en annan
    (SBUX: Sällanköp, kaffe: Dagligvaror) — då försvann temats läge (granskning 3, T8). cycle_tag ändras inte."""
    from valhall import segments as sgm
    from valhall.paper import cycle_tag
    tag = cycle_tag(inner, ident.segment, None)
    if ident.theme:
        th = cycle_tag(inner, sgm.segment_of_theme(ident.theme) or ident.segment, ident.theme)
        tag.update({k: v for k, v in th.items() if k.startswith("t_")})
    return tag


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
    home = getattr(ident, "home_label", "") or ident.home_index      # visning — grinden läser bara aktiens egen fas
    bench, berr = _safe(phase_fn, home)
    seg: dict = {}
    inner = (board or {}).get("board") if isinstance((board or {}).get("board"), dict) else board
    if inner and (ident.segment or ident.theme):
        tag = valhall_tag(inner, ident)
        g = "t" if ident.theme and tag.get("t_status") not in (None, vcy.NODATA) else "n" if ident.nordic else "g"
        seg = {"view": {"t": "temat", "n": "Norden", "g": "globalt"}[g], "status": tag.get(f"{g}_status"),
               "phase": tag.get(f"{g}_phase"), "cykel": tag.get(f"{g}_cykel"), "all": tag}
    if pb:
        sets = cy.playbook_phases(pb)
        status, note = cy.gate_from_market_phase(own, pb)
    else:
        sets, (status, note) = {"buy": set(), "hold": set(), "sell": set()}, ("MANUAL", NO_PLAYBOOK)
    return {"phase": own, "error": err, "benchmark": {"index": home, "phase": bench, "error": berr},
            "segment": seg, "playbook": pb, "phases": {k: sorted(v) for k, v in sets.items()},
            "gate": {"status": status, "note": note}}


def held_forms(ticker) -> list:
    """Formerna en innehavs-, journal- eller Copilotrad kan ha för aktien. Raderna skrivs i Yahoo-form
    (positions.add, Thor), och en bar rad är en amerikansk ticker: 'TEL' är TE Connectivity och hör aldrig till
    Telenor TEL.OL — och TEL.OL aldrig till TEL. Bara Börsdatas form med mellanslag ('ERIC B'), som ingen
    amerikansk ticker har, når en nordisk sida."""
    t = tc.yf(ticker)
    if "." not in t:
        return forms(t)
    return [t] + ([f for f in forms(t) if " " in f] if tc.is_nordic(t) else [])


def sheet_forms(ticker, sheet_map: Optional[dict] = None) -> list:
    """Formerna en arkrad kan ha för aktien. Arken sparar tickern som den skrevs: en nordisk sida tar Börsdatas
    former ('BOL', 'ERIC B', 'ERIC-B') — med registrets karta (Universe.sheet_map) bara de bara former kartan ger
    just den här noteringen ('TEL' är TE Connectivity, inte Telenor TEL.OL); formen med mellanslag är alltid
    nordisk. En utländsk notering (EQX.TO, TTE.PA) tar bara sin egen form — en bar arkrad är en amerikansk
    ticker, som i listan (candidates.sheet_yf)."""
    t = tc.yf(ticker)
    if "." not in t:
        return forms(t)
    if not tc.is_nordic(t):
        return [t]
    ctx = {"sheet_map": {str(k).strip().upper(): v for k, v in (sheet_map or {}).items() if v}}
    return [f for f in forms(t) if not ctx["sheet_map"] or f == t or " " in f or tc.sheet_yf(f, ctx) == t]


def _sheet_rows(fs: list, stores: dict, full: dict) -> tuple:
    """(granskningarna, Confidence-raden, scorecardets kort, {form: [ark]}) för arkformerna fs — formen raden
    matchades på följer med, så att dossiern kan säga vilka rader som står på ett kortnamn utan börssuffix."""
    import review_link
    reviews: dict = {}
    used: dict = {}
    for key in review_link.SHEET:
        for f in fs:
            rev = review_link.review(key, f, full)
            if rev and rev.get("found"):
                reviews[key] = rev
                used.setdefault(f, []).append(str(rev.get("sheet") or key))
                break
    comps = (stores.get("confidence") or {}).get("companies") or {}
    hit = next((f for f in fs if isinstance(comps.get(f), dict)), None)
    if hit is not None:
        used.setdefault(hit, []).append("Durrett/Confidence")
    cards: dict = {}
    for k, v in ((stores.get("scorecard") or {}).get("cards") or {}).items():
        f = next((f for f in fs if str(k).endswith("|" + f)), None)
        if f is not None:
            cards[k] = v
            used.setdefault(f, []).append("Scorecardet")
    return reviews, (dict(comps[hit]) if hit is not None else None), cards, used


def _tk(r) -> str:
    return str(r.get("ticker") or "").strip().upper() if isinstance(r, dict) else ""


def sheets_for(ticker, stores: Optional[dict], sheet_map: Optional[dict] = None) -> dict:
    """Vad arken säger om bolaget, bara läsning: granskningarnas utfall (review_link, PASS/MANUAL/FAIL),
    Durrett/Wolf Asymmetry-raden (confidence), scorecardets kort, registrets positioner (öppna och
    stängda), journalens affärer och Copilotens anteckningar. stores = {arknamn: data} ur
    storage.session_load — aldrig modulernas _load(). Arken matchas på sheet_forms (sheet_map = registrets
    karta för arkens bara nordiska former), innehav/journal/Copilot på held_forms: en bar 'TEL' hamnar aldrig
    på Telenor TEL.OL.

    Inget försvinner tyst: elsewhere = rader på aktiens kortnamn som räknas som en annan notering ('BOL' i
    innehaven, 'EQX' i Rick Rule på EQX.TO), ambiguous = arkens bara former en nordisk sida matchade utan
    registrets karta — då går 'TEL' inte att skilja från Telenor."""
    import review_link
    from positions import BUCKET_KEYS, normalize_row
    stores = stores or {}
    fs = sheet_forms(ticker, sheet_map)
    fset = set(held_forms(ticker))
    full = {name: stores.get(name) or review_link.STORE_DEFAULTS[name] for name in review_link.STORE_DEFAULTS}
    reviews, conf, cards, used = _sheet_rows(fs, stores, full)
    away = [f for f in forms(ticker) if f not in fs]          # kortnamnet räknas som en annan notering här
    elsewhere = [f"{place} '{f}'" for f, places in (_sheet_rows(away, stores, full)[3] if away else {}).items()
                 for place in places]
    ambiguous = ([f for f in used if "." not in f and " " not in f]       # utan kartan kan 'TEL' vara TE Connectivity
                 if tc.is_nordic(tc.yf(ticker)) and not any((sheet_map or {}).values()) else [])
    entries: dict = {}
    if cards:                                  # kortens kandidatrader — samma som 🎯 Scorecard räknar grinden på
        try:
            import scorecard
            entries = scorecard.entries_for(fs, stores)
        except Exception:
            entries = {}

    hold = stores.get("holdings") or {}
    held = ([("innehav", b, r) for b in BUCKET_KEYS for r in (hold.get(b) or [])]
            + [("stängd position", None, r) for r in (hold.get("closed") or [])]
            + [("journalen", None, t) for t in (stores.get("journal") or [])]
            + [("Copiloten", None, e) for e in ((stores.get("copilot_journal") or {}).get("entries") or [])])
    mine = [(place, b, r) for place, b, r in held if _tk(r) in fset]   # Yahoo-form, aldrig den bara formen
    open_rows = [normalize_row(r, b) for place, b, r in mine if place == "innehav"]
    closed = [normalize_row(r) for place, _b, r in mine if place == "stängd position"]
    journal = [r for place, _b, r in mine if place == "journalen"]
    copilot = [r for place, _b, r in mine if place == "Copiloten"]
    other = set(forms(ticker)) - fset                         # 'BOL' i innehaven är en amerikansk ticker i panelen
    elsewhere += list(dict.fromkeys(f"{place} '{_tk(r)}'" for place, _b, r in held if _tk(r) in other))
    return {"reviews": reviews, "confidence": conf, "scorecard": cards, "positions": open_rows, "closed": closed,
            "journal": journal, "copilot": copilot, "found": bool(reviews or conf or cards or open_rows),
            "rotation": stores.get("rotation"), "confidence_store": stores.get("confidence"),
            "scorecard_entries": {k: entries.get(k) for k in cards}, "sheet_forms": fs,
            "elsewhere": elsewhere, "ambiguous": ambiguous}


def assemble(raw, rows: Optional[list] = None, board: Optional[dict] = None, stores: Optional[dict] = None,
             df=None, api=None, sheets: Optional[dict] = None, phase_fn: Optional[Callable] = None,
             strategy: Optional[str] = None, table: Optional[dict] = None,
             sheet_map: Optional[dict] = None) -> dict:
    """Dossierns data i ett dict: ticker (identiteten), strategy, candidates, snapshot, cycle, sheets.
    Allt valfritt — det som inte skickas in blir None/tomt, aldrig ett fel. sheet_map = registrets karta för
    arkens bara nordiska former (Universe.sheet_map), samma som sidan skickar till sheets_for."""
    sh = sheets if sheets is not None else stores
    ident = identify(raw, api, sh, table)
    cands = candidates_for(rows or [], ident.yf)
    exact = bare_rows(cands, ident.yf)
    if exact:                                                          # 'TEL' = TE Connectivity, inte TEL.OL (T1)
        cands = exact
    elif "." not in ident.yf and cands and "." in cands[0].ticker:    # 'eric b' → listan vet att det är ERIC-B.ST
        ident = identify(cands[0].ticker, api, sh, table)
    strategy = strategy or default_strategy(ident, cands)
    snap, snap_err = snapshot(df, ident.yf) if df is not None else (None, "")
    return {"ticker": ident.as_dict(), "strategy": strategy, "strategy_label": reg.label(strategy),
            "candidates": [c.as_dict() for c in cands], "snapshot": snap, "snapshot_error": snap_err,
            "cycle": cycle(ident, strategy, board, phase_fn) if (phase_fn is not None or board) else None,
            "sheets": sheets_for(ident.yf, stores, sheet_map)}
