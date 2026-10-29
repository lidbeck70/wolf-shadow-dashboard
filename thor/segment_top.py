"""
thor/segment_top.py — "Största i segmentet": segmentets största bolag i Thors lista, även utan signal.

Skannrarna sparar bara sina träffar (BERSERK KÖP/BEVAKA, EMBER KÖPLÄGE/BEVAKA …), så ett tema där ingen strategi
har en signal i dag visade 0 rader (uran, palladium). Väljer du ett segment, ett tema eller ett komplex står nu de
största bolagen med positivt kassaflöde under de riktiga raderna, som INGEN SIGNAL (tc.INGEN, radslaget tc.TOP).
Raderna ritas bara i listan (thor.ui._list) och hamnar aldrig i data["rows"]: Kandidater-rutan, candidates_for,
default_strategy och 'Bedöm som' räknar bara strategiernas egna rader. Inget köp, ingen entry, inget stopp.

  tema      universe/theme_members.py (Börsdata: kassaflöde > 0, störst börsvärde) — eller bloben segment_top.json
            när den har temat (universe/segment_top.py, Universum-registret varje måndag)
  komplex   komplexets teman i kartans ordning, varvet runt: #1 i varje tema, sedan #2 … ('STÖRST #1 i Olja')
  segment   bloben: 5 Norden + 5 USA, 10 på en marknad när en marknad (eller 'bara Norden') är vald, delat lika
            på flera. Utan blob tar Energi, Material och Dagligvaror sina teman som ett komplex; övriga segment
            får en rad om att listan byggs av Universum-registret.
Ett bolag med en rad i Screening-valet visas med sin riktiga rad, inte här. 'bara KÖP' och 'bara nya' döljer
raderna; 'bara Norden' och marknaderna gäller. Ren logik — inga nätanrop, inga regler.
"""

from __future__ import annotations

from typing import Optional

from thor import candidates as tc
from thor import registry as reg
from universe import registry as ur

N = 10                                              # högst så många rader
HIDE = frozenset({"bara KÖP", "bara nya"})          # filtren som döljer raderna (de är aldrig köp, aldrig nya)
THIN = 5                                            # färre bolag än så = tunt tema (universe.theme_probe.MIN_N)


def ordered_themes(themes) -> list:
    """Temana i kartans ordning (valhall.segments) — komplexens teman är en mängd utan fast ordning."""
    from valhall import segments as sgm
    want = set(themes or ())
    return [th for s in sgm.SEGMENTS for th in s.themes if th in want]


def wording(family: Optional[str] = None, strategy: Optional[str] = None) -> str:
    """'Ingen signal i dag' — 'Ingen BERSERK-signal i dag' när Screening-valet är en strategi (eller en familj)."""
    if strategy:
        return f"Ingen {reg.label(strategy)}-signal i dag"
    if family:
        return f"Ingen {reg.FAMILY_LABEL.get(family, family)}-signal i dag"
    return "Ingen signal i dag"


def choice_label(choice: str) -> str:
    """Valets namn i klartext: 'Palladium', 'komplexet Ädelmetaller', 'Hälsovård'."""
    from berserk import themes as bth
    from ember.config import COMPLEX_LABEL
    from thor import ui as tui
    from valhall import segments as sgm
    s = str(choice or "")
    if s.startswith(tui.THEME_PREFIX):
        return bth.label(s[len(tui.THEME_PREFIX):])
    if s.startswith(tui.COMPLEX_PREFIX):
        cx = s[len(tui.COMPLEX_PREFIX):]
        return f"komplexet {str(COMPLEX_LABEL.get(cx, cx)).capitalize()}"
    seg = sgm.get(s)
    return seg.label if seg else s


def title(choice: str, pick: str) -> str:
    """Avsnittets rubrik: 'STÖRST I PALLADIUM — INGEN SIGNAL I DAG'."""
    from thor import ui as tui
    return f"STÖRST I {choice_label(choice).upper()} — {wording(*tui.pick_to_filter(pick)).upper()}"


def market_split(keys, nordic_only: bool = False, n: int = N) -> list:
    """[(marknad, antal)] för sektorlistan: 5 Norden + 5 USA som förval, n på en vald marknad (eller 'bara
    Norden'), delat lika på flera — bara marknaderna bloben räknar (Norden, USA, Kanada)."""
    from universe import segment_top as ust
    want = set(keys or ())
    chosen = [m.key for m in ur.MARKETS if m.key in want] or None
    if nordic_only:
        chosen = [ur.NORDEN] if chosen is None or ur.NORDEN in chosen else []
    if chosen is None:
        return [(ur.NORDEN, n - n // 2), (ur.USA, n // 2)]
    chosen = [k for k in chosen if k in ust.MARKETS]
    if not chosen:
        return []
    base, rest = divmod(n, len(chosen))
    return [(k, base + (1 if i < rest else 0)) for i, k in enumerate(chosen)]


def _date(blob) -> str:
    return str(blob.get("generated") or "")[:10] if isinstance(blob, dict) else ""


def theme_list(theme: str, blob: Optional[dict] = None) -> tuple:
    """(temats bolag störst först som [{ticker, name, mcap_usd}], datum) — bloben när den har temat, annars
    temalistan (universe/theme_members.py)."""
    themes = blob.get("themes") if isinstance(blob, dict) else None
    rows = themes.get(theme) if isinstance(themes, dict) else None
    rows = [r for r in rows if isinstance(r, dict) and r.get("ticker")] if isinstance(rows, list) else []
    if rows:
        return rows, _date(blob)
    from universe import theme_members as tm
    return [{"ticker": t} for t in tm.members(theme)], tm.PROBE_DATE


def _ok(ticker, keys: set, nordic_only: bool) -> bool:
    """'bara Norden' och marknaderna (registrets nycklar; tomt = alla) — samma filter som listans rader."""
    if nordic_only and not tc.is_nordic(ticker):
        return False
    return not keys or ur.market_of(ticker) in keys


def _round_robin(themes, blob, keys: set, nordic_only: bool, names: Optional[dict], n: int = N) -> tuple:
    """([bolag], datum) varvet runt över temana i kartans ordning: #1 i varje tema, sedan #2 … — ett bolag en gång
    (samma ticker eller samma bolagsnamn: HBM och HBM.TO)."""
    from berserk import themes as bth
    from valhall import picks as vp
    lists, asof = [], ""
    for th in ordered_themes(themes):
        rows, d = theme_list(th, blob)
        asof = max(asof, d)
        lists.append([{**r, "rank": i, "theme": th, "where": bth.label(th)} for i, r in enumerate(rows, 1)
                      if _ok(tc.yf(r["ticker"]), keys, nordic_only)])
    out, seen = [], set()
    for i in range(max((len(x) for x in lists), default=0)):
        for lst in lists:
            if i >= len(lst) or len(out) == n:
                continue
            t = tc.yf(lst[i]["ticker"])
            k = vp.company_key({"name": lst[i].get("name") or (names or {}).get(t, ""), "ticker": t})
            if t in seen or k in seen:
                continue
            seen.update((t, k))
            out.append(lst[i])
    return out, asof


def top_list(choice: str, blob: Optional[dict] = None, keys=(), nordic_only: bool = False,
             names: Optional[dict] = None, n: int = N) -> tuple:
    """(bolagen [{ticker, name, mcap_usd, rank, theme, segment, where}], noteringar, datum) för segmentvalet —
    störst först, filtrerade på 'bara Norden' och marknaderna. where = temat (komplex) eller marknaden (sektor)."""
    from berserk import themes as bth
    from thor import ui as tui
    from valhall import segments as sgm
    choice, keys = str(choice or ""), set(keys or ())
    notes: list = []
    where_txt = "nordiska bolag" if nordic_only and not keys else "bolag på vald marknad"
    if choice.startswith(tui.THEME_PREFIX):
        th = choice[len(tui.THEME_PREFIX):]
        full, asof = theme_list(th, blob)
        out = [{**r, "rank": i, "theme": th, "where": ""} for i, r in enumerate(full, 1)
               if _ok(tc.yf(r["ticker"]), keys, nordic_only)][:n]
        if 0 < len(full) < THIN:
            notes.append(f"{bth.label(th)} har bara {len(full)} noterade bolag med positivt kassaflöde i Börsdata "
                         "— resten prospekterar.")
        if full and not out:
            notes.append(f"Inga {where_txt} bland de största i {bth.label(th)}.")
        return out, notes, asof
    themes = tui._segment_themes(choice)
    if themes is not None:                                           # komplex
        out, asof = _round_robin(themes, blob, keys, nordic_only, names, n)
        if not out:
            notes.append(f"Inga {where_txt} bland de största i {choice_label(choice)}.")
        return out, notes, asof
    segs = blob.get("segments") if isinstance(blob, dict) else None
    per = segs.get(choice) if isinstance(segs, dict) else None
    if isinstance(per, dict):                                        # sektorlistan ur Universum-registret
        from universe import segment_top as ust
        out = []
        for mkey, k in market_split(keys, nordic_only, n):
            rows = [r for r in (per.get(mkey) or []) if isinstance(r, dict) and r.get("ticker")]
            out += [{**r, "rank": i, "segment": choice, "where": ur.label(mkey)} for i, r in enumerate(rows[:k], 1)]
        if keys - set(ust.MARKETS):
            notes.append("Sektorlistan räknar bara Norden, USA och Kanada.")
        if not out:
            notes.append(f"Inga {where_txt} med positivt kassaflöde i sektorlistan.")
        return out, notes, _date(blob)
    seg = sgm.get(choice)
    if seg and seg.themes:                                           # utan blob: segmentets teman som ett komplex
        out, asof = _round_robin(seg.themes, blob, keys, nordic_only, names, n)
        notes.append("Sektorlistan saknas än (Universum-registret, måndagar) — här står temanas största.")
        return out, notes, asof
    notes.append("Listan över segmentets största bolag saknas än — den byggs av Universum-registret (måndagar). "
                 "Kör universe-scan i GitHub Actions för att få den nu.")
    return [], notes, ""


def _candidate(e: dict, word: str, asof: str, names: Optional[dict], ctx: dict) -> tc.Candidate:
    """En rad 'Största i segmentet': INGEN SIGNAL, rang i statusen, börsvärdet som siffra, kartans läge."""
    from valhall import segments as sgm
    t = tc.yf(e["ticker"])
    th = e.get("theme") or None
    mc = tc._f(e.get("mcap_usd"))
    where = e.get("where") or ""
    c = tc.Candidate(
        t, str(e.get("name") or (names or {}).get(t) or ""), tc.TOP,
        f"STÖRST #{e['rank']}" + (f" i {where}" if where else ""), tc.INGEN, score=None,
        score_text=f"≈ {mc:,.0f} MUSD börsvärde".replace(",", " ") if mc else "",
        segment=(sgm.segment_of_theme(th) if th else None) or e.get("segment"), theme=th, nordic=tc.is_nordic(t),
        asof=asof, why=f"{word} · störst börsvärde med positivt kassaflöde (Börsdata {asof})")
    return tc._tag(c, ctx)


def top_rows(choice: str, rows: list, pick: str, filters=(), markets=(), board: Optional[dict] = None,
             blob: Optional[dict] = None, names: Optional[dict] = None, shown: Optional[list] = None) -> tuple:
    """(rader, notering) för listans avsnitt 'Största i …'. Tomt och '' när avsnittet inte visas ('Alla segment',
    'bara KÖP', 'bara nya'). rows = data["rows"] (rörs inte), shown = listans synliga rader (thor.ui.shown_rows).
    Ett bolag med en rad i Screening-valet tas inte med — noteringen säger att det har signal ovan."""
    from thor import ui as tui
    flt = set(filters or [])
    if choice in (None, "", tui.SEGMENT_ALL) or flt & HIDE:
        return [], ""
    family, strategy = tui.pick_to_filter(pick)
    keys = {tui.MARKET_LABELS.get(m, m) for m in (markets or [])}
    entries, notes, asof = top_list(choice, blob, keys, "bara Norden" in flt, names)
    if shown is None:
        shown = tui.shown_rows(rows or [], pick, filters, markets, choice)
    have = {str(c.ticker).upper() for c in tc.filter_rows(rows or [], family=family, strategy=strategy)}
    visible = {str(c.ticker).upper() for c in shown or []}
    word, ctx = wording(family, strategy), tc.segment_context(board)
    out, above, other = [], [], []
    for e in entries:
        t = tc.yf(e["ticker"])
        if t in have:
            (above if t in visible else other).append(t)
            continue
        out.append(_candidate(e, word, asof, names, ctx))
    if above:
        notes.append(f"Har signal ovan: {', '.join(above)}.")
    if other:
        notes.append(f"Har en rad i listan men utanför det här urvalet: {', '.join(other)}.")
    if entries:
        notes.append(f"Störst på börsvärde med positivt kassaflöde (Börsdata {asof}) — ingen köpsignal, "
                     "tryck Öppna → för analysen.")
    return out, " ".join(notes)
