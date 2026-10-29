"""
universe/segment_top.py — de största bolagen MED kassaflöde per Valhall-segment och marknad (Börsdata).

Thors segmentval visar segmentets största bolag även när ingen strategi har en signal (thor/segment_top.py).
Temana har sina listor i universe/theme_members.py; för de vanliga segmenten (Teknik, Hälsovård, Industri …)
räknas listan här, varje måndag i Universum-registrets jobb (universe_scan.py), och sparas som segment_top.json:

  rank_segments  Börsdatas instrument (nordiskt + globalt): segmentet ur sektor/bransch
                 (valhall.segments.segment_for — Fastighet är bransch 75–76), marknaderna Norden, USA och Kanada
                 på huvudlistorna (universe.scan.MAIN_BOARD_IDS — inte First North, TSX Venture eller OTC),
                 kassaflöde från löpande verksamhet (KPI 62) > 0, ett aktieslag per bolag
                 (valhall.picks.company_key; B-aktien och huvudnoteringen visas, inte A-aktien eller en
                 valutalinje), störst börsvärde (KPI 50) i USD — topp 10 per segment och marknad.
  theme_tops     temaprobens urval (universe/theme_probe.py) rangordnat på samma sätt — topp 20 per tema.
  build          de fyra KPI-anropen och bloben. probe = samma tabeller i borsdata_probe (PROBE_SECTION=sectors),
                 med bolagets land så att ADR:er syns.

Ren Python; Börsdata-klienten skickas in. Läser — ändrar inga skanningar och inga regler.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from universe import registry as ur

BLOB = "segment_top.json"
TOP_N = 10                                    # per segment och marknad
THEME_N = 20                                  # per tema (samma tak som temaproben)
MARKETS = (ur.NORDEN, ur.USA, ur.KANADA)      # bara dessa räknas (Nordnet handlar dem, Börsdata har dem)
SOURCE = "Börsdata KPI 50/62"
CCY_LINES = {"SEK", "EUR", "USD", "NOK", "DKK"}   # 'NOKIA SEK' — bolagets andra valutalinje, inte huvudnoteringen


def _market(inst: dict, table: Optional[dict], main_boards: bool = True) -> Optional[str]:
    """Registrets marknad för instrumentets lista — None för OTC, index, valutor och (som förval) listor
    utanför huvudlistorna."""
    import markets as mk
    from universe import scan as uscan
    try:
        mid = int(inst.get("marketId"))
    except (TypeError, ValueError):
        return None
    if mid in ur.EXCLUDE_MARKET_IDS or not mk.is_stock(mid, table):
        return None
    if main_boards and mid not in uscan.MAIN_BOARD_IDS:
        return None
    m = mk.get(mid, table)
    return ur.market_for_country(m.country if m else "")


def _usd(inst: dict, value: float, fx: dict) -> float:
    ccy = str(inst.get("stockPriceCurrency") or inst.get("reportCurrency") or "USD").upper()
    return float(value) * fx.get(ccy, 1.0)


def _company(r: dict) -> str:
    """valhall.picks.company_key utan valutalinjen: 'Nokia SEK' och 'Nokia' är samma bolag."""
    from valhall import picks as vp
    words = vp.company_key(r).split()
    while len(words) > 1 and words[-1].upper() in CCY_LINES:
        words.pop()
    return " ".join(words)


def _class_rank(bd: str) -> int:
    """Aktieslaget att visa ur Börsdatas ticker: B (den handlade i Norden) 0, A eller ett ensamt slag 1,
    C/D/pref 2 ('SEB C' är den illikvida, 'SEB A' den handlade)."""
    parts = bd.split()
    last = parts[-1] if len(parts) > 1 else ""
    if last == "B" or bd.endswith(".B"):
        return 0
    if last in ("C", "D", "R", "PREF") or bd.endswith((".C", ".D")):
        return 2
    return 1


def _one_per_company(rows: list, n: int) -> list:
    """Bolagen störst först (bolagets största aktieslag i USD), ett aktieslag per bolag, högst n rader.
    Slaget som visas: huvudnoteringen före en valutalinje ('NOKIA SEK'), sedan B, A, C/D (_class_rank) — Börsdata
    räknar börsvärdet på varje slags egen kurs, så A- och C-aktien ser ofta större ut fast B är den handlade —
    annars det största."""
    best: dict = {}
    for usd, cls, ccy_line, r in rows:
        k = _company(r)
        size, pick = best.get(k, (0.0, None))
        pref = (ccy_line, cls, -usd, r["ticker"])
        if pick is None or pref < pick[0]:
            pick = (pref, r)
        best[k] = (max(size, usd), pick)
    ranked = sorted(best.values(), key=lambda v: (-v[0], v[1][1]["ticker"]))
    return [pick[1] for _size, pick in ranked[:n]]


def rank_segments(nordic: list, global_: list, table: Optional[dict], mcap: dict, ocf: dict, fx: dict,
                  n: int = TOP_N) -> dict:
    """{segment: {marknad: [{ticker, name, mcap_usd, ins_id}]}} — kartans segment i kartans ordning, Norden,
    USA och Kanada (tom lista när ingen finns). mcap/ocf = {insId: värde} (KPI 50/62), fx = valuta → USD."""
    import markets as mk
    from valhall import segments as sgm
    by: dict = {}
    for inst in list(nordic or []) + list(global_ or []):
        if not isinstance(inst, dict) or not inst.get("ticker") or inst.get("insId") is None:
            continue
        key = _market(inst, table)
        seg = sgm.segment_for(inst.get("sectorId"), inst.get("branchId"))
        if key not in MARKETS or not seg:
            continue
        iid = int(inst["insId"])
        o, m = ocf.get(iid), mcap.get(iid)
        if o is None or o <= 0 or m is None or m <= 0:
            continue
        usd = _usd(inst, m, fx)
        row = {"ticker": mk.to_yf(inst["ticker"], inst.get("marketId"), table), "name": str(inst.get("name") or ""),
               "mcap_usd": round(usd), "ins_id": iid}
        bd = str(inst["ticker"]).strip().upper()
        by.setdefault(seg, {}).setdefault(key, []).append((usd, _class_rank(bd), bd.split()[-1] in CCY_LINES, row))
    return {s.key: {k: _one_per_company((by.get(s.key) or {}).get(k) or [], n) for k in MARKETS}
            for s in sgm.SEGMENTS}


def theme_tops(index: dict, table: Optional[dict], mcap: dict, ocf: dict, fx: dict, n: int = THEME_N) -> dict:
    """{tema: [{ticker, name, mcap_usd}]} — temaprobens urval (panelens bolag + kandidaterna) med kassaflöde > 0,
    störst börsvärde i USD, registrets marknader utan OTC, ett aktieslag per bolag."""
    from universe import theme_probe as tp
    from valhall import picks as vp
    have = tp.panel_themes()
    out: dict = {}
    for th in tp.map_themes():
        pool = set(have.get(th, set())) | set(tp.CANDIDATES.get(th, ()))
        rows, _missing, _negative = tp.rank(pool, index, mcap, ocf, {}, fx)
        seen, picked = set(), []
        for r in rows:
            if r.get("mcap_usd") is None or _market(index.get(r["ticker"]) or {}, table, main_boards=False) is None:
                continue
            k = vp.company_key(r)
            if k in seen:
                continue
            seen.add(k)
            picked.append({"ticker": r["ticker"], "name": r["name"], "mcap_usd": round(r["mcap_usd"])})
            if len(picked) == n:
                break
        out[th] = picked
    return out


def kpis(api) -> tuple:
    """(börsvärde, kassaflöde) = ({insId: KPI 50}, {insId: KPI 62}) för nordiska och globala instrument."""
    from universe import theme_probe as tp
    mcap = {**tp._kpi(api.get_kpi_screener(tp.KPI_MCAP)), **tp._kpi(api.get_kpi_screener_global(tp.KPI_MCAP))}
    ocf = {**tp._kpi(api.get_kpi_screener(tp.KPI_OCF)), **tp._kpi(api.get_kpi_screener_global(tp.KPI_OCF))}
    return mcap, ocf


def build(api, nordic: list, global_: list, table: Optional[dict], fx_getter: Optional[Callable] = None,
          now: Optional[str] = None) -> dict:
    """Bloben: {generated, source, segments, themes}. Ett fel när Börsdata svarar tomt — då sparas ingenting
    och förra veckans lista står kvar."""
    from universe import theme_probe as tp
    mcap, ocf = kpis(api)
    if not mcap or not ocf:
        raise ValueError(f"Börsdata KPI {tp.KPI_MCAP}/{tp.KPI_OCF} svarade tomt (börsvärde {len(mcap)}, kassaflöde {len(ocf)})")
    fx = tp.fx_rates(fx_getter)
    segments = rank_segments(nordic, global_, table, mcap, ocf, fx)
    if not any(rows for per in segments.values() for rows in per.values()):
        raise ValueError("inget bolag med kassaflöde på huvudlistorna — instrumentlistorna tomma?")
    index = tp.instrument_index(nordic, global_, table)
    return {"generated": now or datetime.now(tz=timezone.utc).isoformat(), "source": SOURCE,
            "segments": segments, "themes": theme_tops(index, table, mcap, ocf, fx)}


def describe(payload: Optional[dict], top: int = 3) -> list:
    """En loggrad per segment: 'Teknik · Norden ERIC-B.ST, … · USA NVDA, …' (de top första)."""
    from valhall import segments as sgm
    out = []
    for seg, per in ((payload or {}).get("segments") or {}).items():
        s = sgm.get(seg)
        parts = [f"{ur.label(k)} " + (", ".join(r["ticker"] for r in rows[:top]) or "—") for k, rows in per.items()]
        out.append(f"{s.label if s else seg} · " + " · ".join(parts))
    return out


def probe(api, out=print, fx_getter: Optional[Callable] = None) -> dict:
    """borsdata_probe PROBE_SECTION=sectors: tabellerna per segment och marknad med bolagets land (ADR:er syns
    som USA-noteringar med ett annat land), och temanas fem största. Läser, skriver ingenting."""
    import markets as mk
    from valhall import segments as sgm
    raw_countries = api.get_countries() or []
    table = mk._merge(mk.from_api(api.get_markets(), raw_countries))
    countries = {ur._int(c.get("id")): str(c.get("name") or "") for c in raw_countries if isinstance(c, dict)}
    nordic, global_ = api.get_instruments() or [], api.get_global_instruments_list() or []
    country_of = {int(i["insId"]): countries.get(ur._int(i.get("countryId")), "?") for i in list(nordic) + list(global_)
                  if isinstance(i, dict) and i.get("insId") is not None}
    payload = build(api, nordic, global_, table, fx_getter)
    out("=" * 72)
    out("SEGMENTEN — största bolagen med kassaflöde (KPI 62 > 0) per marknad, börsvärde (KPI 50) i USD")
    out(f"  instrument: Norden {len(nordic)} · globalt {len(global_)} · huvudlistorna, ett aktieslag per bolag")
    out("=" * 72)
    for seg, per in payload["segments"].items():
        s = sgm.get(seg)
        out(f"\n── {s.label if s else seg}")
        for k, rows in per.items():
            out(f"   {ur.label(k)}: {len(rows)}")
            for i, r in enumerate(rows, 1):
                out(f"   {i:>3}. {r['ticker']:12} {r['name'][:30]:30} {country_of.get(r['ins_id'], '?')[:14]:14} "
                    f"≈{r['mcap_usd']:,.0f} MUSD")
    out("\n" + "=" * 72)
    out("TEMANA — de fem största (hela listan, topp 20, sparas i bloben)")
    out("=" * 72)
    for th, rows in payload["themes"].items():
        out(f"  {th:12} {len(rows):>2} · " + ", ".join(r["ticker"] for r in rows[:5]))
    return payload
