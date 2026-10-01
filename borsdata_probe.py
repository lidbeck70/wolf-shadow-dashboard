#!/usr/bin/env python3
"""
borsdata_probe.py — engångsdiagnostik mot Börsdata (körs i Actions där nyckeln finns).

Skriver ut det som behövs för att rätta Contrarian Alpha-screenern på fakta
i stället för gissningar:
  1. Börsdatas sektor- och branschlista (id → namn) + antal nordiska bolag
     per bransch — underlaget för necessity-namnkartan.
  2. KPI-metadata (id → namn) för de KPI:er motorn använder — bekräftar att
     rätt id går till rätt fält.
  3. Råa screenervärden för referensbolag (G5EN, QAIR, BOL, EQNR) — så
     enheterna (procent vs kvot, MSEK vs per aktie) kan fastställas mot
     kända siffror.
  4. Råvaror: alla instrument på Börsdatas råvaru-/valutamarknader (76 Forex,
     77 Nymex och andra marknader av typen "other") med antal år prishistorik,
     plus en namnsökning efter uran, kol, zink, sällsynta metaller m.fl.
     PROBE_SECTION=commodities kör bara den delen (snabbt).

Skriver ingenting, ändrar ingenting. Exit 0 alltid.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

REF_TICKERS = ["G5EN", "QAIR", "BOL", "EQNR"]
PROBE_KPIS = {
    "ebitda_margin": 32, "operating_margin": 29, "gross_margin": 28,
    "fcf_margin": 31, "roic": 37, "roe": 33, "debt_to_equity": 40,
    "equity_ratio": 39, "net_debt_ebitda": 42, "ev_ebitda": 11, "pe": 2,
    "p_fcf": 76, "fcf_m": 63, "revenue_m": 53, "ebitda_m": 54,
    "total_equity_m": 58, "total_assets_m": 57, "market_cap": 50,
    "net_debt_m": 60, "short_selling": 207,
}

COMMODITY_MARKET_IDS = (76, 77)
# Namnsökning (engelska och svenska) — vad finns som råvara hos Börsdata?
COMMODITY_KEYWORDS = ("uran", "coal", "kol", "zinc", "zink", "rare", "sällsynt",
                      "lithium", "litium", "nickel", "cobalt", "kobolt", "alumin",
                      "iron", "järn", "tin", "lead", "bly", "gold", "guld", "silver",
                      "platin", "pallad", "copper", "koppar", "oil", "olja", "brent",
                      "gas", "wheat", "vete", "coffee", "kaffe", "cocoa", "kakao",
                      "lumber", "timber", "sugar", "corn", "soy", "molyb")
MAX_PRICE_PROBES = 120


def _history(api, ins_id) -> str:
    """Antal prisrader, första/sista datum och senaste pris (upp till 20 år)."""
    try:
        rows = api.get_stockprices(ins_id, max_count=5040)
    except Exception as e:
        return f"pris FEL: {str(e)[:60]}"
    if not rows:
        return "pris: inga rader"
    dates = sorted(str(r.get("d", ""))[:10] for r in rows if r.get("d"))
    if not dates:
        return f"pris: {len(rows)} rader utan datum"
    last = max(rows, key=lambda r: str(r.get("d", "")))
    try:
        years = (int(dates[-1][:4]) - int(dates[0][:4])) + (int(dates[-1][5:7]) - int(dates[0][5:7])) / 12
    except ValueError:
        years = 0.0
    return (f"pris: {len(rows)} rader {dates[0]} → {dates[-1]} (≈{years:.1f} år) "
            f"senast={last.get('c')}")


def _keyword_hits(name: str) -> list:
    low = (name or "").lower()
    words = set(low.replace("-", " ").replace("/", " ").split())
    hits = []
    for k in COMMODITY_KEYWORDS:
        # korta ord (uran, kol, tin, gas) måste inleda ett ord — "Uranium" ja, "Petrol" nej
        if (len(k) <= 4 and any(w.startswith(k) for w in words)) or (len(k) > 4 and k in low):
            hits.append(k)
    return hits


def probe_commodities(api, out=print) -> list:
    """
    Listar Börsdatas råvaru-/valutainstrument med prishistorik. Rent
    läsande. Returnerar raderna (dict) så att testet kan granska urvalet.
    """
    from markets import OTHER, from_api
    out("\n" + "=" * 72)
    out("RÅVAROR — marknader, instrument och prishistorik")
    out("=" * 72)
    try:
        markets = from_api(api.get_markets(), api.get_countries())
    except Exception as e:
        out(f"  marknader FEL: {e}")
        markets = {}
    for mid, m in sorted(markets.items()):
        out(f"  marknad {mid:>4}  {m.name:28} land={m.country or '-':12} typ={m.kind}"
            f"{'  ← råvaror/valuta' if m.kind == OTHER else ''}")
    target = set(COMMODITY_MARKET_IDS) | {mid for mid, m in markets.items() if m.kind == OTHER}
    out(f"\n  Råvaru-/valutamarknader som undersöks: {sorted(target)}")

    rows, seen = [], set()
    for source, getter in (("nordisk", api.get_instruments), ("global", api.get_global_instruments_list)):
        try:
            lst = getter() or []
        except Exception as e:
            out(f"  {source} instrumentlista FEL: {e}")
            continue
        out(f"\n  {source}: {len(lst)} instrument; typer (instrumentType → antal): "
            f"{dict(Counter(i.get('instrumentType') for i in lst).most_common())}")
        per_market = Counter(i.get("marketId") for i in lst if i.get("marketId") in target)
        out(f"  {source}: på råvarumarknaderna: {dict(per_market)}")
        for i in lst:
            iid = i.get("insId")
            if iid in seen:
                continue
            on_market = i.get("marketId") in target
            if on_market:
                seen.add(iid)
                rows.append({"source": source, "insId": iid, "name": i.get("name", ""),
                             "ticker": i.get("ticker", ""), "marketId": i.get("marketId"),
                             "type": i.get("instrumentType"), "hits": _keyword_hits(i.get("name", ""))})

    out(f"\n  INSTRUMENT PÅ RÅVARU-/VALUTAMARKNADERNA ({len(rows)} st):")
    for n, r in enumerate(rows):
        hist = _history(api, r["insId"]) if n < MAX_PRICE_PROBES else "pris: ej hämtat (tak)"
        r["history"] = hist
        out(f"   {r['insId']:>7} m={r['marketId']!s:>3} typ={r['type']!s:>3} {r['ticker'][:12]:12} "
            f"{r['name'][:34]:34} {hist}")

    out("\n  NAMNSÖKNING bland dem (uran, kol, zink, sällsynta m.fl.):")
    for k in ("uran", "coal", "kol", "zinc", "zink", "rare", "sällsynt", "lithium", "nickel", "cobalt"):
        found = [f"{r['ticker'] or r['name']} ({r['insId']})" for r in rows if k in r["hits"]]
        out(f"   {k:10} → {', '.join(found) if found else 'INGEN TRÄFF'}")
    return rows


def main() -> int:
    if not os.environ.get("BORSDATA_API_KEY"):
        print("BORSDATA_API_KEY saknas — inget att göra.")
        return 0
    from borsdata_api import BorsdataAPI, ALL_NORDIC_MARKETS
    api = BorsdataAPI()

    if os.environ.get("PROBE_SECTION", "all") == "commodities":
        probe_commodities(api)
        return 0

    print("=" * 72)
    print("SEKTORER (id → namn)")
    print("=" * 72)
    sectors = {s["id"]: s.get("name", "") for s in api.get_sectors()}
    for sid, name in sorted(sectors.items()):
        print(f"  {sid:>4}  {name}")

    instruments = api.get_instruments()
    nordic = [i for i in instruments
              if i.get("marketId") in ALL_NORDIC_MARKETS
              and i.get("instrumentType", 1) in (1, None)]
    per_branch = Counter(i.get("branchId") for i in nordic)
    branch_sector: dict = {}
    for i in nordic:
        branch_sector.setdefault(i.get("branchId"), i.get("sectorId"))

    print("\n" + "=" * 72)
    print(f"BRANSCHER (id → namn · sektor · antal nordiska bolag av {len(nordic)})")
    print("=" * 72)
    for b in sorted(api.get_branches(), key=lambda x: x["id"]):
        bid = b["id"]
        sname = sectors.get(b.get("sectorId", branch_sector.get(bid)), "?")
        print(f"  {bid:>4}  {b.get('name', ''):40} sektor={sname:22} n={per_branch.get(bid, 0)}")

    print("\n" + "=" * 72)
    print("KPI-METADATA (id → namn) för motorns KPI:er")
    print("=" * 72)
    try:
        meta = api._get("/instruments/kpis/metadata")
        by_id = {m.get("kpiId"): m for m in meta.get("kpiHistoryMetadatas", [])}
        for key, kid in PROBE_KPIS.items():
            m = by_id.get(kid, {})
            print(f"  {kid:>4}  {key:18} → {m.get('nameSv') or m.get('nameEn', '?')}"
                  f"  [{m.get('format', '')}]")
    except Exception as e:
        print(f"  metadata misslyckades: {e}")

    print("\n" + "=" * 72)
    print("REFERENSBOLAG — råa screenervärden (last/latest)")
    print("=" * 72)
    refs = {}
    for i in instruments:
        t = str(i.get("ticker", "")).upper()
        if t in REF_TICKERS:
            refs[t] = i
    for t, i in refs.items():
        print(f"\n  {t}: {i.get('name')}  insId={i.get('insId')}  "
              f"sectorId={i.get('sectorId')} ({sectors.get(i.get('sectorId'), '?')})  "
              f"branchId={i.get('branchId')}  marketId={i.get('marketId')}")
    ids = {i.get("insId"): t for t, i in refs.items()}
    for key, kid in PROBE_KPIS.items():
        try:
            vals = api.get_kpi_screener(kid, "last", "latest")
        except Exception as e:
            print(f"  {key:18} (id {kid}): FEL {e}")
            continue
        row = {ids[v.get('i')]: v.get("n") for v in vals if v.get("i") in ids}
        cells = "  ".join(f"{t}={row.get(t)}" for t in REF_TICKERS)
        print(f"  {key:18} (id {kid:>3}): {cells}")

    # ── Blankningsregistret: rå form ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("HOLDINGS/SHORTS — rå payload (scheduled-scan loggade 0 instrument)")
    print("=" * 72)
    try:
        raw = api._get("/holdings/shorts")
        print(f"  typ={type(raw).__name__} nycklar={list(raw)[:8] if isinstance(raw, dict) else '-'}")
        txt = json.dumps(raw, ensure_ascii=False)
        print(f"  längd={len(txt)} tecken; början: {txt[:600]}")
        print(f"  get_short_positions() → {len(api.get_short_positions())} instrument")
    except Exception as e:
        print(f"  FEL: {e}")

    # ── Batch-historik och insiders: rå form ───────────────────────────────
    print("\n" + "=" * 72)
    print("KPI-HISTORIK BATCH (ev_ebitda, BOL+EQNR) och INSIDERS (BOL) — rå form")
    print("=" * 72)
    try:
        ids = [i.get("insId") for t, i in refs.items() if t in ("BOL", "EQNR")]
        bol_eqnr = ",".join(str(x) for x in ids)
        raw = api._get("/instruments/kpis/11/year/mean/history",
                       params={"instList": ",".join(str(x) for x in ids)})
        txt = json.dumps(raw, ensure_ascii=False)
        print(f"  batch: typ={type(raw).__name__} nycklar={list(raw)[:8] if isinstance(raw, dict) else '-'} "
              f"längd={len(txt)}; början: {txt[:500]}")
        parsed = {k: len(v) for k, v in api.get_kpi_history_batch(ids, 11).items()}
        print(f"  get_kpi_history_batch → {parsed}")
        one = api.get_kpi_history(ids[0], 11, "year", "mean")
        print(f"  per-instrument get_kpi_history(BOL, 11) → {len(one)} punkter; första: {str(one[:2])[:200]}")
        bol = refs.get("BOL", {}).get("insId")
        try:
            hi = api._get("/holdings/insider", params={"instList": bol_eqnr})
            t2 = json.dumps(hi, ensure_ascii=False)
            print(f"  holdings/insider: typ={type(hi).__name__} "
                  f"nycklar={list(hi)[:8] if isinstance(hi, dict) else '-'} längd={len(t2)}; "
                  f"början: {t2[:500]}")
            from collections import Counter as _Ct
            for grp in (hi.get("list") or []):
                vals = grp.get("values") or []
                types = _Ct(v.get("transactionType") for v in vals)
                print(f"  insId={grp.get('insId')} transaktioner={len(vals)} typkoder={dict(types)}")
                seen = set()
                for v in sorted(vals, key=lambda x: str(x.get("transactionDate")), reverse=True):
                    tt = v.get("transactionType")
                    if tt in seen:
                        continue
                    seen.add(tt)
                    print(f"    typ {tt}: {v.get('transactionDate', '')[:10]} {v.get('ownerName')} "
                          f"({v.get('ownerPosition')}) shares={v.get('shares')} price={v.get('price')} "
                          f"amount={v.get('amount')} misc={v.get('misc')} eq={v.get('equityProgram')}")
        except Exception as e:
            print(f"  holdings/insider FEL: {e}")
        ins = api._get(f"/insiders/{bol}")
        txt = json.dumps(ins, ensure_ascii=False)
        print(f"  insiders: typ={type(ins).__name__} nycklar={list(ins)[:8] if isinstance(ins, dict) else '-'} "
              f"längd={len(txt)}; början: {txt[:400]}")
    except Exception as e:
        print(f"  FEL: {e}")

    try:
        probe_commodities(api)
    except Exception as e:
        print(f"  råvaror FEL: {e}")

    # ── Deep Contrarian: hela pipelinen med elimineringsorsaker ────────────
    if os.environ.get("PROBE_PIPELINE", "1") == "1":
        print("\n" + "=" * 72)
        print("DEEP CONTRARIAN — pipeline med elimineringsorsaker")
        print("=" * 72)
        import re as _re
        from collections import Counter as _C
        from contrarian_alpha.engine import PipelineConfig, run_pipeline
        cfg = PipelineConfig(mode="deep_contrarian",
                             market_ids=list(ALL_NORDIC_MARKETS), top_n=40)
        res = run_pipeline(cfg)
        print(f"  universum={res.universe_count} necessity={res.necessity_passed} "
              f"hate={res.hate_passed} bs={res.bs_passed} rankade={res.composite_ranked} "
              f"tid={res.run_duration_s}s")
        print("\n  RANKADE:")
        for r in res.results:
            print(f"   #{r.rank:>2} {r.ticker:12} {r.name[:22]:22} comp={r.composite_score:5.1f} "
                  f"N={r.necessity_score:3.0f} H={r.hat_score:4.1f} Q={r.quality_score or 0:4.1f} "
                  f"V={r.value_score or 0:4.1f} C={r.catalyst_score or 0:4.1f} "
                  f"nd/e={r.net_debt_ebitda} roic={r.roic} {r.branch[:22]} "
                  f"flags={[f for f in r.all_flags][:4]}")
        stages = _C(r.elimination_stage for r in res.eliminated)
        print(f"\n  ELIMINERADE per steg: {dict(stages)}")
        for stage in ("BALANCE_SHEET", "QUALITY_GATE"):
            rows = [r for r in res.eliminated if r.elimination_stage == stage]
            reasons = _C(_re.sub(r"-?\d+(\.\d+)?", "#", r.elimination_reason) for r in rows)
            print(f"\n  {stage} ({len(rows)}) — orsaker:")
            for reason, n in reasons.most_common(12):
                print(f"    {n:>3}  {reason[:110]}")
            print(f"  {stage} — de 40 första:")
            for r in rows[:40]:
                print(f"    {r.ticker:12} {r.name[:20]:20} {r.branch[:18]:18} N={r.necessity_score:3.0f} "
                      f"H={r.hat_score:4.1f} | {r.elimination_reason[:90]}")
        hated = [r for r in res.eliminated if r.elimination_stage == "HATE"]
        hs = sorted(r.hat_score for r in hated)
        if hs:
            q = lambda p: hs[min(len(hs) - 1, int(p * len(hs)))]
            print(f"\n  HATE-eliminerade ({len(hs)}): hat p25={q(.25):.1f} p50={q(.5):.1f} "
                  f"p75={q(.75):.1f} p90={q(.9):.1f} max={hs[-1]:.1f}")
            fl = _C(f for r in hated if r.hate_result for f in r.hate_result.flags)
            print(f"  hat-flaggor bland dem: {dict(fl.most_common(8))}")
            conf = sorted(r.hate_result.confidence for r in hated if r.hate_result)
            if conf:
                print(f"  hat-täckning (confidence) p50={conf[len(conf)//2]:.2f} "
                      f"min={conf[0]:.2f}")
            guard = [r for r in hated if "över SMA200" in (r.elimination_reason or "")]
            print(f"  varav unloved guard (>max över SMA200): {len(guard)}")
            for r in sorted(guard, key=lambda r: -r.hat_score)[:15]:
                print(f"    {r.ticker:12} {r.name[:20]:20} H={r.hat_score:4.1f} | "
                      f"{r.elimination_reason[:80]}")
            print("  HATE — högst hat som ändå föll (topp 25):")
            for r in sorted(hated, key=lambda r: -r.hat_score)[:25]:
                bd_ = r.hate_result.breakdown if r.hate_result else {}
                print(f"    {r.ticker:12} {r.name[:20]:20} {r.branch[:18]:18} H={r.hat_score:4.1f} "
                      f"conf={r.hate_result.confidence if r.hate_result else 0:.2f} "
                      f"{ {k: v for k, v in bd_.items() if v} } | "
                      f"{(r.elimination_reason or '')[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
