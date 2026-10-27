"""
universe/theme_probe.py — de största bolagen MED kassaflöde per tema på Valhall-kartan (Börsdata).

Underlag för att fylla tunna teman i universumet (Thor, skannrarna): för varje tema på kartan tas panelens
nuvarande bolag (EMBER-kartan och BERSERK-listorna) plus kandidaterna här, slås upp i Börsdata (nordiska och
globala listor), och bara bolag med positivt kassaflöde från löpande verksamhet (KPI 62) behålls — rangordnade
på börsvärde (KPI 50) omräknat till USD. Kandidaterna är förslag; Börsdata avgör vilka som finns, vad de är
värda och om de tjänar pengar. Dessutom Börsdatas egna branscher i råvarusektorerna, så stora bolag som saknas
bland kandidaterna syns.

Körs i borsdata_probe (PROBE_SECTION=themes). Läser, skriver ingenting.
"""

from __future__ import annotations

from typing import Callable, Optional

TOP_N = 20
MIN_N = 5
KPI_MCAP, KPI_OCF, KPI_FCF = 50, 62, 63

# Förslag per tema (Yahoo-form). Producenter och förädlare med egna intäkter — inte prospektering.
CANDIDATES: dict = {
    "naturgas": ("EQT", "EXE", "CTRA", "AR", "RRC", "CNX", "CRK", "GPOR", "TOU.TO", "ARX.TO", "PEY.TO", "BIR.TO",
                 "KEL.TO", "NVA.TO", "AAV.TO", "SM", "CRGY", "CIVI"),
    "naturgas_eu": ("EQNR.OL", "VAR.OL", "AKRBP.OL", "OKEA.OL", "DNO.OL", "HBR.L", "ENOG.L", "SQZ.L", "PNOR.OL",
                    "BNOR.OL", "OMV.VI", "ENI.MI", "TTE.PA", "REP.MC", "WINT.DE"),
    "kol": ("BTU", "CNR", "HCC", "AMR", "ARLP", "NRP", "METC", "GLEN.L", "CEIX", "HNRG", "NC"),
    "uran": ("CCJ", "KAP.L", "UEC", "UUUU", "CCO.TO", "NXE", "DNN", "EFR.TO", "BWXT", "LEU"),
    "tank": ("FRO", "DHT", "INSW", "STNG", "TNK", "TRMD", "HAFNI.OL", "ECO", "NAT", "OKEANIS.OL", "ASC", "PSHG",
             "TEN", "CMBT", "TK"),
    "torrbulk": ("SBLK", "GNK", "2020.OL", "BELCO.OL", "DSX", "SB", "PANL", "GOGL", "CMBT", "SMHN.DE", "EDRY",
                 "HSHP.OL", "JIN.OL", "WAWI.OL", "NORDEN.CO", "DFDS.CO"),
    "koppar": ("FCX", "SCCO", "TECK", "FM.TO", "LUN.TO", "HBM.TO", "CS.TO", "ERO.TO", "TGB", "BOL.ST", "ANTO.L",
               "GLEN.L", "KGH.WA", "AAL.L", "BHP", "RIO"),
    "aluminium": ("AA", "CENX", "KALU", "NHY.OL", "ACH", "CSTM", "ALU.PA", "S32.L", "RIO"),
    "zink_nickel": ("BOL.ST", "TECK", "VALE", "LUN.TO", "HBM.TO", "GLEN.L", "NEXA", "TKO.TO", "ERO.TO", "S.TO"),
    "stal": ("NUE", "STLD", "CLF", "CMC", "MT", "SSAB-A.ST", "SSAB-B.ST", "OUT1V.HE", "PKX", "GGB", "TX", "RS",
             "ATI", "CRS", "TKA.DE", "SZG.DE", "VOE.VI", "APAM.AS", "ACX.MC", "STLR.TO"),
    "jarnmalm": ("VALE", "RIO", "BHP", "CIA.TO", "LIF.TO", "MSB", "FEF.L", "GRNG.ST", "CLF"),
    "litium": ("ALB", "SQM", "LAC", "SGML", "PLL", "ALTM", "LI.V", "FL.V", "ATLX", "IONR", "PMET.TO"),
    "sallsynta": ("MP", "UUUU", "NEO.TO", "USAR", "ARU", "LYC", "ILU", "ARR", "VUL.DE"),
    "guld": ("NEM", "AEM", "B", "WPM", "FNV", "KGC", "AU", "GFI", "AGI", "RGLD", "PAAS", "EQX", "OR", "IAG", "NGD",
             "BTG", "LUG.TO", "K.TO", "EDV.TO", "DPM.TO", "TXG.TO", "OGC.TO", "WDO.TO", "CG.TO", "TFPM", "SSRM",
             "HMY", "EGO", "CDE", "HL", "ARTG.V", "FRES.L", "EDV.L", "POG.L", "CEY.L"),
    "silver": ("PAAS", "HL", "AG", "FSM", "CDE", "SVM", "MAG", "EXK", "GATO", "FRES.L", "HOC.L", "SSRM", "WPM",
               "TFPM", "ASM", "SKE", "AYA.TO"),
    "platina": ("SBSW", "PLG", "IMPUY", "ANGPY", "NMPNF", "SFR", "TBX", "JMAT.L"),
    "palladium": ("SBSW", "IMPUY", "ANGPY", "NMPNF", "JMAT.L", "SFR", "NILSY", "PDL"),
    "godsel": ("NTR", "MOS", "CF", "YAR.OL", "ICL", "IPI", "SDF.DE", "LXU", "UAN", "CVR", "OCI.AS", "SQM", "AGRO",
               "FMC", "CTVA", "AVD", "BIOX"),
    "skog": ("WY", "RYN", "PCH", "WFG", "IFP.TO", "CFP.TO", "HOLM-B.ST", "SCA-B.ST", "STE-R.HE", "UPM.HE", "SUZ",
             "MERC", "CLW", "SLVM", "IP", "MNDI.L", "SMURF.L", "BILL.ST", "MCOM.ST", "RFP", "SXY.TO"),
    "vete": ("ADM", "BG", "ANDE", "INGR", "GNC.AX", "VITR", "AGRO", "CTVA"),
    "majs": ("ADM", "BG", "INGR", "ANDE", "DAR", "CTVA", "GPRE", "REX", "AGRO", "ALTO", "LND", "FDP"),
    "soja": ("BG", "ADM", "AGRO", "CTVA", "LND", "ANDE", "DAR", "GPRE", "INGR", "CALM", "FDP"),
    "socker": ("SZU.DE", "AGRO", "CSAN", "CZZ", "ABF.L", "TATE.L", "AAK.ST", "BG", "ADM"),
    "kaffe": ("JDEP.AS", "KDP", "SBUX", "NESN.SW", "SJM", "FARM", "JVA", "CAFFE.MI"),
    "kakao": ("BARN.SW", "LISN.SW", "HSY", "MDLZ", "OLAM.SI", "TR", "RMCF", "AAK.ST", "CLOE", "NESN.SW"),
    "lax": ("MOWI.OL", "SALM.OL", "LSG.OL", "GSF.OL", "BAKKA.OL", "AUSS.OL", "MAS.OL", "ISLAX.OL", "KING.OL",
            "NOFI.OL", "ICEFISH.OL", "PROT.OL", "SACAM.OL", "BAKKA.CO", "ATLA.OL", "ANDF.OL"),
    "olja": ("XOM", "CVX", "COP", "EOG", "OXY", "DVN", "FANG", "APA", "HES", "CNQ.TO", "SU.TO", "CVE.TO",
             "IMO.TO", "EQNR.OL", "AKRBP.OL", "VAR.OL", "SHEL.L", "BP.L", "TTE.PA", "ENI.MI", "REP.MC", "OMV.VI",
             "GALP.LS", "PNOR.OL", "DNO.OL", "OKEA.OL", "BWE.OL", "ENQ.L", "HBR.L"),
}

# Europa (hemmanoteringar) — registrets marknad Europa tar bara in temalistornas bolag.
EUROPE_CANDIDATES: dict = {
    "olja": ("TTE.PA", "ENI.MI", "REP.MC", "GALP.LS", "PKN.WA"),
    "naturgas_eu": ("TTE.PA", "ENI.MI", "REP.MC", "ENGI.PA", "SRG.MI", "PKN.WA"),
    "kol": ("JSW.WA",),
    "koppar": ("KGH.WA", "NDA.DE"),
    "aluminium": ("AMG.AS",),
    "zink_nickel": ("ERA.PA",),
    "stal": ("MT.AS", "TKA.DE", "SZG.DE", "ACX.MC", "APAM.AS", "KCO.DE"),
    "litium": ("AMG.AS",),
    "platina": ("UMI.BR",),
    "palladium": ("UMI.BR",),
    "godsel": ("SDF.DE", "OCI.AS", "ATT.WA"),
    "skog": ("NVG.LS", "ALTR.LS", "ENC.MC", "SEM.LS"),
    "vete": ("KWS.DE", "RIN.PA"),
    "majs": ("KWS.DE", "RIN.PA"),
    "soja": ("KWS.DE",),
    "socker": ("SZU.DE", "KWS.DE"),
    "kaffe": ("NESN.SW",),
    "kakao": ("BARN.SW", "LISN.SW", "NESN.SW"),
}
for _th, _ts in EUROPE_CANDIDATES.items():
    CANDIDATES[_th] = tuple(dict.fromkeys(CANDIDATES.get(_th, ()) + _ts))

# Börsdatas sektorer att lista branschvis (Energi 3, Material 7, Dagligvaror 2) och branschnamn (delsträng)
# i andra sektorer som bär råvaror (frakt, skog, jordbruk, fiske).
BRANCH_SECTORS = (2, 3, 7)
BRANCH_WORDS = ("rederi", "frakt", "shipping", "sjöfart", "skog", "papper", "massa", "jordbruk", "lantbruk",
                "fisk", "odling", "livsmedel", "gödsel", "kemi", "gruv", "stål", "metall")
BRANCH_TOP = 15

# Grov växelkurs till USD för rangordningen (bara ordningen räknas, inte nivån). Okänd valuta → 1.
FX_USD_FALLBACK: dict = {"USD": 1.0, "CAD": 0.73, "GBP": 1.33, "GBX": 0.0133, "EUR": 1.16, "SEK": 0.105,
                         "NOK": 0.099, "DKK": 0.156, "CHF": 1.25, "AUD": 0.66, "PLN": 0.27, "ZAR": 0.056,
                         "JPY": 0.0067, "SGD": 0.78, "HKD": 0.128, "CNY": 0.14, "BRL": 0.18, "MXN": 0.054,
                         "ILS": 0.30}


def panel_themes() -> dict:
    """{tema: {Yahoo-ticker}} — panelens nuvarande bolag per tema (EMBER-kartan och BERSERK-listorna)."""
    out: dict = {}
    try:
        from ember.config import TICKER_THEME_MAP
        for t, th in TICKER_THEME_MAP.items():
            out.setdefault(th, set()).add(str(t).upper())
    except Exception:
        pass
    try:
        from berserk import universe as bu
        for region in (bu.NORDIC, bu.US, bu.CANADA, bu.LONDON):
            for t, th in dict(region).items():
                out.setdefault(th, set()).add(str(t).upper())
    except Exception:
        pass
    return out


def map_themes() -> list:
    """Kartans teman i kartans ordning (valhall.segments)."""
    from valhall import segments as sgm
    return [th for s in sgm.SEGMENTS for th in s.themes]


def _kpi(rows) -> dict:
    out = {}
    for r in rows or []:
        i, v = r.get("i"), r.get("n")
        if i is not None and v is not None:
            out[int(i)] = float(v)
    return out


def instrument_index(nordic: list, global_: list, table: dict) -> dict:
    """{Yahoo-ticker: instrument} för nordiska och globala listor (marknadstabellen ger suffixet). Globala
    tickers utan suffix kan krocka — den första vinner, och bolagsnamnet visas så krockar syns."""
    import markets as mk
    out: dict = {}
    for inst in list(nordic or []) + list(global_ or []):
        t = inst.get("ticker")
        if not t or inst.get("insId") is None:
            continue
        out.setdefault(mk.to_yf(t, inst.get("marketId"), table), inst)
    return out


def rank(pool, index: dict, mcap: dict, ocf: dict, fcf: dict, fx: dict) -> tuple:
    """(rader med kassaflöde > 0 sorterade på börsvärde i USD, saknas i Börsdata, utan positivt kassaflöde)."""
    rows, missing, negative = [], [], []
    seen = set()
    for t in sorted({str(x).upper() for x in pool}):
        inst = index.get(t)
        if inst is None:
            missing.append(t)
            continue
        iid = int(inst["insId"])
        if iid in seen:
            continue
        seen.add(iid)
        ccy = str(inst.get("stockPriceCurrency") or inst.get("reportCurrency") or "USD").upper()
        rate = fx.get(ccy, 1.0)
        m, o, f = mcap.get(iid), ocf.get(iid), fcf.get(iid)
        row = {"ticker": t, "name": str(inst.get("name") or ""), "ins_id": iid, "currency": ccy,
               "mcap": m, "mcap_usd": None if m is None else m * rate, "ocf": o, "fcf": f}
        if o is None or o <= 0:
            negative.append(row)
            continue
        rows.append(row)
    rows.sort(key=lambda r: -(r["mcap_usd"] or 0))
    return rows, missing, negative


def fx_rates(getter: Optional[Callable] = None) -> dict:
    """Växelkurser till USD — Yahoo när det går (Actions), annars de grova reservtalen."""
    out = dict(FX_USD_FALLBACK)
    if getter is None:
        return out
    for ccy in list(out):
        if ccy in ("USD", "GBX"):
            continue
        try:
            s = getter(f"{ccy}USD=X")
            if s is not None and len(s):
                out[ccy] = float(s.iloc[-1])
        except Exception:
            continue
    out["GBX"] = out.get("GBP", 1.33) / 100
    return out


def _fmt(v) -> str:
    return "—" if v is None else f"{v:,.0f}"


def probe(api, out=print, fx_getter: Optional[Callable] = None) -> dict:
    """Skriver tabellerna och returnerar {tema: [rader]} (topp TOP_N med kassaflöde)."""
    import markets as mk
    table = mk._merge(mk.from_api(api.get_markets(), api.get_countries()))
    nordic, global_ = api.get_instruments() or [], api.get_global_instruments_list() or []
    index = instrument_index(nordic, global_, table)
    mcap = {**_kpi(api.get_kpi_screener(KPI_MCAP)), **_kpi(api.get_kpi_screener_global(KPI_MCAP))}
    ocf = {**_kpi(api.get_kpi_screener(KPI_OCF)), **_kpi(api.get_kpi_screener_global(KPI_OCF))}
    fcf = {**_kpi(api.get_kpi_screener(KPI_FCF)), **_kpi(api.get_kpi_screener_global(KPI_FCF))}
    fx = fx_rates(fx_getter)
    out("=" * 72)
    out(f"TEMAN — största bolagen med kassaflöde (KPI {KPI_OCF} > 0), rangordnade på börsvärde (KPI {KPI_MCAP}) i USD")
    out(f"  instrument: Norden {len(nordic)} · globalt {len(global_)} · börsvärde {len(mcap)} · kassaflöde {len(ocf)}")
    out("=" * 72)
    have = panel_themes()
    result: dict = {}
    for th in map_themes():
        pool = set(have.get(th, set())) | set(CANDIDATES.get(th, ()))
        rows, missing, negative = rank(pool, index, mcap, ocf, fcf, fx)
        top = rows[:TOP_N]
        result[th] = top
        flag = "" if len(top) >= MIN_N else f"  ⚠️ färre än {MIN_N}"
        out(f"\n── {th}: {len(rows)} med kassaflöde · {len(negative)} utan · {len(missing)} saknas i Börsdata{flag}")
        for i, r in enumerate(top, 1):
            mark = "panel" if r["ticker"] in have.get(th, set()) else "NY"
            out(f"  {i:>2}. {r['ticker']:12} {r['name'][:30]:30} {r['currency']:4} mcap {_fmt(r['mcap']):>9} "
                f"(≈{_fmt(r['mcap_usd'])} MUSD) · OCF {_fmt(r['ocf'])} · FCF {_fmt(r['fcf'])}  [{mark}]")
        if negative:
            out("     utan positivt kassaflöde: " + ", ".join(f"{r['ticker']} ({_fmt(r['ocf'])})" for r in negative[:12]))
        if missing:
            out("     saknas i Börsdata: " + ", ".join(missing[:25]))
    _branches(api, out, nordic, global_, table, mcap, ocf, fx)
    return result


def _branches(api, out, nordic, global_, table, mcap, ocf, fx) -> None:
    """Börsdatas egna branscher i råvarusektorerna: de största med kassaflöde i varje."""
    import markets as mk
    try:
        branches = {int(b["id"]): b for b in api.get_branches()}
        sectors = {int(s["id"]): str(s.get("name") or "") for s in api.get_sectors()}
    except Exception as exc:
        out(f"\n  branscherna gick inte att läsa: {exc}")
        return
    picked = {bid: b for bid, b in branches.items()
              if b.get("sectorId") in BRANCH_SECTORS
              or any(w in str(b.get("name") or "").lower() for w in BRANCH_WORDS)}
    out("\n" + "=" * 72)
    out(f"BÖRSDATAS BRANSCHER — topp {BRANCH_TOP} med kassaflöde per bransch (Norden och globalt)")
    out("=" * 72)
    for bid, b in sorted(picked.items(), key=lambda kv: (kv[1].get("sectorId") or 0, kv[0])):
        rows = []
        for inst in list(nordic) + list(global_):
            if inst.get("branchId") != bid or inst.get("insId") is None:
                continue
            iid = int(inst["insId"])
            o = ocf.get(iid)
            if o is None or o <= 0:
                continue
            ccy = str(inst.get("stockPriceCurrency") or "USD").upper()
            m = mcap.get(iid)
            rows.append((m * fx.get(ccy, 1.0) if m is not None else 0.0,
                         mk.to_yf(inst.get("ticker"), inst.get("marketId"), table), str(inst.get("name") or "")))
        rows.sort(reverse=True)
        out(f"\n  bransch {bid} {b.get('name')} · sektor {sectors.get(b.get('sectorId'), '?')} · {len(rows)} med kassaflöde")
        for usd, t, n in rows[:BRANCH_TOP]:
            out(f"     {t:12} {n[:32]:32} ≈{_fmt(usd)} MUSD")
