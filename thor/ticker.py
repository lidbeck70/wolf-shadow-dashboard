"""
thor/ticker.py — identiteten: vad panelen vet om en ticker innan någon motor körs.

Panelen har sju råvaruvokabulärer, flera tickerformer och tre olika sektoruppslag. Här finns ETT uppslag,
och källan följer alltid med:

  yf          Yahoo-form ('ERIC-B.ST'). Börsdatas 'ERIC B' får suffix ur marknaden när markets-tabellen skickas in.
  Börsdata    insId, namn, sektor/bransch, marknad, valuta — nordiska listan först, sedan den globala
              (sheets_refresh.resolve). Bara den cachade instrumentlistan, inga nya anrop per aktie.
  segment     Valhalls segment ur Börsdatas sektor/bransch (valhall.segments), annars ur råvarutemat.
  hemmaindex  ^OMX / OBX.OL / ^OMXC25 / ^OMXH25 / SPY (alpha_regime.quality_scan.benchmark_for).
  marknad     OMXS30 / SPY — Viking Nines marknadslager (samma regel som ovtlyr_nine.market_for).
  sektor-ETF  Börsdatas sektor → SPDR (ovtlyr_nine.BORSDATA_SECTOR_ETFS), annars segmentets SPDR.
  råvarutema  i ordning: dina ark (Durrett/producent/Poängmodellen) → BERSERK-listan → EMBER-listan →
              Börsdata-bransch (olja 1–5, kol 6, uran 7, guld 18). BERSERK-vokabulär, så att Valhall, BERSERK
              och Kartan menar samma sak. De gamla tabellerna ändras inte.

Rena funktioner: Börsdata-klienten, arken och markets-tabellen skickas in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from thor.candidates import NORDIC_SUFFIX, yf as yf_form

BRANCH_THEME = {1: "olja", 2: "olja", 3: "olja", 4: "olja", 5: "olja", 6: "kol", 7: "uran", 18: "guld"}
SOURCE_SHEET, SOURCE_BERSERK, SOURCE_EMBER, SOURCE_BRANCH = "arket", "BERSERK-listan", "EMBER-listan", "Börsdata-bransch"
NORDIC_MARKET, US_MARKET = "OMXS30", "SPY"


@dataclass
class Ticker:
    raw: str
    yf: str
    name: str = ""
    ins_id: Optional[int] = None
    sector_id: Optional[int] = None
    branch_id: Optional[int] = None
    market_id: Optional[int] = None
    currency: str = ""
    nordic: bool = False
    region: str = ""
    segment: Optional[str] = None
    segment_label: str = ""
    home_index: str = ""
    market: str = ""
    sector_etf: Optional[str] = None
    sector_source: str = ""
    theme: Optional[str] = None
    theme_label: str = ""
    theme_source: str = ""
    complex: Optional[str] = None

    @property
    def is_commodity(self) -> bool:
        return self.theme is not None

    def as_dict(self) -> dict:
        return {**asdict(self), "is_commodity": self.is_commodity}


# ── Former ──────────────────────────────────────────────────────────────────
def base(ticker) -> str:
    """Tickern utan börssuffix: 'ERIC-B.ST' → 'ERIC-B'."""
    t = yf_form(ticker)
    for s in NORDIC_SUFFIX + (".TO", ".V", ".L", ".AX"):
        if t.endswith(s):
            return t[:-len(s)]
    return t


def forms(ticker) -> list:
    """Alla former arken kan ha skrivit tickern i: 'ERIC-B.ST' → ['ERIC-B.ST', 'ERIC-B', 'ERIC B']."""
    t, b = yf_form(ticker), base(ticker)
    out = []
    for f in (t, b, b.replace("-", " ")):
        if f and f not in out:
            out.append(f)
    return out


# ── Uppslagen ───────────────────────────────────────────────────────────────
def lookup(api, ticker) -> Optional[dict]:
    """Börsdata-instrumentet ur klientens cachade listor (nordiska först, sedan globala). None utan klient,
    nyckel eller träff. Gör inga nya anrop utöver listorna."""
    if api is None or not getattr(api, "is_configured", True):
        return None
    try:
        from sheets_refresh import resolve
        iid = resolve(api, ticker)
    except Exception:
        return None
    if iid is None:
        return None
    info = None
    try:
        info = api.get_instrument_info(iid)
    except Exception:
        info = None
    if info is None:
        try:
            info = next((i for i in api.get_global_instruments_list() or [] if int(i.get("insId", -1)) == int(iid)), None)
        except Exception:
            info = None
    return {**(info or {}), "insId": int(iid)}


def theme_for(ticker, sheets: Optional[dict] = None, branch_id=None) -> tuple:
    """(BERSERK-tema | None, källa) — arken, BERSERK-listan, EMBER-listan, Börsdata-branschen, i den ordningen."""
    from berserk import themes as bth
    from berserk import universe as buv
    fs = forms(ticker)
    if sheets:
        try:
            from valhall.picks import _commodity_lookup
            table = _commodity_lookup(sheets)
            th = next((table[f] for f in fs if f in table), None)
            if th:
                return th, SOURCE_SHEET
        except Exception:
            pass
    th = buv.theme_of(fs[0])
    if th:
        return th, SOURCE_BERSERK
    try:
        from ember.config import TICKER_THEME_MAP
        e = TICKER_THEME_MAP.get(fs[0])
        if e in bth.THEMES:
            return e, SOURCE_EMBER
    except Exception:
        pass
    try:
        bid = int(branch_id) if branch_id is not None else None
    except (TypeError, ValueError):
        bid = None
    if bid in BRANCH_THEME:
        return BRANCH_THEME[bid], f"{SOURCE_BRANCH} {bid}"
    return None, ""


def segment_for(sector_id, branch_id, theme: Optional[str]) -> Optional[str]:
    """Valhalls segment ur Börsdatas sektor/bransch, annars ur råvarutemat."""
    from valhall import segments as sgm
    seg = sgm.segment_for(sector_id, branch_id) if sector_id is not None else None
    return seg or (sgm.segment_of_theme(theme) if theme else None)


def sector_etf_for(sector_id, segment: Optional[str]) -> tuple:
    """(SPDR | None, källa): Börsdatas sektor först (samma tabell som Viking Nine), annars segmentets drivare."""
    try:
        from ovtlyr_nine import BORSDATA_SECTOR_ETFS
        sid = int(sector_id) if sector_id is not None else None
        if sid in BORSDATA_SECTOR_ETFS:
            name, etf = BORSDATA_SECTOR_ETFS[sid]
            return etf, f"Börsdata · {name} → {etf}"
    except Exception:
        pass
    if segment:
        from valhall import segments as sgm
        s = sgm.get(segment)
        if s:
            return s.spdr, f"segmentet {s.label} → {s.spdr}"
    return None, ""


def home_index(ticker) -> str:
    from alpha_regime.quality_scan import benchmark_for
    return benchmark_for(yf_form(ticker))


def market_for(ticker) -> str:
    """Viking Nines marknad: nordiskt suffix → OMXS30, annars SPY (ovtlyr_nine.market_for — testet håller dem lika)."""
    return NORDIC_MARKET if yf_form(ticker).endswith(NORDIC_SUFFIX) else US_MARKET


def identify(raw, api=None, sheets: Optional[dict] = None, table: Optional[dict] = None) -> Ticker:
    """Allt om tickern i ett objekt. api = Börsdata-klient (valfri), sheets = arken (valfria),
    table = markets-tabellen (ger suffix åt Börsdata-former som 'ERIC B')."""
    from berserk import themes as bth
    from berserk import universe as buv
    from valhall import segments as sgm
    t = yf_form(raw)
    inst = lookup(api, t) if api is not None else None
    if inst and "." not in t and inst.get("marketId") is not None and table is not None:
        import markets
        t = markets.to_yf(inst.get("ticker") or t, inst.get("marketId"), table)
    inst = inst or {}
    sector_id, branch_id = inst.get("sectorId"), inst.get("branchId")
    theme, source = theme_for(t, sheets, branch_id)
    seg = segment_for(sector_id, branch_id, theme)
    etf, etf_src = sector_etf_for(sector_id, seg)
    region = buv.region_of(t)
    label, cx = "", None
    if theme:
        try:
            label, cx = bth.label(theme), bth.complex_of(theme)
        except Exception:
            label = theme
    return Ticker(raw=str(raw or "").strip(), yf=t, name=str(inst.get("name") or ""), ins_id=inst.get("insId"),
                  sector_id=sector_id, branch_id=branch_id, market_id=inst.get("marketId"),
                  currency=str(inst.get("stockPriceCurrency") or ""), nordic=region == "Norden", region=region,
                  segment=seg, segment_label=sgm.get(seg).label if seg and sgm.get(seg) else "",
                  home_index=home_index(t), market=market_for(t), sector_etf=etf, sector_source=etf_src,
                  theme=theme, theme_label=label, theme_source=source, complex=cx)
