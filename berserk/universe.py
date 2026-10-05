"""
berserk/universe.py — producentbolag och råvaru-ETF:er för 🪓 BERSERK.

Varje aktie är kopplad till det tema (themes.THEMES) vars råvara driver dess
intjäning mest. Bolag med flera råvaror står under den största (Boliden →
koppar, Equinor → olja). Oljeservice och borrning står under olja; Neste under
olja (raffinering). Tickers i Yahoo-form; datasonden visar vilka som saknas.

  NORDIC     Norden — primärt universum
  US, CANADA, LONDON
             sekundärt — de ledande råvarubolagen i varje tema (GLOBAL = alla tre).
             Australien är borttaget (2026-10): ASX går inte att handla hos Nordnet.
  ETFS       råvaru-ETF:er (themes.ETFS) — handlas med setup 2 och 3
  REGION_INDEX  marknadsgrindens index per region (OMXS30, SPY, TSX, FTSE)

Ett bolag finns bara med en gång (ingen dubbelnotering): Rio Tinto och BHP via
London, Kinross via NYSE osv. Datasonden 2026-10 (berserk-probe) sorterade bort
CTRA, NGD, PCH, MEG.TO (ingen kurshistorik — uppköpta/sammanslagna) och
ARX.TO (slutade uppdateras).

Uppköpta och omdöpta tickers sorteras via dead_tickers (som i Ember).
"""

from __future__ import annotations

from berserk import themes as th

_NORDIC_RAW: dict = {
    # Energi
    "EQNR.OL": "olja", "AKRBP.OL": "olja", "VAR.OL": "olja", "OKEA.OL": "olja", "DNO.OL": "olja",
    "PEN.OL": "olja", "BNOR.OL": "olja", "TGS.OL": "olja", "SUBC.OL": "olja", "AKSO.OL": "olja",
    "BORR.OL": "olja", "ODL.OL": "olja", "NESTE.HE": "olja",
    # Basmetaller
    "BOL.ST": "koppar", "LUMI.ST": "koppar", "NHY.OL": "aluminium", "SSAB-A.ST": "stal",
    "OUT1V.HE": "zink_nickel", "ALLEI.ST": "stal",
    # Agri, skog och hav
    "YAR.OL": "godsel",
    "MOWI.OL": "lax", "SALM.OL": "lax", "LSG.OL": "lax", "BAKKA.OL": "lax", "GSF.OL": "lax",
    "SCA-B.ST": "skog", "HOLM-B.ST": "skog", "BILL.ST": "skog", "STERV.HE": "skog", "UPM.HE": "skog",
    "METSB.HE": "skog",
    # Frakt
    "FRO.OL": "tank", "HAFNI.OL": "tank", "OET.OL": "tank", "TRMD-A.CO": "tank", "BWLPG.OL": "tank",
    "2020.OL": "torrbulk",
}

_US_RAW: dict = {
    # Energi — olja (producenter, raffinering, service)
    "XOM": "olja", "CVX": "olja", "COP": "olja", "EOG": "olja", "DVN": "olja", "OXY": "olja", "FANG": "olja",
    "OVV": "olja", "APA": "olja", "MTDR": "olja", "PR": "olja", "CHRD": "olja", "MGY": "olja",
    "SM": "olja", "MUR": "olja", "NOG": "olja",
    "MPC": "olja", "VLO": "olja", "PSX": "olja", "PBF": "olja", "DK": "olja", "DINO": "olja",
    "SLB": "olja", "HAL": "olja", "BKR": "olja", "NOV": "olja", "FTI": "olja", "RIG": "olja", "VAL": "olja",
    "NE": "olja", "HP": "olja", "PTEN": "olja", "LBRT": "olja", "WHD": "olja", "OII": "olja", "TDW": "olja",
    # Energi — gas, kol, uran
    "EQT": "naturgas", "AR": "naturgas", "RRC": "naturgas", "CNX": "naturgas", "EXE": "naturgas", "LNG": "naturgas",
    "BTU": "kol", "AMR": "kol", "ARLP": "kol", "HCC": "kol", "CNR": "kol", "METC": "kol",
    "CCJ": "uran", "NXE": "uran", "DNN": "uran", "UUUU": "uran", "UEC": "uran", "LEU": "uran", "URG": "uran",
    # Basmetaller
    "FCX": "koppar", "SCCO": "koppar", "TECK": "koppar", "HBM": "koppar", "ERO": "koppar",
    "AA": "aluminium", "CENX": "aluminium", "KALU": "aluminium",
    "NUE": "stal", "STLD": "stal", "CMC": "stal", "RS": "stal", "MT": "stal", "TX": "stal",
    "CLF": "jarnmalm", "VALE": "jarnmalm",
    "ALB": "litium", "SQM": "litium", "SGML": "litium", "LAC": "litium", "MP": "sallsynta",
    # Ädelmetaller
    "NEM": "guld", "AEM": "guld", "GOLD": "guld", "KGC": "guld", "GFI": "guld", "AU": "guld", "HMY": "guld",
    "EGO": "guld", "IAG": "guld", "AGI": "guld", "BTG": "guld", "OR": "guld", "SSRM": "guld",
    "EQX": "guld", "DRD": "guld", "RGLD": "guld", "WPM": "guld", "FNV": "guld",
    "PAAS": "silver", "HL": "silver", "AG": "silver", "CDE": "silver", "FSM": "silver", "EXK": "silver",
    "SVM": "silver", "ASM": "silver", "SBSW": "platina",
    # Agri och skog
    "NTR": "godsel", "MOS": "godsel", "CF": "godsel", "IPI": "godsel", "LXU": "godsel", "ICL": "godsel",
    "ADM": "majs", "ANDE": "majs", "INGR": "majs", "CTVA": "majs", "BG": "soja",
    "WY": "skog", "RYN": "skog", "LPX": "skog", "UFPI": "skog", "BCC": "skog",
    # Frakt
    "DHT": "tank", "INSW": "tank", "TNK": "tank", "STNG": "tank", "ASC": "tank", "NAT": "tank",
    "SBLK": "torrbulk", "GNK": "torrbulk", "SB": "torrbulk",
}

_CANADA_RAW: dict = {
    "SU.TO": "olja", "CNQ.TO": "olja", "CVE.TO": "olja", "IMO.TO": "olja", "WCP.TO": "olja", "BTE.TO": "olja",
    "VET.TO": "olja", "PEY.TO": "olja", "PXT.TO": "olja",
    "TOU.TO": "naturgas", "BIR.TO": "naturgas", "AAV.TO": "naturgas",
    "LUN.TO": "koppar", "FM.TO": "koppar", "CS.TO": "koppar", "IVN.TO": "koppar", "TKO.TO": "koppar",
    "CIA.TO": "jarnmalm", "LIF.TO": "jarnmalm",
    "LUG.TO": "guld", "WDO.TO": "guld", "TXG.TO": "guld", "DPM.TO": "guld", "OGC.TO": "guld",
    "WFG.TO": "skog", "CFP.TO": "skog", "IFP.TO": "skog",
}

_LONDON_RAW: dict = {
    "BP.L": "olja", "SHEL.L": "olja", "HBR.L": "olja", "TLW.L": "olja", "ENQ.L": "olja",
    "ANTO.L": "koppar", "GLEN.L": "koppar", "AAL.L": "koppar", "RIO.L": "jarnmalm", "BHP.L": "jarnmalm",
    "FRES.L": "silver", "HOC.L": "guld", "EDV.L": "guld",
}

try:
    from dead_tickers import alive_map as _alive
except Exception:  # pragma: no cover
    def _alive(m):
        return dict(m)

NORDIC: dict = _alive(_NORDIC_RAW)
US: dict = _alive(_US_RAW)
CANADA: dict = _alive(_CANADA_RAW)
LONDON: dict = _alive(_LONDON_RAW)
GLOBAL: dict = {**US, **CANADA, **LONDON}                     # allt utanför Norden

ETFS: dict = dict(th.ETFS)
PRODUCERS: dict = {**NORDIC, **GLOBAL}
REGIONS: dict = {"Norden": NORDIC, "USA": US, "Kanada": CANADA, "London": LONDON}
LISTS: dict = {**{name: tuple(m) for name, m in REGIONS.items()}, "Råvaru-ETF:er": tuple(ETFS)}

# Marknadsgrinden per region (indexet över SMA200). Norden = OMXS30 (Börsdata), övriga Yahoo-index.
REGION_INDEX: dict = {"Norden": "OMXS30", "USA": "SPY", "Kanada": "^GSPTSE", "London": "^FTSE"}


def region_of(ticker: str) -> str:
    """Region ur tickerns suffix (ETF:erna är amerikanska)."""
    t = str(ticker or "").upper()
    if t.endswith((".ST", ".OL", ".CO", ".HE")):
        return "Norden"
    if t.endswith((".TO", ".V")):
        return "Kanada"
    if t.endswith(".L"):
        return "London"
    return "USA"


def theme_of(ticker: str) -> str:
    t = str(ticker or "").upper()
    return PRODUCERS.get(t) or ETFS.get(t) or ""


def kind_of(ticker: str) -> str:
    """'producent', 'etf' eller '' (okänd)."""
    t = str(ticker or "").upper()
    return "etf" if t in ETFS else "producent" if t in PRODUCERS else ""


def driver_candidates(ticker: str) -> tuple:
    """Drivarna för tickerns tema (preferensordning). En ETF utan eget tema är sin egen drivare."""
    theme = theme_of(ticker)
    return th.drivers(theme) or ((str(ticker).upper(),) if kind_of(ticker) == "etf" else ())


def by_theme(tickers=None) -> dict:
    """{tema: [tickers]} för universumet (eller en given lista)."""
    out: dict = {}
    for t in (tickers if tickers is not None else list(PRODUCERS) + list(ETFS)):
        theme = theme_of(t)
        if theme:
            out.setdefault(theme, []).append(t)
    return out
