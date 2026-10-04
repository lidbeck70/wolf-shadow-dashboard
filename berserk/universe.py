"""
berserk/universe.py — producentbolag och råvaru-ETF:er för ⚔️ BERSERK.

Varje aktie är kopplad till det tema (themes.THEMES) vars råvara driver dess
intjäning mest. Bolag med flera råvaror står under den största (Boliden →
koppar, Equinor → olja). Oljeservice och borrning står under olja; Neste under
olja (raffinering). Tickers i Yahoo-form; datasonden visar vilka som saknas.

  NORDIC   Norden — primärt universum
  GLOBAL   Nordamerika och London — sekundärt (de största i varje tema)
  ETFS     råvaru-ETF:er (themes.ETFS) — handlas med setup 2 och 3

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
    "2020.OL": "torrbulk", "BELCO.OL": "torrbulk",
}

_GLOBAL_RAW: dict = {
    # Energi
    "XOM": "olja", "CVX": "olja", "COP": "olja", "OXY": "olja", "SLB": "olja", "SU.TO": "olja", "CNQ.TO": "olja",
    "EQT": "naturgas", "AR": "naturgas", "TOU.TO": "naturgas",
    "BTU": "kol", "AMR": "kol",
    "CCJ": "uran", "NXE": "uran", "DNN": "uran", "UUUU": "uran",
    # Basmetaller
    "FCX": "koppar", "SCCO": "koppar", "TECK": "koppar", "LUN.TO": "koppar", "FM.TO": "koppar",
    "ANTO.L": "koppar", "AA": "aluminium", "NUE": "stal", "STLD": "stal", "CLF": "jarnmalm",
    "RIO.L": "jarnmalm", "BHP.L": "jarnmalm", "ALB": "litium", "SQM": "litium", "MP": "sallsynta",
    # Ädelmetaller
    "NEM": "guld", "AEM": "guld", "GOLD": "guld", "KGC": "guld", "WPM": "guld", "FNV": "guld",
    "PAAS": "silver", "HL": "silver", "AG": "silver", "SBSW": "platina",
    # Agri
    "NTR": "godsel", "MOS": "godsel", "CF": "godsel", "ADM": "majs", "BG": "soja",
    # Frakt
    "DHT": "tank", "SBLK": "torrbulk",
}

try:
    from dead_tickers import alive_map as _alive_map
    NORDIC: dict = _alive_map(_NORDIC_RAW)
    GLOBAL: dict = _alive_map(_GLOBAL_RAW)
except Exception:  # pragma: no cover
    NORDIC, GLOBAL = dict(_NORDIC_RAW), dict(_GLOBAL_RAW)

ETFS: dict = dict(th.ETFS)
PRODUCERS: dict = {**NORDIC, **GLOBAL}
LISTS: dict = {"Norden": tuple(NORDIC), "Nordamerika & London": tuple(GLOBAL), "Råvaru-ETF:er": tuple(ETFS)}


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
