"""
valhall/etf.py — ETF-benet (K6): UCITS-kandidater per ruta och rankning av exponeringar vid cykelvändning.

Två nivåer (analysen 2026-10-24 §6.2, beslut 14–15):
  A  vilken ruta som vänder — det gör kartan (valhall.cycle: fas, status, bekräftad vändning).
  B  vilken EXPONERING i rutan — rankas här med Deep Contrarian-logik på BARA prisdata, räknad på den LÅNGA
     amerikanska signalserien (SPDR sedan 1998, GLD/GDX/COPX …): Blindspot-produkten (nödvändighet × hatad ×
     billig mot egen tioårshistoria) + vändande styrka (fasen) + relativ styrka mot SPY. Aktiesegment väger
     vändande styrka tyngre än billighet (fas 4b: trend slog contrarian där), råvarukomplex tvärtom.

UCITS-linjen (iShares, VanEck, WisdomTree, Global X …) är bara HANDELSINSTRUMENTET: ung historia (TIDIG-bias i
cykelmotorn), valutaklass och Nordnet-tillgänglighet kan inte verifieras i kod. Därför har varje kandidat fältet
verified=False tills ägaren kontrollerat linjen hos Nordnet; inget nedan är bekräftat köpbart. En rad per
exponering — linjer på samma index (IUES.L mot QDVF.DE) rankas inte mot varandra.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import pandas as pd

from valhall import complexes as vcx
from valhall import cycle
from valhall import segments as sgm

AKTIE, FYSISK, TERMIN = "aktie-ETF", "fysisk ETC", "termins-ETC"
# Vikter (beslut 3 i karta_etf): råvarukomplex = billig/hatad först; aktiesegment = vändande styrka först.
WEIGHTS = {"komplex": {"blindspot": 1.0, "turn": 0.5, "rs": 0.25}, "segment": {"blindspot": 0.5, "turn": 1.0, "rs": 0.25}}
TURN_SCORE = {cycle.VANDER: 100.0, cycle.STARK: 100.0, cycle.UPPTREND: 50.0}
MIN_BARS = cycle.PCT_MIN


@dataclass(frozen=True)
class Etf:
    key: str                 # rutan: komplexnyckel eller segmentnyckel
    exposure: str            # vad linjen exponerar mot ("guldgruvor", "bred energi" …)
    name: str                # fondens namn — KANDIDAT att sondera, inte verifierad
    ticker: Optional[str]    # Yahoo-form för UCITS-linjen (None = ingen känd linje)
    signal: str              # lång amerikansk serie som rankningen räknas på
    kind: str = AKTIE
    verified: bool = False   # sätts av ägaren efter kontroll hos Nordnet (handelsbar, valuta, courtage)
    note: str = ""


CANDIDATES: list = [
    Etf("energi", "bred energi", "iShares S&P 500 Energy Sector UCITS", "IUES.L", "XLE"),
    Etf("energi", "olja & gas E&P", "iShares Oil & Gas Exploration & Production UCITS", "SPOG.L", "XOP"),
    Etf("energi", "uran", "Sprott Uranium Miners UCITS (HANetf)", "URNM.L", "URA"),
    Etf("energi", "brent spot", "WisdomTree Brent Crude Oil ETC", "BRNT.L", "BZ=F", TERMIN, note="rullkostnad"),
    Etf("basmetaller", "koppargruvor", "Global X Copper Miners UCITS", "COPX.L", "COPX"),
    Etf("basmetaller", "bred gruvkorg", "iShares MSCI Global Metals & Mining Producers UCITS", "SPMP.L", "PICK"),
    Etf("basmetaller", "koppar spot", "WisdomTree Copper ETC", "COPA.L", "HG=F", TERMIN, note="rullkostnad"),
    Etf("basmetaller", "batterikedja", "L&G Battery Value-Chain UCITS", "BATT.L", "LIT"),
    Etf("adelmetaller", "fysiskt guld", "iShares Physical Gold ETC", "IGLN.L", "GLD", FYSISK),
    Etf("adelmetaller", "guldgruvor", "VanEck Gold Miners UCITS", "GDX.L", "GDX"),
    Etf("adelmetaller", "juniorgruvor", "VanEck Junior Gold Miners UCITS", "GDXJ.L", "GDXJ"),
    Etf("adelmetaller", "fysiskt silver", "iShares Physical Silver ETC", "ISLN.L", "SLV", FYSISK),
    Etf("agri", "jordbruksbolag", "iShares Agribusiness UCITS", "SPAG.L", "MOO"),
    Etf("agri", "jordbrukskorg", "WisdomTree Agriculture ETC", "AIGA.L", "DBA", TERMIN, note="rullkostnad"),
    Etf("skog", "skog & timmer", "iShares Global Timber & Forestry UCITS", "WOOD.L", "WOOD"),
    Etf("frakt", "rederier", "sjöfarts-UCITS — ingen känd linje, slå upp", None, "BDRY", note="ingen UCITS-kandidat"),
]
# Aktiesegmenten: en linje per segment (första kandidaten i valhall.segments), signal = SPDR
CANDIDATES += [Etf(s.key, s.label.lower(), f"iShares S&P 500 {s.label} Sector UCITS (kandidat)",
                   (s.ucits[0] if s.ucits else None), s.spdr) for s in sgm.SEGMENTS]


def candidates(key: str) -> list:
    return [e for e in CANDIDATES if e.key == key]


def level_of(key: str) -> str:
    return "komplex" if key in vcx.KEYS else "segment"


def necessity_of(key: str) -> int:
    if key in vcx.NECESSITY:
        return vcx.NECESSITY[key]
    s = sgm.get(key)
    return s.necessity if s else 50


def score_exposure(st: dict, level: str) -> Optional[float]:
    """Poängen ur ett cycle.state-läge: None när hårda grindar faller (ingen percentil eller cykeltopp)."""
    if st.get("pct10") is None or st.get("cykel") in (cycle.TOPP, cycle.NODATA) or st.get("phase") == cycle.NODATA:
        return None
    w = WEIGHTS[level]
    turn = TURN_SCORE.get(st.get("phase"), 0.0)
    rs = st.get("rs_spy")
    rs_score = max(0.0, min(100.0, 50.0 + float(rs) / 0.6)) if rs is not None else 50.0
    return round((st.get("blindspot") or 0.0) * w["blindspot"] + turn * w["turn"] + rs_score * w["rs"], 1)


def rank_exposures(key: str, getter: Optional[Callable] = None, spy: Optional[pd.Series] = None, today=None,
                   period: str = "15y") -> list:
    """Rutans exponeringar rankade på signalserien: [{exposure, name, ticker, kind, verified, signal, status, phase,
    cykel, pct10, hat, blindspot, turn, score, blocked}] — bäst först, spärrade sist med orsak."""
    if getter is None:
        from market_prices import ohlcv as getter
    level, nec = level_of(key), necessity_of(key)
    out = []
    for e in candidates(key):
        try:
            df = getter(e.signal, period)
        except Exception:
            df = None
        row = {**asdict(e), "status": cycle.NODATA, "phase": cycle.NODATA, "cykel": cycle.NODATA, "pct10": None,
               "hat": None, "blindspot": None, "turn": dict(cycle.NO_TURN), "score": None, "blocked": None}
        if df is None or len(df) < MIN_BARS or "Close" not in df:
            row["blocked"] = f"{e.signal}: ingen serie (minst {MIN_BARS} dagar krävs)"
            out.append(row)
            continue
        st = cycle.state(cycle.cycle_frame(df["Close"], df.get("Volume"), spy, nec))
        row.update({k: st.get(k) for k in ("status", "phase", "cykel", "pct10", "hat", "blindspot", "turn")})
        row["score"] = score_exposure(st, level)
        if row["score"] is None:
            row["blocked"] = "cykeltopp — köp ej" if st.get("cykel") == cycle.TOPP else "percentil saknas (för kort serie)"
        out.append(row)
    out.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0), r["exposure"]))
    return out


def pick(key: str, getter: Optional[Callable] = None, spy: Optional[pd.Series] = None, today=None) -> Optional[dict]:
    """Bästa exponeringen med en UCITS-linje (verifierad först) — None när allt är spärrat eller saknar linje."""
    ranked = [r for r in rank_exposures(key, getter, spy, today) if r["score"] is not None and r["ticker"]]
    ranked.sort(key=lambda r: (not r["verified"], -(r["score"] or 0.0)))
    return ranked[0] if ranked else None
