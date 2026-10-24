"""
complex_profiles.py — premisserna per komplex och segment för Viking Nine-varianterna (K3).

Grunden är Viking Nine oförändrad (ovtlyr_nine, viking_execution, viking_exit — "Rör inte viking"). Det som får
skilja sig per ruta är BARA det som redan är injicerbart i motorn (analysen 2026-10-24 §5.4, beslut 5–7):

  sector_symbol   SECTOR-lagret (Nine faktor 4–5): råvarukomplexens ETF (COPX, GDX, XLE …) i stället för SPDR;
                  segmenten behåller sin SPDR — Fastighet får XLRE (Börsdata-sektor 1 gav XLF)
  market          MARKET-lagret: OMXS30 via ovtlyr_nine.nordic_market (OMXSPI/DBC är framtida beslut)
  breadth         "largecap" = svensk Large Cap över EMA50 (som i dag) · "producers" = komplexets nordiska
                  producenter över EMA50
  driver_theme    BERSERK-temat vars drivare grindar entryn ("turn": EMA20 stigande och över EMA50) och ger
                  exit när drivaren stänger under EMA50 (BERSERK S2:s regel) — None för aktiesegment
  universe        "komplex" (bolag med komplexets teman) · "segment" (Börsdata-sektorn)
  risk_pct / max_position_pct   1,25 % / 20 % för råvarukomplex (BERSERK:s värden), 1,5 % / 25 % för segment

ATR-multipel, R/R, volymkrav, F&G-trösklar och rapportfönster är modulkonstanter i motorn och finns INTE här —
de prövas i backtestet (viking_backtest.Config) innan någon ändring föreslås (beslut 6).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

from valhall import complexes as vcx
from valhall import segments as sgm

KOMPLEX, SEGMENT = "komplex", "segment"
LARGECAP, PRODUCERS = "largecap", "producers"
GATE_OFF, GATE_TURN = "off", "turn"
NORDIC_MARKET = "OMXS30"
# Allt i en profil måste gå att skicka in i motorn utan att röra den — testerna låser listan.
INJECTABLE = ("key", "label", "level", "sector_symbol", "market", "breadth", "driver_theme", "driver_gate",
              "universe", "risk_pct", "max_position_pct", "min_turnover")


@dataclass(frozen=True)
class ComplexProfile:
    key: str
    label: str
    level: str                       # KOMPLEX | SEGMENT
    sector_symbol: str               # ETF med OHLCV för Nine-lagret SECTOR
    market: str = NORDIC_MARKET
    breadth: str = LARGECAP
    driver_theme: Optional[str] = None
    driver_gate: str = GATE_OFF
    universe: str = SEGMENT
    risk_pct: float = 1.5
    max_position_pct: float = 25.0
    min_turnover: float = 1_000_000.0

    @property
    def is_complex(self) -> bool:
        return self.level == KOMPLEX


_COMMODITY = dict(level=KOMPLEX, breadth=PRODUCERS, driver_gate=GATE_TURN, universe=KOMPLEX,
                  risk_pct=1.25, max_position_pct=20.0)

PROFILES: dict = {
    # Råvarukomplexen (valhall.complexes): sektorserie = komplexets ETF, drivare = komplexets ledande tema
    "energi": ComplexProfile("energi", "Energi", sector_symbol="XLE", driver_theme="olja", **_COMMODITY),
    "basmetaller": ComplexProfile("basmetaller", "Basmetaller", sector_symbol="COPX", driver_theme="koppar", **_COMMODITY),
    "adelmetaller": ComplexProfile("adelmetaller", "Ädelmetaller", sector_symbol="GDX", driver_theme="guld", **_COMMODITY),
    "agri": ComplexProfile("agri", "Agri", sector_symbol="DBA", driver_theme="agri_korg", **_COMMODITY),
    "skog": ComplexProfile("skog", "Skog & massa", sector_symbol="WOOD", driver_theme="skog", **_COMMODITY),
    "frakt": ComplexProfile("frakt", "Frakt", sector_symbol="BDRY", driver_theme="torrbulk", **_COMMODITY),
}
# Aktiesegmenten: dagens Viking Nine-premisser, med Fastighet mätt mot XLRE i stället för XLF
PROFILES.update({s.key: ComplexProfile(s.key, s.label, SEGMENT, sector_symbol="XLRE" if s.key == "real_estate" else s.spdr)
                 for s in sgm.SEGMENTS})
assert set(vcx.KEYS) <= set(PROFILES) and all(s.key in PROFILES for s in sgm.SEGMENTS)
assert all(f.name in INJECTABLE for f in fields(ComplexProfile)), "profilen bär ett fält motorn inte kan ta emot"


def get(key: str) -> Optional[ComplexProfile]:
    return PROFILES.get(str(key or "").strip().lower())


def complexes() -> list:
    return [PROFILES[k] for k in vcx.KEYS]


def segments() -> list:
    return [PROFILES[s.key] for s in sgm.SEGMENTS]


def as_dict(p: ComplexProfile) -> dict:
    return {f.name: getattr(p, f.name) for f in fields(ComplexProfile)}
