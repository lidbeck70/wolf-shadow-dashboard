"""
valhall/segments.py — Valhalls elva segment.

Signalen räknas på de amerikanska SPDR-sektorfonderna (lång historik, sedan 1998 — XLC 2018, XLRE
2015). Utanför Norden handlas segmentet i en UCITS-ETF som går att köpa hos Nordnet (amerikanska
ETF:er saknar KID och kan inte köpas av privatpersoner i EU). UCITS-tickrarna är kandidater som
verifieras (valhall/probe.py + Nordnet) innan de används.

Norden: Börsdatas tio sektorer, där Finans & Fastighet (sektor 1) delas på bransch — fastighetsbolag
och REIT (bransch 75–76) blir Real Estate. Nödvändigheten är contrarian_alpha.necessity per
Börsdatasektor (samma tal som Quality Contrarian använder).

Nischen är segmentets särdrag i strategin — grunden är densamma överallt (cykel → kvalitet →
swing i Norden med Viking Nine som bas).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Segment:
    key: str
    label: str
    spdr: str                      # signaldrivare (Yahoo)
    ucits: tuple                   # UCITS-kandidater för handel utanför Norden (Yahoo, verifieras)
    bd_sectors: tuple              # Börsdatas sectorId
    necessity: int                 # 0–100 (Rick Rule: behövt)
    niche: str                     # strategins särdrag i segmentet
    bd_branches: tuple = ()        # bara dessa branchId (tom = hela sektorn)
    bd_exclude: tuple = ()         # utom dessa branchId
    themes: tuple = field(default=())   # BERSERK-råvaruteman under segmentet


REAL_ESTATE_BRANCHES = (75, 76)    # Fastighetsbolag, Fastighet - REIT (contrarian_alpha.necessity)

SEGMENTS: tuple = (
    Segment("energy", "Energi", "XLE", ("IUES.L", "QDVF.DE"), (3,), 88,
            "Råvarucykeln (BERSERK S1/S2) — olja, gas, kol, uran; producenter med Rick Rule-krav",
            themes=("olja", "naturgas", "naturgas_eu", "kol", "uran", "tank", "torrbulk")),
    Segment("materials", "Material", "XLB", ("IUMS.L",), (7,), 78,
            "Råvarucykeln — metaller, gödsel, skog; Durrett (guld/silver) och Tiggre (övriga) för hävstång",
            themes=("koppar", "aluminium", "zink_nickel", "stal", "jarnmalm", "litium", "sallsynta",
                    "guld", "silver", "platina", "palladium", "godsel", "skog")),
    Segment("industrials", "Industri", "XLI", ("IUIS.L",), (5,), 58,
            "Kapitalvarucykeln — orderingång och PMI; Nordens bredaste segment för Viking Nine"),
    Segment("financials", "Finans", "XLF", ("IUFS.L", "QDVH.DE"), (1,), 55,
            "Räntekurvan och kreditcykeln — banker och försäkring; investmentbolag med substansrabatt",
            bd_exclude=REAL_ESTATE_BRANCHES),
    Segment("real_estate", "Fastighet", "XLRE", ("IUSP.L",), (1,), 62,
            "Räntecykeln — köp hatade fastighetsbolag under substans när räntetoppen passerats",
            bd_branches=REAL_ESTATE_BRANCHES),
    Segment("technology", "Teknik", "XLK", ("IUIT.L", "QDVE.DE"), (6,), 20,
            "Momentum och kvalitet — låg nödvändighet: bara starka bolag, aldrig fallande knivar"),
    Segment("healthcare", "Hälsovård", "XLV", ("IUHC.L", "QDVG.DE"), (4,), 75,
            "Defensiv kvalitet — hatad efter patent- och prisoro; kassaflöde före pipeline"),
    Segment("communication", "Kommunikation", "XLC", ("IUCM.L",), (9,), 75,
            "Infrastruktur och kassaflöde — telekom som obligationssubstitut"),
    Segment("consumer_disc", "Sällanköp", "XLY", ("IUCD.L", "QDVK.DE"), (8,), 25,
            "Konsumentcykeln — köp i recession när hatet är störst, sälj vid konsumtionstopp"),
    Segment("consumer_staples", "Dagligvaror", "XLP", ("IUCS.L",), (2,), 75,
            "Defensiv nödvändighet — stabilt kassaflöde, köp när marginaloron är störst; livsmedelsråvaror "
            "och lax följer råvarucykeln",
            themes=("vete", "majs", "soja", "socker", "kaffe", "kakao", "lax")),
    Segment("utilities", "Kraftförsörjning", "XLU", ("IUUS.L",), (10,), 88,
            "Elpris och ränta — nödvändig infrastruktur; köp vid lågt elpris och räntetopp"),
)

BY_KEY: dict = {s.key: s for s in SEGMENTS}


def get(key: str) -> Optional[Segment]:
    return BY_KEY.get(key)


def segment_for(sector_id, branch_id=None) -> Optional[str]:
    """Börsdatas sectorId (+ branchId) → Valhalls segment. None när sektorn saknas."""
    try:
        sid = int(sector_id)
    except (TypeError, ValueError):
        return None
    try:
        bid = int(branch_id) if branch_id is not None else None
    except (TypeError, ValueError):
        bid = None
    for s in SEGMENTS:
        if sid not in s.bd_sectors:
            continue
        if s.bd_branches and bid not in s.bd_branches:
            continue
        if bid is not None and bid in s.bd_exclude:
            continue
        return s.key
    return None


def segment_of_theme(theme: str) -> Optional[str]:
    """BERSERK-temat → segmentet det hör till (energi eller material)."""
    return next((s.key for s in SEGMENTS if theme in s.themes), None)
