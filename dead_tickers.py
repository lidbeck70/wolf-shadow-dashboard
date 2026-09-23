"""
dead_tickers.py — tickers som bytt namn eller försvunnit ur marknaden.

Panelen har ett tjugotal handskrivna tickerlistor (fallback-universum,
Ember-kartor, heatmap, presets, screener-etiketter). Ingen av dem hade en
status-kolumn, så "GOLD" (Barrick, som heter B sedan 2025) och "MRO"
(uppköpt 2024) låg kvar och gav tysta DATA_GAP i stället för kurser.

Ett register, två tabeller:

  RENAMED   gammal → ny ticker; listorna byter automatiskt (alive()).
  DELISTED  ticker → varför; listorna släpper dem.

Panelen kan inte verifiera mot marknaden härifrån (ingen nätåtkomst i
testerna), så varje rad har orsak och år. Är en rad fel: ta bort den, så
återvänder tickern i alla listor på en gång.
"""

from __future__ import annotations

from typing import Iterable, Optional

# gammal ticker → ny ticker (samma bolag, nytt kortnamn)
RENAMED: dict = {
    "GOLD": "B",              # Barrick Gold → Barrick Mining, NYSE: B (2025)
    "ERICB.ST": "ERIC-B.ST",  # Yahoo-form för Ericsson B
    "YARA.OL": "YAR.OL",      # Yara International heter YAR på Oslo Børs
    "NZYM-B.CO": "NSIS-B.CO",  # Novozymes → Novonesis (2024)
    "CEIX": "CNR",            # CONSOL + Arch → Core Natural Resources (2025)
    "ARCH": "CNR",
    "SWN": "EXE",             # Southwestern → Expand Energy (2024)
    "PLL": "ELVR",            # Piedmont + Sayona → Elevra Lithium (2025)
}

# ticker → orsak (uppköpt, avnoterad, fanns aldrig)
DELISTED: dict = {
    "MRO": "uppköpt av ConocoPhillips 2024",
    "HES": "uppköpt av Chevron 2025",
    "X": "uppköpt av Nippon Steel 2025",
    "LTHM": "Livent → Arcadium 2024, uppköpt av Rio Tinto 2025",
    "ALTM": "Arcadium uppköpt av Rio Tinto 2025",
    "FCU.TO": "Fission Uranium uppköpt av Paladin 2024",
    "GATO": "Gatos Silver uppköpt av First Majestic 2025",
    "SILV": "SilverCrest uppköpt av Coeur 2025",
    "MAG": "MAG Silver uppköpt av Pan American 2025",
    "MAG.TO": "MAG Silver uppköpt av Pan American 2025",
    "SAND": "Sandstorm Gold uppköpt av Royal Gold 2025",
    "SSL.TO": "Sandstorm Gold uppköpt av Royal Gold 2025",
    "UEC.TO": "UEC har ingen TSX-notering (NYSE American: UEC)",
    "KAZ.L": "KAZ Minerals avnoterat 2021",
    "SWMA.ST": "Swedish Match uppköpt av PMI 2022",
    "GOGL": "Golden Ocean fusionerat in i CMB.TECH 2025",
    "GOGL.OL": "Golden Ocean fusionerat in i CMB.TECH 2025",
    "KOL": "VanEck Coal ETF stängd 2020",
    "JJN": "iPath Nickel ETN förfallen",
    "PHNX.L": "Phoenix Group är ett försäkringsbolag — inte råvaror",
    "AKZA.AS": "Akzo Nobel är färg/kemi — inte råvaror",
    "SMR": "NuScale är kärnkraftsteknik, inte uran",
}


def status(ticker: str) -> Optional[str]:
    """'renamed → X', 'delisted: orsak' eller None när tickern är i bruk."""
    t = str(ticker or "").strip().upper()
    if t in RENAMED:
        return f"renamed → {RENAMED[t]}"
    if t in DELISTED:
        return f"delisted: {DELISTED[t]}"
    return None


def rename(ticker: str) -> str:
    t = str(ticker or "").strip().upper()
    return RENAMED.get(t, t)


def alive(tickers: Iterable[str]) -> list:
    """Listan utan avnoterade, med omdöpta ersatta. Ordning och dubbletter
    som förut i övrigt — kartor byggda på listan får samma nycklar."""
    out, seen = [], set()
    for t in tickers or []:
        key = str(t or "").strip().upper()
        if not key or key in DELISTED:
            continue
        key = RENAMED.get(key, key)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def alive_map(mapping: dict) -> dict:
    """Samma sak för {ticker: värde}-kartor (Ember-temat, heatmap-namnen)."""
    out = {}
    for t, v in (mapping or {}).items():
        key = str(t or "").strip().upper()
        if key in DELISTED:
            continue
        out.setdefault(RENAMED.get(key, key), v)
    return out
