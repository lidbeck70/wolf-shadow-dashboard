"""
berserk/themes.py — råvaruteman, komplex och drivare för 🪓 BERSERK.

Varje tema har en eller flera DRIVARE (Yahoo-symboler) i preferensordning:
råvaruterminen först, en ETF/ETC med samma exponering som reserv. Datasonden
(probe.py) avgör vilken som faktiskt går att hämta och hur långt tillbaka.
Teman utan prisserie (lax, nordisk el) har inga drivare — aktierna handlas då
bara på sin egen kurva (setup 3 i PR 1).

Teman som också finns i Ember (ember.config.THEME_TO_COMPLEX) har samma komplex
där — testerna kontrollerar det.
"""

from __future__ import annotations

COMPLEXES = {"energi": "ENERGI", "basmetaller": "BASMETALLER", "adelmetaller": "ÄDELMETALLER",
             "agri": "AGRI, SKOG & HAV", "frakt": "FRAKT"}

# tema → (etikett, komplex, drivare i preferensordning)
THEMES: dict = {
    # ── Energi ──────────────────────────────────────────────────────────────
    "olja":        ("Olja", "energi", ("BZ=F", "CL=F", "BNO", "USO")),
    "naturgas":    ("Naturgas (USA)", "energi", ("NG=F", "UNG")),
    "naturgas_eu": ("Naturgas (Europa, TTF)", "energi", ("TTF=F",)),
    "kol":         ("Kol", "energi", ()),                                # MTF=F slutade uppdateras 2025
    "uran":        ("Uran", "energi", ("U-UN.TO", "URNM", "URA")),
    # ── Basmetaller ─────────────────────────────────────────────────────────
    "koppar":      ("Koppar", "basmetaller", ("HG=F", "CPER", "COPX")),
    "aluminium":   ("Aluminium", "basmetaller", ("ALI=F",)),
    "zink_nickel": ("Zink & nickel", "basmetaller", ("PICK",)),          # ingen termin hos Yahoo — gruvkorg
    "stal":        ("Stål", "basmetaller", ("HRC=F", "SLX")),
    "jarnmalm":    ("Järnmalm", "basmetaller", ("TIO=F",)),
    "litium":      ("Litium & batteri", "basmetaller", ("LIT",)),
    "sallsynta":   ("Sällsynta jordartsmetaller", "basmetaller", ("REMX",)),
    # ── Ädelmetaller ────────────────────────────────────────────────────────
    "guld":        ("Guld", "adelmetaller", ("GC=F", "GLD")),
    "silver":      ("Silver", "adelmetaller", ("SI=F", "SLV")),
    "platina":     ("Platina", "adelmetaller", ("PL=F", "PPLT")),
    "palladium":   ("Palladium", "adelmetaller", ("PA=F", "PALL")),
    # ── Agri, skog och hav ──────────────────────────────────────────────────
    "vete":        ("Vete", "agri", ("ZW=F", "WEAT")),
    "majs":        ("Majs", "agri", ("ZC=F", "CORN")),
    "soja":        ("Soja", "agri", ("ZS=F", "SOYB")),
    "socker":      ("Socker", "agri", ("SB=F", "CANE")),
    "kaffe":       ("Kaffe", "agri", ("KC=F",)),
    "kakao":       ("Kakao", "agri", ("CC=F",)),
    "godsel":      ("Gödsel", "agri", ("UFV=F",)),                       # urea — finns den inte: aktiens kurva
    "skog":        ("Skog & trävara", "agri", ("LBR=F", "WOOD")),
    "lax":         ("Lax", "agri", ()),                                  # Fish Pool finns inte hos Yahoo
    # ── Frakt ───────────────────────────────────────────────────────────────
    "tank":        ("Tank & produkt", "frakt", ("BWET",)),               # Breakwave tanker, kort historik
    "torrbulk":    ("Torrbulk", "frakt", ("BDRY",)),
}

# Råvaru-ETF:er att handla själva (tema). Amerikanska, lång historik — i Sverige
# handlas motsvarande ETC:er; exponeringen är densamma, kursen inte identisk.
ETFS: dict = {
    "USO": "olja", "BNO": "olja", "UNG": "naturgas", "URA": "uran", "URNM": "uran",
    "CPER": "koppar", "COPX": "koppar", "PICK": "zink_nickel", "SLX": "stal", "LIT": "litium", "REMX": "sallsynta",
    "GLD": "guld", "GDX": "guld", "GDXJ": "guld", "SLV": "silver", "SIL": "silver", "PPLT": "platina",
    "PALL": "palladium", "WEAT": "vete", "CORN": "majs", "SOYB": "soja", "CANE": "socker", "DBA": "agri_korg",
    "WOOD": "skog", "BDRY": "torrbulk", "XLE": "olja", "XOP": "olja",
}
# Korgar utan eget tema (ETF:en är sin egen drivare)
BASKETS = {"agri_korg": ("Jordbrukskorg", "agri", ("DBA",))}


def label(theme: str) -> str:
    t = THEMES.get(theme) or BASKETS.get(theme)
    return t[0] if t else theme


def complex_of(theme: str) -> str:
    t = THEMES.get(theme) or BASKETS.get(theme)
    return t[1] if t else ""


def drivers(theme: str) -> tuple:
    """Temats drivare i preferensordning (tom = ingen prisserie)."""
    t = THEMES.get(theme) or BASKETS.get(theme)
    return tuple(t[2]) if t else ()


def all_driver_symbols() -> list:
    out = []
    for theme in list(THEMES) + list(BASKETS):
        for s in drivers(theme):
            if s not in out:
                out.append(s)
    return out
