"""
ui/tokens.py — panelens färger och typografi, med semantiska namn.

47 filer hade egna färgkonstanter, och namnen ljög på flera ställen: GOLD
var cyan i tre moduler och guld i fyra, CYAN var guld i fem, och i PALETTE
var "red" och "amber" samma orange så att fel och varning inte gick att
skilja åt. Här finns EN uppsättning, namngiven efter vad färgen betyder.

Värdena är de som granskningsarken redan använder (varm text #e8e4dc, dämpad
#8a8578, grön/gul/röd som trafikljuset) — de är panelens faktiska utseende.
ui/theme.PALETTE behåller sina nycklar för bakåtkompatibilitet och läser
härifrån där de betyder samma sak.

Moduler byter till det här kitet stegvis: `from ui.tokens import TEXT, DIM,
GREEN, AMBER, RED, ACCENT, GOLD` ersätter de lokala konstanterna utan att
något annat ändras.
"""

# ── Text ─────────────────────────────────────────────────────────────────────
TEXT = "#e8e4dc"          # brödtext
DIM = "#8a8578"           # dämpad text, etiketter, hjälp

# ── Semantik ─────────────────────────────────────────────────────────────────
POS = GREEN = "#2d8a4e"   # godkänt, grönt läge, vinst
WARN = AMBER = "#d4943a"  # varning, gult läge, bevaka
NEG = RED = "#c44545"     # fel, rött läge, sälj
ACCENT = CYAN = "#00E5FF" # panelens accent (Arc Cyan)
GOLD = "#c9a84c"          # guld — granskningsarkens rubrikfärg
EMBER = "#FF6B3D"         # råvaruglöd (EMBER-strategin)
PURPLE = "#B400FF"        # Aurora — regim/sekundär accent
GREY = "#6b7280"          # okänt, avstängt

# ── Ytor ─────────────────────────────────────────────────────────────────────
BG = "#0c0c12"
BG_CARD = "#14141e"
BG_ALT = "#1a1f25"
BORDER = "#2a2a38"
BORDER_FAINT = "rgba(255,255,255,0.06)"

# ── Typografi ────────────────────────────────────────────────────────────────
FONT_UI = "'Space Grotesk', sans-serif"
FONT_MONO = "'JetBrains Mono', monospace"

# Trafikljuset som text → färg, så att GRÖN/GUL/RÖD ser likadant ut överallt.
REGIME_COLOR = {"GRÖN": GREEN, "GUL": AMBER, "RÖD": RED, "OKÄND": GREY,
                "GREEN": GREEN, "ORANGE": AMBER, "RED": RED, "UNKNOWN": GREY}


def regime_color(label) -> str:
    return REGIME_COLOR.get(str(label or "").strip().upper(), GREY)
