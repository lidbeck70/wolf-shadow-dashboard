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
DIM = "#8a8578"           # dämpad text — bara text ≥14 px och avstängda lägen (5,5:1 mot APP_BG)
NOTE = "#c4bfb3"          # underrubriker och förklaringar (11:1) — flyttad hit ur ui/components, som exporterar om den
MUTED = "#a8a397"         # etiketter, meta, sekundär text (8:1 mot APP_BG, 6,6:1 mot BG_ALT)

# ── Semantik ─────────────────────────────────────────────────────────────────
POS = GREEN = "#2d8a4e"   # godkänt, grönt läge, vinst
WARN = AMBER = "#d4943a"  # varning, gult läge, bevaka
NEG = RED = "#c44545"     # fel, rött läge, sälj
ACCENT = CYAN = "#00E5FF" # panelens accent (Arc Cyan)
GOLD = "#c9a84c"          # guld — granskningsarkens rubrikfärg
EMBER = "#FF6B3D"         # råvaruglöd (EMBER-strategin)
PURPLE = "#B400FF"        # Aurora — regim/sekundär accent
GREY = "#6b7280"          # okänt, avstängt

# Bläcktoner (designbeslut 5): samma färg, ljusare, för färgad text under 19 px — fyllningar och kanter behåller
# basfärgen. ink() nedan väljer. Samma toner är temats redTextColor/greenTextColor/violetTextColor (config.toml).
GREEN_INK = "#45b36b"     # 7,6:1 mot APP_BG
RED_INK = "#e36464"       # 6,0:1
PURPLE_INK = "#c35cff"    # 6,1:1

# ── Ytor ─────────────────────────────────────────────────────────────────────
APP_BG = "#05070A"        # sidans bakgrund (.stApp) och den klistrade flikraden
BG = "#0c0c12"            # varumärkesraden
BG_CARD = "#14141e"       # kort, chip
BG_ALT = "#1a1f25"        # rutor, remsor, fält
BORDER = "#2a2a38"
BORDER_FAINT = "rgba(255,255,255,0.06)"
BORDER_CARD = "rgba(255,255,255,0.08)"   # kortens och remsornas kant (§3.4) — ingen skugga

# ── Typografi ────────────────────────────────────────────────────────────────
FONT_UI = "'Space Grotesk', sans-serif"
FONT_MONO = "'JetBrains Mono', monospace"

# Typskalan i px (§3.3). Golvet är 12 px överallt.
FS_TITLE = "clamp(20px, 5.6vw, 26px)"    # sidrubriken (page_header)
FS_HERO = 32              # Thors verdiktord
FS_STRIP = 24             # beslutsremsans ord
FS_VALUE = 20             # rutans värde
FS_VALUE_LONG = 16        # rutans ord över 14 tecken
FS_STACK = 15             # rutans värde en del per rad (" · ")
FS_CARD = 16              # kortrubrik, ticker
FS_BODY = 15              # brödtext, beslutsremsans mening
FS_NOTE = 13              # underrubrik, förklaring, kortets text
FS_NAV = 14               # chip-raden, knappar
FS_BUTTON = 14
FS_BUTTON_SM = 13         # kompakta knappar (huvudraden, sparraden)
FS_LABEL = 12             # etikett, sektionsrubrik, chip, meta
FS_MIN = 12               # golvet
FS_INPUT = 16             # fält (iOS zoomar under 16 px)

# Avstånd (§3.4): stegen 4–32
SP_1, SP_2, SP_3, SP_4, SP_5, SP_6, SP_8 = 4, 8, 12, 16, 20, 24, 32

# Hörnradier (§3.4)
R_CHIP = 6
R_TILE = R_FIELD = R_BUTTON = 10
R_CARD = R_STRIP = 12
R_HERO = 14
R_POPOVER = 16
R_PILL = 999

TAP = 44                  # minsta tryckyta (Apple HIG)
FIELD_H = 48              # fältens höjd (inloggningen 52)

# Trafikljuset som text → färg, så att GRÖN/GUL/RÖD ser likadant ut överallt.
REGIME_COLOR = {"GRÖN": GREEN, "GUL": AMBER, "RÖD": RED, "OKÄND": GREY,
                "GREEN": GREEN, "ORANGE": AMBER, "RED": RED, "UNKNOWN": GREY}


def regime_color(label) -> str:
    return REGIME_COLOR.get(str(label or "").strip().upper(), GREY)


# ── Bläck och toner ──────────────────────────────────────────────────────────
INK = {GREEN: GREEN_INK, RED: RED_INK, PURPLE: PURPLE_INK}
INK_ON = True             # påslaget i PR 16 (designbeslut 5) — False ger basfärgen överallt igen


def ink(color: str) -> str:
    """Färgen för färgad text under 19 px: bläcktonen för GREEN, RED och PURPLE, alla andra oförändrade.
    Med INK_ON = False returneras basfärgen (så var det före PR 16)."""
    if not INK_ON:
        return color
    key = str(color or "").strip().lower()
    return next((v for k, v in INK.items() if k.lower() == key), color)


def tint(color: str, a: float) -> str:
    """En genomskinlig ton av en hexfärg: tint('#d4943a', .14) → 'rgba(212,148,58,0.14)'. Ersätter de lösa
    f'{färg}22'-strängarna. Annat än #rgb/#rrggbb lämnas som det är."""
    s = str(color or "").strip()
    h = s[1:] if s.startswith("#") else ""
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        return color
    try:
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return color
    a = max(0.0, min(1.0, float(a)))
    return f"rgba({r},{g},{b},{a:g})"
