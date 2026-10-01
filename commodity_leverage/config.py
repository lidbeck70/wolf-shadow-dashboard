"""commodity_leverage/config.py — tal för Råvaruhävstång-fliken."""

# Kurshävstång: veckoavkastning för aktien mot råvaran
BETA_YEARS = 3                     # mätfönster
BETA_MIN_WEEKS = 52                # färre veckor → DATA_GAP
BETA_MIN_SIDE_WEEKS = 15           # minst så många upp- resp. nedveckor för upp/ned-beta
BETA_PERIOD = "5y"                 # hämtas (delad priscache), sedan kapas till BETA_YEARS
# R² för kursbetat: under det här är sambandet svagt (visas, men märks)
BETA_WEAK_R2 = 0.15
# Upp mot ned: upp-beta minst så här mycket över ned-beta → asymmetri
ASYMMETRY_MIN_GAP = 0.2

MAX_TICKERS = 8

# Råvaror med prisserie på Yahoo (samma som Snabbkollens Commodity Leverage)
COMMODITY_LABELS = {"guld": "Guld", "silver": "Silver", "koppar": "Koppar", "olja": "Olja",
                    "naturgas": "Naturgas", "platina": "Platina", "palladium": "Palladium",
                    "vete": "Vete", "kaffe": "Kaffe", "kakao": "Kakao"}
AUTO = "Auto (bolagets tema)"
# Fritt pris för ett valt bolag: steg i % mot dagens råvarupris
CUSTOM_PCT_RANGE = (-50, 150)
