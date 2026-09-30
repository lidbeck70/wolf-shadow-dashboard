"""
asymmetry/quick_config.py — trösklarna för Wolf Asymmetrys Snabbkoll.

Varje kort är GRÖN (100), GUL (50) eller RÖD (0). Ett kort utan data räknas
inte (DATA_GAP), gruppens poäng är snittet av de kort som gick att mäta.
Tabellerna är (gräns_grön, gräns_gul): för "högre är bättre" gäller
värde ≥ grön → GRÖN, ≥ gul → GUL, annars RÖD; för "lägre är bättre" tvärtom.
Talen är VAL och syns i varje korts förklaring.
"""

from __future__ import annotations

STATUS_POINTS = {"GREEN": 100.0, "AMBER": 50.0, "RED": 0.0}

# ── Survival ────────────────────────────────────────────────────────────────
RUNWAY_YEARS = (2.0, 1.0)          # kassa / årligt underskott (bara vid negativt FCF)
ND_EBITDA = (1.0, 3.0)             # lägre bättre; nettokassa = GRÖN
CURRENT_RATIO = (1.5, 1.0)         # omsättningstillgångar / korta skulder
DILUTION_3Y_PCT = (15.0, 45.0)     # lägre bättre: aktieantal +x % på tre år (≈ 5 resp. 15 %/år)
EQUITY_RATIO_PCT = (50.0, 30.0)    # soliditet

# ── Margin of Safety ────────────────────────────────────────────────────────
PRICE_BUFFER_PCT = (40.0, 20.0)    # EBITDA-marginal = hur långt priset kan falla innan EBITDA = 0
EV_EBITDA_VS_MEDIAN_PCT = (-25.0, 25.0)   # lägre bättre: % mot egen median
P_FCF_VS_MEDIAN_PCT = (-25.0, 25.0)       # lägre bättre
FCF_YIELD_PCT = (8.0, 4.0)                # reserv när P/FCF-historik saknas
FROM_52W_HIGH_PCT = (40.0, 20.0)          # % under 52-veckorshögsta
VS_SMA200_PCT = (-10.0, 5.0)              # lägre bättre: under snittet = hat = marginal
MIN_HISTORY = 3                           # minst så många år för en median

# ── Confidence (auto) ───────────────────────────────────────────────────────
COVERAGE_PCT = (80.0, 50.0)        # andel nyckeltal som fanns
STABILITY = (0.7, 0.4)             # Börsdatas vinst-/FCF-stabilitet (0–1)  — info (cykelvolatilitet), räknas inte
STABILITY_MIN_YEARS = 5            # minst så många årsrapporter för egen beräkning
# Resultatkvalitet (ersätter F-score i poängen — F-score straffar råvarucykeln):
FCF_POSITIVE_SHARE = (0.8, 0.5)    # andel år med positivt FCF, t.ex. 8 av 10 grönt
CASH_CONVERSION = (1.0, 0.7)       # operativt kassaflöde / nettoresultat, summerat över åren
QUALITY_MIN_YEARS = 5              # minst så många årsrapporter
REPORT_YEARS = (5, 3)              # år med årsrapporter
# FCF-data (datakonfidens, inte volatilitet): andel årsrapporter med FCF,
# och Börsdata mot Yahoo för senaste 12 mån (samma valuta)
FCF_DATA_COMPLETE = (0.9, 0.6)
FCF_DATA_MIN_YEARS = (5, 3)
FCF_SOURCE_GAP_PCT = 25.0          # FCF-definitionerna skiljer mer än börsvärdet — större tolerans
SOURCE_GAP_PCT = 10.0              # Börsdata mot Yahoo börsvärde: inom 10 % = samstämmigt

# ── Totalverdikt ────────────────────────────────────────────────────────────
VERDICT_GREEN_MIN = 65.0           # alla tre ≥ 65 → GRÖN
VERDICT_RED_BELOW = 40.0           # någon < 40 → RÖD, annars GUL
MIN_MEASURED = 2                   # färre mätta kort i en grupp → gruppen visas som DATA_GAP

# ── Commodity Leverage (auto) — utanför 300 ─────────────────────────────────
# Skattas ur historiken: bolagets årliga EBITDA och FCF mot råvarans årssnitt
# (omräknat till rapportvalutan). Poängtabell, prissteg och prob återanvänds ur
# asymmetry/config.ASYMMETRY_CONFIG["commodity_leverage"].
LEV_MIN_YEARS = 5                  # minst så många år med både resultat och pris
LEV_MIN_R2 = 0.30                  # svagare samband än så → visas men poängsätts inte
# Tema (ember.regime.detect_theme) → Yahoo-serie för råvarupriset. Uran, kol,
# sällsynta, skog och agri saknar en ren prisserie på Yahoo → DATA_GAP.
LEV_PRICE_TICKERS = {"guld": "GC=F", "silver": "SI=F", "platina": "PL=F", "palladium": "PA=F",
                     "koppar": "HG=F", "olja": "CL=F", "naturgas": "NG=F",
                     "vete": "ZW=F", "kaffe": "KC=F", "kakao": "CC=F"}
LEV_PRICE_UNITS = {"GC=F": "USD/oz", "SI=F": "USD/oz", "PL=F": "USD/oz", "PA=F": "USD/oz",
                   "HG=F": "USD/lb", "CL=F": "USD/fat", "NG=F": "USD/MMBtu",
                   "ZW=F": "USc/bu", "KC=F": "USc/lb", "CC=F": "USD/t"}
# Break-even-marginal (pris − break-even) / pris i % → band (spec §5)
BREAK_EVEN_BANDS = ((40.0, "UTMÄRKT"), (25.0, "STARK"), (10.0, "MÅTTLIG"), (0.0, "SVAG"))
BREAK_EVEN_FAILED = "UNDER BREAK-EVEN"
# Nedsidan (spec §12): FCF (annars EBITDA) vid pris −20 % och −30 %
LEV_DOWNSIDE_PCT = (-20.0, -30.0)
LEV_HIGH_SCORE = 6                 # från den här poängen kallas hävstången hög
