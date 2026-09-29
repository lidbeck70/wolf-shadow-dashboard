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
STABILITY = (0.7, 0.4)             # Börsdatas vinst-/FCF-stabilitet (0–1)
STABILITY_MIN_YEARS = 5            # minst så många årsrapporter för egen beräkning
F_SCORE = (7.0, 4.0)               # Piotroski 0–9
REPORT_YEARS = (5, 3)              # år med årsrapporter
SOURCE_GAP_PCT = 10.0              # Börsdata mot Yahoo börsvärde: inom 10 % = samstämmigt

# ── Totalverdikt ────────────────────────────────────────────────────────────
VERDICT_GREEN_MIN = 65.0           # alla tre ≥ 65 → GRÖN
VERDICT_RED_BELOW = 40.0           # någon < 40 → RÖD, annars GUL
MIN_MEASURED = 2                   # färre mätta kort i en grupp → gruppen visas som DATA_GAP
