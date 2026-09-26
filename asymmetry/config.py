"""
asymmetry/config.py — alla trösklar för Wolf Asymmetry på ett ställe.

Ändra här, inte i motorerna. Varje tabell läses uppifrån: första steget
vars gräns är uppfylld ger poängen. Talen är VAL — de syns i varje
förklaring ("Varför 9?") så ingen tröskel är dold.
"""

from __future__ import annotations

ASYMMETRY_CONFIG: dict = {
    # Prisgriden: % mot dagens råvarupris. Griden visar intäkt, EBITDA, FCF
    # och marginal per steg; leverage-poängen mäts vid probe_pct.
    "price_steps_pct": (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 50.0),

    "commodity_leverage": {
        "probe_pct": 20.0,             # +20 % pris → hur mycket rör sig FCF?
        "downside_probe_pct": -20.0,   # nedsideskontrollen
        # (FCF-/EBITDA-svar i % vid probe → poäng). Under första gränsen = 0.
        "table": ((75.0, 10), (50.0, 8), (35.0, 6), (20.0, 4), (10.0, 2)),
        "max": 10,
        # kassa måste täcka så här många års negativt FCF vid nedsidan
        "cash_cover_years": 1.0,
    },

    "margin_of_safety": {
        "max_per_component": 2.0,
        # A. prismarginal mot break-even: (pris − break-even) / pris i %
        "price": ((40.0, 2.0), (25.0, 1.5), (10.0, 1.0), (0.0, 0.5)),
        # B. capex-marginal (utvecklare): NPV > 0 upp till och med +x % capex
        "capex_steps_pct": (10.0, 20.0, 30.0, 50.0),
        "capex": ((50.0, 2.0), (30.0, 1.5), (20.0, 1.0), (10.0, 0.5)),
        # C. kostnadsmarginal: FCF > 0 upp till och med +x % AISC/opex
        "opex_steps_pct": (10.0, 20.0, 30.0),
        "opex": ((30.0, 2.0), (20.0, 1.5), (10.0, 1.0)),
        "opex_base_positive": 0.5,     # bara basen positiv
        # D. balansräkning: överlever pris −20/−30/−40 % utan destruktiv emission
        "balance_steps_pct": (-20.0, -30.0, -40.0),
        "balance": ((3, 2.0), (2, 1.5), (1, 1.0)),    # antal överlevda steg → poäng
        "balance_cash_cover_years": 2.0,   # kassan täcker två års underskott = överlever
        "balance_max_nd_ebitda": 3.0,      # nettoskuld/EBITDA över detta = överlever inte
        # E. värdering: uppsida mot börsvärde vid pris −20 %
        "valuation_probe_pct": -20.0,
        "valuation": ((0.0, 2.0), (-25.0, 1.5), (-50.0, 1.0), (-75.0, 0.5)),
    },

    "break_even": {
        # marginal till break-even i % → band
        "strong": 30.0,
        "moderate": 15.0,
    },

    "stress_matrix": {
        "price_pct": (-20.0, 0.0, 20.0),
        "capex_pct": (-10.0, 0.0, 20.0, 40.0),
    },

    # Confidence-justerad uppsida. "linear" = uppsida × confidence/100.
    # Byt formel här när en olinjär straffkurva är på plats.
    "adjusted_upside": {"formula": "linear"},
}

BREAK_EVEN_STRONG, BREAK_EVEN_MODERATE, BREAK_EVEN_WEAK = "Stark", "Måttlig", "Svag"
