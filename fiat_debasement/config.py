"""
fiat_debasement/config.py — vilka serier modulen använder, i preferensordning.

Varje (begrepp, valuta) har en lista av källor: den första som fungerar
används, resten är reserver. Ordningen bygger på datasonden (PR 1,
2026-10-03): primärkällan (centralbank/statistikmyndighet) först.
Källorna skarvas aldrig ihop i tysthet — enda undantaget är guld och silver,
där Yahoo-terminer fyller 2000–2006 före Börsdatas spotserie och märks som
"terminspris" (se ASSET_SPLICE).

En källa är en dict: kind (fred/ecb/eurostat/scb/riksbank/yahoo/borsdata),
id/args för källan, samt unit, label och definition som visas i UI:t.
"""

from __future__ import annotations

CURRENCIES = ("SEK", "EUR", "USD")

# Begreppen — hålls isär (inflation ≠ penningmängd ≠ valutaförsvagning)
M2, CPI, CPI_LONG, CORE, GDP, DEBT, DEBT_INFO, FX = (
    "m2", "cpi", "cpi_long", "core_cpi", "gdp", "debt", "debt_info", "fx")
GOLD, SILVER, COPPER, OIL, BTC = "gold", "silver", "copper", "oil", "bitcoin"

CONCEPT_LABEL = {
    M2: "Penningmängd (M2)", CPI: "KPI", CPI_LONG: "KPI, lång historik", CORE: "Kärn-KPI",
    GDP: "Real BNP", DEBT: "Statsskuld/BNP", DEBT_INFO: "Statsskuld/BNP (tillägg)", FX: "Växelkurs mot USD",
    GOLD: "Guld", SILVER: "Silver", COPPER: "Koppar", OIL: "Olja (Brent)", BTC: "Bitcoin",
}

# Hur gammalt sista datum får vara (dagar från periodens början) innan data räknas som inaktuell.
# Kvartal: Q1 (1 jan) publiceras först i juli och är den senaste siffran ända till oktober.
STALE_DAYS = {"D": 10, "W": 21, "M": 100, "Q": 370, "A": 640}

_M2_SE = ("SCB:s penningmängdsmått följer ECB:s definitioner. M2 = M1 (sedlar, mynt, inlåning på "
          "transaktionskonton) + inlåning med upp till två års bindningstid och uppsägningstid upp till "
          "tre månader. Riksbanken följer främst M3.")
_M2_EU = ("ECB:s M2 = M1 + inlåning med bindningstid upp till två år + inlåning med uppsägningstid upp "
          "till tre månader. Före 1999 är euroområdets siffror bakåtberäknade av ECB.")
_M2_US = ("Federal Reserves M2 = M1 + småsparkonton (< 100 000 USD) + privata penningmarknadsfonder. "
          "Inte exakt samma mått som ECB:s M2.")

SERIES: dict = {
    # ── Penningmängd ──
    (M2, "SEK"): [
        {"kind": "scb", "id": "FM/FM5001/FM5001A/FM5001penningmangd", "prefer_text": ("M2", "utestående", "mnkr"),
         "unit": "mn SEK", "label": "M2 Sverige (SCB)", "definition": _M2_SE},
    ],
    (M2, "EUR"): [
        {"kind": "ecb", "flow": "BSI", "key": "M.U2.Y.V.M20.X.1.U2.2300.Z01.E", "unit": "mn EUR",
         "label": "M2 euroområdet (ECB)", "definition": _M2_EU},
    ],
    (M2, "USD"): [
        {"kind": "fred", "id": "M2SL", "unit": "mdr USD", "label": "M2 USA (Fed)", "definition": _M2_US},
    ],
    # ── KPI ──
    (CPI, "SEK"): [
        # "Fastställda tal" i 2020=100-tabellen börjar först 2026; skuggindex är hela serien 1980–
        {"kind": "scb", "id": "PR/PR0101/PR0101A/KPI2020M", "prefer_text": ("skugg", "2020=100", "index"),
         "unit": "index 2020=100", "label": "KPI (SCB, skuggindex 2020=100)",
         "definition": "SCB:s skuggindex är KPI med fler decimaler — samma serie som de fastställda talen "
                       "men utan avrundning, och den enda med historik före 2026 i basen 2020=100."},
        {"kind": "fred", "id": "CP0000SEM086NEST", "unit": "index", "label": "HIKP Sverige (Eurostat via FRED)"},
    ],
    (CPI, "EUR"): [
        # Eurostat har inte HICP 2025=100 i API:t än (sond 2026-10-03) — FRED har redan ombasad serie
        {"kind": "fred", "id": "CP0000EZ19M086NEST", "unit": "index", "label": "HICP euroområdet (Eurostat via FRED)"},
    ],
    (CPI, "USD"): [
        {"kind": "fred", "id": "CPIAUCSL", "unit": "index 1982–84=100", "label": "CPI-U (BLS via FRED)"},
    ],
    # ── KPI med lång historik (köpkraft från 1970 och tidigare) ──
    (CPI_LONG, "SEK"): [
        {"kind": "scb", "id": "PR/PR0101/PR0101A/KPILevindexM", "prefer_text": ("1914=100", "index"),
         "unit": "index juli 1914=100", "label": "KPI/levnadskostnadsindex (SCB) sedan 1914"},
    ],
    (CPI_LONG, "USD"): [
        {"kind": "fred", "id": "CPIAUCNS", "unit": "index 1982–84=100", "label": "CPI-U ej säsongsjusterad sedan 1913"},
    ],
    # ── Kärn-KPI ──
    (CORE, "SEK"): [
        {"kind": "scb", "id": "PR/PR0101/PR0101J/KPIFXE2020", "prefer_text": ("2020=100", "index"),
         "unit": "index 2020=100", "label": "KPIF-XE (SCB)"},
    ],
    (CORE, "EUR"): [
        {"kind": "fred", "id": "00XEFDEZ19M086NEST", "unit": "index",
         "label": "HICP exkl. energi, livsmedel, alkohol och tobak (via FRED)"},
    ],
    (CORE, "USD"): [
        {"kind": "fred", "id": "CPILFESL", "unit": "index", "label": "CPI exkl. livsmedel och energi (via FRED)"},
    ],
    # ── Real BNP ──
    (GDP, "SEK"): [
        {"kind": "eurostat", "dataset": "namq_10_gdp",
         "params": {"geo": "SE", "unit": "CLV10_MNAC", "s_adj": "SCA", "na_item": "B1GQ"},
         "unit": "mn kedjade 2010 SEK", "label": "Real BNP Sverige (Eurostat)"},
        {"kind": "fred", "id": "CLVMNACSCAB1GQSE", "unit": "mn kedjade 2010 SEK", "label": "Real BNP Sverige (via FRED)"},
    ],
    (GDP, "EUR"): [
        {"kind": "eurostat", "dataset": "namq_10_gdp",
         "params": {"geo": "EA20", "unit": "CLV10_MEUR", "s_adj": "SCA", "na_item": "B1GQ"},
         "unit": "mn kedjade 2010 EUR", "label": "Real BNP euroområdet (Eurostat)"},
        {"kind": "fred", "id": "CLVMNACSCAB1GQEA19", "unit": "mn kedjade 2010 EUR", "label": "Real BNP EA19 (via FRED)"},
    ],
    (GDP, "USD"): [
        {"kind": "fred", "id": "GDPC1", "unit": "mdr kedjade 2017 USD", "label": "Real BNP USA (BEA via FRED)"},
    ],
    # ── Statsskuld/BNP: hela offentliga sektorns bruttoskuld (jämförbart mått) ──
    (DEBT, "SEK"): [
        {"kind": "eurostat", "dataset": "gov_10q_ggdebt",
         "params": {"geo": "SE", "unit": "PC_GDP", "sector": "S13", "na_item": "GD"},
         "unit": "% av BNP", "label": "Offentlig bruttoskuld, Maastricht (Eurostat)"},
    ],
    (DEBT, "EUR"): [
        {"kind": "eurostat", "dataset": "gov_10q_ggdebt",
         "params": {"geo": "EA20", "unit": "PC_GDP", "sector": "S13", "na_item": "GD"},
         "unit": "% av BNP", "label": "Offentlig bruttoskuld, Maastricht (Eurostat)"},
    ],
    (DEBT, "USD"): [
        {"kind": "fred", "id": "GGGDTAUSA188N", "unit": "% av BNP",
         "label": "Offentlig bruttoskuld, hela offentliga sektorn (IMF, årlig)"},
    ],
    (DEBT_INFO, "USD"): [
        {"kind": "fred", "id": "GFDEGDQ188S", "unit": "% av BNP",
         "label": "Federal skuld (kvartal) — annat mått än EU:s, bara information"},
    ],
    # ── Växelkurs: valutans pris i USD-termer (SEK per USD, USD per EUR) ──
    (FX, "SEK"): [
        {"kind": "riksbank", "id": "SEKUSDPMI", "unit": "SEK per USD", "label": "USD/SEK (Riksbanken)"},
        {"kind": "fred", "id": "DEXSDUS", "unit": "SEK per USD", "label": "USD/SEK (Fed via FRED)"},
    ],
    (FX, "EUR"): [
        {"kind": "ecb", "flow": "EXR", "key": "D.USD.EUR.SP00.A", "unit": "USD per EUR", "label": "EUR/USD (ECB)"},
        {"kind": "fred", "id": "DEXUSEU", "unit": "USD per EUR", "label": "EUR/USD (Fed via FRED)"},
    ],
    # ── Reala tillgångar (USD) ──
    (COPPER, "USD"): [
        {"kind": "borsdata", "id": 21035, "unit": "USD/lb", "label": "Koppar (Börsdata)"},
        {"kind": "yahoo", "id": "HG=F", "unit": "USD/lb", "label": "Koppar terminer (Yahoo)"},
    ],
    (OIL, "USD"): [
        {"kind": "borsdata", "id": 21046, "unit": "USD/fat", "label": "Brent (Börsdata)"},
        {"kind": "yahoo", "id": "BZ=F", "unit": "USD/fat", "label": "Brent terminer (Yahoo)"},
    ],
    (BTC, "USD"): [
        {"kind": "yahoo", "id": "BTC-USD", "unit": "USD", "label": "Bitcoin (Yahoo)"},
    ],
}

# Guld och silver: Börsdata spot är primär (2006–); Yahoo-terminer fyller tiden före
# Börsdatas första datum och märks "terminspris". Ingen nivåjustering vid skarven.
ASSET_SPLICE: dict = {
    GOLD: {"primary": {"kind": "borsdata", "id": 21031, "unit": "USD/oz", "label": "Guld spot (Börsdata)"},
           "backfill": {"kind": "yahoo", "id": "GC=F", "unit": "USD/oz", "label": "Guld terminspris (Yahoo GC=F)"}},
    SILVER: {"primary": {"kind": "borsdata", "id": 21032, "unit": "USD/oz", "label": "Silver spot (Börsdata)"},
             "backfill": {"kind": "yahoo", "id": "SI=F", "unit": "USD/oz", "label": "Silver terminspris (Yahoo SI=F)"}},
}
SPOT, FUTURES = "spot", "terminspris"
