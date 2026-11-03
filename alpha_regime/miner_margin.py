"""
alpha_regime/miner_margin.py
Gruvorna mot marginalen — gruv-ETF:ns pris per dollar marginal per uns. KONTEXT, ingen signal.

GLD/GDX-kvoten bortser från marginalerna. Gruvorna tjänar (guldpris − AISC) per uns, och med guldet
långt över branschens AISC är marginalen per uns flera gånger 2015–2019 års nivå. Måttet:

    M = gruv-ETF:ns kurs / (guldpris − branschens AISC)

Låg percentil av M i sin egen historik = gruvorna billiga per dollar marginal. GDXJ räknas mot samma
bransch-AISC som GDX (en approximation — juniorbolagens egna kostnader finns inte i tabellen).

CONTEXT ONLY — never feeds any signal, count or confirmation: not the engine's +1 ACCUMULATE
confirmation, not _classify / _classify_driver, not the 'Gummisnoddar' count in the UI.

Lägga in ett nytt kvartal (gör det varje kvartal)
-------------------------------------------------
World Gold Council publicerar kvartal Q:s AISC ungefär ett kvartal senare: i nästa Gold Demand Trends
(supply-avsnittet) eller i en Gold Focus-artikel 3–5 månader efter kvartalets slut.

1. Lägg en rad SIST i AISC_TABLE enligt mallen (platshållare — fyll i de publicerade värdena):
       AiscRow("YYYYQn", <USD/uns>, "YYYY-MM-DD", "MF_Q", "World Gold Council/Metals Focus",
               "<källa: rapport och avsnitt>", "<url>", "<not: förändring k/k, å/å, osäkerhet>"),
   - period: "YYYYQn" för ett kvartal, "YYYY" för ett helår.
   - aisc: USD/uns, som det FÖRST publicerades.
   - published: dagen siffran först publicerades (backtestet ser värdet först då). Vet du bara
     månaden — ta månadens sista dag.
2. Ändra aldrig en gammal rad till ett reviderat värde: tabellen håller första publicerade värdet,
   annars ser backtestet in i framtiden. Skriv revisionen i note i stället.
3. Kör: python -m pytest -q tests/test_miner_margin.py

Panelen varnar när det nyaste kvartalet i tabellen är mer än två kvartal äldre än i dag.

Serier (series): MF_Q = Metals Focus kvartal via WGC (huvudserien) · MF_A = Metals Focus helår ·
MF_A_DERIVED = härlett ur en WGC-mening · WGC_GDT_EARLY = tidiga WGC-tal, leverantör okänd ·
SP_A = S&P Global (primära guldgruvor, annan metod) · GFMS_A = Refinitiv GFMS (annan metod).
Helår och kvartal ligger i samma lista men blandas inte: ett helår gäller bara de kvartal i sitt år
som saknar en egen kvartalsrad (år med bara helår: alla fyra kvartalen).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from alpha_regime.commodity_ratios import _cache_6h, _download_close, _percentile_of

logger = logging.getLogger(__name__)

GOLD_TICKER = "GC=F"          # gold via market_prices — no fallback (DATA_GAP when missing)
MIN_MARGIN_USD = 50.0         # margin ≤ this (USD/oz) → M is NaN: a thin margin makes M explode
STALE_AFTER_QUARTERS = 2      # newest AISC period more than this many quarters before today → stale
_MIN_POINTS = 100             # same floor as the ratio cards


# ── Branschens AISC (USD/uns), första publicerade värdet ─────────────────────
@dataclass(frozen=True)
class AiscRow:
    period: str      # "2025Q4" (quarter) or "2025" (full year)
    aisc: float      # USD/oz as first published
    published: str   # "YYYY-MM-DD" first publication (approximate: month-end when only the month is known)
    series: str      # MF_Q | MF_A | MF_A_DERIVED | WGC_GDT_EARLY | SP_A | GFMS_A (see the module docstring)
    publisher: str   # short name shown on the card
    source: str
    url: str
    note: str = ""


_WGC_MF = "World Gold Council/Metals Focus"
MAIN_SERIES = "MF_Q"          # the quarterly main series; every other series is named on the card's caption
MAIN_PUBLISHER = _WGC_MF
_GF_2022_03 = ("https://www.gold.org/goldhub/gold-focus/2022/03/"
               "gold-mining-aisc-remained-flat-q421-significantly-higher-2021")
_GF_2023_04 = ("https://www.gold.org/goldhub/gold-focus/2023/04/"
               "gold-miners-costs-reached-record-high-2022-dropped-final-quarter-year")
_GDT = "https://www.gold.org/goldhub/research/gold-demand-trends/"
_SEC_GDT_Q3_2015 = "https://www.sec.gov/Archives/edgar/data/1222333/000119312515375901/d30771dfwp.htm"

AISC_TABLE: list[AiscRow] = [
    AiscRow("2012", 1116, "2023-04-30", "MF_A_DERIVED", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Gold miners' costs reached a record high in 2022 but dropped in "
            "the final quarter of the year'", _GF_2023_04,
            "Härlett, inte angivet: 2022 = 1 276 var 'US$160/oz över förra rekordet 2012' → 1 116. "
            "Datum ungefärligt (månadsslut)."),
    AiscRow("2013Q1", 1113, "2015-11-12", "WGC_GDT_EARLY", "World Gold Council",
            "WGC Gold Demand Trends Q3 2015, supply section", _SEC_GDT_Q3_2015,
            "'fallen from US$1,113/oz in Q1 2013 to US$900/oz in Q2 2015'. Datum ungefärligt. Okänt om "
            "talet kommer från GFMS eller Metals Focus — nivån kan avvika från den senare MF-serien."),
    AiscRow("2015Q2", 900, "2015-11-12", "WGC_GDT_EARLY", "World Gold Council",
            "WGC Gold Demand Trends Q3 2015, supply section", _SEC_GDT_Q3_2015,
            "Samma mening som 2013Q1 (−19 % sedan Q1 2013). Datum ungefärligt, leverantör ej bekräftad."),
    AiscRow("2015Q4", 832, "2016-05-12", "WGC_GDT_EARLY", "World Gold Council",
            "WGC Gold Demand Trends Q1 2016, supply section",
            "https://www.sec.gov/Archives/edgar/data/1222333/000119312516588423/d164713dfwp.htm",
            "Låg–medel säkerhet: 'AISC fell 11% y/y to US$832/oz in Q4 2015' sågs i en sökning men "
            "hittades inte igen. Datum ungefärligt."),
    AiscRow("2016", 879, "2017-06-30", "SP_A", "S&P Global",
            "S&P Global Market Intelligence, primary gold mines (Mine Economics data)",
            "https://spglobal.com/market-intelligence/en/news-insights/research/gold-mining-companies-hold-operating-costs",
            "Annan metod och täckning (primära guldgruvor), ungefär oförändrat mot 2015. Publiceringsdatum "
            "ej verifierat — 2017-06-30 är en försiktig platshållare."),
    AiscRow("2018", 897, "2019-04-30", "GFMS_A", "Refinitiv GFMS",
            "Refinitiv GFMS Gold Survey 2019 (press coverage)",
            "https://www.businessghana.com/site/news/Business/186786/Refinitiv-publishes-GFMS-Gold-Survey-2019",
            "Annan metod (GFMS): +4 % till 897, total cash cost 696. Metals Focus 2018 är implicit ~923 "
            "(941/1,02), ~3 % högre. Datum ungefärligt."),
    AiscRow("2019", 941, "2020-06-24", "MF_A", "Metals Focus",
            "Metals Focus Gold Focus 2020 (reported by Kitco)",
            "https://www.kitco.com/news/2020-06-24/Metals-Focus-gold-prices-to-challenge-all-time-highs-by-year-end.html",
            "'average all-in sustaining costs climbed 2% last year to $941 an ounce'. WGC:s marginal 451 "
            "mot LBMA-snittet ~1 393 ger ~942."),
    AiscRow("2020Q2", 972, "2022-03-31", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Gold mining AISC remained flat in Q4'21, but significantly "
            "higher in 2021'", _GF_2022_03,
            "972 står bara i den här artikeln från mars 2022 — därav det sena, försiktiga datumet "
            "(Q3'20-artikeln i dec 2020 antydde det: 953 var −2 % k/k)."),
    AiscRow("2020Q3", 953, "2020-12-31", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Higher prices and lower costs significantly raise miners' "
            "margins in Q3'20'",
            "https://www.gold.org/goldhub/gold-focus/2020/12/higher-prices-and-lower-costs-significantly-raise-miners-margins-q320",
            "−2 % k/k; marginal 959. Datum ungefärligt (artikelns månadsslut)."),
    AiscRow("2020Q4", 983, "2021-03-31", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Sustaining capital expenditure pushes AISC higher in Q4'20'",
            "https://www.gold.org/goldhub/gold-focus/2021/03/sustaining-capital-expenditure-pushes-aisc-higher-q420",
            "+1,8 % k/k. Senare reviderat till ~998. Datum ungefärligt."),
    AiscRow("2021Q1", 1048, "2021-06-30", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Rising gold mining costs Q1'21'",
            "https://www.gold.org/goldhub/gold-focus/2021/06/gold-mining-costs-continue-rising-q1-21",
            "+5 % k/k, högst sedan Q2'13. Datum ungefärligt."),
    AiscRow("2021Q2", 1067, "2021-09-30", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'AISC up 10% year-on-year in Q2'21'",
            "https://www.gold.org/goldhub/gold-focus/2021/09/aisc-10-year-year-q2%E2%80%9921",
            "+10 % å/å; marginal 748. Datum ungefärligt."),
    AiscRow("2021Q3", 1123, "2021-12-31", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Gold miners' AISC reach their highest since 2013'",
            "https://www.gold.org/goldhub/gold-focus/2021/12/gold-miners%E2%80%99-aisc-reach-their-highest-2013",
            "+4 % k/k. Reviderat till 1 131 i mars 2022. Datum ungefärligt."),
    AiscRow("2021Q4", 1129, "2022-03-31", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Gold mining AISC remained flat in Q4'21, but significantly "
            "higher in 2021'", _GF_2022_03,
            "Oförändrat k/k; marginal 731. Datum ungefärligt."),
    AiscRow("2021", 1068, "2022-03-31", "MF_A", _WGC_MF,
            "WGC Gold Focus (Metals Focus) Mar 2022; the same figure is in Metals Focus Gold Focus 2022",
            _GF_2022_03,
            "+7 % å/å. Senare reviderat till ~1 081. Alla fyra kvartal finns, så helåret används inte."),
    AiscRow("2022Q1", 1232, "2022-07-31", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'AISC in the gold mining industry reached their highest level on "
            "record in Q1'22'",
            "https://www.gold.org/goldhub/gold-focus/2022/07/aisc-gold-mining-industry-reached-their-highest-level-record-q122-0",
            "+9 % k/k, rekord sedan 2012; marginal 646. Datum ungefärligt."),
    AiscRow("2022Q2", 1289, "2022-09-30", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Rising energy prices push AISC to a record high in Q2'22'",
            "https://www.gold.org/goldhub/gold-focus/2022/09/rising-energy-prices-push-aisc-record-high-q222",
            "+3 % k/k, +18 % å/å; marginal 585. Q3'22-trycket (1 289 vid +1 % k/k) antyder att Q2 "
            "reviderades till ~1 276. Datum ungefärligt."),
    AiscRow("2022Q3", 1289, "2022-12-31", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Gold miners' costs rise again in Q3'22 but the rate of increase "
            "has slowed'",
            "https://www.gold.org/goldhub/gold-focus/2022/12/gold-miners-costs-rise-again-q322-rate-increase-has-slowed",
            "+1 % k/k, +14 % å/å. Samma tal som Q2:s första tryck eftersom Q2 reviderades ned. "
            "Datum ungefärligt."),
    AiscRow("2022Q4", 1279, "2023-04-30", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Gold miners' costs reached a record high in 2022 but dropped in "
            "the final quarter of the year'", _GF_2023_04,
            "−0,9 % k/k. Datum ungefärligt."),
    AiscRow("2022", 1276, "2023-04-30", "MF_A", _WGC_MF,
            "WGC Gold Focus (Metals Focus) Apr 2023; also WGC Gold Demand Trends Q1 2023 supply section",
            _GF_2023_04,
            "Rekord, +18 % å/å. Alla fyra kvartal finns, så helåret används inte."),
    AiscRow("2023Q1", 1358, "2023-07-31", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Gold miners' average AISC resumes rising trend in Q1'23 and "
            "reaches a new record high'",
            "https://www.gold.org/goldhub/gold-focus/2023/07/"
            "gold-miners-average-aisc-resumes-rising-trend-in-q1-23-and-reaches-new-record-high",
            "+6 % k/k, +10 % å/å. Senare reviderat till ~1 302 (Q2'23 = 1 315 var +1 % k/k). "
            "Datum ungefärligt."),
    AiscRow("2023Q2", 1315, "2023-11-30", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Gold miners' AISC still rising, but at a slower pace'",
            "https://www.gold.org/goldhub/gold-focus/2023/11/gold-miners-aisc-still-rising-slower-pace",
            "+1 % k/k, +6 % å/å. Kan ha publicerats först i GDT Q3 2023 (31 okt 2023) — ej bekräftat."),
    AiscRow("2023Q3", 1343, "2024-01-31", "MF_Q", _WGC_MF,
            "WGC Gold Demand Trends Full Year 2023, supply section",
            _GDT + "gold-demand-trends-full-year-2023/supply",
            "Rekord, +5 % å/å, +3 % k/k. Datum = GDT-släppet (ungefärligt)."),
    AiscRow("2023Q4", 1342, "2024-04-30", "MF_Q", _WGC_MF,
            "WGC Gold Demand Trends Q1 2024, supply section; also WGC Gold Focus May 2024",
            _GDT + "gold-demand-trends-q1-2024/supply",
            "+2 % k/k (alltså Q3 reviderat till ~1 316), +7 % å/å; marginal 635. Datum ungefärligt."),
    AiscRow("2024Q1", 1439, "2024-07-30", "MF_Q", _WGC_MF,
            "WGC Gold Demand Trends Q2 2024, supply section",
            _GDT + "gold-demand-trends-q2-2024/supply",
            "LÅG–MEDEL säkerhet: från en enda sökningssammanfattning, ej bekräftad. Senare tryck antyder "
            "~1 374–1 384 efter revision (2024Q2 = 1 388 var +1 % k/k, 2025Q1 = 1 536 var +11 % å/å). "
            "Ägaren: kontrollera mot supply-avsnittet i GDT Q2 2024 och rätta till första trycket om det "
            "skiljer — gissa inte ett tal."),
    AiscRow("2024Q2", 1388, "2024-10-31", "MF_Q", _WGC_MF,
            "WGC Gold Focus (Metals Focus): 'Gold cost drivers: a veritable pick and mix'",
            "https://www.gold.org/goldhub/gold-focus/2024/10/gold-cost-drivers-veritable-pick-and-mix",
            "+1 % k/k, +6 % å/å; marginal 950, högst sedan Q1'12. Datum ungefärligt."),
    AiscRow("2024Q3", 1456, "2025-02-05", "MF_Q", _WGC_MF,
            "WGC Gold Demand Trends Q4/FY 2024 supply section; Metals Focus 'Ever upwards for AISC' "
            "(WGC, Mar 2025)",
            "https://www.gold.org/download/file/19538/MetalsFocus_Production_costs_Q324.pdf",
            "+4 % k/k, +9 % å/å, högst i serien sedan Q1'10. Senare reviderat till ~1 472. Datum ungefärligt."),
    AiscRow("2024Q4", 1438, "2025-04-30", "MF_Q", _WGC_MF,
            "WGC Gold Demand Trends Q1 2025, supply section",
            _GDT + "gold-demand-trends-q1-2025/supply",
            "+8 % å/å, ned k/k. Senare reviderat till ~1 422. Datum ungefärligt."),
    AiscRow("2024", 1399, "2025-06-05", "MF_A", "Metals Focus",
            "Metals Focus Gold Focus 2025 (Modern Mining summary)",
            "https://www.crown.co.za/latest-news/modern-mining-latest-news/32899-metals-focus-publishes-gold-focus-2025",
            "+8 % å/å. Reviderat till ~1 386 i Gold Focus 2026. Alla fyra kvartal finns, så helåret "
            "används inte."),
    AiscRow("2025Q1", 1536, "2025-07-31", "MF_Q", _WGC_MF,
            "WGC Gold Demand Trends Q2 2025, supply section",
            _GDT + "gold-demand-trends-q2-2025/supply",
            "Rekord, +11 % å/å, +6 % k/k. Datum ungefärligt."),
    AiscRow("2025Q2", 1578, "2025-10-30", "MF_Q", _WGC_MF,
            "WGC Gold Demand Trends Q3 2025, supply section",
            _GDT + "gold-demand-trends-q3-2025/supply",
            "+12 % å/å. Senare implicit ~1 589. Datum ungefärligt."),
    AiscRow("2025Q3", 1605, "2026-01-29", "MF_Q", _WGC_MF,
            "WGC Gold Demand Trends Q4 and Full Year 2025, supply section",
            _GDT + "gold-demand-trends-full-year-2025/supply",
            "+9 % å/å, +1 % k/k. Datum ungefärligt."),
    AiscRow("2025Q4", 1706, "2026-04-29", "MF_Q", _WGC_MF,
            "WGC Gold Demand Trends Q1 2026, supply section",
            _GDT + "gold-demand-trends-q1-2026/supply",
            "Rekord, +20 % å/å. Inte samma sak som en tredjeparts produktionsviktade median på 1 709. "
            "Datum ungefärligt."),
    AiscRow("2025", 1552, "2026-06-05", "MF_A", "Metals Focus",
            "Metals Focus Gold Focus 2026 (reported by Kitco and Modern Mining)",
            "https://www.kitco.com/news/article/2026-06-05/"
            "gold-demand-will-drop-year-even-supply-increases-average-price-will-still",
            "+12 % å/å. ~3,4 % under snittet av kvartalens första tryck (~1 606) — helår och kvartal går "
            "inte att byta mot varandra. Alla fyra kvartal finns, så helåret används inte."),
    AiscRow("2026Q1", 1785, "2026-08-25", "MF_Q", _WGC_MF,
            "WGC Gold Focus: 'Record gold prices outpace rising mining costs in Q1'26'",
            "https://www.gold.org/goldhub/gold-focus/2026/08/record-gold-prices-outpace-rising-mining-costs-q126",
            "+5 % k/k, +16 % å/å (28:e raka ökningen å/å); rekordmarginal 3 076; royalties ~12 % av AISC. "
            "Kan ha publicerats först i GDT Q2 2026 (sent juli 2026) — ej bekräftat. Datum ungefärligt."),
]


# ── Periods ──────────────────────────────────────────────────────────────────
def _parse_period(period: str) -> tuple:
    """'2025Q4' → (2025, 4); '2025' → (2025, None)."""
    p = str(period).strip().upper()
    if len(p) == 6 and p[:4].isdigit() and p[4] == "Q" and p[5] in "1234":
        return int(p[:4]), int(p[5])
    if len(p) == 4 and p.isdigit():
        return int(p), None
    raise ValueError(f"okänd AISC-period: {period!r}")


def _qord(year: int, quarter: int) -> int:
    """Quarter ordinal: consecutive quarters differ by 1."""
    return int(year) * 4 + int(quarter) - 1


def _period_end_ord(period: str) -> int:
    """Ordinal of the last quarter a period covers (a full year ends in its Q4)."""
    y, q = _parse_period(period)
    return _qord(y, q or 4)


def newest_period(rows: Optional[Iterable[AiscRow]] = None) -> str:
    """The newest period in the table ('2026Q1'); a quarter row wins over a full year ending in it."""
    rows = AISC_TABLE if rows is None else list(rows)
    if not rows:
        return ""
    best = max(rows, key=lambda r: (_period_end_ord(r.period), _parse_period(r.period)[1] is not None))
    return best.period


def is_stale(newest: str, today=None) -> bool:
    """True when the newest AISC period is more than STALE_AFTER_QUARTERS quarters before today's quarter."""
    if not newest:
        return True
    t = pd.Timestamp(today if today is not None else date.today())
    return _qord(t.year, t.quarter) - _period_end_ord(newest) > STALE_AFTER_QUARTERS


# ── AISC per day ─────────────────────────────────────────────────────────────
def _quarter_rows(rows: list) -> tuple:
    """(sorted quarter ordinals, the row in force from each). A quarter row applies to its quarter; a
    full-year row applies to every quarter of its year without a quarter row of its own."""
    by_q: dict = {}
    for row in rows:
        y, q = _parse_period(row.period)
        if q is not None:
            by_q[_qord(y, q)] = row
    for row in rows:
        y, q = _parse_period(row.period)
        if q is None:
            for k in range(1, 5):
                by_q.setdefault(_qord(y, k), row)
    keys = sorted(by_q)
    return np.asarray(keys, dtype=np.int64), [by_q[k] for k in keys]


def _lookup(date_ords: np.ndarray, rows: list) -> list:
    """Row in force for each quarter ordinal: its own quarter's row, else the latest earlier one
    (forward-fill over gaps and after the table ends). None before the first row."""
    keys, vals = _quarter_rows(rows)
    if len(keys) == 0:
        return [None] * len(date_ords)
    pos = np.searchsorted(keys, date_ords, side="right") - 1
    return [vals[p] if p >= 0 else None for p in pos]


def _applied_rows(index, point_in_time: bool = False, rows: Optional[Iterable[AiscRow]] = None) -> list:
    """The AiscRow in force on each date of `index` (None where nothing is known).

    point_in_time=True: on each date only rows published on or before that date exist — the table as it
    looked that day — so the date's own quarter (not yet published) gets the latest value known then."""
    rows = AISC_TABLE if rows is None else list(rows)
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    ords = np.asarray(idx.year * 4 + idx.quarter - 1, dtype=np.int64)
    if not point_in_time:
        return _lookup(ords, rows)

    pubs = sorted({pd.Timestamp(r.published) for r in rows})
    vintage = np.searchsorted(np.asarray(pubs, dtype="datetime64[ns]"),
                              idx.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    out: list = [None] * len(idx)
    for v in np.unique(vintage):
        if v < 0:
            continue
        known = [r for r in rows if pd.Timestamp(r.published) <= pubs[v]]
        sel = np.nonzero(vintage == v)[0]
        for i, row in zip(sel, _lookup(ords[sel], known)):
            out[i] = row
    return out


def aisc_series(index, point_in_time: bool = False, rows: Optional[Iterable[AiscRow]] = None) -> pd.Series:
    """Daily industry AISC (USD/oz) on `index`.

    A quarter's value applies to its quarter; a year with only a full-year value applies it to all four
    quarters; the latest known value is forward-filled over gaps and after the table ends; NaN before the
    first row. point_in_time=True (the backtest): a value is used only from its 'published' date — before
    that the previous known value."""
    applied = _applied_rows(index, point_in_time, rows)
    return pd.Series([float(r.aisc) if r is not None else np.nan for r in applied],
                     index=index, dtype=float, name="aisc")


def margin_ratio(miner_close: pd.Series, gold_close: pd.Series, aisc) -> pd.Series:
    """M = miner ETF price / (gold − AISC). A margin ≤ MIN_MARGIN_USD (50 USD/oz) is NaN — a thin or
    negative margin makes M explode or flip sign — and so is a bad bar (miner ≤ 0, inf, NaN; gold or AISC
    inf/NaN: an infinite margin would otherwise give M = 0 and count as history)."""
    margin = gold_close - aisc
    m = miner_close / margin.where(np.isfinite(margin) & (margin > MIN_MARGIN_USD))
    m = m.where(miner_close > 0)
    return m.replace([np.inf, -np.inf], np.nan).rename("M")


def _quarter_label(ts) -> str:
    return f"{ts.year}Q{ts.quarter}"


def _aisc_spans(dates, applied) -> list:
    """Runs of the same AISC row over `dates`, in order: [(period, publisher, series, first quarter, last
    quarter)], e.g. ('2016', 'S&P Global', 'SP_A', '2016Q4', '2017Q4') — which source the window really
    uses, so the card's caption can name the years that are not the quarterly main series."""
    spans: list = []
    for d, row in zip(dates, applied):
        if row is None:
            continue
        q = _quarter_label(d)
        if spans and spans[-1][0] == row.period:
            spans[-1][4] = q
        else:
            spans.append([row.period, row.publisher, row.series, q, q])
    return [tuple(s) for s in spans]


def _sv_int(v: float) -> str:
    """Whole number the Swedish way: '−1 285' (minus sign, no-break space)."""
    return f"{v:,.0f}".replace(",", " ").replace("-", "−")


# ── Result ───────────────────────────────────────────────────────────────────
@dataclass
class MarginResult:
    miner: str
    status: str = "DATA_GAP"              # OK | DATA_GAP
    error: Optional[str] = None
    current: Optional[float] = None      # M on the latest bar
    percentile: Optional[float] = None   # of current M in its own history (whole window)
    median: Optional[float] = None       # median M over the same window
    miner_price: Optional[float] = None
    gold_price: Optional[float] = None
    aisc: Optional[float] = None         # AISC used on the latest bar
    aisc_period: str = ""                # its period, e.g. "2026Q1"
    aisc_source: str = ""                # its publisher, e.g. "World Gold Council/Metals Focus"
    margin: Optional[float] = None       # gold − AISC on the latest bar, USD/oz
    as_of: str = ""                      # date of the latest bar
    history_start: str = ""              # first date with a valid M
    history_years: float = 0.0
    stale: bool = False                  # newest AISC period > 2 quarters before today
    newest_period: str = ""
    aisc_spans: list = field(default_factory=list)   # AISC rows used over the valid window (_aisc_spans)
    sparkline_dates: list = field(default_factory=list)
    sparkline_values: list = field(default_factory=list)


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def build_result(miner: str, miner_close: Optional[pd.Series], gold_close: Optional[pd.Series],
                 today=None, rows: Optional[Iterable[AiscRow]] = None) -> MarginResult:
    """M for one miner ETF from given close series (no network). Percentile and median over the whole
    aligned window — the same convention as the ratio cards."""
    rows = AISC_TABLE if rows is None else list(rows)
    newest = newest_period(rows)
    res = MarginResult(miner=miner, newest_period=newest, stale=is_stale(newest, today))

    if gold_close is None or gold_close.dropna().empty:
        res.error = (f"Guldpriset ({GOLD_TICKER}) saknas — utan det går marginalen inte att räkna "
                     f"(ingen reservkälla).")
        return res
    if miner_close is None or miner_close.dropna().empty:
        res.error = f"Kursen för {miner} saknas."
        return res

    aligned = pd.concat({"m": miner_close, "g": gold_close}, axis=1, sort=True).dropna()
    if aligned.empty:
        res.error = f"{miner} och guldpriset har inga gemensamma handelsdagar."
        return res
    applied = _applied_rows(aligned.index, False, rows)
    aisc = pd.Series([float(r.aisc) if r is not None else np.nan for r in applied], index=aligned.index)
    m = margin_ratio(aligned["m"], aligned["g"], aisc)
    valid = m.dropna()
    if len(valid) < _MIN_POINTS:
        res.error = f"För kort historik för {miner} mot marginalen ({len(valid)} dagar, behöver {_MIN_POINTS})."
        return res

    last_row = applied[-1]
    gold_now, miner_now = float(aligned["g"].iloc[-1]), float(aligned["m"].iloc[-1])
    cur = float(m.iloc[-1])
    if last_row is None or not _finite(cur):
        if last_row is None:
            res.error = f"Ingen AISC för {aligned.index[-1].date()}."
        elif not (_finite(gold_now) and gold_now > 0):
            res.error = f"Sista stängningen för guldet ({GOLD_TICKER}) är trasig."
        elif gold_now - float(last_row.aisc) <= MIN_MARGIN_USD:
            res.error = (f"Marginalen är bara {_sv_int(gold_now - float(last_row.aisc))} USD/uns "
                         f"(≤ {_sv_int(MIN_MARGIN_USD)}) — måttet säger inget då.")
        else:
            res.error = f"Sista stängningen för {miner} är trasig."
        return res

    arr = valid.to_numpy(dtype=float)
    med = float(np.median(arr))
    res.current = float(f"{cur:.6g}")
    res.percentile = round(_percentile_of(arr, cur), 1)
    res.median = float(f"{med:.6g}")
    res.miner_price = round(miner_now, 4)
    res.gold_price = round(gold_now, 2)
    res.aisc = float(last_row.aisc)
    res.aisc_period = last_row.period
    res.aisc_source = last_row.publisher
    res.margin = round(gold_now - float(last_row.aisc), 2)
    res.as_of = str(aligned.index[-1].date())
    start, end = valid.index[0], valid.index[-1]
    res.history_start = str(start.date())
    res.history_years = round((end - start).days / 365.25, 1)
    ok = m.notna().to_numpy()
    res.aisc_spans = _aisc_spans(aligned.index[ok], [row for row, keep in zip(applied, ok) if keep])
    weekly = valid.resample("W").last().dropna()
    res.sparkline_dates = [str(d.date()) for d in weekly.index]
    res.sparkline_values = [round(float(v), 6) for v in weekly.values]
    res.status = "OK"
    return res


class _NoPrices(Exception):
    """Gold or the miner download came back empty. Raised INSIDE the cached function so a transient failure
    (e.g. Yahoo's 'Too Many Requests') is never cached for six hours — the next render tries again (the same
    pattern as market_prices._NoData). Carries the DATA_GAP result to show meanwhile."""

    def __init__(self, result: MarginResult):
        super().__init__(result.error)
        self.result = result


@_cache_6h
def _compute_cached(miner: str) -> MarginResult:
    """Download and compute; only a result built from real closes reaches the 6h cache."""
    gold = _download_close(GOLD_TICKER)
    if gold.dropna().empty:
        raise _NoPrices(build_result(miner, None, gold))
    close = _download_close(miner)
    if close.dropna().empty:
        raise _NoPrices(build_result(miner, close, gold))
    return build_result(miner, close, gold)


def compute(miner: str = "GDX") -> MarginResult:
    """GDX or GDXJ against the industry AISC, on the 10-year closes the ratio cards use. Cached 6h — but a
    missing download is a DATA_GAP that is NOT cached, so the next render retries.

    CONTEXT ONLY — never feeds any signal, count or confirmation."""
    miner = (miner or "GDX").strip().upper()
    try:
        return _compute_cached(miner)
    except _NoPrices as gap:
        return gap.result
