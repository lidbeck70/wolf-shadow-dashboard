#!/usr/bin/env python3
"""
pgm_band_study.py — PGM-snoddarna (docs/forskning/pgm_snoddar_spec.md, version 1).

Förregistrerat backtest av köpstödet i Gold/Platinum (GC=F/PL=F) och Gold/Palladium (GC=F/PA=F): hade det
fungerat, med platinans långa fall och sidledes period 2015–2024 (risken: köpstöd för tidigt, en fallande kniv)?

Signalerna räknas point-in-time varje handelsdag på terminerna med panelens egna funktioner (_percentile_of,
_classify, _classify_driver och buy_support.ENGINE_DRIVERS): S0 nuvarande regel (specens namn för drivarregeln som
gällde när testet registrerades — sedan ägarens beslut 2026-11-03 är båda PGM-snoddarna bara kontext,
buy_support.CONTEXT_ONLY), S1 bara kvoten, S2 kvoten och metallterminen över MA200. Utfallen räknas på de fysiskt
backade ETF:erna (PPLT, PALL) mot GLD, med köp första dagen efter signaldagen. Därefter episoder, utfall på
63/126/252 dagar, baslinjer, bootstrapintervall (beskrivande), domarna H1–H3 per metall, gruvorna (IMPUY, SBSW)
och korsutfallen (beskrivande) och förslagen.

Percentiler, episoder, utfall, sammanfattningar och bootstrap är miner_margin_study:s funktioner (samma metod som
gruvornas test). De rena funktionerna står först; run()/main() hämtar kurserna (market_prices, "max") och sparar
pgm_band_study.json i Gisten. Körs manuellt i Actions (pgm-band-study.yml). Studien ändrar inga regler. Avslutas
med 0 även när hypoteserna faller, och med 1 bara när kurser som krävs saknas.
Env: GITHUB_TOKEN (Gisten), BORSDATA_API_KEY (market_prices provar Börsdata först).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

import miner_margin_study as ms
from alpha_regime import buy_support as bs
from alpha_regime import commodity_ratios as cr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pgm_band_study")

BLOB = "pgm_band_study.json"
SPEC = "docs/forskning/pgm_snoddar_spec.md"
SPEC_VERSION = 1

START = "2012-01-01"          # utfall räknas för episoder som börjar den här dagen eller senare
PERIOD = "max"                # längsta historiken market_prices ger
WINDOW_YEARS = 10             # percentiler över (t − 10 år, t], som kvotkorten
MIN_YEARS = 3                 # fönstret måste börja minst tre år före t
GAP = 63                      # handelsdagar utan signal innan en ny episod börjar
HORIZONS = (63, 126, 252)     # handelsdagar efter köpet (miner_margin_study.summarize läser samma värden)
DD_DAYS = 252                 # värsta nedgången inom så här många dagar efter köpet
BASE_STEP = 63                # baslinjen "var 63:e dag"
RATIO_MIN = 90.0              # S1 och S2: P_kvot ≥ 90
MA_DAYS = 200                 # S2: metallterminens stängning över sitt 200-dagars medelvärde
MIN_EPISODES = 3              # episoder med hel 252-dagarshorisont för en dom (H1–H3)
BLOCK = 63                    # bootstrapens blocklängd
N_BOOT = 5000
SEED = 20261104
MIN_DAYS = 756                # färre dagskurser än tre år = kursen saknas (miner_margin_study.missing)

GOLD_ETF = "GLD"
BANDS = (("platina", "gold_platinum"), ("palladium", "gold_palladium"))   # metall och panelens kvotnyckel
GENITIVE = {"platina": "platinans", "palladium": "palladiums"}           # rapportens text
MINERS = ("IMPUY", "SBSW")                                                # sekundärt, bara beskrivande
REQUIRED = ("GC=F", "PL=F", "PA=F", GOLD_ETF, "PPLT", "PALL")
TICKERS = REQUIRED + MINERS
STRETCHED = bs.STRETCHED                  # "RUBBER_BAND_STRETCHED"
S0_DRIVERS = bs.ENGINE_DRIVERS            # drivarregeln (DENOMINATOR_CHEAP, BOTH): PGM-snoddarnas +1 till 2026-11-03
SIGNALS = (("S0", "nuvarande regel"), ("S1", "bara kvoten"), ("S2", "kvoten och trenden"))
HYPOTHESES = (("H1", "S0"), ("H2", "S1"), ("H3", "S2"))
PASS, FAIL, TOO_FEW = "GODKÄND", "EJ GODKÄND", "FÖR FÅ EPISODER"


# ── Signaler (point-in-time) ─────────────────────────────────────────────────
def ma_frame(metal, days: int = MA_DAYS) -> pd.DataFrame:
    """Metallterminens MA200 och "över MA200" (1.0 eller 0.0) på terminens egna handelsdagar.

    MA200(t) är snittet av de `days` senaste stängningarna till och med t, alltså bara data ≤ t. NaN där färre
    än `days` stängningar finns."""
    s = ms._clean(metal)
    ma = s.rolling(days, min_periods=days).mean()
    above = (s > ma).astype(float).where(ma.notna())
    return pd.DataFrame({"close": s, "ma200": ma, "above_ma": above})


def signal_frame(gold, metal, direction: str = "high", years: int = WINDOW_YEARS, min_years: int = MIN_YEARS,
                 ma_days: int = MA_DAYS) -> pd.DataFrame:
    """Signalernas underlag och S0–S2 per handelsdag i signalkalendern (dagar med både GC=F och metallterminen).

    Kvoten och benen kopplas som kvotkortet (inre koppling) och får sina percentiler point-in-time över
    (t − 10 år, t] med minst tre år (miner_margin_study.pit_percentiles). MA200 räknas på terminens egna dagar.
    ok_S* = signalen kan räknas den dagen."""
    if direction != "high":
        raise ValueError("S1 och S2 är förregistrerade för riktningen 'high' (P_kvot ≥ 90)")
    gold, metal = ms._clean(gold), ms._clean(metal)
    fr = pd.concat({"gc": gold, "metal": metal}, axis=1, sort=True).dropna()
    fr["ratio"] = fr["gc"] / fr["metal"]
    fr["p_ratio"] = ms.pit_percentiles(fr["ratio"], years, min_years)
    fr["p_gc"] = ms.pit_percentiles(fr["gc"], years, min_years)
    fr["p_metal"] = ms.pit_percentiles(fr["metal"], years, min_years)
    ma = ma_frame(metal, ma_days)
    fr["ma200"] = ma["ma200"].reindex(fr.index)
    fr["above_ma"] = ma["above_ma"].reindex(fr.index)

    ok_ratio = fr[["p_ratio", "p_gc", "p_metal"]].notna().all(axis=1).to_numpy()
    status = [cr._classify(p, direction) if ok else None for p, ok in zip(fr["p_ratio"], ok_ratio)]
    driver = [cr._classify_driver(a, b, direction) if ok else None
              for a, b, ok in zip(fr["p_gc"], fr["p_metal"], ok_ratio)]
    band = ok_ratio & (np.where(ok_ratio, fr["p_ratio"].to_numpy(dtype=float), -np.inf) >= RATIO_MIN)
    above = fr["above_ma"].to_numpy(dtype=float)
    ok_ma = np.isfinite(above)
    fr["status"] = status
    fr["driver"] = driver
    fr["S0"] = (np.array([st == STRETCHED for st in status], dtype=bool)
                & np.array([d in S0_DRIVERS for d in driver], dtype=bool))
    fr["S1"] = band
    fr["S2"] = band & ok_ma & (np.where(ok_ma, above, 0.0) == 1.0)
    fr["ok_S0"] = ok_ratio
    fr["ok_S1"] = ok_ratio
    fr["ok_S2"] = ok_ratio & ok_ma
    return fr


# ── Utfall (utfallskalendern) ────────────────────────────────────────────────
def outcome_frame(asset, bench) -> pd.DataFrame:
    """Utfallskalendern: dagar där både tillgången (metall-ETF:en eller gruvan) och GLD har en stängning."""
    return pd.concat({"asset": ms._clean(asset), "bench": ms._clean(bench)}, axis=1, sort=True).dropna()


def entry_positions(signal_index, outcome_index) -> np.ndarray:
    """För varje signaldag t: positionen i utfallskalendern för sista dagen ≤ t (−1 när ingen finns).

    Köpet sker på positionen efter, alltså första dagen efter t då tillgången och GLD båda har en stängning."""
    out = pd.DatetimeIndex(outcome_index).to_numpy(dtype="datetime64[ns]")
    sig = pd.DatetimeIndex(signal_index).to_numpy(dtype="datetime64[ns]")
    return np.searchsorted(out, sig, side="right").astype(np.int64) - 1


def outcomes_on(signal_index, asset, bench) -> dict:
    """Utfallen (miner_margin_study.outcome_arrays) för köp första dagen efter varje signaldag, i signalkalenderns
    ordning: {"arrays": {"tot_63": …, "rel_63": …, …, "dd_252": …}, "pos": …, "index": utfallskalendern}."""
    of = outcome_frame(asset, bench)
    n = len(signal_index)
    if len(of) == 0:
        keys = [f"{k}_{h}" for h in HORIZONS for k in ("tot", "rel")] + [f"dd_{DD_DAYS}"]
        return {"arrays": {k: np.full(n, np.nan) for k in keys}, "pos": np.full(n, -1, dtype=np.int64),
                "index": of.index}
    arr = ms.outcome_arrays(of, asset="asset", bench="bench")
    pos = entry_positions(signal_index, of.index)
    ok = pos >= 0
    safe = np.where(ok, pos, 0)
    return {"arrays": {k: np.where(ok, v[safe], np.nan) for k, v in arr.items()}, "pos": pos, "index": of.index}


def entry_date(out: dict, i: int) -> Optional[str]:
    """Köpdagen för signaldagen på position i: första dagen efter den i utfallskalendern (None när den saknas)."""
    p = int(out["pos"][i])
    return ms._d(out["index"][p + 1]) if 0 <= p and p + 1 < len(out["index"]) else None


def _span(out: dict) -> dict:
    idx = out["index"]
    return {"first": ms._d(idx[0]), "last": ms._d(idx[-1]), "days": int(len(idx))} if len(idx) else None


# ── Domar ────────────────────────────────────────────────────────────────────
def baseline_days(arrays: dict, cov: Optional[dict]) -> tuple:
    """(positioner, 252-dagarsutfall mot GLD) för köp varje dag i täckningen med hel horisont."""
    rel = arrays[f"rel_{HORIZONS[-1]}"]
    if cov is None:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    pos = np.arange(cov["pos"], len(rel))
    pos = pos[np.isfinite(rel[pos])]
    return pos, rel[pos]


def judge(sig: str, arrays: dict, starts: dict, cov: dict, base: Optional[float], vs_s0: bool = False) -> dict:
    """En dom: snittet av signalens 252-dagarsavkastning mot GLD är > 0 och > baslinjens snitt (H3 dessutom ≥ S0:s,
    eller så har S0 ingen hel episod). Golvet: minst MIN_EPISODES hela episoder, annars "för få episoder"."""
    c = cov.get(sig)
    out = {"signal": sig, "pass": False, "floor": False, "verdict": TOO_FEW, "horizon": HORIZONS[-1],
           "period_from": c["first"] if c else None, "baseline_mean": ms._f(base)}
    if c is None:
        out.update({"episodes": 0, "complete": 0, "mean": None, "why": [f"{sig} kan inte räknas"]})
        return out
    comp = ms._complete(arrays, starts[sig])
    m = ms._mean(comp)
    floor = len(comp) >= MIN_EPISODES
    why = []
    if not floor:
        why.append(f"{sig} har färre än {MIN_EPISODES} hela episoder")
    if m is None or not m > 0:
        why.append(f"{sig}:s snitt är inte > 0")
    if base is None or m is None or not m > base:
        why.append(f"{sig}:s snitt är inte över baslinjen")
    if vs_s0:
        c0 = ms._complete(arrays, starts["S0"])
        m0 = ms._mean(c0)
        out.update({"S0_complete": len(c0), "S0_mean": ms._f(m0), "S0_no_signal": not c0})
        if m0 is not None and (m is None or not m >= m0):
            why.append(f"{sig}:s snitt är under S0:s")
    out.update({"episodes": len(starts[sig]), "complete": len(comp), "mean": ms._f(m), "floor": floor,
                "pass": not why, "why": why, "verdict": PASS if not why else (FAIL if floor else TOO_FEW)})
    return out


def hypotheses(arrays: dict, starts: dict, cov: dict, base: Optional[float], base_days: int = 0) -> dict:
    """H1 (S0), H2 (S1) och H3 (S2, mot S0) för en metall."""
    out = {h: judge(sig, arrays, starts, cov, base, vs_s0=(h == "H3")) for h, sig in HYPOTHESES}
    for h in out.values():
        h["baseline_days"] = int(base_days)
    return out


def bootstrap(x, pos, starts, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """95 %-intervall för signalens snitt (miner_margin_study.block_bootstrap, cirkulära block om 63 dagar) över
    baslinjens dagar, med baslinjens snitt fast bredvid. Beskrivande, inget villkor för godkännande."""
    x = np.asarray(x, dtype=float)
    mask = np.isin(np.asarray(pos, dtype=np.int64), np.asarray(list(starts), dtype=np.int64))
    boot = ms.block_bootstrap(x, mask, seed=seed, n_boot=n_boot, block=BLOCK)
    base = float(x.mean()) if len(x) else None
    lo, hi = boot["ci"]
    boot.update({"episodes": int(mask.sum()), "mean": ms._f(x[mask].mean()) if mask.any() else None,
                 "baseline_mean": ms._f(base),
                 "vs_baseline": [ms._f(v - base) if v is not None and base is not None else None for v in (lo, hi)]})
    boot["ci"] = [ms._f(v) for v in (lo, hi)]
    return boot


def proposal(ratio: str, key: str, den: str, hyp: dict) -> dict:
    """Vad utfallet leder till enligt specen, för en snodd: det första fallet som stämmer. Förslag — ägarens
    beslut; studien ändrar inga regler."""
    h = {k: hyp.get(k) or {} for k, _ in HYPOTHESES}
    ok = {k: bool(v.get("pass")) for k, v in h.items()}
    head = f"{ratio} ({key})"
    if ok["H1"]:
        return {"case": "behåll", "text": f"{head}: H1 håller → förslag: behåll regeln (sträckt kvot och drivaren "
                                          f"{'/'.join(S0_DRIVERS)})."}
    if ok["H3"]:
        return {"case": "trendfilter",
                "text": f"{head}: H1 håller inte men H3 → förslag: ersätt drivarvillkoret med trendfiltret (kvoten "
                        f"≥ {RATIO_MIN:.0f} och {den} över MA{MA_DAYS})."}
    if ok["H2"]:
        return {"case": "utan drivaren",
                "text": f"{head}: H1 och H3 håller inte men H2 → förslag: ta bort drivarens nedgradering, så att "
                        f"kvoten ≥ {RATIO_MIN:.0f} räcker."}
    if all(v.get("floor") for v in h.values()):
        return {"case": "bara kontext",
                "text": f"{head}: inget håller → förslag: snodden blir bara kontext (inget +1 och ingen "
                        f"KÖP-STÖD-räkning) tills bättre belägg finns."}
    state = ", ".join(f"{k} {(v.get('verdict') or TOO_FEW).lower()}" for k, v in h.items())
    return {"case": "för få episoder",
            "text": f"{head}: för få episoder ({state}) → inget förslag; snodden behålls som den är, med en "
                    f"varning."}


# ── En snodd: signaler, utfall, baslinjer, domar och det beskrivande ─────────
def secondary(out: dict, starts: dict, cov: Optional[dict], n: int) -> dict:
    """En annan tillgångs utfall mot GLD (korsvis eller en gruva) på snoddens episoder, med baslinjen i samma
    täckning. Bara beskrivande."""
    arrays = out["arrays"]
    return {"data": _span(out),
            "signals": {sig: ms.summarize(arrays, starts[sig]) for sig, _ in SIGNALS},
            "baseline": None if cov is None else ms.summarize(arrays, range(cov["pos"], n))}


def episode_rows(fr: pd.DataFrame, own: dict, positions, cross: Optional[dict] = None,
                 miners: Optional[dict] = None) -> list:
    """En rad per episod: datum, percentilerna och MA200 på signaldagen, utfallen och det beskrivande."""
    a = own["arrays"]
    rows = []
    for i in positions:
        above = fr["above_ma"].iloc[i]
        driver = fr["driver"].iloc[i]
        r = {"start": ms._d(fr.index[i]), "entry": entry_date(own, i),
             "p_ratio": ms._f(fr["p_ratio"].iloc[i], 1), "p_gc": ms._f(fr["p_gc"].iloc[i], 1),
             "p_metal": ms._f(fr["p_metal"].iloc[i], 1), "driver": driver if isinstance(driver, str) else None,
             "metal": ms._f(fr["metal"].iloc[i], 2), "ma200": ms._f(fr["ma200"].iloc[i], 2),
             "above_ma": bool(above == 1.0) if np.isfinite(above) else None}
        for h in HORIZONS:
            r[f"tot_{h}"] = ms._f(a[f"tot_{h}"][i])
            r[f"rel_{h}"] = ms._f(a[f"rel_{h}"][i])
        r[f"dd_{DD_DAYS}"] = ms._f(a[f"dd_{DD_DAYS}"][i])
        if cross is not None:
            r[f"cross_rel_{HORIZONS[-1]}"] = ms._f(cross["arrays"][f"rel_{HORIZONS[-1]}"][i])
        if miners:
            key = f"rel_{HORIZONS[-1]}"
            r[f"miners_{key}"] = {m: ms._f(o["arrays"][key][i]) if o else None for m, o in miners.items()}
        rows.append(r)
    return rows


def analyze(fr: pd.DataFrame, etf, gld, cross: Optional[tuple] = None, miners: Optional[dict] = None,
            start=START, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """Allt för en snodd ur signal_frame. etf och gld är utfallens kursserier, cross = (ticker, serie) för den
    andra metallens ETF och miners = {ticker: serie} för gruvorna (en tom eller saknad serie = saknas)."""
    own = outcomes_on(fr.index, etf, gld)
    arrays = own["arrays"]
    start_ts = pd.Timestamp(start)
    n = len(fr)
    cov = {sig: ms.coverage(fr, sig, start) for sig, _ in SIGNALS}
    starts = {sig: [i for i in ms.episode_starts(fr[sig].to_numpy(dtype=bool), GAP) if fr.index[i] >= start_ts]
              for sig, _ in SIGNALS}
    base_cov = cov["S1"]                                   # täckningen: kvotens fönster från 2012-01-01
    pos, x = baseline_days(arrays, base_cov)
    base = float(x.mean()) if len(x) else None

    cross_out = None if cross is None else outcomes_on(fr.index, cross[1], gld)
    miner_out = {m: (outcomes_on(fr.index, s, gld) if len(ms._clean(s)) else None)
                 for m, s in (miners or {}).items()}

    signals = {}
    for sig, label in SIGNALS:
        days = int((fr[sig].to_numpy(dtype=bool) & (fr.index >= start_ts)).sum())
        signals[sig] = {"label": label, "signal_days": days, "coverage": ms._public(cov[sig]),
                        "summary": ms.summarize(arrays, starts[sig]),
                        "bootstrap": bootstrap(x, pos, starts[sig], seed=seed, n_boot=n_boot),
                        "episodes": episode_rows(fr, own, starts[sig], cross_out, miner_out)}
    baseline = None
    if base_cov is not None:
        baseline = {"first": base_cov["first"], "last": base_cov["last"],
                    "alla dagar": ms.summarize(arrays, range(base_cov["pos"], n)),
                    "var 63:e dag": ms.summarize(arrays, range(base_cov["pos"], n, BASE_STEP))}
    out = {"coverage": {k: ms._public(v) for k, v in cov.items()}, "outcome_data": _span(own),
           "signals": signals, "baseline": baseline,
           "hypotheses": hypotheses(arrays, starts, cov, base, len(x))}
    if cross is not None:
        out["cross"] = dict(secondary(cross_out, starts, base_cov, n), etf=cross[0])
    out["miners"] = {m: (secondary(o, starts, base_cov, n) if o else None) for m, o in miner_out.items()}
    return out


# ── Hela studien ─────────────────────────────────────────────────────────────
def band_spec(key: str):
    """Panelens RatioSpec för snodden (gold_platinum = GC=F/PL=F med GLD/PPLT, gold_palladium = GC=F/PA=F med
    GLD/PALL)."""
    return ms.live_spec(key)


def study(series: dict, start=START, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """Hela studien ur färdiga kursserier (inget nätverk): båda metallerna, korsvis och gruvorna."""
    out: dict = {"metals": {}, "proposals": []}
    specs = {key: band_spec(key) for _, key in BANDS}
    for name, key in BANDS:
        spec = specs[key]
        other = next(specs[k] for _, k in BANDS if k != key)
        fr = signal_frame(series[spec.primary_num], series[spec.primary_den], direction=spec.cheap_direction)
        res = analyze(fr, series[spec.fallback_den], series[spec.fallback_num],
                      cross=(other.fallback_den, series[other.fallback_den]),
                      miners={m: series.get(m) for m in MINERS}, start=start, seed=seed, n_boot=n_boot)
        ratio = f"{spec.primary_num}/{spec.primary_den}"
        res.update({"metal": name, "key": key, "ratio": ratio, "metal_ticker": spec.primary_den,
                    "etf": spec.fallback_den, "bench": spec.fallback_num})
        res["proposal"] = proposal(ratio, key, spec.primary_den, res["hypotheses"])
        out["metals"][name] = res
        out["proposals"].append(res["proposal"]["text"])
        log.info("%s: %d dagar, signaldagar %s", ratio, len(fr),
                 " · ".join(f"{s} {res['signals'][s]['signal_days']}" for s, _ in SIGNALS))
    return out


# ── Rapport ──────────────────────────────────────────────────────────────────
def _header(metal_col: str) -> str:
    return (f"  {'signal':<28}{'dagar':>7}{'ep.':>5}  {'hela 63/126/252':<16}"
            f"{'GLD 63':>10}{'GLD 126':>10}{'GLD 252':>10}{'median':>10}{'träff':>8}"
            f"{metal_col + ' 252':>10}{'nedg.':>10}")


def _span_text(c) -> str:
    return f"{c['first']}–{c['last']} ({ms._int(c['days'])} dagar)" if c else "kan inte räknas"


def _hit(v) -> str:
    return f"{v * 100:.0f} %" if v is not None else "–"


def _miner_cell(s: Optional[dict]) -> str:
    h = HORIZONS[-1]
    if not s:
        return f"{'–':>6}{'–':>10}{'–':>10}{'–':>8}"
    return (f"{ms._int(s.get(f'n_{h}', 0)):>6}{ms._pct(s.get(f'rel_{h}')):>10}"
            f"{ms._pct(s.get(f'rel_{h}_median')):>10}{_hit(s.get(f'hit_{h}')):>8}")


def _hyp_lines(hyp: dict) -> list:
    labels = dict(SIGNALS)
    out = []
    for h, sig in HYPOTHESES:
        v = hyp[h]
        reasons = "" if v.get("pass") else " (" + "; ".join(v.get("why") or ["–"]) + ")"
        extra = ""
        if h == "H3" and "S0_complete" in v:
            extra = (" · S0 ingen hel episod" if v["S0_no_signal"]
                     else f" · S0 snitt {ms._pct(v['S0_mean'])} (hela: {v['S0_complete']})")
        out.append(f"    {h} {sig} {labels[sig]}: {v.get('episodes', 0)} ep. (hela: {v.get('complete', 0)}), snitt "
                   f"{ms._pct(v.get('mean'))}{extra} → {v['verdict']}{reasons}")
    return out


def _metal_lines(name: str, m: dict) -> list:
    sig, cov, etf = m["signals"], m["coverage"], m["etf"]
    lines = [f"== {name.capitalize()} ({m['ratio']}, panelens {m['key']}; utfall {etf} mot {m['bench']}) ==",
             f"  Täckning från {START}: S0/S1 {_span_text(cov['S1'])} · S2 {_span_text(cov['S2'])}",
             _header(etf)]
    for s, label in SIGNALS:
        lines.append(ms._table_line(f"{s} {label}", sig[s]["summary"], sig[s]["signal_days"]))
    b = m.get("baseline")
    if b:
        for kind in ("alla dagar", "var 63:e dag"):
            lines.append(ms._table_line(f"bas {kind}", b[kind]))
    else:
        lines.append("  bas: kan inte räknas")
    lines.append(f"  (dagar = signaldagar, för baslinjen köpdagar; ep. = episoder; hela = antal med hel horisont; "
                 f"GLD x = snitt mot GLD över x dagar; median, träff (andel > 0) och {etf} 252 (totalavkastning) "
                 f"gäller 252 dagar; nedg. = snitt av värsta nedgången inom 252 dagar)")
    lines.append(f"  Episoder (start → köp · percentil kvot/GC/{m['metal_ticker']} · över MA{MA_DAYS} · mot GLD "
                 f"63/126/252 · {etf} 252 · värsta nedgång):")
    any_ep = False
    for s, _ in SIGNALS:
        for e in sig[s]["episodes"]:
            any_ep = True
            ma = "–" if e["above_ma"] is None else ("ja" if e["above_ma"] else "nej")
            lines.append(f"    {s} {e['start']} → {e['entry'] or 'i morgon'} · "
                         f"{ms._p(e['p_ratio'])}/{ms._p(e['p_gc'])}/{ms._p(e['p_metal'])} · MA{MA_DAYS} {ma} · "
                         + "/".join(ms._pct(e.get(f"rel_{x}")) for x in HORIZONS)
                         + f" · {ms._pct(e.get(f'tot_{HORIZONS[-1]}'))} · {ms._pct(e.get(f'dd_{DD_DAYS}'))}")
    if not any_ep:
        lines.append("    inga")
    h1 = m["hypotheses"]["H1"]
    lines.append(f"  Domar (252 dagar mot GLD; baslinje {ms._pct(h1['baseline_mean'])} över "
                 f"{ms._int(h1['baseline_days'])} dagar; golv {MIN_EPISODES} hela episoder):")
    lines += _hyp_lines(m["hypotheses"])
    b0 = sig[SIGNALS[0][0]]["bootstrap"]
    lines.append(f"  Bootstrap, beskrivande (95 %, cirkulära block om {b0['block']} dagar, {ms._int(b0['n_boot'])} "
                 f"dragningar, frö {b0['seed']}):")
    for s, _ in SIGNALS:
        b = sig[s]["bootstrap"]
        lo, hi = b["ci"]
        dlo, dhi = b["vs_baseline"]
        lines.append(f"    {s}: snitt {ms._pct(b['mean'])} (hela ep.: {b['episodes']}) · 95 % [{ms._pct(lo)}; "
                     f"{ms._pct(hi)}] · mot baslinjen [{ms._pct(dlo)}; {ms._pct(dhi)}] · "
                     f"{ms._int(b['empty'])} av {ms._int(b['n_boot'])} dragningar utan episod")
    c = m.get("cross")
    if c:
        lines.append(f"  Korsvis, beskrivande ({c['etf']} mot GLD på {GENITIVE.get(name, name)} episoder):")
        lines.append(_header(c["etf"]))
        for s, label in SIGNALS:
            lines.append(ms._table_line(f"{s} {label}", c["signals"][s], sig[s]["signal_days"]))
        if c.get("baseline"):
            lines.append(ms._table_line("bas alla dagar", c["baseline"]))
    miners = m.get("miners") or {}
    if miners:
        lines.append("  Gruvorna, beskrivande (252 dagar mot GLD: hela · snitt · median · träff):")
        lines.append(f"  {'signal':<28}" + "".join(f"{t:>34}" for t in miners))
        for s, label in SIGNALS:
            lines.append(f"  {s + ' ' + label:<28}"
                         + "".join(_miner_cell(o["signals"][s] if o else None) for o in miners.values()))
        lines.append(f"  {'bas alla dagar':<28}"
                     + "".join(_miner_cell(o["baseline"] if o else None) for o in miners.values()))
        for t, o in miners.items():
            d = (o or {}).get("data")
            lines.append(f"    {t}: " + (f"kurser {d['first']}–{d['last']}" if d else "saknas"))
    lines.append(f"  Förslag: {m['proposal']['text']}")
    return lines


def report_lines(res: dict) -> list:
    """Kompakt svensk rapport för loggen."""
    lines = [f"PGM-SNODDARNA — spec version {res.get('spec_version', SPEC_VERSION)} ({SPEC})",
             "Forskning: ändrar inga regler. Signaler på terminerna (GC=F, PL=F, PA=F), utfall på PPLT och PALL mot "
             "GLD; gruvorna och korsutfallen är bara beskrivande."]
    data = res.get("data") or {}
    if data:
        lines.append("Kurser: " + " · ".join(
            f"{t} {d['first']}–{d['last']} ({ms._int(d['days'])} dagar)" if d.get("days") else f"{t} saknas"
            for t, d in data.items()))
    if res.get("error"):
        return lines + [f"FEL: {res['error']}"]
    if res.get("optional_missing"):
        lines.append(f"Saknas (krävs inte, bara beskrivande): {', '.join(res['optional_missing'])}")
    for name, m in (res.get("metals") or {}).items():
        lines += _metal_lines(name, m)
    lines.append("Förslag enligt specen (bara förslag — ägarens beslut; studien ändrar inga regler):")
    lines += [f"  • {p}" for p in res.get("proposals") or []]
    return lines


# ── I/O ──────────────────────────────────────────────────────────────────────
def run(getter=None, save=None, no_save: bool = False, n_boot: int = N_BOOT) -> dict:
    res = {"spec_version": SPEC_VERSION, "spec": SPEC, "generated": datetime.now(tz=timezone.utc).isoformat()}
    series = ms.load(getter, TICKERS)
    if getter is None:
        try:
            import market_prices
            log.info("%s", market_prices.stats_text())
        except Exception:
            pass
    res["data"] = ms.data_info(series, TICKERS)
    gaps = ms.missing(series, REQUIRED)
    if gaps:
        res["missing"] = gaps
        res["error"] = (f"kurser saknas för {', '.join(gaps)} (behöver minst {ms._int(MIN_DAYS)} dagskurser) — "
                        f"inget sparas")
        for line in report_lines(res):
            log.error("%s", line)
        return res
    res["optional_missing"] = [t for t in MINERS if not len(ms._clean(series.get(t)))]
    res.update(study(series, n_boot=n_boot))
    for line in report_lines(res):
        log.info("%s", line)
    if not no_save:
        if save is None:
            from gist_storage import save_blob as save
        res["saved"] = bool(save(BLOB, res))
        log.info("Gisten (%s): %s", BLOB, "sparad" if res["saved"] else "inte sparad — kontrollera GITHUB_TOKEN")
    return res


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PGM-snoddarna — förregistrerat backtest.")
    parser.add_argument("--no-save", action="store_true", help="spara inte i Gisten")
    args = parser.parse_args(argv)
    res = run(no_save=args.no_save)
    return 1 if res.get("missing") else 0


if __name__ == "__main__":
    sys.exit(main())
