#!/usr/bin/env python3
"""
miner_margin_study.py — Gruvorna mot marginalen (docs/forskning/gruvor_marginal_spec.md, version 1).

Förregistrerat backtest av gruvornas köpstöd: hade marginalmätaren, M = gruv-ETF:ns kurs / (guldpris − branschens
AISC), varit en bättre bekräftelse än drivarens prisvillkor när GLD/GDX eller GLD/GDXJ står sträckt?

Signalerna S0–S3 räknas point-in-time varje handelsdag med panelens egna funktioner (_percentile_of, _classify,
_classify_driver, margin_ratio, aisc_series(point_in_time=True)). Därefter episoder, utfall på 63/126/252 dagar,
baslinjer, en moving-block-bootstrap och domarna H1–H3.

De rena funktionerna (signaler, episoder, utfall, bootstrap, domar, rapport) står först. run()/main() hämtar
kurserna (market_prices, "max") och sparar miner_margin_study.json i Gisten. Körs manuellt i Actions
(miner-margin-study.yml). Ändrar inga regler: mätaren är fortfarande bara kontext i panelen. Avslutas med 0 även
när hypoteserna faller, och med 1 bara när kurser saknas. Env: GITHUB_TOKEN (Gisten), BORSDATA_API_KEY
(market_prices provar Börsdata först).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from alpha_regime import commodity_ratios as cr
from alpha_regime import miner_margin as mm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("miner_margin_study")

BLOB = "miner_margin_study.json"
SPEC = "docs/forskning/gruvor_marginal_spec.md"
SPEC_VERSION = 1

START = "2012-01-01"          # utfall räknas för episoder som börjar den här dagen eller senare
PERIOD = "max"                # längsta historiken market_prices ger
WINDOW_YEARS = 10             # percentiler över (t − 10 år, t], som kvotkorten
MIN_YEARS = 3                 # fönstret måste börja minst tre år före t
GAP = 63                      # handelsdagar utan signal innan en ny episod börjar
HORIZONS = (63, 126, 252)     # handelsdagar efter köpet
DD_DAYS = 252                 # värsta nedgången inom så här många dagar efter köpet
BASE_STEP = 63                # baslinjen "var 63:e dag"
S2_M_MAX = 20.0               # S2: P_M ≤ 20
S3_M_MAX = 30.0               # S3: P_kvot ≥ 90 och P_M ≤ 30
MIN_EPISODES = 3              # episoder med hel 252-dagarshorisont för en dom (H1–H3)
BLOCK = 63                    # bootstrapens blocklängd
N_BOOT = 5000
SEED = 20261103
MIN_DAYS = 756                # färre dagskurser än tre år = kursen saknas

GOLD_ETF = "GLD"
GOLD = mm.GOLD_TICKER                                     # "GC=F"
MINERS = (("GDX", "metal_miners"), ("GDXJ", "gold_gdxj"))  # gruv-ETF och panelens kvotnyckel
TICKERS = (GOLD_ETF, "GDX", "GDXJ", GOLD)
STRETCHED = "RUBBER_BAND_STRETCHED"
S0_DRIVERS = ("DENOMINATOR_CHEAP", "BOTH")                # motorns +1 ACCUMULATE (alpha_regime/engine.py)
SIGNALS = (("S0", "nuvarande regel"), ("S1", "bara kvoten"), ("S2", "bara marginalen"),
           ("S3", "föreslagen ersättare"))


# ── Små hjälpare ─────────────────────────────────────────────────────────────
def _clean(s) -> pd.Series:
    """Dagsserie utan NaN, dubbletter och tidszon, sorterad på datum."""
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(s, dtype=float).dropna()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s[np.isfinite(s.to_numpy())]


def _f(v, nd: int = 5) -> Optional[float]:
    """Avrundat flyttal till JSON, None när värdet saknas eller inte är ändligt."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _d(ts) -> str:
    return str(pd.Timestamp(ts).date())


def _mean(vals) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def live_spec(key: str):
    """Panelens RatioSpec för kvoten (metal_miners = GLD/GDX, gold_gdxj = GLD/GDXJ)."""
    return next(s for s in cr._RATIO_SPECS if s.key == key)


# ── Signaler (point-in-time) ─────────────────────────────────────────────────
def pit_percentiles(s: pd.Series, years: int = WINDOW_YEARS, min_years: int = MIN_YEARS) -> pd.Series:
    """Percentilen (_percentile_of) av varje dags värde bland de giltiga värdena i (t − years, t].

    NaN när dagens värde saknas eller när fönstrets första giltiga värde ligger mindre än min_years före t.
    Bara data ≤ t används, så en ändring efter t ändrar aldrig värdet på t."""
    s = pd.Series(s, dtype=float)
    v = s.to_numpy(dtype=float)
    ok = np.isfinite(v)
    out = np.full(len(v), np.nan)
    if ok.any():
        idx = pd.DatetimeIndex(s.index[ok])
        vals = v[ok]
        days = idx.to_numpy()
        lo = np.searchsorted(days, (idx - pd.DateOffset(years=years)).to_numpy(), side="right")
        need = (idx - pd.DateOffset(years=min_years)).to_numpy()
        res = np.full(len(vals), np.nan)
        for j in range(len(vals)):
            st = lo[j]
            if days[st] <= need[j]:
                res[j] = cr._percentile_of(vals[st:j + 1], vals[j])
        out[ok] = res
    return pd.Series(out, index=s.index, dtype=float)


def signal_frame(gld, miner, gold, rows=None, direction: str = "high",
                 years: int = WINDOW_YEARS, min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Signalernas underlag och S0–S3 per handelsdag (kalender: dagar med både GLD och gruv-ETF:n).

    Kvoten och benen kopplas som kvotkortet (inre koppling GLD/gruv-ETF), M som mätaren (gruv-ETF/guld) med
    den AISC som var publicerad varje dag (aisc_series(point_in_time=True)). ok_S* = signalen kan räknas."""
    gld, miner, gold = _clean(gld), _clean(miner), _clean(gold)
    fr = pd.concat({"gld": gld, "miner": miner}, axis=1, sort=True).dropna()
    fr["ratio"] = fr["gld"] / fr["miner"]
    fr["p_ratio"] = pit_percentiles(fr["ratio"], years, min_years)
    fr["p_gld"] = pit_percentiles(fr["gld"], years, min_years)
    fr["p_miner"] = pit_percentiles(fr["miner"], years, min_years)

    mg = pd.concat({"m": miner, "g": gold}, axis=1, sort=True).dropna()
    aisc = mm.aisc_series(mg.index, point_in_time=True, rows=rows)
    m = mm.margin_ratio(mg["m"], mg["g"], aisc)
    fr["gold"] = mg["g"].reindex(fr.index)
    fr["aisc"] = aisc.reindex(fr.index)
    fr["M"] = m.reindex(fr.index)
    fr["p_m"] = pit_percentiles(m, years, min_years).reindex(fr.index)

    ok_ratio = fr[["p_ratio", "p_gld", "p_miner"]].notna().all(axis=1).to_numpy()
    status = [cr._classify(p, direction) if ok else None for p, ok in zip(fr["p_ratio"], ok_ratio)]
    driver = [cr._classify_driver(a, b, direction) if ok else None
              for a, b, ok in zip(fr["p_gld"], fr["p_miner"], ok_ratio)]
    stretched = np.array([st == STRETCHED for st in status], dtype=bool)
    pm = fr["p_m"].to_numpy(dtype=float)
    ok_m = np.isfinite(pm)
    fr["status"] = status
    fr["driver"] = driver
    fr["S0"] = stretched & np.array([d in S0_DRIVERS for d in driver], dtype=bool)
    fr["S1"] = stretched
    fr["S2"] = ok_m & (np.where(ok_m, pm, np.inf) <= S2_M_MAX)
    fr["S3"] = stretched & ok_m & (np.where(ok_m, pm, np.inf) <= S3_M_MAX)
    fr["ok_S0"] = ok_ratio
    fr["ok_S1"] = ok_ratio
    fr["ok_S2"] = ok_m
    fr["ok_S3"] = ok_ratio & ok_m
    return fr


# ── Episoder och utfall ──────────────────────────────────────────────────────
def episode_starts(sig, gap: int = GAP) -> list:
    """Positioner där en episod börjar: en signaldag efter minst `gap` dagar utan signalen. Dagar före serien
    räknas som utan signal, så den första signaldagen börjar alltid en episod."""
    out, last = [], None
    for i, on in enumerate(np.asarray(sig, dtype=bool)):
        if on:
            if last is None or i - last > gap:
                out.append(i)
            last = i
    return out


def forward_returns(miner, gld, h: int) -> tuple:
    """(total, mot GLD) för köp på stängningen dagen efter varje position och sälj h dagar senare.
    NaN där horisonten inte är hel."""
    m, g = np.asarray(miner, dtype=float), np.asarray(gld, dtype=float)
    n = len(m)
    tot, rel = np.full(n, np.nan), np.full(n, np.nan)
    entry = np.arange(n) + 1
    exit_ = entry + h
    ok = exit_ < n
    tot[ok] = m[exit_[ok]] / m[entry[ok]] - 1.0
    rel[ok] = tot[ok] - (g[exit_[ok]] / g[entry[ok]] - 1.0)
    return tot, rel


def worst_drawdown(miner, h: int = DD_DAYS) -> np.ndarray:
    """Värsta nedgången (topp till botten) från köpet dagen efter varje position och h dagar framåt.
    NaN där horisonten inte är hel."""
    m = np.asarray(miner, dtype=float)
    n = len(m)
    out = np.full(n, np.nan)
    for i in range(max(0, n - h - 1)):
        seg = m[i + 1:i + h + 2]
        out[i] = float(np.min(seg / np.maximum.accumulate(seg) - 1.0))
    return out


def outcome_arrays(fr: pd.DataFrame) -> dict:
    """{"tot_63": …, "rel_63": …, …, "dd_252": …} per position i ramen."""
    out = {}
    for h in HORIZONS:
        out[f"tot_{h}"], out[f"rel_{h}"] = forward_returns(fr["miner"], fr["gld"], h)
    out[f"dd_{DD_DAYS}"] = worst_drawdown(fr["miner"], DD_DAYS)
    return out


def summarize(arrays: dict, positions) -> dict:
    """Snitt, median, antal hela horisonter, träffsäkerhet och nedgång över positionerna."""
    pos = np.asarray(list(positions), dtype=np.int64)
    out: dict = {"n": int(len(pos))}
    for h in HORIZONS:
        rel = arrays[f"rel_{h}"][pos] if len(pos) else np.array([])
        tot = arrays[f"tot_{h}"][pos] if len(pos) else np.array([])
        rel, tot = rel[np.isfinite(rel)], tot[np.isfinite(tot)]
        out[f"n_{h}"] = int(len(rel))
        out[f"rel_{h}"] = _f(rel.mean()) if len(rel) else None
        out[f"rel_{h}_median"] = _f(np.median(rel)) if len(rel) else None
        out[f"tot_{h}"] = _f(tot.mean()) if len(tot) else None
    rel = arrays[f"rel_{HORIZONS[-1]}"][pos] if len(pos) else np.array([])
    rel = rel[np.isfinite(rel)]
    out[f"hit_{HORIZONS[-1]}"] = _f((rel > 0).mean(), 4) if len(rel) else None
    dd = arrays[f"dd_{DD_DAYS}"][pos] if len(pos) else np.array([])
    dd = dd[np.isfinite(dd)]
    out[f"dd_{DD_DAYS}"] = _f(dd.mean()) if len(dd) else None
    out[f"dd_{DD_DAYS}_worst"] = _f(dd.min()) if len(dd) else None
    return out


def coverage(fr: pd.DataFrame, sig: str, start=START) -> Optional[dict]:
    """Första dagen (≥ start) signalen kan räknas, till ramens sista dag. None när den aldrig kan räknas."""
    ok = fr[f"ok_{sig}"].to_numpy(dtype=bool) & (fr.index >= pd.Timestamp(start))
    if not ok.any():
        return None
    first = int(np.argmax(ok))
    return {"first": _d(fr.index[first]), "last": _d(fr.index[-1]), "pos": first, "days": int(len(fr) - first)}


def h1_starts(fr: pd.DataFrame, start=START) -> dict:
    """Episodstarterna som H1 jämför, S3 och S0 räknade på samma dagar.

    S3 kan bara räknas där M finns (ok_S3), så dagarna före marginalperioden är "ingen signal" för S3. S0:s episoder
    räknas här på samma sätt, på S0 och ok_S3: en S0-period som redan pågick när marginalperioden började öppnar då
    en episod den första dagen, precis som S3. Annars fick S3 en extra episod vid periodens början som S0 inte fick.
    Bara starter från S3:s täckning (≥ start) räknas. S0:s rad i signaltabellen räknas över hela kvotperioden."""
    c = coverage(fr, "S3", start)
    if c is None:
        return {"S3": [], "S0": []}
    ok = fr["ok_S3"].to_numpy(dtype=bool)
    return {sig: [i for i in episode_starts(fr[sig].to_numpy(dtype=bool) & ok) if i >= c["pos"]]
            for sig in ("S3", "S0")}


def episode_rows(fr: pd.DataFrame, arrays: dict, positions) -> list:
    """En rad per episod: datum, percentilerna på signaldagen och utfallen."""
    rows = []
    for i in positions:
        entry = fr.index[i + 1] if i + 1 < len(fr) else None
        driver = fr["driver"].iloc[i]
        r = {"start": _d(fr.index[i]), "entry": _d(entry) if entry is not None else None,
             "p_ratio": _f(fr["p_ratio"].iloc[i], 1), "p_gld": _f(fr["p_gld"].iloc[i], 1),
             "p_miner": _f(fr["p_miner"].iloc[i], 1), "p_m": _f(fr["p_m"].iloc[i], 1),
             "driver": driver if isinstance(driver, str) else None}
        for h in HORIZONS:
            r[f"tot_{h}"] = _f(arrays[f"tot_{h}"][i])
            r[f"rel_{h}"] = _f(arrays[f"rel_{h}"][i])
        r[f"dd_{DD_DAYS}"] = _f(arrays[f"dd_{DD_DAYS}"][i])
        rows.append(r)
    return rows


# ── Bootstrap ────────────────────────────────────────────────────────────────
def block_bootstrap(x, mask, seed: int = SEED, n_boot: int = N_BOOT, block: int = BLOCK) -> dict:
    """95 %-intervall för snittet av x över episodstarterna (mask): moving-block-bootstrap av dagarna med
    cirkulära block (som odin/rotation_backtest.block_boot).

    Varje dragning lägger ceil(N/block) block om `block` dagar i följd efter varandra (start likformigt bland alla
    N dagar, blocket fortsätter från början efter sista dagen) och kapar till N dagar. Statistiken är snittet av x
    över de episodstarter som kom med. Cirkulära block ger varje dag samma chans att dras; annars drogs en episod
    på periodens första eller sista dag nästan aldrig. Dragningar utan episodstart räknas inte ("empty"). Samma
    frö ger samma intervall."""
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n = len(x)
    res = {"ci": [None, None], "draws": 0, "empty": 0, "n_boot": int(n_boot), "block": int(block),
           "seed": int(seed), "days": int(n)}
    if n < block or not mask.any() or not np.isfinite(x[mask]).all():
        return res
    k = -(-n // block)
    last = n - (k - 1) * block
    xm = np.where(mask, x, 0.0)
    cx = np.concatenate([[0.0], np.cumsum(np.concatenate([xm, xm]))])          # två varv: block runt slutet
    cc = np.concatenate([[0], np.cumsum(np.concatenate([mask, mask]).astype(np.int64))])
    s = np.arange(n)
    full_x, full_c = cx[s + block] - cx[s], cc[s + block] - cc[s]
    last_x, last_c = cx[s + last] - cx[s], cc[s + last] - cc[s]
    draw = np.random.default_rng(seed).integers(0, n, size=(n_boot, k))
    tot = full_x[draw[:, :-1]].sum(axis=1) + last_x[draw[:, -1]]
    cnt = full_c[draw[:, :-1]].sum(axis=1) + last_c[draw[:, -1]]
    ok = cnt > 0
    res["draws"], res["empty"] = int(ok.sum()), int((~ok).sum())
    if ok.any():
        means = tot[ok] / cnt[ok]
        res["ci"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    return res


# ── En gruv-ETF: signaler, utfall, baslinjer och domar ───────────────────────
def analyze(fr: pd.DataFrame, start=START, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """Allt för en gruv-ETF ur signal_frame: täckning, episoder, sammanfattningar, baslinjer, H1–H3."""
    arrays = outcome_arrays(fr)
    start_ts = pd.Timestamp(start)
    h = HORIZONS[-1]
    cov = {sig: coverage(fr, sig, start) for sig, _ in SIGNALS}
    starts = {sig: [i for i in episode_starts(fr[sig].to_numpy()) if fr.index[i] >= start_ts]
              for sig, _ in SIGNALS}

    signals = {}
    for sig, label in SIGNALS:
        days = int((fr[sig].to_numpy(dtype=bool) & (fr.index >= start_ts)).sum())
        signals[sig] = {"label": label, "signal_days": days, "coverage": _public(cov[sig]),
                        "summary": summarize(arrays, starts[sig]),
                        "episodes": episode_rows(fr, arrays, starts[sig])}

    baseline = {}
    for name, sig in (("kvotperioden", "S1"), ("marginalperioden", "S2")):     # S2:s täckning = M:s, som i H2
        c = cov[sig]
        if c is None:
            baseline[name] = None
            continue
        baseline[name] = {"first": c["first"], "last": c["last"],
                          "alla dagar": summarize(arrays, range(c["pos"], len(fr))),
                          "var 63:e dag": summarize(arrays, range(c["pos"], len(fr), BASE_STEP))}

    h1 = h1_starts(fr, start)
    hyp = {"H1": _h1(arrays, h1, cov), "H2": _h2(arrays, starts, cov, seed, n_boot),
           "H3": _h3(arrays, starts, cov)}
    if cov["S3"] is not None:
        hyp["H1"].update({f"{s}_dates": [_d(fr.index[i]) for i in h1[s]] for s in ("S3", "S0")})
    for k in hyp:
        hyp[k]["horizon"] = h
    return {"coverage": {k: _public(v) for k, v in cov.items()}, "signals": signals, "baseline": baseline,
            "hypotheses": hyp}


def _public(c: Optional[dict]) -> Optional[dict]:
    return None if c is None else {k: v for k, v in c.items() if k != "pos"}


def _complete(arrays: dict, positions) -> list:
    rel = arrays[f"rel_{HORIZONS[-1]}"]
    return [float(rel[i]) for i in positions if math.isfinite(rel[i])]


def _h1(arrays, starts, cov) -> dict:
    """S3 mot S0 över S3:s täckning (episoder som börjar i marginalperioden). starts ur h1_starts, så att S0:s
    episoder räknas på samma dagar som S3:s."""
    c = cov["S3"]
    out = {"pass": False, "period_from": c["first"] if c else None}
    if c is None:
        out["why"] = ["S3 kan inte räknas (ingen marginalhistorik)"]
        return out
    s3 = [i for i in starts["S3"] if i >= c["pos"]]
    s0 = [i for i in starts["S0"] if i >= c["pos"]]
    c3, c0 = _complete(arrays, s3), _complete(arrays, s0)
    m3, m0 = _mean(c3), _mean(c0)
    why = []
    if not len(s3) > len(s0):
        why.append("S3 har inte fler episoder än S0")
    if len(c3) < MIN_EPISODES:
        why.append(f"S3 har färre än {MIN_EPISODES} hela episoder")
    if m3 is None or not m3 > 0:
        why.append("S3:s snitt är inte > 0")
    if m0 is not None and (m3 is None or not m3 >= m0):
        why.append("S3:s snitt är under S0:s")
    out.update({"S3_episodes": len(s3), "S3_complete": len(c3), "S3_mean": _f(m3),
                "S0_episodes": len(s0), "S0_complete": len(c0), "S0_mean": _f(m0),
                "S0_no_signal": not c0, "pass": not why, "why": why})
    return out


def _h2(arrays, starts, cov, seed, n_boot) -> dict:
    """S2 mot baslinjen (köp varje dag i S2:s täckning med hel 252-dagarshorisont), med bootstrap."""
    c = cov["S2"]
    out = {"pass": False, "period_from": c["first"] if c else None}
    if c is None:
        out["why"] = ["S2 kan inte räknas (ingen marginalhistorik)"]
        return out
    rel = arrays[f"rel_{HORIZONS[-1]}"]
    pos = np.arange(c["pos"], len(rel))
    pos = pos[np.isfinite(rel[pos])]
    x = rel[pos]
    mask = np.isin(pos, starts["S2"])
    base = float(x.mean()) if len(x) else None
    m2 = float(x[mask].mean()) if mask.any() else None
    boot = block_bootstrap(x, mask, seed=seed, n_boot=n_boot)
    lo = boot["ci"][0]
    why = []
    if int(mask.sum()) < MIN_EPISODES:
        why.append(f"S2 har färre än {MIN_EPISODES} hela episoder")
    if base is None or m2 is None or not m2 > base:
        why.append("S2:s snitt är inte över baslinjen")
    if base is None or lo is None or not lo > base:
        why.append("intervallets nedre gräns är inte över baslinjen")
    boot["ci"] = [_f(v) for v in boot["ci"]]                 # avrundat först efter domen
    out.update({"S2_complete": int(mask.sum()), "S2_mean": _f(m2), "baseline_days": int(len(x)),
                "baseline_mean": _f(base), "bootstrap": boot, "pass": not why, "why": why})
    return out


def _h3(arrays, starts, cov) -> dict:
    """S1 (utan drivarens nedgradering) från 2012-01-01."""
    c = cov["S1"]
    out = {"pass": False, "period_from": c["first"] if c else None}
    if c is None:
        out["why"] = ["S1 kan inte räknas"]
        return out
    c1 = _complete(arrays, starts["S1"])
    m1 = _mean(c1)
    why = []
    if len(c1) < MIN_EPISODES:
        why.append(f"S1 har färre än {MIN_EPISODES} hela episoder")
    if m1 is None or not m1 > 0:
        why.append("S1:s snitt är inte > 0")
    out.update({"S1_episodes": len(starts["S1"]), "S1_complete": len(c1), "S1_mean": _f(m1),
                "pass": not why, "why": why})
    return out


def proposals(miner: str, ratio: str, hyp: dict) -> list:
    """Vad utfallet leder till enligt specen, för en kvot. Förslag — ägarens beslut."""
    h1, h2, h3 = (bool((hyp.get(k) or {}).get("pass")) for k in ("H1", "H2", "H3"))
    weak = " (svagare: AISC är en approximation för juniorerna)" if miner == "GDXJ" else ""
    out = []
    if h1:
        out.append(f"{ratio}: H1 håller → förslag: ersätt drivarens prisvillkor ({miner}:s eget prisläge ≤ 20) med "
                   f"marginalvillkoret (P_M ≤ {S3_M_MAX:.0f}) för KÖP-STÖD och motorns +1{weak}.")
        if h3:
            out.append(f"{ratio}: H3 håller också → alternativet att bara ta bort drivarens nedgradering redovisas "
                       f"bredvid.")
    elif h3:
        out.append(f"{ratio}: H3 håller men inte H1 → förslag: ta bort drivarens nedgradering, så att kvoten ≥ 90 "
                   f"räcker{weak}.")
    if h2:
        out.append(f"{ratio}: H2 håller → förslag: marginalmätaren som egen bekräftelse, prövas först på "
                   f"papper{weak}.")
    if not out:
        out.append(f"{ratio}: inget håller → regeln behålls, och marginalmätaren är kvar som kontext.")
    return out


# ── Data och AISC-tabellen (beskrivning) ─────────────────────────────────────
def data_info(series: dict) -> dict:
    out = {}
    for t in TICKERS:
        s = _clean(series.get(t))
        out[t] = ({"first": _d(s.index[0]), "last": _d(s.index[-1]), "days": int(len(s))} if len(s)
                  else {"first": None, "last": None, "days": 0})
    return out


def missing(series: dict) -> list:
    """Tickers utan kurser (eller med färre än tre års dagskurser)."""
    return [t for t in TICKERS if len(_clean(series.get(t))) < MIN_DAYS]


def aisc_info(rows=None, used: Optional[pd.Series] = None, today=None) -> dict:
    """Tabellens omfång och vilka AISC-värden studien faktiskt använde (point-in-time)."""
    rows = mm.AISC_TABLE if rows is None else list(rows)
    out = {"rows": len(rows)}
    if rows:
        first = min(rows, key=lambda r: (mm._period_end_ord(r.period), mm._parse_period(r.period)[1] is None))
        pubs = sorted(pd.Timestamp(r.published) for r in rows)
        newest = mm.newest_period(rows)
        out.update({"first_period": first.period, "newest_period": newest,
                    "first_published": _d(pubs[0]), "last_published": _d(pubs[-1]),
                    "pit_from": _d(pubs[0]), "stale": bool(mm.is_stale(newest, today))})
    if used is not None:
        u = pd.Series(used, dtype=float).dropna()
        if len(u):
            out.update({"used_min": _f(u.min(), 1), "used_max": _f(u.max(), 1),
                        "used_from": _d(u.index[0]), "used_to": _d(u.index[-1])})
    return out


def study(series: dict, rows=None, start=START, seed: int = SEED, n_boot: int = N_BOOT, today=None) -> dict:
    """Hela studien ur färdiga kursserier (inget nätverk)."""
    out: dict = {"miners": {}, "proposals": []}
    used = []
    for miner, key in MINERS:
        spec = live_spec(key)
        num, den = spec.primary_num, spec.primary_den
        fr = signal_frame(series[num], series[den], series[GOLD], rows=rows, direction=spec.cheap_direction)
        res = analyze(fr, start=start, seed=seed, n_boot=n_boot)
        res.update({"ratio": f"{num}/{den}", "live_key": key})
        res["proposals"] = proposals(den, res["ratio"], res["hypotheses"])
        out["miners"][den] = res
        out["proposals"] += res["proposals"]
        used.append(fr["aisc"][fr.index >= pd.Timestamp(start)])
        log.info("%s: %d dagar, signaldagar %s", res["ratio"], len(fr),
                 " · ".join(f"{s} {res['signals'][s]['signal_days']}" for s, _ in SIGNALS))
    allused = pd.concat(used).sort_index() if used else None
    if allused is not None:
        allused = allused[~allused.index.duplicated()]
    out["aisc"] = aisc_info(rows, allused, today)
    return out


# ── Rapport ──────────────────────────────────────────────────────────────────
def _pct(v, nd: int = 1) -> str:
    """'+5,1 %' / '−3,2 %' / '–'."""
    if v is None or not math.isfinite(float(v)):
        return "–"
    s = f"{float(v) * 100:+.{nd}f}".replace(".", ",").replace("-", "−")
    return f"{s} %"


def _p(v) -> str:
    return "–" if v is None else f"{float(v):.0f}"


def _int(v) -> str:
    return f"{int(v):,}".replace(",", " ")


def _table_line(label: str, s: dict, days=None) -> str:
    """En rad i tabellen. Signal: dagar = signaldagar, ep. = episoder. Baslinje (days=None): dagar = köpdagar."""
    h = HORIZONS
    full = "/".join(str(s.get(f"n_{x}", 0)) for x in h)
    d, ep = (_int(s["n"]), "") if days is None else (_int(days), str(s["n"]))
    hit = s.get(f"hit_{h[-1]}")
    hit_s = f"{hit * 100:.0f} %" if hit is not None else "–"
    return (f"  {label:<28}{d:>7}{ep:>5}  {full:<16}"
            + "".join(f"{_pct(s.get(f'rel_{x}')):>10}" for x in h)
            + f"{_pct(s.get(f'rel_{h[-1]}_median')):>10}{hit_s:>8}"
            + f"{_pct(s.get(f'tot_{h[-1]}')):>10}{_pct(s.get(f'dd_{DD_DAYS}')):>10}")


def _miner_lines(miner: str, m: dict) -> list:
    sig = m["signals"]
    cov = m["coverage"]

    def span(c):
        return f"{c['first']}–{c['last']} ({_int(c['days'])} dagar)" if c else "kan inte räknas"
    lines = [f"== {miner} ({m['ratio']}, panelens {m['live_key']}) ==",
             f"  Täckning från {START}: S0/S1 {span(cov['S1'])} · S2/S3 {span(cov['S3'])}",
             f"  {'signal':<28}{'dagar':>7}{'ep.':>5}  {'hela 63/126/252':<16}"
             f"{'GLD 63':>10}{'GLD 126':>10}{'GLD 252':>10}{'median':>10}{'träff':>8}"
             f"{miner + ' 252':>10}{'nedg.':>10}"]
    for s, label in SIGNALS:
        lines.append(_table_line(f"{s} {label}", sig[s]["summary"], sig[s]["signal_days"]))
    for period, short in (("kvotperioden", "kvot"), ("marginalperioden", "marginal")):
        b = (m.get("baseline") or {}).get(period)
        if not b:
            lines.append(f"  bas {short}: kan inte räknas")
            continue
        for kind in ("alla dagar", "var 63:e dag"):
            lines.append(_table_line(f"bas {kind}, {short}", b[kind]))
    lines.append(f"  (dagar = signaldagar, för baslinjen köpdagar; ep. = episoder; hela = antal med hel horisont; "
                 f"GLD x = snitt mot GLD över x dagar; median, träff (andel > 0) och {miner} 252 "
                 f"(totalavkastning) gäller 252 dagar; nedg. = snitt av värsta nedgången inom 252 dagar; "
                 f"bas kvot/marginal = baslinjen i kvot-/marginalperioden)")
    lines.append(f"  Episoder (start → köp · percentil kvot/GLD/{miner}/M · mot GLD 63/126/252 · värsta nedgång):")
    any_ep = False
    for s, _ in SIGNALS:
        for e in sig[s]["episodes"]:
            any_ep = True
            lines.append(f"    {s} {e['start']} → {e['entry'] or 'i morgon'} · "
                         f"{_p(e['p_ratio'])}/{_p(e['p_gld'])}/{_p(e['p_miner'])}/{_p(e['p_m'])} · "
                         + "/".join(_pct(e.get(f"rel_{x}")) for x in HORIZONS)
                         + f" · {_pct(e.get(f'dd_{DD_DAYS}'))}")
    if not any_ep:
        lines.append("    inga")
    lines += _hyp_lines(m["hypotheses"])
    return lines


def _verdict(h: dict) -> str:
    if h.get("pass"):
        return "GODKÄND"
    return "EJ GODKÄND (" + "; ".join(h.get("why") or ["–"]) + ")"


def _hyp_lines(hyp: dict) -> list:
    h1, h2, h3 = hyp["H1"], hyp["H2"], hyp["H3"]
    out = []
    if h1.get("period_from"):
        s0 = "ingen signal" if h1["S0_no_signal"] else f"snitt mot GLD 252 {_pct(h1['S0_mean'])}"
        out.append(f"  H1 S3 mot S0 från {h1['period_from']}: S3 {h1['S3_episodes']} ep. (hela: {h1['S3_complete']}, "
                   f"snitt mot GLD 252 {_pct(h1['S3_mean'])}) · S0 {h1['S0_episodes']} ep. (hela: "
                   f"{h1['S0_complete']}, {s0}) → {_verdict(h1)}")
        if "S3_dates" in h1:
            out.append(f"    H1:s episoder (S0 räknad på S3:s dagar): S3 {', '.join(h1['S3_dates']) or 'inga'} · "
                       f"S0 {', '.join(h1['S0_dates']) or 'inga'}")
    else:
        out.append(f"  H1 → {_verdict(h1)}")
    if h2.get("period_from"):
        b = h2["bootstrap"]
        lo, hi = b["ci"]
        out.append(f"  H2 S2 mot baslinjen från {h2['period_from']} ({_int(h2['baseline_days'])} dagar): S2 snitt "
                   f"{_pct(h2['S2_mean'])} (hela ep.: {h2['S2_complete']}) · baslinje {_pct(h2['baseline_mean'])} · "
                   f"95 % [{_pct(lo)}; {_pct(hi)}] ({_int(b['empty'])} av {_int(b['n_boot'])} dragningar utan episod)"
                   f" → {_verdict(h2)}")
    else:
        out.append(f"  H2 → {_verdict(h2)}")
    if h3.get("period_from"):
        out.append(f"  H3 S1 utan drivaren från {h3['period_from']}: {h3['S1_episodes']} ep. (hela: "
                   f"{h3['S1_complete']}), snitt mot GLD 252 {_pct(h3['S1_mean'])} → {_verdict(h3)}")
    else:
        out.append(f"  H3 → {_verdict(h3)}")
    return out


def report_lines(res: dict) -> list:
    """Kompakt svensk rapport för loggen."""
    lines = [f"GRUVORNA MOT MARGINALEN — spec version {res.get('spec_version', SPEC_VERSION)} ({SPEC})",
             "Forskning: ändrar inga regler. Marginalmätaren i panelen är fortfarande bara kontext."]
    data = res.get("data") or {}
    if data:
        lines.append("Kurser: " + " · ".join(
            f"{t} {d['first']}–{d['last']} ({_int(d['days'])} dagar)" if d.get("days") else f"{t} saknas"
            for t, d in data.items()))
    if res.get("error"):
        return lines + [f"FEL: {res['error']}"]
    a = res.get("aisc") or {}
    if a.get("rows"):
        lines.append(f"AISC-tabellen: {a['rows']} rader, perioder {a['first_period']}–{a['newest_period']}"
                     f"{' (INAKTUELL)' if a.get('stale') else ''}, publicerade {a['first_published']}–"
                     f"{a['last_published']}; point-in-time känd från {a['pit_from']}"
                     + (f"; använt {_int(a['used_min'])}–{_int(a['used_max'])} USD/uns ({a['used_from']}–{a['used_to']})"
                        if a.get("used_min") is not None else ""))
    elif a:
        lines.append("AISC-tabellen: tom — S2 och S3 kan inte räknas")
    for miner, m in (res.get("miners") or {}).items():
        lines += _miner_lines(miner, m)
    lines.append("Förslag enligt specen (bara förslag — ägarens beslut; studien ändrar inga regler):")
    lines += [f"  • {p}" for p in res.get("proposals") or []]
    return lines


# ── I/O ──────────────────────────────────────────────────────────────────────
def load(getter=None) -> dict:
    """{ticker: dagsstängningar} för GLD, GDX, GDXJ och GC=F, längsta historiken (market_prices, "max")."""
    if getter is None:
        def getter(ticker):
            return cr._download_close(ticker, PERIOD)
    return {t: getter(t) for t in TICKERS}


def run(getter=None, save=None, no_save=False, rows=None, n_boot: int = N_BOOT, today=None) -> dict:
    res = {"spec_version": SPEC_VERSION, "spec": SPEC, "generated": datetime.now(tz=timezone.utc).isoformat()}
    series = load(getter)
    if getter is None:
        try:
            import market_prices
            log.info("%s", market_prices.stats_text())
        except Exception:
            pass
    res["data"] = data_info(series)
    gaps = missing(series)
    if gaps:
        res["missing"] = gaps
        res["error"] = (f"kurser saknas för {', '.join(gaps)} (behöver minst {_int(MIN_DAYS)} dagskurser) — "
                        f"inget sparas")
        for line in report_lines(res):
            log.error("%s", line)
        return res
    res.update(study(series, rows=rows, n_boot=n_boot, today=today))
    for line in report_lines(res):
        log.info("%s", line)
    if not no_save:
        if save is None:
            from gist_storage import save_blob as save
        res["saved"] = bool(save(BLOB, res))
        log.info("Gisten (%s): %s", BLOB, "sparad" if res["saved"] else "inte sparad — kontrollera GITHUB_TOKEN")
    return res


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Gruvorna mot marginalen — förregistrerat backtest.")
    parser.add_argument("--no-save", action="store_true", help="spara inte i Gisten")
    args = parser.parse_args(argv)
    res = run(no_save=args.no_save)
    return 1 if res.get("missing") else 0


if __name__ == "__main__":
    sys.exit(main())
