#!/usr/bin/env python3
"""
valhall_fas3.py — Valhall fas 3a: förbättrar cykelstatusen Viking Nine och BERSERK? (2008–idag)

  1. Valhalls dagliga cykelserier: nordiska sektorindex (likviditetsfilter), SPDR-fonderna och
     råvarutemanas drivare (berserk.backtest.pick_driver — samma drivare som BERSERK-backtestet).
  2. Viking Nine-backtestet (viking_backtest, oförändrat) på Norden 50 + Norden OOS 100 och BERSERK
     (gällande regler, nordiska producenter) från 2008.
  3. Varje affär märks med läget på signaldagen (valhall.backtest.tag) → tabeller per segment/tema,
     status och period, och portföljen med filter (bara GRÖN, GRÖN + GUL, …).

Körs manuellt i Actions (valhall-backtest.yml). Resultatet skrivs i loggen och sparas i Gisten
(valhall_fas3.json) om GITHUB_TOKEN finns. Env: BORSDATA_API_KEY.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("valhall_fas3")

BLOB = "valhall_fas3.json"
START_YEAR = 2008


def _close(df):
    if df is None or len(df) == 0 or "Close" not in df:
        return None
    s = df["Close"].astype(float).dropna()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s


def frames(api, getter, data_getter, nordic_market=None, keep: dict = None) -> dict:
    """{"nordic": {seg: frame}, "global": {seg: frame}, "themes": {tema: frame}, "universe": {...},
    "bench": {"nordic": hela Norden, "global": SPY}}. keep = {} fylls med aktiernas Close/Volume (fas 4b)."""
    import markets as _markets
    from berserk import backtest as bbt
    from berserk import themes as bth
    from valhall import board as vb
    from valhall import cycle
    from valhall import nordic_index as ni
    from valhall import segments as sgm
    from valhall import swing
    table = _markets.load(api=api)
    nordic_ids = _markets.nordic_stock_ids(table)
    instruments = api.get_instruments()
    to_yf = lambda t, m: _markets.to_yf(t, m, table)  # noqa: E731
    universe = swing.nordic_universe(instruments, nordic_ids, to_yf)
    indices = ni.build_indices(ni.members(instruments, nordic_ids, to_yf),
                               {t: v["currency"] for t, v in universe.items()}, data_getter, keep=keep)
    nordic_all = (indices.get(ni.ALL) or {}).get("index")
    spy = _close(getter("SPY", "max"))
    out = {"nordic": {}, "global": {}, "themes": {}, "universe": universe, "bench": {"nordic": nordic_all, "global": spy}}
    for s in sgm.SEGMENTS:
        idx = (indices.get(s.key) or {}).get("index")
        if idx is not None:
            out["nordic"][s.key] = cycle.cycle_frame(idx, None, nordic_market if nordic_market is not None
                                                     else nordic_all, s.necessity, cykel_base=nordic_all)
        df = getter(s.spdr, "max")
        c = _close(df)
        if c is not None:
            out["global"][s.key] = cycle.cycle_frame(c, df["Volume"] if "Volume" in df else None, spy,
                                                     s.necessity, cykel_base=spy)
    series = {}
    for theme in [t for s in sgm.SEGMENTS for t in s.themes]:
        for sym in bth.drivers(theme):
            if sym not in series:
                series[sym] = _close(getter(sym, "max"))
        sym, d = bbt.pick_driver(theme, f"{START_YEAR}-01-01", series)
        if d is None:
            d = vb.producer_basket(theme, getter, "max")
        if d is not None:
            out["themes"][theme] = cycle.cycle_frame(d, None, spy, vb.THEME_NECESSITY.get(theme, 50))
    log.info("Cykelserier: Norden %d, globalt %d, teman %d", len(out["nordic"]), len(out["global"]), len(out["themes"]))
    return out


def _fmt_table(rows: list, title: str) -> list:
    lines = [f"== {title}"]
    for r in rows:
        lines.append(f"  {str(r['group']):18s} {r['status']:11s} {r['period']:10s} n={r['n']:4d} exp={r['exp']:+.3f} "
                     f"win={r['win']:5.1f} sumR={r['sum_r']:+7.1f} pf={r['pf']}")
    return lines


def _fmt_port(rows: list, title: str) -> list:
    lines = [f"== {title}"]
    for r in rows:
        lines.append(f"  {r['filter']:26s} affärer={r['trades']:4d} tagna={r['taken']:4d} exp={r['exp']} "
                     f"portf.exp={r['port_exp']} CAGR={r['cagr']} DD={r['dd']} MC p5={r['mc_p5']} "
                     f"p50={r['mc_p50']} DD95={r['mc_dd95']}")
    return lines


def run(api=None, getter=None, data_getter=None, viking_run=None, berserk_run=None, today=None,
        save=None, no_save=False) -> dict:
    import viking_backtest as vbt
    from berserk import backtest as bbt
    from berserk import universe as buv
    from valhall import backtest as vbk
    if api is None:
        from borsdata_api import get_api
        api = get_api()
    if getter is None:
        from market_prices import ohlcv as getter
    if data_getter is None:
        from market_prices import closes_volumes as data_getter
    fr = frames(api, getter, data_getter)
    universe = fr["universe"]
    seg_of = lambda t: (universe.get(t) or {}).get("segment")  # noqa: E731
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    years = (today - pd.Timestamp(f"{START_YEAR}-01-01")).days / 365.25
    yrs = {"2008–2020": 13.0, "2021–": (today - pd.Timestamp("2021-01-01")).days / 365.25}

    tickers = list(dict.fromkeys(vbt.NORDIC_50 + vbt.NORDIC_OOS_100))
    vres = (viking_run or vbt.run)(tickers, cfg=vbt.Config(start_year=START_YEAR), today=today)
    vtag = vbk.tag(vres["trades"], seg_of, fr)
    bres = (berserk_run or bbt.run)(list(buv.NORDIC), cfg=bbt.live_config(start_year=START_YEAR), today=today)
    btag = vbk.tag(bres["trades"], seg_of, fr, theme_of=buv.theme_of)

    vfil = {k: v for k, v in vbk.FILTERS.items() if not k.startswith("Tema")}
    bfil = {k: vbk.FILTERS[k] for k in ("Alla affärer", "Tema GRÖN", "Tema ej RÖD", "Norden GRÖN", "Norden ej RÖD")}
    result = {
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "viking": {"by_segment": vbk.table(vtag, "segment"), "all": vbk.table(vtag, "all"),
                   "global": vbk.table(vtag, "segment", key="g_status"),
                   "portfolio": vbk.portfolios(vtag, years, filters=vfil),
                   "portfolio_by_period": {p: vbk.portfolios(vtag, yrs[p], filters=vfil, period=p) for p in yrs}},
        "berserk": {"by_theme": vbk.table(btag, "theme", key="t_status"), "all": vbk.table(btag, "all", key="t_status"),
                    "portfolio": vbk.portfolios(btag, years, pc=bbt.live_portfolio(), filters=bfil),
                    "portfolio_by_period": {p: vbk.portfolios(btag, yrs[p], pc=bbt.live_portfolio(), filters=bfil,
                                                              period=p) for p in yrs}},
    }
    # Fas 3b (K9): VÄNDER-fasen och bekräftad vändning som grind, och Viking Nine per komplex
    result["fas3b"] = {"viking": vbk.portfolios(vtag, years, filters=vbk.FILTERS_3B),
                       "viking_by_period": {p: vbk.portfolios(vtag, yrs[p], filters=vbk.FILTERS_3B, period=p) for p in yrs},
                       "berserk": vbk.portfolios(btag, years, pc=bbt.live_portfolio(), filters=vbk.FILTERS_3B_THEME),
                       "berserk_by_period": {p: vbk.portfolios(btag, yrs[p], pc=bbt.live_portfolio(),
                                                               filters=vbk.FILTERS_3B_THEME, period=p) for p in yrs}}
    # Kostnadsmodellen (backtest_costs): samma affärer och filter, R efter courtage, spread, växling och stoppglidning
    import backtest_costs as bc
    vnet = [{**r, "trade": t} for r, t in zip(vtag, bc.net_trades([r["trade"] for r in vtag]))]
    bnet = [{**r, "trade": t} for r, t in zip(btag, bc.net_trades([r["trade"] for r in btag]))]
    result["costs"] = {
        "model_bps": dict(bc.ROUND_TRIP_BPS),
        "viking": {p: bc.compare([r["trade"] for r in vtag if p == "2008–idag" or r["period"] == p])
                   for p in ("2008–idag",) + tuple(yrs)},
        "berserk": {p: bc.compare([r["trade"] for r in btag if p == "2008–idag" or r["period"] == p])
                    for p in ("2008–idag",) + tuple(yrs)},
        "viking_portfolio": vbk.portfolios(vnet, years, filters=vbk.FILTERS_3B),
        "berserk_portfolio": vbk.portfolios(bnet, years, pc=bbt.live_portfolio(), filters=vbk.FILTERS_3B_THEME)}
    try:
        import complex_backtest as cbt
        import ovtlyr_nine as on
        from valhall import complexes as vcx
        drivers = {}
        for k in vcx.KEYS:
            closes = {t: fr["themes"][t]["close"] for t in vcx.themes_of(k) if "close" in fr["themes"].get(t, {})}
            _syms, drivers[k] = vcx.basket(k, {t: (t, c) for t, c in closes.items()})
        result["fas3b"]["complex"] = cbt.run(getter, nordic=on.nordic_market(int((years + 2) * 262)), drivers=drivers,
                                             cfg=vbt.Config(start_year=START_YEAR))
    except Exception as exc:
        log.exception("Fas 3b — komplexvarianten fallerade: %s", exc)
        result["fas3b"]["complex"] = None
    lines_3b = (_fmt_port(result["fas3b"]["viking"], "FAS 3B — VIKING NINE, VÄNDNINGSGRINDEN 2008–idag")
                + sum((_fmt_port(v, f"FAS 3B — VIKING NINE {p}") for p, v in result["fas3b"]["viking_by_period"].items()), [])
                + _fmt_port(result["fas3b"]["berserk"], "FAS 3B — BERSERK, TEMATS VÄNDNING 2008–idag")
                + sum((_fmt_port(v, f"FAS 3B — BERSERK {p}") for p, v in result["fas3b"]["berserk_by_period"].items()), [])
                + [""] + ["FAS 3B — VIKING NINE PER KOMPLEX (standard SPDR · komplexets ETF · + drivargrind)"]
                + [f"  {r['complex']:<13} {r['variant']:<9} n {r['n']:>4} · exp {r['exp']} R · vinst {r['win']} % · "
                   f"summa {r['sum_r']} R · netto exp {r.get('exp_net')} R" for r in result["fas3b"]["complex"] or []])
    lines_cost = ([f"KOSTNADER — tur och retur {result['costs']['model_bps']} bp + stoppglidning (backtest_costs)"]
                  + [f"  {eng:<8} {p:<10} n {c['n']:>4} · exp {c['exp']} → {c['exp_net']} R · summa {c['sum_r']} → "
                     f"{c['sum_r_net']} R · kostnad {c['cost_r']} R/affär"
                     for eng in ("viking", "berserk") for p, c in result["costs"][eng].items()]
                  + _fmt_port(result["costs"]["viking_portfolio"], "KOSTNADER — VIKING NINE NETTO, VÄNDNINGSGRINDEN")
                  + _fmt_port(result["costs"]["berserk_portfolio"], "KOSTNADER — BERSERK NETTO, TEMATS VÄNDNING"))
    for line in (_fmt_table(result["viking"]["all"], "VIKING NINE — ALLA SEGMENT (Nordens status)")
                 + _fmt_table(result["viking"]["by_segment"], "VIKING NINE — PER SEGMENT (Nordens status)")
                 + _fmt_table([r for r in result["viking"]["global"] if r["status"] != "alla"],
                              "VIKING NINE — PER SEGMENT (global status)")
                 + _fmt_port(result["viking"]["portfolio"], "VIKING NINE — PORTFÖLJ 2008–idag")
                 + sum((_fmt_port(v, f"VIKING NINE — PORTFÖLJ {p}") for p, v in result["viking"]["portfolio_by_period"].items()), [])
                 + _fmt_table(result["berserk"]["all"], "BERSERK — ALLA (temats status)")
                 + _fmt_table(result["berserk"]["by_theme"], "BERSERK — PER TEMA (temats status)")
                 + _fmt_port(result["berserk"]["portfolio"], "BERSERK — PORTFÖLJ 2008–idag")
                 + sum((_fmt_port(v, f"BERSERK — PORTFÖLJ {p}") for p, v in result["berserk"]["portfolio_by_period"].items()), [])
                 + lines_3b + lines_cost):
        log.info("%s", line)
    if not no_save:
        if save is None:
            from gist_storage import save_blob as save
        result["saved"] = bool(save(BLOB, result))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    run(no_save=args.no_save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
