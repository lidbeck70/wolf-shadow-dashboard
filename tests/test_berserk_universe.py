"""
🪓 BERSERK PR 0 — teman, drivare, universum och datasonden. Inget nätverk:
sonden körs med en fejkad hämtare.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from berserk import probe  # noqa: E402
from berserk import themes as th  # noqa: E402
from berserk import universe as uv  # noqa: E402


def test_every_theme_has_a_complex_and_label():
    for theme, (lbl, cx, drv) in th.THEMES.items():
        assert lbl and cx in th.COMPLEXES, theme
        assert isinstance(drv, tuple)
    assert th.drivers("lax") == () and th.drivers("olja")[0] == "BZ=F"


def test_themes_shared_with_ember_have_the_same_complex():
    from ember.config import THEME_TO_COMPLEX
    shared = set(th.THEMES) & set(THEME_TO_COMPLEX)
    assert {"olja", "koppar", "guld", "uran", "skog", "vete"} <= shared
    for theme in shared:
        assert th.complex_of(theme) == THEME_TO_COMPLEX[theme], theme


def test_whole_commodity_market_is_covered():
    cx = {th.complex_of(t) for t in th.THEMES}
    assert cx == set(th.COMPLEXES)
    for theme in ("olja", "naturgas", "kol", "uran", "koppar", "aluminium", "stal", "jarnmalm", "guld", "silver",
                  "vete", "majs", "soja", "godsel", "skog", "lax", "tank", "torrbulk"):
        assert theme in th.THEMES, theme


def test_universe_maps_every_ticker_to_a_known_theme():
    for t, theme in {**uv.PRODUCERS, **uv.ETFS}.items():
        assert theme in th.THEMES or theme in th.BASKETS, (t, theme)
    assert len(uv.NORDIC) >= 35 and len(uv.GLOBAL) >= 40 and len(uv.ETFS) >= 25
    assert all(t.endswith((".ST", ".OL", ".CO", ".HE")) for t in uv.NORDIC)
    assert not set(uv.NORDIC) & set(uv.GLOBAL) and not set(uv.PRODUCERS) & set(uv.ETFS)


def test_dead_tickers_are_renamed_or_dropped():
    assert "GOLD" not in uv.GLOBAL and uv.GLOBAL.get("B") == "guld"         # Barrick → B
    assert "GOGL.OL" not in uv.NORDIC


def test_kind_theme_and_driver_lookup():
    assert uv.kind_of("BOL.ST") == "producent" and uv.theme_of("bol.st") == "koppar"
    assert uv.driver_candidates("BOL.ST") == th.drivers("koppar")
    assert uv.kind_of("GLD") == "etf" and uv.driver_candidates("GLD") == th.drivers("guld")
    assert uv.driver_candidates("DBA") == ("DBA",)                          # korgen är sin egen drivare
    assert uv.driver_candidates("MOWI.OL") == () and uv.kind_of("XYZ") == ""
    assert set(uv.by_theme()["lax"]) == {"MOWI.OL", "SALM.OL", "LSG.OL", "BAKKA.OL", "GSF.OL"}


# ── Datasonden ──────────────────────────────────────────────────────────────
TODAY = pd.Timestamp("2026-10-02")


def _df(start, end="2026-10-01"):
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({"Close": np.linspace(10, 20, len(idx))}, index=idx)


def _getter(symbol, period):
    assert period == "max"
    if symbol in ("BZ=F", "CL=F"):
        raise RuntimeError("blockerad")                                    # olja faller till BNO
    if symbol == "TTF=F":
        return _df("2017-10-02")
    if symbol == "MTF=F":
        return _df("2015-01-01", "2024-01-01")                             # nedlagd → GAMMAL
    if symbol in ("UFV=F", "BWET"):
        return pd.DataFrame()
    return _df("2005-01-03")


def test_probe_picks_first_working_driver_and_flags_problems():
    cands = [("Drivare", t, s) for t in ("olja", "naturgas_eu", "godsel", "lax") for s in th.drivers(t)]
    cands += [("Drivare", "kol", "MTF=F")]                                   # nedlagd termin → GAMMAL
    cands += [("ETF", "guld", "GLD"), ("Norden", "koppar", "BOL.ST")]
    rows = probe.run(cands, getter=_getter, out=lambda *_: None, today=TODAY)
    by = {(r["group"], r["symbol"]): r for r in rows}
    assert by[("Drivare", "BZ=F")]["status"] == "FEL" and "blockerad" in by[("Drivare", "BZ=F")]["error"]
    assert by[("Drivare", "MTF=F")]["status"] == "GAMMAL"
    ch = probe.chosen_drivers(rows)
    assert ch["olja"]["symbol"] == "BNO" and ch["olja"]["oos"] is True
    assert ch["naturgas_eu"]["symbol"] == "TTF=F" and ch["naturgas_eu"]["oos"] is False
    assert ch["kol"] is None and ch["godsel"] is None
    md = probe.markdown(rows)
    assert "Vald drivare per tema" in md and "`BNO`" in md and "**DATA UNAVAILABLE**" in md
    assert "ingen prisserie" in md and "Drivare saknas:" in md and "Kol" in md


def test_probe_candidates_cover_everything_and_never_crash():
    cands = probe.candidates()
    syms = {s for _g, _t, s in cands}
    assert set(th.all_driver_symbols()) <= syms and set(uv.ETFS) <= syms and set(uv.PRODUCERS) <= syms

    def boom(symbol, period):
        raise ValueError("nätet nere")
    rows = probe.run(cands[:5], getter=boom, out=lambda *_: None, today=TODAY)
    assert all(r["status"] == "FEL" for r in rows)


def test_workflow_runs_the_probe():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".github", "workflows",
                        "berserk-probe.yml")
    text = open(path, encoding="utf-8").read()
    assert "python -m berserk.probe" in text and "workflow_dispatch" in text


def test_probe_findings_are_applied():
    assert "BELCO.OL" not in uv.NORDIC                                       # ingen kurshistorik hos Yahoo
    assert th.drivers("kol") == ()                                           # MTF=F slutade uppdateras 2025


# ── PR 1b: större universum och regioner ────────────────────────────────────
def test_regions_and_sizes():
    sizes = {k: len(v) for k, v in uv.REGIONS.items()}
    assert sizes["USA"] >= 120 and sizes["Kanada"] >= 25 and sizes["London"] >= 10 and sizes["Australien"] >= 20
    assert len(uv.GLOBAL) == sum(v for k, v in sizes.items() if k != "Norden") >= 190
    assert set(uv.LISTS) == set(uv.REGIONS) | {"Råvaru-ETF:er"}
    for region, members in uv.REGIONS.items():
        assert all(uv.region_of(t) == region for t in members), region


def test_one_listing_per_company_and_every_theme_known():
    for dup in ("RIO", "BHP", "K.TO", "CCO.TO", "NXE.TO", "DML.TO", "BHP.AX", "RIO.AX", "FRO", "EDV.TO"):
        assert dup not in uv.PRODUCERS, dup
    keys = [k for m in uv.REGIONS.values() for k in m]
    assert len(keys) == len(set(keys))
    assert all(theme in th.THEMES for theme in uv.GLOBAL.values())


def test_region_index_and_region_of():
    assert uv.REGION_INDEX == {"Norden": "OMXS30", "USA": "SPY", "Kanada": "^GSPTSE", "London": "^FTSE",
                               "Australien": "^AXJO"}
    assert [uv.region_of(t) for t in ("BOL.ST", "SU.TO", "GLEN.L", "FMG.AX", "FCX", "GLD")] == \
        ["Norden", "Kanada", "London", "Australien", "USA", "USA"]


def test_probe_checks_region_indices():
    syms = {s for g, _t, s in probe.candidates() if g == "Index"}
    assert syms == {"SPY", "^GSPTSE", "^FTSE", "^AXJO"}
    groups = {g for g, _t, _s in probe.candidates()}
    assert set(uv.REGIONS) <= groups


def test_probe_2026_10_removals():
    for t in ("CTRA", "NGD", "PCH", "MEG.TO", "ARX.TO"):
        assert t not in uv.PRODUCERS, t
