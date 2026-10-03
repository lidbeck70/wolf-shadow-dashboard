"""
🐺 Fiat Debasement PR 3 — sidan REGIME → Makro → 🐺 Fiat Debasement.
Syntetiska serier via en falsk fetch — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiat_debasement import config as cfg  # noqa: E402
from fiat_debasement import data as fd  # noqa: E402
from fiat_debasement import snapshot as fs  # noqa: E402
from fiat_debasement import sources as src  # noqa: E402
from fiat_debasement import ui as fu  # noqa: E402
from ui import nav  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
END = "2026-10-02"


def _growth(idx, pct, base):
    yrs = (idx - idx[0]).days / 365.25
    return pd.Series(base * (1 + pct / 100) ** np.asarray(yrs), index=idx)


GROWTH = {"SEK": 6.0, "EUR": 5.0, "USD": 7.0}


def _series_for(concept, ccy, spec):
    if concept == cfg.M2:
        return _growth(pd.date_range("1999-01-01", "2026-08-01", freq="MS"), GROWTH[ccy], 1000.0)
    if concept == cfg.CPI_LONG:
        return _growth(pd.date_range("1914-07-01", "2026-08-01", freq="MS"), 3.0, 100.0)
    if concept in (cfg.CPI, cfg.CORE):
        start = "1996-12-01" if ccy == "EUR" else "1980-01-01"
        return _growth(pd.date_range(start, "2026-08-01", freq="MS"), 2.5, 100.0)
    if concept == cfg.GDP:
        return _growth(pd.date_range("1995-01-01", "2026-04-01", freq="QS"), 2.0, 1000.0)
    if concept in (cfg.DEBT, cfg.DEBT_INFO):
        return pd.Series(60.0, index=pd.date_range("2000-01-01", "2026-01-01", freq="QS"))
    if concept == cfg.FX:
        idx = pd.bdate_range("1999-01-04", END)
        return pd.Series(10.0 if ccy == "SEK" else 1.1, index=idx)
    return _growth(pd.bdate_range("2006-10-03", END), 5.0, 100.0)


def _fake_fetch_factory(fail=False):
    lookup = {}
    for (concept, ccy), specs in cfg.SERIES.items():
        for spec in specs:
            lookup[fd._cache_key(spec)] = (concept, ccy)

    def fetch(spec, **_kw):
        if fail:
            return src.SeriesData(spec.get("kind", "?"), str(spec.get("id")), error="HTTP 503", label=spec.get("label", ""))
        sid = str(spec.get("id") or spec.get("key") or spec.get("dataset"))
        if spec.get("id") in (21031, "GC=F", 21032, "SI=F"):
            gold = spec.get("id") in (21031, "GC=F")
            base = 280.0 if gold else 5.0
            full = _growth(pd.bdate_range("2000-08-30", END), 9.0 if gold else 7.0, base)
            s = full[full.index >= "2006-10-03"] if spec["kind"] == "borsdata" else full * 1.005
        else:
            concept, ccy = lookup[fd._cache_key(spec)]
            s = _series_for(concept, ccy, spec)
        sd = src.SeriesData(spec["kind"], sid, values=s, unit=spec.get("unit", ""), label=spec.get("label", ""),
                            last_updated="2026-10-03 10:00 UTC")
        sd.frequency = src.frequency_of(s.index)
        return sd
    return fetch


@pytest.fixture
def fake_data(monkeypatch):
    fd.clear_cache()
    monkeypatch.setattr(fd, "fetch", _fake_fetch_factory())
    yield
    fd.clear_cache()


# ── Rena hjälpare ───────────────────────────────────────────────────────────
def test_colors_are_only_a_visual_aid_and_numbers_are_formatted():
    assert fu.color_for("m2_yoy", 2.0) == fu.GREEN and fu.color_for("m2_yoy", 5.0) == fu.AMBER
    assert fu.color_for("m2_yoy", 9.0) == fu.RED and fu.color_for("m2_yoy", None) == fu.GREY
    assert fu.color_for("gdp_yoy", 3.0) == fu.GREEN and fu.color_for("gdp_yoy", -1.0) == fu.RED
    assert fu.color_for("gold_5y", -45.0) == fu.RED and fu.color_for("okänd", 1.0) == fu.GREY
    assert fu.fmt(None) == fu.NA and fu.fmt(4.04, "pe") == "+4.0 pe" and fu.fmt(62.5, sign=False) == "62.5 %"


def test_overview_rows_show_unavailable_not_zero():
    rows = fu.overview_rows({"SEK": None})
    assert [r["currency"] for r in rows] == list(cfg.CURRENCIES)
    assert all(c["text"] == fu.NA and c["value"] is None for r in rows for c in r["cells"])


def test_principle_sentence_describes_and_never_advises():
    snap = fs.Snapshot("SEK", {"monetary_gap": fs.Metric(4.0, "2026-04-01", "x", "pe")})
    txt = fu.principle_sentence(snap)
    assert "växer penningmängden över den reala produktionen" in txt and "+4.0" in txt
    for word in ("köp", "sälj", "kollaps", "buy", "sell"):
        assert word not in txt.lower()
    assert fu.NA in fu.principle_sentence(fs.Snapshot("EUR"))


def test_start_dates_and_usable_series():
    assert fu.start_date("1980") == pd.Timestamp("1980-01-01")
    assert fu.start_date("Eget datum", pd.Timestamp("2003-05-01").date()) == pd.Timestamp("2003-05-01")
    s = pd.Series(1.0, index=pd.date_range("1996-12-01", periods=12, freq="MS"))
    assert fu.usable_from(s, pd.Timestamp("1996-01-01")) == (True, pd.Timestamp("1996-12-01"))
    assert fu.usable_from(s, pd.Timestamp("1990-01-01"))[0] is False             # börjar för sent
    assert fu.usable_from(None, pd.Timestamp("2000-01-01")) == (False, None)


def test_power_lines_mark_missing_euro_history(fake_data):
    lines = fu.power_lines(pd.Timestamp("1970-01-01"))
    assert lines["EUR"][0] is None and fu.NA in lines["EUR"][1] and "1996-12-01" in lines["EUR"][1]
    sek, _txt = lines["SEK"]
    assert sek.iloc[0] == pytest.approx(100.0) and sek.iloc[-1] < 100           # lång KPI-serie sedan 1914
    assert lines["USD"][0] is not None
    later = fu.power_lines(pd.Timestamp("2000-01-01"))
    assert all(s is not None for s, _t in later.values())


def test_hundred_units_adjusts_start_and_marks_futures(fake_data):
    res = fu.hundred_units("SEK", pd.Timestamp("2000-01-01"))
    assert res["start"] == pd.Timestamp("2000-08-30")
    assert any("Startdatum justerat" in n for n in res["notes"])
    assert set(res["lines"]) == {"Kontanter, köpkraft (KPI)", "I guld", "I silver"}
    assert res["lines"]["I guld"].iloc[0] == pytest.approx(100.0) and res["lines"]["I guld"].iloc[-1] > 100
    assert res["lines"]["Kontanter, köpkraft (KPI)"].iloc[-1] < 100
    assert res["segments"] and all(s["kind"] == cfg.FUTURES for s in res["segments"])
    old = fu.hundred_units("USD", pd.Timestamp("1980-01-01"))
    assert "I guld" not in old["lines"] and any(fu.NA in n for n in old["notes"])    # inget guld 1980


def test_source_rows_show_what_was_used(fake_data):
    snaps = {c: fs.snapshot(c) for c in cfg.CURRENCIES}
    rows = fu.source_rows(snaps)
    used = [r for r in rows if r["Status"] == "används"]
    assert used and all(r["Hämtat"] != "—" for r in used)
    assert any(r["Begrepp"] == "Penningmängd (M2)" and r["Valuta"] == "SEK" for r in used)


# ── Navigation och sidan ────────────────────────────────────────────────────
def test_navigation():
    assert nav.options("regime") == ["Marknad", "Råvaror", "Makro"]
    assert nav.options("regime/Makro") == ["🐺 Fiat Debasement"]
    assert "REGIME → Makro → 🐺 Fiat Debasement" in nav.paths()
    assert nav.STATE_KEY["regime/Makro"] == "sub_regime_macro"
    src_panel = open(os.path.join(ROOT, "wolf_panel.py"), encoding="utf-8").read()
    assert "render_fiat_debasement_page" in src_panel and 'sub == "🐺 Fiat Debasement"' in src_panel


def _app():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["FD_TEST_ROOT"])
    from fiat_debasement.ui import render_fiat_debasement_page
    render_fiat_debasement_page()


def test_page_renders_every_section(fake_data, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FD_TEST_ROOT", ROOT)
    at = AppTest.from_function(_app, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    for part in ("FIAT OVERVIEW", "MONEY SUPPLY", "INFLATION &amp; PURCHASING POWER", "WHAT HAPPENED TO 100 UNITS?",
                 "METHODOLOGY", "växer penningmängden över den reala produktionen", "Monetary Gap",
                 "ESTIMATED PURCHASING POWER REMAINING", "Startdatum justerat"):
        assert part in html or part.replace("&amp;", "&") in html, part
    assert fu.NA not in html.split("FIAT OVERVIEW")[1].split("</table>")[0]   # alla översiktsvärden finns
    at.selectbox(key="fd_pp_start").set_value("1970").run()
    html = " ".join(m.value for m in at.markdown)
    assert "EUR: DATA UNAVAILABLE för 1970-01-01" in html and not at.exception
    at.radio(key="fd_ccy").set_value("EUR").run()
    assert not at.exception


def test_page_survives_total_api_failure(monkeypatch):
    from streamlit.testing.v1 import AppTest
    fd.clear_cache()
    monkeypatch.setattr(fd, "fetch", _fake_fetch_factory(fail=True))
    monkeypatch.setenv("FD_TEST_ROOT", ROOT)
    at = AppTest.from_function(_app, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert fu.NA in html and "0.0 %" not in html.split("FIAT OVERVIEW")[1].split("</table>")[0]
    fd.clear_cache()
