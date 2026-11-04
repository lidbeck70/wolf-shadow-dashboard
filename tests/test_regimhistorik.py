"""
Regim-historiken på Regim-sidan: överblick, stapeldiagram och perioder (bara visning).

Datan (wolf_regime.json → history, äldst först) och trösklarna 55 % / 45 % är oförändrade.
Hjälparna är rena och tål trasiga rader; diagrammet har en stapel per handelsdag i regimens
färg med höjden = bredd. Inget nätverk.
"""
import datetime as dt
import json

import pytest

import wolf_regime_ui as w

NB = "\u00a0"


def _bdays(n, end=dt.date(2026, 10, 30)):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= dt.timedelta(days=1)
    return out[::-1]


def _hist():
    """20 handelsdagar 5–30 okt: GRÖN 6 dagar, sedan GUL 14; bredd 52 % → 38 %, kvalande 412 → 269."""
    days = _bdays(20)
    out = []
    for i, d in enumerate(days):
        out.append({"date": d, "regime": "GRÖN" if i < 6 else "GUL",
                    "breadth": round(0.52 - 0.14 * i / 19, 4), "qualifying": 412 - round(143 * i / 19)})
    return out


def _plain(s):
    return s.replace(NB, " ")


# ── rena hjälpare ────────────────────────────────────────────────────────────

def test_clean_history_sorts_dedupes_and_tolerates_junk():
    raw = [
        {"date": "2026-10-03", "regime": "GUL", "breadth": 0.4, "qualifying": 300},
        {"date": "2026-10-01", "regime": "grön", "breadth": "0.55", "qualifying": 410.4},
        {"date": "2026-10-02", "regime": "BLÅ", "breadth": None},                 # okänd regim, saknad bredd
        {"date": "2026-10-03", "regime": "RÖD", "breadth": 0.3, "qualifying": 250},   # dubblett: senaste vinner
        {"date": "2026-10-04", "regime": None, "breadth": float("nan"), "qualifying": -5},
        {"date": "2026-10-05T18:00", "regime": " gul ", "breadth": 1.7, "qualifying": "x"},
        {"date": "inte ett datum", "regime": "GRÖN"}, {"regime": "GRÖN"}, "skräp", None, 42,
    ]
    rows = w.clean_history(raw)
    assert [r["date"] for r in rows] == ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"]
    assert [r["regime"] for r in rows] == ["GRÖN", "OKÄND", "RÖD", "OKÄND", "GUL"]
    assert [r["breadth"] for r in rows] == [0.55, None, 0.3, None, None]           # 1.7 är inte en andel
    assert [r["qualifying"] for r in rows] == [410, None, 250, None, None]
    for bad in ([], None, {}, "text", [None, "x"]):
        assert w.clean_history(bad) == []


def test_regime_periods_are_consecutive_runs_oldest_first():
    raw = [{"date": d, "regime": r} for d, r in (
        ("2026-10-06", "GRÖN"), ("2026-10-01", "GRÖN"), ("2026-10-02", "GRÖN"), ("2026-10-05", "GUL"),
        ("2026-10-07", "GRÖN"), ("2026-10-08", "?"), ("2026-10-09", "konstig"))]
    assert w.regime_periods(raw) == [
        {"regime": "GRÖN", "start": "2026-10-01", "end": "2026-10-02", "days": 2},
        {"regime": "GUL", "start": "2026-10-05", "end": "2026-10-05", "days": 1},
        {"regime": "GRÖN", "start": "2026-10-06", "end": "2026-10-07", "days": 2},
        {"regime": "OKÄND", "start": "2026-10-08", "end": "2026-10-09", "days": 2},
    ]
    assert w.regime_periods([]) == [] and w.regime_periods(None) == []


def test_history_summary_normal_window():
    s = w.history_summary(_hist())
    assert s["n"] == 20 and s["first"] == "2026-10-05" and s["last"] == "2026-10-30"
    assert (s["regime"], s["since"], s["days"], s["since_window_start"]) == ("GUL", "2026-10-13", 14, False)
    assert s["counts"] == {"GRÖN": 6, "GUL": 14, "RÖD": 0}
    assert (s["breadth_first"], s["breadth_last"]) == (0.52, 0.38)
    assert (s["qualifying_first"], s["qualifying_last"]) == (412, 269)


def test_history_summary_edges():
    empty = w.history_summary([])
    assert empty["n"] == 0 and empty["regime"] is None and empty["since"] is None and empty["days"] == 0
    assert empty["counts"] == {"GRÖN": 0, "GUL": 0, "RÖD": 0} and empty["breadth_last"] is None
    assert w.overview_parts(empty) == []
    s = w.history_summary([{"date": "2026-10-02", "regime": "RÖD", "qualifying": 90},
                           {"date": "2026-10-01", "regime": "RÖD", "breadth": 0.31},
                           {"date": "2026-10-05", "regime": "X", "breadth": 0.29, "qualifying": 80}])
    assert s["regime"] == "OKÄND" and s["days"] == 1 and s["counts"]["OKÄND"] == 1
    assert (s["breadth_first"], s["breadth_last"], s["breadth_points"]) == (0.31, 0.29, 2)
    assert (s["qualifying_first"], s["qualifying_last"], s["qualifying_points"]) == (90, 80, 2)


def test_overview_text_is_swedish_with_dates_counts_and_first_to_last():
    parts = w.overview_parts(w.history_summary(_hist()))
    assert parts == ["Nu: GUL sedan 13 okt (14 handelsdagar)",
                     "Senaste 20 dagar: GRÖN 6 · GUL 14 · RÖD 0",
                     f"Bredd 52{NB}% → 38{NB}%",
                     "Kvalande 412 → 269"]


def test_overview_text_edge_cases():
    whole = w.overview_parts(w.history_summary(
        [{"date": f"2026-10-0{i}", "regime": "RÖD", "breadth": 0.3} for i in (1, 2, 5)]))
    assert whole[0] == "Nu: RÖD sedan minst 1 okt (3+ handelsdagar)"         # perioden kan ha börjat före fönstret
    assert whole[1] == "Senaste 3 dagar: GRÖN 0 · GUL 0 · RÖD 3"
    assert whole[2] == f"Bredd 30{NB}% → 30{NB}%" and len(whole) == 3      # inga kvalande → ingen sådan bit
    one = w.overview_parts(w.history_summary(
        [{"date": "2026-10-01", "regime": "GRÖN"}, {"date": "2026-10-02", "regime": "GUL", "breadth": 0.381}]))
    assert one[0] == "Nu: GUL sedan 2 okt (1 handelsdag)"
    assert one[2] == f"Bredd 38{NB}%"                                       # en enda mätpunkt: ingen pil
    single = w.overview_parts(w.history_summary([{"date": "2026-10-02", "regime": "GUL"}]))
    assert single[1].startswith("Senaste dagen:")


def test_period_text():
    raw = [{"date": d, "regime": r} for d, r in (
        ("2026-09-12", "GRÖN"), ("2026-10-02", "GRÖN"), ("2026-10-03", "GUL"), ("2026-10-05", "GUL"),
        ("2026-10-06", "RÖD"), ("2026-10-07", "GUL"))]
    assert [_plain(p) for p in w.period_parts(w.regime_periods(raw))] == [
        "GRÖN 12 sep–2 okt (2 d)", "GUL 3–5 okt (2 d)", "RÖD 6 okt (1 d)", "GUL 7 okt–nu (1 d)"]
    assert [_plain(p) for p in w.period_parts(w.regime_periods(_hist()))] == [
        "GRÖN 5–12 okt (6 d)", "GUL 13 okt–nu (14 d)"]
    assert w.period_parts([]) == []


def test_swedish_formatting_helpers():
    assert w._sv_date("2026-10-08") == "8 okt" and w._sv_date("2026-01-31", year=True) == "31 jan 2026"
    assert w._sv_date("trasigt") == "trasigt" and w._sv_date(None) == "–"
    assert w._sv_pct(0.523, 1) == f"52,3{NB}%" and w._sv_pct(0.5) == f"50{NB}%" and w._sv_pct(None) == "saknas"
    assert w._sv_int(1012) == f"1{NB}012" and w._sv_int(None) == "saknas"
    assert w._tick_indices(60) == [11, 23, 35, 47, 59] and w._tick_indices(3) == [0, 1, 2]
    assert w._tick_indices(0) == [] and w._tick_indices(1) == [0]


# ── diagrammet ───────────────────────────────────────────────────────────────

def test_figure_has_one_bar_per_day_in_regime_colours_with_breadth_heights():
    hist = _hist() + [{"date": "2026-11-02", "regime": "RÖD", "breadth": 0.523, "qualifying": 250},
                      {"date": "2026-11-03", "regime": "okänd", "qualifying": 240}]
    fig = w.history_figure(hist)
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 1
    b = bars[0]
    assert list(b.x) == [r["date"] for r in w.clean_history(hist)] and len(b.x) == 22
    colors = list(b.marker.color)
    assert colors[:6] == [w.GREEN] * 6 and colors[6:20] == [w.AMBER] * 14
    assert colors[20] == w.RED and colors[21] == w.GREY
    assert b.y[0] == 52.0 and b.y[19] == 38.0 and b.y[20] == 52.3 and b.y[21] is None
    assert b.hovertext[20] == f"2 nov 2026<br>Regim: RÖD<br>Bredd: 52,3{NB}%<br>Kvalande bolag: 250"
    assert "Bredd: saknas" in b.hovertext[21] and "Regim: OKÄND" in b.hovertext[21]
    assert b.hoverinfo == "text"


def test_figure_threshold_lines_axis_and_styling():
    fig = w.history_figure(_hist())
    lines = sorted((s.y0, s.y1, s.line.dash) for s in fig.layout.shapes)
    assert lines == [(45, 45, "dash"), (55, 55, "dash")]
    assert sorted(_plain(a.text) for a in fig.layout.annotations) == ["dör 45 %", "friskt 55 %"]
    lo, hi = fig.layout.yaxis.range
    assert lo == 0 and 60 <= hi <= 100                                    # nollbaslinje, båda linjerna syns
    assert fig.layout.xaxis.type == "category"
    assert list(fig.layout.xaxis.ticktext)[-1] == "30 okt" and len(fig.layout.xaxis.tickvals) <= 5
    assert fig.layout.height == 220 and fig.layout.plot_bgcolor == "#0D1117"
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)" and fig.layout.yaxis.gridcolor == "rgba(138,133,120,0.08)"
    assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange
    tall = w.history_figure([{"date": "2026-10-01", "regime": "GRÖN", "breadth": 0.97}])
    assert tall.layout.yaxis.range[1] == 100


def test_figure_handles_unsorted_duplicate_and_empty_input():
    fig = w.history_figure([{"date": "2026-10-02", "regime": "GUL", "breadth": 0.4},
                            {"date": "2026-10-01", "regime": "GRÖN", "breadth": 0.5},
                            {"date": "2026-10-02", "regime": "RÖD", "breadth": 0.3}])
    assert list(fig.data[0].x) == ["2026-10-01", "2026-10-02"] and list(fig.data[0].y) == [50.0, 30.0]
    assert list(fig.data[0].marker.color) == [w.GREEN, w.RED]
    assert len(w.history_figure([]).data[0].x or []) == 0


# ── sidan (AppTest) ──────────────────────────────────────────────────────────

def _page_script():
    import wolf_regime_ui
    wolf_regime_ui.render_wolf_regime_page()


def _history_script():
    import streamlit as st

    import wolf_regime_ui
    wolf_regime_ui._history(st.session_state.get("_hist"))


@pytest.fixture
def page_data(monkeypatch):
    data = {"regime": "GUL", "generated": "2026-10-30T18:00", "rules": ["Halv storlek."], "index": None,
            "breadth": 0.38, "qualifying": 269, "universe": 700, "history": _hist()}
    monkeypatch.setattr(w, "_get_data", lambda force=False: data)
    return data


def _text(at):
    return _plain(" ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption))


def test_page_shows_overview_chart_and_periods(page_data):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_page_script)
    at.run(timeout=60)
    assert not at.exception, at.exception
    at.run(timeout=60)                                                    # omkörning: samma nyckel, inget DuplicateWidgetID
    assert not at.exception, at.exception
    text = _text(at)
    assert "Regim-historik — senaste 20 handelsdagarna" in text
    assert "Nu: <b>GUL</b> sedan 13 okt (14 handelsdagar)" in text
    assert "Senaste 20 dagar:" in text and "GRÖN 6" in text and "GUL 14" in text and "RÖD 0" in text
    assert "Bredd 52 % → 38 %" in text and "Kvalande 412 → 269" in text
    assert "Perioder:" in text and "GRÖN 5–12 okt (6 d)" in text and "GUL 13 okt–nu (14 d)" in text
    assert ("Stapelns höjd = bredd (andel över MA200) · färg = regim den dagen · streckat: 55 % friskt, "
            "45 % bredden dör. GRÖN med fallande staplar är hur toppar ser ut.") in text
    charts = at.get("plotly_chart")
    assert len(charts) == 1
    spec = json.loads(charts[0].proto.spec)
    assert len(spec["data"]) == 1 and spec["data"][0]["type"] == "bar" and len(spec["data"][0]["x"]) == 20
    assert {a["text"].replace(NB, " ") for a in spec["layout"]["annotations"]} == {"friskt 55 %", "dör 45 %"}
    assert json.loads(charts[0].proto.config).get("displayModeBar") is False
    assert w.CHART_KEY in charts[0].proto.id
    assert "fyllnad" not in text                                          # gamla rutnätets text är borta


def test_history_falls_back_to_bright_box_grid_without_plotly(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(w, "_plotly_go", lambda: None)
    at = AppTest.from_function(_history_script)
    at.session_state["_hist"] = _hist()
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert not at.get("plotly_chart")
    text = _text(at)
    assert "Nu: <b>GUL</b> sedan 13 okt (14 handelsdagar)" in text and "GUL 13 okt–nu (14 d)" in text
    assert f"background:{w.GREEN};" in text and f"background:{w.AMBER};" in text
    assert "#12351f" not in text and "#3a2a10" not in text                # de mörka tonerna är borta
    assert "Stapelns höjd = bredd" in text


@pytest.mark.parametrize("hist", [[], None, [{"regime": "GRÖN"}, "skräp", {"date": "fel"}]])
def test_empty_or_unusable_history_renders_nothing(hist):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_history_script)
    at.session_state["_hist"] = hist
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert "Regim-historik" not in _text(at) and not at.get("plotly_chart") and not at.caption
