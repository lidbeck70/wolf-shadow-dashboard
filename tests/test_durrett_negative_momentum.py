"""
Kurs mot MA200 och kursutveckling 6 mån får vara negativa (ägarens beslut 2026-11-25).

En aktie i nedtrend har negativt momentum; Durretts momentumtabell har redan stegen −10 % → 35 och −25 % → 20.
Förr kastade kontrollen bort varje negativt värde, så aktien visades som "saknas" i stället för att få låg poäng.
Nu godtas de i valideringen, i arkets ruta (ingen nedre gräns), i ChatGPT-importen och i motorn — med samma tabell,
gränser och vikter som förut. Andra fält som inte får vara negativa är oförändrade.

Påhittade bolag (EXA.TO), ingen riktig data, inget nätverk.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
from confidence.data import validation as cval  # noqa: E402
from confidence.data.provenance import dp  # noqa: E402

FIELDS = ("price_vs_ma200_pct", "momentum_6m_pct")


def _company(**vals):
    c = dcs.gold_producer()
    c.ticker, c.name = "EXA.TO", "Exempel Guld"
    for k, v in vals.items():
        c.set(k, dp(v, kind="ACTUAL", source="exempel", source_type="secondary", pub_date="2026-11-01", unit="%"))
    return c


@pytest.mark.parametrize("key", FIELDS)
@pytest.mark.parametrize("v", [-5.0, -20.0, -60.0])
def test_validation_accepts_a_negative_momentum_and_scoring_may_read_it(key, v):
    c = _company(**{key: v})
    errors = [i for i in cval.validate(c) if i.field == key and i.level == cval.ERROR]
    assert errors == []
    assert cval.usable(c, key)


def test_other_fields_still_may_not_be_negative():
    assert "aisc" not in cval._MAY_BE_NEGATIVE and "cash_musd" not in cval._MAY_BE_NEGATIVE
    c = _company()
    c.set("cash_musd", dp(-1.0, kind="ACTUAL", source="exempel", source_type="primary", pub_date="2026-11-01"))
    assert not cval.usable(c, "cash_musd")
    assert any(i.field == "cash_musd" and i.level == cval.ERROR for i in cval.validate(c))


def test_the_sheet_field_has_no_lower_bound_for_momentum_only():
    from review import catalog as cat
    fields = {f.path: f for f in cat.fields_for("confidence", _company().as_dict())}
    for key in FIELDS:
        assert fields[f"fields.{key}"].lo is None
    assert fields["fields.aisc"].lo == 0.0


def test_the_chatgpt_import_accepts_a_negative_momentum():
    from confidence import store as cs
    from review import answers as an
    from review import export as ex
    from review import importer as im
    data = cs.default()
    cs.put(data, _company())
    exp = ex.build("EXA.TO", {"confidence": data}, ["confidence"])
    answers = an.read_text("\n".join([ex.ANSWER_HEADER,
                                      "EXA.TO;confidence:fields.price_vs_ma200_pct;-5;%;Kursdata;secondary;"
                                      "2026-11-20;high;",
                                      "EXA.TO;confidence:fields.momentum_6m_pct;-20;%;Kursdata;secondary;"
                                      "2026-11-20;high;"]))
    res = im.check(answers, exp)
    assert res.issues == []
    got = {c.field.path: c.new for c in res.changes}
    assert float(got["fields.price_vs_ma200_pct"]) == -5.0
    assert float(got["fields.momentum_6m_pct"]) == -20.0


@pytest.mark.parametrize("v, pts", [(-5.0, 35.0), (-10.0, 35.0), (-20.0, 20.0), (-25.0, 20.0), (-40.0, 10.0),
                                    (0.0, 50.0), (12.0, 65.0)])
def test_durrett_momentum_uses_the_existing_steps_for_negative_values(v, pts):
    """Tabellen är oförändrad: ≥0 → 50, ≥−10 → 35, ≥−25 → 20, under → score_beyond 10."""
    from engines.durrett import momentum
    from engines.durrett._base import Ctx
    c = _company(price_vs_ma200_pct=v, momentum_6m_pct=v)                  # volymtrend 1 → ingen justering
    s = momentum.score(Ctx(c))
    assert s.components["price_trend"][0] == pts
    assert any(f"price_trend {pts:g}: kurs {v:+.0f} % mot MA200, 6 mån {v:+.0f} %" in x
               for x in s.positive + s.negative)
