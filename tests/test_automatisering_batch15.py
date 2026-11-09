"""
Insider steg 2 (automatisering batch 15): klustret räknas om i nattjobbet (sheets_refresh) — bara förslag.

  Fönstret     upptäckt − 30 dagar till i dag (ägarens beslut); utan upptäcktsdatum skanningens rullande 30 dagar
  Kluster      insider_scan.build_cluster per arkrad ur get_insider_transactions_batch (ett anrop per 50 id)
  Grind        insider_scan.auto_gate på snapshoten som redan hämtats — bara 'Nej' sparas och föreslås, aldrig 'Ja'
  Trigger      insider_scan.auto_trigger på längre kurshistorik i samma kursanrop — bara A eller B, aldrig en
               nedgradering (en engångshändelse)
  Efter fall   kursen vid klustrets första köp mot 52-veckorstoppen före köpet, och F-score
  Sidan        rader som skanningen inte har får nattjobbets förslag med samma offer() som batch 10; att öppna
               sidan skriver ingenting
"""
import copy
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import date, timedelta

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import insider as ins  # noqa: E402
import insider_scan as isc  # noqa: E402
import refresh_ui as rui  # noqa: E402
import sheets_refresh as sr  # noqa: E402
import storage  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = date(2026, 11, 9)


def _tx(owner, pos, shares, price, day, eq=False):
    return {"ownerName": owner, "ownerPosition": pos, "shares": shares, "price": price, "amount": shares * price,
            "equityProgram": eq, "transactionDate": f"{day}T00:00:00"}


def _ago(days) -> str:
    return (TODAY - timedelta(days=days)).isoformat()


# ── Fönstret och klustret ────────────────────────────────────────────────────
def test_the_window_runs_from_found_minus_30_days_to_today():
    values = [_tx("Anna", "cfo", 1000, 50.0, _ago(40)),          # utanför skanningens 30 dagar
              _tx("Bertil", "board member", 2000, 45.0, _ago(5)),
              _tx("Carl", "ceo", 500, 60.0, _ago(80)),           # före upptäckt − 30
              _tx("Doris", "ceo", 900, 40.0, _ago(10), eq=True)]  # tilldelningsprogram räknas inte
    row = {"found": _ago(20)}                                     # fönstret: 50 dagar sedan till i dag
    assert sr.insider_window_start(row) == TODAY - timedelta(days=50)
    cl = sr.insider_cluster(values, row, "SEK", TODAY)
    assert cl["insiders"] == 2 and cl["role"] == ins.ROLE_TOP and cl["amount"] == 140.0
    assert cl["first_buy"] == _ago(40) and cl["last_buy"] == _ago(5)
    # skanningens eget fönster (utan since) är oförändrat: bara köpet för 5 dagar sedan
    plain = isc.build_cluster(values, TODAY)
    assert plain["insiders"] == 1 and plain["first_buy"] == _ago(5)
    # utan upptäcktsdatum: skanningens rullande 30 dagar
    assert sr.insider_window_start({"found": ""}) is None
    assert sr.insider_cluster(values, {}, "SEK", TODAY)["insiders"] == 1
    assert sr.insider_cluster([], row, "SEK", TODAY) is None


def test_buys_in_an_old_window_still_count_for_recurring_buyers():
    """En rad upptäckt för 14 månader sedan: köpen i fönstret är äldre än skanningens 12 månader men räknas."""
    values = [_tx("Anna", "cfo", 100, 10.0, _ago(450)), _tx("Anna", "cfo", 100, 10.0, _ago(440))]
    cl = sr.insider_cluster(values, {"found": _ago(430)}, "SEK", TODAY)
    assert cl["insiders"] == 1 and cl["aterkommande"] is True
    assert isc.build_cluster(values, TODAY) is None                       # skanningen ser dem inte


def test_unknown_currency_leaves_the_amount_in_tkr_empty():
    assert sr.insider_fx(None) == ("SEK", 1.0) and sr.insider_fx("nok") == ("NOK", 0.95)
    assert sr.insider_fx("GBX") == ("GBX", None)
    cl = sr.insider_cluster([_tx("A", "ceo", 1000, 100.0, _ago(1))], {"found": _ago(1)}, "GBX", TODAY)
    f = sr.insider_fields(cl, {"found": _ago(1)}, {"market_cap": 900.0, "f_score": 6}, [], "GBX", TODAY)
    assert f["cluster_amount_local"] == 100.0 and f["cluster_amount_tkr"] is None and f["cluster_fx_to_sek"] is None
    mcap = next(c for c in f["gate_checks"] if c["label"].startswith("Börsvärde"))
    assert mcap["status"] == "unknown" and "GBX saknas i växelkurstabellen" in mcap["detail"]
    cl_nok = sr.insider_cluster([_tx("A", "ceo", 1000, 100.0, _ago(1))], {"found": _ago(1)}, "NOK", TODAY)
    nok = sr.insider_fields(cl_nok, {"found": _ago(1)}, {}, [], "NOK", TODAY)
    assert nok["cluster_amount_tkr"] == 95.0 and nok["cluster_currency"] == "NOK" and nok["cluster_fx_to_sek"] == 0.95
    assert "insider_scan.FX_TO_SEK" in nok["cluster_fx_table"]


# ── Grind, trigger och efter fall ────────────────────────────────────────────
def _snap(**over):
    base = {"market_cap": 900.0, "f_score": 6, "net_debt_m": 50.0, "net_debt_ebitda": 1.2, "fcf_m": 30.0,
            "revenue_growth": 0.05}
    base.update(over)
    return base


def test_the_gate_is_only_ever_stored_as_nej():
    ok = sr.insider_fields(None, {}, _snap(), [], "SEK", TODAY)
    assert isc.auto_gate(_snap())[0] == ins.GATE_YES
    assert ok["gate_auto"] == "" and [c["status"] for c in ok["gate_checks"]] == ["ok"] * 5   # 'Ja' sparas aldrig
    fail = sr.insider_fields(None, {}, _snap(f_score=3), [], "SEK", TODAY)
    assert fail["gate_auto"] == ins.GATE_NO
    assert [c["label"] for c in fail["gate_checks"]] == list(ins.GATE_CRITERIA)
    assert sr.insider_fields(None, {}, _snap(fcf_m=-5.0), [], "SEK", TODAY)["gate_auto"] == ""
    assert "gate_auto" not in sr.insider_fields(None, {}, {"ins_id": 1}, [], "SEK", TODAY)   # inga tal: ingen grind


def _bars(closes, end=TODAY):
    n = len(closes)
    return [{"d": (end - timedelta(days=n - 1 - i)).isoformat(), "c": c} for i, c in enumerate(closes)]


def test_efter_fall_is_measured_at_the_first_buy():
    """300 dagar: 100 i 250 dagar, sedan ned till 70 vid första köpet (dag 270) och upp till 90 i dag. Vid köpet
    30 % under 52-veckorstoppen → efter fall; i dag bara 10 % (det gamla måttet hade sagt nej)."""
    closes = [100.0] * 250 + [100.0 - i * 1.5 for i in range(1, 21)] + [70.0 + i * (20.0 / 29) for i in range(1, 30)]
    closes[-1] = 90.0
    bars = _bars(closes)
    first = bars[269]["d"]
    assert bars[269]["c"] == 70.0
    cl = {"insiders": 2, "role": ins.ROLE_TOP, "amount": 800.0, "amount_local": 800.0, "cluster_avg": 71.0,
          "aterkommande": False, "first_buy": first, "last_buy": bars[280]["d"], "buys": []}
    f = sr.insider_fields(cl, {"found": first}, _snap(), bars, "SEK", TODAY)
    assert f["drawdown"] == 30.0 and f["drawdown_ref"] == first and f["efter_fall_auto"] is True
    assert f["efter_fall_note"].startswith(f"vid första köpet {first}: 30 % under 52v-topp")
    assert isc.drawdown_52w([b["c"] for b in bars]) == 10.0                  # dagens fall räknas inte
    unknown = sr.insider_fields(cl, {"found": first}, _snap(f_score=None), bars, "SEK", TODAY)
    assert unknown["efter_fall_auto"] is False and "kryssa själv" in unknown["efter_fall_note"]
    short = sr.insider_fields(cl, {"found": first}, _snap(), bars[200:], "SEK", TODAY)
    assert "toppen på 70 handelsdagar" in short["efter_fall_note"]
    late = sr.insider_fields(cl, {"found": first}, _snap(), bars[275:], "SEK", TODAY)   # kurserna når inte köpet
    assert late["drawdown"] is None and late["efter_fall_auto"] is False


def test_the_trigger_uses_the_rows_own_cluster_average():
    rising = _bars([100.0] * 10 + [100.0 + i for i in range(20)])
    f = sr.insider_fields(None, {"cluster_avg": 90.0}, {}, rising, "SEK", TODAY)
    assert f["trigger_auto"] == "A" and "MA20" in f["trigger_note"]
    b = _bars([90.0] * 10 + [130.0] * 15 + [100.0, 101.0, 102.0, 103.0, 104.0])
    assert sr.insider_fields(None, {"cluster_avg": 90.0}, {}, b, "SEK", TODAY)["trigger_auto"] == "B"
    assert sr.insider_fields(None, {"cluster_avg": 120.0}, {}, b, "SEK", TODAY)["trigger_auto"] == "Nej"
    # utan radens snitt: nattens klustersnitt
    cl = {"insiders": 1, "role": ins.ROLE_OTHER, "amount": 1.0, "amount_local": 1.0, "cluster_avg": 90.0,
          "aterkommande": False, "first_buy": _ago(3), "last_buy": _ago(3), "buys": []}
    assert sr.insider_fields(cl, {}, {}, b, "SEK", TODAY)["trigger_auto"] == "B"
    assert "trigger_auto" not in sr.insider_fields(None, {}, {}, [], "SEK", TODAY)


def test_the_price_history_covers_52_weeks_before_the_first_buy():
    assert sr.insider_bars_needed(None, TODAY) == sr.INSIDER_BARS_MIN == 260
    assert sr.insider_bars_needed(_ago(0), TODAY) == 262
    assert sr.insider_bars_needed(_ago(365), TODAY) == 252 + 365 * 5 // 7 + 10
    assert sr.insider_bars_needed("1990-01-01", TODAY) == sr.MAX_BARS


# ── Hela nattjobbet mot ett fejkat API ───────────────────────────────────────
class _API:
    def __init__(self):
        self.calls = []
        self.max_counts = {}
        self.tx = {1: [_tx("Anna", "cfo", 10000, 50.0, _ago(4)), _tx("Bertil", "board member", 8000, 48.0, _ago(2))]}
        self.snaps = {1: _snap(f_score=3), 40: {"ev_ebitda": 5.0}}

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if callable(attr) and not name.startswith("_"):
            object.__getattribute__(self, "calls").append(name)
        return attr

    def get_instruments(self):
        return [{"insId": 1, "ticker": "EKTA B", "name": "Elekta B", "stockPriceCurrency": "SEK"},
                {"insId": 40, "ticker": "BOL", "name": "Boliden", "stockPriceCurrency": "SEK"}]

    def get_global_instruments_list(self):
        return []

    def resolve_instrument_id(self, form):
        return {"EKTA B": 1, "BOL": 40}.get(form)

    def get_fundamentals_snapshot_fast(self, ids, scope="nordic"):
        return {i: dict(self.snaps.get(i, {}), ins_id=i) for i in ids}

    def get_stockprices(self, ins_id, max_count=5):
        self.max_counts[ins_id] = max_count
        return _bars([100.0] * 10 + [100.0 + i for i in range(20)])[-max_count:]

    def get_reports(self, ins_id, kind, max_count=7):
        return []

    def get_insider_transactions_batch(self, ids):
        self.batch_ids = list(ids)
        return {i: self.tx[i] for i in ids if i in self.tx}


def _sheets():
    return {"insider": {"signals": [{"id": "m1", "ticker": "EKTA-B.ST", "found": _ago(3), "cluster_avg": 49.0},
                                    {"id": "m2", "ticker": "EKTA B", "found": ""}]},
            "producers": {"producers": [{"id": "r1", "ticker": "BOL.ST"}]}}


def test_the_nightly_job_adds_the_cluster_with_one_batch_call():
    api = _API()
    out = sr.refresh(api, _sheets(), today=TODAY)
    assert out["error"] is None
    c = Counter(api.calls)
    assert c["get_insider_transactions_batch"] == 1 and api.batch_ids == [1]          # bara insider-radernas bolag
    assert c["get_stockprices"] == 2 and c["get_fundamentals_snapshot_fast"] == 1     # en kurs per bolag, som förut
    assert api.max_counts == {1: 252 + 4 * 5 // 7 + 10, 40: 5}                       # längre historik, samma anrop
    m1 = out["rows"]["insider:m1"]
    assert m1["cluster_insiders"] == 2 and m1["cluster_role"] == ins.ROLE_TOP
    assert m1["cluster_amount_tkr"] == 884.0 and m1["cluster_amount_local"] == 884.0 and m1["cluster_currency"] == "SEK"
    assert m1["first_buy"] == _ago(4) and m1["last_buy"] == _ago(2) and len(m1["cluster_buys"]) == 2
    assert m1["cluster_since"] == _ago(33) and m1["cluster_until"] == TODAY.isoformat()
    assert m1["gate_auto"] == ins.GATE_NO and m1["trigger_auto"] == "A" and m1["f_score"] == 3.0
    assert m1["price"] == 119.0                                                        # kursen som förut: sista stängningen
    assert out["rows"]["insider:m2"]["cluster_since"] == _ago(30)                      # utan upptäckt: 30 dagar
    r1 = out["rows"]["producers:r1"]
    assert not any(k.startswith(("cluster_", "gate_", "trigger_")) for k in r1)


def test_the_job_goes_on_when_the_insider_data_fails():
    class _Broken(_API):
        def get_insider_transactions_batch(self, ids):
            raise RuntimeError("503")
    api = _Broken()
    out = sr.refresh(api, _sheets(), today=TODAY)
    assert out["error"] is None
    m1 = out["rows"]["insider:m1"]
    assert "cluster_insiders" not in m1 and "cluster_since" not in m1                # inget fönster att visa
    assert m1["gate_auto"] == ins.GATE_NO and m1["trigger_auto"] == "A"              # grind och trigger ändå
    assert api.max_counts[1] == sr.INSIDER_BARS_MIN


def test_the_batch_call_takes_50_ids_per_request(tmp_path):
    from borsdata_api import BorsdataAPI
    api = BorsdataAPI(api_key="test", cache_dir=str(tmp_path))
    seen = []
    api._get = lambda path, params=None, use_cache=True: seen.append(params["instList"]) or {"list": []}
    api.get_insider_transactions_batch(list(range(1, 121)))
    assert len(seen) == 3 and [len(s.split(",")) for s in seen] == [50, 50, 20]


def test_insider_scan_does_not_reconfigure_the_jobs_logging():
    """sheets_refresh sätter rotloggern först; insider_scans basicConfig gör då ingenting — och sheets_refresh
    importerar inte insider_scan förrän insider-raderna räknas."""
    code = ("import logging, sys; import sheets_refresh as sr; assert 'insider_scan' not in sys.modules; "
            "h = list(logging.root.handlers); lvl = logging.root.level; fmt = h[0].formatter._fmt; sr._isc(); "
            "assert list(logging.root.handlers) == h and logging.root.level == lvl and h[0].formatter._fmt == fmt; "
            "print('ok')")
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0 and res.stdout.strip() == "ok", res.stderr


# ── Förslagen på sidan (rena funktioner) ─────────────────────────────────────
def _night(**over) -> dict:
    """Nattjobbets rad för en handskriven rad: 3 köpare, VD/CFO, 1 235 tkr (1 300 tNOK), grind Nej, trigger A."""
    base = {"price": 50.0, "asof": "2026-11-08", "currency": "NOK", "f_score": 3.0,
            "cluster_since": "2026-10-07", "cluster_until": "2026-11-09", "cluster_insiders": 3,
            "cluster_role": ins.ROLE_TOP, "cluster_amount_tkr": 1235.0, "cluster_amount_local": 1300.0,
            "cluster_currency": "NOK", "cluster_fx_to_sek": 0.95, "cluster_avg": 48.2, "cluster_aterkommande": True,
            "first_buy": "2026-10-20", "last_buy": "2026-11-05",
            "cluster_buys": [{"date": "2026-11-05", "owner": "Anna", "position": "cfo", "shares": 1000, "price": 48.2}],
            "gate_auto": "Nej",
            "gate_checks": [{"label": "Börsvärde > 300 MSEK", "status": "ok", "detail": "1 200 MSEK"},
                            {"label": "F-score ≥ 5", "status": "fail", "detail": "F-score 3"}],
            "trigger_auto": "A", "trigger_note": "stängning 50.00 > MA20 48.00, MA20 planat/vänt",
            "drawdown": 34.0, "drawdown_ref": "2026-10-20", "efter_fall_auto": True,
            "efter_fall_note": "vid första köpet 2026-10-20: 34 % under 52v-topp, F-score 6"}
    base.update(over)
    return base


def _row(**over) -> dict:
    base = {"id": "m1", "ticker": "NOD", "name": "Nordic", "found": "2026-11-06", "role": ins.ROLE_OTHER,
            "gate": ins.GATE_BLANK, "trigger": ""}
    base.update(over)
    return base


def test_a_hand_typed_row_gets_every_nightly_field():
    assert ins.nightly_offers(_row(), _night()) == {
        "insiders": 3, "role": ins.ROLE_TOP, "amount": 1235.0, "aterkommande": True, "cluster_avg": 48.2,
        "efter_fall": True, "gate": ins.GATE_NO, "trigger": "A"}
    assert ins.amount_note(ins.nightly_cluster(_night())) == "1 300 tNOK × kurs 0,95 (grov)"
    assert ins.nightly_row({"rows": {"insider:m1": _night()}}, _row()) == _night()
    assert ins.nightly_row(None, _row()) == {} and ins.nightly_cluster({}) is None


def test_the_gate_is_only_offered_as_nej():
    assert ins.gate_offer(_row(), _night(gate_auto="Ja")) is None                 # 'Ja' föreslås aldrig
    assert ins.gate_offer(_row(), _night(gate_auto="")) is None
    assert ins.gate_offer(_row(gate=ins.GATE_NO), _night()) is None               # redan Nej
    assert ins.gate_offer(_row(gate=ins.GATE_YES), _night()) == ins.GATE_NO      # något mätbart faller nu
    assert "gate" not in ins.nightly_offers(_row(), _night(gate_auto="Ja"))


def test_the_trigger_is_offered_once_and_never_downgraded():
    assert ins.trigger_offer(_row(trigger="Nej"), _night()) == "A"
    assert ins.trigger_offer(_row(), _night(trigger_auto="B")) == "B"
    assert ins.trigger_offer(_row(trigger="A"), _night(trigger_auto="B")) is None
    assert ins.trigger_offer(_row(trigger="B"), _night(trigger_auto="A")) is None
    assert ins.trigger_offer(_row(trigger="C"), _night()) is None
    assert ins.trigger_offer(_row(), _night(trigger_auto="Nej")) is None          # Nej föreslås aldrig
    assert ins.trigger_offer(_row(), _night(trigger_auto="")) is None


def test_the_cluster_rules_are_the_scans():
    """Aldrig en nedgradering, inget efter ett nytt kluster, och tomt belopp när valutan är okänd."""
    full = _row(insiders=4, role=ins.ROLE_TOP, amount=2000.0, aterkommande=True, efter_fall=True, cluster_avg=48.2,
                gate=ins.GATE_NO, trigger="A")
    assert ins.nightly_offers(full, _night()) == {}
    new = _row(found="2026-10-01")                                                # köpen började efter Upptäckt
    assert ins.nightly_offers(new, _night()) == {"gate": ins.GATE_NO, "trigger": "A"}
    assert "amount" not in ins.nightly_offers(_row(), _night(cluster_amount_tkr=None))
    assert "efter_fall" not in ins.nightly_offers(_row(), _night(efter_fall_auto=False))


# ── Insider-sidan (AppTest) ──────────────────────────────────────────────────
SCAN = {"generated": "2026-11-09T07:01:00+00:00", "clusters": [
    {"ticker": "VOLV-B.ST", "name": "Volvo B", "ins_id": 123, "currency": "SEK", "found": "2026-11-09",
     "insiders": 3, "role": ins.ROLE_TOP, "amount": 1235.0, "efter_fall": False, "aterkommande": False,
     "cluster_avg": 48.2, "gate": "", "trigger": "", "score": 7, "status": ins.S_RUN_GATE,
     "auto": {"gate_checks": [], "first_buy": "2026-10-20", "last_buy": "2026-11-05", "buys": []}}]}
STORES = {"insider": {"signals": [_row(), _row(id="v1", ticker="VOLV B", name="Volvo")]}}
REFRESH = {"generated": "2026-11-09T05:10:00+00:00",
           "rows": {"insider:m1": _night(), "insider:v1": _night(gate_auto="Nej", trigger_auto="B")}}


def _same_json(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@pytest.fixture
def page(monkeypatch):
    """Insider-sidan ur fejkad lagring, fejkad nattblob och fejkad skanning — inget GitHub, ingen Gist."""
    state = {"stores": copy.deepcopy(STORES), "refresh": REFRESH, "scan": SCAN}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(state["stores"].get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    import gist_storage
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
    monkeypatch.setattr(rui, "load_refresh", lambda: state["refresh"])
    monkeypatch.setattr(ins, "_load_auto_scan", lambda: copy.deepcopy(state["scan"]))

    def run(stores=None, refresh=None):
        for k, v in (("stores", stores), ("refresh", refresh)):
            if v is not None:
                state[k] = copy.deepcopy(v)
        from streamlit.testing.v1 import AppTest

        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import insider as i
            i.render_insider_page()
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        return at
    return run


def _texts(at) -> str:
    return " ".join(getattr(e, "value", "") or "" for coll in (at.markdown, at.caption) for e in coll)


def test_a_row_the_scan_does_not_have_gets_the_nightly_suggestions(page):
    at = page()
    txt = _texts(at)
    keys = {b.key for b in at.button}
    for k in ins.SCAN_FIELDS + ("gate", "trigger"):
        assert f"rf_offer_ins_night_m1_{k}" in keys, k
    assert ("🌙 Nattjobbet 2026-11-09: 3 köpare, köp 2026-10-20 – 2026-11-05 i fönstret 2026-10-07 – 2026-11-09 "
            "(upptäckt − 30 dagar till i dag)") in txt
    assert "🤖 Nattjobbet 2026-11-09: Belopp 1 235 tkr — 1 300 tNOK × kurs 0,95 (grov)" in txt
    assert "Kvalitetsgrind Nej — något mätbart faller" in txt
    assert "Teknisk trigger A — stängning 50.00 &gt; MA20 48.00" in txt
    assert "Efter fall &gt; 20 % ja — vid första köpet 2026-10-20: 34 % under 52v-topp" in txt
    assert "❌ F-score ≥ 5 — F-score 3" in txt and "✅ Börsvärde &gt; 300 MSEK — 1 200 MSEK" in txt
    assert "Bara tilldelningsprogram räknas bort — gåvor och interna överföringar kan räknas med." in txt
    assert "2026-11-05 Anna (cfo) 1,000 st @ 48.2" in txt
    assert "Grinden mäts med närmevärden" in txt and "'Ja' väljer du själv" in txt
    # raden som skanningen har får skanningens förslag, inte nattjobbets (batch 10 som förut)
    assert not any(k.startswith("rf_offer_ins_night_v1_") for k in keys)
    assert "rf_offer_ins_scan_v1_insiders" in keys
    assert _same_json(at.session_state["insider"], STORES["insider"])              # att öppna sidan skriver ingenting
    at.run()
    assert _same_json(at.session_state["insider"], STORES["insider"])


def test_anvand_writes_gate_trigger_and_numbers_only_on_click(page):
    at = page()
    at.button(key="rf_offer_ins_night_m1_gate").click().run()
    sig = at.session_state["insider"]["signals"][0]
    assert sig["gate"] == ins.GATE_NO and at.selectbox(key="ins_g_m1").value == ins.GATE_NO
    at.button(key="rf_offer_ins_night_m1_trigger").click().run()
    assert at.session_state["insider"]["signals"][0]["trigger"] == "A"
    assert at.selectbox(key="ins_t_m1").value == "A"
    at.button(key="rf_offer_ins_night_m1_insiders").click().run()
    sig = at.session_state["insider"]["signals"][0]
    assert sig["insiders"] == 3 and isinstance(sig["insiders"], int) and at.number_input(key="ins_n_m1").value == 3
    at.button(key="rf_offer_ins_night_m1_efter_fall").click().run()
    assert at.session_state["insider"]["signals"][0]["efter_fall"] is True
    assert at.checkbox(key="ins_b_m1_efter_fall").value is True
    assert not at.exception, at.exception
    keys = {b.key for b in at.button}
    assert not {"rf_offer_ins_night_m1_gate", "rf_offer_ins_night_m1_trigger", "rf_offer_ins_night_m1_insiders"} & keys
    assert at.session_state["insider"]["signals"][1] == STORES["insider"]["signals"][1]   # den andra raden orörd


def test_ja_and_a_downgrade_never_get_a_button(page):
    rows = {"generated": "2026-11-09T05:10:00+00:00",
            "rows": {"insider:m1": _night(gate_auto="Ja", trigger_auto="B")}}
    stores = {"insider": {"signals": [_row(trigger="A")]}}
    at = page(stores=stores, refresh=rows)
    keys = {b.key for b in at.button}
    assert "rf_offer_ins_night_m1_gate" not in keys and "rf_offer_ins_night_m1_trigger" not in keys
    txt = _texts(at)
    assert "Kvalitetsgrind Ja" not in txt and "Trigger i natt" not in txt
    assert _same_json(at.session_state["insider"], stores["insider"])


def test_no_cluster_says_so_and_the_trigger_note_is_text(page):
    base = _night(trigger_auto="Nej", trigger_note="stängning 40.00 vs MA20 45.00, 1 mån -3.0 %", gate_auto="")
    night = {k: v for k, v in base.items()
             if not k.startswith(("cluster_", "first_", "last_", "drawdown", "efter_fall"))}
    night.update(cluster_since="2026-10-07", cluster_until="2026-11-09")
    at = page(refresh={"generated": "2026-11-09T05:10:00+00:00", "rows": {"insider:m1": night}})
    txt = _texts(at)
    keys = {b.key for b in at.button}
    assert not any(k.startswith("rf_offer_ins_night_m1_") for k in keys)
    assert "🌙 Nattjobbet 2026-11-09: inga riktiga köp i Börsdatas insynsregister 2026-10-07 – 2026-11-09" in txt
    assert "🌙 Trigger i natt: Nej — stängning 40.00 vs MA20 45.00, 1 mån -3.0 %. Bara A eller B föreslås." in txt
    assert _same_json(at.session_state["insider"], STORES["insider"])
