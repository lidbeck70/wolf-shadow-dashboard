"""
Nya larm (automatisering batch 17, ägarens beslut): arkens egna rena funktioner på nattens färska tal, i
sheets_refresh.build_events — inga regler, trösklar eller poäng ändras.

  rule_verdict      Rick Rule: producers.producer_verdict med dagens råvarupris mot det sparade
  rule_ev_nav       Rick Rule: producers.ev_nav_verdict med färskt EV mot det sparade
  insider_trigger   Insider: 'Väntar på teknisk trigger' → KÖP med nattens trigger A/B (insider.trigger_offer), utan att
                    jaga kursen (insider.is_chase, passa över +30 %)

Larm bara när bedömningen byter, och för Rick Rule bara när det sparade talet ligger inom ±50 % av det färska (annars
står de nog i olika enheter). Inget larm när data saknas. Nycklarna är stabila, så alert_rules.sheet_alerts larmar en
gång per övergång, och 'rule_'/'insider_'-prefixet leder larmet till rätt ark. Inget säljklusterlarm (brus).

Fix: Rick Rules två larm har tre lägen var, så nyckeln bär den nya bedömningen — ett andra byte larmar också.
Insider-triggern larmar inte när nattens grind säger Nej (kortet föreslår Nej samma natt), och texten lovar ingen
Använd: för rader som skanningen har visar kortet ingen.
"""
import logging
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alert_rules as ar  # noqa: E402
import insider as ins  # noqa: E402
import producers as prod  # noqa: E402
import sheets_refresh as sr  # noqa: E402
from tests import test_automatisering_batch15 as b15  # noqa: E402
from tests.test_automatisering_batch15 import page  # noqa: E402,F401  (Insider-sidan, batch 15)

TODAY = date(2026, 11, 9)


# ── Enhetsvakten ─────────────────────────────────────────────────────────────
def test_the_unit_guard_is_plus_minus_50_percent_of_the_fresh_number():
    assert sr.UNIT_GUARD_PCT == 50.0
    assert sr.unit_guard_ok(50.0, 100.0) and sr.unit_guard_ok(150.0, 100.0) and sr.unit_guard_ok(100.0, 100.0)
    assert not sr.unit_guard_ok(49.0, 100.0) and not sr.unit_guard_ok(151.0, 100.0)
    assert not sr.unit_guard_ok(9000.0, 4.5)                     # koppar i USD/t mot terminen i USD/lb
    for saved, fresh in ((None, 100.0), (100.0, None), (0.0, 100.0), (100.0, 0.0), (-5.0, 100.0), ("", 1.0)):
        assert not sr.unit_guard_ok(saved, fresh)


# ── rule_verdict: Rick Rules bedömning med dagens råvarupris ─────────────────
def _rule(**over) -> dict:
    """Guld, AISC 1 300, två disciplinkryss: sparat pris 2 000 = 35 % marginal (1 p) + 2 = 3 p, Bevaka."""
    row = {"id": "r1", "ticker": "BOL.ST", "commodity": "Guld", "price": 2000.0, "unit_cost": 1300.0,
           "jurisdiktion": True, "kapitaldisciplin": True, "insyn": False}
    row.update(over)
    return row


def _night(**over) -> dict:
    s = {"ticker": "BOL.ST", "price": 101.0, "commodity_price": 2500.0, "commodity_unit": "USD/oz",
         "commodity_asof": "2026-11-06", "commodity_ticker": "GC=F"}
    s.update(over)
    return s


def _events(sheets: dict, rows: dict) -> dict:
    return {e["key"]: e for e in sr.build_events(sheets, rows)}


def _rule_events(row: dict, night: dict) -> dict:
    return _events({"producers": {"producers": [row]}}, {"producers:r1": night})


def test_the_verdict_alarm_fires_when_todays_price_changes_the_verdict():
    row = _rule()
    assert prod.producer_verdict(prod.producer_score(row)).label == prod.P_WATCH
    ev = _rule_events(row, _night())[f"rule_verdict:producers:r1:{prod.P_BUY}"]   # 2 500: 48 % (2 p) + 2 = 4 p
    assert ev["kind"] == "rule_verdict" and ev["sheet"] == "producers" and ev["ticker"] == "BOL.ST"
    assert ev["title"] == f"🪨 Rick Rule: BOL.ST → {prod.P_BUY}"
    assert "2,500.00 USD/oz (sparat 2,000.00)" in ev["body"] and "marginal 48.0 % (var 35.0 %)" in ev["body"]
    assert "poäng 4/5 (var 3)" in ev["body"] and f"från '{prod.P_WATCH}' till '{prod.P_BUY}'" in ev["body"]
    assert "$" not in ev["title"] + ev["body"]
    assert row["price"] == 2000.0                                           # raden rörs inte
    down = _rule_events(_rule(price=2500.0), _night(commodity_price=2000.0))
    assert down[f"rule_verdict:producers:r1:{prod.P_WATCH}"]["title"].endswith(prod.P_WATCH)   # åt andra hållet också


def test_no_verdict_alarm_without_a_change_or_without_data():
    assert _rule_events(_rule(), _night(commodity_price=2100.0)) == {}       # 38 %: fortfarande 1 p, Bevaka
    assert _rule_events(_rule(), _night(commodity_price=None)) == {}         # inget nattpris
    assert _rule_events(_rule(price=None), _night()) == {}                   # inget sparat pris
    assert _rule_events(_rule(unit_cost=None), _night()) == {}               # ingen kostnad → ingen poäng
    assert _rule_events(_rule(mine_life=3.0), _night()) == {}                # döende tillgång: Passa före och efter


def test_no_verdict_alarm_when_the_units_differ():
    # sparat 1 000 mot 2 500 i natt: 60 % ifrån → troligen olika enheter, fast bedömningen hade bytt
    row = _rule(price=1000.0)
    assert prod.producer_verdict(prod.producer_score(row)).label != \
        prod.producer_verdict(prod.producer_score(dict(row, price=2500.0))).label
    assert _rule_events(row, _night()) == {}
    # koppar: kortets pris i USD/t mot terminen i USD/lb
    copper = _rule(commodity="Koppar", price=9000.0, unit_cost=2500.0)
    assert _rule_events(copper, _night(commodity_price=4.5, commodity_unit="USD/lb",
                                       commodity_ticker="HG=F")) == {}


def test_no_verdict_alarm_when_the_nights_price_is_for_another_commodity():
    assert _rule_events(_rule(), _night(commodity_ticker="SI=F")) == {}     # silverpris på ett guldkort
    assert _rule_events(_rule(commodity=""), _night()) == {}


# ── rule_ev_nav: EV/NAV med färskt EV ────────────────────────────────────────
def _nav_row(**over) -> dict:
    """EV 650 / NAV 1 000 = 0,65× — billigt. Ingen kostnad: bara EV/NAV-larmet kan gå."""
    row = {"id": "r1", "ticker": "BOL.ST", "ev_musd": 650.0, "nav_musd": 1000.0, "nav_price": 2000.0}
    row.update(over)
    return row


def test_the_ev_nav_alarm_fires_when_fresh_ev_changes_the_verdict():
    ev = _rule_events(_nav_row(), {"ev_musd": 800.0})[f"rule_ev_nav:producers:r1:{prod.NAV_FAIR}"]
    assert ev["title"] == f"🪨 Rick Rule: BOL.ST 0.80× NAV — {prod.NAV_FAIR}"
    assert "Färskt EV 800 MUSD (sparat 650) mot NAV 1,000 MUSD = 0.80× (var 0.65×)" in ev["body"]
    assert f"från '{prod.NAV_CHEAP}' till '{prod.NAV_FAIR}'" in ev["body"]
    rich = _rule_events(_nav_row(ev_musd=950.0), {"ev_musd": 1100.0})[f"rule_ev_nav:producers:r1:{prod.NAV_RICH}"]
    assert rich["title"].endswith(prod.NAV_RICH)


def test_no_ev_nav_alarm_without_a_change_without_data_or_with_other_units():
    assert _rule_events(_nav_row(), {"ev_musd": 680.0}) == {}                # 0,68×: fortfarande billigt
    assert _rule_events(_nav_row(), {"ev_musd": None}) == {}
    assert _rule_events(_nav_row(ev_musd=None), {"ev_musd": 800.0}) == {}
    assert _rule_events(_nav_row(nav_musd=None), {"ev_musd": 800.0}) == {}
    assert _rule_events(_nav_row(), {"ev_musd": -50.0}) == {}                # negativt EV är en lucka
    assert _rule_events(_nav_row(), {"ev_musd": 1400.0}) == {}               # 650 mot 1 400: annan enhet (MSEK?)


def test_the_three_rick_rule_alarms_can_come_together():
    row = _rule(ev_musd=650.0, nav_musd=1000.0, nd_ebitda=0.7)
    keys = set(_rule_events(row, _night(ev_musd=800.0, nd_ebitda=1.4)))
    assert keys == {"rule_deleveraging:producers:r1", f"rule_verdict:producers:r1:{prod.P_BUY}",
                    f"rule_ev_nav:producers:r1:{prod.NAV_FAIR}"}


# ── insider_trigger: från 'Väntar på teknisk trigger' till KÖP ────────────────
def _sig(**over) -> dict:
    """3 insiders (3 p) + VD/CFO (2 p) + 1 500 tkr (2 p) = 7 p, grinden Ja, ingen trigger: väntar på trigger."""
    row = {"id": "i1", "ticker": "EKTA-B.ST", "insiders": 3, "role": ins.ROLE_TOP, "amount": 1500.0,
           "gate": ins.GATE_YES, "trigger": "", "cluster_avg": 100.0, "price_now": 100.0}
    row.update(over)
    return row


def _ins_events(row: dict, price=110.0, trigger_auto="A", note="stängning 110.00 > MA20 104.00, MA20 planat/vänt",
                **night):
    s = {"ticker": row.get("ticker"), "price": price, "trigger_auto": trigger_auto, "trigger_note": note, **night}
    return _events({"insider": {"signals": [row]}}, {"insider:i1": s})


def test_the_trigger_alarm_fires_when_a_waiting_row_becomes_buy_without_chasing():
    assert ins.status(_sig()) == ins.S_WAIT_TRIGGER
    ev = _ins_events(_sig())["insider_trigger:insider:i1"]
    assert ev["kind"] == "insider_trigger" and ev["sheet"] == "insider" and ev["ticker"] == "EKTA-B.ST"
    assert ev["title"] == "👥 Insider: EKTA-B.ST trigger A — KÖP-läge"
    assert "Nattens trigger A: stängning 110.00 > MA20 104.00" in ev["body"]
    assert "Kurs 110.00 är +10 % mot klustersnittet 100.00 — under +30 %" in ev["body"]
    assert "Poäng 7/10, grinden Ja. Välj trigger A i raden (Teknisk trigger) och lägg stoppen 85.00 som order." \
        in ev["body"]
    assert "Använd" not in ev["body"]                            # kortet visar inte alltid en Använd för triggern
    b = _ins_events(_sig(trigger="Nej"), trigger_auto="B", note="")["insider_trigger:insider:i1"]
    assert b["title"].endswith("trigger B — KÖP-läge") and "Nattens trigger B. Kurs" in b["body"]


def test_no_trigger_alarm_when_the_price_is_chased_or_under_the_stop():
    chase = _ins_events(_sig(), price=131.0)                     # +31 %: passa-regeln, det larmet går i stället
    assert set(chase) == {"insider_chase:insider:i1"}
    assert set(_ins_events(_sig(), price=130.0)) == {"insider_trigger:insider:i1"}   # exakt +30 % är inte jakt
    stop = _ins_events(_sig(), price=84.0)                       # under stoppen: säljregel 2 går först
    assert set(stop) == {"insider_stop:insider:i1"}


def test_no_trigger_alarm_unless_the_row_waits_for_its_trigger():
    assert _ins_events(_sig(trigger="A")) == {}                  # redan KÖP
    assert _ins_events(_sig(trigger="C")) == {}
    assert _ins_events(_sig(gate=ins.GATE_BLANK)) == {}          # grinden inte körd
    assert _ins_events(_sig(gate=ins.GATE_NO)) == {}             # stoppad i grinden
    assert _ins_events(_sig(insiders=2)) == {}                   # 6 p: bevaka
    assert _ins_events(_sig(amount=None)) == {}                  # ingen poäng utan belopp


def test_no_trigger_alarm_without_a_nightly_trigger_or_without_data():
    assert _ins_events(_sig(), trigger_auto="Nej") == {}
    assert _ins_events(_sig(), trigger_auto="") == {}
    assert _ins_events(_sig(), trigger_auto=None) == {}
    assert _ins_events(_sig(), trigger_auto="C") == {}           # C sätts aldrig av natten
    assert _ins_events(_sig(), price=None) == {}
    assert _ins_events(_sig(cluster_avg=None)) == {}             # utan klustersnitt går jakten inte att pröva


def test_no_trigger_alarm_when_the_nights_gate_says_nej():
    """Ny rapport: F-score 3, så nattens grind blir 'Nej'. Kortet föreslår Nej samma natt (insider.gate_offer) — med
    båda nattens förslag står raden i grinden, inte på KÖP. Då går inget KÖP-larm med 'grinden Ja'."""
    s = {"price": 110.0, "trigger_auto": "A", "gate_auto": ins.GATE_NO}
    assert ins.gate_offer(_sig(), s) == ins.GATE_NO and ins.trigger_offer(_sig(), s) == "A"
    applied = dict(_sig(), **{k: v for k, v in ins.nightly_offers(_sig(), s).items() if k in ("gate", "trigger")})
    assert ins.status(applied) == ins.S_GATE_FAIL
    assert _ins_events(_sig(), gate_auto=ins.GATE_NO) == {}
    # samma natt utan grindsvar: larmet går som förut
    assert set(_ins_events(_sig(), gate_auto="")) == {"insider_trigger:insider:i1"}
    assert set(_ins_events(_sig(), gate_auto=None)) == {"insider_trigger:insider:i1"}
    # stopp- och passa-larmen bryr sig inte om grinden
    assert set(_ins_events(_sig(), price=84.0, gate_auto=ins.GATE_NO)) == {"insider_stop:insider:i1"}
    assert set(_ins_events(_sig(), price=131.0, gate_auto=ins.GATE_NO)) == {"insider_chase:insider:i1"}


def test_a_row_the_scan_covers_gets_the_alarm_without_a_promised_button(page):  # noqa: F811
    """En rad ur skanningen (samma ins_id, klustret i skanningens fönster) får ingen Använd för triggern på kortet —
    nattjobbets förslag visas bara för rader som skanningen inte har. Larmet går ändå (skanningens eget köplarm kräver
    skanningens grind Ja), och texten säger att du väljer triggern i raden."""
    row = b15._row(id="v1", ticker="VOLV B", name="Volvo", ins_id=123, insiders=3, role=ins.ROLE_TOP, amount=1500.0,
                   gate=ins.GATE_YES, trigger="", cluster_avg=48.2)
    night = b15._night(gate_auto="", trigger_auto="B", trigger_note="stängning 50.00 > MA20 48.00, 1 mån +4.0 %")
    assert ins.status(row) == ins.S_WAIT_TRIGGER
    assert ins.match_signal(b15.SCAN["clusters"][0], [row]) is row             # skanningen täcker raden
    ev = _events({"insider": {"signals": [row]}}, {"insider:v1": night})
    assert set(ev) == {"insider_trigger:insider:v1"}
    body = ev["insider_trigger:insider:v1"]["body"]
    assert "Använd" not in body
    assert "Välj trigger B i raden (Teknisk trigger) och lägg stoppen 40.97 som order." in body
    stores = {"insider": {"signals": [row]}}
    at = page(stores=stores, refresh={"generated": "2026-11-09T05:10:00+00:00", "rows": {"insider:v1": night}})
    keys = {b.key for b in at.button}
    assert "rf_offer_ins_night_v1_trigger" not in keys
    assert not any(k.startswith("rf_offer_ins_night_v1_") for k in keys)
    assert at.selectbox(key="ins_t_v1").value == ""                            # rutan finns — du väljer själv
    assert at.selectbox(key="ins_t_v1").label == "Teknisk trigger"
    assert "Grind och trigger ur skanningen är bara text" in b15._texts(at)
    assert b15._same_json(at.session_state["insider"], stores["insider"])     # att öppna sidan skriver ingenting


# ── Larmbenet: en gång per övergång, rätt ark ────────────────────────────────
def _blob(events: list) -> dict:
    return {"generated": "2026-11-09T06:00:00", "error": None, "events": events}


def test_the_new_alarms_fire_once_per_transition_and_open_their_sheet():
    row, sig = _rule(ev_musd=650.0, nav_musd=1000.0), _sig()
    sheets = {"producers": {"producers": [row]}, "insider": {"signals": [sig]}}
    night = {"producers:r1": _night(ev_musd=800.0),
             "insider:i1": {"price": 110.0, "trigger_auto": "A", "trigger_note": "x"}}
    _a, state = ar.sheet_alerts(_blob([]), None)                                    # baslinjen
    alerts, state = ar.sheet_alerts(_blob(sr.build_events(sheets, night)), state)
    got = {a["kind"]: a for a in alerts}
    assert set(got) == {"sheet_rule_verdict", "sheet_rule_ev_nav", "sheet_insider_trigger"}
    assert got["sheet_rule_verdict"]["target"] == got["sheet_rule_ev_nav"]["target"] == "rule"
    assert got["sheet_insider_trigger"]["target"] == "insider" and got["sheet_insider_trigger"]["ticker"] == "EKTA-B.ST"
    # samma läge nästa natt: inget nytt larm
    again, state = ar.sheet_alerts(_blob(sr.build_events(sheets, night)), state)
    assert again == []
    # du trycker Använd: raden har dagens tal och triggern — händelserna tar slut, inget larm
    row.update(price=2500.0, ev_musd=800.0)
    sig.update(trigger="A")
    done, state = ar.sheet_alerts(_blob(sr.build_events(sheets, night)), state)
    assert done == [] and state["events"] == {}


def _nights(row: dict, nights: list) -> list:
    """build_events → sheet_alerts natt för natt utan att raden ändras (du trycker inte Använd): larmens titlar."""
    sheets = {"producers": {"producers": [row]}}
    _a, state = ar.sheet_alerts(_blob([]), None)                                    # baslinjen
    out = []
    for s in nights:
        alerts, state = ar.sheet_alerts(_blob(sr.build_events(sheets, {"producers:r1": s})), state)
        out.append([a["title"] for a in alerts])
    return out


def test_a_second_verdict_change_alarms_too():
    """Bedömningarna har tre lägen: ett andra byte utan Använd emellan larmar en gång, samma läge nästa natt inte."""
    nav = _nights(_nav_row(), [{"ev_musd": v} for v in (716.0, 750.0, 1010.0, 1060.0)])   # sparat 650 / NAV 1 000
    assert nav == [[f"🪨 Rick Rule: BOL.ST 0.72× NAV — {prod.NAV_FAIR}"], [],
                   [f"🪨 Rick Rule: BOL.ST 1.01× NAV — {prod.NAV_RICH}"], []]
    # sparat 850 (Rimligt): Billigt i natt, några veckor senare Dyrt — det sista larmet ska inte stå kvar på Billigt
    back = _nights(_nav_row(ev_musd=850.0), [{"ev_musd": 680.0}, {"ev_musd": 1010.0}])
    assert back == [[f"🪨 Rick Rule: BOL.ST 0.68× NAV — {prod.NAV_CHEAP}"],
                    [f"🪨 Rick Rule: BOL.ST 1.01× NAV — {prod.NAV_RICH}"]]
    # guld sparat 3 300, AISC 1 900, två kryss = Köpkandidat
    verdict = _nights(_rule(price=3300.0, unit_cost=1900.0),
                      [_night(commodity_price=v) for v in (3040.0, 2900.0, 2480.0)])
    assert verdict == [[f"🪨 Rick Rule: BOL.ST → {prod.P_WATCH}"], [], [f"🪨 Rick Rule: BOL.ST → {prod.P_PASS}"]]


def test_the_hidden_log_shows_only_the_kind(monkeypatch, caplog):
    """Actions-loggen är publik: med nyckeln skrivs bara händelsens slag, aldrig tickern."""
    import alert_scan
    import borsdata_api
    import private_data
    events = sr.build_events({"producers": {"producers": [_rule()]}, "insider": {"signals": [_sig()]}},
                             {"producers:r1": _night(), "insider:i1": {"price": 110.0, "trigger_auto": "A"}})
    monkeypatch.setenv("BORSDATA_API_KEY", "x")
    monkeypatch.setattr(borsdata_api, "BorsdataAPI", lambda api_key=None: object())
    monkeypatch.setattr(sr, "load_sheets", lambda: {"insider": {}})
    monkeypatch.setattr(sr, "refresh", lambda api, sheets: {"generated": "g", "rows": {}, "events": events,
                                                            "error": None})
    monkeypatch.setattr(alert_scan, "UNREADABLE_PATHS", set())
    monkeypatch.setattr(private_data, "passphrase", lambda: "hemlig")
    monkeypatch.setattr("sys.argv", ["sheets_refresh.py", "--dry-run"])
    caplog.set_level(logging.INFO)
    assert sr.main() == 0
    assert "rule_verdict (rubriken dold)" in caplog.text and "insider_trigger (rubriken dold)" in caplog.text
    assert "BOL" not in caplog.text and "EKTA" not in caplog.text


# ── Hela nattjobbet: talen kommer ur samma hämtning, inga nya anrop ──────────
def _bars(closes, end=TODAY):
    n = len(closes)
    return [{"d": (end - timedelta(days=n - 1 - i)).isoformat(), "c": c} for i, c in enumerate(closes)]


class _API:
    """EKTA (insider, SEK) och BOL (Rick Rule, SEK). Stängningarna stiger 100 → 119: över MA20 som vänt → trigger A."""

    def __init__(self):
        self.calls = []

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
        snaps = {40: {"ev": 8000.0}}                               # 8 000 MSEK × 0,095 = 760 MUSD
        return {i: dict(snaps.get(i, {}), ins_id=i) for i in ids}

    def get_stockprices(self, ins_id, max_count=5):
        return _bars([100.0] * 10 + [100.0 + i for i in range(20)])[-max_count:]

    def get_reports(self, ins_id, kind, max_count=7):
        return []


def test_the_nightly_job_raises_the_new_alarms_from_its_own_numbers(monkeypatch):
    import commodity_prices as cp
    monkeypatch.setattr(cp, "spot_many", lambda names: {"guld": {"price": 2500.0, "unit": "USD/oz",
                                                                 "asof": "2026-11-06", "ticker": "GC=F"}})
    sheets = {"insider": {"signals": [_sig(cluster_avg=110.0)]},
              "producers": {"producers": [_rule(ev_musd=650.0, nav_musd=1000.0)]}}
    api = _API()
    out = sr.refresh(api, sheets, today=TODAY)
    assert out["error"] is None
    s = out["rows"]
    assert s["insider:i1"]["price"] == 119.0 and s["insider:i1"]["trigger_auto"] == "A"
    assert s["producers:r1"]["commodity_price"] == 2500.0 and s["producers:r1"]["ev_musd"] == 760.0
    keys = {e["key"] for e in out["events"]}
    assert keys == {"insider_trigger:insider:i1", f"rule_verdict:producers:r1:{prod.P_BUY}",
                    f"rule_ev_nav:producers:r1:{prod.NAV_FAIR}"}
    assert set(api.calls) <= {"get_instruments", "get_global_instruments_list", "get_fundamentals_snapshot_fast",
                              "get_stockprices", "resolve_instrument_id", "get_reports"}
    assert sheets["insider"]["signals"][0]["trigger"] == "" and sheets["producers"]["producers"][0]["price"] == 2000.0


# ── Larminställningarna (MER → Larm) säger vad benet larmar för ─────────────
def _settings_page():
    from tabs.alerts import _render_scheduled_settings
    _render_scheduled_settings(None)


def test_the_alert_settings_name_the_new_sheet_alarms(monkeypatch):
    from streamlit.testing.v1 import AppTest
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_watch_ui as twu
    twu.fake_repo(monkeypatch, {"alerts": {"sheets": {"enabled": True, "channels": ["discord"]}}})
    at = AppTest.from_function(_settings_page, default_timeout=60).run()
    assert not at.exception
    assert at.toggle(key="sched_sheets_on").label == "📋 Arkens övergångar"
    assert any("nattens trigger som gör raden till KÖP" in c.value
               and "bedömning eller EV/NAV som byter med dagens tal" in c.value for c in at.caption)
