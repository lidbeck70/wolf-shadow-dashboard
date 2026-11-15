"""
portfolio/broker_sync.py — mäklarens innehavsfil mot Positioner (PORTFÖLJ → Positioner → 📥 Uppdatera från mäklaren).

Filerna:
  Nordea ASK   innehavslistan "My holdings" (.xlsx, bladet Holdings): ISIN, MIC, valuta, namn, antal och
               snittkurs per konto. Läses rakt av.
  Nordnet ISK  transaktionsexporten (.csv, samma fil som Journalens import): innehaven räknas fram med FIFO
               (journal_import). Exporten måste täcka hela historiken — ett köp före exportens start syns inte.

Stegen (rena funktioner — inget Streamlit, inget nätverk utom isin_map som anroparen ger instrumenten):
  read_file      fil → (mäklare, [Holding], fel)
  isin_map       Börsdatas instrument → {ISIN: Yahoo-ticker}
  resolve        Holding.ticker ur radens sparade ISIN, Börsdata (på knapp) eller — strängt — namnet på en rad.
                 Journalens ISIN-karta (journal_import.ticker_map) används inte: den kopplar flera nordiska ISIN
                 till fel bolag (Volvo B → Atlas Copco).
  plan           [Change]: ny / ändrad / stämmer / saknas hos mäklaren / bara text
  apply          de markerade ändringarna till registret i EN sparning: antal och snittkurs (aldrig stopp, mål,
                 strategi eller anteckningar); nya rader som Untagged

Samma bolag på flera konton slås ihop till en rad bara när alla kontona finns i de uppladdade filerna — annars
vore ett kontos antal inte radens innehav. En rad som står i flera strategier får bara text (vilken rad ska ändras?).
Snittkursen föreslås bara i radens noteringsvaluta (position_hints.broker_price, London i pence).
Inget skrivs utan klick: apply anropas bara av knappen.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

NORDEA = "nordea_ask"
NORDNET = "nordnet_isk"
BROKER_LABEL = {NORDEA: "Nordea ASK", NORDNET: "Nordnet ISK"}

# Rutornas typer
NEW, CHANGED, SAME, MISSING, INFO = "ny", "ändrad", "stämmer", "saknas", "text"

# MIC → Yahoo-suffix, bara som reserv när Börsdata inte hittar ISIN:et (tickern måste ändå skrivas)
MIC_SUFFIX = {"XSTO": ".ST", "XOSL": ".OL", "XCSE": ".CO", "XHEL": ".HE", "XLON": ".L", "XETR": ".DE",
              "XTSE": ".TO", "XTSX": ".V"}


@dataclass
class Holding:
    """Ett innehav på ett konto enligt mäklaren. avg_price i instrumentets valuta (None när mäklaren inte har den)."""
    broker: str
    account: str
    isin: str
    name: str
    currency: str
    qty: float
    avg_price: Optional[float] = None
    mic: str = ""
    ticker: Optional[str] = None
    how: str = ""                 # hur tickern hittades


@dataclass
class Change:
    kind: str                     # NEW | CHANGED | SAME | MISSING | INFO
    key: str                      # stabil nyckel för kryssrutan
    ticker: Optional[str]
    name: str
    isin: str = ""
    accounts: tuple = ()
    row: Optional[dict] = None
    qty: Optional[float] = None
    price: Optional[float] = None
    old_qty: Optional[float] = None
    old_price: Optional[float] = None
    currency: str = ""
    notes: list = field(default_factory=list)
    pick: bool = True             # kryssad från början (inte när tickern bara kommer ur ett namn)


# ── Hjälpare ────────────────────────────────────────────────────────────────
def _num(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f or f in (float("inf"), float("-inf")) else f


def _qty(v) -> float:
    q = _num(v) or 0.0
    return int(q) if abs(q - round(q)) < 1e-6 else round(q, 6)


def _same(a, b, tol: float) -> bool:
    a, b = _num(a), _num(b)
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol * max(1.0, abs(b))


def _bytes(file_obj) -> bytes:
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)
    if isinstance(file_obj, str):
        with open(file_obj, "rb") as f:
            return f.read()
    data = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
    return data if isinstance(data, bytes) else str(data).encode("utf-8")


# ── 1. Filerna ──────────────────────────────────────────────────────────────
def parse_nordea_holdings(data: bytes) -> tuple:
    """Nordeas innehavslista (.xlsx) → ([Holding], fel, läst). Rubrikraden letas upp (raden med ISIN och HOLDINGS);
    kassakonton och tomma rader hoppas över. Snittkurs 0 (fonder) = saknas, utom när fondens valuta är kontots
    basvaluta — då är anskaffningsvärdet ÷ antal snittkursen."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    ws = wb["Holdings"] if "Holdings" in wb.sheetnames else wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    head_i = next((i for i, r in enumerate(rows)
                   if r and "ISIN" in [str(c).strip() for c in r if c is not None]
                   and "HOLDINGS" in [str(c).strip() for c in r if c is not None]), None)
    if head_i is None:
        return [], ["Nordea: hittar inte rubrikraden (ISIN, HOLDINGS) — är det innehavslistan 'My holdings'?"], False
    head = [str(c).strip() if c is not None else "" for c in rows[head_i]]
    out, errors = [], []
    for n, r in enumerate(rows[head_i + 1:], start=head_i + 2):
        d = dict(zip(head, r))
        if str(d.get("Type") or "").strip() not in ("Custody", ""):
            continue
        isin = str(d.get("ISIN") or "").strip()
        qty = _num(d.get("HOLDINGS"))
        if not isin or not qty:
            continue
        ccy = str(d.get("CURRENCY") or "").strip().upper()
        avg = _num(d.get("Average purchase price"))
        if not avg or avg <= 0:
            base = str(d.get("Base currency") or "").strip().upper()
            cost = _num(d.get("Purchase value of current holdings"))
            avg = cost / qty if cost and cost > 0 and ccy and ccy == base else None
        acct = str(d.get("Account") or d.get("AccountKey") or "").strip()
        out.append(Holding(NORDEA, acct, isin, str(d.get("NAME") or "").strip(), ccy, _qty(qty),
                           round(avg, 6) if avg else None, str(d.get("MIC") or "").strip().upper()))
    if not out and not errors:
        errors.append("Nordea: inga innehav i filen.")
    return out, errors, True


def nordnet_holdings(data: bytes) -> tuple:
    """Nordnets transaktionsexport (.csv) → ([Holding], fel, läst) via Journalens parser och FIFO. Snittkursen står i
    handelsvalutan. Ett sälj utan köp före i filen = exporten börjar för sent; det står i felen. Nordnet listar
    nyast först: köp sorteras före sälj samma dag (FIFO tar ändå de äldsta posterna först), annars blev en
    dagsaffär (köp och sälj samma dag) kvar som innehav. Utan ett enda köp eller sälj (fel teckenkodning, en period
    med bara utdelningar) räknas filen inte som läst — Nordnet-raderna ska då inte se sålda ut."""
    from journal_import.fifo import run_fifo
    from journal_import.parsers import parse_nordnet
    res = parse_nordnet(io.BytesIO(data))
    txs = sorted(res.transactions, key=lambda t: (t.account, t.isin, t.date, t.action != "BUY"))
    _closed, open_pos, fifo_errors = run_fifo(txs, lambda *a, **k: None)
    out = [Holding(NORDNET, str(p.account or NORDNET), str(p.isin or ""), str(p.name or ""),
                   str(p.currency or "").upper(), _qty(p.qty), round(p.avg_price, 6) if p.avg_price else None)
           for p in open_pos if (p.qty or 0) > 0]
    errors = list(res.errors) + list(fifo_errors)
    if not res.transactions:
        errors.append("Nordnet: inga köp eller sälj i filen — är det transaktionsexporten som CSV (UTF-16, så som "
                      "Nordnet sparar den), för hela perioden sedan första köpet?")
        return out, errors, False
    return out, errors, True


def read_file(name: str, file_obj) -> tuple:
    """(mäklare, [Holding], fel) ur en uppladdad fil. Mäklaren känns igen på filtypen: .xlsx = Nordeas
    innehavslista, .csv = Nordnets transaktionsexport. PDF (avräkningsnotor) går inte att läsa som innehav.
    Mäklaren är None när filen inte gick att läsa som sin typ — då räknas den inte som uppladdad (plan: rader hos
    den mäklaren blir varken 'saknas' eller kapade till den andra mäklarens antal)."""
    n = str(name or "").lower()
    try:
        data = _bytes(file_obj)
    except Exception as exc:
        return None, [], [f"Kunde inte läsa filen: {exc}"]
    if n.endswith(".pdf"):
        return None, [], ["PDF:en är avräkningsnotor, inte en innehavslista. Från Nordnet: exportera "
                          "transaktionerna som CSV (hela perioden sedan första köpet)."]
    try:
        if n.endswith((".xlsx", ".xlsm")) or data[:2] == b"PK":
            h, e, ok = parse_nordea_holdings(data)
            return (NORDEA if ok else None), h, e
        if n.endswith((".csv", ".txt")):
            h, e, ok = nordnet_holdings(data)
            return (NORDNET if ok else None), h, e
    except Exception as exc:
        return None, [], [f"Kunde inte läsa {name}: {exc}"]
    return None, [], [f"Okänd filtyp: {name} — Nordea: innehavslistan (.xlsx), Nordnet: transaktionerna (.csv)."]


# ── 2. Tickern ──────────────────────────────────────────────────────────────
def isin_map(instruments: Iterable[dict], table: Optional[dict] = None) -> dict:
    """{ISIN: Yahoo-ticker} ur Börsdatas instrumentlistor (markets.to_yf ur marknads-id). Ett ISIN som står på
    flera listor behåller den första (Nordens lista först)."""
    import markets as mk
    out: dict = {}
    for inst in instruments or []:
        if not isinstance(inst, dict):
            continue
        isin = str(inst.get("isin") or "").strip().upper()
        t = str(inst.get("ticker") or "").strip()
        if isin and t and isin not in out:
            try:
                out[isin] = mk.to_yf(t, inst.get("marketId"), table)
            except Exception:
                continue
    return out


def _row_isin(row: dict) -> str:
    return str(((row or {}).get("extras") or {}).get("isin") or (row or {}).get("isin") or "").strip().upper()


def resolve(holdings: List[Holding], rows: Iterable[dict], bd_map: Optional[dict] = None) -> List[Holding]:
    """Fyller Holding.ticker i ordningen: radens sparade ISIN (en tidigare uppdatering), Börsdata (bd_map), sist ett
    namn som strängt liknar en rads namn i samma valuta. Utan träff förblir tickern None (skrivs i rutan)."""
    rows = [r for r in rows or [] if isinstance(r, dict)]
    by_isin = {}
    for r in rows:
        i = _row_isin(r)
        if i:
            by_isin.setdefault(i, r["ticker"])
    for h in holdings:
        if h.ticker:
            continue
        isin = h.isin.upper()
        if isin in by_isin:
            h.ticker, h.how = by_isin[isin], "ISIN sparat på raden"
        elif bd_map and isin in bd_map:
            h.ticker, h.how = bd_map[isin], "Börsdata (ISIN)"
        else:
            hit = _name_match(h, rows)
            if hit:
                h.ticker, h.how = hit, BY_NAME
    # samma ISIN på flera konton får samma ticker — annars blev ena kontot ett kort och det andra ett annat
    found: dict = {}
    for h in holdings:
        if h.ticker and h.isin:
            prev = found.get(h.isin.upper())
            if prev is None or h.how != BY_NAME and prev[1] == BY_NAME:
                found[h.isin.upper()] = (h.ticker, h.how)
    weak = {i for i in found if any(x.how == BY_NAME for x in holdings if x.isin.upper() == i)}
    for h in holdings:
        hit = found.get(h.isin.upper()) if h.isin else None
        if hit and not h.ticker:
            h.ticker, h.how = hit[0], BY_NAME if h.isin.upper() in weak else hit[1]
    return holdings


BY_NAME = "namnet liknar radens"


def _tokens(name) -> set:
    """Namnets ord utan bolagsformer och klassbeteckning ('ser.', 'B'). Branschord räknas — 'Lundin Gold' och
    'Lundin Mining' är olika bolag."""
    import refresh_ui
    return set(refresh_ui._name_tokens(name)) - {"ser", "serie", "series"}


def _name_match(h: Holding, rows: list) -> Optional[str]:
    """Tickern på den enda rad vars namn har exakt samma ord som mäklarens (bolagsformer och klassbeteckning räknas
    inte: 'Holmen AB ser. B' = 'Holmen B'). 'Hexagon AB' mot 'Hexagon Composites ASA', 'Lundin Gold' mot 'Lundin
    Mining' eller 'Nel' mot 'Nelly' ger ingen träff. Nordeas valuta är instrumentets och måste vara radens
    noteringsvaluta; Nordnets är alltid kontots (SEK) och prövas inte. En träff kryssas inte i från början
    (Change.pick) — namnet är en gissning."""
    import positions
    want = _tokens(h.name)
    if not want:
        return None
    hits = set()
    for r in rows:
        own = _tokens(r.get("name") or "")
        if not own or own != want:
            continue
        listing = positions.currency_for(r["ticker"])
        ccy = (h.currency or "").upper() if h.broker != NORDNET else ""
        if ccy and not (ccy == listing or (r["ticker"].upper().endswith(".L") and ccy in ("GBP", "GBX"))):
            continue
        hits.add(r["ticker"])
    return hits.pop() if len(hits) == 1 else None


# ── 3. Planen ───────────────────────────────────────────────────────────────
def _row_accounts(row: dict) -> set:
    """Kontona raden har hämtats från tidigare (sparat vid uppdatering), annars ur Journalens importanteckning
    ('import · nordnet_isk · SEK')."""
    ex = (row or {}).get("extras") or {}
    acc = ex.get("broker_accounts")
    if isinstance(acc, (list, tuple)) and acc:
        return {str(a) for a in acc}
    notes = str((row or {}).get("notes") or "")
    return {b for b in (NORDEA, NORDNET) if b in notes}


def _brokers_of(accounts: Iterable[str]) -> set:
    return {a.split(":", 1)[0] for a in accounts}


def _acct_key(h: Holding) -> str:
    return f"{h.broker}:{h.account}"


def _avg(hs: List[Holding]) -> tuple:
    """(antalsvägd snittkurs, text) — bara när alla poster har snittkurs i samma valuta."""
    ccys = {h.currency for h in hs}
    if len(ccys) > 1:
        return None, "kontona har olika valuta — snittkursen föreslås inte"
    if not all(h.avg_price for h in hs):
        return None, ""
    return sum(h.avg_price * h.qty for h in hs) / sum(h.qty for h in hs), ""


def _resembles(own: str, broker: str) -> bool:
    import refresh_ui
    return refresh_ui.name_resembles(own, broker)


def plan(rows: Iterable[dict], holdings: List[Holding], uploaded: Iterable[str]) -> List[Change]:
    """Ändringarna mot registrets öppna rader. uploaded = mäklarna vars filer finns ({'nordea_ask', ...}).

    Per ticker: ny (ingen rad), ändrad/stämmer (en rad), text (flera rader med tickern, saknad ticker, eller kontot
    raden också ligger på saknas bland filerna). Rader som har ett konto hos en uppladdad mäklare men inget innehav
    där blir 'saknas'. Rader utan känt konto röras inte — de kan ligga hos en annan mäklare."""
    import positions
    from portfolio import position_hints as ph
    rows = [r for r in rows or [] if isinstance(r, dict) and r.get("ticker")]
    uploaded = set(uploaded or ())
    out: List[Change] = []
    groups: dict = {}
    unknown: dict = {}
    for h in holdings:
        if not h.ticker:
            unknown.setdefault(h.isin.strip().upper() or h.name, []).append(h)
            continue
        groups.setdefault(h.ticker.strip().upper(), []).append(h)
    for key, hs in unknown.items():                  # samma ISIN på flera konton: ett kort, antalet summerat
        accts = tuple(sorted({_acct_key(h) for h in hs}))
        avg, note = _avg(hs)
        notes = ["ingen ticker hittad — skriv den för att lägga till"] + (
            [f"summa av {len(accts)} konton"] if len(accts) > 1 else []) + ([note] if note else [])
        out.append(Change(NEW, f"new:{key}", None, hs[0].name, hs[0].isin, accts, qty=_qty(sum(h.qty for h in hs)),
                          price=round(avg, 6) if avg else None, currency=hs[0].currency, notes=notes))
    rows_by_t: dict = {}
    for r in rows:
        rows_by_t.setdefault(r["ticker"].strip().upper(), []).append(r)
    for t, hs in groups.items():
        accts = tuple(sorted({_acct_key(h) for h in hs}))
        qty = _qty(sum(h.qty for h in hs))
        avg, anote = _avg(hs)
        name, isin, ccy = hs[0].name, hs[0].isin, hs[0].currency
        hows = sorted({h.how for h in hs if h.how and h.how != "ISIN sparat på raden"})
        notes = [f"ticker: {', '.join(hows)}"] if hows else []
        by_name = any(h.how == BY_NAME for h in hs)
        if len(accts) > 1:
            notes.append(f"summa av {len(accts)} konton")
        if anote:
            notes.append(anote)
        isins = sorted({h.isin.upper() for h in hs if h.isin})
        if len(isins) > 1:                           # två olika värdepapper på samma ticker — summeras aldrig
            out.append(Change(INFO, f"info:{t}", t, name, isin, accts, qty=None, currency=ccy,
                              row=(rows_by_t.get(t) or [None])[0] if len(rows_by_t.get(t) or []) == 1 else None,
                              notes=notes + [f"{len(isins)} olika ISIN ({', '.join(isins)}) pekar på {t} — "
                                             f"ändra själv"]))
            continue
        price, pnote = ph.broker_price(t, avg, ccy) if avg is not None else (None, "")
        price = ph._round(price) if price is not None else None
        if pnote:
            notes.append(pnote)
        rs = rows_by_t.get(t, [])
        if not rs:
            out.append(Change(NEW, f"new:{t}", t, name, isin, accts, qty=qty, price=price, currency=ccy, notes=notes))
            continue
        if len(rs) > 1:
            out.append(Change(INFO, f"info:{t}", t, name, isin, accts, qty=qty, price=price, currency=ccy,
                              notes=notes + [f"{t} står på {len(rs)} rader (olika strategier) — ändra raden själv"]))
            continue
        row = rs[0]
        own_name = str(row.get("name") or "").strip()
        saved = all(h.how == "ISIN sparat på raden" for h in hs)      # du har redan bekräftat kopplingen
        if own_name and name and not by_name and not saved and not _resembles(own_name, name):
            out.append(Change(INFO, f"info:{t}", t, name, isin, accts, row=row, qty=qty, price=price, currency=ccy,
                              notes=notes + [f"mäklarens '{name}' liknar inte radens '{own_name}' — kontrollera "
                                             f"tickern, inget ändras"]))
            continue
        other = {a for a in _row_accounts(row) if a.split(":", 1)[0] not in uploaded}
        if other:
            out.append(Change(INFO, f"info:{t}", t, name, isin, accts, row=row, qty=qty, price=price, currency=ccy,
                              notes=notes + ["raden ligger också hos " + ", ".join(
                                  BROKER_LABEL.get(b, b) for b in sorted(_brokers_of(other)))
                                  + " — ladda upp den filen också"]))
            continue
        old_q, old_p = positions._num(row.get("shares")), positions._num(row.get("entry_price"))
        stop = ph._pos(row.get("stop"))
        if price is not None and stop is not None and stop >= price:
            notes.append(f"snittkursen {ph.fmt_num(price)} ligger på eller under stoppen {ph.fmt_num(stop)} — "
                         f"bara antalet uppdateras")
            price = None
        same_q = _same(old_q, qty, 1e-6)
        same_p = price is None or _same(old_p, price, 5e-4)
        if by_name:
            notes.append("tickern gissad ur namnet — kryssa i om det stämmer")
        out.append(Change(SAME if same_q and same_p else CHANGED, f"row:{row['id']}", t, name, isin, accts,
                          row=row, qty=qty, price=price, old_qty=old_q, old_price=old_p, currency=ccy, notes=notes,
                          pick=not by_name))
    seen = {c.row["id"] for c in out if c.row is not None}
    held = set(groups)
    # Ett nytt innehav som heter som en omatchad rad är troligen samma innehav (raden kan ha fått fel ticker av
    # Journalens ISIN-karta: 'ATCO-A.ST' med namnet 'Volvo B'). Raden visas då inte som 'såld?', och kortet kryssas
    # inte i — annars blev det en dubblettrad, eller Stäng på en position du fortfarande äger.
    loose = {c.key: _tokens(c.name) for c in out if c.kind == NEW}
    for r in rows:
        if r["id"] in seen or r["ticker"].strip().upper() in held:
            continue
        own = _tokens(r.get("name") or "")
        twin = next((k for k, t in loose.items() if own and t == own), None)
        if twin is not None:
            card = next(c for c in out if c.key == twin)
            card.pick = False
            card.notes.append(f"heter som raden {r['ticker']} — skriv {r['ticker']} om det är samma" if not card.ticker
                              else f"heter som raden {r['ticker']} — samma innehav? Rätta i så fall radens ticker "
                                   f"till {card.ticker} i stället för att lägga till")
            continue
        accts = _row_accounts(r)
        if accts and _brokers_of(accts) <= uploaded:
            out.append(Change(MISSING, f"miss:{r['id']}", r["ticker"], str(r.get("name") or ""),
                              _row_isin(r), tuple(sorted(accts)), row=r,
                              old_qty=positions._num(r.get("shares")), old_price=positions._num(r.get("entry_price")),
                              notes=["finns inte i mäklarens fil — såld?"]))
    order = {NEW: 0, CHANGED: 1, MISSING: 2, INFO: 3, SAME: 4}
    return sorted(out, key=lambda c: (order[c.kind], c.ticker or c.name))


# ── 4. Sparningen ───────────────────────────────────────────────────────────
def apply(changes: Iterable[Change], typed: Optional[dict] = None, today: str = "") -> tuple:
    """De markerade ändringarna till registret i en sparning. ändrad → antal, snittkurs (när den föreslås), ISIN och
    konton; ny → raden som Untagged med antal, snittkurs, namn, ISIN och konton. typed = {Change.key: ticker} för
    nya utan ticker. Returnerar (gjorda, hoppade över, fel eller None). Stopp, mål, strategi och anteckningar rörs
    aldrig. Stängning går genom positions.close (egen knapp)."""
    import positions
    typed = typed or {}
    data = positions.load()
    done, skipped = [], []
    live = {r["id"]: (b, r) for b in positions.BUCKET_KEYS for r in data[b]}
    for c in changes:
        accts = list(c.accounts)
        if c.kind == CHANGED and c.row is not None:
            hit = live.get(c.row["id"])
            if hit is None:
                skipped.append(f"{c.ticker}: raden finns inte längre")
                continue
            _b, r = hit
            if c.qty is not None:
                r["shares"] = c.qty
            if c.price is not None:
                r["entry_price"] = c.price
            ex = r.setdefault("extras", {})
            if c.isin:
                ex["isin"] = c.isin
            ex["broker_accounts"] = accts
            done.append(c.ticker)
        elif c.kind == NEW:
            t = str(c.ticker or typed.get(c.key) or "").strip().upper()
            if not t:
                skipped.append(f"{c.name or c.isin}: ticker saknas")
                continue
            bucket = positions.bucket_for("Untagged")
            hits = [(b, r) for b in positions.BUCKET_KEYS for r in data[b] if r["ticker"] == t]
            if hits:
                # en skriven ticker som redan står i Positioner: ingen ny rad (det vore dubbelt). En enda rad får
                # ISIN:et, så nästa uppladdning går den vanliga vägen (ändrat: valuta, stopp och konton prövas).
                if len(hits) == 1 and c.isin:
                    hits[0][1].setdefault("extras", {})["isin"] = c.isin
                    done.append(f"{t} (kopplad till raden i {positions.BUCKETS[hits[0][0]]['name']} — ladda upp "
                                f"igen för antal och snittkurs)")
                else:
                    skipped.append(f"{t}: finns redan i Positioner")
                continue
            if len(data[bucket]) >= positions.BUCKETS[bucket]["max"]:
                skipped.append(f"{t}: {positions.BUCKETS[bucket]['name']} är full")
                continue
            data[bucket].append(positions.normalize_row({
                "ticker": t, "name": c.name, "strategy": "Untagged", "entry_price": c.price or 0,
                "shares": c.qty or 0, "entry_date": today or positions._today(),
                "notes": "mäklarfil · " + ", ".join(sorted(_brokers_of(accts))), "source": "broker",
                "extras": {"isin": c.isin, "broker_accounts": accts}}))
            done.append(t)
    if not done:
        return done, skipped, None
    for b in positions.BUCKET_KEYS:
        data[b] = [{k: v for k, v in positions.normalize_row(r).items() if k != "_bucket"} for r in data[b]]
    err = positions._commit(data)
    return done, skipped, err
