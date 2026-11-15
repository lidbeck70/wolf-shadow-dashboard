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
  resolve        Holding.ticker ur radens sparade ISIN, journalens ISIN-karta, Börsdata eller namnet på en rad
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
    """Nordeas innehavslista (.xlsx) → ([Holding], fel). Rubrikraden letas upp (raden med ISIN och HOLDINGS);
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
        return [], ["Nordea: hittar inte rubrikraden (ISIN, HOLDINGS) — är det innehavslistan 'My holdings'?"]
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
    return out, errors


def nordnet_holdings(data: bytes) -> tuple:
    """Nordnets transaktionsexport (.csv) → ([Holding], fel) via Journalens parser och FIFO. Snittkursen står i
    handelsvalutan. Ett sälj utan köp före i filen = exporten börjar för sent; det står i felen."""
    from journal_import.fifo import run_fifo
    from journal_import.parsers import parse_nordnet
    from journal_import.ticker_map import resolve_ticker
    res = parse_nordnet(io.BytesIO(data))
    _closed, open_pos, fifo_errors = run_fifo(res.transactions, resolve_ticker)
    out = [Holding(NORDNET, str(p.account or NORDNET), str(p.isin or ""), str(p.name or ""),
                   str(p.currency or "").upper(), _qty(p.qty), round(p.avg_price, 6) if p.avg_price else None)
           for p in open_pos if (p.qty or 0) > 0]
    errors = list(res.errors) + list(fifo_errors)
    if not res.transactions:
        errors.append("Nordnet: inga köp eller sälj i filen — är det transaktionsexporten som CSV?")
    return out, errors


def read_file(name: str, file_obj) -> tuple:
    """(mäklare, [Holding], fel) ur en uppladdad fil. Mäklaren känns igen på filtypen: .xlsx = Nordeas
    innehavslista, .csv = Nordnets transaktionsexport. PDF (avräkningsnotor) går inte att läsa som innehav."""
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
            h, e = parse_nordea_holdings(data)
            return NORDEA, h, e
        if n.endswith((".csv", ".txt")):
            h, e = nordnet_holdings(data)
            return NORDNET, h, e
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
    """Fyller Holding.ticker i ordningen: radens sparade ISIN (en tidigare uppdatering), journalens ISIN-karta,
    Börsdata (bd_map), sist ett namn som liknar en rads namn. Utan träff förblir tickern None (skrivs i rutan)."""
    from journal_import.ticker_map import ISIN_MAP
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
        elif isin in ISIN_MAP:
            h.ticker, h.how = ISIN_MAP[isin], "journalens ISIN-karta"
        elif bd_map and isin in bd_map:
            h.ticker, h.how = bd_map[isin], "Börsdata (ISIN)"
        else:
            hit = _name_match(h.name, rows)
            if hit:
                h.ticker, h.how = hit, "namnet liknar radens"
    return holdings


def _name_match(name: str, rows: list) -> Optional[str]:
    """Tickern på den enda rad vars namn liknar mäklarens (strängt: ett av namnen ryms i det andra)."""
    import refresh_ui
    if not name:
        return None
    norm = " ".join(refresh_ui._name_tokens(name)) if hasattr(refresh_ui, "_name_tokens") else name.lower()
    hits = set()
    for r in rows:
        own = str(r.get("name") or "").strip()
        if not own:
            continue
        o = " ".join(refresh_ui._name_tokens(own)) if hasattr(refresh_ui, "_name_tokens") else own.lower()
        if o and norm and (o in norm or norm in o):
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
    for h in holdings:
        if not h.ticker:
            out.append(Change(NEW, f"new:{h.isin}", None, h.name, h.isin, (_acct_key(h),), qty=h.qty,
                              price=h.avg_price, currency=h.currency,
                              notes=["ingen ticker hittad — skriv den för att lägga till"]))
            continue
        groups.setdefault(h.ticker.strip().upper(), []).append(h)
    rows_by_t: dict = {}
    for r in rows:
        rows_by_t.setdefault(r["ticker"].strip().upper(), []).append(r)
    for t, hs in groups.items():
        accts = tuple(sorted({_acct_key(h) for h in hs}))
        qty = _qty(sum(h.qty for h in hs))
        ccys = {h.currency for h in hs}
        prices = [h.avg_price for h in hs]
        avg = (sum(h.avg_price * h.qty for h in hs) / sum(h.qty for h in hs)
               if len(ccys) == 1 and all(p for p in prices) else None)
        name, isin, ccy = hs[0].name, hs[0].isin, hs[0].currency
        notes = [f"ticker: {hs[0].how}"] if hs[0].how and hs[0].how != "ISIN sparat på raden" else []
        if len(accts) > 1:
            notes.append(f"summa av {len(accts)} konton")
        if avg is None and len(ccys) > 1:
            notes.append("kontona har olika valuta — snittkursen föreslås inte")
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
        out.append(Change(SAME if same_q and same_p else CHANGED, f"row:{row['id']}", t, name, isin, accts,
                          row=row, qty=qty, price=price, old_qty=old_q, old_price=old_p, currency=ccy, notes=notes))
    seen = {c.row["id"] for c in out if c.row is not None}
    held = set(groups)
    for r in rows:
        if r["id"] in seen or r["ticker"].strip().upper() in held:
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
            if any(r["ticker"] == t for r in data[bucket]):
                skipped.append(f"{t}: finns redan")
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
