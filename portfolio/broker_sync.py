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
import re
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
    unsure: bool = False          # antalet kan vara fel (Nordnet: split, flytt eller sälj utan köp i filen)
    sold: bool = False            # Nordnet: allt sålt enligt filen (antal 0) — beviset för 'såld?'


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


def _nordnet_result(txs: list, shaky: dict, errors: list, what: str) -> tuple:
    """Nordnets köp och sälj → ([Holding], fel, läst) via FIFO. Köp sorteras före sälj samma dag (Nordnet listar
    nyast först; FIFO tar ändå de äldsta posterna först). shaky = {ISIN: namn} där antalet kan vara fel (split,
    flytt, sälj utan köp i filen). Utöver de öppna innehaven:
      - spöken (antal 0, osäkra) för osäkra ISIN utan öppet innehav — raden visas då inte som 'såld?';
      - sålda (antal 0, sold) för ISIN som filen visar sålda helt — bara de får Nordnet-rader att visas som 'såld?'.
        Ett köp före filens period syns inte alls, så frånvaro i filen är inget bevis.
    Utan ett enda köp eller sälj räknas filen inte som läst."""
    from journal_import.fifo import run_fifo
    txs = sorted(txs, key=lambda t: (t.account, t.isin, t.date, t.action != "BUY"))
    closed, open_pos, fifo_errors = run_fifo(txs, lambda *a, **k: None)
    for e in fifo_errors:                           # 'SELL without matching BUY: ISIN (namn) …'
        e = str(e)
        if "SELL without matching BUY:" in e:
            rest = e.split("SELL without matching BUY:", 1)[1].strip()
            isin = rest.split()[0].strip().upper() if rest else ""
            name = rest[rest.find("(") + 1:rest.find(")")] if "(" in rest and ")" in rest else ""
            if isin:
                shaky.setdefault(isin, name)
    out = [Holding(NORDNET, str(p.account or NORDNET), str(p.isin or ""), str(p.name or ""),
                   str(p.currency or "").upper(), _qty(p.qty), round(p.avg_price, 6) if p.avg_price else None,
                   unsure=str(p.isin or "").upper() in shaky)
           for p in open_pos if (p.qty or 0) > 0]
    held = {h.isin.upper() for h in out}
    out += [Holding(NORDNET, NORDNET, isin, name, "", 0, None, unsure=True)
            for isin, name in shaky.items() if isin not in held]
    gone: dict = {}
    for ct in closed:
        i = str(ct.isin or "").upper()
        if i and i not in held and i not in shaky:
            gone.setdefault(i, (str(ct.account or NORDNET), str(ct.name or "")))
    out += [Holding(NORDNET, acct, isin, name, "", 0, None, sold=True) for isin, (acct, name) in gone.items()]
    unmatched = sorted({str(e).split("(", 1)[1].split(")", 1)[0] for e in fifo_errors
                        if "SELL without matching BUY:" in str(e) and "(" in str(e)})
    other = [e for e in fifo_errors if "SELL without matching BUY:" not in str(e)]
    if unmatched:                                   # köpta före filens period — en rad, inte en per affär
        other.insert(0, f"{len(unmatched)} bolag har försäljningar utan köp i filen (köpta före filens period): "
                        f"{', '.join(unmatched)}. De räknas som osäkra och kryssas inte i.")
    errors = list(errors) + other
    if not txs:
        errors.append(f"Nordnet: inga köp eller sälj i filen — {what}")
        return out, errors, False
    return out, errors, True


def nordnet_holdings(data: bytes) -> tuple:
    """Nordnets transaktionsexport (.csv) → ([Holding], fel, läst) via Journalens parser och FIFO."""
    from journal_import.parsers import parse_nordnet
    res = parse_nordnet(io.BytesIO(data))
    return _nordnet_result(res.transactions, _nordnet_shaky_isins(data), res.errors,
                           "är det transaktionsexporten som CSV (UTF-16, så som Nordnet sparar den)?")


# Nordnets avräkningsnotor (PDF). Personnummer och namn läses aldrig ut.
_NOTE = re.compile(r"för Er räkning (köpt|sålt) följande värdepapper:\s*(.+?)\(ISIN ([A-Z0-9]{12})\)\s*-\s*([^\n]+)",
                   re.S)
_SP = "\u00a0\u202f "                               # vanligt, hårt och smalt mellanslag (pypdf ger hårda)
_FILL = re.compile(rf"^(\d+(?:,\d+)?)[{_SP}]+(\d[\d{_SP}]*,\d+)[{_SP}]+([A-Z]{{3}})[{_SP}]")
# sidor som flyttar antal utan köp/sälj (split, in-/uttag av värdepapper, emission, byte, fusion, inlösen)
_MOVE_WORDS = ("VP", "SPLIT", "BYTE", "FUSION", "EMISSION", "INLÖS", "AVREG")


def _num_sv(text: str) -> float:
    return float(re.sub(r"[\s\u00a0\u202f]", "", str(text)).replace(",", "."))


def parse_nordnet_notes(pages: list) -> tuple:
    """Sidornas text → ([Transaction], {ISIN: namn} osäkra, fel). En avräkningsnota per sida: köpt/sålt, värdepapper
    med ISIN, depå, affärsdatum och avsluten ('antal pris valuta …', antalet utan tusentalsmellanslag). Summan
    antal × pris × valutakurs prövas mot köpeskillingen; stämmer den inte blir ISIN:et osäkert. Sidor som flyttar
    antal (VP, split, emission …) gör ISIN:et osäkert. Utdelning, källskatt och ränta hoppas över."""
    from journal_import.parsers import Transaction, _make_hash
    txs, shaky, errors = [], {}, []
    for n, text in enumerate(pages, start=1):
        t = str(text or "")
        head = next((ln.strip() for ln in t.splitlines() if ln.strip() and not ln.strip().startswith("Nr ")), "")
        head = head.split("Utskriftsdatum")[0].strip().upper()
        depa = re.search(r"Depå:\s*(\d+)", t)
        acct = f"depå {depa.group(1)}" if depa else NORDNET
        m = _NOTE.search(t)
        if m and "AVRÄKNINGSNOTA" in head:
            action = "BUY" if m.group(1) == "köpt" else "SELL"
            isin, name = m.group(3), m.group(4).strip()
            day = re.search(r"Affärsdatum:\s*(\d{4}-\d{2}-\d{2})", t)
            fills = [(_num_sv(f.group(1)), _num_sv(f.group(2)), f.group(3))
                     for f in (_FILL.match(ln.strip()) for ln in t.splitlines()) if f]
            fx = re.search(rf"Valutakurs [A-Z]{{3}}/SEK ([\d{_SP},]+)", t)
            fx = _num_sv(fx.group(1)) if fx else 1.0
            kop = re.search(rf"Köpeskilling \(SEK\)[{_SP}]+([\d{_SP},]+)", t)
            qty = sum(q for q, _p, _c in fills)
            if not fills or not day or qty <= 0:
                shaky.setdefault(isin, name)
                errors.append(f"Nordnet-PDF sida {n}: avräkningsnotan för {name} gick inte att läsa")
                continue
            price = sum(q * p for q, p, _c in fills) / qty
            if kop and abs(qty * price * fx - _num_sv(kop.group(1))) > max(0.01 * _num_sv(kop.group(1)), 0.05):
                shaky.setdefault(isin, name)
                errors.append(f"Nordnet-PDF sida {n}: {name} — antal × pris stämmer inte med köpeskillingen")
            txs.append(Transaction(account=acct, isin=isin, name=name, symbol="", exchange="", date=day.group(1),
                                   action=action, qty=qty, price=price, currency=fills[0][2], fx_rate=fx, fee=0.0,
                                   gross_amount=0.0, raw_hash=_make_hash(acct, isin, day.group(1), qty, price)))
            continue
        if any(w in head.split() or w in head for w in _MOVE_WORDS):
            isin = re.search(r"ISIN:\s*([A-Z0-9]{12})", t)
            antal = re.search(rf"Antal:\s*([\d{_SP},]+)", t)
            name = re.search(r"Värdepapper:\s*([^\n]+)", t)
            if isin and antal and _num_sv(antal.group(1)):
                shaky.setdefault(isin.group(1), name.group(1).strip() if name else "")
    return txs, shaky, errors


def _pdf_pages(data: bytes) -> list:
    import pypdf
    return [p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(data)).pages]


def nordnet_pdf_holdings(data: bytes) -> tuple:
    """Nordnets transaktioner som PDF (avräkningsnotor) → ([Holding], fel, läst)."""
    txs, shaky, errors = parse_nordnet_notes(_pdf_pages(data))
    return _nordnet_result(txs, shaky, errors, "är det Nordnets transaktioner och notor som PDF?")


def _nordnet_shaky_isins(data: bytes) -> set:
    """{ISIN: namn} med en rad som flyttar antal utan köp/sälj (split, inlägg/uttag av värdepapper, byte, emission,
    fusion …) — inte utdelning (dess antal är innehavet, inte en flytt). FIFO räknar bara köp och sälj, så antalet för de ISIN:en kan vara fel — NVIDIA köpt före splitten
    10:1 gav 100 → 10. Samma kolumner som journal_import.parsers.parse_nordnet."""
    from journal_import.parsers import _to_df, _parse_decimal
    try:
        df = _to_df(io.BytesIO(data), encoding="utf-16", sep="\t")
    except Exception:
        return {}
    out: dict = {}
    for _i, row in df.iterrows():
        vals = row.tolist()
        if len(vals) < 9:
            continue
        typ, isin = str(vals[5]).strip().upper(), str(vals[7]).strip().upper()
        if not isin or isin in ("NAN", "NONE") or not any(w in typ for w in _MOVE_WORDS):
            continue                                 # köp/sälj, utdelning, källskatt, ränta: inte en flytt
        try:
            qty = _parse_decimal(vals[8])
        except Exception:
            qty = 0
        if qty:
            out.setdefault(isin, str(vals[6]).strip())
    return out


def read_file(name: str, file_obj) -> tuple:
    """(mäklare, [Holding], fel) ur en uppladdad fil. Mäklaren känns igen på filtypen: .xlsx = Nordeas
    innehavslista, .csv och .pdf = Nordnets transaktioner (CSV-exporten, eller avräkningsnotorna som PDF).
    Mäklaren är None när filen inte gick att läsa som sin typ — då räknas den inte som uppladdad (plan: rader hos
    den mäklaren blir varken 'saknas' eller kapade till den andra mäklarens antal)."""
    n = str(name or "").lower()
    try:
        data = _bytes(file_obj)
    except Exception as exc:
        return None, [], [f"Kunde inte läsa filen: {exc}"]
    try:
        if n.endswith(".pdf") or data[:4] == b"%PDF":
            h, e, ok = nordnet_pdf_holdings(data)
            return (NORDNET if ok else None), h, e
        if n.endswith((".xlsx", ".xlsm")) or data[:2] == b"PK":
            h, e, ok = parse_nordea_holdings(data)
            return (NORDEA if ok else None), h, e
        if n.endswith((".csv", ".txt")):
            h, e, ok = nordnet_holdings(data)
            return (NORDNET if ok else None), h, e
    except Exception as exc:
        return None, [], [f"Kunde inte läsa {name}: {exc}"]
    return None, [], [f"Okänd filtyp: {name} — Nordea: innehavslistan (.xlsx), Nordnet: transaktionerna (.csv "
                      f"eller .pdf)."]


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
SHAKY_NOTE = ("split, flytt eller sälj utan köp i Nordnet-filen — antalet ur köp och sälj kan vara fel; kryssa i "
              "bara om det stämmer")


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
    every = list(holdings)                           # spöken och sålda (antal 0) räknas bara mot 'såld?'
    ghosts = {h.isin.upper() for h in every if not h.qty and h.unsure and h.isin}
    sold_isins = {h.isin.upper() for h in every if h.sold and h.isin}
    sold_names = {frozenset(_tokens(h.name)) for h in every if h.sold and _tokens(h.name)}
    holdings = [h for h in every if h.qty]
    # osäkra ISIN från alla innehav, spöken inräknade: Nordnet-aktier som flyttats in eller fått nytt ISIN vid en
    # split syns inte i köpen — samma ISIN hos Nordea är då bara en del av innehavet
    shaky_isins = {h.isin.upper() for h in every if h.unsure and h.isin}
    out: List[Change] = []
    groups: dict = {}
    unknown: dict = {}
    card_names: dict = {}                            # alla mäklarnamn bakom ett nytt kort (Nordea och Nordnet stavar olika)
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
        card_names[f"new:{key}"] = [h.name for h in hs]
        shaky = any(h.unsure or h.isin.upper() in shaky_isins for h in hs)
        if shaky:
            notes.append(SHAKY_NOTE)
        out.append(Change(NEW, f"new:{key}", None, hs[0].name, hs[0].isin, accts, qty=_qty(sum(h.qty for h in hs)),
                          price=round(avg, 6) if avg else None, currency=hs[0].currency, notes=notes, pick=not shaky))
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
            card_names[f"new:{t}"] = [h.name for h in hs]
            shaky = any(h.unsure or h.isin.upper() in shaky_isins for h in hs)
            out.append(Change(NEW, f"new:{t}", t, name, isin, accts, qty=qty, price=price, currency=ccy,
                              notes=notes + ([SHAKY_NOTE] if shaky else []), pick=not shaky))
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
        shaky = any(h.unsure or h.isin.upper() in shaky_isins for h in hs)
        if shaky and not same_q:
            notes.append(SHAKY_NOTE)
        missing_brokers = sorted({NORDEA, NORDNET} - uploaded)
        unsure = not _row_accounts(row) and missing_brokers and not same_q
        if unsure:                                   # raden kan också ligga hos mäklaren vars fil saknas
            notes.append("raden har inget känt konto — innehavet kan också ligga hos "
                         + ", ".join(BROKER_LABEL[b] for b in missing_brokers)
                         + "; kryssa i bara om hela innehavet ligger här")
        out.append(Change(SAME if same_q and same_p else CHANGED, f"row:{row['id']}", t, name, isin, accts,
                          row=row, qty=qty, price=price, old_qty=old_q, old_price=old_p, currency=ccy, notes=notes,
                          pick=not by_name and not unsure and not (shaky and not same_q)))
        card_names[f"row:{row['id']}"] = [h.name for h in hs]
    seen = {c.row["id"] for c in out if c.row is not None}
    held = set(groups)
    # Namnet som skyddsnät. Journalens ISIN-karta kan ha gett äldre rader fel ticker ('ATCO-A.ST' med namnet
    # 'Volvo B', SEB A och Sandvik ombytta). Därför:
    #   - ett nytt kort vars namn står på någon rad med annan ticker kryssas aldrig i (det vore en dubblettrad), och
    #     kortet säger vilken rad det kan vara;
    #   - en rad vars namn finns bland mäklarens innehav visas aldrig som 'såld?' (Stäng på en position du äger).
    # _tokens tappar aktieslaget ('Atlas Copco A' = 'Atlas Copco B'), så skyddet kan ibland avkryssa ett riktigt nytt
    # kort — det är den säkra riktningen; du kryssar själv.
    rows_by_tokens: dict = {}
    for r in rows:
        tk = frozenset(_tokens(r.get("name") or ""))
        if tk:
            rows_by_tokens.setdefault(tk, []).append(r)
    # en rad räknas som upptagen bara när ett eget ändrat/stämmer-kort matchade den — inte när tickern bara finns
    # bland innehaven (Börsdata kan ge en annan aktie raden fel ticker, och det kortet blir då bara text)
    matched = {c.row["id"] for c in out if c.kind in (CHANGED, SAME) and c.row is not None}
    for c in out:
        if c.kind not in (NEW, CHANGED):
            continue
        tks = {frozenset(_tokens(n)) for n in card_names.get(c.key, [c.name]) if _tokens(n)}
        others = list({r["id"]: r for tk in tks for r in rows_by_tokens.get(tk, [])
                       if r["ticker"].strip().upper() != (c.ticker or "").strip().upper()
                       and (c.kind == NEW or r["id"] not in matched)}.values())
        if not others:
            continue
        c.pick = False
        names = ", ".join(sorted(r["ticker"] for r in others))
        if c.kind == CHANGED:
            c.notes.append(f"heter också som raden {names} — står en del av innehavet redan där? Kryssa i bara om "
                           f"inte")
        else:
            c.notes.append(f"heter som raden {names} — skriv {names} om det är samma" if not c.ticker
                           else f"heter som raden {names} — samma innehav? Rätta i så fall radens ticker till "
                                f"{c.ticker} i stället för att lägga till")
    held_names = {frozenset(_tokens(h.name)) for h in every if not h.sold and _tokens(h.name)}
    for r in rows:
        if r["id"] in seen or r["ticker"].strip().upper() in held:
            continue
        if frozenset(_tokens(r.get("name") or "")) in held_names or _row_isin(r) in ghosts:
            continue                                 # mäklaren har ett innehav med radens namn — inte 'såld?'
        accts = _row_accounts(r)
        if accts and _brokers_of(accts) <= uploaded:
            # Nordnets fil är transaktioner för en period: en rad hos Nordnet är 'såld?' bara när filen visar
            # försäljningen (ISIN eller namn bland de helt sålda). Nordeas innehavslista är en ögonblicksbild.
            proven = _row_isin(r) in sold_isins or frozenset(_tokens(r.get("name") or "")) in sold_names
            if NORDNET in _brokers_of(accts) and not proven:
                continue
            out.append(Change(MISSING, f"miss:{r['id']}", r["ticker"], str(r.get("name") or ""),
                              _row_isin(r), tuple(sorted(accts)), row=r,
                              old_qty=positions._num(r.get("shares")), old_price=positions._num(r.get("entry_price")),
                              notes=["finns inte i mäklarens fil — såld?" if NORDNET not in _brokers_of(accts)
                                     else "såld enligt Nordnets transaktioner"]))
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
