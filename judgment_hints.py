"""
judgment_hints.py — förslag på bedömningsfälten, aldrig ifyllnad.

Jurisdiktion, insyn, ägare och utspädningshistorik är bedömningar och ska
förbli det. Men underlaget finns ofta redan: extraktorn läser jurisdiktion
och insynsägande ur presentationen (extract_store), Durrett-arket har land
och jurisdiktion på bolaget, och sifferuppdateringen har aktiehistoriken.
Här översätts underlaget till ett förslag med källa, som arket visar som
text bredvid kryssrutan eller faktorn. Kryssa gör du.

Fraser-listan är Fraser Institutes Investment Attractiveness Index (Annual
Survey of Mining Companies 2023), grovt indelad i övre halvan och botten.
Den är ungefärlig och uppdateras för hand — en jurisdiktion som saknas ger
inget förslag alls, inte ett negativt.
"""

from __future__ import annotations

from typing import Optional

# Övre halvan: delstater/provinser/länder som brukar ligga där. Matchas som
# delsträng, skiftlägesokänsligt, mot jurisdiktionstexten.
FRASER_TOP_HALF: tuple = (
    # USA
    "nevada", "arizona", "utah", "idaho", "alaska", "wyoming", "colorado", "montana",
    "new mexico", "minnesota", "michigan",
    # Kanada
    "quebec", "québec", "saskatchewan", "ontario", "british columbia", "newfoundland",
    "labrador", "manitoba", "yukon", "nunavut", "northwest territories", "new brunswick",
    # Australien
    "western australia", "queensland", "south australia", "northern territory",
    "new south wales", "victoria", "tasmania",
    # Europa och övriga
    "finland", "sweden", "sverige", "norway", "norge", "ireland", "irland",
    "botswana", "namibia", "morocco", "marocko", "ghana", "chile",
)
# Botten: jurisdiktioner som brukar ligga bland de sista.
FRASER_BOTTOM: tuple = (
    "venezuela", "zimbabwe", "bolivia", "democratic republic of congo", "drc", "kongo",
    "kyrgyz", "angola", "la rioja", "chubut", "mendoza", "zacatecas", "nicaragua",
    "south africa", "sydafrika", "tanzania", "papua new guinea", "mongolia", "mongoliet",
)
FRASER_EDITION = "Fraser 2023"

# Insynsägande: Sprotts faktor 4 säger "> 10 %" för 2 p; Tiggre "5–10 %".
INSIDER_STRONG_PCT = 10.0
INSIDER_OK_PCT = 5.0

# DS "Historik": aktier på fem år. Över +50 % är en serieutspädare, +20–50 %
# blandad, under det disciplinerad — grova trösklar, därav "förslag".
DILUTION_5Y_MIXED_PCT = 20.0
DILUTION_5Y_SERIAL_PCT = 50.0


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


# ── Jurisdiktion ─────────────────────────────────────────────────────────────
def jurisdiction_verdict(text) -> Optional[tuple]:
    """(True/False, träffen) för en jurisdiktionstext, None när listan inte
    känner igen den. Botten prövas först: "Chubut, Argentina" är botten även
    om landet inte står i någon lista."""
    t = str(text or "").strip().lower()
    if not t:
        return None
    for name in FRASER_BOTTOM:
        if name in t:
            return False, name
    for name in FRASER_TOP_HALF:
        if name in t:
            return True, name
    return None


def jurisdiction_points(text) -> Optional[int]:
    """Tiggres faktor Jurisdiktion (0–2): toppjurisdiktion 2, botten 0."""
    v = jurisdiction_verdict(text)
    if v is None:
        return None
    return 2 if v[0] else 0


# ── Insyn ────────────────────────────────────────────────────────────────────
def insider_ok(pct) -> Optional[bool]:
    """Rick Rules kryss "Insynsägande": ledningen äger själv (≥ 5 %)."""
    p = _num(pct)
    return None if p is None else p >= INSIDER_OK_PCT


def owner_points(pct) -> Optional[int]:
    """Sprotts faktor 4 "Ägare & management": > 10 % → 2, 5–10 % → 1, annars 0."""
    p = _num(pct)
    if p is None:
        return None
    if p > INSIDER_STRONG_PCT:
        return 2
    return 1 if p >= INSIDER_OK_PCT else 0


# ── Utspädningshistorik ──────────────────────────────────────────────────────
def ds_history_points(growth_5y_pct) -> Optional[int]:
    """DS "Historik" ur fem års aktiehistorik: > +50 % serieutspädare (2),
    +20–50 % blandad (1), annars disciplinerad (0)."""
    g = _num(growth_5y_pct)
    if g is None:
        return None
    if g > DILUTION_5Y_SERIAL_PCT:
        return 2
    return 1 if g >= DILUTION_5Y_MIXED_PCT else 0


# ── Underlaget för ett bolag ─────────────────────────────────────────────────
def gather(ticker: str, confidence_data: Optional[dict] = None,
           extraction: Optional[dict] = None) -> dict:
    """{jurisdiction: (text, källa), insider_pct: (tal, källa)} ur det som
    finns: utdraget (extract_store) först, sedan Durrett-arkets bolag."""
    out: dict = {}
    parsed = (extraction or {}).get("parsed") if extraction else None
    fields = (parsed or {}).get("fields") or {}
    src_x = f"utdrag ur {(extraction or {}).get('doc') or 'dokumentet'}"

    def _x(key):
        raw = fields.get(key)
        return raw.get("value") if isinstance(raw, dict) else None

    jur = _x("jurisdiction")
    if jur:
        out["jurisdiction"] = (str(jur), src_x)
    ins = _num(_x("insider_ownership_pct"))
    if ins is None:
        ins = _num(_x("insider_ownership"))
    if ins is not None:
        out["insider_pct"] = (ins, src_x)

    if confidence_data:
        try:
            from confidence import store as cs
            c = cs.get(confidence_data, ticker)
        except Exception:
            c = None
        if c is not None:
            if "jurisdiction" not in out and (c.jurisdiction or c.country):
                out["jurisdiction"] = (", ".join(x for x in (c.jurisdiction, c.country) if x),
                                       "Durrett-arket")
            if "insider_pct" not in out and c.has("insider_ownership_pct"):
                v = c.num("insider_ownership_pct")
                if v is not None:
                    out["insider_pct"] = (v, "Durrett-arket")
    return out


def gather_for(ticker: str) -> dict:
    """Som gather, men läser sessionen själv (extract_store + confidence-lagret).
    Tomt vid fel — ett förslag får aldrig fälla ett ark."""
    try:
        import extract_store
        import storage
        from confidence import store as cs
        return gather(ticker, storage.session_load(cs.STORE, cs.default()),
                      extract_store.get(ticker))
    except Exception:
        return {}


# ── Texterna arken visar ─────────────────────────────────────────────────────
def jurisdiction_hint(ticker: str, current=None, as_points: bool = False) -> Optional[str]:
    """Förslagstext för jurisdiktionen, eller None när underlaget saknas eller
    redan stämmer med det som är satt (kryss eller poäng)."""
    g = gather_for(ticker)
    if "jurisdiction" not in g:
        return None
    text, src = g["jurisdiction"]
    v = jurisdiction_verdict(text)
    if v is None:
        return None
    ok, hit = v
    want = (2 if ok else 0) if as_points else ok
    if current is not None and current == want:
        return None
    half = "övre halvan" if ok else "botten"
    what = f"Jurisdiktion {want} p" if as_points else ("Jurisdiktion OK" if ok else "Jurisdiktion INTE OK")
    return (f"Förslag: {what} — \"{text}\" ({hit.title()} ligger i {half} av {FRASER_EDITION}; "
            f"{src}). Kryssa själv.")


def insider_hint(ticker: str, current=None, as_points: bool = False) -> Optional[str]:
    """Förslagstext för insyn (kryss) eller Sprotts ägarfaktor (poäng)."""
    g = gather_for(ticker)
    if "insider_pct" not in g:
        return None
    pct, src = g["insider_pct"]
    want = owner_points(pct) if as_points else insider_ok(pct)
    if want is None or (current is not None and current == want):
        return None
    what = f"Ägare & management {want} p" if as_points else ("Insynsägande ✓" if want else "Insynsägande ✗")
    return f"Förslag: {what} — insynsägande {pct:g} % ({src}). Kryssa själv."
