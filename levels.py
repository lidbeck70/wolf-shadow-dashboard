"""
levels.py — entry- och exitnivåer, räknade.

Det här är svaret på "tydligare entry och exit". En språkmodell kan inte göra
en entry skarpare genom att tycka — skärpan kommer från nivåer man kan peka
på: volatiliteten (ATR), senaste swing-low, EMA50, och target som en multipel
av den risk man faktiskt tar.

Modulen föreslår flera stoppnivåer och säger vad var och en KOSTAR i risk och
vad den kräver av targeten för att nå strategins R:R-krav. Den väljer inte åt
dig. Att se att en ATR-stop ger 8 % risk medan swing-low ger 4 % är själva
beslutsunderlaget; AI:n kommenterar valet, den gör det inte.

Rena funktioner, ingen Streamlit och inget nätverk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

ATR_MULT = 2.0            # standardavstånd i ATR till stoppen
MA_BUFFER_PCT = 1.0       # hur långt under EMA50 stoppen läggs
SWING_BUFFER_PCT = 0.5    # marginal under swing-low, så bruset inte plockar den

RR_MIN = 2.0              # panelens krav
RR_PREFERRED = 3.0        # guidens "helst 1:3"

# Namn på nivåerna, så UI och prompt talar samma språk.
ATR_STOP = "ATR-stop"
SWING_STOP = "Swing-low"
MA_STOP = "EMA50"
PCT_STOP = "Fast procent"


@dataclass(frozen=True)
class StopLevel:
    name: str
    price: float
    risk_pct: float           # avstånd från entry, i procent
    why: str
    target_for_min_rr: float  # target som krävs för RR_MIN
    target_for_pref_rr: float


@dataclass(frozen=True)
class Assessment:
    """Vad de valda nivåerna faktiskt innebär."""
    rr: Optional[float]
    risk_pct: Optional[float]
    reward_pct: Optional[float]
    meets_min: bool
    meets_preferred: bool
    notes: tuple


def _f(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _pos(value) -> Optional[float]:
    v = _f(value)
    return v if v is not None and v > 0 else None


def risk_pct(entry, stop) -> Optional[float]:
    """Avståndet till stoppen i procent av entry."""
    e, s = _pos(entry), _pos(stop)
    if e is None or s is None:
        return None
    return round(abs(e - s) / e * 100, 4)


def rr(entry, stop, target) -> Optional[float]:
    """Reward mot risk. None när stoppen ligger på entry — då finns ingen risk
    att dividera med, och 0 hade lästs som 'dåligt R:R' i stället för
    'ofullständigt'."""
    e, s, t = _pos(entry), _pos(stop), _pos(target)
    if e is None or s is None or t is None or e == s:
        return None
    return round(abs(t - e) / abs(e - s), 4)


def target_for_rr(entry, stop, ratio: float) -> Optional[float]:
    """Vilken target som ger exakt det R:R-förhållandet.

    Riktningen följer stoppen: ligger stoppen under entry är det en lång
    position och targeten ligger över.
    """
    e, s = _pos(entry), _pos(stop)
    if e is None or s is None or e == s:
        return None
    distance = abs(e - s) * ratio
    return round(e + distance if s < e else e - distance, 4)


def stop_candidates(entry, snap=None, fixed_pct: Optional[float] = None,
                    atr_mult: Optional[float] = None) -> list:
    """Stoppnivåerna som går att räkna, dyraste risk först.

    Utan ögonblicksbild finns bara den fasta procenten — då säger listan det
    i stället för att hitta på nivåer ur ingenting.

    atr_mult kommer från strategin där den har en: Viking 1,5×, Wolf 2,5×.
    Saknas den används ATR_MULT, och namnet på nivån visar vilken som gällde.
    """
    mult = _pos(atr_mult) or ATR_MULT
    e = _pos(entry)
    if e is None:
        return []

    out = []

    def _add(name, price, why):
        p = _pos(price)
        if p is None or p >= e:      # en stop över entry är inte en stop
            return
        out.append(StopLevel(
            name=name, price=round(p, 4), risk_pct=risk_pct(e, p) or 0.0,
            why=why,
            target_for_min_rr=target_for_rr(e, p, RR_MIN) or 0.0,
            target_for_pref_rr=target_for_rr(e, p, RR_PREFERRED) or 0.0))

    if snap is not None and getattr(snap, "atr14", None):
        _add(f"{ATR_STOP} {mult:g}×", e - mult * snap.atr14,
             f"{mult:g}× ATR({snap.atr14:.2f}) under entry — stoppen sitter "
             f"utanför normalt dagsbrus.")
    if snap is not None and getattr(snap, "swing_low_20", None):
        _add(SWING_STOP, snap.swing_low_20 * (1 - SWING_BUFFER_PCT / 100),
             f"{SWING_BUFFER_PCT:g} % under 20-dagars swing-low "
             f"({snap.swing_low_20:.2f}) — bryts den är strukturen bruten.")
    if snap is not None and getattr(snap, "ema50", None):
        _add(MA_STOP, snap.ema50 * (1 - MA_BUFFER_PCT / 100),
             f"{MA_BUFFER_PCT:g} % under EMA50 ({snap.ema50:.2f}) — samma nivå "
             f"som swingens säljregel.")
    if fixed_pct:
        _add(PCT_STOP, e * (1 - abs(fixed_pct) / 100),
             f"Strategins fasta {abs(fixed_pct):g} % under entry.")

    return sorted(out, key=lambda s: -s.risk_pct)


def assess(entry, stop, target, snap=None) -> Assessment:
    """Vad de valda nivåerna innebär, med invändningarna utskrivna."""
    ratio = rr(entry, stop, target)
    r_pct = risk_pct(entry, stop)
    e, t = _pos(entry), _pos(target)
    reward = round(abs(t - e) / e * 100, 4) if (e and t) else None

    notes = []
    if ratio is None:
        notes.append("R:R går inte att räkna — entry, stop och target måste "
                     "alla vara satta, och stoppen får inte ligga på entry.")
    elif ratio < RR_MIN:
        need = target_for_rr(entry, stop, RR_MIN)
        notes.append(f"R:R {ratio:.1f}x är under kravet {RR_MIN:g}. "
                     f"Target måste till {need:g} — eller stoppen närmare.")
    elif ratio < RR_PREFERRED:
        need = target_for_rr(entry, stop, RR_PREFERRED)
        notes.append(f"R:R {ratio:.1f}x klarar kravet men inte guidens "
                     f"{RR_PREFERRED:g}. Det kräver target {need:g}.")

    if snap is not None:
        atr = getattr(snap, "atr14", None)
        if atr and r_pct is not None:
            dist_in_atr = abs(_pos(entry) - _pos(stop)) / atr
            if dist_in_atr < 1.0:
                notes.append(f"Stoppen ligger {dist_in_atr:.1f} ATR från entry "
                             f"— innanför normalt dagsbrus. Den plockas av "
                             f"slumpen, inte av att du har fel.")
        low = getattr(snap, "swing_low_20", None)
        s = _pos(stop)
        if low and s and s > low:
            notes.append(f"Stoppen ({s:g}) ligger ÖVER 20-dagars swing-low "
                         f"({low:g}) — strukturen är intakt när du redan är ute.")
        high = getattr(snap, "swing_high_20", None)
        if high and t and t < high:
            notes.append(f"Targeten ({t:g}) ligger under 20-dagars högsta "
                         f"({high:g}) — du tar vinst före förra motståndet.")

    return Assessment(
        rr=ratio, risk_pct=r_pct, reward_pct=reward,
        meets_min=ratio is not None and round(ratio, 6) >= RR_MIN,
        meets_preferred=ratio is not None and round(ratio, 6) >= RR_PREFERRED,
        notes=tuple(notes))
