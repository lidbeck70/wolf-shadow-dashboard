"""
fiat_debasement/alerts.py — underlaget för larmbenet "fiat" (alert_scan.py):
dagens nyckeltal per valuta och guld/silver-kvoten. Själva reglerna och
övergångarna finns i alert_rules.fiat_alerts.
"""

from __future__ import annotations

from typing import Optional

from fiat_debasement import config as cfg
from fiat_debasement import data as fd
from fiat_debasement import snapshot as fs

METRICS = ("m2_yoy", "cpi_yoy", "monetary_gap", "gold_1y")


def collect(snapshot_fn=None, asset_loader=None) -> Optional[dict]:
    """{"metrics": {valuta: {mått: värde|None}}, "gs_ratio": float|None, "dates": {...}} —
    None om ingenting alls gick att räkna (benet fryser då sin baslinje)."""
    snap = snapshot_fn or fs.snapshot
    load_asset = asset_loader or fd.load_asset
    out = {"metrics": {}, "dates": {}, "gs_ratio": None}
    for ccy in cfg.CURRENCIES:
        s = snap(ccy)
        out["metrics"][ccy] = {k: s.get(k).value for k in METRICS}
        out["dates"][ccy] = {k: s.get(k).as_of for k in METRICS}
    try:
        from gold_silver import engine as ge
        g, s = load_asset(cfg.GOLD), load_asset(cfg.SILVER)
        if g.ok and s.ok:
            ratio = ge.ratio_series(g.values, s.values)
            if len(ratio):
                out["gs_ratio"] = round(float(ratio.iloc[-1]), 2)
                out["dates"]["gs_ratio"] = str(ratio.index[-1].date())
    except Exception:
        pass
    anything = out["gs_ratio"] is not None or any(v is not None for m in out["metrics"].values() for v in m.values())
    return out if anything else None
