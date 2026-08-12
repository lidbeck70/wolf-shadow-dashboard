#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genererar screener- och regimdata för Wolfpanelen från Börsdata-API:et.

Körs på din maskin (samma cache-mönster som backtestern), 1 gång/vecka:

    python wolf_data.py

Skriver wolf_screener.json + wolf_regime.json till OUTPUT_DIR (panelens
public-/data-mapp). Sätt PUSH_TO_GIST=True (+ GIST_ID/GITHUB_TOKEN) för att
även lägga filerna i samma Gist som holdings/swing — då ser Streamlit Cloud-
flikarna datan utan att du behöver committa JSON.
"""

import os
import sys
import time
import json
import datetime as dt

import pandas as pd
import requests

CONFIG = {
    "API_KEY": os.environ.get("BORSDATA_API_KEY", "DIN_API_NYCKEL_HAR"),
    "CACHE_DIR": os.environ.get("WOLF_CACHE_DIR", "bd_cache"),    # kan peka på backtesterns cache
    "OUTPUT_DIR": os.environ.get("WOLF_OUTPUT_DIR", "public"),    # panelens public-/data-mapp
    "UNIVERSE_CSV": os.environ.get("WOLF_UNIVERSE_CSV", "universe.csv"),  # export från Börsdata-screenern
    "HISTORY_DAYS": 320,                  # räcker för MA200 + 6-mån momentum
    "MIN_MCAP_NOTE": "Screena börsvärde/F-score i Börsdata; detta är rankinglagret",

    # Strategiparametrar (matchar swing-reglerna)
    "MOM_SHORT": 63, "MOM_LONG": 126,
    "MOM_LONG_MIN": 0.10,
    "TOP_N": 20, "RANK_EXIT": 40,
    "NEAR_MA_PCT": 0.02,                  # ±2 % = "nära MA20/MA50"
    "NEAR_HIGH_PCT": 0.03,                # inom 3 % av 52v-högsta
    "INDEX_NAME_HINT": "OMX Stockholm PI",

    # Valfri Gist-publicering (samma Gist som holdings/swing).
    "PUSH_TO_GIST": bool(os.environ.get("WOLF_GIST_ID")),
    "GIST_ID": os.environ.get("WOLF_GIST_ID", ""),
    "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
}

BASE = "https://apiservice.borsdata.se/v1"
_last = [0.0]


def _get(ep, params=None):
    p = dict(params or {}); p["authKey"] = CONFIG["API_KEY"]
    w = 0.12 - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    r = requests.get(f"{BASE}{ep}", params=p, timeout=30)
    _last[0] = time.time()
    if r.status_code == 429:
        time.sleep(10); return _get(ep, params)
    r.raise_for_status(); return r.json()


def get_prices(ins_id, start, end):
    os.makedirs(CONFIG["CACHE_DIR"], exist_ok=True)
    fp = os.path.join(CONFIG["CACHE_DIR"], f"{ins_id}.csv")
    fresh = os.path.exists(fp) and (time.time() - os.path.getmtime(fp)) / 86400 < 3
    if fresh:
        df = pd.read_csv(fp, parse_dates=["d"])
        if len(df):
            return df
    data = _get(f"/instruments/{ins_id}/stockprices",
                {"from": start, "to": end, "maxCount": 20000})
    rows = data.get("stockPricesList", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows); df["d"] = pd.to_datetime(df["d"])
    df = df.sort_values("d").drop_duplicates("d")
    df.to_csv(fp, index=False); return df


def build_universe():
    ins = pd.DataFrame(_get("/instruments")["instruments"])
    if "instrument" in ins.columns:
        ins = ins[ins["instrument"] == 0]
    csvp = CONFIG["UNIVERSE_CSV"]
    if csvp and os.path.exists(csvp):
        uni = pd.read_csv(csvp, sep=None, engine="python")
        col = next((c for c in uni.columns if c.lower() in
                    ("ticker", "kortnamn", "symbol")), uni.columns[0])
        wanted = set(str(t).strip().upper() for t in uni[col].dropna())
        ins = ins[ins["ticker"].astype(str).str.upper().isin(wanted)]
        print(f"Universum: {len(ins)} bolag från {csvp}")
    else:
        print(f"VARNING: ingen universe.csv — kör alla {len(ins)} (långsamt). "
              "Exportera swing-screenern från Börsdata för bäst resultat.")
    return ins[["insId", "name", "ticker"]].reset_index(drop=True)


def find_index_id():
    # 1) Direct override — most reliable. Look up OMXSPI's insId once in Börsdata
    #    and set WOLF_INDEX_ID; skips the fragile name lookup entirely.
    env_id = os.environ.get("WOLF_INDEX_ID", "").strip()
    if env_id.isdigit():
        print(f"Index: använder WOLF_INDEX_ID={env_id}")
        return int(env_id)
    # 2) Name lookup against the indexes endpoint.
    try:
        idx = pd.DataFrame(_get("/instruments/indexes").get("indexes", []))
        if len(idx) and "name" in idx.columns:
            for hint in (CONFIG["INDEX_NAME_HINT"], "OMXSPI", "OMX Stockholm"):
                hit = idx[idx["name"].str.contains(hint, case=False, na=False)]
                if len(hit):
                    print(f"Index: matchade '{hint}' -> insId {int(hit.iloc[0]['insId'])}")
                    return int(hit.iloc[0]["insId"])
            # No match — surface what's available so the hint/id can be fixed.
            names = [str(n) for n in idx["name"].head(30).tolist()]
            print("VARNING: OMXSPI hittades inte. Tillgängliga index (första 30):")
            for n in names:
                print("   -", n)
        else:
            print("VARNING: index-endpointen gav ingen data (licens?).")
    except Exception as e:
        print("VARNING: kunde inte hämta index:", e)
    print("Fix: sätt WOLF_INDEX_ID till OMXSPI:s insId (secret/env), "
          "eller justera INDEX_NAME_HINT. Regimfil får index=null tills dess.")
    return None


def metrics_for(closes):
    """Alla nyckeltal ur en prisserie. Returnerar None om för kort historik."""
    c = closes.dropna()
    if len(c) < 210:
        return None
    px = float(c.iloc[-1])
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma50 = float(c.rolling(50).mean().iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    r3 = px / float(c.iloc[-CONFIG["MOM_SHORT"]]) - 1 if len(c) > CONFIG["MOM_SHORT"] else None
    r6 = px / float(c.iloc[-CONFIG["MOM_LONG"]]) - 1 if len(c) > CONFIG["MOM_LONG"] else None
    hi52 = float(c.iloc[-252:].max()) if len(c) >= 252 else float(c.max())
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    dn = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rsi = float(100 - 100 / (1 + up / dn)) if dn > 0 else 100.0
    return dict(px=px, ma20=ma20, ma50=ma50, ma200=ma200,
                r3=r3, r6=r6, hi52=hi52, rsi=rsi)


def classify_regime(idx_block, breadth):
    """Regim + regelverk ur index-blocket och marknadsbredden. Ren funktion."""
    if idx_block is None:
        return "OKÄND", ["Index saknas — kör manuell koll av OMXSPI vs MA200."]
    if not idx_block["above"]:
        return "RÖD", ["INGA nya swingköp.",
                       "Hantera endast exits (MA50 / stop / rank).",
                       "Kontrarisk sida: notera vilka råvaruscreeners som växer — botten byggs nu."]
    if breadth < 0.45 or idx_block["dist"] < 0.02:
        return "GUL", ["Nya köp tillåtna men HALV positionsstorlek.",
                       "Max 1 nytt köp/vecka. Endast setup A (pullback) — inga utbrottsjakter.",
                       "Bredden dör före index: var beredd på RÖD."]
    return "GRÖN", ["Full positionsstorlek enligt reglerna (12–20 %).",
                    "Max 1–2 nya köp/vecka, setup A eller B.",
                    "Vanlig veckorutin — och rör inte det som fungerar."]


def _push_to_gist(files: dict) -> bool:
    """Valfritt: lägg JSON-filerna i Gist:en så Cloud-flikarna ser datan."""
    if not CONFIG["PUSH_TO_GIST"] or not CONFIG["GIST_ID"] or not CONFIG["GITHUB_TOKEN"]:
        return False
    token = CONFIG["GITHUB_TOKEN"].strip()
    prefix = "Bearer" if token.startswith("github_pat_") else "token"
    headers = {"Authorization": f"{prefix} {token}",
               "Accept": "application/vnd.github.v3+json"}
    payload = {"files": {n: {"content": c} for n, c in files.items()}}
    try:
        r = requests.patch(f"https://api.github.com/gists/{CONFIG['GIST_ID']}",
                           headers=headers, json=payload, timeout=30)
        ok = r.status_code == 200
        print("Gist-push:", "OK" if ok else f"fel {r.status_code}")
        return ok
    except Exception as e:
        print("Gist-push misslyckades:", e); return False


def _write(name, obj):
    path = os.path.join(CONFIG["OUTPUT_DIR"], name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return json.dumps(obj, ensure_ascii=False, indent=1)


def main():
    if CONFIG["API_KEY"] == "DIN_API_NYCKEL_HAR":
        sys.exit("Sätt API_KEY (eller miljövariabeln BORSDATA_API_KEY) först.")
    end = dt.date.today().isoformat()
    start = (dt.date.today() - dt.timedelta(days=CONFIG["HISTORY_DAYS"] + 120)).isoformat()
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    uni = build_universe()
    rows, above200 = [], 0
    for i, r in uni.iterrows():
        df = get_prices(r["insId"], start, end)
        if not len(df):
            continue
        m = metrics_for(df.set_index("d")["c"].astype(float))
        if m is None or m["r3"] is None or m["r6"] is None:
            continue
        in_trend = m["px"] > m["ma200"]
        if in_trend:
            above200 += 1
        qualifies = in_trend and m["r3"] > 0 and m["r6"] > CONFIG["MOM_LONG_MIN"]
        near_ma = (abs(m["px"] / m["ma20"] - 1) <= CONFIG["NEAR_MA_PCT"] or
                   abs(m["px"] / m["ma50"] - 1) <= CONFIG["NEAR_MA_PCT"])
        near_high = m["px"] >= m["hi52"] * (1 - CONFIG["NEAR_HIGH_PCT"])
        rows.append(dict(
            ticker=r["ticker"], name=r["name"],
            price=round(m["px"], 2), mom3=round(m["r3"], 4), mom6=round(m["r6"], 4),
            score=round(0.5 * m["r3"] + 0.5 * m["r6"], 4),
            rsi=round(m["rsi"], 1),
            dist_ma20=round(m["px"] / m["ma20"] - 1, 4),
            dist_ma50=round(m["px"] / m["ma50"] - 1, 4),
            above_ma200=in_trend, qualifies=qualifies,
            setupA=bool(qualifies and near_ma and 35 <= m["rsi"] <= 55),
            nearMA=bool(near_ma), nearHigh=bool(near_high),
        ))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(uni)}")

    allrows = pd.DataFrame(rows)
    ranked = allrows[allrows.qualifies].sort_values("score", ascending=False)
    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    top = ranked.head(CONFIG["RANK_EXIT"])

    screener = dict(
        generated=dt.datetime.now().isoformat(timespec="minutes"),
        universe_size=len(allrows), qualifying=len(ranked),
        top=top.to_dict(orient="records"),
        note=CONFIG["MIN_MCAP_NOTE"],
    )
    screener_str = _write("wolf_screener.json", screener)

    # ---- Regim ----
    idx_block = None
    idx_id = find_index_id()
    if idx_id:
        idf = get_prices(idx_id, start, end)
        if len(idf):
            c = idf.set_index("d")["c"].astype(float)
            ma200 = c.rolling(200).mean()
            idx_block = dict(
                close=round(float(c.iloc[-1]), 1),
                ma200=round(float(ma200.iloc[-1]), 1),
                above=bool(c.iloc[-1] > ma200.iloc[-1]),
                dist=round(float(c.iloc[-1] / ma200.iloc[-1] - 1), 4),
                spark=[round(float(v), 1) for v in c.iloc[-30:].tolist()],
            )
    breadth = round(above200 / max(len(allrows), 1), 4)
    regime, rules = classify_regime(idx_block, breadth)

    hist_fp = os.path.join(CONFIG["OUTPUT_DIR"], "wolf_regime_history.json")
    hist = []
    if os.path.exists(hist_fp):
        try:
            hist = json.load(open(hist_fp, encoding="utf-8"))
        except Exception:
            hist = []
    hist.append(dict(date=end, qualifying=len(ranked), breadth=breadth, regime=regime))
    hist = hist[-26:]  # ett halvårs veckohistorik
    json.dump(hist, open(hist_fp, "w", encoding="utf-8"), indent=1)

    regime_json = dict(
        generated=screener["generated"], regime=regime, rules=rules,
        index=idx_block, breadth=breadth,
        qualifying=len(ranked), universe=len(allrows), history=hist,
    )
    regime_str = _write("wolf_regime.json", regime_json)

    _push_to_gist({"wolf_screener.json": screener_str, "wolf_regime.json": regime_str})

    print(f"KLART: regim={regime}, bredd={breadth:.0%}, "
          f"{len(ranked)} kvalar, topp {len(top)} skrivna.")


if __name__ == "__main__":
    main()
