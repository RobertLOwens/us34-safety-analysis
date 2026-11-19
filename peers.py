"""Peer comparison: US 34 Big Thompson Canyon against other Colorado
two-lane mountain canyon highways, 2019-2025.

Same sources and method for every corridor:
  - crashes: CDOT statewide crash listings (scanned once for the peer routes,
    cached in data/raw/crashes/peers_<file>.csv)
  - traffic: CDOT AADT All Years layer, length-weighted by year -> VMT
  - rate: fatal + serious (K+A) crashes per 100 million vehicle-miles, with a
    95% Poisson interval

Output: data/processed/eda/peer_corridors.csv
Usage:  python peers.py [--rescan]
"""
import sys
from math import exp, factorial

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import config
from arcgis import query
from collect_crashes import COLS, crash_files, normalise

YEARS = (2019, 2025)
CACHE = config.RAW / "crashes"
AGOL = "https://services.arcgis.com/yzB9WM8W0BO3Ql7d/ArcGIS/rest/services"

# name, CDOT route, MP from, MP to, route number in crash listings, (lat_min, lat_max, lon_min, lon_max)
PEERS = [
    ("US 34 Big Thompson Canyon", "034A", 65.0, 83.5, 34, (40.35, 40.46, -105.50, -105.15)),
    ("US 36 Estes Park to Lyons", "036B", 2.0, 19.8, 36, (40.18, 40.40, -105.52, -105.25)),
    ("SH 7 Estes Park to Lyons", "007A", 1.0, 32.5, 7, (40.10, 40.40, -105.60, -105.25)),
    ("SH 119 Boulder Canyon", "119A", 26.5, 41.0, 119, (39.93, 40.05, -105.55, -105.27)),
    ("US 6 Clear Creek Canyon", "006G", 260.5, 271.5, 6, (39.70, 39.80, -105.45, -105.20)),
    ("SH 14 Poudre Canyon", "014B", 90.0, 121.5, 14, (40.55, 40.80, -105.80, -105.15)),
    ("SH 72 Coal Creek Canyon", "072A", 11.0, 29.0, 72, (39.85, 39.95, -105.55, -105.20)),
]
ROUTE_NUMS = {p[4] for p in PEERS}


def _num(v):
    """Route number from '34', '34.0' (2007-2020 files) or '034A' (2021+)."""
    import re
    m = re.match(r"\s*0*(\d+)", str(v))
    return int(m.group(1)) if m else None


def scan(path, rescan):
    cache = CACHE / f"peers_{path.stem}.csv"
    if cache.exists() and not rescan:
        return pd.read_csv(cache, dtype=str)
    print(f"scanning {path.name} ...", flush=True)
    wb = load_workbook(path, read_only=True)
    rows = wb[wb.sheetnames[0]].iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else f"col{i}" for i, h in enumerate(next(rows))]
    low = [h.lower() for h in header]
    i_route = next(low.index(n) for n in COLS["route"] if n in low)
    keep = [r for r in rows if _num(r[i_route]) in ROUTE_NUMS]
    wb.close()
    df = pd.DataFrame(keep, columns=header).astype(str)
    df.insert(0, "source_file", path.name)
    df.to_csv(cache, index=False)
    print(f"  kept {len(df):,}", flush=True)
    return df


def crashes(rescan=False):
    files = [p for p in crash_files() if p.suffix == ".xlsx"
             and YEARS[0] <= int(p.stem[-4:]) <= YEARS[1]]
    df = pd.concat([normalise(scan(p, rescan)) for p in files], ignore_index=True)
    df["rnum"] = df.route.map(_num)
    # exact CDOT route id: '034A' (2021+) or number + section letter (older files)
    r = df.route.astype(str).str.strip().str.upper()
    sec = df.section.astype(str).str.strip().str.upper()
    old = ~r.str.fullmatch(r"\d+[A-Z]") & sec.str.fullmatch(r"[A-Z]")
    df["route_id"] = np.where(old, df.rnum.map(lambda n: f"{n:03d}" if n is not None else "") + sec, r)
    # state highways only (city-street codes share the route column in 2021+ files)
    sysc = df.system.astype(str).str.upper()
    df = df[sysc.str.contains("STATE") & ~sysc.str.contains("FRONTAGE")]
    return df


def vmt(route, a, b):
    """Vehicle-miles 2019-2025 from the AADT All Years layer."""
    d = query(f"{AGOL}/AADT_All_Years_Data/FeatureServer/0",
              where=f"route='{route}' AND endRefPt>{a} AND refPt<{b}", geometry=False)
    d["len"] = d.endRefPt.clip(upper=b) - d.refPt.clip(lower=a)
    d = d[d.len > 0]
    total, aadt = 0.0, []
    for y in range(YEARS[0], YEARS[1] + 1):
        names = {f"AADT{y}", f"AADT{y % 100:02d}"}          # columns are AADT19 ... AADT25
        col = next((c for c in d.columns if c.upper() in names), None)
        if col is None:            # newest year missing -> carry last year forward
            prev = {f"AADT{y - 1}", f"AADT{(y - 1) % 100:02d}"}
            col = next(c for c in d.columns if c.upper() in prev)
        v = pd.to_numeric(d[col], errors="coerce")
        total += (v * d.len).sum() * (366 if y % 4 == 0 else 365)
        aadt.append((v * d.len).sum() / d.len.sum())
    return total, round(float(np.mean(aadt))), round(d.len.sum(), 1)


def pois_ci(k, z=1.96):
    """Approximate 95% interval for a Poisson count (Byar's method)."""
    if k == 0:
        return 0.0, 3.69
    lo = k * (1 - 1 / (9 * k) - z / (3 * np.sqrt(k))) ** 3
    k1 = k + 1
    hi = k1 * (1 - 1 / (9 * k1) + z / (3 * np.sqrt(k1))) ** 3
    return lo, hi


def main():
    cr = crashes("--rescan" in sys.argv)
    cr = cr[cr.year.between(*YEARS)]
    rows = []
    for name, route, a, b, rn, (la0, la1, lo0, lo1) in PEERS:
        s = cr[(cr.route_id == route) & cr.mp_reported.between(a, b)]
        # drop rows whose coordinates put them somewhere else with the same number
        bad = s.lat.notna() & ~(s.lat.between(la0, la1) & s.lon.between(lo0, lo1))
        s = s[~bad]
        v, aadt, miles = vmt(route, a, b)
        ka = int(s.severity.isin(["K", "A"]).sum())
        k = int((s.severity == "K").sum())
        lo, hi = pois_ci(ka)
        rows.append({"corridor": name, "route": route, "mp_from": a, "mp_to": b, "miles": miles,
                     "aadt_avg": aadt, "crashes": len(s), "ka": ka, "k": k,
                     "crash_rate_per_mvmt": round(len(s) / v * 1e6, 2),
                     "ka_per_100mvmt": round(ka / v * 1e8, 1),
                     "ka_per_100mvmt_lo": round(lo / v * 1e8, 1),
                     "ka_per_100mvmt_hi": round(hi / v * 1e8, 1),
                     "ka_per_mile_yr": round(ka / miles / (YEARS[1] - YEARS[0] + 1), 2),
                     "pct_injury": round(100 * s.severity.isin(["K", "A", "B", "C"]).mean(), 1),
                     "dropped_by_coords": int(bad.sum())})
    out = pd.DataFrame(rows).sort_values("ka_per_100mvmt", ascending=False)
    out.to_csv(config.PROCESSED / "eda" / "peer_corridors.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
