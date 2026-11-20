"""Build the visual web report (web/us34_safety.html) from the analysis outputs.

Reads data/processed and data/raw, packs what the page needs into JSON and
injects it into web/template.html. Re-run after eda.py, eda_glare_drivers.py
and priorities.py.
"""
import base64
import json
import math

import numpy as np
import pandas as pd

from config import PROCESSED, RAW, ROOT, SEGMENTS
from eda_glare_drivers import load as load_glare

EDA = PROCESSED / "eda"
WEB = ROOT / "web"
CUR = (2019, 2025)


def r(x, n=1):
    return None if pd.isna(x) else round(float(x), n)


def centerline():
    cl = pd.read_csv(RAW / "roadway" / "centerline_tenth_mile.csv")
    cl = cl[(cl.ROUTE == "034A") & cl.REFPT.between(60.9, 92.05)].sort_values("REFPT")
    return cl


def terrain():
    """USGS 3DEP hillshade (Web Mercator) embedded as a data URI, with its
    lon/lat bounds so the page can place it under the map."""
    img = RAW / "terrain" / "hillshade.jpg"
    meta = RAW / "terrain" / "hillshade_meta.json"
    if not img.exists():
        return None
    e = json.loads(meta.read_text())["extent"]
    lon = lambda x: x / 6378137 * 180 / math.pi
    lat = lambda y: math.degrees(2 * math.atan(math.exp(y / 6378137)) - math.pi / 2)
    return {"src": "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode(),
            "w": lon(e["xmin"]), "e": lon(e["xmax"]), "s": lat(e["ymin"]), "n": lat(e["ymax"])}


def places():
    """Town limits (Census TIGERweb incorporated places) and point landmarks
    (data/raw/places/landmarks.json, from OpenStreetMap), for the map."""
    out = {"towns": [], "points": []}
    tb = RAW / "places" / "town_boundaries.geojson"
    if tb.exists():
        for f in json.loads(tb.read_text())["features"]:
            g = f["geometry"]
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            out["towns"].append({"name": f["properties"]["BASENAME"],
                                 "rings": [[[round(x, 5), round(y, 5)] for x, y in ring]
                                           for poly in polys for ring in poly]})
    lm = RAW / "places" / "landmarks.json"
    if lm.exists():
        out["points"] = json.loads(lm.read_text())
    return out


def existing(cl):
    """What is on the road now, for the project views. Coverage differs a lot:
    - shoulders: CDOT Highways Geometrics (2025), every segment, primary-direction outside shoulder
    - survey: Street View spot check, 12 points (rumble strips, chevrons, guardrail)
    - guardrail: OpenStreetMap barrier=guard_rail within 30 m of US 34 (partly mapped)
    - signals: OpenStreetMap highway=traffic_signals, grouped into intersections
    """
    def mp_of(lat, lon):
        d = np.hypot((cl.lat.values - lat) * 111000, (cl.lon.values - lon) * 85000)
        i = d.argmin()
        return float(cl.REFPT.values[i]), float(d[i])

    g = pd.read_csv(RAW / "roadway" / "geometrics.csv")
    g = g[(g.ROUTE == "034A") & (g.ENDREFPT > 60.95) & (g.REFPT < 92.1)].sort_values("REFPT")
    sh = [[round(a, 3), round(b, 3), int(w)] for a, b, w in zip(g.REFPT, g.ENDREFPT, g.PRIOUTSHLDWD.fillna(0))]
    merged = []   # merge runs of the same width so the page draws fewer paths
    for a, b, w in sh:
        if merged and merged[-1][2] == w and abs(merged[-1][1] - a) < 0.002:
            merged[-1][1] = b
        else:
            merged.append([a, b, w])

    sv = pd.read_csv(RAW / "roadway" / "streetview_survey.csv").fillna("")
    survey = sv.to_dict("records")

    rails, signals = [], []
    f = RAW / "places" / "osm_features.json"
    if f.exists():
        for e in json.loads(f.read_text())["elements"]:
            t = e.get("tags", {})
            if t.get("barrier") == "guard_rail":
                pts = [[round(p["lon"], 5), round(p["lat"], 5)] for p in e["geometry"]]
                rails.append({"id": e["id"], "pts": pts, "mp": round(mp_of(pts[0][1], pts[0][0])[0], 1)})
            elif t.get("highway") == "traffic_signals":
                m, dist = mp_of(e["lat"], e["lon"])
                if dist < 120 and 61 <= m <= 92.05:
                    signals.append([e["lat"], e["lon"], m])
    groups = []   # signal heads within ~0.08 mi are one intersection
    for la, lo, m in sorted(signals, key=lambda s: s[2]):
        if groups and m - groups[-1][-1][2] < 0.08:
            groups[-1].append([la, lo, m])
        else:
            groups.append([[la, lo, m]])
    inter = [{"lat": round(float(np.mean([s[0] for s in gp])), 5), "lon": round(float(np.mean([s[1] for s in gp])), 5),
              "mp": round(float(np.mean([s[2] for s in gp])), 2)} for gp in groups]
    return {"shoulders": merged, "survey": survey, "rails": rails, "signals": inter}


def curve_assets(a, b, sv):
    """Chevrons and guardrail seen in the Street View survey within 0.15 mi of a curve
    (chevrons only from points approaching it)."""
    near = sv[sv.mp.between(a - 0.15, b + 0.15)]
    # a chevron sighting only counts from a point approaching the curve in its direction of travel
    approach = near[((near.direction == "EB") & (near.mp <= a + 0.02)) | ((near.direction == "WB") & (near.mp >= b - 0.02))]
    def seen(col):
        v = (approach if col == "chevrons_visible" else near)[col].astype(str).str.lower()
        v = v[~v.isin(["", "nan", "n/a"])]
        return "not checked" if v.empty else "yes" if v.str.startswith("yes").any() else "no"
    return {"chevrons": seen("chevrons_visible"), "guardrail": seen("guardrail")}


def top_curves(t, other):
    """Top-curve records for the page, with what is on the road and whether the
    curve is also in the other ranking."""
    sv = pd.read_csv(RAW / "roadway" / "streetview_survey.csv")
    cols = ["mp_start", "mp_end", "min_radius_ft", "deflection_deg", "crashes",
            "exp_crashes", "ka", "exp_ka", "k", "departure"]
    out = []
    for row in t.itertuples():
        d = {k: getattr(row, k) for k in cols}
        d.update(curve_assets(row.mp_start, row.mp_end, sv))
        d["in_other"] = bool((other.curve_id == row.curve_id).any())
        out.append(d)
    return out


def main():
    cl = centerline()
    line = [[r(m, 1), round(la, 5), round(lo, 5), r(e, 0)]
            for m, la, lo, e in zip(cl.REFPT, cl.lat, cl.lon, cl.elev_m)]

    cr, _ = load_glare()
    c = cr[cr.year.between(*CUR)].copy()
    # crashes without coordinates go on the centerline at their milepost
    lat = np.interp(c.mp, cl.REFPT, cl.lat)
    lon = np.interp(c.mp, cl.REFPT, cl.lon)
    bad = ~(c.lat.abs() > 1) | (c.dist_to_us34_m.fillna(0) > 300)
    c["plat"] = np.where(bad, lat, c.lat)
    c["plon"] = np.where(bad, lon, c.lon)
    sev = c.severity.fillna("O")
    sp = c.animal_species.fillna("")
    crashes = []
    for i, row in enumerate(c.itertuples()):
        flags = ""
        flags += "m" if str(row.motorcycle) == "True" else ""
        flags += "a" if str(row.alcohol) == "True" else ""
        flags += "c" if str(row.on_curve) == "True" else ""
        flags += "g" if bool(row.sun_ahead) else ""
        flags += "w" if row.crash_cat == "Wild animal" else ""
        crashes.append([round(row.plat, 5), round(row.plon, 5), r(row.mp, 2), sev.iloc[i],
                        row.crash_cat, str(row.light_cat), flags,
                        str(sp.iloc[i]) if row.crash_cat == "Wild animal" else "",
                        row.dir if isinstance(row.dir, str) else "", int(row.year),
                        str(row.date)[:10] if isinstance(row.date, str) or pd.notna(row.date) else "",
                        int(row.hour) if pd.notna(row.hour) else -1,
                        str(row.surface_cat) if pd.notna(row.surface_cat) else "",
                        int(row.vehicles) if pd.notna(row.vehicles) else 0,
                        row.segment if isinstance(row.segment, str) else ""])

    seg = pd.read_csv(EDA / "crashes_by_segment.csv")
    segments = [{"name": n, "a": a, "b": b,
                 **seg.set_index("segment").loc[n][["crashes", "ka", "k", "rate_per_mvmt",
                                                    "ka_per_100mvmt", "pct_injury"]].to_dict()}
                for n, a, b in SEGMENTS]
    for d in segments:   # rate from counts: the CSV column is pre-rounded (8.75 vs 8.746)
        row = seg.set_index("segment").loc[d["name"]]
        d["ka_per_100mvmt"] = round(float(row.ka / row.mvmt * 100), 3)
    by_year = pd.read_csv(EDA / "by_year.csv")
    by_year = by_year[by_year.year.between(2007, 2025)]
    by_month = pd.read_csv(EDA / "by_month.csv")
    by_hour = pd.read_csv(EDA / "by_hour.csv")
    per_mile = pd.read_csv(EDA / "per_mile_current.csv")
    pri = pd.read_csv(EDA / "project_priorities.csv")
    pk = pd.read_csv(EDA / "project_packages.csv")
    drv = pd.read_csv(EDA / "drivers_factors.csv")
    gl = pd.read_csv(EDA / "glare_by_segment.csv")
    cv = pd.read_csv(EDA / "curves_inventory.csv")
    cv = cv[(cv.mp_mid >= 65) & (cv.mp_mid < 88)]
    rk_all = pd.read_csv(EDA / "curves_ranked.csv")
    rk_all = rk_all[(rk_all.mp_mid >= 65) & (rk_all.mp_mid < 88)]
    rk = rk_all.head(7)                                   # ranked by fatal + serious, then crashes
    rk_n = rk_all.sort_values(["crashes", "ka", "injury"], ascending=False).head(7)   # by all crashes

    # per-project detail and the positions (in `crashes`) of the crashes each targets
    pos = {idx: i for i, idx in enumerate(c.index)}
    tg = pd.read_csv(EDA / "project_targets.csv")
    det = json.loads((EDA / "project_details.json").read_text())
    pri_cols = ["project", "where", "target_ka_10yr", "reduction_low_pct",
                "reduction_high_pct", "ka_prevented_10yr_low",
                "ka_prevented_10yr_high", "injury_prevented_10yr_low",
                "injury_prevented_10yr_high", "relative_cost", "cost_low", "cost_high",
                "bc_low", "bc_mid", "bc_high", "life_yr", "cost_basis", "basis",
                "eb_adjust_ka", "untreated_share", "status"]
    priorities = pri[pri_cols].fillna("").to_dict("records")
    for p in priorities:
        p.update(det[p["project"]])
        p["crashes"] = sorted(pos[r] for r in tg[tg.project == p["project"]].row if r in pos)

    # fatal + serious per 100M vehicle-miles in 3-mile sections (last one MP 88-92 = Loveland)
    ah = pd.read_csv(RAW / "roadway" / "aadt_history_long.csv")
    ah = ah[ah.year.between(*CUR)]
    edges = [61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 92]
    rate3 = []
    for a, b in zip(edges[:-1], edges[1:]):
        mv = float((ah.aadt * (ah.endRefPt.clip(upper=b) - ah.refPt.clip(lower=a)).clip(lower=0) * 365).sum() / 1e6)
        s = c[(c.mp >= a) & ((c.mp < b) | ((b == 92) & (c.mp <= 92)))]
        ka = int(s.severity.isin(["K", "A"]).sum())
        rate3.append({"a": a, "b": b, "crashes": int(len(s)), "ka": ka, "k": int((s.severity == "K").sum()),
                      "mvmt": round(mv, 1), "rate": round(ka / mv * 100, 1)})
    assert sum(r["ka"] for r in rate3) == int(c.severity.isin(["K", "A"]).sum())

    pf = EDA / "peer_corridors.csv"
    peers = pd.read_csv(pf).to_dict("records") if pf.exists() else []
    data = {
        "terrain": terrain(),
        "places": places(),
        "existing": existing(cl),
        "peers": peers,
        "line": line,
        "crashes": crashes,
        "segments": segments,
        "rate3": rate3,
        "byYear": by_year[["year", "crashes", "ka", "k", "rate_per_mvmt", "ka_per_100mvmt"]]
        .round(2).to_dict("records"),
        "byMonth": by_month[["month", "crashes", "ka", "rate_per_mvmt"]].round(2).to_dict("records"),
        "byHour": by_hour.round(2).to_dict("records"),
        "perMile": per_mile.to_dict("records"),
        "priorities": priorities,
        "packages": pk.to_dict("records"),
        "drivers": drv.round(1).to_dict("records"),
        "glare": gl.round(2).to_dict("records"),
        "curves": [[r(a, 3), r(b, 3), int(rad), int(n), int(ka)] for a, b, rad, n, ka in
                   zip(cv.mp_start, cv.mp_end, cv.min_radius_ft, cv.crashes, cv.ka)],
        "topCurves": top_curves(rk, rk_n),
        "topCurvesAll": top_curves(rk_n, rk),
        "totals": {"crashes": int(len(c)), "ka": int(c.severity.isin(["K", "A"]).sum()),
                   "k": int((c.severity == "K").sum()), "killed": int(c.inj4.sum()),
                   "all_years_crashes": int(cr.year.between(2007, 2026).sum()),
                   "corridor_ka_rate": round(float(seg.set_index("segment").loc["Corridor", "ka"]
                                                    / seg.set_index("segment").loc["Corridor", "mvmt"] * 100), 3)},
    }
    tpl = (WEB / "template.html").read_text()
    out = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (WEB / "us34_safety.html").write_text(out)
    # GitHub Pages serves the docs/ folder
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(out)
    print("wrote", WEB / "us34_safety.html", f"{len(out)/1e3:.0f} kB",
          "crashes", len(crashes), "killed", data["totals"]["killed"])


if __name__ == "__main__":
    main()
