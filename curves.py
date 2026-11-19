"""Curve-level crash analysis for US 34, MP 61-92.

1. Curve inventory from the CDOT Highways polyline (data/raw/roadway/
   route_polyline.json, ~2,000 vertices): resample every 10 m with a milepost
   for each point, estimate radius from the heading change over a 60 m
   window, and call a run of points with radius < CURVE_R_FT a curve.
2. Crash position: project the crash's lat/lon onto the polyline when it lies
   within 60 m of the road (more precise), else use the reported milepost.
3. A crash belongs to a curve if it falls within the curve or APPROACH_MI of
   either end (crash locations are often rounded to 0.1 mi).
4. Rank curves by fatal + serious crashes, then all crashes, 2019-2025 (after
   the post-2013-flood canyon rebuild, so geometry matches today's road).

Outputs (data/processed/eda/): curves_inventory.csv, curves_ranked.csv,
curves_by_section.csv, curves_crash_positions.csv
"""
import json

import numpy as np
import pandas as pd

from config import PROCESSED, RAW, SEGMENTS

EDA = PROCESSED / "eda"
YEARS = (2019, 2025)
N_YEARS = YEARS[1] - YEARS[0] + 1
STEP_M = 10
WIN_M = 60
CURVE_R_FT = 1000          # radius below this = curve
SHARP_R_FT = 400
MIN_DEFLECT = 20           # degrees; ignores small bends
APPROACH_MI = 0.05
M_PER_FT = 0.3048
R_EARTH = 6371000


def xy(lon, lat, lat0):
    return (np.radians(lon) * R_EARTH * np.cos(np.radians(lat0)), np.radians(lat) * R_EARTH)


def polyline():
    d = pd.read_json(RAW / "roadway" / "route_polyline.json").sort_values("REFPT")
    pts = []
    for _, s in d.iterrows():
        c = np.array(s.coords)
        lat0 = 40.4
        x, y = xy(c[:, 0], c[:, 1], lat0)
        seg = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        frac = seg / seg[-1] if seg[-1] > 0 else seg
        mp = s.REFPT + frac * (s.ENDREFPT - s.REFPT)
        for i in range(len(c)):
            if pts and i == 0:
                continue
            pts.append((mp[i], c[i, 1], c[i, 0], x[i], y[i]))
    p = pd.DataFrame(pts, columns=["mp", "lat", "lon", "x", "y"]).drop_duplicates("mp")
    # resample every STEP_M along the line
    s = np.r_[0, np.cumsum(np.hypot(np.diff(p.x), np.diff(p.y)))]
    si = np.arange(0, s[-1], STEP_M)
    r = pd.DataFrame({k: np.interp(si, s, p[k]) for k in ["mp", "lat", "lon", "x", "y"]})
    r["s"] = si
    return r


def radius(r):
    h = np.degrees(np.arctan2(np.diff(r.y), np.diff(r.x)))
    h = np.r_[h, h[-1]]
    hu = np.degrees(np.unwrap(np.radians(h)))
    k = WIN_M // STEP_M // 2
    dh = pd.Series(hu).shift(-k) - pd.Series(hu).shift(k)       # deg over WIN_M
    r["heading"] = (90 - h) % 360                                 # compass bearing
    r["turn_deg_per_win"] = dh
    rad = WIN_M / np.radians(dh.abs().clip(lower=1e-3))
    r["radius_ft"] = (rad / M_PER_FT).clip(upper=20000)
    r["turn"] = np.sign(dh)                                       # + = left (CCW) heading EB
    return r


def find_curves(r):
    curvy = (r.radius_ft < CURVE_R_FT).values
    out, i = [], 0
    while i < len(r):
        if not curvy[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(r) and curvy[j + 1] and r.turn.iloc[j + 1] == r.turn.iloc[i]:
            j += 1
        a, b = r.iloc[i], r.iloc[j]
        seg = r.iloc[i:j + 1]
        deflect = abs(np.degrees(np.unwrap(np.radians([0, 0]))).sum())  # placeholder
        h = np.unwrap(np.radians(seg.heading.values))
        deflect = abs(np.degrees(h[-1] - h[0])) + abs(seg.turn_deg_per_win.iloc[[0, -1]]).mean() / 2
        out.append({"mp_start": round(a.mp, 3), "mp_end": round(b.mp, 3),
                    "length_ft": round((b.s - a.s + STEP_M) / M_PER_FT),
                    "min_radius_ft": round(seg.radius_ft.min()),
                    "deflection_deg": round(deflect),
                    "turn_eb": "left" if a.turn > 0 else "right",
                    "lat": round(seg.lat.iloc[len(seg) // 2], 5),
                    "lon": round(seg.lon.iloc[len(seg) // 2], 5)})
        i = j + 1
    c = pd.DataFrame(out)
    c = c[(c.deflection_deg >= MIN_DEFLECT) & (c.mp_start >= 61) & (c.mp_end <= 92)].reset_index(drop=True)
    c["mp_mid"] = ((c.mp_start + c.mp_end) / 2).round(2)
    c["sharp"] = c.min_radius_ft < SHARP_R_FT
    c["section"] = c.mp_mid.map(section_of)
    return c


def section_of(mp):
    for n, a, b in SEGMENTS:
        if a <= mp < b:
            return n
    return SEGMENTS[-1][0]


def crash_positions(r):
    c = pd.read_csv(PROCESSED / "crashes_us34.csv", low_memory=False)
    c = c[c.year.between(*YEARS) & c.mp.between(61, 92)].copy()
    lat0 = 40.4
    rx, ry = r.x.values, r.y.values
    ok = (c.lat.abs() > 1) & c.lon.notna()
    pos = c.mp.astype(float).copy()
    src = pd.Series("reported", index=c.index)
    cx, cy = xy(c.lon[ok].values, c.lat[ok].values, lat0)
    for idx, x, y, mp0 in zip(c.index[ok], cx, cy, c.mp[ok]):
        near = np.abs(r.mp.values - mp0) < 1.0      # stay near the reported mp
        if not near.any():
            continue
        d = np.hypot(rx[near] - x, ry[near] - y)
        k = d.argmin()
        if d[k] < 60:
            pos[idx] = r.mp.values[near][k]
            src[idx] = "gps"
    c["mp_pos"] = pos
    c["pos_src"] = src
    c["ka"] = c.severity.isin(["K", "A"])
    c["k"] = c.severity.eq("K")
    c["inj"] = c.severity.isin(["K", "A", "B", "C"])
    c["departure"] = c.crash_cat.isin(["Fixed object", "Rollover"])
    c["head_on_ss"] = c.crash_cat.isin(["Head-on"]) | (c.crash_cat.eq("Sideswipe") & c.vehicles.ge(2))
    c["moto"] = c.motorcycle.astype(str).eq("True")
    c["curve_coded"] = c.on_curve.astype(str).eq("True")
    c["wet_icy"] = ~c.surface_cat.astype(str).isin(["Dry", "nan"])
    c["dark"] = c.light_cat.astype(str).str.startswith("Dark")
    c["eb"] = c.tu1_direction.astype(str).str.upper().str.match(r"^(E|NE|SE|EAST)")
    return c


def assign(cr, cv):
    cr["curve_id"] = -1
    for i, cu in cv.iterrows():
        m = cr.mp_pos.between(cu.mp_start - APPROACH_MI, cu.mp_end + APPROACH_MI) & (cr.curve_id < 0)
        cr.loc[m, "curve_id"] = i
    return cr


def main():
    r = radius(polyline())
    cv = find_curves(r)
    cv.index.name = "curve_id"
    cr = assign(crash_positions(r), cv)
    g = cr[cr.curve_id >= 0].groupby("curve_id")
    agg = g.agg(crashes=("crash_id", "size"), ka=("ka", "sum"), k=("k", "sum"), injury=("inj", "sum"),
                departure=("departure", "sum"), head_on_ss=("head_on_ss", "sum"), moto=("moto", "sum"),
                wet_icy=("wet_icy", "sum"), dark=("dark", "sum"), eb=("eb", "sum"),
                killed=("inj4", "sum"))
    cv = cv.join(agg).fillna({c: 0 for c in agg.columns})
    for col in agg.columns:
        cv[col] = cv[col].astype(int)
    cv["zone_mi"] = (cv.mp_end - cv.mp_start + 2 * APPROACH_MI).round(3)
    cv.to_csv(EDA / "curves_inventory.csv")

    # rank: severe first, then all crashes; label the curve by its mid milepost
    rk = cv[cv.crashes > 0].sort_values(["ka", "crashes", "injury"], ascending=False).copy()
    rk.insert(0, "rank", range(1, len(rk) + 1))
    rk.to_csv(EDA / "curves_ranked.csv")

    # section summary: share of crashes and KA that are at mapped curves
    rows = []
    for n, a, b in SEGMENTS:
        s = cr[cr.mp_pos.between(a, b, inclusive="left") if n != SEGMENTS[-1][0] else cr.mp_pos.between(a, b)]
        cs = cv[cv.section == n]
        on = s.curve_id >= 0
        rows.append({"section": n, "miles": b - a, "curves": len(cs), "sharp_curves": int(cs.sharp.sum()),
                     "curves_per_mile": round(len(cs) / (b - a), 1),
                     "curve_zone_miles": round(cs.zone_mi.sum(), 1),
                     "crashes": len(s), "crashes_at_curves": int(on.sum()),
                     "ka": int(s.ka.sum()), "ka_at_curves": int(s[on].ka.sum()),
                     "ka_per_curve_mile": round(s[on].ka.sum() / max(cs.zone_mi.sum(), 1e-9), 2),
                     "ka_per_tangent_mile": round(s[~on].ka.sum() / ((b - a) - cs.zone_mi.sum()), 2)})
    sec = pd.DataFrame(rows)
    sec.to_csv(EDA / "curves_by_section.csv", index=False)
    cr[["crash_id", "year", "mp", "mp_pos", "pos_src", "severity", "crash_cat", "road_desc", "curve_id"]].to_csv(
        EDA / "curves_crash_positions.csv", index=False)

    extras(cr, cv)
    pd.set_option("display.width", 250)
    print(f"{len(cv)} curves (R<{CURVE_R_FT} ft, deflection>={MIN_DEFLECT}°); "
          f"{(cr.pos_src == 'gps').mean():.0%} of crashes placed by GPS")
    print(sec.to_string(index=False))
    cols = ["rank", "section", "mp_start", "mp_end", "min_radius_ft", "deflection_deg", "turn_eb", "crashes",
            "ka", "k", "killed", "departure", "head_on_ss", "moto", "wet_icy", "dark"]
    print(rk[cols].head(20).to_string())


def poisson_sf(k, mu):
    """P(X >= k) for Poisson(mu)."""
    from math import exp, factorial
    return 1 - sum(exp(-mu) * mu ** i / factorial(i) for i in range(int(k)))


def extras(cr, cv):
    """Curve types, driver-view turn direction, half-mile hot zones and a
    Poisson check of whether each ranked curve is above the canyon norm."""
    rural = (cr.mp_pos >= 65) & (cr.mp_pos < 88)
    cvr = cv[(cv.mp_mid >= 65) & (cv.mp_mid < 88)].copy()

    # 1. curve type: radius bins, crashes per curve
    cvr["radius_bin"] = pd.cut(cvr.min_radius_ft, [0, 250, 400, 600, 1000],
                               labels=["under 250 ft", "250-400 ft", "400-600 ft", "600-1,000 ft"])
    cvr["deflect_bin"] = pd.cut(cvr.deflection_deg, [0, 45, 90, 360], labels=["20-45°", "45-90°", "over 90°"])
    rows = []
    for col in ["radius_bin", "deflect_bin"]:
        for b, s in cvr.groupby(col, observed=True):
            rows.append({"by": col.replace("_bin", ""), "bin": b, "curves": len(s), "crashes": int(s.crashes.sum()),
                         "ka": int(s.ka.sum()), "crashes_per_curve": round(s.crashes.mean(), 2),
                         "ka_per_curve": round(s.ka.mean(), 2),
                         "departure_per_curve": round(s.departure.mean(), 2),
                         "crashes_per_curve_mile": round(s.crashes.sum() / s.zone_mi.sum(), 1),
                         "ka_per_curve_mile": round(s.ka.sum() / s.zone_mi.sum(), 2)})
    pd.DataFrame(rows).to_csv(EDA / "curves_by_type.csv", index=False)

    # 2. driver's view: is the curve to the driver's left or right?
    at = cr[rural & (cr.curve_id >= 0)].copy()
    d = at.tu1_direction.astype(str).str.upper().str.strip()
    at["dir"] = np.where(d.str.match(r"^(E|NE|SE)"), "EB", np.where(d.str.match(r"^(W|NW|SW)"), "WB", ""))
    at = at[at.dir != ""]
    turn_eb = cv.turn_eb.reindex(at.curve_id).values
    at["driver_turn"] = np.where(at.dir == "EB", turn_eb, np.where(turn_eb == "left", "right", "left"))
    dv = at.groupby(["driver_turn"]).agg(crashes=("crash_id", "size"), ka=("ka", "sum"),
                                         departure=("departure", "sum"), head_on_ss=("head_on_ss", "sum"),
                                         moto=("moto", "sum")).reset_index()
    dv2 = at.groupby(["dir", "driver_turn"]).agg(crashes=("crash_id", "size"), ka=("ka", "sum"),
                                                 departure=("departure", "sum")).reset_index()
    pd.concat([dv.assign(dir="both"), dv2]).to_csv(EDA / "curves_driver_view.csv", index=False)

    # 3. half-mile hot zones (MP 65-88), greedy non-overlapping by KA then crashes
    rc = cr[rural]
    starts = np.round(np.arange(65, 87.51, 0.05), 2)
    w = pd.DataFrame({"mp_start": starts, "mp_end": starts + 0.5})
    w["ka"] = [int(rc[rc.mp_pos.between(a, a + 0.5, inclusive="left")].ka.sum()) for a in starts]
    w["crashes"] = [int(rc.mp_pos.between(a, a + 0.5, inclusive="left").sum()) for a in starts]
    w["k"] = [int(rc[rc.mp_pos.between(a, a + 0.5, inclusive="left")].k.sum()) for a in starts]
    w["curves"] = [int(((cv.mp_mid >= a) & (cv.mp_mid < a + 0.5)).sum()) for a in starts]
    ka_mi = rc.ka.sum() / 23
    w["p_value"] = [round(poisson_sf(k, ka_mi * 0.5), 4) for k in w.ka]
    zones, used = [], np.zeros(len(w), bool)
    for i in w.sort_values(["ka", "crashes"], ascending=False).index:
        a = w.mp_start[i]
        if any(abs(a - z.mp_start) < 0.5 for z in zones):
            continue
        zones.append(w.loc[i])
        if len(zones) == 8:
            break
    zz = pd.DataFrame(zones)
    zz["section"] = zz.mp_start.map(section_of)
    zz.to_csv(EDA / "curves_hot_zones.csv", index=False)

    # 4. per-curve excess over the MP 65-88 curve norm
    cz = cr[rural & (cr.curve_id >= 0)]
    ka_per_zone_mi = cz.ka.sum() / cvr.zone_mi.sum()
    cr_per_zone_mi = len(cz) / cvr.zone_mi.sum()
    rk = pd.read_csv(EDA / "curves_ranked.csv", index_col="curve_id")
    rk["exp_ka"] = (rk.zone_mi * ka_per_zone_mi).round(2)
    rk["exp_crashes"] = (rk.zone_mi * cr_per_zone_mi).round(1)
    rk["p_ka"] = [round(poisson_sf(k, m), 3) for k, m in zip(rk.ka, rk.exp_ka)]
    rk["p_crashes"] = [round(poisson_sf(k, m), 3) for k, m in zip(rk.crashes, rk.exp_crashes)]
    rk["intersection_turn"] = (rk.section == "Estes Park (town)") & (rk.min_radius_ft < 200)
    rk.to_csv(EDA / "curves_ranked.csv")

    pd.set_option("display.width", 250)
    print(pd.read_csv(EDA / "curves_by_type.csv").to_string(index=False))
    print(pd.read_csv(EDA / "curves_driver_view.csv").to_string(index=False))
    print(zz.to_string(index=False))
    print(f"norm: {ka_per_zone_mi:.2f} KA and {cr_per_zone_mi:.1f} crashes per curve-zone mile")
    print(rk[(rk.mp_mid >= 65) & (rk.mp_mid < 88)][["section", "mp_start", "mp_end", "min_radius_ft", "deflection_deg",
              "crashes", "exp_crashes", "p_crashes", "ka", "exp_ka", "p_ka", "departure", "moto"]].head(15).to_string())


if __name__ == "__main__":
    main()
