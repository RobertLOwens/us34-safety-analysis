"""Sun glare by travel direction, and driver factors (drinking, distraction).

Glare: a crash counts as "sun ahead" when the sun is 0-15 degrees above the
horizon and within 30 degrees of the vehicle's heading at that milepost
(heading from the tenth-mile centerline; EB = direction of increasing MP).
Expected share = share of directional traffic (counter 000205 hourly volumes,
weighted by section vehicle-miles) driving with the sun ahead.
Terrain is ignored, so canyon walls that block low sun are not modelled.

Outputs go to data/processed/eda/glare_*.csv and drivers_*.csv.
"""
import numpy as np
import pandas as pd

from config import PROCESSED, RAW, SEGMENTS

OUT = PROCESSED / "eda"
OUT.mkdir(parents=True, exist_ok=True)
YEARS = (2007, 2025)          # glare uses all full years for sample size
CURRENT = (2019, 2025)        # driver factors, matching the rest of the report
SUN_MAX_ALT, AZ_TOL = 15.0, 30.0
LAT, LON = 40.40, -105.30     # corridor midpoint for exposure sun positions


# --- sun position (NOAA approximation, good to ~0.5 degree) ---------------
def sun_alt_az(ts_utc, lat, lon):
    ts = pd.DatetimeIndex(ts_utc)
    doy = ts.dayofyear.values
    hr = ts.hour.values + ts.minute.values / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
           - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hr * 60 + eqt + 4 * np.asarray(lon)
    ha = np.radians(tst / 4 - 180)
    la = np.radians(np.asarray(lat))
    cosz = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    zen = np.arccos(np.clip(cosz, -1, 1))
    alt = 90 - np.degrees(zen)
    az = np.degrees(np.arctan2(np.sin(ha),
                               np.cos(ha) * np.sin(la) - np.tan(dec) * np.cos(la))) + 180
    return alt, az % 360


def ang_diff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def direction(v):
    v = str(v).strip().upper()
    if v in ("E", "EAST", "NE", "NORTHEAST", "SE", "SOUTHEAST"):
        return "EB"
    if v in ("W", "WEST", "NW", "NORTHWEST", "SW", "SOUTHWEST"):
        return "WB"
    return None


def load():
    cr = pd.read_csv(PROCESSED / "crashes_us34.csv", low_memory=False,
                     parse_dates=["datetime"])
    cr = cr[cr.datetime.notna() & cr.mp.notna()].copy()
    cl = pd.read_csv(RAW / "roadway" / "centerline_tenth_mile.csv")
    cl = cl[cl.ROUTE == "034A"].dropna(subset=["bearing_deg"])
    # bearing at REFPT r = heading from r-0.1 to r; use the tenth that contains the crash
    head = cl.set_index(cl.REFPT.round(1))["bearing_deg"]
    key = (np.floor(cr.mp * 10) / 10 + 0.1).round(1).clip(head.index.min(), head.index.max())
    cr["eb_heading"] = head.reindex(key).values
    cr["dir"] = cr.tu1_direction.map(direction)
    cr["heading"] = np.where(cr["dir"] == "WB", (cr.eb_heading + 180) % 360, cr.eb_heading)
    utc = cr.datetime.dt.tz_localize("America/Denver", ambiguous="NaT",
                                     nonexistent="shift_forward").dt.tz_convert("UTC")
    lat = cr.lat.where(cr.lat.abs() > 1, 40.40)
    lon = cr.lon.where(cr.lon.abs() > 1, -105.30)
    ok = utc.notna()
    cr["sun_alt"], cr["sun_az"] = np.nan, np.nan
    alt, az = sun_alt_az(utc[ok].dt.tz_localize(None), lat[ok], lon[ok])
    cr.loc[ok, "sun_alt"], cr.loc[ok, "sun_az"] = alt, az
    cr["sun_ahead"] = ((cr.sun_alt.between(0, SUN_MAX_ALT))
                       & (ang_diff(cr.sun_az, cr.heading) <= AZ_TOL))
    cr["sun_behind"] = ((cr.sun_alt.between(0, SUN_MAX_ALT))
                        & (ang_diff(cr.sun_az, (cr.heading + 180) % 360) <= AZ_TOL))
    cr["sun_part"] = np.where(cr.datetime.dt.hour < 12, "morning", "evening")
    return cr, cl


def exposure(cl):
    """Share of each direction's vehicle-hours driven with low sun ahead."""
    hv = pd.read_csv(RAW / "exposure" / "station_000205_hourly.csv", parse_dates=["date"])
    hv = hv[hv.date.dt.year.between(2013, YEARS[1])]
    prof = (hv.assign(month=hv.date.dt.month)
              .groupby(["month", "hour", "direction"]).volume.mean())
    # section weights = vehicle-miles share; headings sampled every tenth mile
    al = pd.read_csv(RAW / "roadway" / "aadt_history_long.csv")
    w = {}
    for name, a, b in SEGMENTS:
        pts = cl[(cl.REFPT > a) & (cl.REFPT <= b)]
        w[name] = pts.bearing_deg.values
    aadt_cols = [c for c in al.columns if c.lower() in ("aadt", "value")]
    seg_aadt = {"Estes Park (town)": 1.0, "Big Thompson Canyon": 1.0,
                "Canyon mouth to Loveland": 1.0, "Loveland (urban)": 1.0}
    try:
        yr = al[al.columns[al.columns.str.lower() == "year"][0]].max()
        last = al[al[al.columns[al.columns.str.lower() == "year"][0]] == yr]
        mpcol = [c for c in al.columns if "refpt" in c.lower() or c.lower() == "mp"][0]
        for name, a, b in SEGMENTS:
            s = last[(last[mpcol] >= a) & (last[mpcol] < b)][aadt_cols[0]]
            if len(s):
                seg_aadt[name] = s.mean()
    except Exception:
        pass

    times = pd.date_range("2019-01-01", "2025-12-31 23:45", freq="15min", tz="America/Denver")
    alt, az = sun_alt_az(times.tz_convert("UTC").tz_localize(None), LAT, LON)
    low = (alt >= 0) & (alt <= SUN_MAX_ALT)
    t = pd.DataFrame({"month": times.month, "hour": times.hour, "alt": alt, "az": az,
                      "low": low, "morning": times.hour < 12})
    rows, seg_rows = [], []
    for d in ("EB", "WB"):
        vol = prof.xs(d, level="direction").reindex(
            pd.MultiIndex.from_arrays([t.month, t.hour])).values
        tot = ahead_m = ahead_e = 0.0
        for name, a, b in SEGMENTS:
            hd = w[name] if d == "EB" else (w[name] + 180) % 360
            miles = b - a
            frac = np.zeros(len(t))
            for h in hd:
                frac += (t.low.values & (ang_diff(t.az.values, h) <= AZ_TOL))
            frac /= len(hd)
            wt = seg_aadt[name] * miles
            tot += np.nansum(vol) * wt
            ahead_m += np.nansum(vol * frac * t.morning.values) * wt
            ahead_e += np.nansum(vol * frac * ~t.morning.values) * wt
            seg_rows.append({"segment": name, "dir": d,
                             "expected_pct": 100 * np.nansum(vol * frac * (t.morning.values if d == "EB" else ~t.morning.values)) / np.nansum(vol)})
        rows.append({"dir": d, "expected_sun_ahead_morning_pct": 100 * ahead_m / tot,
                     "expected_sun_ahead_evening_pct": 100 * ahead_e / tot})
    return pd.DataFrame(rows), pd.DataFrame(seg_rows)


def glare_tables(cr, cl):
    g = cr[cr.year.between(*YEARS) & cr["dir"].notna()].copy()
    exp, exp_seg = exposure(cl)
    exp = exp.set_index("dir")
    exp_seg = exp_seg.set_index(["segment", "dir"]).expected_pct
    rows = []
    for d in ("EB", "WB"):
        s = g[g["dir"] == d]
        for part in ("morning", "evening"):
            n = int((s.sun_ahead & (s.sun_part == part)).sum())
            rows.append({"dir": d, "sun": part, "crashes_total": len(s),
                         "crashes_sun_ahead": n,
                         "pct_sun_ahead": 100 * n / len(s),
                         "expected_pct": exp.loc[d, f"expected_sun_ahead_{part}_pct"],
                         "ka_sun_ahead": int((s.sun_ahead & (s.sun_part == part)
                                              & s.severity.isin(["K", "A"])).sum())})
    t = pd.DataFrame(rows)
    t["ratio"] = t.pct_sun_ahead / t.expected_pct
    t.to_csv(OUT / "glare_by_direction.csv", index=False)

    # by section: EB morning and WB evening sun-ahead crashes vs the same
    # direction-and-time-of-day crashes with the sun behind (a within-road control)
    sec = []
    for name, *_ in SEGMENTS:
        s = g[g.segment == name]
        for d, part in (("EB", "morning"), ("WB", "evening")):
            x = s[(s["dir"] == d) & (s.sun_part == part)]
            n_dir = int((s["dir"] == d).sum())
            sec.append({"segment": name, "dir": d, "sun": part,
                        "crashes_dir": n_dir,
                        "sun_ahead": int(x.sun_ahead.sum()),
                        "pct_sun_ahead": 100 * x.sun_ahead.sum() / n_dir,
                        "expected_pct": exp_seg.loc[(name, d)],
                        "opposite_dir_sun_behind": int(
                            s[(s["dir"] != d) & (s.sun_part == part)].sun_behind.sum())})
    sec = pd.DataFrame(sec)
    sec["ratio"] = sec.pct_sun_ahead / sec.expected_pct
    sec.to_csv(OUT / "glare_by_segment.csv", index=False)

    # profile of the sun-ahead crashes
    sa = g[g.sun_ahead]
    prof = {
        "n": len(sa),
        "rear_end_pct": 100 * (sa.crash_cat == "Rear-end").mean(),
        "rear_end_pct_all": 100 * (g.crash_cat == "Rear-end").mean(),
        "multi_vehicle_pct": 100 * (sa.vehicles >= 2).mean(),
        "multi_vehicle_pct_all": 100 * (g.vehicles >= 2).mean(),
        "dark_pct_all": 0,
        "officer_coded_sun_glare": int((cr.tu1_human_factor.astype(str)
                                        .str.contains("glare", case=False)).sum()),
    }
    pd.Series(prof).to_csv(OUT / "glare_profile.csv")
    by_month = (sa.assign(m=sa.datetime.dt.month)
                  .groupby(["dir", "m"]).size().unstack(0, fill_value=0))
    by_month.to_csv(OUT / "glare_by_month.csv")
    return t, prof


DISTRACT = r"distract|preoccupied|passenger|phone|electronic|eating|manipulating|looked/did not see"
FATIGUE = r"asleep|fatigue"


def driver_tables(cr):
    c = cr[cr.year.between(*CURRENT)].copy()
    hf = c.tu1_human_factor.astype(str).str.lower()
    c["distracted"] = hf.str.contains(DISTRACT, regex=True)
    c["phone"] = hf.str.contains(r"phone|electronic|cell", regex=True)
    c["fatigue"] = hf.str.contains(FATIGUE, regex=True)
    al = c.tu1_alcohol.astype(str).str.lower()
    c["drugs"] = al.str.contains("drug|medication")
    c["ka"] = c.severity.isin(["K", "A"])
    c["k"] = c.severity == "K"
    c["night"] = c.hour.isin([21, 22, 23, 0, 1, 2, 3, 4])
    rows = []
    for f, label in (("alcohol", "Alcohol or drugs suspected"),
                     ("distracted", "Distracted or inattentive"),
                     ("phone", "  of which phone or device"),
                     ("fatigue", "Asleep or fatigued")):
        x = c[c[f]]
        rows.append({"factor": label, "crashes": len(x),
                     "pct_all": 100 * len(x) / len(c),
                     "ka": int(x.ka.sum()), "pct_of_ka": 100 * x.ka.sum() / c.ka.sum(),
                     "fatal": int(x.k.sum()),
                     "ka_rate_pct": 100 * x.ka.mean() if len(x) else np.nan,
                     "night_pct": 100 * x.night.mean() if len(x) else np.nan,
                     "single_vehicle_pct": 100 * (x.vehicles == 1).mean() if len(x) else np.nan})
    base = {"factor": "All crashes", "crashes": len(c), "pct_all": 100.0,
            "ka": int(c.ka.sum()), "pct_of_ka": 100.0, "fatal": int(c.k.sum()),
            "ka_rate_pct": 100 * c.ka.mean(), "night_pct": 100 * c.night.mean(),
            "single_vehicle_pct": 100 * (c.vehicles == 1).mean()}
    t = pd.DataFrame(rows + [base])
    t.to_csv(OUT / "drivers_factors.csv", index=False)
    seg = c.groupby("segment")[["alcohol", "distracted", "fatigue"]].agg(["sum", "mean"])
    seg.to_csv(OUT / "drivers_by_segment.csv")
    alc_hour = c[c.alcohol].hour.value_counts().sort_index()
    alc_hour.to_csv(OUT / "drivers_alcohol_by_hour.csv")
    # trend across both layouts (approximate: code lists differ before/after 2021)
    t2 = cr[cr.year.between(2007, 2025)].copy()
    hf2 = t2.tu1_human_factor.astype(str).str.lower()
    t2["distracted"] = hf2.str.contains(DISTRACT, regex=True)
    yr = t2.groupby("year").agg(crashes=("crash_id", "size"),
                                alcohol=("alcohol", "sum"),
                                distracted=("distracted", "sum"))
    yr.to_csv(OUT / "drivers_by_year.csv")
    return t, seg, c


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    cr, cl = load()
    t, prof = glare_tables(cr, cl)
    print(t.round(2).to_string()); print(prof)
    print(pd.read_csv(OUT / "glare_by_segment.csv").to_string())
    print(pd.read_csv(OUT / "glare_by_month.csv").to_string())
    d, seg, c = driver_tables(cr)
    print(d.round(1).to_string()); print(seg.round(3).to_string())
    print(pd.read_csv(OUT / "drivers_alcohol_by_hour.csv").to_string())
    print(pd.read_csv(OUT / "drivers_by_year.csv").to_string())
