"""Extract US 34 (Estes Park - Loveland) crashes from CDOT statewide crash listings.

CDOT publishes one statewide Excel file per year (2007-present) at
https://www.codot.gov/safety/traffic-safety/data-analysis/crash-data
The files sit behind a document viewer, so download them by hand and drop
them anywhere in this project folder (e.g. CDOTRM_CD_Crash_Listing_-_2023.xlsx).

Steps:
  1. Stream each workbook and keep rows on US 34 or inside the corridor's
     bounding box. Cached in data/raw/crashes/<file>.csv, since a full
     statewide file takes ~1 min to scan.
  2. Normalise the column names (they differ between the 2007-2020 and
     2021+ formats), snap each crash to the CDOT tenth-mile centerline
     (needs collect_roadway.py run first), and keep the crashes inside
     MP_START..MP_END.
  3. Write data/processed/crashes_us34.csv.

Usage:  python collect_crashes.py [--rescan]
"""
import re
import sys

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import config

CACHE = config.RAW / "crashes"
CACHE.mkdir(parents=True, exist_ok=True)

# Generous box around the corridor (lat/lon) used for the first-pass filter.
BBOX = (40.35, 40.46, -105.58, -105.04)  # lat_min, lat_max, lon_min, lon_max

# Candidate column names (lower-cased, stripped) for each normalised field.
COLS = {
    "crash_id": ["cuid", "crash id", "caseid", "id"],
    "system": ["system code", "system", "sys"],
    "route": ["rd_number", "rte", "route", "hwy"],
    "section": ["rd_section", "sec", "section"],
    "mp_raw": ["mp", "milepost", "milepoint", "refpt"],
    "date": ["crash date", "date", "acc_date"],
    "time": ["crash time", "time"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "long"],
    "severity_raw": ["severity", "sev"],
    "killed": ["number killed", "killed", "num_killed"],
    "injured": ["number injured", "injured", "num_injured"],
    "inj0": ["injury 00", "injury_00", "inj00"],
    "inj1": ["injury 01", "injury_01", "inj01"],
    "inj2": ["injury 02", "injury_02", "inj02"],
    "inj3": ["injury 03", "injury_03", "inj03"],
    "inj4": ["injury 04", "injury_04", "inj04"],
    "vehicles": ["total vehicles", "vehicles", "num_veh"],
    "crash_type": ["mhe"],
    "crash_type2": ["acctype", "crash type"],
    "first_he": ["first he", "event_1"],
    "location": ["location", "loc_01"],
    "road_desc": ["road description", "road_desc"],
    "curves": ["road contour curves", "contour"],
    "grade": ["road contour grade", "contour"],
    "road_condition": ["road condition", "condition"],
    "lighting": ["lighting conditions", "lighting"],
    "weather": ["weather condition", "weather"],
    "weather2": ["weather condition 2"],
    "wild_animal": ["wild animal", "wan_type"],
    "construction_zone": ["construction zone"],
    "speed_limit": ["tu-1 speed limit", "limit1"],
    "tu1_est_speed": ["tu-1 estimated speed", "speed_1"],
    "tu1_direction": ["tu-1 direction", "dir_1"],
    "tu1_vehicle": ["tu-1 type", "vehicle_1"],
    "tu1_action": ["tu-1 driver action", "violcode_1"],
    "tu1_human_factor": ["tu-1 human contributing factor", "factor_1"],
    "tu1_age": ["tu-1 age", "age_1"],
    "tu1_alcohol": ["tu-1 alcohol suspected", "driver_1"],
    "tu1_dui": ["dui_1"],
    "tu2_vehicle": ["tu-2 type", "vehicle_2"],
    "tu2_action": ["tu-2 driver action", "violcode_2"],
}

# Harmonised categories, so the 2007-2020 layout (upper-case codes) and the
# 2021+ layout (title-case codes) can be compared. Keyword rules, first match wins.
CRASH_CAT_RULES = [
    ("Wild animal", ["WILD ANIMAL"]),
    ("Domestic animal", ["DOMESTIC"]),
    ("Rocks in road", ["ROCK", "BOULDER"]),
    ("Pedestrian", ["PEDESTRIAN"]),
    ("Bicycle", ["BICYCLE"]),
    ("Rear-end", ["REAR"]),
    ("Head-on", ["HEAD-ON", "FRONT TO FRONT"]),
    ("Sideswipe", ["SIDESWIPE", "SIDE TO SIDE"]),
    ("Angle or turning", ["BROADSIDE", "FRONT TO SIDE", "APPROACH TURN", "OVERTAKING TURN"]),
    ("Rollover", ["OVERTURN", "ROLLOVER"]),
    ("Parked vehicle", ["PARKED"]),
    ("Fixed object", ["SIGN", "TREE", "GUARD", "EMBANK", "DELINEATOR", "POLE", "FENCE", "CULVERT",
                      "BRIDGE", "BARRIER", "CURB", "WALL", "MAILBOX", "FIXED OBJECT", "DITCH",
                      "CRASH CUSHION", "HYDRANT", "OBJECT"]),
]


def _cat(value, rules, default="Other"):
    s = str(value).upper()
    if s in ("NAN", "NONE", ""):
        return "Unknown"
    for name, keys in rules:
        if any(k in s for k in keys):
            return name
    return default


def harmonise(out):
    up = lambda c: out[c].astype(str).str.upper()
    cat = out["crash_type"].map(lambda v: _cat(v, CRASH_CAT_RULES))
    # Fall back to the accident type, then the first harmful event, when the
    # most harmful event is missing or coded unknown.
    for col in ["crash_type2", "first_he"]:
        alt = out[col].map(lambda v: _cat(v, CRASH_CAT_RULES))
        cat = cat.where(~cat.isin(["Unknown", "Other"]) | alt.isin(["Unknown", "Other"]), alt)
    out["crash_cat"] = cat
    light = up("lighting")
    out["light_cat"] = np.select(
        [light.str.contains("UNLIGHT"), light.str.contains("DARK") & light.str.contains("LIGHT"),
         light.str.contains("DAYLIGHT"), light.str.contains("DAWN|DUSK")],
        ["Dark, unlighted", "Dark, lighted", "Daylight", "Dawn or dusk"], default="Unknown")
    surf = up("road_condition")
    out["surface_cat"] = np.select(
        [surf.str.contains("ICY|SNOW|SLUSH|ICE"), surf.str.contains("WET"), surf.str.contains("DRY")],
        ["Snow, ice or slush", "Wet", "Dry"], default="Other or unknown")
    wx = up("weather")
    out["weather_cat"] = np.select(
        [wx.str.contains("SNOW|SLEET|HAIL|FREEZING"), wx.str.contains("RAIN"), wx.str.contains("FOG"),
         wx.str.contains("WIND"), wx.str.contains("CLOUD"), wx.str.contains("CLEAR|NONE")],
        ["Snow or sleet", "Rain", "Fog", "Wind", "Cloudy", "Clear"], default="Other or unknown")
    out["on_curve"] = up("curves").str.contains("CURVE")
    out["motorcycle"] = up("tu1_vehicle").str.contains("MOTORCYCLE") | up("tu2_vehicle").str.contains("MOTORCYCLE")
    alc = up("tu1_alcohol")
    out["alcohol"] = (alc.str.contains("ALCOHOL") | alc.str.startswith("YES")
                      | up("tu1_dui").eq("Y"))
    out["dark"] = out["light_cat"].str.startswith("Dark")
    wa = up("wild_animal")
    out["animal_species"] = np.select(
        [wa.str.contains("ELK"), wa.str.contains("DEER"), wa.str.contains("BEAR"),
         wa.str.contains("SHEEP|BIGHORN"), out["crash_cat"].eq("Wild animal")],
        ["Elk", "Deer", "Bear", "Bighorn sheep", "Not recorded"], default="")
    return out


def crash_files():
    files = [p for p in config.ROOT.rglob("*.xls*")
             if ".venv" not in p.parts and not p.name.startswith("~$")]
    # The same year may be saved in more than one folder: keep one copy per name.
    by_name = {}
    for p in sorted(files):
        by_name.setdefault(p.name, p)
    return sorted(by_name.values(), key=lambda p: p.name)


def _is_us34(route):
    """Loose match used when scanning (US 34 with or without a section letter)."""
    s = str(route or "").strip().upper()
    return bool(re.fullmatch(r"0*34(\.0)?[A-Z]?", s))


def _is_us34_mainline(route):
    """US 34 mainline (034A). A bare '34' (no section letter) is accepted too."""
    s = str(route or "").strip().upper()
    return bool(re.fullmatch(r"0*34(\.0)?A?", s))


def scan_file(path, rescan=False):
    """First pass: keep rows on US 34 or inside the corridor bounding box."""
    cache = CACHE / (path.stem + ".csv")
    if cache.exists() and not rescan:
        return pd.read_csv(cache, dtype=str)
    print(f"scanning {path.name} ...", flush=True)
    wb = load_workbook(path, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else f"col{i}" for i, h in enumerate(next(rows))]
    low = [h.lower() for h in header]

    def find(names):
        return next((low.index(n) for n in names if n in low), None)

    i_route, i_lat, i_lon = find(COLS["route"]), find(COLS["lat"]), find(COLS["lon"])
    keep, total = [], 0
    for r in rows:
        total += 1
        hit = i_route is not None and _is_us34(r[i_route])
        if not hit and i_lat is not None and i_lon is not None:
            try:
                lat, lon = float(r[i_lat]), float(r[i_lon])
                hit = BBOX[0] <= lat <= BBOX[1] and BBOX[2] <= lon <= BBOX[3]
            except (TypeError, ValueError):
                pass
        if hit:
            keep.append(r)
    wb.close()
    df = pd.DataFrame(keep, columns=header).astype(str)
    df.insert(0, "source_file", path.name)
    df.to_csv(cache, index=False)
    print(f"  {total:,} statewide rows -> {len(df):,} kept", flush=True)
    return df


def normalise(raw):
    low = {c.strip().lower(): c for c in raw.columns}
    out = pd.DataFrame({"source_file": raw["source_file"]})
    for field, names in COLS.items():
        src = next((low[n] for n in names if n in low), None)
        out[field] = raw[src] if src else np.nan
    out = out.replace({"None": np.nan, "nan": np.nan, "": np.nan})

    num = ["lat", "lon", "killed", "injured", "inj0", "inj1", "inj2", "inj3", "inj4",
           "vehicles", "speed_limit", "tu1_est_speed", "tu1_age"]
    for c in num:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # Older files store missing coordinates as 0.
    out.loc[(out["lat"] == 0) | (out["lon"] == 0), ["lat", "lon"]] = np.nan
    out.loc[out["tu1_age"] <= 0, "tu1_age"] = np.nan

    # Milepost: 2021+ files put it in Rd_Section for state highways; older
    # files have a separate MP column (and SEC holds the route letter).
    mp = pd.to_numeric(out["mp_raw"], errors="coerce")
    sec = pd.to_numeric(out["section"], errors="coerce")
    out["mp_reported"] = mp.fillna(sec)
    out.loc[(out["mp_reported"] >= 999) | (out["mp_reported"] <= 0), "mp_reported"] = np.nan
    sec_letter = out["section"].astype(str).str.strip().str.upper()
    out["route"] = np.where(out["route"].astype(str).str.fullmatch(r"0*34(\.0)?") & sec_letter.str.fullmatch(r"[A-Z]"),
                            "034" + sec_letter, out["route"].astype(str).str.replace(r"\.0$", "", regex=True))

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    t = out["time"].astype(str).str.extract(r"(\d{1,2}):?(\d{2})")
    out["hour"] = pd.to_numeric(t[0], errors="coerce")
    out["datetime"] = out["date"] + pd.to_timedelta(out["hour"].fillna(0), unit="h") \
        + pd.to_timedelta(pd.to_numeric(t[1], errors="coerce").fillna(0), unit="m")
    out["year"] = out["date"].dt.year
    out["on_us34_route"] = out["route"].map(_is_us34_mainline)

    # KABCO severity from injury counts (04 = fatal ... 00 = no injury).
    sev = np.select(
        [out["inj4"] > 0, out["inj3"] > 0, out["inj2"] > 0, out["inj1"] > 0],
        ["K", "A", "B", "C"], default="O")
    sev = pd.Series(sev, index=out.index)
    has_counts = out[["inj0", "inj1", "inj2", "inj3", "inj4"]].notna().any(axis=1)
    sev[~has_counts] = out.loc[~has_counts, "severity_raw"].map(
        {"FAT": "K", "INJ": "INJ", "PDO": "O"}).fillna("UNK")
    out["severity"] = sev
    return harmonise(out)


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))


def snap(df):
    cl = pd.read_csv(config.RAW / "roadway" / "centerline_tenth_mile.csv")
    d = haversine_m(df["lat"].values[:, None], df["lon"].values[:, None],
                    cl["lat"].values[None, :], cl["lon"].values[None, :])
    d = np.where(np.isnan(d), np.inf, d)
    j = d.argmin(axis=1)
    df["dist_to_us34_m"] = d[np.arange(len(df)), j]
    df["mp_snapped"] = np.where(np.isfinite(df["dist_to_us34_m"]), cl["REFPT"].values[j], np.nan)
    df.loc[~np.isfinite(df["dist_to_us34_m"]), "dist_to_us34_m"] = np.nan
    return df


def select_corridor(df):
    in_rng = lambda s: s.between(config.MP_START, config.MP_END)
    near = df["dist_to_us34_m"] <= config.SNAP_TOLERANCE_M
    by_mp = df["on_us34_route"] & in_rng(df["mp_reported"])
    # US 34 crash with no usable milepost: fall back to its coordinates.
    by_xy = df["on_us34_route"] & df["mp_reported"].isna() & near & in_rng(df["mp_snapped"])
    df = df[by_mp | by_xy].copy()
    df["mp"] = df["mp_reported"].fillna(df["mp_snapped"])
    df["mp_source"] = np.where(df["mp_reported"].notna(), "reported", "snapped")
    # Reported MP and coordinates more than 1 mile apart -> flag for review.
    df["location_mismatch"] = (df["mp_reported"] - df["mp_snapped"]).abs() > 1.0
    df["segment"] = df["mp"].map(config.segment_for)
    return df


def main():
    rescan = "--rescan" in sys.argv
    files = crash_files()
    if not files:
        print("No crash listing .xlsx files found in the project folder.")
        return
    raw = [scan_file(p, rescan) for p in files]
    # Normalise each file on its own: column names differ between layouts.
    df = pd.concat([normalise(r) for r in raw], ignore_index=True)
    df = select_corridor(snap(df))
    df = df.drop(columns=["mp_raw", "section"]).sort_values("datetime")
    out = config.PROCESSED / "crashes_us34.csv"
    df.to_csv(out, index=False)

    print(f"\n{len(df):,} crashes on US 34 MP {config.MP_START}-{config.MP_END} -> {out.relative_to(config.ROOT)}")
    print(pd.crosstab(df["year"], df["severity"], margins=True).to_string())
    print("\nBy segment:\n" + df["segment"].value_counts().to_string())
    print(f"\nMP from coordinates (no reported MP): {(df.mp_source == 'snapped').sum()}")
    print(f"Reported MP vs coordinates >1 mi apart: {int(df.location_mismatch.sum())}")


if __name__ == "__main__":
    main()
