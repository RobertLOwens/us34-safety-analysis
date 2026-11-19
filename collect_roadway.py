"""Collect roadway characteristics for US 34 between Estes Park and Loveland.

Sources (all public CDOT ArcGIS services):
  - Highways (master segment table): speed limit, lanes, lane/shoulder width,
    terrain, functional class, AADT, surface, median.
  - Highways: Geometrics: detailed cross-section by direction.
  - Highways: Traffic Counts: AADT, trucks, design-hour volume, V/C ratio.
  - Highways: Access Categories.
  - AADT All Years (1997-2025): traffic volume history.
  - Milepoints (tenths): centerline points used for crash matching and to
    derive curvature; elevations from the USGS 3DEP point query service.

Outputs go to data/raw/roadway/ as CSV.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import config
from arcgis import query

SDE = "https://dtdapps.codot.gov/server/rest/services/Webapps/open_data_sde/MapServer"
AGOL = "https://services.arcgis.com/yzB9WM8W0BO3Ql7d/ArcGIS/rest/services"
OUT = config.RAW / "roadway"
OUT.mkdir(parents=True, exist_ok=True)

SEG_WHERE = (f"ROUTE='{config.ROUTE}' AND ENDREFPT>={config.MP_START} "
             f"AND REFPT<={config.MP_END}")

SEGMENT_LAYERS = {
    "highways_master": f"{SDE}/7",
    "geometrics": f"{SDE}/12",
    "traffic_counts": f"{SDE}/13",
    "access_categories": f"{SDE}/9",
}


def collect_segment_layers():
    for name, url in SEGMENT_LAYERS.items():
        df = query(url, where=SEG_WHERE, geometry=False)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Shape")], errors="ignore")
        df = df.sort_values("REFPT")
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} segments")


def collect_aadt_history():
    df = query(f"{AGOL}/AADT_All_Years_Data/FeatureServer/0",
               where=f"route='{config.ROUTE}' AND endRefPt>={config.MP_START} "
                     f"AND refPt<={config.MP_END}",
               geometry=False)
    df = df.drop(columns=["Shape__Length"], errors="ignore").sort_values("refPt")
    df.to_csv(OUT / "aadt_history_wide.csv", index=False)
    year_cols = [c for c in df.columns if c.startswith("AADT")]
    long = df.melt(id_vars=["route", "refPt", "endRefPt"], value_vars=year_cols,
                   var_name="year", value_name="aadt")
    long["year"] = long["year"].str[4:].astype(int).map(lambda y: 1900 + y if y >= 90 else 2000 + y)
    long.dropna(subset=["aadt"]).to_csv(OUT / "aadt_history_long.csv", index=False)
    print(f"aadt history: {len(df)} segments, years {long.year.min()}-{long.year.max()}")


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _usgs_elev(latlon):
    """USGS 3DEP Elevation Point Query (1 m lidar where available)."""
    lat, lon = latlon
    for attempt in range(5):
        try:
            r = requests.get("https://epqs.nationalmap.gov/v1/json",
                             params={"x": lon, "y": lat, "units": "Meters", "wkid": 4326},
                             timeout=30)
            r.raise_for_status()
            return float(r.json()["value"])
        except Exception:
            time.sleep(2 ** attempt)
    return np.nan


def elevations(lats, lons):
    with ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(_usgs_elev, zip(lats, lons)))


def collect_centerline():
    """Tenth-mile points on the whole route (with a margin around the corridor)."""
    df = query(f"{SDE}/38",
               where=f"ROUTE='{config.ROUTE}' AND REFPT>={config.MP_START - 1} "
                     f"AND REFPT<={config.MP_END + 1}")
    df = df[["ROUTE", "REFPT", "lat", "lon"]].sort_values("REFPT").drop_duplicates("REFPT")
    df = df.reset_index(drop=True)

    # Heading change between consecutive tenth-mile chords = curvature proxy
    # (degrees per 0.1 mi). Large values flag sharp curves.
    b = [np.nan] + [bearing(df.lat[i - 1], df.lon[i - 1], df.lat[i], df.lon[i])
                    for i in range(1, len(df))]
    df["bearing_deg"] = b
    d = df["bearing_deg"].diff()
    df["heading_change_deg"] = ((d + 180) % 360 - 180).abs()

    df["elev_m"] = elevations(df.lat.tolist(), df.lon.tolist())
    # Grade over each tenth mile (161 m), percent.
    df["grade_pct"] = df["elev_m"].diff() / 160.934 * 100
    df["segment"] = df["REFPT"].map(config.segment_for)
    df.to_csv(OUT / "centerline_tenth_mile.csv", index=False)
    print(f"centerline: {len(df)} tenth-mile points, elev {df.elev_m.min():.0f}-{df.elev_m.max():.0f} m")


if __name__ == "__main__":
    collect_segment_layers()
    collect_aadt_history()
    collect_centerline()
