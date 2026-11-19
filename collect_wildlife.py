"""Wildlife data for the US 34 corridor.

1. CDOT animal-involved crashes 2018-2023 (statewide layer with coordinates),
   kept when within SNAP_TOLERANCE_M of the US 34 centerline. Adds years the
   crash listings in this project don't cover yet. No species field: species
   comes only from the CDOT listings (crashes_us34.csv, `wild_animal`).
2. CPW elk ranges (Colorado Parks and Wildlife species activity mapping):
   overall range, winter range, severe winter range, winter concentration
   area, migration corridors, and mapped highway crossings. Each tenth-mile
   centerline point is tagged with the ranges it falls in.
3. CDOT statewide wildlife-transportation priority scores for US 34
   segments (elk/deer collision risk and winter density).

Run collect_roadway.py first (needs the centerline).
Outputs in data/raw/wildlife/.
"""
import time

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union

import config
from arcgis import query
from collect_crashes import haversine_m

OUT = config.RAW / "wildlife"
OUT.mkdir(parents=True, exist_ok=True)
AGOL = "https://services.arcgis.com/yzB9WM8W0BO3Ql7d/ArcGIS/rest/services"
CPW = "https://services5.arcgis.com/ttNGmDvKQA7oeDQ3/arcgis/rest/services/CPWSpeciesData/FeatureServer"
BBOX = "-105.60,40.34,-105.03,40.48"  # xmin,ymin,xmax,ymax
CPW_LAYERS = {
    "elk_overall_range": 44,
    "elk_winter_range": 43,
    "elk_severe_winter_range": 41,
    "elk_winter_concentration": 42,
    "elk_migration_corridor": 40,
    "elk_highway_crossing": 34,
}


def centerline():
    return pd.read_csv(config.RAW / "roadway" / "centerline_tenth_mile.csv")


def snap(df, cl):
    d = haversine_m(df["lat"].values[:, None], df["lon"].values[:, None],
                    cl["lat"].values[None, :], cl["lon"].values[None, :])
    j = d.argmin(axis=1)
    df["dist_to_us34_m"] = d[np.arange(len(df)), j]
    df["mp_snapped"] = cl["REFPT"].values[j]
    return df


def collect_animal_crashes(cl):
    df = query(f"{AGOL}/Animal_Involved_Crash_2018_2023/FeatureServer/0",
               where="Latitude>40.34 AND Latitude<40.48 AND Longitude>-105.60 AND Longitude<-105.03")
    df["lat"], df["lon"] = df["Latitude"], df["Longitude"]
    df = snap(df, cl)
    df = df[(df.dist_to_us34_m <= config.SNAP_TOLERANCE_M)
            & df.mp_snapped.between(config.MP_START, config.MP_END)].copy()
    df["crash_type"] = df["Crash_Type"].str.title()
    df["date"] = pd.to_datetime(df["Crash_Date"])
    df["segment"] = df["mp_snapped"].map(config.segment_for)
    df = df.drop(columns=["Latitude", "Longitude", "Crash_Type", "Crash_Date"]).sort_values("date")
    df.to_csv(OUT / "animal_crashes_2018_2023.csv", index=False)
    print(f"animal crashes on corridor 2018-2023: {len(df)} "
          f"({(df.crash_type == 'Wild Animal').sum()} wild, {(df.crash_type == 'Domestic Animal').sum()} domestic)")


def cpw_polygons(layer):
    params = {"where": "1=1", "geometry": BBOX, "geometryType": "esriGeometryEnvelope",
              "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": "FID",
              "returnGeometry": "true", "outSR": 4326, "f": "geojson"}
    for attempt in range(4):
        try:
            r = requests.get(f"{CPW}/{layer}/query", params=params, timeout=180)
            r.raise_for_status()
            feats = r.json().get("features", [])
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    geoms = [shape(f["geometry"]).buffer(0) for f in feats if f.get("geometry")]
    return unary_union(geoms) if geoms else None


def collect_elk_ranges(cl):
    pts = [Point(lon, lat) for lat, lon in zip(cl.lat, cl.lon)]
    out = cl[["REFPT", "lat", "lon", "segment"]].copy()
    for name, layer in CPW_LAYERS.items():
        g = cpw_polygons(layer)
        out[name] = [bool(g is not None and g.intersects(p.buffer(0.0005))) for p in pts]  # ~50 m
        inside = out[out.REFPT.between(config.MP_START, config.MP_END)][name].mean() * 31
        print(f"{name}: {inside:.1f} of 31 corridor miles")
    out.to_csv(OUT / "elk_ranges_by_tenth_mile.csv", index=False)


def collect_priority_scores():
    df = query(f"{AGOL}/Statewide_Wildlife_and_Transportation_Priority_Results/FeatureServer/0",
               where=f"Route='{config.ROUTE}' AND End_MP>={config.MP_START} AND Beg_MP<={config.MP_END}",
               geometry=False)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Shape")], errors="ignore").sort_values("Beg_MP")
    df.to_csv(OUT / "cdot_wildlife_priority_us34.csv", index=False)
    print(f"CDOT wildlife priority segments: {len(df)}")


if __name__ == "__main__":
    cl = centerline()
    collect_animal_crashes(cl)
    collect_elk_ranges(cl)
    collect_priority_scores()
