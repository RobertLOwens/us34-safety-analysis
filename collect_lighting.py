"""Roadway lighting evidence for US 34.

No agency publishes a street light inventory for this corridor (CDOT asset
data is not public; Loveland and Estes Park publish none). Two proxies:

1. OpenStreetMap: street lamps mapped within LAMP_DIST_M of the centerline,
   and `lit=yes/no` tags on the US 34 road segments. Volunteer-mapped, so
   coverage is partial: absence of a lamp in OSM does not mean no lamp.
2. Officer-recorded lighting on night crashes in the CDOT listings
   ("Dark - Lighted" vs "Dark - Unlighted"), already in crashes_us34.csv;
   summarised in eda.py.

Output: data/raw/lighting/osm_lamps.csv, osm_lit_ways.csv
"""
import time

import pandas as pd
import requests

import config
from collect_crashes import haversine_m

OUT = config.RAW / "lighting"
OUT.mkdir(parents=True, exist_ok=True)
LAMP_DIST_M = 60
BBOX = "40.34,-105.60,40.48,-105.03"  # south,west,north,east
QUERY = f"""[out:json][timeout:120];
(node["highway"="street_lamp"]({BBOX});
 way["highway"]["ref"~"(^|;)US 34($|;)"]({BBOX}););
out tags center;"""


def main():
    for attempt in range(4):
        try:
            r = requests.post("https://overpass-api.de/api/interpreter", data={"data": QUERY},
                              headers={"User-Agent": "us34-safety-analysis/0.1"}, timeout=180)
            r.raise_for_status()
            els = r.json()["elements"]
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10 * 2 ** attempt)
    cl = pd.read_csv(config.RAW / "roadway" / "centerline_tenth_mile.csv")

    def snap(df):
        d = haversine_m(df.lat.values[:, None], df.lon.values[:, None], cl.lat.values[None, :], cl.lon.values[None, :])
        j = d.argmin(axis=1)
        df["dist_to_us34_m"] = d.min(axis=1)
        df["mp_snapped"] = cl.REFPT.values[j]
        df["segment"] = df.mp_snapped.map(config.segment_for)
        return df

    lamps = pd.DataFrame([{"osm_id": e["id"], "lat": e["lat"], "lon": e["lon"]}
                          for e in els if e["type"] == "node"])
    lamps = snap(lamps)
    lamps = lamps[(lamps.dist_to_us34_m <= LAMP_DIST_M) & lamps.mp_snapped.between(config.MP_START, config.MP_END)]
    lamps.to_csv(OUT / "osm_lamps.csv", index=False)

    ways = pd.DataFrame([{"osm_id": e["id"], "name": e["tags"].get("name"), "ref": e["tags"].get("ref"),
                          "lit": e["tags"].get("lit"), "lat": e["center"]["lat"], "lon": e["center"]["lon"]}
                         for e in els if e["type"] == "way"])
    ways = snap(ways)
    ways = ways[(ways.dist_to_us34_m <= 100) & ways.mp_snapped.between(config.MP_START, config.MP_END)]
    ways.sort_values("mp_snapped").to_csv(OUT / "osm_lit_ways.csv", index=False)

    print(f"OSM lamps within {LAMP_DIST_M} m of US 34: {len(lamps)}")
    print(lamps.groupby("segment").size().to_string())
    print(f"\nUS 34 OSM ways in corridor: {len(ways)}; lit tag:")
    print(ways.groupby(["segment", ways.lit.fillna("untagged")]).size().to_string())


if __name__ == "__main__":
    main()
