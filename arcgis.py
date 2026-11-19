"""Minimal ArcGIS REST query helper (paged, returns a DataFrame)."""
import time

import pandas as pd
import requests

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "us34-safety-analysis/0.1"


def query(layer_url, where="1=1", out_fields="*", geometry=True, page_size=1000,
          order_by="OBJECTID"):
    """Return all features of an ArcGIS layer as a DataFrame.

    With geometry=True, points get `lon`/`lat` columns and polylines get a
    `coords` column holding a list of [lon, lat] vertices (WGS84).
    """
    rows, offset = [], 0
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": str(geometry).lower(),
            "outSR": 4326,
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "orderByFields": order_by,
        }
        for attempt in range(4):
            try:
                r = SESSION.get(layer_url + "/query", params=params, timeout=120)
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        if "error" in data:
            raise RuntimeError(f"{layer_url}: {data['error']}")
        feats = data.get("features", [])
        for f in feats:
            row = dict(f["attributes"])
            g = f.get("geometry")
            if g:
                if "x" in g:
                    row["lon"], row["lat"] = g["x"], g["y"]
                elif "paths" in g:
                    row["coords"] = [pt for path in g["paths"] for pt in path]
            rows.append(row)
        if not data.get("exceededTransferLimit") and len(feats) < page_size:
            break
        offset += len(feats)
    return pd.DataFrame(rows)
