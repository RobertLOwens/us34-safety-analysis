"""Shared settings for the US 34 (Loveland - Estes Park) safety analysis."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
for p in (RAW, PROCESSED):
    p.mkdir(parents=True, exist_ok=True)

# CDOT linear referencing: US 34 mainline is route "034A".
ROUTE = "034A"
ROAD_NUMBER = "034"  # as written in the CDOT crash listings (Rd_Number)

# Corridor limits in CDOT reference points (mileposts).
# MP 61 ~ downtown Estes Park, MP 92 ~ US 287 (Lincoln Ave) in Loveland.
MP_START = 61.0
MP_END = 92.0

# Sub-segments for reporting. Breakpoints are approximate; adjust after
# looking at the roadway data.
SEGMENTS = [
    ("Estes Park (town)", 61.0, 65.0),
    ("Big Thompson Canyon", 65.0, 83.5),
    ("Canyon mouth to Loveland", 83.5, 88.0),
    ("Loveland (urban)", 88.0, 92.0),
]

# Max distance (metres) from the US 34 centerline for a crash point to be
# considered on the corridor when matching by coordinates.
SNAP_TOLERANCE_M = 150

# Analysis period.
START_DATE = "2007-01-01"
END_DATE = "2026-11-15"

# Weather: NOAA GHCN-Daily stations along the corridor.
NOAA_STATIONS = {
    "USC00052759": "Estes Park",          # reports 2022+ in this period
    "USC00052761": "Estes Park 3 SSE",    # 1998-2021, fills the gap above
    "USC00058839": "Waterdale (canyon mouth)",
    "USC00055236": "Loveland 2 N",
}

# Weather: Open-Meteo (ERA5 reanalysis) points, used for hourly conditions.
WEATHER_POINTS = {
    "estes_park": (40.3772, -105.5217),   # ~MP 62
    "drake": (40.4320, -105.3340),        # ~MP 76, mid-canyon
    "canyon_mouth": (40.4114, -105.2153), # ~MP 84
    "loveland": (40.4072, -105.0910),     # ~MP 91
}


def segment_for(mp):
    for name, a, b in SEGMENTS:
        if a <= mp < b:
            return name
    # The corridor end point (e.g. MP 92.0, US 287) belongs to the last section.
    if mp == SEGMENTS[-1][2]:
        return SEGMENTS[-1][0]
    return None
