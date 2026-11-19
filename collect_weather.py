"""Collect weather for the US 34 corridor.

1. NOAA GHCN-Daily observations (precip, snowfall, snow depth, max/min temp)
   from stations in Estes Park, at the canyon mouth, and in Loveland.
   Source: NOAA GHCN-D open data on AWS (no API key needed).
2. Hourly reanalysis (ERA5 via Open-Meteo) at four points along the road, so
   each crash can be matched to conditions at its hour and location.

Outputs:
  data/raw/weather/noaa_daily.csv
  data/raw/weather/hourly_<point>.csv
"""
import io
import time
from datetime import date

import pandas as pd
import requests

import config

OUT = config.RAW / "weather"
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "us34-safety-analysis/0.1"}


def get(url, params, tries=5):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=180)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"  retry ({e.__class__.__name__})", flush=True)
            time.sleep(5 * 2 ** attempt)


GHCN_COLS = ["STATION", "DATE", "ELEMENT", "VALUE", "MFLAG", "QFLAG", "SFLAG", "OBS_TIME"]
# GHCN-D native units -> inches / deg F
CONVERT = {
    "PRCP": lambda v: v / 254.0,          # tenths of mm
    "SNOW": lambda v: v / 25.4,           # mm
    "SNWD": lambda v: v / 25.4,           # mm
    "TMAX": lambda v: v / 10 * 9 / 5 + 32,  # tenths of deg C
    "TMIN": lambda v: v / 10 * 9 / 5 + 32,
}


def collect_noaa():
    """GHCN-Daily station files from NOAA's open-data bucket on AWS."""
    frames = []
    for sid, label in config.NOAA_STATIONS.items():
        r = get(f"https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{sid}.csv", {})
        df = pd.read_csv(io.StringIO(r.text), header=None, names=GHCN_COLS,
                         dtype={"DATE": str}, low_memory=False)
        if df.iloc[0, 0] == "ID":  # some files carry a header row
            df = df.iloc[1:]
        df = df[df["ELEMENT"].isin(CONVERT) & df["QFLAG"].isna()]  # drop QC-failed values
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
        df = df[df["DATE"].between(config.START_DATE, config.END_DATE)]
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
        wide = df.pivot_table(index="DATE", columns="ELEMENT", values="VALUE", aggfunc="first")
        for el, f in CONVERT.items():
            if el in wide:
                wide[el] = f(wide[el]).round(2)
        wide = wide.reset_index()
        wide.insert(0, "station", sid)
        wide.insert(1, "station_label", label)
        frames.append(wide)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"DATE": "date", "PRCP": "precip_in", "SNOW": "snowfall_in",
                            "SNWD": "snow_depth_in", "TMAX": "tmax_f", "TMIN": "tmin_f"})
    df.to_csv(OUT / "noaa_daily.csv", index=False)
    print("NOAA daily:\n" + df.groupby("station_label")["date"].agg(["min", "max", "count"]).to_string())


HOURLY = ["temperature_2m", "precipitation", "rain", "snowfall", "snow_depth",
          "weather_code", "cloud_cover", "wind_speed_10m", "wind_gusts_10m"]


def collect_hourly():
    start, end = date.fromisoformat(config.START_DATE), date.fromisoformat(config.END_DATE)
    for name, (lat, lon) in config.WEATHER_POINTS.items():
        dest = OUT / f"hourly_{name}.csv"
        if dest.exists():
            print(f"hourly {name}: cached")
            continue
        parts = []
        for y in range(start.year, end.year + 1):
            a = max(start, date(y, 1, 1)); b = min(end, date(y, 12, 31))
            r = get("https://archive-api.open-meteo.com/v1/archive", {
                "latitude": lat, "longitude": lon,
                "start_date": a.isoformat(), "end_date": b.isoformat(),
                "hourly": ",".join(HOURLY),
                "timezone": "America/Denver",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
            })
            parts.append(pd.DataFrame(r.json()["hourly"]))
            time.sleep(1)  # be polite to the free API
        df = pd.concat(parts, ignore_index=True)
        df.insert(0, "point", name)
        df.to_csv(dest, index=False)
        print(f"hourly {name}: {len(df):,} hours {df.time.min()} .. {df.time.max()}")


if __name__ == "__main__":
    collect_noaa()
    collect_hourly()
