"""Collect traffic exposure and context data for US 34.

1. CDOT continuous count station 000205 (US 34, MP ~64.2-64.8, east Estes
   Park): monthly average daily traffic since 1996, and hourly counts by
   direction for every day from HOURLY_START_YEAR on. Source: CDOT OTIS.
2. Rocky Mountain National Park monthly recreation visits (NPS IRMA Stats),
   which track the tourist traffic that uses US 34.
3. CDOT geohazard event log (rockfall, rockslide, debris flow) on US 34,
   with closure flags. Coverage ends around 2017.

Outputs in data/raw/exposure/.
"""
import io
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

import config
from arcgis import query

OUT = config.RAW / "exposure"
OUT.mkdir(parents=True, exist_ok=True)
OTIS = "https://dtdapps.codot.gov/otis/API/TRANSYS"
STATION = "000205"
HOURLY_START_YEAR = 2013
UA = {"User-Agent": "us34-safety-analysis/0.1"}


def get(url, **kw):
    for attempt in range(5):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            r.raise_for_status()
            return r
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def collect_monthly():
    df = pd.read_csv(io.StringIO(get(f"{OTIS}/GetMonthlyTrafficVolumeForStation/{STATION}.csv").text),
                     dtype={"COUNTSTATIONID": str})
    months = [c for c in df.columns if c.endswith("COUNT")]
    long = df.melt(id_vars=["COUNTSTATIONID", "CALYR"], value_vars=months,
                   var_name="month", value_name="adt")
    long["month"] = long["month"].str[:3].map(
        {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)})
    long = long.dropna(subset=["adt"]).rename(columns={"CALYR": "year"})
    long = long.sort_values(["year", "month"])
    long.to_csv(OUT / f"station_{STATION}_monthly_adt.csv", index=False)
    print(f"monthly ADT: {len(long)} months {long.year.min()}-{long.year.max()}")


def _day(d):
    url = f"{OTIS}/GetDailyTrafficVolumeForStationByDay/{STATION}/true/{d.year}-{d.month}-{d.day}.csv"
    try:
        t = get(url).text
    except Exception:
        return None
    if not t.startswith("COUNTDIR"):
        return None
    df = pd.read_csv(io.StringIO(t))
    if df.empty:
        return None
    df.insert(0, "date", d.isoformat())
    return df


def collect_hourly():
    dest = OUT / f"station_{STATION}_hourly.csv"
    have = set()
    if dest.exists():
        have = set(pd.read_csv(dest, usecols=["date"])["date"])
    start, end = date(HOURLY_START_YEAR, 1, 1), date.today() - timedelta(days=2)
    days = [start + timedelta(n) for n in range((end - start).days + 1)]
    days = [d for d in days if d.isoformat() not in have]
    print(f"hourly counts: fetching {len(days)} days ...", flush=True)
    frames = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, df in enumerate(ex.map(_day, days)):
            if df is not None:
                frames.append(df)
            if i and i % 500 == 0:
                print(f"  {i}/{len(days)}", flush=True)
    if frames:
        new = pd.concat(frames, ignore_index=True).drop(columns=["sort"], errors="ignore")
        hours = [f"HOUR{h}" for h in range(24)]
        new = new.melt(id_vars=["date", "COUNTDIR"], value_vars=hours, var_name="hour", value_name="volume")
        new["hour"] = new["hour"].str[4:].astype(int)
        # COUNTDIR: P = primary (increasing MP, eastbound), S = secondary (westbound)
        new["direction"] = new["COUNTDIR"].map({"P": "EB", "S": "WB"}).fillna(new["COUNTDIR"])
        new = new.drop(columns="COUNTDIR")
        if dest.exists():
            new = pd.concat([pd.read_csv(dest), new], ignore_index=True)
        new.sort_values(["date", "hour", "direction"]).to_csv(dest, index=False)
    total = pd.read_csv(dest, usecols=["date"])["date"].nunique() if dest.exists() else 0
    print(f"hourly counts: {total} days with data")


def collect_park_visits():
    r = get("https://irmaservices.nps.gov/Stats/v1/visitation",
            params={"unitCodes": "ROMO", "startMonth": 1, "startYear": 2000,
                    "endMonth": 12, "endYear": date.today().year},
            )
    r = requests.get(r.url, headers={**UA, "Accept": "application/json"}, timeout=60)
    df = pd.DataFrame(r.json())[["Year", "Month", "RecreationVisitors", "NonRecreationVisitors"]]
    df.columns = ["year", "month", "recreation_visits", "non_recreation_visits"]
    df.to_csv(OUT / "rmnp_monthly_visits.csv", index=False)
    print(f"RMNP visits: {len(df)} months {df.year.min()}-{df.year.max()}")


def collect_geohazards():
    df = query("https://services.arcgis.com/yzB9WM8W0BO3Ql7d/ArcGIS/rest/services/Geohazards_Update/FeatureServer/0",
               where="Route=34", geometry=False, order_by="FID")
    df["Begin_MP"] = pd.to_numeric(df["Begin_MP"], errors="coerce")
    df = df[df["Begin_MP"].between(config.MP_START, config.MP_END)].sort_values(["Year", "Begin_MP"])
    df["segment"] = df["Begin_MP"].map(config.segment_for)
    df.to_csv(OUT / "geohazard_events.csv", index=False)
    print(f"geohazard events: {len(df)} ({df.Year.min()}-{df.Year.max()})")


if __name__ == "__main__":
    collect_monthly()
    collect_park_visits()
    collect_geohazards()
    collect_hourly()
