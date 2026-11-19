"""Exploratory summary tables for the report. Writes CSVs to data/processed/eda/.

Two periods:
  - TREND_YEARS: every full year in the crash listings (2007-2025), for trends.
  - CURRENT_YEARS: 2019-2025, the rebuilt canyon after the 2016-2019 flood
    reconstruction. Used for "today's road" patterns and rates.
2026 is preliminary and partial, so it is shown in counts only.

Rates use vehicle-miles from CDOT's annual AADT history per segment.
"""
import numpy as np
import pandas as pd

import config

OUT = config.PROCESSED / "eda"
OUT.mkdir(parents=True, exist_ok=True)
R = config.RAW
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 80)

TREND_YEARS = list(range(2007, 2026))
CURRENT_YEARS = list(range(2019, 2026))
SEG = [s[0] for s in config.SEGMENTS]
INJ = ["K", "A", "B", "C"]


def show(name, df):
    print(f"\n== {name}\n{df}")
    df.to_csv(OUT / f"{name}.csv")


cr_all = pd.read_csv(config.PROCESSED / "crashes_us34.csv", parse_dates=["date", "datetime"], low_memory=False)
cr_all["ka"] = cr_all.severity.isin(["K", "A"])
cr_all["inj"] = cr_all.severity.isin(INJ)
cr_all["month"] = cr_all.date.dt.month
cr_all["single_vehicle"] = cr_all.vehicles.eq(1)
cr = cr_all[cr_all.year.isin(CURRENT_YEARS)].copy()
ny = len(CURRENT_YEARS)
print(f"crashes: {len(cr_all)} total; {len(cr)} in {CURRENT_YEARS[0]}-{CURRENT_YEARS[-1]}")

# --- Vehicle-miles by section and year (AADT history) -----------------------
ah = pd.read_csv(R / "roadway" / "aadt_history_long.csv")
ah["a"] = ah.refPt.clip(lower=config.MP_START)
ah["b"] = ah.endRefPt.clip(upper=config.MP_END)
ah = ah[ah.b > ah.a].copy()
ah["segment"] = ((ah.a + ah.b) / 2).map(config.segment_for)
ah["mvmt"] = ah.aadt * (ah.b - ah.a) * 365 / 1e6
vmt = ah.pivot_table(index="year", columns="segment", values="mvmt", aggfunc="sum")[SEG]
vmt["Corridor"] = vmt.sum(axis=1)
show("mvmt_by_year", vmt.loc[TREND_YEARS].round(1))

# --- Yearly trend ------------------------------------------------------------
yr = cr_all.groupby("year").agg(crashes=("severity", "size"), injury=("inj", "sum"), ka=("ka", "sum"),
                                k=("severity", lambda s: (s == "K").sum()),
                                canyon=("segment", lambda s: (s == "Big Thompson Canyon").sum()))
yr["mvmt"] = vmt["Corridor"]
yr["rate_per_mvmt"] = yr.crashes / yr.mvmt
yr["ka_per_100mvmt"] = yr.ka / yr.mvmt * 100
show("by_year", yr.round(2))
show("ka_by_year_segment", pd.crosstab(cr_all[cr_all.ka].year, cr_all[cr_all.ka].segment).reindex(columns=SEG, fill_value=0))

# --- Roadway by section (current inventory) ----------------------------------
hw = pd.read_csv(R / "roadway" / "highways_master.csv")
hw["len"] = (hw.ENDREFPT.clip(upper=config.MP_END) - hw.REFPT.clip(lower=config.MP_START)).clip(lower=0)
hw["segment"] = ((hw.REFPT + hw.ENDREFPT) / 2).map(config.segment_for)
hw = hw[hw.segment.notna() & (hw.len > 0)]
wavg = lambda g, c: np.average(g[c], weights=g.len)
cl = pd.read_csv(R / "roadway" / "centerline_tenth_mile.csv")
cl = cl[cl.REFPT.between(config.MP_START, config.MP_END)]
road = pd.DataFrame({
    "length_mi": hw.groupby("segment").len.sum(),
    "aadt_2025": hw.groupby("segment").apply(lambda g: wavg(g, "AADT")),
    "speed_min": hw.groupby("segment").SPEEDLIM.min(), "speed_max": hw.groupby("segment").SPEEDLIM.max(),
    "lanes_max": hw.groupby("segment").THRULNQTY.max(),
    "shoulder_ft": hw.groupby("segment").apply(lambda g: wavg(g, "PRIOUTSHLDWD")),
    "pct_shoulder_le_4ft": hw.groupby("segment").apply(lambda g: g.loc[g.PRIOUTSHLDWD <= 4, "len"].sum() / g.len.sum() * 100),
    "curv_deg_per_mi": cl.groupby("segment").heading_change_deg.sum() / cl.groupby("segment").size() * 10,
    "elev_change_m": cl.groupby("segment").elev_m.agg(lambda s: s.max() - s.min()),
}).loc[SEG].round(1)
show("roadway_by_segment", road)

# --- Sections, current period ------------------------------------------------
mv_cur = vmt.loc[CURRENT_YEARS, SEG].sum()
seg = cr.groupby("segment").agg(crashes=("severity", "size"), injury=("inj", "sum"), ka=("ka", "sum"),
                                k=("severity", lambda s: (s == "K").sum())).reindex(SEG)
seg["mvmt"] = mv_cur
seg["per_mile_yr"] = seg.crashes / road.length_mi / ny
seg["rate_per_mvmt"] = seg.crashes / seg.mvmt
seg["ka_per_100mvmt"] = seg.ka / seg.mvmt * 100
seg["pct_injury"] = seg.injury / seg.crashes * 100
seg.loc["Corridor"] = seg.sum(numeric_only=True)
c = seg.loc["Corridor"]
seg.loc["Corridor", ["per_mile_yr", "rate_per_mvmt", "ka_per_100mvmt", "pct_injury"]] = [
    c.crashes / 31 / ny, c.crashes / c.mvmt, c.ka / c.mvmt * 100, c.injury / c.crashes * 100]
show("crashes_by_segment", seg.round(2))
show("severity_current", pd.crosstab(cr.year, cr.severity, margins=True))

# --- Crash types -------------------------------------------------------------
ct = pd.crosstab(cr.crash_cat, cr.segment).reindex(columns=SEG, fill_value=0)
ct["total"] = ct.sum(axis=1)
ct["ka"] = cr[cr.ka].crash_cat.value_counts()
show("crash_cat_by_segment", ct.fillna(0).astype(int).sort_values("total", ascending=False))
can = cr[cr.segment == "Big Thompson Canyon"]
show("canyon_curve_share", pd.Series({"crashes": len(can), "on_curve": can.on_curve.sum(),
                                       "single_vehicle": can.single_vehicle.sum()}).to_frame())

# --- Per mile ----------------------------------------------------------------
cr["mi"] = np.floor(cr.mp).clip(upper=91).astype(int)
pm = cr.groupby("mi").agg(total=("severity", "size"), ka=("ka", "sum")).reindex(range(61, 92), fill_value=0)
show("per_mile_current", pm)

# --- Time --------------------------------------------------------------------
mon = pd.read_csv(R / "exposure" / "station_000205_monthly_adt.csv")
mon = mon[mon.year.isin(CURRENT_YEARS)].groupby("month").adt.mean()
mt = pd.DataFrame({"crashes": cr.groupby("month").size().reindex(range(1, 13), fill_value=0),
                   "ka": cr[cr.ka].groupby("month").size().reindex(range(1, 13), fill_value=0),
                   "station205_adt": mon.round(0)})
mt["rate_per_mvmt"] = mt.crashes / (mv_cur.sum() / 12 * mt.station205_adt / mt.station205_adt.mean())
show("by_month", mt.round(2))
hr = pd.read_csv(R / "exposure" / "station_000205_hourly.csv")
hr = hr[pd.to_datetime(hr.date).dt.year.isin(CURRENT_YEARS)]
hp = hr.groupby("hour").volume.sum()
show("by_hour", pd.DataFrame({"crashes": cr.groupby("hour").size().reindex(range(24), fill_value=0),
                              "pct_crashes": (cr.groupby("hour").size() / cr.hour.notna().sum() * 100).reindex(range(24), fill_value=0).round(2),
                              "pct_traffic": (hp / hp.sum() * 100).round(2)}))
show("by_weekday", cr.date.dt.day_name().value_counts().to_frame("crashes"))

# --- Conditions --------------------------------------------------------------
for col in ["surface_cat", "weather_cat", "light_cat"]:
    t = pd.crosstab(cr[col], cr.segment).reindex(columns=SEG, fill_value=0)
    t["total"] = t.sum(axis=1)
    t["ka"] = cr[cr.ka][col].value_counts()
    show(f"{col}_by_segment", t.fillna(0).astype(int).sort_values("total", ascending=False))
show("night_share", pd.DataFrame({
    "pct_9pm_5am": cr.groupby("segment").hour.apply(lambda s: s.isin([21, 22, 23, 0, 1, 2, 3, 4]).mean() * 100),
    "pct_dark": cr.groupby("segment").dark.mean() * 100}).reindex(SEG).round(1))
dk = cr[cr.dark]
show("dark_injury_by_lighting", pd.crosstab(dk.light_cat, dk.inj, margins=True))
lit = cr[cr.light_cat == "Dark, lighted"]
show("lighted_dark_crash_mp", lit.groupby("segment").mp.describe().round(1))

# --- Drivers -----------------------------------------------------------------
show("tu1_action_top", cr.tu1_action.str.strip().str.title().value_counts().head(12).to_frame())
show("tu1_human_factor_top", cr.tu1_human_factor.str.strip().str.title().value_counts().head(12).to_frame())

# --- Severe crashes ----------------------------------------------------------
ka = cr[cr.ka]
prof = pd.Series({
    "ka_crashes": len(ka), "fatal_crashes": (ka.severity == "K").sum(),
    "killed": ka.inj4.sum(), "seriously_injured": ka.inj3.sum(),
    "canyon": (ka.segment == "Big Thompson Canyon").sum(),
    "motorcycle": ka.motorcycle.sum(), "motorcycle_all_crashes": cr.motorcycle.sum(),
    "single_vehicle": ka.single_vehicle.sum(), "rollover": (ka.crash_cat == "Rollover").sum(),
    "on_curve": ka.on_curve.sum(), "apr_sep": ka.month.between(4, 9).sum(),
    "all_apr_sep_pct": cr.month.between(4, 9).mean() * 100,
    "daylight": (ka.light_cat == "Daylight").sum(), "dry": (ka.surface_cat == "Dry").sum(),
    "alcohol": ka.alcohol.sum(), "alcohol_all_pct": cr.alcohol.mean() * 100,
    "driver_65plus": (ka.tu1_age >= 65).sum(), "driver_under25": (ka.tu1_age < 25).sum(),
    "age_known": ka.tu1_age.notna().sum(), "sunday": (ka.date.dt.dayofweek == 6).sum(),
})
show("ka_profile_current", prof.round(1).to_frame("value"))
moto = cr[cr.motorcycle]
show("motorcycle_by_segment", pd.crosstab(moto.segment, moto.ka, margins=True))
show("ka_crash_cat", ka.crash_cat.value_counts().to_frame())
show("ka_action", ka.tu1_action.str.strip().str.title().value_counts().head(8).to_frame())

# --- Wildlife ----------------------------------------------------------------
an = cr_all[cr_all.crash_cat == "Wild animal"]
show("animal_by_year", an.groupby("year").size().to_frame("wild_animal_crashes"))
anc = an[an.year.isin(CURRENT_YEARS)].copy()
show("animal_species_current", anc.animal_species.value_counts().to_frame())
anc["mi"] = np.floor(anc.mp).clip(upper=91).astype(int)
show("animal_per_mile_current", anc.mi.value_counts().sort_index().to_frame())
show("animal_by_month_current", anc.month.value_counts().sort_index().to_frame())
show("animal_light_current", anc.light_cat.value_counts().to_frame())
show("animal_severity_current", anc.severity.value_counts().to_frame())
show("animal_by_segment_current", anc.segment.value_counts().to_frame())

# --- Weather at crash time (hourly reanalysis) ------------------------------
pts = {"Estes Park (town)": "estes_park", "Big Thompson Canyon": "drake",
       "Canyon mouth to Loveland": "canyon_mouth", "Loveland (urban)": "loveland"}
hw_ = []
for s, p in pts.items():
    h = pd.read_csv(R / "weather" / f"hourly_{p}.csv", parse_dates=["time"])
    h = h[h.time.dt.year.isin(CURRENT_YEARS)]
    h["segment"] = s
    hw_.append(h)
hw_ = pd.concat(hw_)
hw_["wet"] = (hw_.precipitation > 0.01) | (hw_.snowfall > 0)
cr["hour_ts"] = cr.datetime.dt.floor("h")
m = cr.merge(hw_[["segment", "time", "wet"]], left_on=["segment", "hour_ts"], right_on=["segment", "time"], how="left")
show("precip_share", pd.DataFrame({"pct_hours_wet": hw_.groupby("segment").wet.mean() * 100,
                                   "pct_crashes_wet": m.groupby("segment").wet.mean() * 100}).reindex(SEG).round(1))

# --- Weather climatology & exposure context ---------------------------------
nd = pd.read_csv(R / "weather" / "noaa_daily.csv", parse_dates=["date"])
nd["year"] = nd.date.dt.year
w = nd[nd.year.between(2008, 2025)]
yrs = w.groupby("station_label").year.nunique()
clim = w.groupby("station_label").agg(precip_in=("precip_in", "sum"), snow_in=("snowfall_in", "sum"),
                                      freeze_days=("tmin_f", lambda s: (s <= 32).sum()))
show("weather_climate", clim.div(yrs, axis=0).round(1))
