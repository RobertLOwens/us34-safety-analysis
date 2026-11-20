# US 34 Safety Analysis — Loveland to Estes Park

**Interactive crash map and findings:** https://robertlowens.github.io/us34-safety-analysis/ (rebuilt by `build_web.py` into `docs/index.html`).

Data collection for a safety report on US 34 (CDOT route `034A`), MP 61–92:
downtown Estes Park → Big Thompson Canyon → US 287 in Loveland.
Corridor limits and sub-segments are in `config.py`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Collect data (run in this order)

| Script | What it pulls | Output |
|---|---|---|
| `collect_roadway.py` | CDOT segment data (speed limit, lanes, lane and shoulder widths, terrain, AADT, geometrics, access category), AADT history 1997–2025, tenth-mile centerline with curvature and USGS elevation/grade | `data/raw/roadway/` |
| `collect_crashes.py` | US 34 crashes from CDOT statewide crash listings (manual download, see below) | `data/processed/crashes_us34.csv` |
| `collect_exposure.py` | CDOT continuous counter 000205 (east Estes Park): monthly ADT since 1991 and hourly counts by direction since 2013; Rocky Mountain NP monthly visits; CDOT geohazard (rockfall) events | `data/raw/exposure/` |
| `collect_wildlife.py` | CDOT animal-involved crashes 2018–2023 snapped to US 34; CPW elk ranges (winter, severe winter, concentration, highway crossings) per tenth mile; CDOT wildlife priority scores | `data/raw/wildlife/` |
| `collect_lighting.py` | OpenStreetMap street lamps and `lit` tags near US 34 (none mapped on the corridor; lighting is inferred from crash records instead) | `data/raw/lighting/` |
| *(hand-curated)* | Closures and construction: 2013 flood, 2016–2019 reconstruction, 2020 Cameron Peak Fire | `data/raw/events/corridor_events.csv` |
| `collect_weather.py` | NOAA GHCN-Daily (Estes Park, Waterdale/canyon mouth, Loveland) and hourly ERA5 reanalysis at 4 points along the road | `data/raw/weather/` |

### Crash files (manual step)

CDOT's yearly statewide listings (2007–2026) are behind a document viewer
and can't be scripted. Download them from
<https://www.codot.gov/safety/traffic-safety/data-analysis/crash-data>
and save them anywhere in this folder (currently `data/raw/crashes/`).
Duplicate copies of the same file name are read once. `collect_crashes.py` finds every
`.xlsx` file and caches the US 34 rows in `data/raw/crashes/` (about a minute
per file the first time). Use `--rescan` to rebuild the cache.

## Notes

- Crashes are matched by CDOT's route and milepost. A crash coded to US 34
  with no milepost falls back to its coordinates (within 150 m of the
  centerline). Crashes on side streets near US 34 are excluded.
- Severity is KABCO from the injury counts: K = fatal, A = serious, B = minor, C = possible, O = property damage only.
- The 2007–2020 crash files use a different column layout (RTE/SEC/MP, upper-case
  codes, missing coordinates stored as 0). Each file is normalised on its own, and
  codes are harmonised into common categories (`crash_cat`, `light_cat`,
  `surface_cat`, `weather_cat`, `on_curve`, `motorcycle`, `alcohol`,
  `animal_species`).
- `eda.py` uses 2007–2025 for trends and 2019–2025 (post-rebuild) for current
  patterns; rates use vehicle-miles from the yearly AADT history.
- NOAA's Estes Park record has a gap from Jan 2021 to Jul 2022; the hourly
  reanalysis covers it.
- NHTSA FARS (fatal crashes) blocks automated access from here; CDOT listings
  include fatal crashes.

## Analysis and report (run in this order, or `python run_all.py`)

| Script | What it does | Output |
|---|---|---|
| `eda.py` | Exploratory tables: sections, trend, time, weather, severe crashes, wildlife, lighting | `data/processed/eda/*.csv` |
| `eda_glare_drivers.py` | Sun-glare model (sun ahead of the driver) and driver factors | `glare_*.csv`, `drivers_*.csv` |
| `curves.py` | Curve inventory from the CDOT route polyline (`data/raw/roadway/route_polyline.json`), crashes placed by GPS and matched to curves | `curves_*.csv` |
| `eb.py` | Empirical Bayes screening, HSM rural two-lane model with curve, lane, shoulder and grade adjustments, calibrated to MP 65–88; intersection crashes excluded | `eb_segments.csv`, `eb_hot_zones.csv` |
| `priorities.py` | Project list: FHWA reductions applied to targeted crashes, EB adjustment for sites picked for high counts, existing-feature adjustments, benefit-cost with MnDOT/FHWA crash costs | `project_priorities.csv`, `project_packages.csv` |
| `peers.py` | Same rates for six other Colorado canyon highways (rescans statewide crash files, slow) | `peer_corridors.csv` |
| `build_web.py` | Builds the visual page from `web/template.html` | `web/us34_safety.html` |
| `check_numbers.py` | Writes `data/processed/key_numbers.json` and flags numbers in the page (and an optional doc export) that no longer match the data | exit code 1 on mismatch |

```bash
.venv/bin/python run_all.py            # analysis, page, number check
.venv/bin/python run_all.py --peers    # plus the peer-corridor rescan
.venv/bin/python check_numbers.py report_export.txt   # also check an exported copy of the report
```

Hand-collected inputs: `data/raw/roadway/streetview_survey.csv` (Street View survey of rumble strips, chevrons, guardrail: 71 points, every half mile MP 65–88 plus the approaches to the 7 most-severe curves and the 4 other busiest curves; 2023–2024 imagery; center-line rumble strips confirmed with close-up views; `priorities.py` reads it for the center-line rumble strip adjustment) and `data/raw/roadway/cdot_projects_us34.csv` (CDOT project records on the corridor).

Map layers: `data/raw/places/town_boundaries.geojson` (Estes Park and Loveland incorporated limits, US Census Bureau TIGERweb, Places layer 4) and `data/raw/places/landmarks.json` (CR 43 junction to Glen Haven at MP 75.6 and Loveland Fire Rescue Authority Station 8 at MP 77.0, from OpenStreetMap; the CR 43 line is simplified from OSM way 845782154).

Project detail on the page: `priorities.py` also writes `data/processed/eda/project_details.json` (plain-language description, treated milepost ranges, FHWA Proven Safety Countermeasures source links) and `project_targets.csv` (row index in `crashes_us34.csv` of every crash each project targets). `build_web.py` turns these into the clickable project cards and map highlights.

Existing conditions on the project views (`build_web.py: existing()`): shoulder widths from CDOT Highways Geometrics (2025, complete, one side only); guardrail (`barrier=guard_rail`) and traffic signals from OpenStreetMap (`data/raw/places/osm_features.json`, partial for guardrail); rumble strips, chevrons and guardrail from the 71-point Street View survey (guardrail shows as squares under the road). CDOT's rumble strip fields are empty for US 34 and CDOT publishes no guardrail inventory.

Project targets (`priorities.py`, revised after a code review): each project counts only the crash types its FHWA reduction was measured on, where it would be built. Curve delineation applies 60% only at the 10 worst curves (16% elsewhere); center-line rumble strips count opposite-direction sideswipes only; edge lines and SafetyEdge count run-off-road and opposing-direction crashes; shoulder rumble strips count MP 75.5-88 where CDOT records 4 ft or more; intersection redesign counts Loveland's two unsignalized intersections with the most angle crashes; lighting counts crashes at unlit intersections; backplates count crashes within 0.05 mi of a signal (signals from OpenStreetMap). Methods are written up in section 3 of the report.
