"""Project priority list: estimated crashes prevented per countermeasure.

For each project, target crashes = 2019-2025 crashes of the types and places
the countermeasure treats (per year). Estimated prevented = target x FHWA
Proven Safety Countermeasures reduction (low-high). Effects overlap between
projects that treat the same crashes, so they must not be summed.

Output: data/processed/eda/project_priorities.csv
"""
import json

import pandas as pd

from config import PROCESSED, RAW

YEARS = (2019, 2025)
N_YEARS = YEARS[1] - YEARS[0] + 1

c = pd.read_csv(PROCESSED / "crashes_us34.csv", low_memory=False)
c = c[c.year.between(*YEARS)].copy()
c["ka"] = c.severity.isin(["K", "A"])
c["inj"] = c.severity.isin(["K", "A", "B", "C"])
rd = c.road_desc.astype(str).str.strip().str.lower()
c["non_int"] = rd.str.contains("non-intersection") | rd.eq("nan")
c["at_int"] = rd.str.contains("intersection") & ~rd.str.contains("non-")
canyon = c.segment == "Big Thompson Canyon"
rural2 = c.segment.isin(["Big Thompson Canyon", "Canyon mouth to Loveland"])
loveland = c.segment == "Loveland (urban)"
departure = c.crash_cat.isin(["Fixed object", "Rollover"])
on_curve = c.on_curve.astype(str).eq("True")
head_on = c.crash_cat.eq("Head-on")
opp_sideswipe = c.crash_cat.eq("Sideswipe") & c.vehicles.ge(2)
dark = c.light_cat.astype(str).str.startswith("Dark")
unlit = c.light_cat.astype(str).eq("Dark, unlighted")
angle = c.crash_cat.eq("Angle or turning")
# opposite-direction sideswipes only: the "Sideswipe" category mixes both kinds
_ct = (c.crash_type.astype(str) + " " + c.crash_type2.astype(str)).str.upper()
opp_sideswipe = c.crash_cat.eq("Sideswipe") & _ct.str.contains("OPPOSITE")
# crashes an edge treatment can affect: leaving the road or crossing into the other lane
departure_or_opposing = departure | head_on | opp_sideswipe


def _signal_mps():
    """Signalized intersections on US 34 (OpenStreetMap highway=traffic_signals
    within 120 m, heads within 0.08 mi grouped), as mileposts."""
    import numpy as np
    cl = pd.read_csv(RAW / "roadway" / "centerline_tenth_mile.csv")
    cl = cl[(cl.ROUTE == "034A") & cl.REFPT.between(60.9, 92.05)]
    els = json.loads((RAW / "places" / "osm_features.json").read_text())["elements"]
    mps = []
    for e in els:
        if e.get("tags", {}).get("highway") != "traffic_signals":
            continue
        d = np.hypot((cl.lat.values - e["lat"]) * 111000, (cl.lon.values - e["lon"]) * 85000)
        if d.min() < 120:
            mps.append(float(cl.REFPT.values[d.argmin()]))
    groups = []
    for m in sorted(mps):
        if groups and m - groups[-1][-1] < 0.08:
            groups[-1].append(m)
        else:
            groups.append([m])
    return [sum(g) / len(g) for g in groups]


SIGNALS = _signal_mps()
near_signal = c.mp.map(lambda m: min(abs(m - s) for s in SIGNALS) <= 0.05)


def _top_unsignalized(mask, n=2):
    """The n unsignalized intersections (crashes grouped within 0.05 mi) with the
    most crashes in `mask`; returns the mask limited to those intersections."""
    pts = c[mask].mp.sort_values()
    groups = []
    for i, m in pts.items():
        if groups and m - groups[-1][-1][1] <= 0.05:
            groups[-1].append((i, m))
        else:
            groups.append([(i, m)])
    groups = sorted(groups, key=len, reverse=True)[:n]
    keep = {i for g in groups for i, _ in g}
    return mask & c.index.isin(keep), [round(sum(m for _, m in g) / len(g), 1) for g in groups]


RCUT_MASK, RCUT_SITES = _top_unsignalized(loveland & angle & c.at_int & ~near_signal)
_, LIGHT_SITES = _top_unsignalized(loveland & unlit & c.at_int, n=99)

# Shoulder rumble strips need about 4 ft of shoulder. CDOT records 4 ft or more
# on most of MP 65-88, but the Street View survey saw 1-3 ft against rock
# through MP 65-75.5, so only MP 75.5-88 where CDOT records 4 ft or more counts.
_geo = pd.read_csv(RAW / "roadway" / "geometrics.csv")
_geo = _geo[(_geo.ROUTE == "034A") & (_geo.ENDREFPT > 75.5) & (_geo.REFPT < 88)]
_wide = _geo[_geo.PRIOUTSHLDWD >= 4]
SRS_MILES = float((_wide.ENDREFPT.clip(upper=88) - _wide.REFPT.clip(lower=75.5)).sum())
srs_ok = c.mp.map(lambda m: bool(((_wide.REFPT <= m) & (_wide.ENDREFPT > m)).any()))

# Sequential dynamic chevrons (60%) go on the ~10 worst canyon curves only;
# plain chevrons (16%) everywhere else.
_rk = pd.read_csv(PROCESSED / "eda" / "curves_ranked.csv")
_rk = _rk[(_rk.mp_mid >= 65) & (_rk.mp_mid < 83.5)].head(10)
worst10 = pd.Series(False, index=c.index)
for _r in _rk.itertuples():
    worst10 |= c.mp.between(_r.mp_start - 0.05, _r.mp_end + 0.05)

# (rank label, where, mask, low %, high %, FHWA basis, relative cost).
# A reduction may be a number or a per-crash Series (percent).
PROJECTS = [
    ("Curve delineation package", "Canyon, MP 65-83.5",
     canyon & departure & on_curve, 16, worst10.map({True: 60, False: 16}),
     "Chevrons 16% (non-intersection FI) on every curve; sequential dynamic chevrons 60% (FI) on the 10 worst", "Low"),
    ("Center line rumble strips", "MP 65-88 (two-lane)",
     rural2 & (head_on | opp_sideswipe), 44, 64,
     "44-64% head-on FI, rural two-lane", "Low"),
    ("Wider edge lines", "MP 65-88 (two-lane)",
     rural2 & c.non_int & departure_or_opposing, 0, 37,
     "Up to 37% non-intersection FI, rural two-lane (departure and opposing-direction crashes)", "Low"),
    ("High-friction surface on worst curves", "Curves at MP 66.9-67.3, 70.5-72.2, 81.9-82.5",
     rural2 & on_curve & (c.mp.between(66.85, 67.35) | c.mp.between(70.5, 72.2) | c.mp.between(81.9, 82.55)), 0, 48,
     "Up to 48% injury crashes at curves", "Medium"),
    ("Shoulder rumble strips where shoulders allow", f"MP 75.5-88 where the shoulder is 4 ft or wider ({SRS_MILES:.1f} mi)",
     rural2 & departure & srs_ok, 13, 51,
     "13-51% single-vehicle run-off-road FI, rural two-lane", "Low"),
    ("Intersection redesign (roundabout or RCUT)",
     "Loveland's 2 busiest unsignalized intersections (MP " + " and ".join(f"{m:.1f}" for m in RCUT_SITES) + ")",
     RCUT_MASK, 22, 82,
     "Signalized to RCUT 22%; stop-controlled to roundabout 82% (FI)", "High"),
    ("SafetyEdge at next repaving", "MP 65-88 (two-lane)",
     rural2 & departure_or_opposing, 11, 11,
     "11% FI (departure and opposing-direction crashes)", "Low (with repaving)"),
    ("Lighting gaps", "Unlit Loveland intersections, MP 88-92",
     loveland & unlit & c.at_int, 33, 38,
     "33-38% night crashes at intersections", "Medium"),
    ("Signal backplates", "Loveland signals",
     loveland & angle & c.at_int & near_signal, 15, 15,
     "15% total crashes", "Low"),
]


def _r(x):
    """Reduction as a per-crash Series (fraction)."""
    return (x if isinstance(x, pd.Series) else pd.Series(float(x), index=c.index)) / 100

# ---------------------------------------------------------------------------
# Existing features and regression to the mean
# Street View survey (data/raw/roadway/streetview_survey.csv, 2023-2024
# imagery, every half mile MP 65-88 plus both approaches to the top curves):
# the share of points with a center-line rumble strip, among points where it
# could be judged ("unclear" left out), is taken as already treated.
_SV = pd.read_csv(RAW / "roadway" / "streetview_survey.csv")
_CL = _SV[_SV.mp.between(65, 88) & _SV.centerline_rumble.isin(["yes", "no"])].centerline_rumble
CLRS_YES, CLRS_N = int((_CL == "yes").sum()), int(len(_CL))
# CDOT project 23015 "R4 Signal Head Backplate Install" covers MP 62-101, so
# Loveland's signals may already have backplates: kept, but flagged.
UNTREATED = {"Center line rumble strips": 1 - CLRS_YES / CLRS_N}
STATUS = {"Signal backplates": "May already be installed (CDOT project 23015)",
          "Center line rumble strips": f"Seen at {CLRS_YES} of the {CLRS_N} points where it could be judged; "
                                       f"benefits cut by {CLRS_YES / CLRS_N:.0%}"}


def eb_factor():
    """The HFST sites were chosen because they had many crashes; their counts
    regress toward the mean. Scale them by Empirical Bayes expected / observed."""
    s = pd.read_csv(PROCESSED / "eda" / "eb_segments.csv")
    mid = (s.mp_start + s.mp_end) / 2
    m = mid.between(66.85, 67.35) | mid.between(70.5, 72.2) | mid.between(81.9, 82.55)
    return s[m].eb_ka.sum() / s[m].obs_ka.sum(), s[m].eb.sum() / s[m].obs.sum()


EB_KA, EB_ALL = eb_factor()
EB_ADJ = {"High-friction surface on worst curves": (EB_KA, EB_ALL)}

# ---------------------------------------------------------------------------
# Benefit-cost
# Crash costs: comprehensive cost per crash, MnDOT Table A.1 (SFY 2025; FHWA
# 2018 crash-cost method with USDOT 2021 value of a statistical life).
# K and A are combined at this corridor's K/A mix, as the HSM recommends, so
# one fatal crash does not swing a result.
COST = {"K": 15.9e6, "A": 1.7e6, "B": 380e3, "C": 180e3}
DISCOUNT = 0.031
K_SHARE = (c.severity == "K").sum() / c.ka.sum()
COST_KA = K_SHARE * COST["K"] + (1 - K_SHARE) * COST["A"]

# (low $, high $, service life in years, basis). Planning-level costs.
CAPITAL = {
    "Curve delineation package": (0.55e6, 1.1e6, 10,
        "Chevrons on 89 canyon curves (~$6K each), plus sequential dynamic chevrons on ~10 worst ($30-50K each)"),
    "Center line rumble strips": (0.12e6, 0.35e6, 10, "23 mi at $5-15K per mile"),
    "Wider edge lines": (0.03e6, 0.08e6, 4, "Extra cost of 6-inch over 4-inch lines when restriping, 46 line-miles"),
    "High-friction surface on worst curves": (1.0e6, 1.5e6, 10,
        "17 curves plus approaches, ~32,500 sq yd at $30-45"),
    "Shoulder rumble strips where shoulders allow": (2 * SRS_MILES * 2e3, 2 * SRS_MILES * 6e3, 10,
        f"{2 * SRS_MILES:.0f} side-miles (both sides of {SRS_MILES:.1f} mi) at $2-6K"),
    "Intersection redesign (roundabout or RCUT)": (5e6, 12e6, 25, "2 unsignalized intersections at $2.5-6M"),
    "SafetyEdge at next repaving": (0.03e6, 0.07e6, 15, "$1-3K per mile when repaving"),
    "Lighting gaps": (len(LIGHT_SITES) * 60e3, len(LIGHT_SITES) * 150e3, 20,
        f"{len(LIGHT_SITES)} unlit intersections with night crashes at $60-150K"),
    "Signal backplates": (0.05e6, 0.12e6, 10, "~15 signals at $3-8K"),
}


def pv_factor(n, i=DISCOUNT):
    return (1 - (1 + i) ** -n) / i


def add_benefit_cost(t):
    out = []
    for r in t.itertuples():
        lo_c, hi_c, life, cbasis = CAPITAL[r.project]
        # per-year crash cost treated, by severity, times each crash's reduction
        m = MASKS[r.project]
        f_ka, f_all = EB_ADJ.get(r.project, (1.0, 1.0))
        f_u = UNTREATED.get(r.project, 1.0)
        unit = c.severity.map({"K": COST_KA * f_ka, "A": COST_KA * f_ka,
                               "B": COST["B"] * f_all, "C": COST["C"] * f_all}).fillna(0) * m * f_u / N_YEARS
        pv = pv_factor(life)
        lo_r, hi_r = REDUCTIONS[r.project]
        b_lo, b_hi = (unit * lo_r).sum() * pv, (unit * hi_r).sum() * pv
        mid_c = (lo_c + hi_c) / 2
        out.append({"cost_low": lo_c, "cost_high": hi_c, "life_yr": life, "cost_basis": cbasis,
                    "benefit_pv_low": round(b_lo), "benefit_pv_high": round(b_hi),
                    "bc_low": round(b_lo / hi_c, 1), "bc_high": round(b_hi / lo_c, 1),
                    "bc_mid": round((b_lo + b_hi) / 2 / mid_c, 1),
                    "eb_adjust_ka": round(f_ka, 2), "untreated_share": round(f_u, 2),
                    "status": STATUS.get(r.project, "")})
    return pd.concat([t.reset_index(drop=True), pd.DataFrame(out)], axis=1)


MASKS, REDUCTIONS = {}, {}
rows = []
for name, where, m, lo, hi, basis, cost in PROJECTS:
    MASKS[name] = m
    lo_r, hi_r = _r(lo), _r(hi)
    REDUCTIONS[name] = (lo_r, hi_r)
    f_ka, f_all = EB_ADJ.get(name, (1.0, 1.0))
    f_u = UNTREATED.get(name, 1.0)
    w_ka = (m & c.ka) * f_ka * f_u / N_YEARS                       # per-crash yearly weight
    w_inj = (m & c.ka) * f_ka * f_u / N_YEARS + (m & c.inj & ~c.ka) * f_all * f_u / N_YEARS
    ka, inj = w_ka.sum(), w_inj.sum()
    # reductions shown in the table: weighted by the injury crashes treated
    eff = lambda r: round(100 * (w_inj * r).sum() / inj) if inj else round(100 * r[m].mean() if m.any() else 0)
    rows.append({"project": name, "where": where,
                 "target_ka_10yr": round(ka * 10, 1),
                 "target_injury_per_yr": round(inj, 1),
                 "reduction_low_pct": eff(lo_r), "reduction_high_pct": eff(hi_r),
                 "ka_prevented_10yr_low": round((w_ka * lo_r).sum() * 10, 1),
                 "ka_prevented_10yr_high": round((w_ka * hi_r).sum() * 10, 1),
                 "injury_prevented_10yr_low": round((w_inj * lo_r).sum() * 10, 1),
                 "injury_prevented_10yr_high": round((w_inj * hi_r).sum() * 10, 1),
                 "basis": basis, "relative_cost": cost})
t = pd.DataFrame(rows)
t["ka_mid"] = (t.ka_prevented_10yr_low + t.ka_prevented_10yr_high) / 2
t = add_benefit_cost(t)
t = t.sort_values(["ka_mid", "injury_prevented_10yr_high"], ascending=False)
t.to_csv(PROCESSED / "eda" / "project_priorities.csv", index=False)

# ---------------------------------------------------------------------------
# Per-project detail for the web page: what it is, where it goes (milepost
# ranges, or "curves" = the inventory curves inside those ranges), the FHWA
# source page, and which crashes it targets (row index of crashes_us34.csv).
PSC = "https://highways.dot.gov/safety/proven-safety-countermeasures/"
HFST_RANGES = [[66.85, 67.35], [70.5, 72.2], [81.9, 82.55]]
DETAILS = {
    "Curve delineation package": dict(
        existing=["chevrons", "rails"], 
        what="Chevron signs on every canyon curve, plus sequential flashing chevrons on the 10 worst curves, so drivers see how sharp each bend is in time to slow down. The 60% reduction counts only at those 10 curves; 16% elsewhere.",
        targets="Fixed-object and rollover crashes on canyon curves",
        ranges=[[65, 83.5]], show="curves",
        sources=[("FHWA: Enhanced delineation for horizontal curves", PSC + "enhanced-delineation-horizontal-curves")]),
    "Center line rumble strips": dict(
        existing=["clrs"], 
        what="Grooves milled into the center line that rumble when a driver drifts toward oncoming traffic.",
        targets="Head-on and opposite-direction sideswipe crashes, MP 65–88 (same-direction sideswipes left out)",
        ranges=[[65, 88]], show="range",
        sources=[("FHWA: Longitudinal rumble strips on two-lane roads", PSC + "longitudinal-rumble-strips-and-stripes-two-lane-roads")]),
    "Wider edge lines": dict(
        existing=["none:edge line width"], 
        what="6-inch edge lines instead of 4-inch, painted at the next restriping, so the road edge is easier to see at night and in bad weather.",
        targets="Run-off-road, head-on and opposite-direction sideswipe crashes away from intersections, MP 65–88",
        ranges=[[65, 88]], show="range",
        sources=[("FHWA: Wider edge lines", PSC + "wider-edge-lines")]),
    "High-friction surface on worst curves": dict(
        existing=["chevrons", "rails", "none:pavement friction"], 
        what="A bonded high-friction aggregate overlay on the worst curves, giving tires more grip when drivers brake or steer hard.",
        targets="Crashes on the curves with the most excess crashes after Empirical Bayes screening",
        ranges=HFST_RANGES, show="curves",
        sources=[("FHWA: Pavement friction management", PSC + "pavement-friction-management")]),
    "Shoulder rumble strips where shoulders allow": dict(
        existing=["shoulders", "srs", "rails"], 
        what="Rumble strips on the shoulder that alert drivers drifting off the road, only where the shoulder is wide enough: MP 75.5–88 where CDOT records 4 ft or more. Street View shows 1–3 ft against rock through MP 65–75.5, so that stretch is left out.",
        targets="Fixed-object and rollover crashes where the shoulder is 4 ft or wider, MP 75.5–88",
        ranges=[[75.5, 88]], show="range",
        sources=[("FHWA: Longitudinal rumble strips on two-lane roads", PSC + "longitudinal-rumble-strips-and-stripes-two-lane-roads")]),
    "Intersection redesign (roundabout or RCUT)": dict(
        existing=["signals"], 
        what="Rebuild Loveland's two unsignalized intersections with the most angle and turning crashes as a roundabout or a restricted-crossing U-turn. Most of Loveland's angle crashes, and half its severe ones, are at signals, which this does not treat.",
        targets="Angle and turning crashes at the 2 busiest unsignalized intersections",
        ranges=[[m - 0.1, m + 0.1] for m in sorted(RCUT_SITES)], show="range",
        sources=[("FHWA: Roundabouts", PSC + "roundabouts"),
                 ("FHWA: Reduced left-turn conflict intersections", PSC + "reduced-left-turn-conflict-intersections")]),
    "SafetyEdge at next repaving": dict(
        existing=["shoulders", "rails", "none:pavement edge shape"], 
        what="A 30° sloped pavement edge, shaped during repaving, so a tire that drops off the edge can climb back on without the driver over-steering.",
        targets="Run-off-road, head-on and opposite-direction sideswipe crashes, MP 65–88",
        ranges=[[65, 88]], show="range",
        sources=[("FHWA: SafetyEdge", PSC + "safetyedgesm")]),
    "Lighting gaps": dict(
        existing=["signals", "none:street lighting"], 
        what="Light the Loveland intersections that are dark now. Officers recorded most night crashes at Loveland intersections as already lighted, so only the unlit ones count.",
        targets="Night crashes at unlit Loveland intersections",
        ranges=[[88, 92.1]], show="range",
        sources=[("FHWA: Lighting", PSC + "lighting")]),
    "Signal backplates": dict(
        existing=["signals", "none:backplates on each signal"], 
        what="Black backplates with a reflective yellow border on the signal heads, so signals stand out against the sky and stay visible in a power outage.",
        targets="Angle and turning crashes at Loveland signals",
        ranges=[[88, 92.1]], show="range",
        sources=[("FHWA: Backplates with retroreflective borders", PSC + "backplates-retroreflective-borders")]),
}
assert set(DETAILS) == set(MASKS), "every project needs page details"
(PROCESSED / "eda" / "project_details.json").write_text(json.dumps(DETAILS, indent=1))
pd.DataFrame([(name, i) for name, m in MASKS.items() for i in c.index[m]],
             columns=["project", "row"]).to_csv(PROCESSED / "eda" / "project_targets.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(f"EB adjustment for HFST sites: K+A x{EB_KA:.2f}, all x{EB_ALL:.2f}; K+A cost ${COST_KA/1e6:.2f}M")
    print(t.drop(columns=["basis", "cost_basis"]).to_string(index=False))


def package(names, which):
    """Combined effect: per crash, multiply the remaining share (1 - r) of each
    project that treats it (standard way to combine CMFs; still optimistic)."""
    keep = pd.Series(1.0, index=c.index)
    for name, where, m, *_ in PROJECTS:
        if name in names:
            lo_r, hi_r = REDUCTIONS[name]
            r = {"low": lo_r, "high": hi_r, "mid": (lo_r + hi_r) / 2}[which]
            f = UNTREATED.get(name, 1.0) * EB_ADJ.get(name, (1.0, 1.0))[0]
            keep[m] *= (1 - r[m] * f)
    prevented = (1 - keep)
    return (prevented * c.ka).sum() / N_YEARS * 10, (prevented * c.inj).sum() / N_YEARS * 10


PACKAGES = {
    "Rural two-lane package (MP 65-88)": [
        "Curve delineation package", "Center line rumble strips", "Wider edge lines",
        "Shoulder rumble strips where shoulders allow", "High-friction surface on worst curves",
        "SafetyEdge at next repaving"],
    "Loveland intersection package (MP 88-92)": [
        "Intersection redesign (roundabout or RCUT)", "Lighting gaps", "Signal backplates"],
}

if __name__ == "__main__":
    out = []
    for p, names in PACKAGES.items():
        r = {w: package(names, w) for w in ("low", "mid", "high")}
        out.append({"package": p, **{f"ka_10yr_{w}": round(v[0], 1) for w, v in r.items()},
                    **{f"injury_10yr_{w}": round(v[1], 1) for w, v in r.items()}})
    out = pd.DataFrame(out)
    out.to_csv(PROCESSED / "eda" / "project_packages.csv", index=False)
    print(out.to_string(index=False))
    print("KA per 10 yr, all corridor:", round(c.ka.sum() / N_YEARS * 10, 1))
