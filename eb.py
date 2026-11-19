"""Empirical Bayes network screening for US 34, MP 65-88 (rural two-lane).

Method: Highway Safety Manual (HSM, 1st ed.) Part C, Chapter 10, rural
two-lane two-way roadway segments.

  N_spf  = AADT x L x 365 x 1e-6 x e^-0.312            (all crashes per year)
  N_pred = N_spf x CMF_lane x CMF_shoulder x CMF_curve x CMF_grade x C
  CMF_curve = (1.55 Lc + 80.2/R) / (1.55 Lc)           (no spiral transitions)
  k = 0.236 / L                                        (overdispersion)
  w = 1 / (1 + k x sum(N_pred over the years))
  N_expected = w x N_pred + (1 - w) x N_observed

C (calibration) makes total predicted = total observed over the study area,
done separately for all crashes and for fatal + serious (K+A), so the K+A
model has the same shape with its own level. Sites are ranked by EXCESS =
N_expected - N_pred: how many more crashes a site has than a typical
rural two-lane site with the same traffic and geometry. EB pulls short,
low-count sites towards the prediction, which removes most of the
regression-to-the-mean and multiple-testing noise in raw counts.

Intersection and intersection-related crashes are excluded, as in the HSM
segment models; they are screened with the intersection projects instead.

Segments: each curve from curves.py (padded by PAD_MI each side, since
crashes on curve approaches are curve-related) and the tangents between.

Outputs (data/processed/eda/): eb_segments.csv, eb_hot_zones.csv
"""
import numpy as np
import pandas as pd

from config import PROCESSED, RAW

EDA = PROCESSED / "eda"
YEARS = range(2019, 2026)
A, B = 65.0, 88.0
PAD_MI = 0.02
P_RA = 0.574          # HSM share of crashes related to lane/shoulder width


def cmf_lane(w):
    t = {9: 1.50, 10: 1.30, 11: 1.05, 12: 1.00}
    ra = t.get(int(round(min(max(w, 9), 12))), 1.0)
    return (ra - 1) * P_RA + 1


def cmf_shoulder(w):
    xs, ys = [0, 2, 4, 6, 8], [1.50, 1.30, 1.15, 1.00, 0.87]
    ra = float(np.interp(min(max(w, 0), 8), xs, ys))
    return (ra - 1) * P_RA + 1


def cmf_grade(g):
    g = abs(g)
    return 1.00 if g <= 3 else 1.10 if g <= 6 else 1.16


def segments():
    cv = pd.read_csv(EDA / "curves_inventory.csv")
    cv = cv[(cv.mp_mid >= A) & (cv.mp_mid < B)].sort_values("mp_start")
    segs, pos = [], A
    for c in cv.itertuples():
        a, b = max(c.mp_start - PAD_MI, pos), min(c.mp_end + PAD_MI, B)
        if a - pos > 0.005:
            segs.append(("tangent", pos, a, np.nan, 0, -1))
        segs.append(("curve", a, b, c.min_radius_ft, c.deflection_deg, c.curve_id))
        pos = b
    if B - pos > 0.005:
        segs.append(("tangent", pos, B, np.nan, 0, -1))
    s = pd.DataFrame(segs, columns=["kind", "mp_start", "mp_end", "radius_ft", "deflection_deg", "curve_id"])
    s["L"] = s.mp_end - s.mp_start
    return s


def attach_road(s):
    g = pd.read_csv(RAW / "roadway" / "geometrics.csv")
    g["shld"] = ((g.PRIOUTSHLDWD.clip(upper=8) + g.SECOUTSHLDWD.clip(upper=8)) / 2)
    g["lane"] = (g.PRITHRULNWD + g.SECTHRULNWD) / 2
    cl = pd.read_csv(RAW / "roadway" / "centerline_tenth_mile.csv")
    cl = cl[cl.ROUTE == "034A"]
    ad = pd.read_csv(RAW / "roadway" / "aadt_history_long.csv")
    ad["year"] = pd.to_numeric(ad.year.astype(str).str.extract(r"(\d{4})")[0])
    lane, shld, grade, aadt = [], [], [], {y: [] for y in YEARS}
    for r in s.itertuples():
        m = (r.mp_start + r.mp_end) / 2
        gg = g[(g.REFPT <= m) & (g.ENDREFPT > m)]
        lane.append(gg.lane.mean() if len(gg) else 12)
        shld.append(gg.shld.mean() if len(gg) else 4)
        cc = cl[cl.REFPT.between(r.mp_start - 0.1, r.mp_end + 0.1)]
        grade.append(cc.grade_pct.abs().mean() if len(cc) else 0)
        for y in YEARS:
            a = ad[(ad.year == y) & (ad.refPt <= m) & (ad.endRefPt > m)].aadt
            aadt[y].append(float(a.mean()) if len(a) else np.nan)
    s["lane_ft"], s["shoulder_ft"], s["grade_pct"] = lane, shld, grade
    for y in YEARS:
        s[f"aadt_{y}"] = pd.Series(aadt[y]).ffill().bfill().values
    return s


def main():
    s = attach_road(segments())
    # curve CMF uses the curve's own length (without padding)
    lc = (s.L - 2 * PAD_MI).clip(lower=0.01)
    s["cmf_curve"] = np.where(s.kind == "curve", (1.55 * lc + 80.2 / s.radius_ft) / (1.55 * lc), 1.0)
    s["cmf_lane"] = s.lane_ft.map(cmf_lane)
    s["cmf_shoulder"] = s.shoulder_ft.map(cmf_shoulder)
    s["cmf_grade"] = s.grade_pct.map(cmf_grade)
    cmf = s.cmf_curve * s.cmf_lane * s.cmf_shoulder * s.cmf_grade
    spf = sum(s[f"aadt_{y}"] * s.L * 365 * 1e-6 * np.exp(-0.312) for y in YEARS)
    s["n_uncal"] = spf * cmf

    cr = pd.read_csv(EDA / "curves_crash_positions.csv")
    cr = cr[cr.year.between(min(YEARS), max(YEARS)) & cr.mp_pos.between(A, B)]
    # HSM segment models exclude intersection crashes (driveway crashes stay in)
    at_int = cr.road_desc.astype(str).str.lower().str.contains("intersection") & \
        ~cr.road_desc.astype(str).str.lower().str.contains("non-")
    print(f"excluded {at_int.sum()} intersection crashes ({(at_int & cr.severity.isin(['K', 'A'])).sum()} K+A)")
    cr = cr[~at_int].copy()
    cr["ka"] = cr.severity.isin(["K", "A"])
    idx = np.searchsorted(s.mp_end.values, cr.mp_pos.values, side="left").clip(0, len(s) - 1)
    cr["seg"] = idx
    s["obs"] = cr.groupby("seg").size().reindex(s.index, fill_value=0)
    s["obs_ka"] = cr[cr.ka].groupby("seg").size().reindex(s.index, fill_value=0)

    out = {}
    for tag, col in [("", "obs"), ("_ka", "obs_ka")]:
        C = s[col].sum() / s.n_uncal.sum()
        pred = s.n_uncal * C
        k = 0.236 / s.L
        w = 1 / (1 + k * pred)
        exp_ = w * pred + (1 - w) * s[col]
        s[f"pred{tag}"] = pred.round(2)
        s[f"w{tag}"] = w.round(3)
        s[f"eb{tag}"] = exp_.round(2)
        s[f"excess{tag}"] = (exp_ - pred).round(2)
        out[tag] = C
    s["excess_ka_per_mi"] = (s.excess_ka / s.L).round(2)
    s = s.round({"mp_start": 3, "mp_end": 3, "L": 3, "lane_ft": 1, "shoulder_ft": 1, "grade_pct": 1,
                 "cmf_curve": 2, "cmf_lane": 2, "cmf_shoulder": 2, "cmf_grade": 2, "n_uncal": 2})
    s.to_csv(EDA / "eb_segments.csv", index=False)

    # half-mile zones: sum EB excess (K+A) over sliding windows, keep non-overlapping best
    starts = np.round(np.arange(A, B - 0.5 + 1e-9, 0.05), 2)
    rows = []
    for a in starts:
        b = a + 0.5
        f = ((s.mp_end.clip(upper=b) - s.mp_start.clip(lower=a)).clip(lower=0) / s.L)
        rows.append({"mp_start": a, "mp_end": b, "excess_ka": (f * s.excess_ka).sum(),
                     "excess": (f * s.excess).sum(), "obs_ka": (f * s.obs_ka).sum(),
                     "obs": (f * s.obs).sum(), "pred_ka": (f * s.pred_ka).sum(), "pred": (f * s.pred).sum()})
    w = pd.DataFrame(rows)
    zones = []
    for i in w.sort_values("excess_ka", ascending=False).index:
        if any(abs(w.mp_start[i] - z) < 0.5 for z in [q.mp_start for q in zones]):
            continue
        zones.append(w.loc[i])
        if len(zones) == 8:
            break
    zz = pd.DataFrame(zones).round(2)
    zz.to_csv(EDA / "eb_hot_zones.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"{len(s)} segments ({(s.kind == 'curve').sum()} curves); calibration C = {out['']:.2f} (all), "
          f"{out['_ka']:.3f} (K+A, relative to the all-crash SPF)")
    cols = ["kind", "mp_start", "mp_end", "radius_ft", "cmf_curve", "obs", "pred", "eb", "excess",
            "obs_ka", "pred_ka", "eb_ka", "excess_ka", "w_ka"]
    print("Top by EB excess, fatal + serious:")
    print(s.sort_values("excess_ka", ascending=False)[cols].head(12).to_string(index=False))
    print("Top by EB excess, all crashes:")
    print(s.sort_values("excess", ascending=False)[cols].head(10).to_string(index=False))
    print(zz.to_string(index=False))


if __name__ == "__main__":
    main()
