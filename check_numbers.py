"""Check that numbers written in the report match the analysis outputs.

1. Computes the key numbers from data/processed and writes them to
   data/processed/key_numbers.json (the single source for hand-written text).
2. Scans text files (the built web page, or a plain-text export of the
   report doc) for phrases that state those numbers, and lists every place
   where the stated value differs from the data.

Usage:
  python check_numbers.py                      # page only
  python check_numbers.py report_export.txt    # page + a doc export
Exit code 1 if any mismatch is found.
"""
import html
import json
import re
import sys

import pandas as pd

from config import PROCESSED, ROOT

EDA = PROCESSED / "eda"


def key_numbers():
    c = pd.read_csv(PROCESSED / "crashes_us34.csv", low_memory=False)
    c = c[c.year.between(2019, 2025)]
    seg = pd.read_csv(EDA / "crashes_by_segment.csv").set_index("segment")
    cv = pd.read_csv(EDA / "curves_by_section.csv").set_index("section")
    pk = pd.read_csv(EDA / "project_packages.csv")
    can, love = seg.loc["Big Thompson Canyon"], seg.loc["Loveland (urban)"]
    k = {
        "crashes": int(len(c)),
        "ka": int(c.severity.isin(["K", "A"]).sum()),
        "k": int((c.severity == "K").sum()),
        "killed": int(c.inj4.sum()),
        "canyon_crashes": int(can.crashes),
        "canyon_ka": int(can.ka),
        # rates from counts and vehicle-miles (the CSV rate columns are pre-rounded)
        "canyon_ka_rate": round(float(can.ka / can.mvmt * 100), 1),
        "loveland_ka_rate": round(float(love.ka / love.mvmt * 100), 1),
        "corridor_ka_rate": round(float(seg.loc["Corridor"].ka / seg.loc["Corridor"].mvmt * 100), 1),
        "canyon_vs_loveland": round(float((can.ka / can.mvmt) / (love.ka / love.mvmt)), 1),
        "canyon_curves": int(cv.loc["Big Thompson Canyon"].curves),
        "rural_package_mid": round(float(pk.iloc[0].ka_10yr_mid)),
        "loveland_package_mid": round(float(pk.iloc[1].ka_10yr_mid)),
    }
    (PROCESSED / "key_numbers.json").write_text(json.dumps(k, indent=2))
    return k


def fmt(v):
    return f"{v:,}" if isinstance(v, int) else f"{v}"


# (key, regex with one capture group). Patterns are written to match how the
# report states each number, and to avoid matching subsets.
CHECKS = [
    ("killed", r"\b(\d+) (?:people )?killed\b|killing (\d+)\b"),
    ("crashes", r"of the ([\d,]{5}) crashes|([\d,]{5}) crashes, 2019"),
    ("ka", r"of (?:the |all )?(\d+) fatal (?:and|or) serious|(\d+) fatal or serious crashes,? 2019|Fatal and serious, 2019–2025\W+(\d+)"),
    ("canyon_ka", r"\b(\d+) of (?:the )?72 fatal[^.]{0,80}canyon|[Cc]anyon had (\d+) of them|it had (\d+) of the 72"),
    ("canyon_vs_loveland", r"(\d\.\d) times Loveland"),
    ("loveland_ka_rate", r"Loveland's (\d+\.\d)\b|Loveland's level \((\d+\.\d)\)|Loveland \(urban\)\W+20\W+9\W+(\d+\.\d)"),
    ("canyon_ka_rate", r"canyon rate of (\d+\.\d)|severe-crash rate is (\d+\.\d)(?! times)|Big Thompson Canyon\W+41\W+12\W+(\d+\.\d)"),
    ("canyon_curves", r"(\d+) curves in the canyon|It has (\d+) of the corridor's"),
    ("rural_package_mid", r"[Rr]ural two-lane package[^.]{0,120}?midpoint (\d+)"),
]


def check_text(name, text, k):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\s+", " ", text)
    bad = 0
    for key, pat in CHECKS:
        for m in re.finditer(pat, text):
            got = next(g for g in m.groups() if g)
            want = fmt(k[key])
            if got.replace(",", "") != want.replace(",", ""):
                bad += 1
                ctx = text[max(0, m.start() - 60): m.end() + 40]
                print(f"  MISMATCH [{name}] {key}: text says {got}, data says {want}\n      …{ctx}…")
    return bad


def main():
    k = key_numbers()
    print("key numbers:", json.dumps(k))
    targets = [("web page", (ROOT / "web" / "us34_safety.html").read_text())]
    targets += [(p, open(p).read()) for p in sys.argv[1:]]
    bad = sum(check_text(n, t, k) for n, t in targets)
    print("OK: no mismatches" if not bad else f"{bad} mismatch(es)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
