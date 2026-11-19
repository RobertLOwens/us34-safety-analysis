"""Run the whole US 34 analysis in order, then check the numbers.

  python run_all.py              # analysis + web page + number check
  python run_all.py --collect    # also re-download source data first (slow)
  python run_all.py --peers      # also rescan statewide files for peer corridors

Stops at the first step that fails.
"""
import subprocess
import sys
import time

PY = sys.executable
COLLECT = ["collect_roadway.py", "collect_crashes.py", "collect_exposure.py",
           "collect_wildlife.py", "collect_lighting.py", "collect_weather.py"]
ANALYSIS = ["eda.py", "eda_glare_drivers.py", "curves.py", "eb.py", "priorities.py"]
BUILD = ["build_web.py", "check_numbers.py"]


def run(script, *args):
    t = time.time()
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    r = subprocess.run([PY, script, *args])
    if r.returncode:
        sys.exit(f"{script} failed (exit {r.returncode})")
    print(f"--- {script} done in {time.time() - t:.0f}s")


def main():
    steps = (COLLECT if "--collect" in sys.argv else []) + ANALYSIS
    if "--peers" in sys.argv:
        steps.append("peers.py")
    for s in steps + BUILD:
        run(s)


if __name__ == "__main__":
    main()
