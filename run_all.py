"""Entry point: run the whole project from a clean start.

    python run_all.py              verify the data snapshot, then run every step and render the report
    python run_all.py --refresh    also re-download both sources into data/fresh/ and compare
    python run_all.py --final      refuse to render while a "::: write" block remains in report/report.md
    python run_all.py --pdf        also print the report to PDF (needs Chrome, Chromium or Edge)

Each step is a separate script in scripts/ and can also be run on its own. Steps run in
Python's isolated mode (-I) so that nothing outside this project's environment is imported.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "00_download_data.py",      # verify snapshot checksums (+ optional fresh download)
    "01_clean_audit.py",        # audit, clean, reshape, join
    "02_eda.py",                # descriptive tables and figures
    "03_q1_social_needs.py",    # Question 1 models and checks
    "04_q2_air_quality.py",     # Question 2 models
    "05_q3_prediction.py",      # Question 3 nested cross-validation (about 2 minutes)
    "06_build_report.py",       # render report/report.md to HTML
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true", help="re-download the sources for comparison")
    parser.add_argument("--final", action="store_true", help="final report build (no unwritten blocks allowed)")
    parser.add_argument("--pdf", action="store_true", help="also print the report to PDF")
    args = parser.parse_args()

    extra = {"00_download_data.py": ["--refresh"] if args.refresh else [],
             "06_build_report.py": (["--final"] if args.final else []) + (["--pdf"] if args.pdf else [])}
    start = time.time()
    for step in STEPS:
        print(f"\n=== {step} " + "=" * (60 - len(step)))
        subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / step), *extra.get(step, [])],
                       cwd=ROOT, check=True)
    print(f"\nAll steps finished in {time.time() - start:.0f} seconds.")


if __name__ == "__main__":
    main()
