"""Local copies of the slide deck, built from the slide source.

    python -I slides/export_static_pdf.py                         # -> slides/slides_backup_static.pdf
    python -I slides/export_static_pdf.py --png-dir DIR           # also one PNG per slide (quick layout check)
    python -I slides/export_static_pdf.py --png-dir DIR --steps   # one PNG per click state instead
    python -I slides/export_static_pdf.py --check                 # only list slide images that changed since upload

The published deck refers to its images as /_blob/<id>. slides/asset_map.json maps each id to the
local file that was uploaded (path relative to slides/) with that file's SHA-256 at upload time, so
an image that was regenerated but not uploaded again is reported here.

The static PDF shows every click step at once; animations are not played and speaker notes are
left out. Needs Chrome, Chromium or Edge. The deck's faces, Inter and Inter Tight, are read from
slides/fonts/ (the online deck loads the same faces from Google Fonts), so these copies look the same
with or without a network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

SLIDES = Path(__file__).resolve().parent
DECK = SLIDES / "deck" / "project"
ASSET_MAP = SLIDES / "asset_map.json"
FONTS = SLIDES / "fonts"
LOCAL_FACES = [("Inter", 400, "Inter-Regular.ttf"), ("Inter", 700, "Inter-Bold.ttf"),
               ("Inter Tight", 700, "InterTight-Bold.ttf")]
OUT_PDF = SLIDES / "slides_backup_static.pdf"

BASE_CSS = """
* { margin: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact }
@page { size: 1920px 1080px; margin: 0 }
html, body { width: 1920px }
body { font: 32px/1.2 sans-serif }
body > section { width: 1920px !important; height: 1080px !important; overflow: hidden !important;
                 box-sizing: border-box !important; position: relative; isolation: isolate;
                 break-after: page; break-inside: avoid }
section[hidden], aside, noscript { display: none !important }
p, li, h1, h2, h3, td, th { overflow-wrap: break-word; text-wrap: pretty }
*, *::before, *::after { animation: none !important; transition: none !important }
"""

# Runs in the local page only (slide files hold no script). It shows the click steps up to the one
# named in the address (#step=N; all of them without it) and keeps a backdrop listed before the
# first flowing element underneath it, as the deck player does.
PAGE_SCRIPT = """
const asked = location.hash.match(/step=(\\d+)/);
const shown = asked ? Number(asked[1]) : Infinity;
for (const el of document.querySelectorAll('.fragment')) {
  if (Number(el.dataset.fragmentIndex) <= shown) el.classList.add('visible');
}
for (const section of document.querySelectorAll('body > section')) {
  for (const el of section.children) {
    if (getComputedStyle(el).position !== 'absolute') break;
    el.style.zIndex = -1;
  }
}
"""


def browser() -> str:
    candidates = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                  "/Applications/Chromium.app/Contents/MacOS/Chromium",
                  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    found = next((c for c in candidates if Path(c).exists()), None) or next(
        (shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge")
         if shutil.which(n)), None)
    if not found:
        raise SystemExit("No Chrome, Chromium or Edge found.")
    return found


def load_index() -> dict:
    return json.loads((DECK / "deck.json").read_text(encoding="utf-8"))


def load_asset_map() -> dict:
    return json.loads(ASSET_MAP.read_text(encoding="utf-8"))


def stale_assets(asset_map: dict) -> list[str]:
    """Images whose local file no longer matches what was uploaded to the deck."""
    stale = []
    for blob, item in asset_map.items():
        path = SLIDES / item["file"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            stale.append(f"{item['file']} (deck image /_blob/{blob})")
    return stale


def slide_source(slide_id: str) -> str:
    return (DECK / "slides" / f"{slide_id}.html").read_text(encoding="utf-8")


def click_steps(slide_id: str) -> list[int]:
    """The click steps of one slide, in order. Every fragment in this deck names its step."""
    html = slide_source(slide_id)
    tags = re.findall(r"<[^>]*\bclass=\"[^\"]*\bfragment\b[^\"]*\"[^>]*>", html)
    steps = set()
    for tag in tags:
        index = re.search(r'data-fragment-index="(\d+)"', tag)
        if not index:
            raise SystemExit(f"{slide_id}: a fragment has no data-fragment-index")
        steps.add(int(index.group(1)))
    return sorted(steps)


def page_html(slide_ids: list[str], index: dict, asset_map: dict, counter_start: int = 0,
              extra_script: str = "") -> str:
    def local(match: re.Match) -> str:
        blob = match.group(1)
        if blob not in asset_map:
            raise SystemExit(f"/_blob/{blob} is not listed in {ASSET_MAP.name}")
        return (SLIDES / asset_map[blob]["file"]).as_uri()

    sections = [re.sub(r"/_blob/([0-9a-f]{32})", local, slide_source(sid)) for sid in slide_ids]
    files = [(family, weight, FONTS / name) for family, weight, name in LOCAL_FACES if (FONTS / name).exists()]
    faces = "".join(f"@font-face{{font-family:'{family}';font-weight:{weight};font-style:normal;"
                    f"src:url('{path.as_uri()}') format('truetype')}}" for family, weight, path in files)
    covered = {family for family, _, _ in files}
    fonts = "".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in index["faces"].values()
                    if "href" in f and f["family"] not in covered)
    css = (DECK / "deck.css").read_text(encoding="utf-8")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{index['title']}</title>{fonts}"
            f"<style>{faces}{BASE_CSS}</style><style>{css}</style>"
            f"<style>body{{counter-reset: page {counter_start}}}</style></head><body>{''.join(sections)}"
            f"<script>{PAGE_SCRIPT}{extra_script}</script></body></html>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--png-dir", type=Path, help="also write PNG files into this folder")
    parser.add_argument("--steps", action="store_true", help="with --png-dir: one PNG per click state")
    parser.add_argument("--check", action="store_true", help="only report images that changed since upload")
    args = parser.parse_args()

    index, asset_map = load_index(), load_asset_map()
    stale = stale_assets(asset_map)
    for item in stale:
        print(f"CHANGED SINCE UPLOAD: {item}")
    if args.check:
        print("All slide images match the uploaded versions." if not stale else
              "Upload the changed images to the deck, then update the slide src and asset_map.json.")
        raise SystemExit(1 if stale else 0)

    chrome = browser()
    quiet = dict(check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "deck.html"
        page.write_text(page_html(index["order"], index, asset_map), encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=8000", f"--print-to-pdf={OUT_PDF}", page.as_uri()], **quiet)
        print(f"Wrote {OUT_PDF.relative_to(SLIDES.parent)} ({len(index['order'])} slides)")

        if args.png_dir:
            args.png_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            for n, sid in enumerate(index["order"], start=1):
                one = Path(tmp) / f"{sid}.html"
                one.write_text(page_html([sid], index, asset_map, counter_start=n - 1), encoding="utf-8")
                # without --steps: one picture with every step shown. With it: one picture on arrival
                # (_0) and one after each click (_1, _2, ...), whatever index numbers the slide uses.
                shown = [0] + click_steps(sid) if args.steps else [None]
                for state, index_shown in enumerate(shown):
                    address = one.as_uri() if index_shown is None else f"{one.as_uri()}#step={index_shown}"
                    name = f"{n:02d}_{sid}.png" if index_shown is None else f"{n:02d}_{sid}_{state}.png"
                    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                                    "--force-device-scale-factor=1", "--window-size=1920,1080",
                                    "--virtual-time-budget=4000", f"--screenshot={args.png_dir / name}",
                                    address], **quiet)
                    written += 1
            print(f"Wrote {written} PNG files to {args.png_dir}")


if __name__ == "__main__":
    main()
