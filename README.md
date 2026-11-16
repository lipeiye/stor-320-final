# Respiratory Health and Social Needs in North Carolina Counties

Final project for STOR 320-003 Methods of Data Science (UNC Chapel Hill, Fall 2026).
Python scripts paired with a Markdown report source.

## The question

Across North Carolina's 100 counties, are transportation difficulties, lack of health insurance and
housing insecurity associated with age-adjusted COPD prevalence after adjusting for smoking and
physical inactivity?

Two secondary questions: whether median AQI is associated with COPD among the counties that have 2023
AQI records, and whether the indicators flag high-COPD counties better than smoking alone.

The study is cross-sectional and ecological: one row per county, modeled estimates for 2023,
associations rather than causes.

## Deliverables

| Deliverable | Where |
| --- | --- |
| Report | source `report/report.md`; rendered `report/report.html` and `report/report.pdf` |
| Slides | source `slides/deck/project/`; PowerPoint copy `slides/copd_social_needs_slides.pptx`; static copy `slides/slides_backup_static.pdf`; live deck at https://claude.ai/artifact/Giu5vWC7EQepR7SDY9ji21 (private until shared by its owner); a Google Slides copy of the 12 main slides and a script-only Google Doc in the owner's Google Drive |
| Code | `run_all.py` (entry point) and `scripts/` |
| Data | `data/raw/` (frozen snapshot), `data/fresh/` (dated re-download), `data/processed/` |
| Results | `results/` (audit outputs, tables, figures, `results.json`) |

## This repository

The public GitHub repository holds everything needed to rerun the analysis and to read its results: code,
data snapshots, results, the report source with its rendered draft, and the slides. A few files stay in the
team's local folder and are not published: the team's working notes (`team_notes/`) and the guides for AI
coding agents (`AGENTS.md`, `CLAUDE.md`), the proposal (it carries the members' names) and the instructor's
course document. `.gitignore` lists them. The report is still a draft: the yellow "To write" blocks are
for the team.

## Reproduce from a clean start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/) (or any virtual environment plus `pip`).
Run from this folder:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python run_all.py
```

`run_all.py` verifies the data snapshot, runs every step and renders `report/report.html`.
A full run takes about two minutes; almost all of it is step 05.

| Option | Effect |
| --- | --- |
| `--refresh` | also re-download every source into `data/fresh/<date>/` and compare it with the snapshot (needs network) |
| `--final` | refuse to render the report while any `::: write` block remains in `report/report.md` |
| `--pdf` | also print the report to `report/report.pdf` (needs Chrome, Chromium or Edge) |

Every step can be run alone, for example `.venv/bin/python -I scripts/03_q1_social_needs.py`.

## Pipeline

| Step | What it does | Main outputs |
| --- | --- | --- |
| `scripts/00_download_data.py` | verifies snapshot checksums; with `--refresh` downloads a fresh copy; compares snapshot and fresh copy | `data/raw/SNAPSHOT_MANIFEST.json`, `results/audit/snapshot_vs_fresh.csv` |
| `scripts/01_clean_audit.py` | 42 executable checks, selection of seven measures, pivot to one row per county, join of EPA data; stops if a check fails | `data/processed/nc_county_health_environment.csv`, `data_dictionary.csv`, `results/audit/*.csv` |
| `scripts/02_eda.py` | descriptive tables, county map, CDC intervals, correlations, scatterplots | `results/tables/eda_*.csv`, figures 01-05 |
| `scripts/03_q1_social_needs.py` | Question 1: baseline and expanded regressions, three kinds of interval, diagnostics, sensitivity analyses | `results/tables/q1_*.csv`, figures 06-08 |
| `scripts/04_q2_air_quality.py` | Question 2: air-quality models on the 35 counties with records | `results/tables/q2_*.csv`, figure 09 |
| `scripts/05_q3_prediction.py` | Question 3: nested cross-validation, repeated partitions, shuffled-outcome check | `results/tables/q3_*.csv`, figure 10 |
| `scripts/06_build_report.py` | renders the report from its Markdown source and the result files | `report/report.html`, `report/report.pdf` |

Each step also stores its key numbers in `results/results.json`, which the report reads at build time.
Shared settings are in `scripts/config.py` (paths, sources, seed, variable roles) and figure styling in
`scripts/plotstyle.py`.

## Folder map

```
run_all.py                 entry point
requirements.txt           pinned direct dependencies (requirements-lock.txt: every installed package)
README.md                  this file
AGENTS.md, CLAUDE.md       working guide for AI coding agents (local folder only)
proposal.docx              approved proposal (local folder only)
final_project_instructions_and_rubrics_updated.pdf   course instructions (local folder only)
scripts/                   pipeline steps 00-06, config.py, plotstyle.py
data/
  raw/                     analysis snapshot with SNAPSHOT_MANIFEST.json (SHA-256): CDC and EPA files
                           retrieved 2026-09-28, Census map outlines retrieved 2026-11-15
  fresh/2026-11-15/        re-download kept apart from the snapshot, with retrieval_log.json
  processed/               nc_county_health_environment.csv (one row per county), data_dictionary.csv
results/
  audit/                   audit_checks.csv, row_counts.csv, missingness.csv, snapshot_vs_fresh.csv, join_county_match.csv
  tables/                  eda_*.csv, q1_*.csv, q2_*.csv, q3_*.csv
  figures/                 fig01 ... fig10 (report versions, with title and source note)
  results.json             key numbers used by the report
report/
  report.md                report source
  report.html / .pdf       rendered report
slides/
  deck/project/            slide source (deck.json, deck.css, slides/*.html with speaker notes)
  build_figures.py         draws the deck's figures as layered PNGs from results/
  layers/                  those PNGs (transparent; one file per click step of a chart)
  assets/                  slide versions of the report figures, written by the pipeline
  asset_map.json           which uploaded deck image corresponds to which local file
  fonts/                   Inter and Inter Tight (SIL Open Font License), used by the slides and their figures
  export_static_pdf.py     builds slides_backup_static.pdf (and PNGs per slide or per click state)
  build_pptx.py            builds the PowerPoint copy from the same slide source
  copd_social_needs_slides.pptx   editable copy with the click steps and the notes
  slides_backup_static.pdf static copy of the deck (one page per slide, no animation)
team_notes/                working notes for the team; not part of the report (local folder only)
```

## Data

| | CDC PLACES | EPA AirData | Census boundary file |
| --- | --- | --- | --- |
| Publisher | Centers for Disease Control and Prevention, Division of Population Health | U.S. Environmental Protection Agency | U.S. Census Bureau |
| Dataset | PLACES: Local Data for Better Health, County Data, 2025 release (`swc5-untb`) | Annual AQI by County, 2023 | Cartographic Boundary Files, 2023: counties, 1:5,000,000 |
| Retrieval | `https://data.cdc.gov/resource/swc5-untb.json?stateabbr=NC&$limit=50000` | `https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2023.zip`, North Carolina rows | `https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip`, North Carolina polygons |
| Snapshot | `data/raw/cdc_places_nc_raw.csv`, retrieved 2026-09-28 | `data/raw/epa_aqi_nc_raw.csv`, retrieved 2026-09-28 | `data/raw/census_nc_county_boundaries.geojson`, retrieved 2026-11-15 |
| Use | outcomes and predictors | air-quality measures (Question 2) | outlines of the county map only |
| Terms | Public domain | U.S. federal government public data | U.S. Census Bureau public data product |

The files hold aggregate county records and county outlines; nothing personal or restricted is used.
PLACES values are model-based small-area estimates, each published with 95% confidence limits; the
limits are kept in the processed table (`*_lcl`, `*_ucl`). Thirty-five counties have 2023 AQI records;
the other 65 are kept with missing AQI values, which are never imputed.

The snapshot is never overwritten: `scripts/00_download_data.py` checks each SHA-256 against
`data/raw/SNAPSHOT_MANIFEST.json` on every run. The re-download of 2026-11-15 is identical to the CDC
and EPA snapshot in every cell (`results/audit/snapshot_vs_fresh.csv`). The boundary file is a
generalized outline for mapping and is not used for any measurement or model.

## Report

`report/report.md` holds the text. Tables, figures and numbers are inserted at build time, so the
rendered report always matches the result files:

| In the Markdown | Becomes |
| --- | --- |
| `{{figure:fig06_q1_specifications}}` | the figure with its number and caption |
| `{{table:q1_coefs}}` | a table built from the result files |
| `{{ref:q1_coefs}}` | "Table 7" (numbering follows order of appearance) |
| `{{q1.fit.copd.expanded.r2|.3f}}` | a value from `results/results.json`, formatted |
| `::: write` ... `:::` | a block still to be written; `--final` refuses to build while one remains |

Build a draft with `.venv/bin/python -I scripts/06_build_report.py` (add `--pdf` for a PDF) and the
final version with `.venv/bin/python run_all.py --final --pdf`. Figure captions are in `FIGURES` and
table captions and notes in the `t_*` functions of `scripts/06_build_report.py`.

## Slides

The deck has 15 slides (12 main, 3 backup) and 14 click steps. Its source is `slides/deck/project/`. Charts
that are revealed in steps are stacks of transparent PNG layers drawn by `slides/build_figures.py`, which
reads every number from `results/`. The deck is published as a private Claude artifact and can be presented
from there; `slides/copd_social_needs_slides.pptx` is an editable PowerPoint copy with the same click steps
and speaker notes, and `slides/slides_backup_static.pdf` shows every step at once. A Google Slides copy of
the 12 main slides (no backup slides) was made from the last command below, uploaded to Google Drive and
converted there; `--google` adjusts line spacing, line breaks and box widths for Google's import. Text is
set in Inter, with titles and big numbers in Inter Tight: open-source faces whose files and licences are in
`slides/fonts/`. Install them on the computer that opens the PowerPoint copy.

```bash
.venv/bin/python -I slides/build_figures.py                # redraw the deck's figures into slides/layers/
.venv/bin/python -I slides/export_static_pdf.py --check    # list slide images that changed since upload
.venv/bin/python -I slides/export_static_pdf.py            # rebuild slides/slides_backup_static.pdf
uv run --no-project --with python-pptx python slides/build_pptx.py   # rebuild the PowerPoint copy
uv run --no-project --with python-pptx python slides/build_pptx.py --main-only --google --out FILE.pptx
```

The last three need Chrome, Chromium or Edge. `python-pptx` is fetched by `uv` for that one command and is not
part of the analysis environment.

## Reproducibility

- One seed (320, in `scripts/config.py`) drives every resampling step; results are deterministic.
- Package versions are pinned in `requirements.txt`.
- All paths are relative to the project folder.
- The project was rerun in an empty folder with a new environment and every regenerated file compared
  by SHA-256 with this folder: `team_notes/reproducibility_check.txt`.

## Working on the project

The files named here stay in the team's local folder; they are not in the public repository.

- `AGENTS.md` is a compact guide for AI coding agents: rules, file map, what to rerun after a change,
  where each number lives. `CLAUDE.md` loads it automatically in Claude Code.
- `team_notes/` holds the team's working material: a study guide (Chinese), the presentation script for
  six speakers, the slide redesign brief, the AI use log for the Disclosure Note, a rubric checklist and
  the reproducibility record.

## Submission bundle

Everything needed to reproduce the report is in this folder. The local environment (`.venv/`) and
Python caches are not part of the bundle; `team_notes/`, `AGENTS.md` and `CLAUDE.md` are working
material and are not needed to rebuild anything. To zip the bundle from this folder:

```bash
zip -r stor320_final_project_bundle.zip . -x ".venv/*" "*/__pycache__/*" "*.DS_Store" "team_notes/*" "AGENTS.md" "CLAUDE.md"
```
