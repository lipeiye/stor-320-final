"""Shared paths, constants and small helpers used by every pipeline step.

Everything is relative to the project root, so the bundle can be moved to any
folder and run from there.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project root)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"            # frozen analysis snapshot (never overwritten)
DATA_FRESH = ROOT / "data" / "fresh"        # dated re-downloads kept apart from the snapshot
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
AUDIT = RESULTS / "audit"
REPORT = ROOT / "report"

SNAPSHOT_CDC = DATA_RAW / "cdc_places_nc_raw.csv"
SNAPSHOT_EPA = DATA_RAW / "epa_aqi_nc_raw.csv"
SNAPSHOT_BOUNDARIES = DATA_RAW / "census_nc_county_boundaries.geojson"   # map outlines only
SNAPSHOT_MANIFEST = DATA_RAW / "SNAPSHOT_MANIFEST.json"
PROCESSED_TABLE = DATA_PROCESSED / "nc_county_health_environment.csv"
DATA_DICTIONARY = DATA_PROCESSED / "data_dictionary.csv"
RESULTS_JSON = RESULTS / "results.json"

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
CDC_DATASET_ID = "swc5-untb"
CDC_DATA_URL = "https://data.cdc.gov/resource/swc5-untb.json?stateabbr=NC&$limit=50000"
CDC_METADATA_URL = "https://data.cdc.gov/api/views/swc5-untb.json"
EPA_ZIP_URL = "https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2023.zip"
EPA_ZIP_MEMBER = "annual_aqi_by_county_2023.csv"
CENSUS_ZIP_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip"
CENSUS_ZIP_STEM = "cb_2023_us_county_5m"
STATE_ABBR = "NC"
STATE_NAME = "North Carolina"
STATE_FIPS = "37"
ANALYSIS_YEAR = 2023

# --------------------------------------------------------------------------
# Analysis constants (fixed in advance; see the proposal)
# --------------------------------------------------------------------------
SEED = 320            # one seed for every random step
N_BOOT = 2000         # county bootstrap resamples
AQI_MIN_DAYS = 300    # coverage sensitivity threshold (not an EPA standard)
HIGH_QUANTILE = 0.80  # "high COPD" = at or above the training-set 80th percentile

# PLACES MeasureId -> column stem, plain-language label, role in the analysis.
# Only these seven age-adjusted measures enter the analysis.
MEASURES = {
    "COPD":       {"col": "copd",       "label": "COPD",                         "role": "primary outcome"},
    "CASTHMA":    {"col": "asthma",     "label": "Current asthma",               "role": "secondary outcome"},
    "CSMOKING":   {"col": "smoking",    "label": "Current cigarette smoking",    "role": "behavioral indicator"},
    "LPA":        {"col": "inactivity", "label": "No leisure-time physical activity", "role": "behavioral indicator"},
    "ACCESS2":    {"col": "uninsured",  "label": "No health insurance (ages 18-64)", "role": "social indicator"},
    "LACKTRPT":   {"col": "transport",  "label": "Lack of reliable transportation", "role": "social indicator"},
    "HOUSINSECU": {"col": "housing",    "label": "Housing insecurity",           "role": "social indicator"},
}
DATA_VALUE_TYPE_ID = "AgeAdjPrv"   # age-adjusted prevalence

OUTCOME = "copd"
SECONDARY_OUTCOME = "asthma"
BEHAVIORAL = ["smoking", "inactivity"]
SOCIAL = ["transport", "uninsured", "housing"]
ALL_PREDICTORS = BEHAVIORAL + SOCIAL

LABELS = {m["col"]: m["label"] for m in MEASURES.values()}
LABELS.update({
    "aqi_median": "Median AQI",
    "aqi_prop_gt100": "Share of monitored days with AQI > 100",
})
SHORT_LABELS = {
    "copd": "COPD",
    "asthma": "Asthma",
    "smoking": "Smoking",
    "inactivity": "Inactivity",
    "uninsured": "Uninsured",
    "transport": "No transport",
    "housing": "Housing insecurity",
    "aqi_median": "Median AQI",
    "aqi_prop_gt100": "Days AQI > 100",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_FRESH, DATA_PROCESSED, FIGURES, TABLES, AUDIT, REPORT):
        d.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_jsonable)
        f.write("\n")


def _jsonable(x):
    """Convert numpy / pandas scalars so they can be written to JSON."""
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(x)


def update_results(section: str, values: dict) -> None:
    """Store one step's key numbers in results/results.json.

    The report and the slides read their numbers from this file, so every
    reported value traces back to the code that produced it.
    """
    current = read_json(RESULTS_JSON) if RESULTS_JSON.exists() else {}
    current[section] = values
    write_json(RESULTS_JSON, current)


def polygon_rings(geometry: dict) -> list[list[list[float]]]:
    """All rings (outer boundaries and holes) of a GeoJSON Polygon or MultiPolygon."""
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return [ring for polygon in polygons for ring in polygon]


def rel(path: Path) -> str:
    """Path shown in logs and manifests: always relative to the project root."""
    return str(Path(path).resolve().relative_to(ROOT))
