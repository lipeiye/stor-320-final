"""Step 00 - acquire the raw data and protect the analysis snapshot.

Sources
    1. CDC PLACES: Local Data for Better Health, County Data, 2025 release
       (data.cdc.gov dataset swc5-untb), North Carolina rows.
    2. U.S. EPA AirData, annual AQI by county, 2023, North Carolina rows.
    3. U.S. Census Bureau cartographic boundary file, counties, 2023, 1:5,000,000,
       North Carolina polygons. Used only to draw the county map; no model uses it.

What this script does
    * The analysis snapshot in data/raw/ is never overwritten. Sources 1 and 2 were
      retrieved on 2026-09-28, source 3 on 2026-11-15. SHA-256 checksums are
      registered in data/raw/SNAPSHOT_MANIFEST.json and verified on every run.
    * With --refresh, a fresh copy of every source is downloaded into
      data/fresh/<UTC date>/ together with the CDC dataset metadata, the HTTP
      headers and checksums. It is kept apart from the snapshot.
    * The snapshot is compared, record by record, with the most recent fresh copy
      and the comparison is written to results/audit/snapshot_vs_fresh.csv.

Usage
    python -I scripts/00_download_data.py            # verify + compare (no network)
    python -I scripts/00_download_data.py --refresh  # also download a fresh copy
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config as C

USER_AGENT = "python-urllib (course data retrieval script)"

# How each snapshot file was produced (same queries as download_fresh below).
SOURCES = {
    "cdc": {
        "file": C.SNAPSHOT_CDC.name,
        "retrieved_local_date": "2026-09-28",
        "publisher": "Centers for Disease Control and Prevention, Division of Population Health",
        "dataset": "PLACES: Local Data for Better Health, County Data, 2025 release",
        "dataset_id": C.CDC_DATASET_ID,
        "source_url": C.CDC_DATA_URL,
        "processing": "JSON response converted to CSV with pandas; no rows or columns dropped",
        "terms": "Public Domain (dataset metadata, licenseId PUBLIC_DOMAIN)",
    },
    "epa": {
        "file": C.SNAPSHOT_EPA.name,
        "retrieved_local_date": "2026-09-28",
        "publisher": "U.S. Environmental Protection Agency (AirData / Air Quality System)",
        "dataset": "Annual AQI by County, 2023",
        "source_url": C.EPA_ZIP_URL,
        "processing": "CSV inside the archive filtered to State == 'North Carolina'",
        "terms": "U.S. federal government public data",
    },
    "census": {
        "file": C.SNAPSHOT_BOUNDARIES.name,
        "retrieved_local_date": "2026-11-15",
        "publisher": "U.S. Census Bureau",
        "dataset": "Cartographic Boundary Files, 2023: counties, 1:5,000,000",
        "source_url": C.CENSUS_ZIP_URL,
        "processing": "polygons with STATEFP == '37' converted to GeoJSON; coordinates rounded to 5 decimals",
        "terms": "U.S. Census Bureau public data product; the publisher states no restriction",
        "use": "map outlines only; generalized shapes, not used for any measurement or model",
    },
}
SNAPSHOT_FILES = {"cdc": C.SNAPSHOT_CDC, "epa": C.SNAPSHOT_EPA, "census": C.SNAPSHOT_BOUNDARIES}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------
def fetch(url: str) -> tuple[bytes, dict]:
    """Return the response body and a small record of how it was retrieved."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read()
        info = {
            "url": url,
            "retrieved_utc": utc_now(),
            "http_status": resp.status,
            "last_modified": resp.headers.get("Last-Modified"),
            "etag": resp.headers.get("ETag"),
            "bytes": len(body),
        }
    return body, info


def boundaries_to_geojson(zip_bytes: bytes) -> tuple[str, dict]:
    """North Carolina county polygons from the Census shapefile archive, as GeoJSON text.

    The archive is read in memory; nothing is extracted to disk.
    """
    import shapefile   # pyshp; only needed when a fresh copy is downloaded

    def rounded(x):
        return [rounded(v) for v in x] if isinstance(x, (list, tuple)) else round(float(x), 5)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        parts = {ext: io.BytesIO(z.read(f"{C.CENSUS_ZIP_STEM}.{ext}")) for ext in ("shp", "shx", "dbf")}
        crs = z.read(f"{C.CENSUS_ZIP_STEM}.prj").decode("utf-8", "replace")
    reader = shapefile.Reader(shp=parts["shp"], shx=parts["shx"], dbf=parts["dbf"])
    features, n_all = [], 0
    for item in reader.iterShapeRecords():
        n_all += 1
        rec = item.record.as_dict()
        if rec["STATEFP"] != C.STATE_FIPS:
            continue
        geometry = item.shape.__geo_interface__
        features.append({
            "type": "Feature",
            "properties": {"GEOID": rec["GEOID"], "NAME": rec["NAME"], "STATEFP": rec["STATEFP"]},
            "geometry": {"type": geometry["type"], "coordinates": rounded(geometry["coordinates"])},
        })
    features.sort(key=lambda f: f["properties"]["GEOID"])
    text = json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")) + "\n"
    name = re.search(r'GEOGCS\["([^"]+)"', crs)
    return text, {"counties_all_states": n_all, "counties_nc": len(features),
                  "coordinate_system": name.group(1) if name else crs[:60]}


def download_fresh() -> Path:
    """Download every source into data/fresh/<UTC date>/ and log the retrieval."""
    out = C.DATA_FRESH / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    log = {}

    # --- CDC PLACES rows for North Carolina (exact API response is kept) ---
    body, info = fetch(C.CDC_DATA_URL)
    (out / "cdc_places_nc.json").write_bytes(body)
    records = json.loads(body.decode("utf-8"))
    cdc = pd.DataFrame(records)
    cdc.to_csv(out / C.SNAPSHOT_CDC.name, index=False)
    footnote_keys = sorted({k for r in records for k in r if "footnote" in k})
    info.update(rows=len(cdc), columns=list(cdc.columns), footnote_fields_present=footnote_keys)
    log["cdc_places_nc.json"] = info

    # --- CDC dataset metadata (release name, update timestamps, licence) ---
    body, info = fetch(C.CDC_METADATA_URL)
    (out / "cdc_places_metadata.json").write_bytes(body)
    meta = json.loads(body.decode("utf-8"))
    info.update(
        dataset_name=meta.get("name"),
        rows_updated_utc=_epoch(meta.get("rowsUpdatedAt")),
        view_last_modified_utc=_epoch(meta.get("viewLastModified")),
        license_id=meta.get("licenseId"),
        attribution=meta.get("attribution"),
    )
    log["cdc_places_metadata.json"] = info

    # --- EPA annual AQI archive: keep the original zip, read one member ---
    body, info = fetch(C.EPA_ZIP_URL)
    (out / "annual_aqi_by_county_2023.zip").write_bytes(body)
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        if C.EPA_ZIP_MEMBER not in z.namelist():
            raise SystemExit(f"Unexpected archive contents: {z.namelist()}")
        with z.open(C.EPA_ZIP_MEMBER) as f:
            epa_all = pd.read_csv(f)
    epa_nc = epa_all[epa_all["State"] == C.STATE_NAME]
    epa_nc.to_csv(out / C.SNAPSHOT_EPA.name, index=False)
    info.update(rows_all_states=len(epa_all), rows_nc=len(epa_nc))
    log["annual_aqi_by_county_2023.zip"] = info

    # --- Census county boundaries: keep the original zip, convert NC polygons ---
    body, info = fetch(C.CENSUS_ZIP_URL)
    (out / f"{C.CENSUS_ZIP_STEM}.zip").write_bytes(body)
    text, facts = boundaries_to_geojson(body)
    (out / C.SNAPSHOT_BOUNDARIES.name).write_text(text, encoding="utf-8")
    info.update(facts)
    log[f"{C.CENSUS_ZIP_STEM}.zip"] = info

    for name in ("cdc_places_nc.json", C.SNAPSHOT_CDC.name, "cdc_places_metadata.json",
                 "annual_aqi_by_county_2023.zip", C.SNAPSHOT_EPA.name,
                 f"{C.CENSUS_ZIP_STEM}.zip", C.SNAPSHOT_BOUNDARIES.name):
        log.setdefault(name, {})["sha256"] = C.sha256_file(out / name)
    C.write_json(out / "retrieval_log.json", log)
    print(f"Fresh download saved in {C.rel(out)}")
    return out


def _epoch(seconds) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(int(seconds), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def latest_fresh() -> Path | None:
    folders = sorted(p for p in C.DATA_FRESH.glob("*") if (p / "retrieval_log.json").exists())
    return folders[-1] if folders else None


# --------------------------------------------------------------------------
# Snapshot protection
# --------------------------------------------------------------------------
def verify_snapshot() -> None:
    """Register a snapshot file's checksum the first time it is seen, then verify all of them."""
    manifest = C.read_json(C.SNAPSHOT_MANIFEST) if C.SNAPSHOT_MANIFEST.exists() else {}
    files = manifest.setdefault("files", {})

    # First acquisition of the map boundaries: adopt them from the fresh download.
    fresh = latest_fresh()
    path = C.SNAPSHOT_BOUNDARIES
    if not path.exists() and path.name not in files and fresh and (fresh / path.name).exists():
        shutil.copyfile(fresh / path.name, path)
        print(f"Adopted {C.rel(fresh / path.name)} as the analysis snapshot {C.rel(path)}")

    for path in SNAPSHOT_FILES.values():
        if not path.exists():
            raise SystemExit(
                f"Missing {C.rel(path)}. The analysis snapshot ships with this bundle; "
                "restore it before running the pipeline."
            )
        if path.name not in files:
            files[path.name] = {
                "sha256": C.sha256_file(path),
                "bytes": path.stat().st_size,
                "file_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "registered_utc": utc_now(),
            }
            print(f"Registered checksum of {path.name}")

    manifest["sources"] = SOURCES
    manifest.setdefault("manifest_created_utc", utc_now())
    manifest["note"] = (
        "Checksums are registered once and verified on every run. The CDC dataset metadata was "
        "not saved on 2026-09-28; see data/fresh/ for metadata retrieved later and "
        "results/audit/snapshot_vs_fresh.csv for the comparison."
    )
    C.write_json(C.SNAPSHOT_MANIFEST, manifest)

    for path in SNAPSHOT_FILES.values():
        if C.sha256_file(path) != files[path.name]["sha256"]:
            raise SystemExit(f"Checksum mismatch for {path.name}: the analysis snapshot was modified.")
    print("Snapshot checksums verified.")


# --------------------------------------------------------------------------
# Snapshot versus fresh copy
# --------------------------------------------------------------------------
def _normalise_cdc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # The point geometry is stored as text; compare its coordinates, not its formatting.
    coords = df["geolocation"].map(lambda s: ast.literal_eval(s)["coordinates"])
    df["geolocation"] = coords.map(lambda xy: f"{float(xy[0]):.10f},{float(xy[1]):.10f}")
    return df


def _boundary_table(path: Path) -> pd.DataFrame:
    """One row per county polygon: code, name, and the geometry as text."""
    features = C.read_json(path)["features"]
    return pd.DataFrame([{"GEOID": f["properties"]["GEOID"], "NAME": f["properties"]["NAME"],
                          "geometry": json.dumps(f["geometry"], sort_keys=True)} for f in features])


def compare(snapshot: pd.DataFrame, fresh: pd.DataFrame, keys: list[str], numeric: list[str]) -> dict:
    """Row-by-row comparison of two tables that share a unique key."""
    s = snapshot.set_index(keys).sort_index()
    f = fresh.set_index(keys).sort_index()
    only_s, only_f = s.index.difference(f.index), f.index.difference(s.index)
    cols = [c for c in s.columns if c in f.columns]
    common = s.index.intersection(f.index)
    a, b = s.loc[common, cols].copy(), f.loc[common, cols].copy()
    for c in numeric:
        if c in cols:
            a[c], b[c] = pd.to_numeric(a[c]), pd.to_numeric(b[c])
    different = (a.astype(str) != b.astype(str))
    return {
        "rows_snapshot": len(s),
        "rows_fresh": len(f),
        "keys_only_in_snapshot": len(only_s),
        "keys_only_in_fresh": len(only_f),
        "columns_only_in_snapshot": ";".join(sorted(set(s.columns) - set(f.columns))),
        "columns_only_in_fresh": ";".join(sorted(set(f.columns) - set(s.columns))),
        "cells_compared": int(different.size),
        "cells_different": int(different.to_numpy().sum()),
    }


def compare_with_latest_fresh() -> None:
    fresh_dir = latest_fresh()
    if fresh_dir is None:
        print("No fresh download stored; run with --refresh to create one.")
        return
    log = C.read_json(fresh_dir / "retrieval_log.json")
    census_zip = f"{C.CENSUS_ZIP_STEM}.zip"
    rows = []

    snap = _normalise_cdc(pd.read_csv(C.SNAPSHOT_CDC, dtype=str))
    fresh = _normalise_cdc(pd.read_csv(fresh_dir / C.SNAPSHOT_CDC.name, dtype=str))
    r = compare(snap, fresh, ["locationid", "measureid", "datavaluetypeid"],
                ["data_value", "low_confidence_limit", "high_confidence_limit",
                 "totalpopulation", "totalpop18plus", "year"])
    rows.append({"source": "CDC PLACES (NC rows)", "snapshot_date": SOURCES["cdc"]["retrieved_local_date"], **r,
                 "fresh_retrieved_utc": log["cdc_places_nc.json"]["retrieved_utc"],
                 "source_last_changed": log["cdc_places_metadata.json"]["rows_updated_utc"]})

    snap = pd.read_csv(C.SNAPSHOT_EPA, dtype=str)
    fresh = pd.read_csv(fresh_dir / C.SNAPSHOT_EPA.name, dtype=str)
    r = compare(snap, fresh, ["State", "County", "Year"],
                [c for c in snap.columns if c not in ("State", "County")])
    rows.append({"source": "EPA annual AQI by county 2023 (NC rows)",
                 "snapshot_date": SOURCES["epa"]["retrieved_local_date"], **r,
                 "fresh_retrieved_utc": log["annual_aqi_by_county_2023.zip"]["retrieved_utc"],
                 "source_last_changed": log["annual_aqi_by_county_2023.zip"]["last_modified"]})

    r = compare(_boundary_table(C.SNAPSHOT_BOUNDARIES), _boundary_table(fresh_dir / C.SNAPSHOT_BOUNDARIES.name),
                ["GEOID"], [])
    rows.append({"source": "Census county boundaries 2023 (NC polygons)",
                 "snapshot_date": SOURCES["census"]["retrieved_local_date"], **r,
                 "fresh_retrieved_utc": log[census_zip]["retrieved_utc"],
                 "source_last_changed": log[census_zip]["last_modified"]})

    out = pd.DataFrame(rows)
    out["identical_content"] = (
        (out["keys_only_in_snapshot"] == 0) & (out["keys_only_in_fresh"] == 0)
        & (out["cells_different"] == 0)
        & (out["columns_only_in_snapshot"] == "") & (out["columns_only_in_fresh"] == "")
    )
    out.to_csv(C.AUDIT / "snapshot_vs_fresh.csv", index=False)
    print(out.T.to_string(header=False))
    print(f"Comparison written to {C.rel(C.AUDIT / 'snapshot_vs_fresh.csv')}")

    C.update_results("acquisition", {
        "snapshot_date": SOURCES["cdc"]["retrieved_local_date"],
        "boundaries_date": SOURCES["census"]["retrieved_local_date"],
        "fresh_folder": C.rel(fresh_dir),
        "cdc_dataset_name": log["cdc_places_metadata.json"]["dataset_name"],
        "cdc_rows_updated_utc": log["cdc_places_metadata.json"]["rows_updated_utc"],
        "cdc_license_id": log["cdc_places_metadata.json"]["license_id"],
        "cdc_fresh_retrieved_utc": log["cdc_places_nc.json"]["retrieved_utc"],
        "cdc_footnote_fields_in_fresh": log["cdc_places_nc.json"]["footnote_fields_present"],
        "epa_zip_last_modified": log["annual_aqi_by_county_2023.zip"]["last_modified"],
        "epa_rows_all_states": log["annual_aqi_by_county_2023.zip"]["rows_all_states"],
        "census_zip_last_modified": log[census_zip]["last_modified"],
        "census_zip_bytes": log[census_zip]["bytes"],
        "census_counties_all_states": log[census_zip]["counties_all_states"],
        "census_coordinate_system": log[census_zip]["coordinate_system"],
        "snapshot_sha256": {p.name: C.sha256_file(p) for p in SNAPSHOT_FILES.values()},
        "comparison": out.to_dict(orient="records"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true",
                        help="download a fresh copy of every source (needs network access)")
    args = parser.parse_args()

    C.ensure_dirs()
    if args.refresh:
        download_fresh()
    verify_snapshot()
    compare_with_latest_fresh()


if __name__ == "__main__":
    main()
