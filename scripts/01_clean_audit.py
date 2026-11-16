"""Step 01 - audit, clean, reshape and join the raw snapshot.

Input
    data/raw/cdc_places_nc_raw.csv                 long table: county x measure x value type
    data/raw/epa_aqi_nc_raw.csv                    one row per county with 2023 AQI records
    data/raw/census_nc_county_boundaries.geojson   county outlines for the map (checked here, not merged)

Output
    data/processed/nc_county_health_environment.csv   one row per county (n = 100)
    data/processed/data_dictionary.csv
    results/audit/audit_checks.csv       every check: expected, observed, PASS / FAIL
    results/audit/row_counts.csv         rows kept at each preparation step
    results/audit/missingness.csv        missing values per analysis variable
    results/audit/join_county_match.csv  EPA county name -> CDC county and FIPS

The script stops with an error if any check fails, so later steps never run on
data that broke an assumption.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config as C

CHECKS: list[dict] = []
ROW_COUNTS: list[dict] = []


def check(check_id: str, area: str, description: str, passed: bool, observed, expected="") -> None:
    """Record one executable check."""
    CHECKS.append({"id": check_id, "area": area, "check": description,
                   "expected": str(expected), "observed": str(observed),
                   "result": "PASS" if bool(passed) else "FAIL"})


def count(step: str, rows: int, counties: int, note: str = "") -> None:
    ROW_COUNTS.append({"step": step, "rows": int(rows), "counties": int(counties), "note": note})


def norm_name(name: str) -> str:
    """County name reduced to lower-case letters, used only for the name match."""
    name = re.sub(r"\bcounty\b", "", str(name), flags=re.IGNORECASE)
    return re.sub(r"[^a-z]", "", name.casefold())


# --------------------------------------------------------------------------
# 1. CDC PLACES
# --------------------------------------------------------------------------
def prepare_places() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(C.SNAPSHOT_CDC, dtype=str)
    count("CDC raw snapshot", len(raw), raw["locationid"].nunique(),
          f"{raw['measureid'].nunique()} measures x {raw['datavaluetypeid'].nunique()} value types")

    expected_cols = {"year", "stateabbr", "locationname", "locationid", "measure", "measureid",
                     "data_value_unit", "data_value_type", "datavaluetypeid", "data_value",
                     "low_confidence_limit", "high_confidence_limit", "totalpopulation",
                     "totalpop18plus", "geolocation"}
    check("P01", "CDC structure", "Required columns are present",
          expected_cols <= set(raw.columns), sorted(expected_cols - set(raw.columns)) or "none missing",
          "none missing")
    check("P02", "CDC coverage", "Every row belongs to North Carolina",
          raw["stateabbr"].eq(C.STATE_ABBR).all(), raw["stateabbr"].unique().tolist(), [C.STATE_ABBR])
    fips_ok = raw["locationid"].str.fullmatch(rf"{C.STATE_FIPS}\d{{3}}").all()
    check("P03", "CDC keys", "County FIPS codes are five digits starting with 37",
          fips_ok, f"{raw['locationid'].min()} to {raw['locationid'].max()}", "37001 to 37199")
    check("P04", "CDC coverage", "Number of distinct counties",
          raw["locationid"].nunique() == 100, raw["locationid"].nunique(), 100)
    pairs = raw[["locationid", "locationname"]].drop_duplicates()
    check("P05", "CDC keys", "FIPS code and county name map one-to-one",
          pairs["locationid"].is_unique and pairs["locationname"].is_unique, len(pairs), 100)
    grid = raw.groupby("locationid").size()
    check("P06", "CDC structure", "Every county has the same number of raw rows",
          grid.nunique() == 1, f"{grid.min()} to {grid.max()} rows per county", "constant")
    privacy_cols = [c for c in raw.columns if re.search(r"name|address|birth|respondent|person", c)
                    and c != "locationname"]
    check("P07", "Privacy", "No person-level fields; smallest unit is a county",
          not privacy_cols, privacy_cols or "aggregate county fields only", "aggregate county fields only")

    # --- keep the seven prespecified measures, age-adjusted prevalence only ---
    sel = raw[raw["measureid"].isin(C.MEASURES)].copy()
    count("Seven selected measures", len(sel), sel["locationid"].nunique(), "both value types")
    sel = sel[sel["datavaluetypeid"].eq(C.DATA_VALUE_TYPE_ID)].copy()
    count("Age-adjusted prevalence only", len(sel), sel["locationid"].nunique(), "crude prevalence rows dropped")

    check("P08", "CDC structure", "Selected rows = 7 measures x 100 counties",
          len(sel) == 700, len(sel), 700)
    key = ["locationid", "measureid", "datavaluetypeid", "year"]
    check("P09", "CDC keys", "County-measure-type-year is unique before pivoting",
          not sel.duplicated(key).any(), int(sel.duplicated(key).sum()), 0)
    check("P10", "Timing", "All seven selected measures use survey year 2023",
          sel["year"].eq(str(C.ANALYSIS_YEAR)).all(), sorted(sel["year"].unique()), [str(C.ANALYSIS_YEAR)])
    other_year = sorted(raw.loc[raw["year"].ne(str(C.ANALYSIS_YEAR)), "measureid"].unique())
    check("P11", "Timing", "Measures with another survey year are outside the analysis",
          not set(other_year) & set(C.MEASURES), f"2022 measures not used: {', '.join(other_year)}",
          "no overlap with selected measures")
    check("P12", "Units", "Unit is percent for every selected row",
          sel["data_value_unit"].eq("%").all(), sel["data_value_unit"].unique().tolist(), ["%"])

    num_cols = ["data_value", "low_confidence_limit", "high_confidence_limit",
                "totalpopulation", "totalpop18plus"]
    for c in num_cols:
        sel[c] = pd.to_numeric(sel[c], errors="coerce")
    n_missing = int(sel[num_cols].isna().sum().sum())
    check("P13", "Missingness", "No missing or non-numeric values in estimates, limits, population",
          n_missing == 0, n_missing, 0)
    in_range = sel[["data_value", "low_confidence_limit", "high_confidence_limit"]].stack().between(0, 100).all()
    check("P14", "Ranges", "Estimates and confidence limits lie between 0 and 100 percent",
          in_range, f"{sel['data_value'].min()} to {sel['data_value'].max()}", "within 0-100")
    ordered = ((sel["low_confidence_limit"] <= sel["data_value"])
               & (sel["data_value"] <= sel["high_confidence_limit"])).all()
    check("P15", "Ranges", "Lower limit <= estimate <= upper limit in every row",
          ordered, int((~((sel["low_confidence_limit"] <= sel["data_value"])
                          & (sel["data_value"] <= sel["high_confidence_limit"]))).sum()), 0)
    pop_const = sel.groupby("locationid")[["totalpopulation", "totalpop18plus"]].nunique().eq(1).all().all()
    check("P16", "CDC keys", "Population fields are constant within county",
          pop_const, pop_const, True)
    check("P17", "Ranges", "Adult population is positive and no larger than total population",
          ((sel["totalpop18plus"] > 0) & (sel["totalpop18plus"] <= sel["totalpopulation"])).all(),
          f"total population {int(sel['totalpopulation'].min()):,} to {int(sel['totalpopulation'].max()):,}",
          "0 < adults <= total")

    # --- county attributes -------------------------------------------------
    coords = sel["geolocation"].map(lambda s: ast.literal_eval(s)["coordinates"])
    sel["lon"], sel["lat"] = coords.map(lambda xy: float(xy[0])), coords.map(lambda xy: float(xy[1]))
    counties = (sel.groupby("locationid", as_index=False)
                .agg(county=("locationname", "first"), population=("totalpopulation", "first"),
                     pop18plus=("totalpop18plus", "first"), lon=("lon", "first"), lat=("lat", "first"))
                .rename(columns={"locationid": "fips"}))
    counties[["population", "pop18plus"]] = counties[["population", "pop18plus"]].astype(int)
    in_nc = counties["lon"].between(-84.5, -75.3).all() and counties["lat"].between(33.7, 36.7).all()
    check("P18", "Ranges", "County centroids fall inside the North Carolina bounding box",
          in_nc, f"lon {counties['lon'].min():.2f} to {counties['lon'].max():.2f}, "
                 f"lat {counties['lat'].min():.2f} to {counties['lat'].max():.2f}",
          "lon -84.5 to -75.3, lat 33.7 to 36.7")

    # --- long -> wide: one row per county, three columns per measure ---------
    sel["col"] = sel["measureid"].map(lambda m: C.MEASURES[m]["col"])
    wide = counties.copy()
    for source_col, suffix in (("data_value", ""), ("low_confidence_limit", "_lcl"),
                               ("high_confidence_limit", "_ucl")):
        part = sel.pivot(index="locationid", columns="col", values=source_col)
        part.columns = [f"{c}{suffix}" for c in part.columns]
        wide = wide.merge(part, left_on="fips", right_index=True, how="left", validate="one_to_one")
    ordered_cols = ["fips", "county", "population", "pop18plus", "lon", "lat"]
    for m in C.MEASURES.values():
        ordered_cols += [m["col"], f"{m['col']}_lcl", f"{m['col']}_ucl"]
    wide = wide[ordered_cols].sort_values("fips").reset_index(drop=True)
    count("Pivot to one row per county", len(wide), wide["fips"].nunique(), "21 measure columns")
    check("P19", "CDC structure", "Wide table has 100 unique counties and no missing cells",
          len(wide) == 100 and wide["fips"].is_unique and not wide.isna().any().any(),
          f"{len(wide)} rows, {int(wide.isna().sum().sum())} missing cells", "100 rows, 0 missing cells")

    measure_info = (sel.groupby("measureid")
                    .agg(measure=("measure", "first"), category=("category", "first"),
                         year=("year", "first"), unit=("data_value_unit", "first"),
                         value_type=("data_value_type", "first"), datasource=("datasource", "first"))
                    .reset_index())
    return wide, measure_info


# --------------------------------------------------------------------------
# 2. EPA annual AQI
# --------------------------------------------------------------------------
EPA_RENAME = {
    "Days with AQI": "aqi_days",
    "Good Days": "aqi_good_days",
    "Moderate Days": "aqi_moderate_days",
    "Unhealthy for Sensitive Groups Days": "aqi_usg_days",
    "Unhealthy Days": "aqi_unhealthy_days",
    "Very Unhealthy Days": "aqi_very_unhealthy_days",
    "Hazardous Days": "aqi_hazardous_days",
    "Max AQI": "aqi_max",
    "90th Percentile AQI": "aqi_p90",
    "Median AQI": "aqi_median",
    "Days CO": "aqi_days_co",
    "Days NO2": "aqi_days_no2",
    "Days Ozone": "aqi_days_ozone",
    "Days PM2.5": "aqi_days_pm25",
    "Days PM10": "aqi_days_pm10",
}
CATEGORY_COLS = ["aqi_good_days", "aqi_moderate_days", "aqi_usg_days", "aqi_unhealthy_days",
                 "aqi_very_unhealthy_days", "aqi_hazardous_days"]
ABOVE_100_COLS = ["aqi_usg_days", "aqi_unhealthy_days", "aqi_very_unhealthy_days", "aqi_hazardous_days"]
POLLUTANT_COLS = ["aqi_days_co", "aqi_days_no2", "aqi_days_ozone", "aqi_days_pm25", "aqi_days_pm10"]


def prepare_epa() -> pd.DataFrame:
    raw = pd.read_csv(C.SNAPSHOT_EPA)
    count("EPA raw snapshot (NC)", len(raw), raw["County"].nunique(), "counties with 2023 AQI records")

    check("E01", "EPA structure", "Required columns are present",
          set(EPA_RENAME) <= set(raw.columns), sorted(set(EPA_RENAME) - set(raw.columns)) or "none missing",
          "none missing")
    check("E02", "EPA coverage", "Every row is North Carolina, year 2023",
          raw["State"].eq(C.STATE_NAME).all() and raw["Year"].eq(C.ANALYSIS_YEAR).all(),
          f"{raw['State'].unique().tolist()} {raw['Year'].unique().tolist()}",
          f"['{C.STATE_NAME}'] [{C.ANALYSIS_YEAR}]")
    check("E03", "EPA keys", "One row per county (no duplicate county names)",
          raw["County"].is_unique, f"{len(raw)} rows, {raw['County'].nunique()} counties", "equal")

    epa = raw.rename(columns=EPA_RENAME)
    num = list(EPA_RENAME.values())
    check("E04", "Missingness", "No missing values in AQI fields",
          not epa[num].isna().any().any(), int(epa[num].isna().sum().sum()), 0)
    check("E05", "Ranges", "Days with AQI between 1 and 365",
          epa["aqi_days"].between(1, 365).all(),
          f"{int(epa['aqi_days'].min())} to {int(epa['aqi_days'].max())}", "1 to 365")
    cat_sum = epa[CATEGORY_COLS].sum(axis=1)
    check("E06", "Units", "Six AQI category counts add up to days with AQI",
          cat_sum.eq(epa["aqi_days"]).all(), int((cat_sum != epa["aqi_days"]).sum()), 0)
    pol_sum = epa[POLLUTANT_COLS].sum(axis=1)
    check("E07", "Units", "Main-pollutant day counts add up to days with AQI",
          pol_sum.eq(epa["aqi_days"]).all(), int((pol_sum != epa["aqi_days"]).sum()), 0)
    order_ok = ((epa["aqi_median"] <= epa["aqi_p90"]) & (epa["aqi_p90"] <= epa["aqi_max"])).all()
    check("E08", "Ranges", "Median AQI <= 90th percentile <= maximum, all within 0-500",
          order_ok and epa[["aqi_median", "aqi_p90", "aqi_max"]].stack().between(0, 500).all(),
          f"median {int(epa['aqi_median'].min())}-{int(epa['aqi_median'].max())}, "
          f"max {int(epa['aqi_max'].min())}-{int(epa['aqi_max'].max())}", "ordered, 0-500")
    above = epa[ABOVE_100_COLS].sum(axis=1)
    check("E09", "Ranges", "Maximum AQI > 100 exactly when at least one day is above 100",
          ((epa["aqi_max"] > 100) == (above > 0)).all(),
          int(((epa["aqi_max"] > 100) != (above > 0)).sum()), 0)

    # --- derived air-quality measures --------------------------------------
    epa["aqi_days_gt100"] = above
    epa["aqi_prop_gt100"] = above / epa["aqi_days"]
    epa["aqi_ge300_days"] = (epa["aqi_days"] >= C.AQI_MIN_DAYS).astype(int)
    check("D01", "Derived", "Share of monitored days with AQI > 100 lies between 0 and 1",
          epa["aqi_prop_gt100"].between(0, 1).all(),
          f"{epa['aqi_prop_gt100'].min():.4f} to {epa['aqi_prop_gt100'].max():.4f}", "0 to 1")
    epa["name_key"] = epa["County"].map(norm_name)
    return epa


# --------------------------------------------------------------------------
# 3. Join
# --------------------------------------------------------------------------
def join(places: pd.DataFrame, epa: pd.DataFrame) -> pd.DataFrame:
    places = places.assign(name_key=places["county"].map(norm_name))
    check("J01", "Join", "Normalised county names are unique on both sides",
          places["name_key"].is_unique and epa["name_key"].is_unique,
          f"CDC {places['name_key'].nunique()}, EPA {epa['name_key'].nunique()}", "CDC 100, EPA = EPA rows")

    unmatched = sorted(set(epa["name_key"]) - set(places["name_key"]))
    check("J02", "Join", "Every EPA county matches a CDC county",
          not unmatched, unmatched or "0 unmatched", "0 unmatched")

    keep = ["name_key", "County"] + list(EPA_RENAME.values()) + ["aqi_days_gt100", "aqi_prop_gt100", "aqi_ge300_days"]
    before = len(places)
    merged = places.merge(epa[keep], on="name_key", how="left", validate="one_to_one", indicator=True)
    check("J03", "Join", "Left join keeps exactly the 100 CDC counties",
          len(merged) == before == 100 and merged["fips"].is_unique, f"{before} before, {len(merged)} after", "100, 100")
    merged["has_aqi"] = merged["_merge"].eq("both").astype(int)
    check("J04", "Join", "Counties with AQI records after the join equal EPA rows",
          merged["has_aqi"].sum() == len(epa), f"{int(merged['has_aqi'].sum())} of {len(epa)}", "all EPA rows matched once")
    check("J05", "Timing", "Health indicators and AQI summaries refer to the same year",
          True, f"PLACES survey year {C.ANALYSIS_YEAR}; EPA summary year {C.ANALYSIS_YEAR}", "same year")

    match = (merged.loc[merged["has_aqi"].eq(1), ["County", "county", "fips", "aqi_days"]]
             .rename(columns={"County": "epa_county", "county": "cdc_county"}))
    match.to_csv(C.AUDIT / "join_county_match.csv", index=False)

    merged["aqi_ge300_days"] = merged["aqi_ge300_days"].fillna(0).astype(int)
    merged = merged.drop(columns=["name_key", "County", "_merge"])
    count("Left join of EPA onto CDC counties", len(merged), merged["fips"].nunique(),
          f"{int(merged['has_aqi'].sum())} counties with AQI records; none imputed")
    count(f"AQI subset with >= {C.AQI_MIN_DAYS} monitored days", int(merged["aqi_ge300_days"].sum()),
          int(merged["aqi_ge300_days"].sum()), "coverage sensitivity subset")
    return merged


# --------------------------------------------------------------------------
# 4. Map boundaries (drawn in the county map only; never merged into the table)
# --------------------------------------------------------------------------
def check_boundaries(table: pd.DataFrame) -> None:
    features = C.read_json(C.SNAPSHOT_BOUNDARIES)["features"]
    props = pd.DataFrame([f["properties"] for f in features])
    check("B01", "Map boundaries", "Boundary file has one record per county, all in state FIPS 37",
          len(props) == 100 and props["GEOID"].is_unique and props["STATEFP"].eq(C.STATE_FIPS).all(),
          f"{len(props)} records, {props['GEOID'].nunique()} distinct codes", "100 records, 100 distinct codes")
    unmatched = set(props["GEOID"]) ^ set(table["fips"])
    check("B02", "Join", "Boundary county codes equal the 100 CDC county FIPS codes",
          not unmatched, f"{len(unmatched)} codes unmatched", "0 codes unmatched")
    both = props.merge(table[["fips", "county", "lon", "lat"]], left_on="GEOID", right_on="fips",
                       validate="one_to_one")
    differ = int((both["NAME"] != both["county"]).sum())
    check("B03", "Join", "County names agree between the boundary file and CDC",
          differ == 0, f"{differ} names differ", "0 names differ")

    boxes = {}
    for f in features:
        xy = np.array([pt for ring in C.polygon_rings(f["geometry"]) for pt in ring])
        boxes[f["properties"]["GEOID"]] = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    box = pd.DataFrame.from_dict(boxes, orient="index", columns=["x0", "x1", "y0", "y1"])
    check("B04", "Ranges", "Boundary coordinates fall inside the North Carolina bounding box",
          box["x0"].min() >= -84.5 and box["x1"].max() <= -75.3 and box["y0"].min() >= 33.7 and box["y1"].max() <= 36.7,
          f"lon {box['x0'].min():.2f} to {box['x1'].max():.2f}, lat {box['y0'].min():.2f} to {box['y1'].max():.2f}",
          "lon -84.5 to -75.3, lat 33.7 to 36.7")
    b = both.set_index("GEOID").join(box)
    outside = int((~(b["lon"].between(b["x0"], b["x1"]) & b["lat"].between(b["y0"], b["y1"]))).sum())
    check("B05", "Map boundaries", "Each CDC county centroid lies within the extent of its boundary polygon",
          outside == 0, f"{outside} centroids outside", "0 centroids outside")


# --------------------------------------------------------------------------
# 5. Data dictionary
# --------------------------------------------------------------------------
def build_dictionary(table: pd.DataFrame, measure_info: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("fips", "Five-digit county FIPS code (join key)", "code", "CDC PLACES LocationID", ""),
        ("county", "County name", "text", "CDC PLACES LocationName", ""),
        ("population", "Total resident population, Census 2023 estimate", "persons", "CDC PLACES TotalPopulation", "2023"),
        ("pop18plus", "Resident population aged 18 or older, Census 2023 estimate", "persons", "CDC PLACES TotalPop18plus", "2023"),
        ("lon", "Longitude of county centroid", "degrees", "CDC PLACES Geolocation", ""),
        ("lat", "Latitude of county centroid", "degrees", "CDC PLACES Geolocation", ""),
    ]
    info = measure_info.set_index("measureid")
    for mid, m in C.MEASURES.items():
        r = info.loc[mid]
        desc = f"{r['measure']}: {r['value_type'].lower()} ({m['role']})"
        src = f"CDC PLACES MeasureId {mid}, model-based estimate from {r['datasource']}"
        rows.append((m["col"], desc, "percent", src, r["year"]))
        rows.append((f"{m['col']}_lcl", f"Lower 95% confidence limit published by CDC for {m['col']}", "percent", src, r["year"]))
        rows.append((f"{m['col']}_ucl", f"Upper 95% confidence limit published by CDC for {m['col']}", "percent", src, r["year"]))
    epa_src = "EPA AirData annual AQI by county"
    rows += [
        ("aqi_days", "Days in the year with an AQI value from any monitoring site in the county", "days", epa_src, "2023"),
        ("aqi_good_days", "Days with AQI 0-50", "days", epa_src, "2023"),
        ("aqi_moderate_days", "Days with AQI 51-100", "days", epa_src, "2023"),
        ("aqi_usg_days", "Days with AQI 101-150 (unhealthy for sensitive groups)", "days", epa_src, "2023"),
        ("aqi_unhealthy_days", "Days with AQI 151-200", "days", epa_src, "2023"),
        ("aqi_very_unhealthy_days", "Days with AQI 201-300", "days", epa_src, "2023"),
        ("aqi_hazardous_days", "Days with AQI 301 or higher", "days", epa_src, "2023"),
        ("aqi_max", "Highest daily AQI value in the year", "AQI index", epa_src, "2023"),
        ("aqi_p90", "90th percentile of daily AQI values", "AQI index", epa_src, "2023"),
        ("aqi_median", "Median of daily AQI values", "AQI index", epa_src, "2023"),
        ("aqi_days_co", "Days on which CO was the main pollutant", "days", epa_src, "2023"),
        ("aqi_days_no2", "Days on which NO2 was the main pollutant", "days", epa_src, "2023"),
        ("aqi_days_ozone", "Days on which ozone was the main pollutant", "days", epa_src, "2023"),
        ("aqi_days_pm25", "Days on which PM2.5 was the main pollutant", "days", epa_src, "2023"),
        ("aqi_days_pm10", "Days on which PM10 was the main pollutant", "days", epa_src, "2023"),
        ("aqi_days_gt100", "Days with AQI above 100 (sum of the four categories above 100)", "days", "derived from EPA counts", "2023"),
        ("aqi_prop_gt100", "aqi_days_gt100 divided by aqi_days", "proportion (0-1)", "derived from EPA counts", "2023"),
        ("aqi_ge300_days", f"1 if aqi_days >= {C.AQI_MIN_DAYS}, else 0 (coverage sensitivity flag)", "0/1", "derived", "2023"),
        ("has_aqi", "1 if the county has a 2023 EPA AQI summary, else 0", "0/1", "derived from the join", "2023"),
    ]
    dd = pd.DataFrame(rows, columns=["variable", "description", "unit", "source", "year"])

    # Denominator of each rate: taken from the publisher's measure name for PLACES measures.
    denominator = {"aqi_prop_gt100": "days with an AQI value (aqi_days)"}
    for mid, m in C.MEASURES.items():
        adults = ("adults aged 18-64 years" if "18-64" in info.loc[mid, "measure"]
                  else "adults aged 18 years or older")
        for suffix in ("", "_lcl", "_ucl"):
            denominator[f"{m['col']}{suffix}"] = adults
    dd.insert(3, "denominator", dd["variable"].map(denominator).fillna(""))
    dd["non_missing"] = dd["variable"].map(lambda v: int(table[v].notna().sum()))
    missing_in_table = set(dd["variable"]) ^ set(table.columns)
    check("D02", "Derived", "Data dictionary documents every column of the processed table",
          not missing_in_table, sorted(missing_in_table) or "all columns documented", "all columns documented")
    return dd


# --------------------------------------------------------------------------
def main() -> None:
    C.ensure_dirs()
    places, measure_info = prepare_places()
    epa = prepare_epa()
    table = join(places, epa)
    check_boundaries(table)

    core = ["population"] + [m["col"] for m in C.MEASURES.values()]
    check("M01", "Missingness", "Seven health indicators complete for all 100 counties",
          not table[core].isna().any().any(), int(table[core].isna().sum().sum()), 0)
    check("M02", "Missingness", "AQI fields are missing exactly for counties without EPA records",
          table.loc[table["has_aqi"].eq(0), "aqi_median"].isna().all()
          and table.loc[table["has_aqi"].eq(1), "aqi_median"].notna().all(),
          f"{int(table['aqi_median'].isna().sum())} counties without AQI summary", "100 - EPA rows")

    dictionary = build_dictionary(table, measure_info)
    table.to_csv(C.PROCESSED_TABLE, index=False)
    dictionary.to_csv(C.DATA_DICTIONARY, index=False)

    missing = (table.isna().sum().rename("n_missing").to_frame()
               .assign(n=len(table), pct_missing=lambda d: (100 * d["n_missing"] / d["n"]).round(1))
               .rename_axis("variable").reset_index())
    missing.to_csv(C.AUDIT / "missingness.csv", index=False)

    checks = pd.DataFrame(CHECKS)
    checks.to_csv(C.AUDIT / "audit_checks.csv", index=False)
    rows = pd.DataFrame(ROW_COUNTS)
    rows.to_csv(C.AUDIT / "row_counts.csv", index=False)

    print(checks[["id", "check", "observed", "result"]].to_string(index=False))
    print()
    print(rows.to_string(index=False))

    aqi = table[table["has_aqi"].eq(1)]
    C.update_results("audit", {
        "n_checks": len(checks),
        "n_passed": int(checks["result"].eq("PASS").sum()),
        "n_failed": int(checks["result"].eq("FAIL").sum()),
        "n_counties": len(table),
        "raw_rows": int(rows.loc[0, "rows"]),
        "selected_rows": 700,
        "n_with_aqi": int(table["has_aqi"].sum()),
        "n_without_aqi": int((table["has_aqi"] == 0).sum()),
        "n_aqi_ge300": int(table["aqi_ge300_days"].sum()),
        "aqi_days_min": int(aqi["aqi_days"].min()),
        "aqi_days_max": int(aqi["aqi_days"].max()),
        "aqi_days_median": float(aqi["aqi_days"].median()),
        "population_min": int(table["population"].min()),
        "population_min_county": table.loc[table["population"].idxmin(), "county"],
        "population_max": int(table["population"].max()),
        "population_max_county": table.loc[table["population"].idxmax(), "county"],
        "population_total": int(table["population"].sum()),
        "processed_sha256": C.sha256_file(C.PROCESSED_TABLE),
    })

    failed = checks[checks["result"].eq("FAIL")]
    if len(failed):
        raise SystemExit(f"{len(failed)} audit check(s) failed: {failed['id'].tolist()}")
    print(f"\nAll {len(checks)} audit checks passed. Processed table: {C.rel(C.PROCESSED_TABLE)}")


if __name__ == "__main__":
    main()
