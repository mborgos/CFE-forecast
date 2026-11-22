# Downloads regional electricity generation and GFS weather data
# File names, column names, source order, emission factors and the weather post-processing follow CarbonCast
# (https://github.com/carbonfirst/CarbonCast, Apache License 2.0); the weather horizon is 168 h instead of 96 h.
# Parts adapted from CarbonCast are marked below.
import os, json, time, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dataclasses import dataclass

from typing import cast
from abc import ABC, abstractmethod

import click

@dataclass
class Bounding_box:
    nlat: float
    slat: float
    wlon: float
    elon: float

@dataclass
class RegionData:
    provider: str
    code: str
    bbox: Bounding_box
    tz: str

load_dotenv('config.env')

DOWNLOADED_REGIONS = [r.strip() for r in os.environ.get("DOWNLOADED_REGIONS", "").split(",") if r.strip()]   # comma separated, empty = all regions
START  = os.environ.get("START", "2024-01-01")
END    = os.environ.get("END", "2024-02-01")

EIA_API_KEY             = os.environ.get("EIA_API_KEY", "")
ENTSOE_API_KEY          = os.environ.get("ENTSOE_API_KEY", "")
OPENELECTRICITY_API_KEY = os.environ.get("OPENELECTRICITY_API_KEY", "")

DOWNLOAD_DA_FORECASTS = True     # day-ahead solar/wind forecasts where available (ENTSO-E, CAISO)
GFS_SOURCE     = "aws"           # "aws" (>= 2021-02-26, no account) or "gdex_local" (files you downloaded from GDEX)
GFS_RESOLUTION = os.environ.get("GFS_RESOLUTION", "0p25")          # "0p25" like the paper; "0p50"/"1p00" are 4x/16x smaller downloads
GFS_WORKERS    = 8               # parallel HTTP downloads

OUT_DIR = Path(os.environ.get("OUT_DIR", "data"))     # files go to <OUT_DIR>/<REGION>/, like CarbonCast's data/<REGION>/
RUN_RETRIES = 3                                       # attempts per daily GFS run

def region_dir(region: str) -> Path:
    out = OUT_DIR / region
    for sub in ("", "fuel_forecast", "weather_data", "weather_data/gfs_cache"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out

def make_session(workers: int = 1) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=workers, pool_maxsize=workers))
    return session

def with_retries(call, what: str, attempts: int = 3, give_up_on: tuple = ()):
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except give_up_on:
            raise
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"  {what}: attempt {attempt}/{attempts} failed ({exc}), retrying")
            time.sleep(5 * attempt)


@click.group()
def main():
    pass


# provider, provider code, bbox (nlat, slat, wlon, elon) from CarbonCast src/weather/ds084.1_control.ctl


REGIONS_DATA_MAP = {
    # --- US (EIA-930 balancing authorities) ---
    "CISO":  RegionData(provider="eia", code="CISO", bbox=Bounding_box(nlat=42, slat=32, wlon=-124.75, elon=-113.5),   tz="US/Pacific"),
    "PJM":   RegionData(provider="eia", code="PJM",  bbox=Bounding_box(nlat=43, slat=34.25, wlon=-91, elon=-73.5),     tz="US/Eastern"),
    "ERCO":  RegionData(provider="eia", code="ERCO", bbox=Bounding_box(nlat=36.5, slat=25.25, wlon=-104.5, elon=-93.25), tz="US/Central"),
    "ISNE":  RegionData(provider="eia", code="ISNE", bbox=Bounding_box(nlat=48, slat=40, wlon=-74.25, elon=-66.5),     tz="US/Eastern"),
    "NYISO": RegionData(provider="eia", code="NYIS", bbox=Bounding_box(nlat=45.5, slat=40, wlon=-80.25, elon=-71.25),  tz="US/Eastern"),
    "FPL":   RegionData(provider="eia", code="FPL",  bbox=Bounding_box(nlat=31.25, slat=24, wlon=-83.5, elon=-79.5),   tz="US/Eastern"),
    "BPAT":  RegionData(provider="eia", code="BPAT", bbox=Bounding_box(nlat=49.5, slat=39.5, wlon=-125.25, elon=-105.5), tz="US/Pacific"),
    "MISO":  RegionData(provider="eia", code="MISO", bbox=Bounding_box(nlat=50, slat=28.5, wlon=-107.75, elon=-81.75), tz="US/Eastern"),
    "SWPP":  RegionData(provider="eia", code="SWPP", bbox=Bounding_box(nlat=49.5, slat=30.25, wlon=-107.75, elon=-89.5), tz="US/Central"),
    "SOCO":  RegionData(provider="eia", code="SOCO", bbox=Bounding_box(nlat=35.5, slat=29.25, wlon=-90.5, elon=-80.25), tz="US/Central"),
    # --- Europe (ENTSO-E areas) ---
    "DE":    RegionData(provider="entsoe", code="DE",   bbox=Bounding_box(nlat=55.25, slat=47.25, wlon=5.75, elon=15),  tz="CET"),
    "NL":    RegionData(provider="entsoe", code="NL",   bbox=Bounding_box(nlat=53.5, slat=50.75, wlon=3.25, elon=7),    tz="CET"),
    "ES":    RegionData(provider="entsoe", code="ES",   bbox=Bounding_box(nlat=43.75, slat=36, wlon=-9.25, elon=3.5),   tz="CET"),
    "SE":    RegionData(provider="entsoe", code="SE",   bbox=Bounding_box(nlat=69, slat=55.25, wlon=11.25, elon=21.25), tz="CET"),
    "PL":    RegionData(provider="entsoe", code="PL",   bbox=Bounding_box(nlat=54.75, slat=49, wlon=14, elon=24),       tz="CET"),
    "FR":    RegionData(provider="entsoe", code="FR",   bbox=Bounding_box(nlat=51.25, slat=42.25, wlon=-5.25, elon=8.25), tz="CET"),
    "BE":    RegionData(provider="entsoe", code="BE",   bbox=Bounding_box(nlat=51.5, slat=49.5, wlon=2.5, elon=6.25),   tz="CET"),
    "FI":    RegionData(provider="entsoe", code="FI",   bbox=Bounding_box(nlat=70, slat=59.75, wlon=20.5, elon=31.5),   tz="EET"),
    "DK-DK2":RegionData(provider="entsoe", code="DK_2", bbox=Bounding_box(nlat=57.75, slat=54.75, wlon=7.25, elon=11.25), tz="CET"),
    # --- Australia (NEM regions via Open Electricity) ---
    "AUS_QLD": RegionData(provider="openelectricity", code="QLD1", bbox=Bounding_box(nlat=-8.75, slat=-29.75, wlon=137.5, elon=154), tz="Australia/Brisbane"),
    "AUS_NSW": RegionData(provider="openelectricity", code="NSW1", bbox=Bounding_box(nlat=-34.75, slat=-36.5, wlon=148.25, elon=150), tz="Australia/Sydney"),
    "AUS_SA":  RegionData(provider="openelectricity", code="SA1",  bbox=Bounding_box(nlat=-25.5, slat=-38.5, wlon=128.5, elon=141.5), tz="Australia/Adelaide"),

}

REGION_NAMES = DOWNLOADED_REGIONS or list(REGIONS_DATA_MAP)
for _region in REGION_NAMES:
    if _region not in REGIONS_DATA_MAP:
        raise SystemExit(f"Unknown region {_region}. Choose from: {', '.join(REGIONS_DATA_MAP)}")
START_TS, END_TS = pd.Timestamp(START), pd.Timestamp(END)
HOURS = pd.date_range(START_TS, END_TS, freq="h", inclusive="left", name="UTC time")

# Source names and column order used by CarbonCast's models for its original regions (from CarbonCast
# src/firstTierConfig.json). Other regions: the order found in the data.
_US_SOURCES = ["coal", "nat_gas", "nuclear", "oil", "hydro", "solar", "wind", "other"]
SOURCE_ORDER = {
    "CISO": _US_SOURCES, "PJM": _US_SOURCES, "ERCO": _US_SOURCES, "ISNE": _US_SOURCES,
    "SE": ["nuclear", "unknown", "wind", "hydro"],
    "DE": ["biomass", "coal", "nat_gas", "geothermal", "hydro", "nuclear", "oil", "solar", "wind", "unknown"],
    "AUS_QLD": ["coal", "nat_gas", "oil", "hydro", "wind", "biomass", "solar"],
    "BPAT": ["nat_gas", "nuclear", "hydro", "solar", "wind", "other"],
    "ES": ["solar", "unknown", "nat_gas", "coal", "wind", "hydro", "nuclear", "biomass", "oil"],
    "FPL": ["nat_gas", "nuclear", "solar", "other"],
    "NYISO": ["nat_gas", "nuclear", "oil", "hydro", "wind", "other"],
    "PL": ["wind", "hydro", "biomass", "nat_gas", "oil", "coal", "solar"],
    "NL": ["hydro", "biomass", "wind", "solar", "unknown", "coal", "nuclear", "nat_gas"],
}

# END OF CONSTANTS

class Electricity_data_fetcher(ABC):
    @abstractmethod
    def fetch(self, region_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError


# EIA - US
EIA_FUEL_MAP: dict[str, str | None] = {"COL": "coal", "NG": "nat_gas", "NUC": "nuclear", "OIL": "oil", "WAT": "hydro",
                "SUN": "solar", "WND": "wind", "GEO": "geothermal", "OTH": "other", "UNK": "unknown",
                # storage: not generation -> dropped
                "BAT": None, "PS": None, "SNB": None, "OES": None}

def month_chunks(start, end):
    edges = list(pd.date_range(start.normalize().replace(day=1), end, freq="MS"))
    edges = [start] + [e for e in edges if start < e < end] + [end]
    return list(zip(edges[:-1], edges[1:]))

class EIA_Fetcher(Electricity_data_fetcher):

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.session = make_session()

    def fetch(self, region_code: str, start: pd.Timestamp, end: pd.Timestamp): 
        assert self.api_key, "Set EIA_API_KEY (https://www.eia.gov/opendata/)"
        url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
        rows = []
        for s, e in month_chunks(start, end):
            offset = 0
            while True:
                params = {"api_key": self.api_key, "frequency": "hourly", "data[0]": "value",
                        "facets[respondent][]": region_code,
                        "start": s.strftime("%Y-%m-%dT%H"), "end": (e - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H"),
                        "sort[0][column]": "period", "sort[0][direction]": "asc",
                        "offset": offset, "length": 5000}
                r = self.session.get(url, params=params, timeout=120)
                r.raise_for_status()
                resp = r.json()["response"]
                data = resp.get("data", [])
                rows += data
                offset += len(data)
                if not data or offset >= int(resp.get("total", 0)):
                    break
            print(f"  EIA {region_code} {s.date()} - {e.date()}: {offset} rows")
        df = pd.DataFrame(rows)
        df["period"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        unknown = sorted(set(df["fueltype"]) - set(EIA_FUEL_MAP))
        if unknown:
            print("  Unmapped EIA fuel types (counted as 'other'):", unknown)
        df["source"] = df["fueltype"].map(lambda f: EIA_FUEL_MAP.get(cast(str, f), "other"))
        df = df[df["source"].notna()]
        return df.pivot_table(index="period", columns="source", values="value", aggfunc="sum")

# ENTSOE - EU

ENTSOE_MAP = {
    "Biomass": "biomass", "Waste": "biomass",
    "Fossil Brown coal/Lignite": "coal", "Fossil Hard coal": "coal", "Fossil Coal-derived gas": "coal",
    "Fossil Peat": "coal", "Fossil Oil shale": "coal",
    "Fossil Gas": "nat_gas", "Fossil Oil": "oil", "Nuclear": "nuclear", "Geothermal": "geothermal",
    "Hydro Run-of-river and poundage": "hydro", "Hydro Water Reservoir": "hydro", "Hydro Pumped Storage": "hydro",
    "Solar": "solar", "Wind Onshore": "wind", "Wind Offshore": "wind",
    "Other": "unknown", "Other renewable": "unknown", "Marine": "unknown",
    "Energy storage": None,
}

def _to_hourly_utc(df: pd.DataFrame):
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").tz_localize(None)
    return df.resample("h").mean()

class Entsoe_Fetcher (Electricity_data_fetcher):

    def __init__(self, api_key: str):
        super().__init__()

        assert api_key, "Set ENTSOE_API_KEY"
        from entsoe.entsoe import EntsoePandasClient
        self.entsoe_client = EntsoePandasClient(api_key=api_key)

    def fetch(self, region_code: str, start: pd.Timestamp, end: pd.Timestamp):
        from entsoe.exceptions import NoMatchingDataError
        client, parts = self.entsoe_client, []
        for s, e in month_chunks(start, end):
            try:
                g = with_retries(lambda: client.query_generation(region_code, start=pd.Timestamp(s, tz="UTC"), end=pd.Timestamp(e, tz="UTC")),
                                 f"ENTSO-E {region_code} {s.date()}", give_up_on=(NoMatchingDataError,))
            except NoMatchingDataError:
                print(f"  ENTSO-E {region_code} {s.date()}: no data"); continue
            if isinstance(g, pd.Series):
                g = g.to_frame()
            if isinstance(g.columns, pd.MultiIndex):
                keep = [c for c in g.columns if c[1] == "Actual Aggregated"]
                g = g[keep]; g.columns = [c[0] for c in keep]
            parts.append(_to_hourly_utc(g))
            print(f"  ENTSO-E {region_code} {s.date()} - {e.date()}: {len(g)} rows, types: {list(g.columns)}")
        raw = pd.concat(parts).sort_index()
        raw = raw[~raw.index.duplicated()]
        unknown = sorted(set(raw.columns) - set(ENTSOE_MAP))
        if unknown:
            print("  Unmapped ENTSO-E types (counted as 'unknown'):", unknown)
        names = [ENTSOE_MAP.get(c, "unknown") for c in raw.columns]
        keep = [n is not None for n in names]
        return raw.loc[:, keep].T.groupby([n for n in names if n is not None]).sum(min_count=1).T

# Open electricity - australia

def map_fueltech(ft):
    ft = str(ft).lower()
    if ft.startswith("coal"): return "coal"
    if ft.startswith("gas"): return "nat_gas"
    if ft in ("distillate",) or ft.startswith("oil"): return "oil"
    if ft.startswith("hydro"): return "hydro"
    if ft.startswith("wind"): return "wind"
    if ft.startswith("solar"): return "solar"
    if ft.startswith("bioenergy") or ft.startswith("biomass"): return "biomass"
    if ft.startswith("nuclear"): return "nuclear"
    return None   # battery, pumps, imports/exports, ...


class OpenElectricity_Fetcher (Electricity_data_fetcher):

    def __init__(self, api_key: str):
        super().__init__()
        assert api_key, "Set OPENELECTRICITY_API_KEY (https://platform.openelectricity.org.au)"
        self.api_key = api_key

    def fetch(self, region_code: str, start: pd.Timestamp, end: pd.Timestamp):
        os.environ["OPENELECTRICITY_API_KEY"] = self.api_key
        from openelectricity import OEClient
        from openelectricity.types import DataMetric
        NEM_OFFSET = pd.Timedelta(hours=10)
        parts = []
        with OEClient() as client:
            for s, e in month_chunks(start, end):
                resp = with_retries(lambda: client.get_network_data(network_code="NEM", metrics=[DataMetric.POWER], interval="1h",
                                            date_start=(s + NEM_OFFSET).to_pydatetime(),
                                            date_end=(e + NEM_OFFSET).to_pydatetime(),
                                            primary_grouping="network_region", secondary_grouping="fueltech"),
                                    f"Open Electricity {region_code} {s.date()}")
                df = resp.to_pandas()
                reg_col = next((c for c in ("network_region", "region") if c in df.columns), None)
                val_col = next((c for c in ("power", "value") if c in df.columns), None)
                assert reg_col and val_col and "fueltech" in df.columns, f"Unexpected columns: {list(df.columns)}"
                df = df[df[reg_col].astype(str).str.upper() == region_code]
                parts.append(df[["interval", "fueltech", val_col]].rename(columns={val_col: "value"}))
                print(f"  Open Electricity {region_code} {s.date()} - {e.date()}: {len(df)} rows")
        df = pd.concat(parts)
        df["interval"] = pd.to_datetime(df["interval"])
        if df["interval"].dt.tz is not None:
            df["interval"] = df["interval"].dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            df["interval"] = df["interval"] - NEM_OFFSET          # naive NEM time -> UTC
        df["source"] = df["fueltech"].map(map_fueltech)
        print("  Fuel techs left out:", sorted(df.loc[df["source"].isna(), "fueltech"].unique()))
        df = df[df["source"].notna()]
        return df.pivot_table(index="interval", columns="source", values="value", aggfunc="sum")

# End of fetchers

def clean_generation(raw, hours, source_order):
    df = raw.copy()
    df.index = pd.DatetimeIndex(df.index)
    df = df[~df.index.duplicated()].reindex(hours)
    n_missing = int(df.isna().all(axis=1).sum())
    df = df.clip(lower=0).interpolate(limit=3, limit_area="inside").ffill().bfill().fillna(0)
    df = df.loc[:, (df != 0).any()] if not source_order else df       # drop all-zero columns for new regions
    if source_order:
        extra = [c for c in df.columns if c not in source_order]
        missing = [c for c in source_order if c not in df.columns]
        if extra:   print("Sources not in the repo's list for this region (kept at the end):", extra)
        if missing: print("Repo sources with no data in this period (filled with 0):", missing)
        for c in missing: df[c] = 0.0
        df = df[source_order + extra]
    print(f"{len(df)} hours, {n_missing} hours had no data at all and were filled")
    return df.round(2)

def write_repo_csv(df, path: Path, time_fmt="%Y-%m-%dT%H:%M"):
    out = df.copy(); out.index = out.index.strftime(time_fmt); out.index.name = "UTC time"
    out.to_csv(path)

# Emission factors (gCO2eq/kWh) and the SE exception: from CarbonCast src/carbonIntensityCalculator.py
CEF_LIFECYCLE: dict[str, float] = {"coal": 820, "biomass": 230, "nat_gas": 490, "geothermal": 38, "hydro": 24, "nuclear": 12,
                 "oil": 650, "solar": 45, "unknown": 700, "other": 700, "wind": 11}
CEF_DIRECT: dict[str, float] = {"coal": 760, "biomass": 0, "nat_gas": 370, "geothermal": 0, "hydro": 0, "nuclear": 0,
                 "oil": 406, "solar": 0, "unknown": 575, "other": 575, "wind": 0}
CEF_LIFECYCLE_SE = {**CEF_LIFECYCLE, "unknown": 292.9, "other": 292.9}   # special case for SE


def carbon_intensity(gen: pd.DataFrame, cef: dict[str, float]):
    total = gen.sum(axis=1).replace(0, np.nan)
    ci = (gen * pd.Series(cef)[gen.columns]).sum(axis=1) / total
    return ci.ffill().round(2)      

@main.command()
def electricity():
    fetchers = {"eia": lambda: EIA_Fetcher(EIA_API_KEY),
            "entsoe": lambda: Entsoe_Fetcher(ENTSOE_API_KEY),
            "openelectricity": lambda: OpenElectricity_Fetcher(OPENELECTRICITY_API_KEY)}
    failed = []
    for region in REGION_NAMES:
        try:
            out = region_dir(region)
            fetcher: Electricity_data_fetcher = fetchers[REGIONS_DATA_MAP[region].provider]()
            gen_raw = fetcher.fetch(REGIONS_DATA_MAP[region].code, START_TS, END_TS)
            gen_raw.to_csv(out / f"{region}_generation_raw.csv")

            gen = clean_generation(gen_raw, HOURS, SOURCE_ORDER.get(region, []))

            write_repo_csv(gen, out / f"{region}.csv", "%Y-%m-%d %H:%M:%S")           # input of CarbonCast's carbonIntensityCalculator.py -r
            write_repo_csv(gen, out / "fuel_forecast" / f"{region}_clean.csv")         # input of CarbonCast's firstTierForecasts.py

            for kind, cef in (("lifecycle", CEF_LIFECYCLE_SE if region == "SE" else CEF_LIFECYCLE), ("direct", CEF_DIRECT)):
                em = gen.copy()
                em.insert(0, "carbon_intensity", carbon_intensity(gen, cef))
                em = em.reset_index()
                em["UTC time"] = em["UTC time"].dt.strftime("%Y-%m-%d %H:%M:%S")
                em.to_csv(out / f"{region}_{kind}_emissions.csv")          # leading unnamed index column, like CarbonCast
        except Exception as ex:
            print(f"{region} FAILED: {ex}")
            failed.append(region)
    if failed:
        raise SystemExit("Electricity data failed for: " + ", ".join(failed))

# NOW weather data

def earth_radius(lat: np.ndarray):
    # WGS84 radius at geodetic latitude (Adapted CarbonCast implementation from src/weather/dataCollectionScript.py)
    a, b = 6378137, 6356752.3142
    e2 = 1 - (b**2 / a**2)
    lat_gc = np.arctan((1 - e2) * np.tan(np.deg2rad(lat)))
    return (a * (1 - e2) ** 0.5) / (1 - (e2 * np.cos(lat_gc) ** 2)) ** 0.5

def area_grid(lat: np.ndarray, lon: np.ndarray):
    # grid-cell areas in m^2, shape (lat, lon)  (Adapted CarbonCast implementation from src/weather/dataCollectionScript.py)
    xlon, ylat = np.meshgrid(lon, lat)
    R = earth_radius(ylat)
    dy = np.deg2rad(np.gradient(ylat, axis=0)) * R
    dx = np.deg2rad(np.gradient(xlon, axis=1)) * R * np.cos(np.deg2rad(ylat))
    return dy * dx

def crop_bbox(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, bbox: Bounding_box):
    # values (lat, lon) on a global grid -> sub-grid inside bbox; handles 0..360 longitudes and the 0° seam
    lon180 = ((np.asarray(lon) + 180) % 360) - 180
    order = np.argsort(lon180)
    lon180, values = lon180[order], np.asarray(values)[:, order]
    li = (lat >= bbox.slat) & (lat <= bbox.nlat)
    lj = (lon180 >= bbox.wlon) & (lon180 <= bbox.elon)
    return values[np.ix_(li, lj)], np.asarray(lat)[li], lon180[lj]

def regional_mean(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, bbox: Bounding_box):
    sub, la, lo = crop_bbox(values, lat, lon, bbox)
    w = np.abs(area_grid(la, lo))
    return float(np.nansum(sub * w) / np.sum(w[~np.isnan(sub)]))

LEADS = list(range(3, 169, 3))
VARS = ["wind", "temp", "dpt", "dswrf", "pcp"]

def interval_start(lead):          # 0-3, 0-6, 6-9, 6-12, ... (GFS 3/6-h buckets)
    return lead - 3 if (lead // 3) % 2 == 1 else lead - 6

@dataclass(frozen=True)
class Database_item:
    variable_name: str
    surface_level: str
    data_type: str

def get_gfs_wanted_fields(lead: int) -> dict[str, Database_item]:
    a = interval_start(lead)
    return {"u":     Database_item("UGRD", "10 m above ground", f"{lead} hour fcst"),
            "v":     Database_item("VGRD", "10 m above ground", f"{lead} hour fcst"),
            "temp":  Database_item("TMP",  "2 m above ground",  f"{lead} hour fcst"),
            "dpt":   Database_item("DPT",  "2 m above ground",  f"{lead} hour fcst"),
            "dswrf": Database_item("DSWRF", "surface", f"{a}-{lead} hour ave fcst"),
            "pcp":   Database_item("APCP",  "surface", f"{a}-{lead} hour acc fcst")}

@dataclass
class Byte_range:
    start: int # start position in the data file
    end: int | None # end position in the data file

@dataclass
class Database_index_item:
    fields: Database_item # what is contained under that item
    byte_range: Byte_range

def parse_idx(idx_text: str) -> dict[Database_item, Byte_range]:
    lines: list[list[str]] = [l.split(":") for l in idx_text.strip().splitlines()]
    out = {}
    for i, p in enumerate(lines):
        start = int(p[1]); end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
        out[Database_item(variable_name=p[3], surface_level=p[4], data_type=p[5])] = Byte_range (start=start, end=end)
    return out

class Gfs_downloader:
    def __init__(self):
        self.session = make_session(GFS_WORKERS)

    def gfs_url(self, downloaded_day: pd.Timestamp, lead: int, res: str):
        base = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
        for path in (f"gfs.{downloaded_day:%Y%m%d}/00/atmos", f"gfs.{downloaded_day:%Y%m%d}/00"):
            url = f"{base}/{path}/gfs.t00z.pgrb2.{res}.f{lead:03d}"
            r = self.session.get(url + ".idx", timeout=60)
            if r.ok: return url, r.text
        raise FileNotFoundError(f"GFS {downloaded_day:%Y-%m-%d} f{lead:03d} not on AWS")

    def download_message(self, url, m: Byte_range, path: str):
        hdr = {"Range": f"bytes={m.start}-{'' if m.end is None else m.end}"}
        for attempt in range(4):
            try:
                r = self.session.get(url, headers=hdr, timeout=120); r.raise_for_status()
                Path(path).write_bytes(r.content); return path
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"failed: {url} {hdr}")

def read_grib_field(grib_file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import xarray as xr
    ds = xr.open_dataset(grib_file_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]].load()
    vals, lat, lon = np.squeeze(da.values), da.latitude.values, da.longitude.values
    ds.close()
    return vals, lat, lon

def process_day_aws(day: pd.Timestamp, bboxes: dict[str, Bounding_box], res: str = GFS_RESOLUTION, workdir: str | None = None):
    # -> {region: {var: {lead: regional mean}}} for one 00 UTC run; every field is downloaded and decoded once for all regions
    workdir_path = Path(workdir or tempfile.mkdtemp())
    jobs = []

    downloader = Gfs_downloader()

    for lead in LEADS:
        url, idx_text = downloader.gfs_url(day, lead, res) # idx_text is the plain text index describing how data are organised under url
        idx: dict[Database_item, Byte_range] = parse_idx(idx_text)
        for column_name, column_info in get_gfs_wanted_fields(lead).items():
            jobs.append((lead, column_name, url, idx[column_info], workdir_path / f"{day:%Y%m%d}_{lead:03d}_{column_name}.grib2"))

    with ThreadPoolExecutor(GFS_WORKERS) as ex:
        list(ex.map(lambda j: downloader.download_message(j[2], j[3], j[4]), jobs))
    paths = {(lead, column_name): p for lead, column_name, _, _, p in jobs}
    out = {name: {v: {} for v in VARS} for name in bboxes}
    for lead in LEADS:                     # decode one lead at a time (global 0.25° fields are large)
        u, lat, lon = read_grib_field(paths[(lead, "u")]); v, _, _ = read_grib_field(paths[(lead, "v")])
        fields = {"wind": np.sqrt(u**2 + v**2)}
        for column_name in ("temp", "dpt", "dswrf", "pcp"):
            fields[column_name] = read_grib_field(paths[(lead, column_name)])[0]
        for name, bbox in bboxes.items():
            for column_name, values in fields.items():
                out[name][column_name][lead] = regional_mean(values, lat, lon, bbox)
    for p in paths.values():
        Path(p).unlink(missing_ok=True)
    return out

DOWNLOADED_DAYS: pd.DatetimeIndex = pd.date_range(START_TS.normalize(), END_TS.normalize() + pd.Timedelta(days=1), freq="D")

REPO_COLS = {
    "wind": lambda L: f"{L} hr fcst", "temp": lambda L: f"{L} hr fcst", "dpt": lambda L: f"{L} hr fcst",
    "dswrf": lambda L: f"{interval_start(L)}-{L} hr avg", "pcp": lambda L: f"{interval_start(L)}-{L} hr acc",
}
AVG_FILE = {"wind": "WIND_SPEED", "temp": "TEMP", "dpt": "DPT", "dswrf": "DSWRF", "pcp": "PCP"}
OUT_COL = {"wind": "forecast_avg_wind_speed_wMean", "temp": "forecast_avg_temperature_wMean",
           "dpt": "forecast_avg_dewpoint_wMean", "dswrf": "forecast_avg_dswrf_wMean",
           "pcp": "forecast_avg_precipitation_wMean"}

HOURS_IN_A_DAY = 24
NUM_TIMES_PREDICTED = HOURS_IN_A_DAY * 7


def load_cache(loaded_days: pd.DatetimeIndex, cache: Path):
    # -> {var: DataFrame(index=run date, columns=lead hours)}; missing runs copy the previous run
    tables = {v: pd.DataFrame(index=loaded_days, columns=LEADS, dtype=float) for v in VARS}
    for loaded_day in loaded_days:
        f = cache / f"{loaded_day:%Y%m%d}.json"
        if f.exists():
            d = json.loads(f.read_text())
            for v in VARS:
                for lead, val in d[v].items():
                    tables[v].loc[loaded_day, int(lead)] = val
    for v in VARS:
        missing_rows = tables[v].isna().all(axis=1)
        if missing_rows.any():
            print(f"{v}: {int(missing_rows.sum())} day(s) data missing -> previous days values used: {list(pd.to_datetime(tables[v].index[missing_rows]).date)[:5]}")
        tables[v] = tables[v].ffill().bfill()
    return tables

def write_avg_csvs(tables: dict[str, pd.DataFrame], region: str, folder: Path):
    # Stage 1 files, same columns as dataCollectionScript.py writes
    for column_name, table in tables.items():
        df = pd.DataFrame({REPO_COLS[column_name](L): table[L].values for L in LEADS}, index=table.index)
        df.index.name = "datetime"
        df.to_csv(folder / f"{region}_AVG_{AVG_FILE[column_name]}.csv")

def read_avg_csvs(region, folder):
    # Read stage-1 files -> tables
    tables = {}
    for v in VARS:
        df = pd.read_csv(folder / f"{region}_AVG_{AVG_FILE[v]}.csv", parse_dates=["datetime"], index_col="datetime")
        tables[v] = pd.DataFrame({L: df[REPO_COLS[v](L)].values for L in LEADS}, index=df.index)
    return tables

def build_hourly_blocks(tables: dict[str, pd.DataFrame]):
    # Stage 2: CarbonCast's src/weather/cleanWeatherData.py logic, for a 168 h horizon
    loaded_days = tables[VARS[0]].index
    lead_for_row = np.array([LEADS[(r - 1) // 3] for r in range(1, NUM_TIMES_PREDICTED)])
    out = {}
    for v in VARS:
        t = tables[v][LEADS].values.astype(float)
        col = {L: i for i, L in enumerate(LEADS)}
        blocks = np.empty((len(loaded_days), NUM_TIMES_PREDICTED))
        blocks[:, 1:] = t[:, [col[L] for L in lead_for_row]]
        # here we get the data as in the main while loop
        # of the createAvgOrAccForecastColumns function in CarbonCast's src/weather/cleanWeatherData.py
        blocks[1:, 0] = t[:-1, col[HOURS_IN_A_DAY]]
        # then we set the first hours of predictions. In the createAvgOrAccForecastColumns
        # function these are later changed in the latter loop
        blocks[0, 0] = blocks[0, 1]
        # this is backfill mirroring the last 
        out[OUT_COL[v]] = blocks.ravel()
    idx = pd.DatetimeIndex(np.concatenate([pd.date_range(r, periods=NUM_TIMES_PREDICTED, freq="h") for r in loaded_days]), name="datetime")
    return pd.DataFrame(out, index=idx)


def shift_to_model_input(agg):
    # Move each run's values onto the preceding labeled day.
    out = agg.shift(-NUM_TIMES_PREDICTED)
    out.index.name = "UTC time"
    return out.iloc[:-NUM_TIMES_PREDICTED]

@main.command()
def weather():
    if DOWNLOADED_DAYS[0] < pd.Timestamp("2021-02-26"):
        print("WARNING: AWS holds GFS 0.25° only from 2021-02-26.")
        return
    outs = {region: region_dir(region) for region in REGION_NAMES}
    caches = {region: out / "weather_data" / "gfs_cache" for region, out in outs.items()}
    for downloaded_day in DOWNLOADED_DAYS:
        # regions that still need this day
        todo = {r: REGIONS_DATA_MAP[r].bbox for r in REGION_NAMES if not (caches[r] / f"{downloaded_day:%Y%m%d}.json").exists()}
        if not todo: continue
        t0 = time.time()
        res = None
        for attempt in range(1, RUN_RETRIES + 1):
            try:
                print(f"downloading day {downloaded_day.date()} ({len(todo)} regions)")
                res = process_day_aws(downloaded_day, todo)
                break
            except Exception as ex:
                print(f"attempt {attempt}/{RUN_RETRIES} failed: {ex}")
                if attempt < RUN_RETRIES:
                    time.sleep(5 * attempt)
        if res is None:
            print(f"FAILED: {downloaded_day.date()}")
            continue
        for region, data in res.items():
            (caches[region] / f"{downloaded_day:%Y%m%d}.json").write_text(json.dumps(data))
        print(f"done in {time.time() - t0:.0f}s")
    for region in REGION_NAMES:
        tables: dict[str, pd.DataFrame] = load_cache(DOWNLOADED_DAYS, caches[region])
        write_avg_csvs(tables, region, outs[region] / "weather_data")
        agg = build_hourly_blocks(tables)
        agg.to_csv(outs[region] / "weather_data" / f"{region}_aggregated_weather_data.csv")
        wfc = shift_to_model_input(agg)
        wfc.to_csv(outs[region] / f"{region}_weather_forecast.csv")

main()
