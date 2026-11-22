"""Download NOAA GFS 0.25-degree forecasts from AWS."""
from __future__ import annotations
import argparse
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BBox = Tuple[float, float, float, float]   # (north lat, south lat, west lon, east lon)
Spec = Tuple[str, str, str]                # (GRIB variable, level, forecast text exactly as written in the .idx file)
RunData = Dict[str, Dict[str, float]]      # variable -> lead (as text, like the JSON cache) -> regional mean

class IdxEntry(NamedTuple):
    start: int
    end: Optional[int]

class Region(NamedTuple):
    rows: np.ndarray                       # latitude indices inside the box
    cols: np.ndarray                       # longitude indices inside the box
    weights: np.ndarray                    # grid-cell areas for those rows and columns

def _project_root() -> Path:
    parents = Path(__file__).resolve().parents
    return parents[2] if len(parents) > 2 else Path.cwd()

def interval_start(lead: int) -> int:
    """Start of GFS's alternating 3-/6-hour radiation and precipitation buckets."""
    return lead - 3 if (lead // 3) % 2 == 1 else lead - 6

DEFAULT_OUT = _project_root() / "data"
AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
RUN_RETRIES = 3
HORIZON = 168
LEADS = list(range(3, HORIZON + 1, 3))
VARS = ["wind", "temp", "dpt", "dswrf", "pcp"]
AVG_FILE = {"wind": "WIND_SPEED", "temp": "TEMP", "dpt": "DPT", "dswrf": "DSWRF", "pcp": "PCP"}
OUT_COL = {
    "wind": "forecast_avg_wind_speed_wMean",
    "temp": "forecast_avg_temperature_wMean",
    "dpt": "forecast_avg_dewpoint_wMean",
    "dswrf": "forecast_avg_dswrf_wMean",
    "pcp": "forecast_avg_precipitation_wMean",
}
REGIONS = {
    "CISO": (42, 32, -124.75, -113.5),
    "PJM": (43, 34.25, -91, -73.5),
    "ERCO": (36.5, 25.25, -104.5, -93.25),
    "ISNE": (48, 40, -74.25, -66.5),
    "NYISO": (45.5, 40, -80.25, -71.25),
    "FPL": (31.25, 24, -83.5, -79.5),
    "BPAT": (49.5, 39.5, -125.25, -105.5),
    "MISO": (50, 28.5, -107.75, -81.75),
    "SWPP": (49.5, 30.25, -107.75, -89.5),
    "SOCO": (35.5, 29.25, -90.5, -80.25),
    "DE": (55.25, 47.25, 5.75, 15),
    "NL": (53.5, 50.75, 3.25, 7),
    "ES": (43.75, 36, -9.25, 3.5),
    "SE": (69, 55.25, 11.25, 21.25),
    "PL": (54.75, 49, 14, 24),
    "FR": (51.25, 42.25, -5.25, 8.25),
    "BE": (51.5, 49.5, 2.5, 6.25),
    "FI": (70, 59.75, 20.5, 31.5),
    "DK-DK2": (57.75, 54.75, 7.25, 11.25),
    "AUS_QLD": (-8.75, -29.75, 137.5, 154),
    "AUS_NSW": (-34.75, -36.5, 148.25, 150),
    "AUS_SA": (-25.5, -38.5, 128.5, 141.5),
}

def column_name(var: str, lead: int):
    if var == "dswrf":
        return f"{interval_start(lead)}-{lead} hr avg"
    if var == "pcp":
        return f"{interval_start(lead)}-{lead} hr acc"
    return f"{lead} hr fcst"

def earth_radius(lat: np.ndarray) -> np.ndarray:
    """Radius of Earth assuming oblate spheroid defined by WGS84"""
    a, b = 6378137.0, 6356752.3142
    e2 = 1 - b**2 / a**2
    lat_gc = np.arctan((1 - e2) * np.tan(np.deg2rad(lat)))
    return (a * np.sqrt(1 - e2)) / np.sqrt(1 - e2 * np.cos(lat_gc)**2)

def area_grid(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculate the area of each grid cell. Area is in square meters
    Based on the function in 
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    xlon, ylat = np.meshgrid(lon, lat)
    r = earth_radius(ylat)
    dy = np.deg2rad(np.gradient(ylat, axis=0)) * r
    dx = np.deg2rad(np.gradient(xlon, axis=1)) * r * np.cos(np.deg2rad(ylat))
    return dy * dx

def make_region(lat: np.ndarray, lon: np.ndarray, bbox: BBox) -> Region:
    north, south, west, east = bbox
    lon = ((lon + 180) % 360) - 180 # GFS grids use 0..360
    order = np.argsort(lon)
    rows = np.flatnonzero((lat >= south) & (lat <= north))
    cols = order[(lon[order] >= west) & (lon[order] <= east)]
    if rows.size == 0 or cols.size == 0:
        raise ValueError(f"Bounding box {bbox} contains no grid points")
    return Region(rows, cols, np.abs(area_grid(lat[rows], lon[cols])))

def region_mean(values: np.ndarray, region: Region) -> float:
    """Area-weighted mean of a global lat/lon field over the region."""
    sub = values[np.ix_(region.rows, region.cols)]
    valid = np.isfinite(sub)
    return float(np.average(sub[valid], weights=region.weights[valid]))

def wanted_fields(lead: int) -> Dict[str, Spec]:
    return {
        "u": ("UGRD", "10 m above ground", f"{lead} hour fcst"),
        "v": ("VGRD", "10 m above ground", f"{lead} hour fcst"),
        "temp": ("TMP", "2 m above ground", f"{lead} hour fcst"),
        "dpt": ("DPT", "2 m above ground", f"{lead} hour fcst"),
        "dswrf": ("DSWRF", "surface", f"{interval_start(lead)}-{lead} hour ave fcst"),
        "pcp": ("APCP", "surface", f"{interval_start(lead)}-{lead} hour acc fcst"),
    }

def parse_idx(text: str) -> Dict[Spec, IdxEntry]:
    rows = [line.split(":") for line in text.splitlines() if line.strip()]
    starts = [int(row[1]) for row in rows]
    ends = [s - 1 for s in starts[1:]] + [None]
    return {(row[3], row[4], row[5]): IdxEntry(start, end) for row, start, end in zip(rows, starts, ends)}

def make_session(workers: int) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=workers, pool_maxsize=workers))
    session.headers.update({"User-Agent": "CFE-forecast"})
    return session

def request(session: requests.Session, url: str, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    for attempt in range(3):
        try:
            response = session.get(url, headers=headers, timeout=(15, 45))
            response.raise_for_status()
            return response
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise Exception("Should be unreachable")

def find_layout(session: requests.Session, run: pd.Timestamp, res: str) -> str:
    for layout in (f"gfs.{run:%Y%m%d}/00/atmos", f"gfs.{run:%Y%m%d}/00"):
        try:
            request(session, f"{AWS}/{layout}/gfs.t00z.pgrb2.{res}.f003.idx")
            return layout
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
    raise FileNotFoundError(f"GFS 00z run unavailable on AWS for {run:%Y-%m-%d}")

def download_range(session: requests.Session, url: str, entry: IdxEntry, target: Path) -> Path:
    end = "" if entry.end is None else entry.end
    response = request(session, url, headers={"Range": f"bytes={entry.start}-{end}"})
    if response.status_code != 206:
        raise IOError(f"Server did not honour byte-range request (HTTP {response.status_code})")
    if entry.end is not None and len(response.content) != entry.end - entry.start + 1:
        raise IOError(f"Truncated range: got {len(response.content)} of {entry.end - entry.start + 1} bytes")
    target.write_bytes(response.content)
    return target

def read_field(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
        data = ds[list(ds.data_vars)[0]].load()
        return np.squeeze(data.values), data.latitude.values, data.longitude.values

def process_run(run: pd.Timestamp, bboxes: Dict[str, BBox], res: str, workers: int) -> Dict[str, RunData]:
    with make_session(workers) as session, tempfile.TemporaryDirectory(prefix="gfs_run_") as tmp:
        base = f"{AWS}/{find_layout(session, run, res)}/gfs.t00z.pgrb2.{res}"

        def fetch_lead(lead: int) -> Dict[str, Path]:
            url = f"{base}.f{lead:03d}"
            inventory = parse_idx(request(session, url + ".idx").text)
            return {key: download_range(session, url, inventory[spec], Path(tmp) / f"{lead:03d}_{key}.grib2")
                    for key, spec in wanted_fields(lead).items()}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            files = dict(zip(LEADS, pool.map(fetch_lead, LEADS)))

        output = {name: {var: {} for var in VARS} for name in bboxes}
        regions = {}
        for lead, paths in files.items():
            u, lat, lon = read_field(paths["u"])
            if not regions: 
                regions = {name: make_region(lat, lon, bbox) for name, bbox in bboxes.items()}
            v, _, _ = read_field(paths["v"])
            fields = {"wind": np.hypot(u, v)}
            for var in ("temp", "dpt", "dswrf", "pcp"):
                fields[var] = read_field(paths[var])[0]
            for name, region in regions.items():
                for var, values in fields.items():
                    output[name][var][str(lead)] = region_mean(values, region)
    return output

def read_cache(file: Path) -> Optional[RunData]:
    try:
        data = json.loads(file.read_text())
        if all(str(lead) in data[var] for var in VARS for lead in LEADS):
            return data
    except (OSError, ValueError, KeyError):
        pass
    return None

def download_runs(runs: pd.DatetimeIndex, bboxes: Dict[str, BBox], res: str, workers: int, caches: Dict[str, Path]) -> None:
    failed = []
    for run in runs:
        cache_name = f"{run:%Y%m%d}.json"
        todo = {name: bbox for name, bbox in bboxes.items() if read_cache(caches[name] / cache_name) is None}
        if not todo:
            print(f"{run:%Y-%m-%d} cached", flush=True)
            continue
        started = time.time()
        data = None
        for attempt in range(1, RUN_RETRIES + 1):
            try:
                data = process_run(run, todo, res, workers)
                break
            except Exception as exc:
                print(f"{run:%Y-%m-%d} attempt {attempt}/{RUN_RETRIES} failed: {exc}", flush=True)
                if attempt < RUN_RETRIES:
                    time.sleep(5 * attempt)
        if data is None:
            failed.append(f"{run:%Y-%m-%d}")
            continue
        for name, region_data in data.items():
            (caches[name] / cache_name).write_text(json.dumps(region_data))
        print(f"{run:%Y-%m-%d} done in {time.time() - started:.0f}s ({len(todo)} regions)", flush=True)
    if failed:
        raise SystemExit(
            "Some GFS runs could not be downloaded: " + ", ".join(failed)
            + "\nNo final CSVs were generated. Run the same command again to retry. Completed dates are cached."
        )

def load_cache(cache: Path, runs: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
    data = {run: read_cache(cache / f"{run:%Y%m%d}.json") for run in runs}
    missing = [f"{run:%Y-%m-%d}" for run, d in data.items() if d is None]
    if missing:
        raise SystemExit(f"Cannot build output: missing/incomplete cached runs in {cache}: " + ", ".join(missing)
                         + "\nRetry the download (do not use --no-download).")
    return {var: pd.DataFrame([[data[run][var][str(lead)] for lead in LEADS] for run in runs], index=runs, columns=LEADS, dtype=float) for var in VARS}

def build_hourly_blocks(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    runs = tables[VARS[0]].index
    hour_to_col = (np.arange(1, HORIZON) - 1) // 3 # hours 1-3 -> lead 3, hours 4-6 -> lead 6, ...
    col_24h = LEADS.index(24)
    columns = {}
    for var in VARS:
        matrix = tables[var][LEADS].values.astype(float)
        blocks = np.empty((len(runs), HORIZON))
        blocks[:, 1:] = matrix[:, hour_to_col]
        blocks[1:, 0] = matrix[:-1, col_24h] # hour 0 = previous run's 24 h forecast
        blocks[0, 0] = blocks[0, 1]
        columns[OUT_COL[var]] = blocks.ravel()
    index = pd.DatetimeIndex(np.concatenate([pd.date_range(run, periods=HORIZON, freq="h") for run in runs]), name="datetime")
    return pd.DataFrame(columns, index=index)

def save_outputs(region: str, runs: pd.DatetimeIndex, out: Path, cache: Path) -> None:
    tables = load_cache(cache, runs)
    weather_dir = out / "weather_data"
    for var, table in tables.items():
        frame = pd.DataFrame({column_name(var, lead): table[lead].values for lead in LEADS}, index=table.index)
        frame.index.name = "datetime"
        frame.to_csv(weather_dir / f"{region}_AVG_{AVG_FILE[var]}.csv")
    aggregated = build_hourly_blocks(tables)
    aggregated.to_csv(weather_dir / f"{region}_aggregated_weather_data.csv")

    # Block labelled day d contains the following day's run (last block dropped)
    n_cols = aggregated.shape[1]
    values = aggregated.values.reshape(-1, HORIZON, n_cols)[1:].reshape(-1, n_cols)
    forecast = pd.DataFrame(values, index=aggregated.index[:-HORIZON].rename("UTC time"), columns=aggregated.columns)
    forecast.to_csv(out / f"{region}_weather_forecast.csv")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regions", nargs="+", default=list(REGIONS), choices=REGIONS, metavar="REGION", 
                        help=f"Regions (default: all). Choose from: {', '.join(REGIONS)}")
    parser.add_argument("--start", default="2026-01-01", help="first date, UTC inclusive")
    parser.add_argument("--end", default="2026-01-31", help="end date, UTC exclusive")
    parser.add_argument("--res", default="0p25", choices=["0p25", "0p50", "1p00"], help="GFS resolution")
    parser.add_argument("--workers", type=int, default=8, help="parallel GRIB range downloads per run")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output root directory")
    parser.add_argument("--no-download", action="store_true", help="build CSVs from existing cache only")
    args = parser.parse_args()
    if pd.Timestamp(args.end) <= pd.Timestamp(args.start):
        parser.error("--end must be later than --start (exclusive)")
    return args

def main():
    args = parse_args()
    bboxes = {name: REGIONS[name] for name in args.regions}
    runs = pd.date_range(pd.Timestamp(args.start).normalize(), pd.Timestamp(args.end).normalize() + pd.Timedelta(days=1), freq="D")
    outs = {name: Path(args.out) / name for name in bboxes}
    caches = {name: out / "weather_data" / f"cache_gfs_{args.res}" for name, out in outs.items()}
    for cache in caches.values():
        cache.mkdir(parents=True, exist_ok=True)
    print(f"GFS {len(bboxes)} regions ({', '.join(bboxes)}): {len(runs)} daily runs, "
          f"{runs[0]:%Y-%m-%d} through {runs[-1]:%Y-%m-%d}, resolution {args.res}, horizon {HORIZON}h")
    if not args.no_download:
        download_runs(runs, bboxes, args.res, args.workers, caches)
    print("Writing files.")
    for name in bboxes:
        save_outputs(name, runs, outs[name], caches[name])
    print("CSVs building and saving complete.")

if __name__ == "__main__":
    main()
