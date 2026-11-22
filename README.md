## GFS weather download

`src/weather/gfs_forecast.py` downloads NOAA GFS 0.25° forecasts from the public AWS bucket (`noaa-gfs-bdp-pds`, no account needed).

### Install

```bash
conda create -n cfe_forecast python=3.11
conda activate cfe_forecast
conda install -c conda-forge cfgrib eccodes xarray numpy pandas requests

# or, with pip only (needs a working eccodes binary)
pip install -r requirements.txt
```

### Usage
To download the weather data for all regions run (set `--start` and `--end` to the dates you need)
```bash
python src/weather/gfs_forecast.py --start 2026-10-01 --end 2026-10-15
```

### Notes
- Uses the **00 UTC run** of every day, lead times **3, 6, ..., 168 h** (3-hourly).
- Variables per region (area-weighted mean over the region's bounding box):
  wind speed (10 m), temperature (2 m), dew point (2 m), downward shortwave radiation, precipitation.
- Finished downloads are cached as JSON. If the download is interrupted or some runs fail, run the same
command again. No CSVs are written until every needed run is available (missing runs are never filled in with older ones).
- The data folder contains the weather forecast data for each region, downloaded for 2026-01-01 to 2026-01-31.
