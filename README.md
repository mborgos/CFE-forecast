# Probabilistic %CFE forecasting
## Install

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```
On macOS, XGBoost also needs the OpenMP runtime: `brew install libomp`.

## Data collection

`data_collection.py` downloads, for one or many regions:

- **electricity**: hourly generation per source from EIA (US), ENTSO-E (Europe) and
  Open Electricity (Australia), plus carbon intensity and emissions files;
- **weather**: NOAA GFS 0.25° forecasts from the public AWS bucket (no account needed),
  averaged over each region's bounding box, with a **168 h** horizon.

CarbonCast (https://github.com/carbonfirst/CarbonCast) was used as a reference while writing this code, and parts of it are adapted from CarbonCast. CarbonCast itself is not needed to run it. All times are UTC.

### Configure

To download the data create a `config.env` file next to the script (or set the same names as environment variables):

```
EIA_API_KEY=...                 # US regions (https://www.eia.gov/opendata/)
ENTSOE_API_KEY=...              # European regions (ENTSO-E Transparency Platform)
OPENELECTRICITY_API_KEY=...     # Australian regions (https://platform.openelectricity.org.au)
DOWNLOADED_REGIONS=CISO,DE,PL   # comma separated; empty = all 22 regions
START=2026-01-01                # first day
END=2026-01-31                  # last day
OUT_DIR=data                    # default: data (relative to where you run the script)
GFS_RESOLUTION=0p25             # 0p25, 0p50 or 1p00
```

Weather needs no key, while electricity needs only the key of the providers your regions use.

Regions available: CISO, PJM, ERCO, ISNE, NYISO, FPL, BPAT, MISO, SWPP, SOCO, DE, NL, ES, SE, PL, FR, BE, FI, DK-DK2,
AUS_QLD, AUS_NSW, AUS_SA.

Created `config.env` is for the data downloading purpose only.

### Run

```bash
python data_collection.py electricity
python data_collection.py weather
```

Failed requests are retried, and finished days are cached, so after an interruption run the same command.

## 7-day baseline

`cfe_7day_baseline.py` sets the bar for 7-day forecasts of the carbon-free energy share (%CFE). It reads the `<REGION>_lifecycle_emissions.csv` files that `data_collection.py electricity` writes to `data/`, and scores every region it finds. It does not use the weather.

Models, all trained on historical lags only:
 
- **last day** and **7-day mean**: naive baselines that repeat yesterday, or the average of the last 7 days, hour by hour;
- **XGBoost**: predicts the correction to the 7-day mean, with one model for day 1 and one for days 2-7;
- **LightGBM**: predicts %CFE directly, with one model for all 7 days;
- **LightGBM quantile**: predicts the 10th, 50th and 90th percentiles of %CFE directly. The median is its point forecast, and the 10th-90th range is an 80% interval.


The last 6 months of the data are the test period and the 6 months before are for validation (early stopping), so each region needs well over a year of data.

### Run
```bash
python cfe_7day_baseline.py
```

**Settings.** Command-line options:
- `--regions CISO DE ...`: regions to score (default: every region in `data/`).
- `--start 2023-01-01`, `--end 2025-06-30`: the period of data to use (default: all available).
- `--models xgboost lightgbm lightgbm-quantile`: the learned models to run (default: all three). 
- `--save-forecasts`: also save every forecast to `forecasts/<REGION>_forecasts.csv`.

Everything else (forecast horizon, test and validation months, model parameters, quantiles) is set by the constants at the top of `cfe_7day_baseline.py`

**Output:** tables in the terminal. In `results/cfe_7day_baseline/`:
- `scores_by_day.csv`: MAE, RMSE and bias for each region, model and forecast day;
- `forecasts/<REGION>_forecasts.csv`, only with `--save-forecasts`: every test forecast, one row per forecast start and hour ahead, with the actual %CFE and one column per model (plus the LightGBM 10th and 90th percentiles);
- `quantile_scores_by_day.csv`: for LightGBM quantile, the pinball loss of each quantile and the coverage and width of the 80% interval;
- three figures: an example forecast and the error by hour ahead for one region (`REGION` at the top of the file, if it is among those scored), and the error of every region.


