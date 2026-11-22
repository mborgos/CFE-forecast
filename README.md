## Data collection for %CFE forecasting

`data_collection.py` downloads, for one or many regions:

- **electricity**: hourly generation per source from EIA (US), ENTSO-E (Europe) and
  Open Electricity (Australia), plus carbon intensity and emissions files;
- **weather**: NOAA GFS 0.25° forecasts from the public AWS bucket (no account needed),
  averaged over each region's bounding box, with a **168 h** horizon.

CarbonCast (https://github.com/carbonfirst/CarbonCast) was used as a reference while writing this code, and parts of it are adapted from CarbonCast. CarbonCast itself is not needed to run it. All times are UTC.

## Install

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Configure

Create a `config.env` file next to the script (or set the same names as environment variables):

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

## Run

```bash
python data_collection.py electricity
python data_collection.py weather
```

Failed requests are retried, and finished days are cached, so after an interruption run the same command.
