The `CarbonCast` directory is a git submodule of the original CarbonCast project. The baseline script reads its data from it.

## Setup

You need git and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/mborgos/CFE-forecast.git
cd CFE-forecast
git submodule update --init          # downloads the CarbonCast data: about 190 MB, 0.9 GB on disk
uv sync                              # creates .venv with Python 3.11 and the locked package versions
uv run python cfe_7day_baseline.py   # about 2 minutes
```

- **Data:** fetch it with `git submodule update --init` as above. Avoid `git clone --recurse-submodules`: it downloads every branch of CarbonCast, which is far larger.
- **macOS:** XGBoost needs the OpenMP runtime. Install it once with `brew install libomp`.
- **VS Code:** select the interpreter in `.venv` to run the script from the editor.
- **Intel Macs:** TensorFlow has no build for them, so `uv sync` fails there.

## `cfe_7day_baseline.py`

Sets the bar for 7-day forecasts of the carbon-free energy share (%CFE): two naive baselines and an XGBoost model trained on historical lags only, scored for all 13 regions. It prints its tables to the terminal. The parameters are constants at the top of the file.

- **Output** in `results/cfe_7day_baseline/`: `scores_by_day.csv` (MAE, RMSE and bias for each region, model and forecast day) and three figures.
- **Scoring another model:** import the data loading, the forecast windows and the scoring from this file:

```python
from cfe_7day_baseline import load_cfe, make_windows, score

starts, history, actual = make_windows(load_cfe("DE"))
print(score(actual, my_forecast))   # my_forecast: one row per forecast, 168 columns
```

![MAE over the 7-day window, all regions](results/cfe_7day_baseline/all_regions_mae.png)
