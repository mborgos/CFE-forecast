The `CarbonCast` directory is a git submodule of the original CarbonCast project. Both notebooks read their data from it.

## Setup

You need git and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/mborgos/CFE-forecast.git
cd CFE-forecast
git submodule update --init   # downloads the CarbonCast data: about 190 MB, 0.9 GB on disk
uv sync                       # creates .venv with Python 3.11 and the locked package versions
uv run jupyter lab            # open a notebook and run all cells
```

- **Data:** fetch it with `git submodule update --init` as above. Avoid `git clone --recurse-submodules`: it downloads every branch of CarbonCast, which is far larger.
- **macOS:** XGBoost needs the OpenMP runtime. Install it once with `brew install libomp`.
- **VS Code:** open a notebook and select the interpreter in `.venv` as its kernel.
- **Intel Macs:** TensorFlow has no build for them, so `uv sync` fails there.

## Notebooks

- **`CarbonCast_CISO_paper_reproduction.ipynb`** – Reproduces the CISO results of the CarbonCast paper. It trains both tiers from scratch and compares them with the paper and with the day-ahead baseline. Trained forecasts are cached in `results/carboncast_ciso/`, so it re-runs in seconds.
- **`CFE_7day_baseline.ipynb`** – Sets the bar for 7-day forecasts of the carbon-free energy share (%CFE): two naive baselines and an XGBoost model trained on historical lags only, scored for all 13 regions. Scores are saved in `results/cfe_7day_baseline/`.
