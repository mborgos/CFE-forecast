The `CarbonCast` directory contains a fork of the original CarbonCast project, along with three additional helper notebooks.

The notebooks provide the following functionality:

- **`CarbonCast_walkthrough.ipynb`** – Demonstrates how to use the second tier of the CarbonCast architecture with the available data.
- **`CarbonCast_download_data.ipynb`** – Provides a straightforward workflow for downloading and refreshing the datasets required by the project.
- **`CFE_LightGBM_baseline.ipynb`** – Uses the available data to train a baseline LightGBM model based exclusively on historical lag features. Its performance is evaluated using several relevant metrics and compared against simple heuristic forecasting approaches.

To download this repo, download recursively using ``git clone --recurse-submodules``