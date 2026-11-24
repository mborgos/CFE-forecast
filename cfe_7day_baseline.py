from pathlib import Path
import argparse
import time

from collections.abc import Sequence
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb


# Region for which more detailed tables and figures are saved. 
# It is ignored if it is not in the list of regions selected for evaluation (look --regions parameter)
SPECIAL_REGION = "CISO"             

INCLUDE_BIOMASS_AS_CFE = False      # definitions differ; keep it explicit
CFE_SOURCES = ["nuclear", "hydro", "solar", "wind", "geothermal"] + (["biomass"] if INCLUDE_BIOMASS_AS_CFE else [])

HORIZON_DAYS = 7                    # forecast window in days
START_HOUR = 0                      # UTC hour at which each forecast window starts (one forecast per day)
TEST_MONTHS = 6                     # the last months of the data are the test period
VAL_MONTHS = 6                      # the months before the test period: XGBoost picks its number of trees on them

XGB_PARAMS = dict(learning_rate=0.05, max_depth=6, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", random_state=0, n_jobs=-1)
LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, max_depth=6, min_child_samples=20, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)      # matched to XGB_PARAMS
QUANTILES = [0.1, 0.5, 0.9]         # LightGBM quantile: the median is its point forecast, 0.1-0.9 its 80% interval
MAX_TREES, PATIENCE = 3000, 50      # early stopping

HORIZON_HOURS = 24 * HORIZON_DAYS # hours to predict
HISTORY_HOURS = 24 * 7 # hours of history each forecast is given
DAY_LABELS = [f"day {k}" for k in range(1, HORIZON_DAYS + 1)] + ["all"]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"            # where data_collection.py writes
OUT_DIR = ROOT / "results" / "cfe_7day_baseline"


# --- The %CFE series --------------------------------------------------------------------------

def load_cfe(region: str, cfe_sources: list[str] = CFE_SOURCES, start: str | None = None, end: str | None = None):
    """Hourly %CFE of a region, from the <region>_lifecycle_emissions.csv that data_collection.py writes:
    every hour present, no gaps and no negative generation. An hour with no generation at all takes the
    previous hour's value. start and end (dates such as "2024-01-01", end included) keep only that period."""
    df = pd.read_csv(DATA_DIR / region / f"{region}_lifecycle_emissions.csv", index_col=0, parse_dates=["UTC time"])
    gen = df.set_index("UTC time").drop(columns="carbon_intensity")
    total = gen.sum(axis=1).replace(0, np.nan)
    cfe = gen[[c for c in gen.columns if c in cfe_sources]].sum(axis=1) / total * 100
    return cfe.ffill().rename("cfe").loc[start:end]


# --- Train, validation and test periods -------------------------------------------------------

def split_dates(cfe: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(val_start, test_start, test_end) of a series. The test period is its last TEST_MONTHS months,
    validation the VAL_MONTHS months before that, and training everything earlier. Test forecasts
    start from test_start, and every target hour is before test_end."""
    test_end = (cfe.index[-1] + pd.Timedelta(hours=1)).floor("D")
    test_start = test_end - pd.DateOffset(months=TEST_MONTHS)
    val_start = test_start - pd.DateOffset(months=VAL_MONTHS)
    return val_start, test_start, test_end


# --- Forecast windows -------------------------------------------------------------------------

def make_windows(cfe: pd.Series, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, 
                 start_hour: int | None = START_HOUR) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """One row per forecast: its start time, the 168 hours before it (history) and the 168 hours to
    predict (actual). Every forecast is built from history alone. Without start and end, the windows
    of the test period. Scoring uses forecasts starting at START_HOUR; start_hour=None returns a
    window for every hour, which XGBoost and LightGBM use for training."""
    _, test_start, test_end = split_dates(cfe)
    start = test_start if start is None else start
    end = test_end if end is None else end
    y, idx = cfe.to_numpy(), cfe.index
    allowed_forecast_start = (idx >= start) & (idx + pd.Timedelta(hours=HORIZON_HOURS) <= end)
    if start_hour is not None:
        allowed_forecast_start &= idx.hour == start_hour
    candidate_rows = np.flatnonzero(allowed_forecast_start)
    start_rows = candidate_rows[(candidate_rows >= HISTORY_HOURS) & (candidate_rows + HORIZON_HOURS <= len(y))]
    history = y[start_rows[:, None] + np.arange(-HISTORY_HOURS, 0)]
    actual = y[start_rows[:, None] + np.arange(HORIZON_HOURS)]
    return idx[start_rows], history, actual


# --- The two naive baselines ------------------------------------------------------------------
# For a target hour t on forecast day k:
#   last day:    forecast(t) = y(t - 24k)
#   7-day mean:  forecast(t) = mean of y(t - 24(k + d)) for d = 0..6

def last_day(history: np.ndarray) -> np.ndarray:
    return np.tile(history[:, -24:], HORIZON_DAYS)

def mean_7d(history: np.ndarray) -> np.ndarray:
    profile = history.reshape(len(history), 7, 24).mean(axis=1)
    return np.tile(profile, HORIZON_DAYS)

NAIVE = {"last day": last_day, "7-day mean": mean_7d}


# --- XGBoost on lags --------------------------------------------------------------------------
# - What it predicts: the correction to the 7-day mean, y(t) - 7-day mean. The forecast is the 7-day
#   mean plus that correction, clipped to 0-100, so with nothing to learn the model falls back to
#   the baseline.
# - Two models: one for day 1 (hours 1-24) and one for days 2-7, each with the hours ahead as a feature.
# - Training: on forecasts starting at every hour of the day, not only START_HOUR. The number of
#   trees is chosen by early stopping: train on windows whose targets end before the validation
#   period, stop when the error on the windows of the validation period stops improving. The model
#   is then refitted on everything before the test period with that number of trees.
# - Features (46):
#     last day against the weekly profile (24)   each of the last 24 hours minus the 7-day mean of that hour
#     level and spread (9)                       mean of each of the last 7 days; standard deviation of the
#                                                last 24 hours and of the last 7 days
#     target hour (8)                            %CFE at the same hour on each of the last 7 days, and their
#                                                mean (the 7-day-mean forecast)
#     calendar (5)                               hours ahead; hour of day, day of week and day of year
#                                                (sin, cos) of the target
#

BLOCKS = {"day 1": (1, 24), f"days 2-{HORIZON_DAYS}": (25, HORIZON_HOURS)}      # hours ahead covered by each model

def make_features(starts: pd.DatetimeIndex, history: np.ndarray) -> np.ndarray:
    # One row per (forecast, hour ahead), forecasts in order, hours 1..HORIZON_HOURS within each forecast.
    n = len(starts)
    days = history.reshape(n, 7, 24)                                # [forecast, day (oldest first), hour]
    profile = days.mean(axis=1)                                     # 7-day mean by hour
    per_forecast = np.hstack([
        history[:, -24:] - profile,                                 # last day against the weekly profile
        days.mean(axis=2),                                          # mean of each of the last 7 days
        history[:, -24:].std(axis=1, keepdims=True), history.std(axis=1, keepdims=True)])
    same_hour = days[:, ::-1][:, :, np.arange(HORIZON_HOURS) % 24].transpose(0, 2, 1)     # (n, HORIZON_HOURS, 7): 1, 2, ... 7 days before day 1
    target = pd.DatetimeIndex((starts.values[:, None] + np.arange(HORIZON_HOURS).astype("timedelta64[h]")).ravel())
    angle = 2 * np.pi * target.day_of_year / 365.25
    per_hour = np.column_stack([
        np.tile(np.arange(1, HORIZON_HOURS + 1), n), same_hour.reshape(n * HORIZON_HOURS, 7), mean_7d(history).ravel(),
        target.hour, target.day_of_week, np.sin(angle), np.cos(angle)])
    return np.hstack([np.repeat(per_forecast, HORIZON_HOURS, axis=0), per_hour]).astype(np.float32)

def block_rows(block: str, n_forecasts: int) -> np.ndarray:
    lo, hi = BLOCKS[block]
    hours_ahead = np.arange(1, HORIZON_HOURS + 1)
    return np.tile((hours_ahead >= lo) & (hours_ahead <= hi), n_forecasts)

def fit_xgb(cfe: pd.Series) -> dict[str, xgb.XGBRegressor]:
    val_start, test_start, _ = split_dates(cfe)
    starts, history, actual = make_windows(cfe, cfe.index[0], test_start, start_hour=None)
    X, y = make_features(starts, history), (actual - mean_7d(history)).ravel()
    train = np.repeat(starts + pd.Timedelta(hours=HORIZON_HOURS) <= val_start, HORIZON_HOURS)
    val = np.repeat(starts >= val_start, HORIZON_HOURS)
    models = {}
    for block in BLOCKS:
        rows = block_rows(block, len(starts))
        stopper = xgb.XGBRegressor(n_estimators=MAX_TREES, early_stopping_rounds=PATIENCE, eval_metric="mae", **XGB_PARAMS)
        stopper.fit(X[rows & train], y[rows & train], eval_set=[(X[rows & val], y[rows & val])], verbose=False)
        models[block] = xgb.XGBRegressor(n_estimators=stopper.best_iteration + 1, **XGB_PARAMS).fit(X[rows], y[rows])
    return models

def xgb_forecast(models: dict[str, xgb.XGBRegressor], starts: pd.DatetimeIndex, history: np.ndarray) -> np.ndarray:
    X = make_features(starts, history)
    correction = np.empty(len(X))
    for block, model in models.items():
        rows = block_rows(block, len(starts))
        correction[rows] = model.predict(X[rows])
    return np.clip(mean_7d(history) + correction.reshape(len(starts), HORIZON_HOURS), 0, 100)


# --- LightGBM on lags -------------------------------------------------------------------------
# Predicts %CFE directly, with one model for all hours ahead (hours ahead is one of its features).
# Same features and training scheme as XGBoost: early stopping on the validation period, then a
# refit on everything before the test period. Forecasts are clipped to 0-100.
# - LightGBM: squared error, early stopping on validation MAE.
# - LightGBM quantile: one model per quantile in QUANTILES, trained with the pinball loss and
#   early-stopped on it. Its point forecast is the median.

def fit_lgb(cfe: pd.Series, alpha: float | None = None) -> lgb.LGBMRegressor:
    """One model for all hours ahead. alpha=None: squared error; otherwise the alpha quantile."""
    val_start, test_start, _ = split_dates(cfe)
    starts, history, actual = make_windows(cfe, cfe.index[0], test_start, start_hour=None)
    X, y = make_features(starts, history), actual.ravel()
    train = np.repeat(starts + pd.Timedelta(hours=HORIZON_HOURS) <= val_start, HORIZON_HOURS)
    val = np.repeat(starts >= val_start, HORIZON_HOURS)
    loss = dict(objective="regression") if alpha is None else dict(objective="quantile", alpha=alpha)
    metric = "l1" if alpha is None else "quantile"
    stopper = lgb.LGBMRegressor(n_estimators=MAX_TREES, **loss, **LGB_PARAMS)
    stopper.fit(X[train], y[train], eval_X=(X[val],), eval_y=(y[val],), eval_metric=metric, callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return lgb.LGBMRegressor(n_estimators=max(stopper.best_iteration_, 1), **loss, **LGB_PARAMS).fit(X, y)

def lgb_forecast(model: lgb.LGBMRegressor, starts: pd.DatetimeIndex, history: np.ndarray) -> np.ndarray:
    return np.clip(model.predict(make_features(starts, history)).reshape(len(starts), HORIZON_HOURS), 0, 100)

def fit_lgb_quantiles(cfe: pd.Series) -> dict[float, lgb.LGBMRegressor]:
    return {q: fit_lgb(cfe, alpha=q) for q in QUANTILES}

def lgb_quantile_forecast(models_by_q: dict[float, lgb.LGBMRegressor], starts: pd.DatetimeIndex, history: np.ndarray) -> np.ndarray:
    """(forecast, hour ahead, quantile) array. Sorting each row keeps the quantiles from crossing."""
    return np.sort(np.stack([lgb_forecast(m, starts, history) for m in models_by_q.values()], axis=-1), axis=-1)


# --- Scores -----------------------------------------------------------------------------------

def score(actual: np.ndarray, forecast: np.ndarray) -> pd.DataFrame:
    """MAE, RMSE and bias of each forecast day and of the whole window, in percentage points."""
    errors = (forecast - actual).reshape(len(actual), HORIZON_DAYS, 24)
    per_day = lambda f: [f(errors[:, k]) for k in range(HORIZON_DAYS)] + [f(errors)]
    return pd.DataFrame({"MAE": per_day(lambda x: np.abs(x).mean()),
                         "RMSE": per_day(lambda x: np.sqrt((x ** 2).mean())),
                         "bias": per_day(np.mean)}, index=DAY_LABELS).T

def score_quantiles(actual: np.ndarray, quantiles: np.ndarray) -> pd.DataFrame:
    """Pinball loss of each quantile, and coverage (%) and width (pp) of the lowest-to-highest
    interval, for each forecast day and the whole window."""
    a = actual.reshape(len(actual), HORIZON_DAYS, 24)
    q = quantiles.reshape(len(actual), HORIZON_DAYS, 24, len(QUANTILES))
    per_day = lambda x: [x[:, k].mean() for k in range(HORIZON_DAYS)] + [x.mean()]
    rows = {}
    for j, alpha in enumerate(QUANTILES):
        diff = a - q[..., j]
        rows[f"pinball q{alpha:g}"] = per_day(np.maximum(alpha * diff, (alpha - 1) * diff))
    lo, hi = q[..., 0], q[..., -1]
    level = round((QUANTILES[-1] - QUANTILES[0]) * 100)
    rows[f"{level}% coverage (%)"] = per_day(((a >= lo) & (a <= hi)) * 100.0)
    rows[f"{level}% width"] = per_day(hi - lo)
    return pd.DataFrame(rows, index=DAY_LABELS).T


# --- Figures ----------------------------------------------------------------------------------

COLORS = {"last day": "#2a78d6", "7-day mean": "#eb6834", "XGBoost": "#1baf7a",
          "LightGBM": "#8a4fd0", "LightGBM quantile": "#d4417a"}      # one fixed colour per model, in every chart
MODELS = list(COLORS)
LEARNED = [m for m in MODELS if m not in NAIVE] # the models compared with the best naive baseline
CLI_NAMES = {m.lower().replace(" ", "-"): m for m in LEARNED} # --models names: xgboost, lightgbm, lightgbm-quantile
INK, MUTED, AXIS, GRID, SURFACE = "#52514e", "#898781", "#c3c2b7", "#e1e0d9", "#fcfcfb"
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": "#0b0b0b",
    "axes.titlelocation": "left", "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False,
}

def plot_example(region: str, starts: pd.DatetimeIndex, history: np.ndarray, actual: np.ndarray, 
                 forecasts: dict[str, np.ndarray], band: np.ndarray | None = None):
    """band: optional (n, HORIZON_HOURS, len(QUANTILES)) quantile forecasts, drawn as the LightGBM quantile interval."""
    i = len(starts) // 2                                            # the forecast in the middle of the test period
    past = pd.date_range(end=starts[i], periods=HISTORY_HOURS + 1, freq="h")[:-1]
    future = pd.date_range(starts[i], periods=HORIZON_HOURS, freq="h")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    if band is not None:
        level = round((QUANTILES[-1] - QUANTILES[0]) * 100)
        ax.fill_between(future, band[i, :, 0], band[i, :, -1], color=COLORS["LightGBM quantile"], alpha=0.15,
                        linewidth=0, label=f"LightGBM {level}% interval")
    ax.plot(past.append(future), np.r_[history[i], actual[i]], color=INK, label="actual")
    for name, p in forecasts.items():
        ax.plot(future, p[i], color=COLORS[name], label=name)
    ax.axvline(starts[i], color=MUTED, linewidth=1)
    ax.annotate("forecast start", (starts[i], 1), xycoords=("data", "axes fraction"), xytext=(5, -3),
                textcoords="offset points", va="top", color=MUTED, fontsize=9)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.margins(x=0, y=0.12)
    ax.set_ylabel("%CFE")
    ax.set_title(f"{region}: one forecast and the 7 days of history it is built from", pad=28)
    ax.legend(ncol=7, loc="lower left", bbox_to_anchor=(-0.01, 1.0))
    return fig

def plot_error_by_hour(region: str, actual: np.ndarray, forecasts: dict[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for name, p in forecasts.items():
        ax.plot(np.arange(1, HORIZON_HOURS + 1), np.abs(p - actual).mean(axis=0), color=COLORS[name], label=name)
    ax.set_xlim(0, HORIZON_HOURS)
    ax.set_xticks(np.arange(0, HORIZON_HOURS + 1, 24))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.grid(axis="x")
    for k in range(HORIZON_DAYS):
        ax.annotate(f"day {k + 1}", (24 * k + 12, 1), xycoords=("data", "axes fraction"), xytext=(0, -4),
                    textcoords="offset points", ha="center", va="top", color=MUTED, fontsize=9)
    ax.set_xlabel("hours ahead")
    ax.set_ylabel("MAE (pp)")
    ax.set_title(f"{region}: error by hour ahead, {len(actual)} forecasts")
    ax.legend(loc="lower right", ncol=5)
    return fig

def plot_regions(summary: pd.DataFrame, models: list[str]):
    order = summary.sort_values("7-day mean").index # hardest region at the top
    ypos = np.arange(len(order))
    shown = summary.loc[order, models]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hlines(ypos, shown.min(axis=1), shown.max(axis=1), color=AXIS, linewidth=1.5, zorder=1)
    for name in models:
        ax.plot(shown[name], ypos, "o", color=COLORS[name], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=2)
    ax.set_yticks(ypos, order)
    ax.set_xlim(0, None)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MAE over the 7-day window (pp)")
    ax.set_title("All regions: error over the whole window")
    ax.legend(loc="lower right")
    return fig


# --- Run: all regions, with tables and figures for SPECIAL_REGION -------------------------------------

def show(title: str, table: pd.DataFrame, decimals: int = 2):
    print(f"\n{title}\n{table.round(decimals).to_string()}")

def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", (OUT_DIR / name).relative_to(ROOT))

def parse_settings(argv=None):
    """Command-line settings: which regions to score and which period of their data to use."""
    def date(text):
        pd.Timestamp(text) # fails on anything that is not a date
        return text
    p = argparse.ArgumentParser(description="Naive, XGBoost and LightGBM baselines for forecasting hourly %CFE.")
    p.add_argument("--regions", nargs="+", metavar="REGION", help=f"regions to score (default: every region in the data folder).")
    p.add_argument("--start", type=date, metavar="DATE", help="first day of data to use, e.g. 2023-01-01 (default: the first day available)")
    p.add_argument("--end", type=date, metavar="DATE", help="last day of data to use, included (default: the last day available)")
    p.add_argument("--models", nargs="+", choices=list(CLI_NAMES), metavar="MODEL", help=f"learned models to run, from: {', '.join(CLI_NAMES)} (default: all).")
    p.add_argument("--save-forecasts", action="store_true", help="Saves every forecast to forecasts/<REGION>_forecasts.csv in the results folder")
    s = p.parse_args(argv)
    s.models = [m for m in LEARNED if s.models is None or m in {CLI_NAMES[c] for c in s.models}]
    if s.start and s.end and pd.Timestamp(s.start) > pd.Timestamp(s.end):
        p.error("--start must not be after --end")
    return s

def find_regions(target_regions: Sequence[str] | None = None) -> list[str]:
    regions = sorted(region_folder.name for region_folder in DATA_DIR.glob("*") if (region_folder / f"{region_folder.name}_lifecycle_emissions.csv").exists())
    if target_regions:
        missing = sorted(set(target_regions) - set(regions))
        if missing:
            raise SystemExit(f"No data for {', '.join(missing)} in {DATA_DIR}. Regions with data: {', '.join(regions) or 'none'}.")
        regions = sorted(target_regions)
    if not regions:
        raise SystemExit(f"No <REGION>/<REGION>_lifecycle_emissions.csv in {DATA_DIR}. "
                         "Download the data with `python data_collection.py electricity` first.")
    return regions

def load_region(region: str, start: str | None = None, end: str | None = None) -> pd.Series:
    """load_cfe, stopping with a clear message if the period is empty or too short to train on."""
    cfe = load_cfe(region, start=start, end=end)
    if cfe.empty:
        raise SystemExit(f"{region}: no data between {start or 'the start'} and {end or 'the end'}.")
    val_start, _, _ = split_dates(cfe)
    if cfe.index[0] + pd.Timedelta(hours=HISTORY_HOURS + HORIZON_HOURS) > val_start: # no room for a single training window
        raise SystemExit(
            f"{region}: the data covers {cfe.index[0]:%Y-%m-%d} to {cfe.index[-1]:%Y-%m-%d}, which is too short. "
            f"The last {TEST_MONTHS} months are the test period and the {VAL_MONTHS} months before are for validation, "
            f"so the baseline needs well over {TEST_MONTHS + VAL_MONTHS} months of data. "
            "Choose a longer period with --start/--end, or download more data with data_collection.py.")
    return cfe

def run_models(cfe: pd.Series, starts: pd.DatetimeIndex, history: np.ndarray,
               learned: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray | None, list[str]]:
    forecasts = {name: f(history) for name, f in NAIVE.items()}
    quantiles, trees = None, []
    if "XGBoost" in learned:
        models = fit_xgb(cfe)
        forecasts["XGBoost"] = xgb_forecast(models, starts, history)
        trees.append(f"XGBoost {[m.n_estimators for m in models.values()]}")
    if "LightGBM" in learned:
        model = fit_lgb(cfe)
        forecasts["LightGBM"] = lgb_forecast(model, starts, history)
        trees.append(f"LightGBM {model.n_estimators}")
    if "LightGBM quantile" in learned:
        models = fit_lgb_quantiles(cfe)
        quantiles = lgb_quantile_forecast(models, starts, history)
        forecasts["LightGBM quantile"] = quantiles[..., QUANTILES.index(0.5)]
        trees.append(f"LightGBM quantile {[m.n_estimators for m in models.values()]}")
    return forecasts, quantiles, trees

def report(scores: dict[str, pd.DataFrame], q_scores: dict[str, pd.DataFrame], mean_cfe: dict[str, float], learned: list[str]):
    models = list(NAIVE) + learned
    all_scores = pd.concat(scores, names=["region"])
    all_mae = all_scores.xs("MAE", level="metric")
    best_naive = all_mae[all_mae.index.get_level_values("model").isin(list(NAIVE))].groupby(level="region").min()
    best_naive["all"] = best_naive[DAY_LABELS[:-1]].mean(axis=1)                        # best naive
    all_q = pd.concat(q_scores, names=["region", "metric"]) if q_scores else None

    # Special region
    if SPECIAL_REGION in scores: 
        by_metric = pd.MultiIndex.from_product([["MAE", "RMSE", "bias"], models], names=["metric", "model"])
        show(f"{SPECIAL_REGION}: scores by forecast day (pp)", all_scores.loc[SPECIAL_REGION].swaplevel().reindex(by_metric))
        against = {"best naive, MAE (pp)": best_naive.loc[SPECIAL_REGION]}
        for name in learned:
            mae = all_mae.loc[(SPECIAL_REGION, name)]
            against[f"{name}, MAE (pp)"] = mae
            against[f"{name} against best naive (%)"] = (mae / best_naive.loc[SPECIAL_REGION] - 1) * 100
        show(f"{SPECIAL_REGION}: learned models against the best naive baseline", pd.DataFrame(against).T)
        if all_q is not None:
            show(f"{SPECIAL_REGION}: LightGBM quantile, pinball loss (pp) and interval", all_q.loc[SPECIAL_REGION])

    # All regions
    summary = all_mae["all"].unstack("model")[models].rename_axis(columns=None)
    summary.insert(0, "mean %CFE", pd.Series(mean_cfe))
    summary["best naive"] = best_naive["all"]
    for name in learned:
        summary[f"{name} against best naive (%)"] = (summary[name] / summary["best naive"] - 1) * 100
    show("All regions: MAE over the 7-day window (pp)", summary)
    for name in learned:
        show(f"All regions: {name} against best naive by forecast day (%)",
             (all_mae.xs(name, level="model") / best_naive - 1) * 100, decimals=1)
    show("All regions: MAE by forecast day (pp)", all_mae)
    if all_q is not None:
        show("All regions: LightGBM quantile over the 7-day window", all_q["all"].unstack("metric"))

    print()
    save(plot_regions(summary, models), "all_regions_mae.png")
    all_scores.round(4).to_csv(OUT_DIR / "scores_by_day.csv")
    print("saved", (OUT_DIR / "scores_by_day.csv").relative_to(ROOT))
    if all_q is not None:
        all_q.round(4).to_csv(OUT_DIR / "quantile_scores_by_day.csv")
        print("saved", (OUT_DIR / "quantile_scores_by_day.csv").relative_to(ROOT))

def forecast_table(starts: pd.DatetimeIndex, actual: np.ndarray, forecasts: dict[str, np.ndarray], quantiles: np.ndarray | None = None) -> pd.DataFrame:
    table = pd.DataFrame({"forecast start": np.repeat(starts, HORIZON_HOURS),
                          "target time": (starts.values[:, None] + np.arange(HORIZON_HOURS).astype("timedelta64[h]")).ravel(),
                          "hours ahead": np.tile(np.arange(1, HORIZON_HOURS + 1), len(starts)),
                          "actual": actual.ravel()})
    for name, forecast in forecasts.items():
        table[name] = forecast.ravel()
    if quantiles is not None:
        for j, alpha in enumerate(QUANTILES):
            if alpha != 0.5:
                table[f"LightGBM quantile q{alpha:g}"] = quantiles[..., j].ravel()
    return table

def main(argv=None):
    settings = parse_settings(argv)
    regions = find_regions(settings.regions)
    plt.switch_backend("Agg")           # figures are written to files, no windows
    plt.style.use(STYLE)
    (OUT_DIR / "forecasts" if settings.save_forecasts else OUT_DIR).mkdir(parents=True, exist_ok=True)

    scores, q_scores, mean_cfe = {}, {}, {}
    for region in regions:
        t0 = time.time()
        cfe = load_region(region, settings.start, settings.end)
        _, test_start, test_end = split_dates(cfe)
        starts, history, actual = make_windows(cfe)
        forecasts, quantiles, trees = run_models(cfe, starts, history, settings.models)
        scores[region] = pd.concat({name: score(actual, forecast) for name, forecast in forecasts.items()}, names=["model", "metric"])
        if quantiles is not None:
            q_scores[region] = score_quantiles(actual, quantiles)
        mean_cfe[region] = actual.mean()
        if settings.save_forecasts:
            forecast_table(starts, actual, forecasts, quantiles).to_csv(OUT_DIR / "forecasts" / f"{region}_forecasts.csv", index=False, float_format="%.3f")
        print(f"{region}: test {test_start:%Y-%m-%d} to {test_end - pd.Timedelta(days=1):%Y-%m-%d}, {len(starts)} forecasts, "
              + (f"trees {', '.join(trees)}, " if trees else "") + f"{time.time() - t0:.0f}s")
        if region == SPECIAL_REGION:
            save(plot_example(region, starts, history, actual, forecasts, band=quantiles), f"{region}_example_forecast.png")
            save(plot_error_by_hour(region, actual, forecasts), f"{region}_error_by_hour.png")

    report(scores, q_scores, mean_cfe, settings.models)

if __name__ == "__main__":
    main()
