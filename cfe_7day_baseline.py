
from pathlib import Path
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


REGION = "DE"                       # region of the detailed tables and figures; all regions are scored
INCLUDE_BIOMASS_AS_CFE = False      # definitions differ; keep it explicit
CFE_SOURCES = ["nuclear", "hydro", "solar", "wind", "geothermal"] + (["biomass"] if INCLUDE_BIOMASS_AS_CFE else [])

HORIZON_DAYS = 7                    # forecast window in days
START_HOUR = 0                      # UTC hour at which each forecast window starts (one forecast per day)
VAL_START = "2021-01-01"            # XGBoost trains on targets before this date and picks its number of trees on VAL_START to TEST_START
TEST_START, TEST_END = "2021-07-01", "2022-01-01"   # test forecasts start from TEST_START; every target hour is before TEST_END
EXAMPLE_START = "2021-10-04"        # the forecast drawn in <REGION>_example_forecast.png

XGB_PARAMS = dict(learning_rate=0.05, max_depth=6, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", random_state=0, n_jobs=-1)
MAX_TREES, PATIENCE = 3000, 50      # early stopping

H = 24 * HORIZON_DAYS               # hours to predict
HIST = 24 * 7                       # hours of history each forecast is given
DAYS = [f"day {k}" for k in range(1, HORIZON_DAYS + 1)] + ["all"]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "CarbonCast" / "data"
OUT_DIR = ROOT / "results" / "cfe_7day_baseline"


# --- The %CFE series --------------------------------------------------------------------------

def load_cfe(region, cfe_sources=CFE_SOURCES):
    """Hourly %CFE of a region. Negative generation is set to zero, gaps of up to 3 hours are
    interpolated and longer ones carried forward."""
    df = pd.read_csv(DATA_DIR / region / f"{region}_lifecycle_emissions.csv")
    df.index = pd.to_datetime(df.pop("UTC time"), utc=True).dt.tz_localize(None)   # some files carry +00:00
    sources = [c for c in df.columns if c != "carbon_intensity" and not c.startswith("Unnamed")]
    gen = df[sources].clip(lower=0)
    gen = gen[~gen.index.duplicated()].asfreq("h").interpolate(limit=3).ffill()
    total = gen.sum(axis=1).replace(0, np.nan)
    cfe = gen[[c for c in sources if c in cfe_sources]].sum(axis=1) / total * 100
    return cfe.ffill().rename("cfe")


# --- Forecast windows -------------------------------------------------------------------------

def make_windows(cfe, start=TEST_START, end=TEST_END, start_hour: int | None = START_HOUR):
    """One row per forecast: its start time, the 168 hours before it (history) and the 168 hours to
    predict (actual). Every forecast is built from history alone. Scoring uses forecasts starting at
    START_HOUR; start_hour=None returns a window for every hour, which XGBoost uses for training."""
    y, idx = cfe.to_numpy(), cfe.index
    ok = (idx >= pd.Timestamp(start)) & (idx + pd.Timedelta(hours=H) <= pd.Timestamp(end))
    if start_hour is not None:
        ok &= idx.hour == start_hour
    pos = np.flatnonzero(ok)
    pos = pos[(pos >= HIST) & (pos + H <= len(y))]
    history = y[pos[:, None] + np.arange(-HIST, 0)]
    actual = y[pos[:, None] + np.arange(H)]
    return idx[pos], history, actual


# --- The two naive baselines ------------------------------------------------------------------
# For a target hour t on forecast day k:
#   last day:    forecast(t) = y(t - 24k)
#   7-day mean:  forecast(t) = mean of y(t - 24(k + d)) for d = 0..6

def last_day(history):
    return np.tile(history[:, -24:], HORIZON_DAYS)

def mean_7d(history):
    profile = history.reshape(len(history), 7, 24).mean(axis=1)
    return np.tile(profile, HORIZON_DAYS)

NAIVE = {"last day": last_day, "7-day mean": mean_7d}


# --- XGBoost on lags --------------------------------------------------------------------------
# - What it predicts: the correction to the 7-day mean, y(t) - 7-day mean. The forecast is the 7-day
#   mean plus that correction, clipped to 0-100, so with nothing to learn the model falls back to
#   the baseline.
# - Two models: one for day 1 (hours 1-24) and one for days 2-7, each with the hours ahead as a feature.
# - Training: on forecasts starting at every hour of the day, not only START_HOUR. The number of
#   trees is chosen by early stopping: train on windows whose targets end before VAL_START, stop
#   when the error on windows from VAL_START to TEST_START stops improving. The model is then
#   refitted on everything before TEST_START with that number of trees.
# - Features (46):
#     last day against the weekly profile (24)   each of the last 24 hours minus the 7-day mean of that hour
#     level and spread (9)                       mean of each of the last 7 days; standard deviation of the
#                                                last 24 hours and of the last 7 days
#     target hour (8)                            %CFE at the same hour on each of the last 7 days, and their
#                                                mean (the 7-day-mean forecast)
#     calendar (5)                               hours ahead; hour of day, day of week and day of year
#                                                (sin, cos) of the target
#
# Choices checked on validation forecasts (Jan-Jun 2021), without using the test period:
# - One model for all 168 hours was worse than "last day" on day 1 in the three regions tried (DE, CISO, NYISO).
# - Predicting %CFE itself instead of the correction was worse than the 7-day mean in SE (0.95 against 0.81 pp).
# - One model per forecast day, source-mix shares and 28-day means gave no consistent gain over this
#   setup (DE, CISO, NYISO, ERCO, SE).
# - One model per hour for day 1 lowered the day-1 error by a further 1-8 % in those five regions, for
#   24 extra models. Left out to keep this simple.
# - Refitting on train + validation helped in 4 of the 5, checked on an earlier split.

BLOCKS = {"day 1": (1, 24), f"days 2-{HORIZON_DAYS}": (25, H)}      # hours ahead covered by each model

def make_features(starts, history):
    # One row per (forecast, hour ahead), forecasts in order, hours 1..H within each forecast.
    n = len(starts)
    days = history.reshape(n, 7, 24)                                # [forecast, day (oldest first), hour]
    profile = days.mean(axis=1)                                     # 7-day mean by hour
    per_forecast = np.hstack([
        history[:, -24:] - profile,                                 # last day against the weekly profile
        days.mean(axis=2),                                          # mean of each of the last 7 days
        history[:, -24:].std(axis=1, keepdims=True), history.std(axis=1, keepdims=True)])
    same_hour = days[:, ::-1][:, :, np.arange(H) % 24].transpose(0, 2, 1)     # (n, H, 7): 1, 2, ... 7 days before day 1
    target = pd.DatetimeIndex((starts.values[:, None] + np.arange(H).astype("timedelta64[h]")).ravel())
    angle = 2 * np.pi * target.day_of_year / 365.25
    per_hour = np.column_stack([
        np.tile(np.arange(1, H + 1), n), same_hour.reshape(n * H, 7), mean_7d(history).ravel(),
        target.hour, target.day_of_week, np.sin(angle), np.cos(angle)])
    return np.hstack([np.repeat(per_forecast, H, axis=0), per_hour]).astype(np.float32)

def block_rows(block, n_forecasts):
    lo, hi = BLOCKS[block]
    hours_ahead = np.arange(1, H + 1)
    return np.tile((hours_ahead >= lo) & (hours_ahead <= hi), n_forecasts)

def fit_xgb(cfe):
    starts, history, actual = make_windows(cfe, cfe.index[0], TEST_START, start_hour=None)
    X, y = make_features(starts, history), (actual - mean_7d(history)).ravel()
    train = np.repeat(starts + pd.Timedelta(hours=H) <= pd.Timestamp(VAL_START), H)
    val = np.repeat(starts >= pd.Timestamp(VAL_START), H)
    models = {}
    for block in BLOCKS:
        rows = block_rows(block, len(starts))
        stopper = xgb.XGBRegressor(n_estimators=MAX_TREES, early_stopping_rounds=PATIENCE, eval_metric="mae", **XGB_PARAMS)
        stopper.fit(X[rows & train], y[rows & train], eval_set=[(X[rows & val], y[rows & val])], verbose=False)
        models[block] = xgb.XGBRegressor(n_estimators=stopper.best_iteration + 1, **XGB_PARAMS).fit(X[rows], y[rows])
    return models

def xgb_forecast(models, starts, history):
    X = make_features(starts, history)
    correction = np.empty(len(X))
    for block, model in models.items():
        rows = block_rows(block, len(starts))
        correction[rows] = model.predict(X[rows])
    return np.clip(mean_7d(history) + correction.reshape(len(starts), H), 0, 100)


# --- Scores -----------------------------------------------------------------------------------

def score(actual, forecast):
    """MAE, RMSE and bias of each forecast day and of the whole window, in percentage points."""
    e = (forecast - actual).reshape(len(actual), HORIZON_DAYS, 24)
    per_day = lambda f: [f(e[:, k]) for k in range(HORIZON_DAYS)] + [f(e)]
    return pd.DataFrame({"MAE": per_day(lambda x: np.abs(x).mean()),
                         "RMSE": per_day(lambda x: np.sqrt((x ** 2).mean())),
                         "bias": per_day(np.mean)}, index=DAYS).T


# --- Figures ----------------------------------------------------------------------------------

COLORS = {"last day": "#2a78d6", "7-day mean": "#eb6834", "XGBoost": "#1baf7a"}     # one fixed colour per model, in every chart
MODELS = list(COLORS)
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

def plot_example(region, starts, history, actual, forecasts, start=EXAMPLE_START):
    i = starts.get_loc(pd.Timestamp(start))
    past = pd.date_range(end=starts[i], periods=HIST + 1, freq="h")[:-1]
    future = pd.date_range(starts[i], periods=H, freq="h")
    fig, ax = plt.subplots(figsize=(11, 3.8))
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
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(-0.01, 1.0))
    return fig

def plot_error_by_hour(region, actual, forecasts):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for name, p in forecasts.items():
        ax.plot(np.arange(1, H + 1), np.abs(p - actual).mean(axis=0), color=COLORS[name], label=name)
    ax.set_xlim(0, H)
    ax.set_xticks(np.arange(0, H + 1, 24))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.grid(axis="x")
    for k in range(HORIZON_DAYS):
        ax.annotate(f"day {k + 1}", (24 * k + 12, 1), xycoords=("data", "axes fraction"), xytext=(0, -4),
                    textcoords="offset points", ha="center", va="top", color=MUTED, fontsize=9)
    ax.set_xlabel("hours ahead")
    ax.set_ylabel("MAE (pp)")
    ax.set_title(f"{region}: error by hour ahead, {len(actual)} forecasts")
    ax.legend(loc="lower right", ncol=3)
    return fig

def plot_regions(summary):
    order = summary.sort_values("7-day mean").index                # hardest region at the top
    ypos = np.arange(len(order))
    shown = summary.loc[order, MODELS]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hlines(ypos, shown.min(axis=1), shown.max(axis=1), color=AXIS, linewidth=1.5, zorder=1)
    for name in MODELS:
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


# --- Run: all regions, with tables and figures for REGION -------------------------------------

def show(title, table, decimals=2):
    print(f"\n{title}\n{table.round(decimals).to_string()}")

def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", (OUT_DIR / name).relative_to(ROOT))

def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"{DATA_DIR} is missing. Run `git submodule update --init` in the repo root first.")
    plt.switch_backend("Agg")           # figures are written to files, no windows
    plt.style.use(STYLE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The same three forecasts for every region in CarbonCast/data/, with one XGBoost fit per region.
    regions = sorted(p.name for p in DATA_DIR.iterdir() if (p / f"{p.name}_lifecycle_emissions.csv").exists())
    scores, mean_cfe = {}, {}
    for region in regions:
        t0 = time.time()
        cfe = load_cfe(region)
        starts, history, actual = make_windows(cfe)
        models = fit_xgb(cfe)
        forecasts = {name: f(history) for name, f in NAIVE.items()}
        forecasts["XGBoost"] = xgb_forecast(models, starts, history)
        scores[region] = pd.concat({name: score(actual, p) for name, p in forecasts.items()}, names=["model", "metric"])
        mean_cfe[region] = actual.mean()
        print(f"{region}: {len(starts)} forecasts, trees {[m.n_estimators for m in models.values()]}, {time.time() - t0:.0f}s")
        if region == REGION:
            save(plot_example(region, starts, history, actual, forecasts), f"{region}_example_forecast.png")
            save(plot_error_by_hour(region, actual, forecasts), f"{region}_error_by_hour.png")

    all_scores = pd.concat(scores, names=["region"])
    all_mae = all_scores.xs("MAE", level="metric")
    bar = all_mae[all_mae.index.get_level_values("model").isin(list(NAIVE))].groupby(level="region").min()
    bar["all"] = bar[DAYS[:-1]].mean(axis=1)                        # best naive

    # One region
    by_metric = pd.MultiIndex.from_product([["MAE", "RMSE", "bias"], MODELS], names=["metric", "model"])
    show(f"{REGION}: scores by forecast day (pp)", all_scores.loc[REGION].swaplevel().reindex(by_metric))
    xgb_mae = all_mae.loc[(REGION, "XGBoost")]
    show(f"{REGION}: XGBoost against the best naive baseline",
         pd.DataFrame({"best naive, MAE (pp)": bar.loc[REGION], "XGBoost, MAE (pp)": xgb_mae,
                       "XGBoost against best naive (%)": (xgb_mae / bar.loc[REGION] - 1) * 100}).T)

    # All regions. Negative percentages: XGBoost beats the best naive baseline.
    summary = all_mae["all"].unstack("model")[MODELS].rename_axis(columns=None)
    summary.insert(0, "mean %CFE", pd.Series(mean_cfe))
    summary["best naive"] = bar["all"]
    summary["XGBoost against best naive (%)"] = (summary["XGBoost"] / summary["best naive"] - 1) * 100
    show("All regions: MAE over the 7-day window (pp)", summary)
    show("All regions: XGBoost against best naive by forecast day (%)",
         (all_mae.xs("XGBoost", level="model") / bar - 1) * 100, decimals=1)
    show("All regions: MAE by forecast day (pp)", all_mae)

    print()
    save(plot_regions(summary), "all_regions_mae.png")
    all_scores.round(4).to_csv(OUT_DIR / "scores_by_day.csv")
    print("saved", (OUT_DIR / "scores_by_day.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()
