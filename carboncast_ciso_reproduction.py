"""CarbonCast reproduction: CISO, two-tier model.

Reproduces the CISO results of "CarbonCast: Multi-Day Forecasting of Grid Carbon Intensity"
(Maji et al., BuildSys '22). It trains both tiers from scratch with the repo's configs and tests on
Jul 1 - Dec 28 2021: 181 daily 96-hour forecasts.

1. Tier 1: one small ANN per energy source forecasts 96 h of production. For day 1 it uses the
   day-ahead forecasts that DACF also uses.
2. Tier 2: a CNN-LSTM forecasts carbon intensity from its own history plus the weather and tier-1
   forecasts. It is trained twice: once on our tier-1 forecasts and once on the authors'.
3. Baseline (DACF): Eq. 1 applied to the same two sets of source forecasts.

Trained forecasts are cached in results/carboncast_ciso/, so nothing is trained: the script takes
about a minute on its first run and seconds after that. Training everything again (RETRAIN = True)
takes about 15 min on a laptop.

Run:     uv run python carboncast_ciso_reproduction.py
Output:  tables in the terminal; CISO_reproduction_summary.csv and CISO_mape_by_day.png in
         results/carboncast_ciso/
"""
import os, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # the figure is written to a file, no window
import matplotlib.pyplot as plt
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
from keras import layers

REGION    = "CISO"
CEF_TYPES = ["lifecycle", "direct"]   # lifecycle and direct emission factors
N_RUNS    = 3                         # the paper averages 3 tier-2 runs
SEED      = 0
RETRAIN   = False                     # True: ignore the cache and train everything again

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "CarbonCast"
DATA = REPO / "data" / REGION
assert DATA.exists(), f"{DATA} is missing. Run `git submodule update --init` in the repo root first."
OUT = ROOT / "results" / "carboncast_ciso"
for d in ("tier1", "tier2"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

def load_config(path):   # the repo's JSON configs contain // comments
    text = re.sub(r'("(?:\\.|[^"\\])*")|//[^\n]*', lambda m: m.group(1) or "", Path(path).read_text())
    return json.loads(re.sub(r",\s*([}\]])", r"\1", text))

T1 = load_config(REPO / "src" / "firstTierConfig.json")
T2 = load_config(REPO / "src" / "secondTierConfig.json")
H, P = 24, 96                         # 24-h input window, 96-h forecast
NDAYS = P // 24
repo_path = lambda p: p.replace("../", f"{REPO}/")

# Paper, MAPE % (Tables 3-6)
PAPER = {"lifecycle": [8.08, 11.19, 12.93, 13.62, 11.45, 9.91, 18.58, 24.27],
         "direct":    [9.40, 13.23, 15.09, 15.75, 13.37, 11.96, 22.21, 25.99]}
PAPER_DACF = {"lifecycle": [6.45, 12.26, 16.02, 17.22]}
COLS = ["day 1", "day 2", "day 3", "day 4", "mean", "median", "p90", "p95"]


# --- Helpers ----------------------------------------------------------------------------------
# These re-implement the original functions in common.py and utility.py:
# - Calendar features: hour and year sine/cosine, plus a weekend flag.
# - Scaling: min-max scaling fitted on the training data only.
# - Training windows: a 24-h history paired with the matching 24 rows of a 96-h forecast block.
# - Rollout: forecasts day by day, feeding the predictions back in as history.

def read_ts(path):
    df = pd.read_csv(path)
    t = df.pop("UTC time")
    try:
        idx = pd.to_datetime(t, format="%Y-%m-%d %H:%M:%S")
    except ValueError:
        idx = pd.to_datetime(t, format="%m/%d/%Y %H:%M:%S")
    return df.loc[:, ~df.columns.str.startswith("Unnamed")].set_index(pd.DatetimeIndex(idx))

def calendar_features(idx):
    hour = 2 * np.pi * idx.hour / 24
    year = 2 * np.pi * (idx - pd.Timestamp("1970-01-01")).total_seconds().values / (365.25 * 86400)
    return np.column_stack([np.sin(hour), np.cos(hour), np.sin(year), np.cos(year), idx.day_of_week >= 5])

def ffill(a):
    return pd.DataFrame(a).ffill().values

def minmax(train):
    # Return a scaler fitted on train. Constant columns are left unscaled, as in the original.
    mn, rng = train.min(0), train.max(0) - train.min(0)
    const, rng = rng == 0, np.where(rng == 0, 1, rng)
    return lambda a: np.where(const, a, (a - mn) / rng), mn[0], rng[0]

def unscale(a, mn, rng):
    return np.round(np.maximum(a * rng + mn, 0), 5)

def make_windows(data, blocks=None):
    # Hour i: X = rows i..i+23 (+ rows i%24..i%24+23 of forecast block i//24), y = column 0 of the next 24 rows.
    n = len(data) - 2 * H + 1
    rows = np.arange(n)[:, None] + np.arange(H)
    X, y = data[rows], data[rows + H, 0]
    if blocks is not None:
        d, h = np.arange(n) // 24, np.arange(n) % 24
        X = np.concatenate([X, blocks[d[:, None], h[:, None] + np.arange(H)]], axis=2)
    return X.astype("float32"), y.astype("float32")

def test_windows(last_val_day, test, n):
    # Starting history (the actual previous day) and true future rows (for the calendar features) of n forecasts.
    days = np.vstack([last_val_day, test])
    return (np.stack([days[i * 24:(i + 1) * 24] for i in range(n)]),
            np.stack([test[i * 24:i * 24 + P] for i in range(n)]))

def rollout(model, hist, future, blocks=None, day1=None):
    # 96-h forecast, 24 h at a time, feeding each predicted day back as history (getDayAheadForecasts).
    preds = []
    for j in range(NDAYS):
        x = hist if blocks is None else np.concatenate([hist, blocks[:, j * 24:(j + 1) * 24]], axis=2)
        y = day1 if (j == 0 and day1 is not None) else np.asarray(model(x.astype("float32")))
        preds.append(y)
        hist = future[:, j * 24:(j + 1) * 24].copy()
        hist[:, :, 0] = y
    return np.concatenate(preds, axis=1)

def fit(model, X, y, Xv, yv, hp, callbacks=()):
    es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)   # keeps the best-validation weights
    model.compile(optimizer=keras.optimizers.Adam(hp["LEARNING_RATE"]), loss=hp["LOSS_FUNC"])
    model.fit(X, y, epochs=hp["EPOCH"], batch_size=hp["BATCH_SIZE"][0], validation_data=(Xv, yv),
              verbose=0, callbacks=[*callbacks, es])
    return model

def daily_mape(actual, pred):   # (n, 96) -> (n, 4): MAPE % of each forecast day
    a, p = actual.reshape(-1, NDAYS, 24), pred.reshape(-1, NDAYS, 24)
    return 100 * np.mean(np.abs(a - p) / np.maximum(np.abs(a), 1e-7), axis=2)

def summarize(dm):
    return dict(zip(COLS, [*dm.mean(0), dm.mean(), np.median(dm), np.percentile(dm, 90), np.percentile(dm, 95)]))


# --- 1. Tier 1: production forecasts per source -----------------------------------------------
# - Inputs: each source's last 24 h of production plus calendar features. Solar, wind and hydro also
#   get the 5 GFS weather-forecast features.
# - Weather timing: the repo dates each 96-h weather block one day early. The block dated day d is
#   the GFS run from 00Z on day d + 1. Pairing the history day's block with the history therefore
#   gives weather for the day being predicted.
# - Model: Dense(50) -> Dense(34) -> Dense(24), early stopping on the last 30 days before the test period.
# - Retraining: four expanding-window models per source, one per half-year from H1 2020 to H2 2021.
#   This gives out-of-sample forecasts for the whole period that tier 2 uses.
# - Day 1: taken from the DACF day-ahead forecast (avg_<source>_production_forecast). The ANN
#   forecasts only days 2-4.

SOURCES = [s.lower() for s in T1[REGION]["SOURCES"]]
fuel = {s: read_ts(DATA / "fuel_forecast" / f"{REGION}_{s}_2019_clean.csv") for s in SOURCES}
weather = read_ts(repo_path(T1[REGION]["WEATHER_FORECAST_IN_FILE_NAME"]))
W_BLOCKS = weather.values.reshape(-1, P, weather.shape[1]).astype(float)   # block k dated 2019-01-01 + k days

def ann(n_feat, hp):
    return keras.Sequential([keras.Input((H, n_feat)), layers.Flatten(),
                             layers.Dense(hp["HIDDEN_UNITS"][0], activation=hp["ACTIVATION_FUNC"]),
                             layers.Dense(hp["HIDDEN_UNITS"][1], activation=hp["ACTIVATION_FUNC"]),
                             layers.Dense(24)])

def tier1_period(source, period, seed):
    hp, n_val = T1["FIRST_TIER_ANN_MODEL_HYPERPARAMS"], T1["NUM_VAL_DAYS"]
    limit, n_test = period["DATASET_LIMITER"], period["NUM_TEST_DAYS"]
    df = fuel[source]
    data = ffill(np.column_stack([df[source].values, calendar_features(df.index)]).astype(float))
    test = data[limit - n_test * 24:limit + P - 24]                    # test days + 3 extra days for the last forecast
    train, val = data[:limit - (n_test + n_val) * 24], data[limit - (n_test + n_val) * 24:limit - n_test * 24]
    scale, mn, rng = minmax(train)
    train, val, test = scale(train), scale(val), scale(test)

    wtr = wva = wte = None
    if source in ("solar", "wind", "hydro"):
        wb = W_BLOCKS[:limit // 24]
        wscale = minmax(ffill(wb[:-(n_test + n_val)].reshape(-1, wb.shape[2])))[0]
        wb = wscale(ffill(wb.reshape(-1, wb.shape[2]))).reshape(wb.shape)
        wtr, wva, wte = wb[:-(n_test + n_val)], wb[-(n_test + n_val):-n_test], wb[-n_test:]

    keras.utils.set_random_seed(seed)
    model = fit(ann(train.shape[1] + (0 if wtr is None else wtr.shape[2]), hp),
                *make_windows(train, wtr), *make_windows(val, wva), hp)

    hist, future = test_windows(val[-24:], test, n_test)
    blocks = None if wte is None else np.concatenate([wva[-1:], wte])[:n_test]   # dated the history day = GFS run for the target day
    day1 = None
    if T1[REGION]["PARTIAL_FORECAST_AVAILABILITY_LIST"][SOURCES.index(source)]:
        day1 = (df[f"avg_{source}_production_forecast"].values[limit - n_test * 24:limit] - mn) / rng
        day1 = day1.reshape(n_test, 24)
    pred = unscale(rollout(model, hist, future, blocks, day1), mn, rng)
    issue = pd.date_range(df.index[limit - n_test * 24], periods=n_test, freq="D")
    return pd.Series(pred.ravel(), index=np.repeat(issue, P), name=source)

tier1 = {}   # source -> Series of 96-h forecasts indexed by issue date
for i, source in enumerate(SOURCES):
    f = OUT / "tier1" / f"{REGION}_ANN_DA_{source}.csv"
    if f.exists() and not RETRAIN:
        d = pd.read_csv(f, parse_dates=["issue_date"])
        tier1[source] = pd.Series(d[f"avg_{source}_production_forecast"].values, index=d["issue_date"].values)
        continue
    tier1[source] = pd.concat([tier1_period(source, p, SEED + 100 * i + j)
                               for j, p in enumerate(T1["TRAIN_TEST_PERIOD"].values())])
    print(f"tier 1 trained: {source}")
    issue = tier1[source].index
    pd.DataFrame({"issue_date": issue,
                  "datetime": issue + pd.to_timedelta(np.tile(np.arange(P), len(issue) // P), "h"),
                  f"avg_{source}_production_forecast": tier1[source].values}).to_csv(f, index=False)
print("tier-1 forecasts:", tier1[SOURCES[0]].index.min().date(), "→", tier1[SOURCES[0]].index.max().date())


# --- 2. Tier 2: CNN-LSTM for carbon intensity -------------------------------------------------
# Input file. The tier-2 input is one 96-row block per day. Block d combines:
# - the weather block dated day d, which is the GFS run for day d + 1 (see section 1);
# - the tier-1 forecasts issued for day d + 1.
# The model uses the history of day d to predict day d + 1, so both the weather and the production
# forecasts line up with the day being predicted.
#
# Tier 2 is trained on two versions of this file. One is rebuilt from our tier-1 forecasts. The other
# is the authors' CISO_96hr_forecasts_DA.csv. The weather columns are identical; only the tier-1
# forecasts differ.
#
# Model setup:
# - Inputs: carbon intensity + 5 calendar features, plus 12 forecast features (5 weather + 7 sources).
# - Architecture: Conv1D(4) -> MaxPool -> Conv1D(16) -> RepeatVector -> LSTM(24) -> Dropout -> Dense(24).
# - Data split: train on 2020, validate on H1 2021, test on H2 2021.

F = T2[REGION]["NUM_FORECAST_FEATURES"]                              # 5 weather + 7 sources
FC_DAYS = pd.date_range("2020-01-01", "2021-12-28")                  # 728 blocks
FC_SOURCES = ["wind", "solar", "nat_gas", "coal", "nuclear", "hydro", "oil"]
w0 = (FC_DAYS[0] - weather.index[0]).days
ours_fc = np.concatenate(
    [W_BLOCKS[w0:w0 + len(FC_DAYS)]] +
    [np.stack([tier1[s].loc[day + pd.Timedelta(days=1)].values for day in FC_DAYS])[..., None] for s in FC_SOURCES],
    axis=2)                                                          # (728, 96, 12)
authors_fc = read_ts(repo_path(T2[REGION]["FORECAST_IN_FILE_NAME"])).iloc[:, :F].values.reshape(-1, P, F)
assert ours_fc.shape == authors_fc.shape and np.allclose(ours_fc[..., :5], authors_fc[..., :5])   # same weather
TIER2_INPUTS = {"ours": ours_fc, "authors": authors_fc}

def cnn_lstm(n_feat, hp):
    act = hp["ACTIVATION_FUNC"]
    return keras.Sequential([
        keras.Input((H, n_feat)),
        layers.Conv1D(hp["CNN_NUM_FILTERS1"], hp["CNN_KERNEL1"], padding="same", activation=act),
        layers.MaxPooling1D(hp["CNN_POOL_SIZE"]),
        layers.Conv1D(hp["CNN_NUM_FILTERS2"], hp["CNN_KERNEL2"], activation=act),
        layers.Flatten(), layers.RepeatVector(24), layers.LSTM(24),
        layers.Dropout(hp["LSTM_DROPOUT_RATE"]), layers.Dense(24)])

def tier2(cef, fc, tag):
    # Return (actual, [forecast of each run]), both (181, 96).
    hp, n_test, n_val = T2["SECOND_TIER_CNN_LSTM_MODEL_HYPERPARAMS"], T2["NUM_TEST_DAYS"], T2["NUM_VAL_DAYS"]
    em = read_ts(repo_path(T2[REGION][f"{cef.upper()}_CEF_IN_FILE_NAME"]))
    data = ffill(np.column_stack([em["carbon_intensity"].values, calendar_features(em.index)]).astype(float))
    n_hist = len(data) - (n_test + NDAYS - 1) * 24
    train, val, test = data[:n_hist - n_val * 24], data[n_hist - n_val * 24:n_hist], data[n_hist:]
    actual = np.stack([test[i * 24:i * 24 + P, 0] for i in range(n_test)])
    scale, mn, rng = minmax(train)
    train, val, test = scale(train), scale(val), scale(test)

    fb = fc.reshape(-1, F)
    fb = minmax(ffill(fb[:-(n_test + n_val) * P]))[0](ffill(fb)).reshape(fc.shape)
    ftr, fva, fte = fb[:-(n_test + n_val)], fb[-(n_test + n_val):-n_test], fb[-n_test:]
    hist, future = test_windows(val[-24:], test, n_test)
    blocks = np.concatenate([fva[-1:], fte])[:n_test]

    runs = []
    for run in range(N_RUNS):
        f = OUT / "tier2" / f"{REGION}_{cef}_{tag}_run{run}.csv"
        if f.exists() and not RETRAIN:
            runs.append(pd.read_csv(f)["forecast"].values.reshape(-1, P))
            continue
        keras.utils.set_random_seed(SEED + 1000 + run)
        lr = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=6, min_lr=hp["MIN_LEARNING_RATE"])
        model = fit(cnn_lstm(train.shape[1] + F, hp), *make_windows(train, ftr), *make_windows(val, fva),
                    hp, callbacks=[lr])
        runs.append(unscale(rollout(model, hist, future, blocks), mn, rng))
        pd.DataFrame({"actual": actual.ravel(), "forecast": runs[-1].ravel()}).to_csv(f, index=False)
        print(f"tier 2 trained: {tag} tier 1, {cef}, run {run}")
    return actual, runs

TEST_ISSUE = pd.date_range("2021-07-01", periods=T2["NUM_TEST_DAYS"])   # Jul 1 - Dec 28 2021
actual, carboncast = {}, {}
for tag, fc in TIER2_INPUTS.items():
    for cef in CEF_TYPES:
        actual[cef], carboncast[(tag, cef)] = tier2(cef, fc, tag)
per_run = pd.DataFrame({(tag, cef, r): summarize(daily_mape(actual[cef], p))
                        for (tag, cef), runs in carboncast.items() for r, p in enumerate(runs)}).T.sort_index()
per_run.index.names = ["tier 1", "CEF", "run"]
print(f"\nCarbonCast, each tier-2 run (MAPE %)\n{per_run.round(2).to_string()}")


# --- 3. DACF baseline -------------------------------------------------------------------------
# DACF computes carbon intensity from the source forecasts with Eq. 1: CI = sum_i E_i * CEF_i / sum_i E_i.
# The emission factors come from carbonIntensityCalculator.py. Eq. 1 is applied to the same two sets
# of source forecasts tier 2 was trained on, so each comparison with CarbonCast is like-for-like:
# - The authors' tier-1 forecasts: the repo's DACF input CISO_96hr_source_prod_forecasts_DA_Jul_Dec_2021.csv.
#   It is identical to the source columns of the authors' tier-2 input, as the check below confirms.
#   This reproduces the paper's DACF numbers exactly.
# - Our tier-1 forecasts.

CEF = {"lifecycle": {"coal": 820, "nat_gas": 490, "nuclear": 12, "oil": 650, "hydro": 24, "solar": 45, "wind": 11, "other": 700},
       "direct":    {"coal": 760, "nat_gas": 370, "nuclear": 0,  "oil": 406, "hydro": 0,  "solar": 0,  "wind": 0,  "other": 575}}

def dacf(source_fc, cef):
    E = np.stack([np.clip(source_fc[s].loc[TEST_ISSUE].values.reshape(-1, P), 0, None) for s in SOURCES], -1)
    return np.round(E @ np.array([CEF[cef][s] for s in SOURCES]) / E.sum(-1), 2)

baseline = pd.read_csv(DATA / f"{REGION}_96hr_source_prod_forecasts_DA_Jul_Dec_2021.csv")
for i, s in enumerate(FC_SOURCES):   # test forecasts issued Jul 1..Dec 28 = blocks dated Jun 30..Dec 27
    assert np.array_equal(baseline[f"avg_{s}_production_forecast"].values.reshape(-1, P), authors_fc[-len(TEST_ISSUE) - 1:-1, :, 5 + i])
authors_t1 = {s: pd.Series(baseline[f"avg_{s}_production_forecast"].values, index=np.repeat(TEST_ISSUE, P)) for s in SOURCES}
dacf_fc = {(tag, cef): dacf(src, cef) for tag, src in {"ours": tier1, "authors": authors_t1}.items() for cef in CEF_TYPES}


# --- 4. Results vs the paper (MAPE %) ---------------------------------------------------------

TIER1_LABEL = {"authors": "authors' tier 1", "ours": "our tier 1"}
rows = {}
for cef in CEF_TYPES:
    rows[(cef, "paper", "CarbonCast")] = dict(zip(COLS, PAPER[cef]))
    if cef in PAPER_DACF:
        rows[(cef, "paper", "DACF")] = dict(zip(COLS, PAPER_DACF[cef]))
    for tag, label in TIER1_LABEL.items():
        rows[(cef, label, "CarbonCast")] = per_run.loc[(tag, cef)].mean()   # mean of the N_RUNS runs
        rows[(cef, label, "DACF")] = summarize(daily_mape(actual[cef], dacf_fc[(tag, cef)]))
summary = pd.DataFrame(rows).T[COLS]
summary.index.names = ["CEF", "source forecasts", "model"]
summary.to_csv(OUT / f"{REGION}_reproduction_summary.csv")
print(f"\nResults vs the paper (MAPE %)\n{summary.round(2).to_string()}\n")
print("saved", (OUT / f"{REGION}_reproduction_summary.csv").relative_to(ROOT))

fig, axes = plt.subplots(1, len(CEF_TYPES), figsize=(7 * len(CEF_TYPES), 4), sharey=True)
for ax, cef in zip(np.atleast_1d(axes), CEF_TYPES):
    s = summary.loc[cef]
    s = s[s.index.get_level_values(0) != "paper"][COLS[:NDAYS]]
    s.index = [f"{model} on {src}" for src, model in s.index]
    s.T.plot.bar(ax=ax, rot=0, width=0.8, color=["#1f77b4", "#aec7e8", "#d62728", "#ff9896"])
    ax.set_title(f"{REGION}, {cef} emission factors"); ax.set_ylabel("MAPE (%)"); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / f"{REGION}_mape_by_day.png", dpi=150)
print("saved", (OUT / f"{REGION}_mape_by_day.png").relative_to(ROOT))
