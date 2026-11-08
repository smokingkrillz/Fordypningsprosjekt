"""Plot forecasts against the measured values on the test period.

Top panel:    the measured indicator and the forecasts from persistence, Linear ARX and XGBoost.
Bottom panel: the forecast residuals (measured - forecast) for the same models.

Each forecast is drawn at the time it is FOR (t_target), so the lines can be compared
directly with the measured value at that time. Segments are drawn separately, so no
line is drawn across a shutdown.

The models are trained exactly as in ModelEvaluation (same dataset, same splits,
hyperparameters picked on validation), so the MAE in the legend matches
model_vs_baselines.csv.

Usage:
    from plot_forecast import forecast_frame, plot_forecast
    fc = forecast_frame(p, "dev_p1", horizon=5)
    plot_forecast(fc, "dev_p1", 5, "results/forecast_dev_p1_h5.png")

Shell (prepared csv from prepare_oo.py):
    python plot_forecast.py <prepared.csv> <target> <horizon> <output.png>
    python plot_forecast.py prepared.csv dev_p1 5 results/forecast_dev_p1_h5.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dev.forecast_data import ForecastDataset
from dev.models import LinearARX, XGBoostForecaster

MODELS = ["persistence", "linear_arx", "xgboost"]
LABELS = {"persistence": "Persistence", "linear_arx": "Linear ARX", "xgboost": "XGBoost"}
COLORS = {"persistence": "#8a8984", "linear_arx": "#2a78d6", "xgboost": "#eb6834"}
STYLES = {"persistence": "--", "linear_arx": "-", "xgboost": "-"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def forecast_frame(source, target, horizon, lookback=10, split="test", train=0.6, val=0.2):
    """One row per forecast: segment, measured value and each model's forecast."""
    ds = ForecastDataset(source, target, horizon=horizon, lookback=lookback).check()
    splits = ds.split(train, val)
    s = splits[split]

    fc = pd.DataFrame({
        "segment": s.meta["segment"].to_numpy(),
        "actual": s.y.to_numpy(),
        "persistence": s.X["lag_0"].to_numpy(),
    }, index=pd.DatetimeIndex(s.meta["t_target"], name="time"))

    for model in (LinearARX(), XGBoostForecaster()):
        model.fit(splits["train"], splits["val"])
        fc[model.name] = model.predict(s.X)
    return fc.sort_index()


def plot_forecast(fc, target, horizon, path, unit="m/s", models=MODELS):
    mae = {m: np.mean(np.abs(fc["actual"] - fc[m])) for m in models}

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True,
                                      height_ratios=[2, 1], constrained_layout=True)

    for i, (_, seg) in enumerate(fc.groupby("segment", sort=False)):
        first = i == 0
        for m in models:
            label = f"{LABELS[m]}  (MAE {mae[m]:.3f} {unit})" if first else None
            top.plot(seg.index, seg[m], color=COLORS[m], ls=STYLES[m], lw=1.3, label=label)
            bottom.plot(seg.index, seg["actual"] - seg[m], color=COLORS[m], ls=STYLES[m], lw=1.1)
        top.plot(seg.index, seg["actual"], color=INK, lw=1.8, zorder=3,
                 label="Measured" if first else None)

    bottom.axhline(0, color=MUTED, lw=0.8)

    top.set_ylabel(f"{target} ({unit})", color=INK)
    bottom.set_ylabel(f"Residual ({unit})", color=INK)
    fig.suptitle(f"{target}: {horizon}-minute-ahead forecasts on the test period",
                 x=0.01, ha="left", fontsize=11, color=INK)
    top.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))

    locator = mdates.AutoDateLocator()
    bottom.xaxis.set_major_locator(locator)
    bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    for ax in (top, bottom):
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)

    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved {path}")
    return mae


if __name__ == "__main__":
    src, target, horizon, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    df = pd.read_csv(src, index_col=0, parse_dates=True)
    fc = forecast_frame(df, target, horizon)
    plot_forecast(fc, target, horizon, out)