"""Step 2 of RQ2: baseline forecasts every later model has to beat.

Baselines (all use only information available at the forecast origin):
    persistence      forecast = last value
    rolling_mean_k   forecast = mean of the last k values
    segment_mean     forecast = mean of the indicator since the segment started
    train_mean       forecast = one constant, the mean of the training targets

Scores: MAE and RMSE (in the indicator's unit, m/s), and skill versus persistence
(1 - MAE / MAE_persistence; 0 = no better than persistence, positive = better, negative = worse).

Usage:
    ev = BaselineEvaluation(p, horizons=(1, 5, 15), lookback=10).run()
    ev.report()
    ev.save("results")

Shell (on a prepared csv written by prepare_oo.py):
    python baselines.py <prepared.csv> <output_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dev.forecast_data import ForecastDataset, INDICATORS


# ---- metrics ----------------------------------------------------------------------------
def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


# ---- baselines --------------------------------------------------------------------------
class Baseline:
    name = "baseline"

    def fit(self, split):
        return self

    def predict(self, X):
        raise NotImplementedError


class Persistence(Baseline):
    name = "persistence"

    def predict(self, X):
        return X["lag_0"].to_numpy()


class RollingMean(Baseline):
    def __init__(self, k=5):
        self.k = k
        self.name = f"rolling_mean_{k}"

    def predict(self, X):
        cols = [f"lag_{i}" for i in range(self.k)]
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"{self.name} needs lookback >= {self.k}")
        return X[cols].mean(axis=1).to_numpy()


class SegmentMean(Baseline):
    name = "segment_mean"

    def predict(self, X):
        return X["seg_mean"].to_numpy()


class TrainMean(Baseline):
    name = "train_mean"

    def fit(self, split):
        self.mu = float(split.y.mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.mu)


def default_baselines(lookback):
    bl = [Persistence(), SegmentMean(), TrainMean()]
    bl += [RollingMean(k) for k in (3, 5, 10) if k <= lookback]
    return bl


# ---- evaluation -------------------------------------------------------------------------
class BaselineEvaluation:
    def __init__(self, source, targets=INDICATORS, horizons=(1, 5, 15), lookback=10,
                 baselines=None, train=0.6, val=0.2, **dataset_kw):
        self.source = source
        self.targets = targets
        self.horizons = horizons
        self.lookback = lookback
        self.baselines = baselines or default_baselines(lookback)
        self.train, self.val = train, val
        self.dataset_kw = dataset_kw
        self.datasets = {}          # (target, horizon) -> ForecastDataset, reused by later models
        self.results = pd.DataFrame()

    def run(self):
        rows = []
        for target in self.targets:
            for h in self.horizons:
                ds = ForecastDataset(self.source, target, horizon=h, lookback=self.lookback,
                                     **self.dataset_kw).check()
                try:
                    splits = ds.split(self.train, self.val)
                except ValueError as e:
                    print(f"  skipped: {e}")
                    continue
                self.datasets[(target, h)] = ds
                for b in self.baselines:
                    b.fit(splits["train"])
                    for name in ("val", "test"):
                        s = splits[name]
                        pred = b.predict(s.X)
                        rows.append({"target": target, "horizon": h, "split": name,
                                     "baseline": b.name, "n": len(s),
                                     "mae": mae(s.y, pred), "rmse": rmse(s.y, pred)})
        res = pd.DataFrame(rows)
        if not res.empty:
            ref = (res[res["baseline"] == "persistence"]
                   .set_index(["target", "horizon", "split"])["mae"].rename("mae_persistence"))
            res = res.join(ref, on=["target", "horizon", "split"])
            res["skill_vs_persistence"] = 1 - res["mae"] / res["mae_persistence"]
            res = res.drop(columns="mae_persistence")
        self.results = res
        return self

    def table(self, split="test", metric="mae"):
        """Rows = target and horizon, columns = baselines."""
        r = self.results[self.results["split"] == split]
        return r.pivot_table(index=["target", "horizon"], columns="baseline", values=metric)

    def best(self, split="test"):
        """Best baseline per target and horizon: the number later models must beat."""
        r = self.results[self.results["split"] == split]
        idx = r.groupby(["target", "horizon"])["mae"].idxmin()
        return r.loc[idx, ["target", "horizon", "baseline", "mae", "rmse", "skill_vs_persistence"]] \
                .set_index(["target", "horizon"])

    def report(self):
        pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format)
        if self.results.empty:
            print("no results")
            return self
        ds = next(iter(self.datasets.values()))
        n = {k: len(v) for k, v in ds.split(self.train, self.val).items()}
        print(f"\nBaselines: lookback {self.lookback} min, samples per split (first target/horizon): {n}")
        print("\nTest MAE (m/s):\n", self.table("test", "mae"))
        print("\nValidation MAE (m/s):\n", self.table("val", "mae"))
        print("\nBest baseline on test (the bar for XGBoost / LSTM):\n", self.best("test"))
        return self

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(out / "baseline_results.csv", index=False)
        print(f"  saved {out / 'baseline_results.csv'}")
        return self
