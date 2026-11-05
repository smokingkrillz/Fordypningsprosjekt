"""Step 3 of RQ2: a statistical model with process inputs, compared against the baselines.

LinearARX is a ridge-regularised linear autoregression with exogenous inputs: it predicts the
indicator `horizon` minutes ahead from its own recent values (lag_*) plus flow, pressure,
temperature, mole weight, the segment mean and the time since the segment started. By default it
predicts the CHANGE from the last value (y - lag_0) and adds that back, so at worst it behaves like
persistence. The ridge strength is picked on the validation split; the test split is only used
for the final scores.

ModelEvaluation reuses the datasets and splits from a BaselineEvaluation, so the comparison is on
exactly the same samples.

Usage:
    ev = BaselineEvaluation(p, horizons=(1, 5, 15), lookback=10).run()
    me = ModelEvaluation(ev).run().report()
    me.save("results")

Shell (prepared csv from prepare_oo.py):
    python models.py <prepared.csv> <output_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dev.baselines import BaselineEvaluation, mae, rmse


class Model:
    name = "model"

    def fit(self, train, val=None):
        return self

    def predict(self, X):
        raise NotImplementedError


class LinearARX(Model):
    name = "linear_arx"

    def __init__(self, alphas=(0.01, 0.1, 1, 10, 100, 1000), predict_delta=True):
        self.alphas = alphas
        self.predict_delta = predict_delta
        self.alpha_ = None          # ridge strength chosen on validation
        self.beta_ = None           # intercept + one weight per standardised input
        self.coef_ = None           # the weights as a Series (per +1 std of each input)

    # ---- helpers ---------------------------------------------------------------------------
    def _design(self, X):
        Z = (X[self.cols] - self.mu) / self.sd
        return np.column_stack([np.ones(len(Z)), Z.to_numpy()])

    def _base(self, X):
        return X["lag_0"].to_numpy() if self.predict_delta else 0.0

    @staticmethod
    def _ridge(D, t, alpha):
        P = np.eye(D.shape[1])
        P[0, 0] = 0.0                                   # do not penalise the intercept
        return np.linalg.lstsq(D.T @ D + alpha * P, D.T @ t, rcond=None)[0]

    # ---- fit / predict ---------------------------------------------------------------------------
    def fit(self, train, val=None):
        self.cols = list(train.X.columns)
        self.mu = train.X[self.cols].mean()
        self.sd = train.X[self.cols].std().replace(0, 1.0).fillna(1.0)
        D = self._design(train.X)
        t = train.y.to_numpy() - self._base(train.X)

        best = None                                     # (validation MAE, alpha, beta)
        for a in self.alphas:
            beta = self._ridge(D, t, a)
            if val is None:
                best = (0.0, a, beta)
                break
            score = mae(val.y, self._base(val.X) + self._design(val.X) @ beta)
            if best is None or score < best[0]:
                best = (score, a, beta)
        _, self.alpha_, self.beta_ = best
        self.coef_ = pd.Series(self.beta_[1:], index=self.cols)
        return self

    def predict(self, X):
        return self._base(X) + self._design(X) @ self.beta_


class ModelEvaluation:
    def __init__(self, baseline_eval, make_models=lambda: [LinearARX()]):
        self.ev = baseline_eval
        self.make_models = make_models
        self.results = pd.DataFrame()
        self.coefs = {}                                     # (target, horizon, model) -> Series

    def run(self):
        ev = self.ev
        if not ev.datasets:
            raise ValueError("run the BaselineEvaluation first")
        rows = []
        for (target, h), ds in ev.datasets.items():
            splits = ds.split(ev.train, ev.val)
            base = ev.results[(ev.results["target"] == target) & (ev.results["horizon"] == h)]
            # honest comparison: choose the best baseline on VALIDATION, report its TEST score
            val = base[base["split"] == "val"].set_index("baseline")["mae"]
            chosen = val.idxmin()
            chosen_test = base[(base["split"] == "test") & (base["baseline"] == chosen)]["mae"].iloc[0]
            pers_test = base[(base["split"] == "test") & (base["baseline"] == "persistence")]["mae"].iloc[0]
            for m in self.make_models():
                m.fit(splits["train"], splits["val"])
                if getattr(m, "coef_", None) is not None:
                    self.coefs[(target, h, m.name)] = m.coef_
                for name in ("val", "test"):
                    s = splits[name]
                    pred = m.predict(s.X)
                    score = mae(s.y, pred)
                    row = {"target": target, "horizon": h, "split": name, "model": m.name,
                           "n": len(s), "mae": score, "rmse": rmse(s.y, pred),
                           "alpha": getattr(m, "alpha_", np.nan)}
                    if name == "test":
                        row.update({"best_baseline": chosen, "baseline_mae": chosen_test,
                                    "skill_vs_persistence": 1 - score / pers_test,
                                    "skill_vs_best_baseline": 1 - score / chosen_test})
                    rows.append(row)
        self.results = pd.DataFrame(rows)
        return self

    def comparison(self):
        """Test MAE: model vs the best baseline (chosen on validation) vs persistence."""
        r = self.results[self.results["split"] == "test"]
        pers = self.ev.results[(self.ev.results["split"] == "test") &
                               (self.ev.results["baseline"] == "persistence")] \
            .set_index(["target", "horizon"])["mae"].rename("persistence_mae")
        out = r.set_index(["target", "horizon"])[["model", "mae", "best_baseline", "baseline_mae",
                                                  "skill_vs_persistence", "skill_vs_best_baseline"]]
        return out.join(pers)

    def top_coefficients(self, n=3):
        """The inputs with the largest effect (per +1 std), per target and horizon."""
        rows = []
        for (target, h, model), c in self.coefs.items():
            top = c.reindex(c.abs().sort_values(ascending=False).index)[:n]
            rows.append({"target": target, "horizon": h,
                         "top_inputs": ", ".join(f"{k} ({v:+.4f})" for k, v in top.items())})
        return pd.DataFrame(rows).set_index(["target", "horizon"])

    def report(self):
        pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 80,
                      "display.float_format", "{:.4f}".format)
        print("\nTest MAE (m/s): model vs best baseline (picked on validation) vs persistence")
        print(self.comparison())
        print("\nLargest effects per +1 std of the input (ridge, standardised):")
        print(self.top_coefficients())
        print("\nskill_vs_best_baseline > 0 means the model beats the best baseline. "
              "With few test samples, differences of a few percent are noise.")
        return self

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(out / "model_results.csv", index=False)
        self.comparison().to_csv(out / "model_vs_baselines.csv")
        print(f"  saved results to {out}")
        return self


if __name__ == "__main__":
    src, out_dir = sys.argv[1], sys.argv[2]
    df = pd.read_csv(src, index_col=0, parse_dates=True)
    ev = BaselineEvaluation(df).run()
    ModelEvaluation(ev).run().report().save(out_dir)
