"""Step 3 of RQ2: forecasting models with process inputs, compared against the baselines.

Two models, both fed exactly the same inputs (see forecast_data.py):
    lag_0 ... lag_{L-1}, seg_mean, since_start, q_vol, p, T, mw

LinearARX
    Ridge-regularised linear autoregression with exogenous inputs. A weighted sum of the
    inputs, so every weight can be read directly. The ridge strength is picked on the
    validation split.

XGBoostForecaster
    Gradient-boosted decision trees. Can learn non-linear effects and interactions (for
    example "low flow AND rising deviation"). The number of trees is picked on the validation
    split with early stopping.

Both predict the CHANGE from the last value (y - lag_0) and add it back, so a model that
learns nothing useful falls back to persistence. The test split is only used for final scores.

ModelEvaluation reuses the datasets and splits from a BaselineEvaluation, so models and
baselines are scored on exactly the same samples.

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
from xgboost import XGBRegressor

from dev.baselines import BaselineEvaluation, mae, rmse


class Model:
    name = "model"

    def fit(self, train, val=None):
        return self

    def predict(self, X):
        raise NotImplementedError


# ---- Linear model -----------------------------------------------------------------------------
class LinearARX(Model):
    name = "linear_arx"

    def __init__(self, alphas=(0.01, 0.1, 1, 10, 100, 1000), predict_delta=True):
        self.alphas = alphas
        self.predict_delta = predict_delta
        self.alpha_ = None          # ridge strength chosen on validation
        self.beta_ = None           # intercept + one weight per standardised input
        self.coef_ = None           # the weights as a Series (per +1 std of each input)

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


# ---- Gradient-boosted trees -------------------------------------------------------------------
class XGBoostForecaster(Model):
    name = "xgboost"

    def __init__(self, predict_delta=True, max_trees=1000, early_stopping=50, **params):
        """params are passed to XGBRegressor and override the defaults below."""
        self.predict_delta = predict_delta
        self.max_trees = max_trees
        self.early_stopping = early_stopping
        self.params = {
            "max_depth": 4,             # shallow trees: less overfitting on small data
            "learning_rate": 0.05,      # small steps, many trees
            "subsample": 0.8,           # each tree sees 80 % of the samples
            "colsample_bytree": 0.8,    # and 80 % of the inputs
            "min_child_weight": 5,      # no leaf for fewer than ~5 samples
            "random_state": 0,
            **params,
        }
        self.n_trees_ = None        # number of trees kept after early stopping
        self.importance_ = None     # share of the model's gain from each input (sums to 1)

    def _base(self, X):
        return X["lag_0"].to_numpy() if self.predict_delta else 0.0

    def fit(self, train, val=None):
        self.cols = list(train.X.columns)
        t = train.y.to_numpy() - self._base(train.X)

        self.model_ = XGBRegressor(
            n_estimators=self.max_trees,
            early_stopping_rounds=self.early_stopping if val is not None else None,
            eval_metric="mae",
            **self.params,
        )
        if val is not None:
            # Stop adding trees once validation MAE has not improved for `early_stopping` rounds.
            t_val = val.y.to_numpy() - self._base(val.X)
            self.model_.fit(train.X[self.cols], t, eval_set=[(val.X[self.cols], t_val)], verbose=False)
            self.n_trees_ = self.model_.best_iteration + 1
        else:
            self.model_.fit(train.X[self.cols], t, verbose=False)
            self.n_trees_ = self.max_trees

        self.importance_ = pd.Series(self.model_.feature_importances_, index=self.cols)
        return self

    def predict(self, X):
        return self._base(X) + self.model_.predict(X[self.cols])


def default_models():
    return [LinearARX(), XGBoostForecaster()]


# ---- Evaluation -------------------------------------------------------------------------------
class ModelEvaluation:
    def __init__(self, baseline_eval, make_models=default_models):
        self.ev = baseline_eval
        self.make_models = make_models
        self.results = pd.DataFrame()
        self.coefs = {}             # (target, horizon, model) -> signed weights (linear models)
        self.importances = {}       # (target, horizon, model) -> importance shares (tree models)

    def run(self):
        ev = self.ev
        if not ev.datasets:
            raise ValueError("run the BaselineEvaluation first")
        rows = []
        for (target, h), ds in ev.datasets.items():
            splits = ds.split(ev.train, ev.val)
            base = ev.results[(ev.results["target"] == target) & (ev.results["horizon"] == h)]

            # Honest comparison: choose the best baseline on VALIDATION, report its TEST score.
            val = base[base["split"] == "val"].set_index("baseline")["mae"]
            chosen = val.idxmin()
            test = base[base["split"] == "test"].set_index("baseline")["mae"]
            chosen_test = test[chosen]
            pers_test = test.get("persistence", np.nan)

            for m in self.make_models():
                m.fit(splits["train"], splits["val"])
                if getattr(m, "coef_", None) is not None:
                    self.coefs[(target, h, m.name)] = m.coef_
                if getattr(m, "importance_", None) is not None:
                    self.importances[(target, h, m.name)] = m.importance_

                for name in ("val", "test"):
                    s = splits[name]
                    pred = m.predict(s.X)
                    score = mae(s.y, pred)
                    row = {"target": target, "horizon": h, "split": name, "model": m.name,
                           "n": len(s), "mae": score, "rmse": rmse(s.y, pred),
                           "alpha": getattr(m, "alpha_", np.nan),
                           "n_trees": getattr(m, "n_trees_", np.nan)}
                    if name == "test":
                        row.update({"best_baseline": chosen, "baseline_mae": chosen_test,
                                    "skill_vs_persistence": 1 - score / pers_test,
                                    "skill_vs_best_baseline": 1 - score / chosen_test})
                    rows.append(row)
        self.results = pd.DataFrame(rows)
        return self

    def comparison(self):
        """Test MAE per model vs the best baseline (chosen on validation) vs persistence."""
        r = self.results[self.results["split"] == "test"]
        pers = self.ev.results[(self.ev.results["split"] == "test") &
                               (self.ev.results["baseline"] == "persistence")] \
            .set_index(["target", "horizon"])["mae"].rename("persistence_mae")
        out = r.set_index(["target", "horizon"])[["model", "mae", "best_baseline", "baseline_mae",
                                                  "skill_vs_persistence", "skill_vs_best_baseline"]]
        return out.join(pers)

    def top_coefficients(self, n=3):
        """Linear model: the inputs with the largest weight (per +1 std), with sign."""
        rows = []
        for (target, h, model), c in self.coefs.items():
            top = c.reindex(c.abs().sort_values(ascending=False).index)[:n]
            rows.append({"target": target, "horizon": h, "model": model,
                         "top_inputs": ", ".join(f"{k} ({v:+.4f})" for k, v in top.items())})
        return pd.DataFrame(rows).set_index(["target", "horizon", "model"])

    def top_importances(self, n=3):
        """Tree model: the inputs the trees used most (share of total gain, no sign)."""
        rows = []
        for (target, h, model), imp in self.importances.items():
            top = imp.sort_values(ascending=False)[:n]
            rows.append({"target": target, "horizon": h, "model": model,
                         "top_inputs": ", ".join(f"{k} ({v:.0%})" for k, v in top.items())})
        return pd.DataFrame(rows).set_index(["target", "horizon", "model"])

    def report(self):
        with pd.option_context("display.width", 220, "display.max_columns", 20,
                               "display.max_colwidth", 80, "display.float_format", "{:.4f}".format):
            print("\nTest MAE (m/s): models vs best baseline (picked on validation) vs persistence")
            print(self.comparison())
            if self.coefs:
                print("\nLinear ARX, largest weights per +1 std of the input:")
                print(self.top_coefficients())
            if self.importances:
                print("\nXGBoost, most used inputs (share of gain):")
                print(self.top_importances())
            print("\nskill_vs_best_baseline > 0 means the model beats the best baseline. "
                  "With few test samples, differences of a few percent are noise.")
        return self

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(out / "model_results.csv", index=False)
        self.comparison().to_csv(out / "model_vs_baselines.csv")
        if self.importances:
            pd.DataFrame(self.importances).T.to_csv(out / "xgboost_importances.csv")
        print(f"  saved results to {out}")
        return self


if __name__ == "__main__":
    src, out_dir = sys.argv[1], sys.argv[2]
    df = pd.read_csv(src, index_col=0, parse_dates=True)
    ev = BaselineEvaluation(df).run()
    ModelEvaluation(ev).run().report().save(out_dir)