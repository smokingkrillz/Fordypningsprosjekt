"""RQ1: which health indicators stay stable under normal operation?

StabilityAnalysis takes a prepared run (a Prepare object from prepare_oo.py, or a dataframe that
has the `valid` and `segment` columns) and produces:

    describe()       spread, drift, and within- vs between-segment variation per indicator
    dependence()     Spearman correlation of each indicator with the process variables
    effects()        regression effect of +1 standard deviation of each process variable,
                     in the indicator's own units (m/s)
    fit()            R2 and unexplained spread of that regression
    step_response()  indicator change across each gas-composition update
    summary()        one row per indicator with a heuristic verdict
    plot()           time series + scatter against each process variable

Usage:
    p = Prepare(runs["Vega 2st stage run 1"], name="run 1").prepare()
    sa = StabilityAnalysis(p)
    sa.report()
    sa.plot("run1_stability.png")

Shell:
    python stability.py <workbook.xlsx> <output_dir>

Read the results with care: correlation is not causation, process variables move together
(see collinearity()), and a short sample cannot say anything about variables that barely vary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INDICATORS = ["dev_p1", "dev_p2", "dev_p3", "gc_disc"]
PROCESS_VARS = ["p", "T", "q_vol", "mw"] 
TRIM_MIN = 1           # drop the first minutes of every segment (restart transient)
RHO_THRESH = 0.5        # |Spearman rho| at or above this = "depends on"
R2_THRESH = 0.3         # regression R2 at or above this = "process-dependent"
MIN_ROWS = 28           # fewer valid rows than this -> no verdict
MW_TOL = 1e-6           # change in mole weight that counts as a composition update


class StabilityAnalysis:
    def __init__(self, source, name=None, indicators=INDICATORS, process_vars=PROCESS_VARS,
                 trim_min=TRIM_MIN, rho_thresh=RHO_THRESH, r2_thresh=R2_THRESH,
                 fault_shift=None):
        """fault_shift: optional {indicator: size of the fault you want to detect}, same units
        as the indicator. Adds a detectability ratio (shift / unexplained spread) to summary()."""
        df = source.df if hasattr(source, "df") else source
        if "valid" not in df.columns or "segment" not in df.columns:
            raise ValueError("expected `valid` and `segment` columns; run Prepare.prepare() first")
        self.name = name or getattr(source, "name", "run")
        self.full = df
        self.indicators = [c for c in indicators if c in df.columns]
        self.trim_min = trim_min
        self.rho_thresh = rho_thresh
        self.r2_thresh = r2_thresh
        self.fault_shift = fault_shift or {}

        self.data = self._trimmed(df)
        wanted = [c for c in process_vars if c in df.columns]
        self.process_vars = [c for c in wanted if self.data[c].nunique() > 1 and self.data[c].std() > 0]
        self.constant_vars = [c for c in wanted if c not in self.process_vars]

    # ---- data ------------------------------------------------------------------------
    def _trimmed(self, df):
        v = df[df["valid"]]
        pos = v.groupby("segment").cumcount()
        return v[pos >= self.trim_min]

    @property
    def enough_data(self):
        print("enough data", len(self.data) >= MIN_ROWS)
        return len(self.data) >= MIN_ROWS

    # ---- 1. baseline -------------------------------------------------------------------
    def describe(self):
        rows = []
        for ind in self.indicators:
            s = self.data[ind].dropna()
            g = self.data.groupby("segment")[ind]
            n, var = g.count(), g.var(ddof=1)
            ok = n > 1
            within = np.sqrt((var[ok] * (n[ok] - 1)).sum() / (n[ok] - 1).sum()) if ok.any() else np.nan
            between = g.mean().std(ddof=1) if g.ngroups > 1 else np.nan
            days = (s.index - s.index[0]).total_seconds().to_numpy() / 86400 if len(s) else np.array([])
            drift = np.polyfit(days, s.to_numpy(), 1)[0] if len(s) > 2 and np.ptp(days) > 0 else np.nan
            rows.append({"indicator": ind, "n": len(s), "segments": g.ngroups,
                         "mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max(),
                         "std_within_segment": within, "std_between_segments": between,
                         "drift_per_day": drift})
        return pd.DataFrame(rows).set_index("indicator")

    # ---- 2. dependence on process conditions ---------------------------------------------
    def dependence(self):
        """Spearman correlation: rows = indicators, columns = process variables."""
        out = pd.DataFrame(index=self.indicators, columns=self.process_vars, dtype=float)
        for ind in self.indicators:
            for var in self.process_vars:
                out.loc[ind, var] = self.data[ind].corr(self.data[var], method="spearman")
        return out

    def _ols(self, ind):
        cols = self.process_vars
        if not cols:
            return None
        d = self.data[[ind] + cols].dropna()
        if len(d) < len(cols) + 5:
            return None
        z = (d[cols] - d[cols].mean()) / d[cols].std()
        X = np.column_stack([np.ones(len(d)), z.to_numpy()])
        y = d[ind].to_numpy()
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        sst = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / sst if sst > 0 else np.nan
        resid_std = np.sqrt((resid ** 2).sum() / (len(d) - X.shape[1]))
        return dict(zip(cols, beta[1:])), r2, resid_std

    def effects(self):
        """Change in the indicator for +1 standard deviation of each process variable (all
        variables fitted together, so shared movement is partly separated)."""
        out = pd.DataFrame(index=self.indicators, columns=self.process_vars, dtype=float)
        for ind in self.indicators:
            res = self._ols(ind)
            if res:
                for var, b in res[0].items():
                    out.loc[ind, var] = b
        return out

    def fit(self):
        rows = {}
        for ind in self.indicators:
            res = self._ols(ind)
            rows[ind] = {"r2": res[1], "resid_std": res[2]} if res else {"r2": np.nan, "resid_std": np.nan}
        return pd.DataFrame(rows).T

    def collinearity(self, thresh=0.9):
        """Pairs of process variables that move together; their separate effects are unreliable."""
        c = self.data[self.process_vars].corr(method="spearman")
        pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(c.columns) for b in c.columns[i + 1:]
                 if abs(c.loc[a, b]) >= thresh]
        return pd.DataFrame(pairs, columns=["var_a", "var_b", "rho"])

    # ---- 3. composition steps -------------------------------------------------------------
    def step_response(self, window=5, mw_tol=MW_TOL):
        """Mean indicator after minus before each composition update, inside one segment."""
        df = self.full
        if "mw" not in df.columns:
            return pd.DataFrame(columns=["time", "d_mw", "indicator", "delta"])
        events = df.index[df["mw"].diff().abs() > mw_tol]
        v = df[df["valid"]]
        w, one = pd.Timedelta(minutes=window), pd.Timedelta(minutes=1)
        rows = []
        for t in events:
            if not df.loc[t, "valid"]:
                continue
            s = v[v["segment"] == df.loc[t, "segment"]]
            before, after = s.loc[t - w:t - one], s.loc[t:t + w - one]
            if len(before) < 2 or len(after) < 2:
                continue
            for ind in self.indicators:
                rows.append({"time": t, "d_mw": df["mw"].diff().loc[t], "indicator": ind,
                             "delta": after[ind].mean() - before[ind].mean()})
        return pd.DataFrame(rows, columns=["time", "d_mw", "indicator", "delta"])

    def step_summary(self, **kw):
        st = self.step_response(**kw)
        if st.empty:
            return pd.DataFrame(columns=["events", "median_abs_delta", "max_abs_delta"])
        g = st.assign(a=st["delta"].abs()).groupby("indicator")["a"]
        return pd.DataFrame({"events": g.count(), "median_abs_delta": g.median(),
                             "max_abs_delta": g.max()})

    # ---- 4. verdict -----------------------------------------------------------------------
    def summary(self):
        desc, rho, fit = self.describe(), self.dependence(), self.fit()
        rows = []
        for ind in self.indicators:
            depends = [v for v in self.process_vars if abs(rho.loc[ind, v]) >= self.rho_thresh]
            r2 = fit.loc[ind, "r2"]
            if not self.enough_data:
                verdict = "insufficient data"
            elif depends or (pd.notna(r2) and r2 >= self.r2_thresh):
                verdict = "process-dependent"
            else:
                verdict = "stable"
            row = {"indicator": ind, "std": desc.loc[ind, "std"],
                   "std_within_segment": desc.loc[ind, "std_within_segment"],
                   "drift_per_day": desc.loc[ind, "drift_per_day"],
                   "r2": r2, "unexplained_std": fit.loc[ind, "resid_std"],
                   "depends_on": ", ".join(depends) or "-", "verdict": verdict}
            if ind in self.fault_shift:
                spread = row["unexplained_std"] if pd.notna(row["unexplained_std"]) else row["std"]
                row["fault_shift"] = self.fault_shift[ind]
                row["shift_over_spread"] = self.fault_shift[ind] / spread if spread else np.nan
            rows.append(row)
        return pd.DataFrame(rows).set_index("indicator")

    # ---- output ---------------------------------------------------------------------------
    def report(self):
        pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format)
        print(f"\n=== Stability: {self.name} ===")
        print(f"valid rows used: {len(self.data)} (first {self.trim_min} min of each segment dropped), "
              f"segments: {self.data['segment'].nunique()}")
        if not self.enough_data:
            print(f"  fewer than {MIN_ROWS} usable rows: no verdicts, results below are not reliable")
        if self.constant_vars:
            print(f"  constant in this data, so their effect cannot be assessed: {self.constant_vars}")
        print("\nBaseline:\n", self.describe())
        print("\nSpearman correlation with process variables:\n", self.dependence())
        print("\nEffect of +1 std of each variable (indicator units):\n", self.effects())
        col = self.collinearity()
        if not col.empty:
            print("\nWarning: these variables move together, separate effects are unreliable:\n", col)
        st = self.step_summary()
        if not st.empty:
            print("\nChange across composition updates:\n", st)
        print("\nSummary (heuristic verdict):\n", self.summary())
        return self

    def plot(self, path):

        cols = self.process_vars
        fig, axes = plt.subplots(len(self.indicators), 1 + len(cols),
                                 figsize=(3.4 * (1 + len(cols)), 2.4 * len(self.indicators)),
                                 squeeze=False, constrained_layout=True)
        rho = self.dependence()
        ink, muted, blue = "#0b0b0b", "#52514e", "#2a78d6"
        for i, ind in enumerate(self.indicators):
            ax = axes[i, 0]
            ax.plot(self.data.index, self.data[ind], ".", ms=2, color=blue)
            ax.set_ylabel(ind, color=ink)
            ax.set_title("over time" if i == 0 else "", fontsize=9, color=muted, loc="left")
            for j, var in enumerate(cols, start=1):
                ax = axes[i, j]
                ax.plot(self.data[var], self.data[ind], ".", ms=2, color=blue, alpha=0.6)
                ax.set_title(f"vs {var}   rho = {rho.loc[ind, var]:+.2f}", fontsize=9, color=muted, loc="left")
            for ax in axes[i]:
                ax.grid(True, color="#e6e5e1", lw=0.6)
                ax.tick_params(labelsize=7, colors=muted)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
        fig.suptitle(f"{self.name}: indicators vs time and process variables (valid, flowing rows)",
                     fontsize=11, x=0.01, ha="left")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return self

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = self.name.replace(" ", "_")
        self.summary().to_csv(out / f"{stem}_stability_summary.csv")
        self.dependence().to_csv(out / f"{stem}_dependence.csv")
        self.effects().to_csv(out / f"{stem}_effects.csv")
        self.step_response().to_csv(out / f"{stem}_step_response.csv", index=False)
        if len(self.data):
            self.plot(out / f"{stem}_stability.png")
        print(f"  saved results to {out}")
        return self
