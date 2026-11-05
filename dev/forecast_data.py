"""Step 1 of RQ2: turn a prepared run into a supervised forecasting dataset.

Task: given the last `lookback` minutes of one indicator plus the process inputs at the forecast
origin, predict that indicator `horizon` minutes later.

Windows are built inside segments only (a segment = unbroken block of valid, flowing minutes), so
no sample ever spans a zero-flow gap. Splits are chronological and keep whole segments together,
so no segment is shared between train, validation and test.

Columns of X
    lag_0 ... lag_{L-1}   indicator at the origin (lag_0) and the L-1 minutes before it
    seg_mean              mean of the indicator from the segment start up to the origin
    since_start           minutes since the segment's first used row
    q_vol, p, T, mw ...   process inputs at the origin

Usage:
    p = Prepare(runs["Vega 2st stage run 1"], name="run 1").prepare()
    ds = ForecastDataset(p, target="dev_p1", horizon=5, lookback=10)
    ds.check()
    splits = ds.split(train=0.6, val=0.2)      # {"train": Split, "val": Split, "test": Split}
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

INDICATORS = ["dev_p1", "dev_p2", "dev_p3", "gc_disc"]
COVARIATES = ["q_vol", "p", "T", "mw"]


@dataclass
class Split:
    name: str
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame          # segment, t_origin, t_target for each sample

    def __len__(self):
        return len(self.y)


class ForecastDataset:
    def __init__(self, source, target, horizon=1, lookback=10, covariates=COVARIATES, trim_min=1):
        df = source.df if hasattr(source, "df") else source
        for col in ("valid", "segment", target):
            if col not in df.columns:
                raise ValueError(f"missing column {col!r}; run Prepare.prepare() first")
        self.full = df
        self.target = target
        self.horizon = horizon
        self.lookback = lookback
        self.covariates = [c for c in covariates if c in df.columns]
        self.trim_min = trim_min
        self.X, self.y, self.meta = self._build()

    # ---- construction --------------------------------------------------------------------
    def _build(self):
        L, h = self.lookback, self.horizon
        v = self.full[self.full["valid"]]
        v = v[v.groupby("segment").cumcount() >= self.trim_min]     # drop restart transient
        g = v.groupby("segment")[self.target]

        feats = {f"lag_{k}": g.shift(k) for k in range(L)}           # shift never crosses segments
        feats["seg_mean"] = g.expanding().mean().reset_index(level=0, drop=True)
        X = pd.DataFrame(feats, index=v.index)
        for c in self.covariates:
            X[c] = v[c]
        X["since_start"] = v.groupby("segment").cumcount()

        y = g.shift(-h)
        meta = pd.DataFrame({"segment": v["segment"], "t_origin": v.index,
                             "t_target": v.index + pd.Timedelta(minutes=h)}, index=v.index)
        ok = X.notna().all(axis=1) & y.notna()
        return X[ok], y[ok].rename("y"), meta[ok]

    def __len__(self):
        return len(self.y)

    # ---- safety check ----------------------------------------------------------------------
    def check(self):
        """Verify against the raw time series that no sample crosses a gap or a segment."""
        one = pd.Timedelta(minutes=1)
        full = self.full
        t0, t1 = self.meta["t_origin"], self.meta["t_target"]
        tl = t0 - (self.lookback - 1) * one
        assert full.loc[t1, "valid"].all(), "a target lies outside valid rows"
        assert full.loc[tl, "valid"].all(), "a lookback start lies outside valid rows"
        seg = self.meta["segment"].to_numpy()
        assert (full.loc[t1, "segment"].to_numpy() == seg).all(), "target in another segment"
        assert (full.loc[tl, "segment"].to_numpy() == seg).all(), "lookback in another segment"
        assert np.allclose(full.loc[t1, self.target].to_numpy(), self.y.to_numpy()), "target mismatch"
        assert np.allclose(full.loc[tl, self.target].to_numpy(),
                           self.X[f"lag_{self.lookback - 1}"].to_numpy()), "lag mismatch"
        return self

    # ---- chronological split by whole segments ------------------------------------------------
    def split(self, train=0.6, val=0.2):
        counts = self.meta.groupby("segment").size().sort_index()    # segment ids increase with time
        total = counts.sum()
        before = counts.cumsum() - counts
        mid = (before + counts / 2) / total                          # where in time the segment sits
        which = np.where(mid < train, "train", np.where(mid < train + val, "val", "test"))
        owner = pd.Series(which, index=counts.index)
        part = self.meta["segment"].map(owner)

        out = {}
        for name in ("train", "val", "test"):
            m = (part == name).to_numpy()
            if not m.any():
                raise ValueError(f"{self.target} h={self.horizon}: the {name} split is empty "
                                 f"({len(counts)} segments, {total} samples); need more data or smaller fractions")
            out[name] = Split(name, self.X[m], self.y[m], self.meta[m])
        return out

    def summary(self):
        return (f"{self.target} h={self.horizon} L={self.lookback}: {len(self)} samples "
                f"in {self.meta['segment'].nunique()} segments")
