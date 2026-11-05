"""Step 1 after loading: clean, flag and mask the USM data.

Takes the output of load_usm.load() and, for each run, returns a dataframe on a complete
1-minute grid with these extra columns:

    missing_flow  q_vol is NaN (data gap, or minute missing from the export)
    zero_flow     q_vol is 0 (or below Q_ZERO). NaN is NOT counted as zero flow here
    snr_ok        all path SNRs above SNR_MIN
    vos_ok        all path VOS inside a plausible range
    valid         the rows models are allowed to use
    segment       id of the contiguous block of valid rows (NaN where not valid)

Nothing is deleted. Filter with df[df.valid] or use `segment` to build training windows
that never span a gap.


"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dev.load_usm import PATHS, USMLoader

# --- thresholds: starting values ---------------------------

FREQ = "1min"
Q_ZERO = 0.0            # |q_vol| <= Q_ZERO counts as zero flow (m3/h)

SNR_MIN = 31.0          # dB, per path
VOS_RANGE = (200.0, 600.0)  # m/s

MAX_FFILL_MIN = 30      # forward-fill gas composition at most this many minutes
MIN_SEGMENT_LEN = 10    # valid blocks shorter than this are marked invalid

COMPOSITION = ["rho_ref", "mw", "N2", "CO2", "C1", "C2", "C3",
               "iC4", "nC4", "iC5", "nC5", "C6plus"]
INDICATORS = ["dev_p1", "dev_p2", "dev_p3", "gc_disc"]

class Prepare:
    """Cleans one run: regular time grid, flags, `valid` mask and `segment` ids."""
 
    def __init__(self, df, name="run", freq=FREQ, q_zero=Q_ZERO, snr_min=SNR_MIN,
                 vos_range=VOS_RANGE, max_ffill=MAX_FFILL_MIN, min_segment_len=MIN_SEGMENT_LEN):
        self.raw = df                     
        self.df = df.copy()
        self.name = name
        self.freq = freq
        self.q_zero = q_zero
        self.snr_min = snr_min
        self.vos_range = vos_range
        self.max_ffill = max_ffill
        self.min_segment_len = min_segment_len
        self.n_added = 0 
 
 
    # ---- steps -------------------------------------------------------------------
    def regularise(self):
        """Put the data on a complete, evenly spaced time grid. Missing minutes become NaN rows."""
        df = self.df.copy()
        df.index = df.index.round(self.freq)
        dups = int(df.index.duplicated().sum())
   
        full = pd.date_range(df.index.min(), df.index.max(), freq=self.freq, name=df.index.name)
        self.n_added = len(full) - len(df)
        self.df = df.reindex(full)
        return self
 
    def fill_composition(self):
        """Gas composition updates every few minutes: carry the last value forward, but not forever."""
        comp = self._composition_cols()
        self.df[comp] = self.df[comp].ffill(limit=self.max_ffill)
        return self
 
    def add_indicators(self):
        """Recompute after reindexing so added rows are NaN, not stale."""
        self.df = USMLoader.add_indicators(self.df)
        return self
 
    def flag(self):
        """Per-row flags. Nothing is deleted."""
        df = self.df
        df["missing_flow"] = df["q_vol"].isna()
        df["zero_flow"] = df["q_vol"].abs().le(self.q_zero) & ~df["missing_flow"]
 
        snr_cols = [f"snr_{p}" for p in PATHS]
        vos_cols = [f"vos_{p}" for p in PATHS]
        # NaN counts as "not bad" here; NaN rows are caught by missing_flow / has_inputs
        df["snr_ok"] = ~(df[snr_cols] <= self.snr_min).any(axis=1)
        df["vos_ok"] = ~((df[vos_cols] < self.vos_range[0]) |
                         (df[vos_cols] > self.vos_range[1])).any(axis=1)
        return self
 
    def mask(self):
        """`valid` = rows models may use; `segment` = id of each unbroken block of valid rows."""
        df = self.df
        has_inputs = df[INDICATORS + ["p", "T"] + self._composition_cols()].notna().all(axis=1)
        df["valid"] = (~df.missing_flow & ~df.zero_flow & df.snr_ok & df.vos_ok & has_inputs)
 
        # drop valid blocks shorter than min_segment_len
        block = (df["valid"] != df["valid"].shift()).cumsum()
        seg_len = df.groupby(block)["valid"].transform("size")
        df.loc[df["valid"] & (seg_len < self.min_segment_len), "valid"] = False
 
        block = (df["valid"] != df["valid"].shift()).cumsum()
        df["segment"] = block.where(df["valid"]).rank(method="dense")   # 1, 2, 3, ... in time order
        return self
 
    def prepare(self):
        """Run all steps in order, starting from the raw data."""
        self.df = self.raw.copy()
        return (self.regularise()
                    .fill_composition()
                    .add_indicators()
                    .flag()
                    .mask())
 
    # ---- output ------------------------------------------------------------------
    @property
    def valid_df(self):
        """Only the rows models may use."""
        return self.df[self.df["valid"]]
 
    def report(self):
        df = self.df
        print(f"\n{self.name}: {len(df)} rows, {df.index.min()} -> {df.index.max()}")
        print(f"  minutes missing from the export (added by reindex): {self.n_added}")
        print(f"  missing flow: {int(df.missing_flow.sum())}   zero flow: {int(df.zero_flow.sum())}"
              f"   low SNR: {int((~df.snr_ok).sum())}   implausible VOS: {int((~df.vos_ok).sum())}")
        n_seg = int(df.segment.max()) if df.valid.any() else 0
        print(f"  valid rows: {int(df.valid.sum())} ({df.valid.mean():.1%}) in {n_seg} segment(s)")
        if df.valid.any():
            lens = df[df.valid].groupby("segment").size()
            print(f"  segment length (min): min {lens.min()}, median {int(lens.median())}, max {lens.max()}")
        steps = df.index.to_series().diff().dropna().value_counts()
        print(f"  time steps after regularising: {dict(steps)}")
        return self
 
    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / (self.name.replace(" ", "_") + "_prepared.csv")
        self.df.to_csv(fname)
        print(f"  saved {fname}")
        return self
 
    # ---- helpers -----------------------------------------------------------------
    def _composition_cols(self):
        return [c for c in COMPOSITION if c in self.df.columns]
 