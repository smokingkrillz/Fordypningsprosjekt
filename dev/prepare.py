"""Step 1 after loading: clean, flag and mask the USM data.

Takes the output of load_usm.load() and, for each run, returns a dataframe on a complete
1-minute grid with these extra columns:

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

SNR_MIN = 25          # dB, per path
VOS_RANGE = (200.0, 600.0)  # m/s

MAX_FFILL_MIN = 30      # forward-fill gas composition at most this many minutes
MIN_SEGMENT_LEN = 10    # valid blocks shorter than this are marked invalid

COMPOSITION = ["rho_ref", "mw", "N2", "CO2", "C1", "C2", "C3",
               "iC4", "nC4", "iC5", "nC5", "C6plus"]
INDICATORS = ["dev_p1", "dev_p2", "dev_p3", "gc_disc"]

class Prepare:
    """Cleans one run: regular time grid, flags, `valid` mask and `segment` ids."""
 
    def __init__(self, df, name="meter_data", freq=FREQ, q_zero=Q_ZERO, snr_min=SNR_MIN,
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
 
 
    def add_indicators(self):
        """Recompute after reindexing so added rows are NaN, not stale."""
        self.df = USMLoader.add_indicators(self.df)
        return self
 
    def flag(self):
        """Per-row flags. Nothing is deleted.
            Checks where there is zero-flow,
            snr values are ok,
            vos values ok.
        """
        df = self.df
        df["zero_flow"] = df["q_vol"].abs().le(self.q_zero)
 
        snr_cols = [f"snr_{p}" for p in PATHS]
        vos_cols = [f"vos_{p}" for p in PATHS]
        # NaN counts as "not bad" here; NaN rows are caught by  has_inputs
        df["snr_ok"] = ~(df[snr_cols] <= self.snr_min).any(axis=1)
        df["vos_ok"] = ~((df[vos_cols] < self.vos_range[0]) |
                         (df[vos_cols] > self.vos_range[1])).any(axis=1)
        return self
 
    def mask(self):
        """`valid` = rows models may use; `segment` = id of each unbroken block of valid rows."""
        df = self.df
        has_inputs = df[INDICATORS + ["p", "T"] + self._composition_cols()].notna().all(axis=1)
        # Valid if flow is ok, no zero-flow 
        df["valid"] = (~df.zero_flow & df.snr_ok & df.vos_ok & has_inputs)
 
        # Step 2: find every run of consecutive valid rows
        # A run is stored as (start, end), where end is the first row AFTER the run.
        valid_flags = df["valid"].to_numpy()
        runs = []
        run_start = None                        

        for row_number, is_valid in enumerate(valid_flags):
            if is_valid and run_start is None:
                run_start = row_number           # a run begins here
            elif not is_valid and run_start is not None:
                runs.append((run_start, row_number))   # the run just ended
                run_start = None

        if run_start is not None:                # the data ended while still inside a run
            runs.append((run_start, len(valid_flags)))

        # Step 3: keep only long runs, and give each one a number 1, 2, 3, ...
        new_valid = np.zeros(len(df), dtype=bool)      # start with everything invalid
        new_segment = np.full(len(df), np.nan)         # start with everything NaN
        segment_number = 0

        for start, end in runs:
            run_length = end - start
            if run_length >= self.min_segment_len:
                segment_number += 1
                new_valid[start:end] = True
                new_segment[start:end] = segment_number

        df["valid"] = new_valid
        df["segment"] = new_segment
        return self
 
    def prepare(self):
        """Run all steps in order, starting from the raw data."""
        self.df = self.raw.copy()

        return (self.add_indicators()
                    .flag()
                    .mask())
 
    # ---- output ------------------------------------------------------------------
    @property
    def valid_df(self):
        """Only the rows models may use."""
        return self.df[self.df["valid"]]
 
    def report(self):
        "makes a report."
        df = self.df
        print(f"\n{self.name}: {len(df)} rows, {df.index.min()} -> {df.index.max()}")
        print(f"   low SNR: {int((~df.snr_ok).sum())}   implausible VOS: {int((~df.vos_ok).sum())}")

        n_seg = int(df.segment.max()) if df.valid.any() else 0
        print(f"  valid rows: {int(df.valid.sum())} ({df.valid.mean():.1%}) in {n_seg} segment(s)")
        if df.valid.any():
            lens = df[df.valid].groupby("segment").size()
            print(f"  segment length (min): min {lens.min()}, median {int(lens.median())}, max {lens.max()}")

        
        return self
 
    def save(self, out_dir):
        """saves the prepared csv. Input is the output directory."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / (self.name.replace(" ", "_") + "_prepared.csv")
        self.df.to_csv(fname)
        print(f"  saved {fname}")
        return self
 
    # ---- helpers -----------------------------------------------------------------
    def _composition_cols(self):
        """Returns the name of composition columns"""
        return [c for c in COMPOSITION if c in self.df.columns]
 