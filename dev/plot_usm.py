"""Plots for the Vega 2nd stage USM runs: one PNG per panel per run, plus a gas-composition overview."""
import os
import re
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dev.load_usm import PATHS

OUT_DIR = "plots"
DPI = 300
RUN_TITLES = ["Run 1 - low flow", "Run 2 - zero flow"]    # matched to the run sheets in order
C = {"p1": "#2a78d6", "p2": "#eb6834", "p3": "#1baf7a"}   # categorical slots 1-3 (validated all-pairs)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SUBTITLE = "USM Gas - raw sample, (1-min data)"
# gas components in the order they appear in GAS_RENAME, with readable labels
GASES = {
    "C1": "Methane C1", "C2": "Ethane C2", "C3": "Propane C3",
    "iC4": "i-Butane iC4", "nC4": "n-Butane nC4",
    "iC5": "i-Pentane iC5", "nC5": "n-Pentane nC5", "C6plus": "Hexane+ C6+",
    "N2": "Nitrogen N2", "CO2": "Carbon dioxide CO2",
}


class USMPlotter:
    """Makes the plots for each run DataFrame produced by USMLoader.

    Usage:
        runs, gas = USMLoader().load()
        plotter = USMPlotter(runs)
        plotter.plot_all()
    Files are written to <out_dir>/<run title>/<panel>.png.
    """

    def __init__(self, runs, out_dir=OUT_DIR, dpi=DPI):
        self.runs = runs
        self.out_dir = out_dir
        self.dpi = dpi
        # sheet name -> readable title; falls back to the sheet name if there are more runs than titles
        self.titles = {n: (RUN_TITLES[i] if i < len(RUN_TITLES) else n) for i, n in enumerate(runs)}

    @staticmethod
    def slug(s):
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

    @staticmethod
    def style(ax):
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # ---- panel drawing functions: each takes (d, ax) ----

    @staticmethod
    def plot_paths(d, ax, col):
        for p in PATHS:
            ax.plot(d.index, d[f"{col}_{p}"], color=C[p], lw=1.2, label=f"path {p[1]}")

    @staticmethod
    def plot_q_vol(d, ax):
        ax.plot(d.index, d.q_vol, color=INK, lw=1.2)

    def plot_vos(self, d, ax):
        self.plot_paths(d, ax, "vos")
        ax.plot(d.index, d.vos_gc, color=INK, lw=1.2, ls="--", label="from GC")

    def plot_dev(self, d, ax):
        self.plot_paths(d, ax, "dev")
        ax.axhline(0, color=MUTED, lw=0.8)

    def plot_snr(self, d, ax):
        self.plot_paths(d, ax, "snr")

    @staticmethod
    def plot_gc_disc(d, ax):
        ax.plot(d.index, d.gc_disc, color=INK, lw=1.2)

    @staticmethod
    def plot_gas(d, ax, g):
        ax.step(d.index, d[g], where="post", color=INK, lw=1.5)

    def panels(self):
        """(file name, y-label, plot function, show legend)"""
        panels = [
            ("q_vol", "Volume flow (m³/h)", self.plot_q_vol, False),
            ("vos", "Measured VOS (m/s)", self.plot_vos, True),
            ("vos_dev", "Cross-path VOS deviation (m/s)", self.plot_dev, True),
            ("snr", "SNR (dB)", self.plot_snr, True),
            ("gc_disc", "GC discrepancy, VOS_GC - VOS_avg (m/s)", self.plot_gc_disc, False),
        ]
        for g, lab in GASES.items():
            panels.append((f"gas_{g}", f"{lab} (mol%)", partial(self.plot_gas, g=g), False))
        return panels

    @staticmethod
    def has_gas(d, g):
        return g in d and not d[g].isna().all()

    def save(self, fig, path):
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        print("saved", path)

    def run_dir(self, name):
        path = os.path.join(self.out_dir, self.slug(self.titles[name]))
        os.makedirs(path, exist_ok=True)
        return path

    def plot_panel(self, name, fname, lab, fn, legend):
        """One figure for one panel of one run."""
        d = self.runs[name]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        fn(d, ax)
        self.style(ax)
        ax.set_ylabel(lab, fontsize=9, color=MUTED)
        ax.set_title(f"{self.titles[name]} - {lab}", fontsize=11, color=INK, loc="left")
        fig.suptitle(SUBTITLE, fontsize=9, color=MUTED, x=0.01, ha="left")
        if legend:
            ax.legend(frameon=False, fontsize=9, loc="best")
        self.save(fig, os.path.join(self.run_dir(name), f"{fname}.png"))

    def plot_gas_all(self, name):
        """All gas components in one picture: small multiples, each with its own y-scale."""
        d = self.runs[name]
        present = [g for g in GASES if self.has_gas(d, g)]
        if not present:
            return
        ncols = 3
        nrows = -(-len(present) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows), sharex=True,
                                 constrained_layout=True, squeeze=False)
        axes = axes.ravel()
        for ax, g in zip(axes, present):
            self.plot_gas(d, ax, g)
            ax.set_title(f"{GASES[g]} (mol%)", fontsize=10, color=INK, loc="left")
            self.style(ax)
            ax.tick_params(labelbottom=True)
        for ax in axes[len(present):]:
            ax.set_visible(False)
        fig.suptitle(f"{self.titles[name]} - gas composition\n{SUBTITLE}", fontsize=12, x=0.01, ha="left")
        self.save(fig, os.path.join(self.run_dir(name), "gas_all.png"))

    def plot_run(self, name):
        d = self.runs[name]
        for fname, lab, fn, legend in self.panels():
            if fname.startswith("gas_") and not self.has_gas(d, fname[4:]):
                continue
            self.plot_panel(name, fname, lab, fn, legend)
        self.plot_gas_all(name)

    def plot_all(self):
        for name in self.runs:
            self.plot_run(name)
