import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import load_usm

SRC, OUT = sys.argv[1], sys.argv[2]
runs, gas = load_usm.load(SRC)
names = list(runs)
titles = {names[0]: "Run 1 - low flow", names[1]: "Run 2 - zero flow"}
C = {"p1": "#2a78d6", "p2": "#eb6834", "p3": "#1baf7a"}   # categorical slots 1-3 (validated all-pairs)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

rows = [
    ("Volume flow (m³/h)", lambda d, ax: ax.plot(d.index, d.q_vol, color=INK, lw=1.5, marker="o", ms=4, label="q_vol")),
    ("Measured VOS (m/s)", None),
    ("Cross-path VOS deviation (m/s)", None),
    ("SNR (dB)", None),
    ("GC discrepancy (m/s)", lambda d, ax: ax.plot(d.index, d.gc_disc, color=INK, lw=1.5, marker="o", ms=4, label="VOS_GC - VOS_avg")),
    ("Methane C1 (mol%)", lambda d, ax: ax.step(d.index, d.C1, where="post", color=INK, lw=1.5, label="C1")),
]
fig, axes = plt.subplots(len(rows), 2, figsize=(12, 14), sharex="col", constrained_layout=True)
for j, n in enumerate(names):
    d = runs[n]
    for i, (lab, fn) in enumerate(rows):
        ax = axes[i, j]
        if lab.startswith("Measured VOS"):
            for p in load_usm.PATHS:
                ax.plot(d.index, d[f"vos_{p}"], color=C[p], lw=1.5, marker="o", ms=3, label=f"path {p[1]}")
            ax.plot(d.index, d.vos_gc, color=INK, lw=1.2, ls="--", label="from GC")
        elif lab.startswith("Cross-path"):
            for p in load_usm.PATHS:
                ax.plot(d.index, d[f"dev_{p}"], color=C[p], lw=1.5, marker="o", ms=3, label=f"path {p[1]}")
            ax.axhline(0, color=MUTED, lw=0.8)
        elif lab.startswith("SNR"):
            for p in load_usm.PATHS:
                ax.plot(d.index, d[f"snr_{p}"], color=C[p], lw=1.5, marker="o", ms=3, label=f"path {p[1]}")
        else:
            fn(d, ax)
        if j == 0:
            ax.set_ylabel(lab, fontsize=9, color=MUTED)
        if i == 0:
            ax.set_title(titles[n], fontsize=11, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
from matplotlib.lines import Line2D
h = [Line2D([0], [0], color=C[p], lw=1.5, marker="o", ms=3, label=f"path {p[1]}") for p in load_usm.PATHS]
h.append(Line2D([0], [0], color=INK, lw=1.2, ls="--", label="VOS from GC (dashed, VOS panel only)"))
fig.legend(handles=h, loc="upper right", ncol=4, frameon=False, fontsize=9)
fig.suptitle("Vega 2nd stage USM - raw sample, 2026-10-01 (1-min data)", fontsize=12, x=0.01, ha="left")
fig.savefig(OUT, dpi=130)
print("saved", OUT)
