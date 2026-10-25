#!/usr/bin/env python3
"""Plot the complete existing seed-0 EV-R101 grid (observed models only)."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5})
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "data" / "cub_ev_r101_rows.csv").open()))
by = {(r["method"], float(r["lambda_value"]), int(r["seed"])): r for r in rows}
STYLE = {"bger": ("BGER", "#1f6f4a", "o"), "aux": ("Aux-BCE", "#5a7fb5", "s"),
         "residual": ("Residual-BCE", "#8a6bb0", "D"), "cam_bce": ("CAM-BCE", "#b5653a", "^")}

fig, ax = plt.subplots(figsize=(3.33, 2.35))
for m, (label, color, marker) in STYLE.items():
    pts = [by[(m, lam, 0)] for lam in (0.125, 0.5, 2.0)]
    xs = [100 * float(r["Acc"]) for r in pts]
    ys = [100 * float(r["V2"]) for r in pts]
    ax.plot(xs, ys, "-", color=color, lw=0.9, alpha=0.6)
    ax.scatter(xs, ys, color=color, marker=marker, s=16, label=label, zorder=3)
    for r, x, y in zip(pts, xs, ys):
        ax.annotate(f"{float(r['lambda_value']):g}", (x, y), xytext=(2.5, 2.5),
                    textcoords="offset points", fontsize=6, color=color)
ce = by[("ce", 2.0, 0)]
ax.scatter([100 * float(ce["Acc"])], [100 * float(ce["V2"])], marker="x", color="black",
           s=22, label="CE", zorder=3)
ax.set_xlabel("Top-1 accuracy (%)")
ax.set_ylabel("MaxBoxAccV2 (%)")
ax.grid(alpha=0.25, lw=0.5)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
ax.legend(fontsize=6, frameon=False, loc="lower left", bbox_to_anchor=(0.50, 0.0))
ax.margins(x=0.08, y=0.06)
fig.tight_layout(pad=0.6)
out = ROOT / "figures" / "seed0_grid.pdf"
fig.savefig(out, metadata={'CreationDate': None, 'ModDate': None})
print("wrote", out)
