"""
plot_maxgap.py — replot the max-gap figure from the saved summary CSV.
Instant: no bootstrap, no permutation, no model.

    python plot_maxgap.py --tag bert-cls
    python plot_maxgap.py --tag modernbert-cls
    python plot_maxgap.py --tag bert-cls --bands noise    # keep the grey band
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--gap_dir", default="../outputs/gap")
ap.add_argument("--bands", choices=["none", "noise", "ci", "both"], default="none")
ap.add_argument("--suffix", default="-plain")
a = ap.parse_args()

d = Path(a.gap_dir)
df = pd.read_csv(d / f"{a.tag}-gap-summary.csv")
L = df["layer"].values

fig, ax = plt.subplots(figsize=(9, 5.2))

if a.bands in ("noise", "both"):
    ax.fill_between(L, 0, df["null_p95"] * 100, color="grey", alpha=.25,
                    label="expected with no real subgroup difference (95%)")
if a.bands in ("ci", "both"):
    ax.fill_between(L, df["ci_lo"] * 100, df["ci_hi"] * 100,
                    color="#D1495B", alpha=.18, label="95% bootstrap CI")

ax.plot(L, df["max_gap"] * 100, marker="o", ms=5, lw=2.2,
        color="#D1495B", label="max gap below mean AUC")

for _, r in df.iterrows():
    ax.annotate(r["worst_group"], (r["layer"], r["max_gap"] * 100), fontsize=7.5,
                xytext=(0, 8), textcoords="offset points", ha="center", color="#5A7184")

ax.set_xlabel("layer")
ax.set_ylabel("max gap below mean AUC (pp)")
ax.set_title(f"Largest subgroup shortfall by layer — {a.tag}", fontsize=12)
ax.set_xticks(L)
ax.set_ylim(0, max(df["max_gap"].max() * 100 * 1.35, 5))
ax.grid(alpha=.3)
if a.bands != "none":
    ax.legend(fontsize=8)
fig.tight_layout()

out = d / f"{a.tag}-fig3-maxgap{a.suffix}.png"
fig.savefig(out, dpi=160)
print("saved", out)
