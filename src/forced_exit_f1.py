"""
forced_exit_f1.py — accuracy, F1 and AUC at each forced-exit depth.

Reads the cached score files only. No retraining, no GPU.

    python forced_exit_f1.py --tag bert-cls
    python forced_exit_f1.py --tag modernbert-cls

Writes outputs/probe/{tag}-forced-exit-f1.csv and prints a table ready to
paste into a slide.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--probe_dir", default="../outputs/probe")
ap.add_argument("--depths", type=float, nargs="+",
                default=[0.0, 0.25, 0.50, 0.75, 1.0])
a = ap.parse_args()

runs = []
for s in a.seeds:
    f = Path(a.probe_dir) / f"{a.tag}-scores-s{s}.npz"
    if not f.exists():
        print(f"  missing {f} (skipped)")
        continue
    d = np.load(f, allow_pickle=True)
    runs.append({"seed": s, "scores": d["scores"], "preds": d["preds"],
                 "labels": d["labels"].astype(int), "n_layers": int(d["n_layers"])})
if not runs:
    raise SystemExit(f"no score files for tag '{a.tag}'")

L = runs[0]["n_layers"]
print(f"\n{a.tag} — {L} layers, {len(runs)} seeds, {len(runs[0]['labels'])} test posts\n")

rows = []
for frac in a.depths:
    layer = max(1, int(round(frac * L)))        # never the embedding layer
    acc, f1b, f1m, auc = [], [], [], []
    for r in runs:
        y = r["labels"]
        p = r["preds"][:, layer]
        s = r["scores"][:, layer]
        acc.append(accuracy_score(y, p))
        f1b.append(f1_score(y, p, average="binary", zero_division=0))
        f1m.append(f1_score(y, p, average="macro",  zero_division=0))
        auc.append(roc_auc_score(y, s))
    rows.append({
        "depth": f"{int(frac*100)}%", "layer": layer,
        "accuracy": np.mean(acc), "accuracy_std": np.std(acc),
        "f1_binary": np.mean(f1b), "f1_binary_std": np.std(f1b),
        "f1_macro": np.mean(f1m), "f1_macro_std": np.std(f1m),
        "auc": np.mean(auc), "auc_std": np.std(auc),
    })

df = pd.DataFrame(rows)
out = Path(a.probe_dir) / f"{a.tag}-forced-exit-f1.csv"
df.to_csv(out, index=False)

print(f"{'depth':>6} {'layer':>6} {'accuracy':>16} {'F1 (hate)':>16} "
      f"{'F1 macro':>16} {'AUC':>16}")
for _, r in df.iterrows():
    print(f"{r['depth']:>6} {int(r['layer']):>6} "
          f"{r['accuracy']*100:>9.2f} ±{r['accuracy_std']*100:>4.2f} "
          f"{r['f1_binary']*100:>9.2f} ±{r['f1_binary_std']*100:>4.2f} "
          f"{r['f1_macro']*100:>9.2f} ±{r['f1_macro_std']*100:>4.2f} "
          f"{r['auc']*100:>9.2f} ±{r['auc_std']*100:>4.2f}")

print(f"\nF1 (hate) is the positive class only; F1 macro averages both classes.")
print(f"Saved {out}")
