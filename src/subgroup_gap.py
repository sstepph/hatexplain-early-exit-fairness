"""
subgroup_gap.py
Per-subgroup AUC at every layer, each subgroup's distance below the mean AUC,
and the maximum of that distance per layer (Julien's proposed measure).

Needs only the cached score files — no GPU, no retraining.

Usage:
    python subgroup_gap.py --tag bert-cls
    python subgroup_gap.py --tag modernbert-cls
    python subgroup_gap.py --tag bert-cls --nboot 1000 --nperm 500

Outputs (in outputs/gap/):
    {tag}-subgroup-auc.csv        per layer x subgroup: AUC mean/std over seeds
    {tag}-gap-summary.csv         per layer: mean AUC, max gap, CI, null band
    {tag}-fig1-subgroup-auc.png   one line per subgroup, AUC vs layer
    {tag}-fig2-gap.png            distance below mean AUC vs layer
    {tag}-fig3-maxgap.png         max gap vs layer, with noise band
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)


def safe_auc(y, s):
    """AUC, or nan if only one class is present."""
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


def load_runs(tag, seeds, probe_dir):
    runs = []
    for s in seeds:
        f = Path(probe_dir) / f"{tag}-scores-s{s}.npz"
        if not f.exists():
            print(f"  missing: {f}  (skipped)")
            continue
        d = np.load(f, allow_pickle=True)
        runs.append({
            "seed": s,
            "scores": d["scores"],                 # (N, L+1)
            "labels": d["labels"].astype(int),     # (N,)
            "groups": d["groups"].astype(str),     # (N,)
            "n_layers": int(d["n_layers"]),
        })
        print(f"  loaded seed {s}: {d['scores'].shape[0]} posts, "
              f"{int(d['n_layers'])} layers")
    if not runs:
        raise SystemExit(f"No score files found for tag '{tag}' in {probe_dir}")
    return runs


def pick_groups(groups, labels, min_n, drop):
    """Subgroups with enough posts and both classes present."""
    keep = []
    for g in sorted(set(groups)):
        if g.lower() in drop:
            continue
        m = groups == g
        if m.sum() < min_n:
            continue
        if len(np.unique(labels[m])) < 2:
            print(f"  '{g}' has only one class — excluded")
            continue
        keep.append(g)
    return keep


def auc_per_group(y, s, groups, glist):
    return np.array([safe_auc(y[groups == g], s[groups == g]) for g in glist])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="bert-cls or modernbert-cls")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--probe_dir", default="../outputs/probe")
    ap.add_argument("--out_dir", default="../outputs/gap")
    ap.add_argument("--min_n", type=int, default=30,
                    help="minimum test posts for a subgroup to be included")
    ap.add_argument("--drop", nargs="*", default=["none"],
                    help="group labels to exclude")
    ap.add_argument("--nboot", type=int, default=500,
                    help="bootstrap resamples for the CI on max gap")
    ap.add_argument("--nperm", type=int, default=300,
                    help="label shuffles for the no-real-difference reference")
    ap.add_argument("--seed_rng", type=int, default=0)
    args = ap.parse_args()

    drop = {d.lower() for d in args.drop}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {args.tag}")
    runs = load_runs(args.tag, args.seeds, args.probe_dir)

    L = runs[0]["n_layers"]
    y0, g0 = runs[0]["labels"], runs[0]["groups"]
    glist = pick_groups(g0, y0, args.min_n, drop)
    sizes = {g: int((g0 == g).sum()) for g in glist}
    print(f"\n{len(glist)} subgroups: " +
          ", ".join(f"{g} ({sizes[g]})" for g in glist))
    print(f"Layers 1..{L}  (layer 0 is the embedding output, excluded)\n")

    layers = list(range(1, L + 1))

    # ---- per-seed AUC: (n_seeds, n_layers, n_groups) ----
    A = np.full((len(runs), len(layers), len(glist)), np.nan)
    for i, r in enumerate(runs):
        for j, l in enumerate(layers):
            A[i, j] = auc_per_group(r["labels"], r["scores"][:, l],
                                    r["groups"], glist)

    auc_mean = np.nanmean(A, axis=0)          # (n_layers, n_groups)
    auc_std = np.nanstd(A, axis=0)
    mean_auc = np.nanmean(auc_mean, axis=1)   # simple mean over subgroups
    gap = mean_auc[:, None] - auc_mean        # positive = below the mean
    max_gap = np.nanmax(gap, axis=1)
    worst = [glist[int(np.nanargmax(gap[j]))] for j in range(len(layers))]

    # ---- bootstrap CI on max gap (resample posts within each subgroup) ----
    rng = np.random.default_rng(args.seed_rng)
    print(f"Bootstrap ({args.nboot} resamples)...")
    boot = np.full((len(layers), args.nboot), np.nan)
    idx_by_g = {g: np.where(g0 == g)[0] for g in glist}
    for b in range(args.nboot):
        pick = {g: rng.choice(ix, size=len(ix), replace=True)
                for g, ix in idx_by_g.items()}
        for j, l in enumerate(layers):
            vals = []
            for g in glist:
                ix = pick[g]
                per_seed = [safe_auc(r["labels"][ix], r["scores"][ix, l])
                            for r in runs]
                vals.append(np.nanmean(per_seed))
            vals = np.array(vals)
            vals_gap = np.nanmean(vals) - vals
            boot[j, b] = np.nanmax(vals_gap)
    ci_lo = np.nanpercentile(boot, 2.5, axis=1)
    ci_hi = np.nanpercentile(boot, 97.5, axis=1)

    # ---- permutation reference: what does max gap look like if the
    #      subgroup labels carry no information at all? ----
    print(f"Permutation reference ({args.nperm} shuffles)...")
    perm = np.full((len(layers), args.nperm), np.nan)
    for p in range(args.nperm):
        shuffled = rng.permutation(g0)
        for j, l in enumerate(layers):
            vals = []
            for g in glist:
                m = shuffled == g
                per_seed = [safe_auc(r["labels"][m], r["scores"][m, l])
                            for r in runs]
                vals.append(np.nanmean(per_seed))
            vals = np.array(vals)
            perm[j, p] = np.nanmax(np.nanmean(vals) - vals)
    null_hi = np.nanpercentile(perm, 95, axis=1)
    null_med = np.nanmedian(perm, axis=1)

    # ---- tables ----
    rows = []
    for j, l in enumerate(layers):
        for k, g in enumerate(glist):
            rows.append({
                "layer": l, "group": g, "n": sizes[g],
                "auc_mean": auc_mean[j, k], "auc_std": auc_std[j, k],
                "gap_below_mean": gap[j, k],
            })
    df_g = pd.DataFrame(rows)
    df_g.to_csv(out / f"{args.tag}-subgroup-auc.csv", index=False)

    df_s = pd.DataFrame({
        "layer": layers,
        "mean_auc": mean_auc,
        "max_gap": max_gap,
        "worst_group": worst,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "null_median": null_med, "null_p95": null_hi,
        "above_null": max_gap > null_hi,
    })
    df_s.to_csv(out / f"{args.tag}-gap-summary.csv", index=False)

    print(f"\n{'layer':>5} {'meanAUC':>8} {'maxgap':>8} {'95% CI':>16} "
          f"{'noise p95':>10}  {'>noise':>6}  worst")
    for j, l in enumerate(layers):
        print(f"{l:>5} {mean_auc[j]*100:>7.2f}% {max_gap[j]*100:>7.2f}pp "
              f"[{ci_lo[j]*100:>5.2f},{ci_hi[j]*100:>5.2f}] "
              f"{null_hi[j]*100:>9.2f}pp  "
              f"{'yes' if max_gap[j] > null_hi[j] else 'no':>6}  {worst[j]}")

    # ---- figures ----
    cmap = plt.get_cmap("tab10")
    colors = {g: cmap(i % 10) for i, g in enumerate(glist)}

    # fig 1 — per-subgroup AUC
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for k, g in enumerate(glist):
        ax.plot(layers, auc_mean[:, k] * 100, marker="o", ms=3,
                lw=1.5, color=colors[g], label=f"{g} (n={sizes[g]})")
    ax.plot(layers, mean_auc * 100, lw=2.5, color="black",
            ls="--", label="mean of subgroups")
    ax.set_xlabel("layer"); ax.set_ylabel("AUC (%)")
    ax.set_title(f"Per-subgroup AUC by layer — {args.tag}")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / f"{args.tag}-fig1-subgroup-auc.png", dpi=160)
    plt.close(fig)

    # fig 2 — distance below the mean
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for k, g in enumerate(glist):
        ax.plot(layers, gap[:, k] * 100, marker="o", ms=3,
                lw=1.5, color=colors[g], label=g)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("layer"); ax.set_ylabel("mean AUC − subgroup AUC (pp)")
    ax.set_title(f"Distance below the mean AUC — {args.tag}\n"
                 f"(positive = worse than the average subgroup)", fontsize=11)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / f"{args.tag}-fig2-gap.png", dpi=160)
    plt.close(fig)

    # fig 3 — max gap with noise band
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.fill_between(layers, 0, null_hi * 100, color="grey", alpha=.25,
                    label="range expected with no real subgroup difference (95%)")
    ax.plot(layers, max_gap * 100, marker="o", ms=4, lw=2,
            color="#D1495B", label="max gap (observed)")
    ax.fill_between(layers, ci_lo * 100, ci_hi * 100, color="#D1495B",
                    alpha=.18, label="95% bootstrap CI")
    for j, l in enumerate(layers):
        if max_gap[j] > null_hi[j]:
            ax.annotate(worst[j], (l, max_gap[j] * 100), fontsize=7,
                        xytext=(0, 7), textcoords="offset points", ha="center")
    ax.set_xlabel("layer"); ax.set_ylabel("max gap below mean AUC (pp)")
    ax.set_title(f"Largest subgroup shortfall by layer — {args.tag}", fontsize=12)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / f"{args.tag}-fig3-maxgap.png", dpi=160)
    plt.close(fig)

    print(f"\nSaved to {out}/")
    for f in sorted(out.glob(f"{args.tag}-*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
