"""
Experiment 1 analysis — forced exit at fixed depths.

Reads outputs/probe/{tag}-scores-s*.npz and writes three CSVs:
    {tag}-overall.csv    accuracy, macro F1, GMB per depth (mean, std)
    {tag}-subgroups.csv  subgroup AUC per depth per group
    {tag}-table1.csv     Table 1 rows

Usage:
    python analyze_forced_exit.py --tag bert-cls
    python analyze_forced_exit.py --tag bert-mean
"""
import argparse, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
DEPTHS = [0.0, 0.25, 0.50, 0.75, 1.0]


def gmb(aucs, p=-5):
    a = np.asarray([x for x in aucs if np.isfinite(x)], float)
    return float(np.power(np.mean(np.power(a, p)), 1.0 / p)) if len(a) else np.nan


def perm_test(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = abs(d.mean())
    null = np.abs((rng.choice([-1., 1.], size=(n, len(d))) * d).mean(1))
    return float((null >= obs).mean())


def analyse(path, min_n):
    z = np.load(path, allow_pickle=True)
    scores, preds, y, groups = z["scores"], z["preds"], z["labels"], z["groups"]
    L = int(z["n_layers"])
    keep = [g for g in sorted(set(groups))
            if g != "none" and (groups == g).sum() >= min_n
            and len(set(y[groups == g])) == 2]
    ref = L
    cref = (preds[:, ref] == y).astype(float)

    out = []
    for frac in DEPTHS:
        l = max(1, int(round(frac * L)))   # never the embedding layer
        c = (preds[:, l] == y).astype(float)
        sa, sp = {}, {}
        for g in keep:
            m = groups == g
            sa[g] = roc_auc_score(y[m], scores[m, l])
            sp[g] = np.nan if l == ref else stats.ttest_rel(c[m], cref[m]).pvalue
        if l == ref:
            p_arr = p_perm = p_acc = np.nan
        else:
            a1 = [accuracy_score(y[groups == g], preds[groups == g, l]) for g in keep]
            a2 = [accuracy_score(y[groups == g], preds[groups == g, ref]) for g in keep]
            p_arr, p_perm = stats.ttest_rel(a1, a2).pvalue, perm_test(a1, a2)
            p_acc = stats.ttest_rel(c, cref).pvalue
        out.append(dict(depth=frac, layer=l,
                        acc=accuracy_score(y, preds[:, l]),
                        f1=f1_score(y, preds[:, l], average="macro"),
                        gmb=gmb(list(sa.values())),
                        p_acc=p_acc, p_arr=p_arr, p_perm=p_perm,
                        sub_auc=sa, sub_p=sp))
    return out, keep, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bert-cls")
    ap.add_argument("--dir", default="../outputs/probe")
    ap.add_argument("--min_n", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.dir}/{args.tag}-scores-s*.npz"))
    if not files:
        raise SystemExit(f"No files for tag '{args.tag}' in {args.dir}")
    runs, keep, L = [], None, None
    for f in files:
        r, keep, L = analyse(f, args.min_n)
        runs.append(r)
    S = len(runs)
    sd = lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0

    print(f"Tag {args.tag} | seeds {S} | layers {L} | alpha {args.alpha}")
    print(f"Subgroups ({len(keep)}): {', '.join(keep)}\n")

    # ---- overall ----
    ov = []
    for i, frac in enumerate(DEPTHS):
        acc = [r[i]["acc"] for r in runs]
        f1  = [r[i]["f1"] for r in runs]
        gm  = [r[i]["gmb"] for r in runs]
        sig = frac != 1.0 and all(np.isfinite(r[i]["p_acc"]) and
                                  r[i]["p_acc"] < args.alpha for r in runs)
        ov.append(dict(depth=f"{frac*100:.0f}%", layer=runs[0][i]["layer"],
                       acc=np.mean(acc)*100, acc_std=sd(acc)*100,
                       macro_f1=np.mean(f1)*100, macro_f1_std=sd(f1)*100,
                       gmb=np.mean(gm)*100, gmb_std=sd(gm)*100,
                       acc_significant=sig,
                       p_subgroup_ttest=np.nanmean([r[i]["p_arr"] for r in runs]),
                       p_subgroup_perm=np.nanmean([r[i]["p_perm"] for r in runs])))
    ov = pd.DataFrame(ov)

    # ---- subgroups ----
    sub = []
    for g in keep:
        row = {"group": g}
        for i, frac in enumerate(DEPTHS):
            row[f"auc_{frac*100:.0f}"] = np.mean([r[i]["sub_auc"][g] for r in runs]) * 100
        sub.append(row)
    sub = pd.DataFrame(sub)

    # ---- table 1 ----
    t1 = []
    for i, frac in enumerate(DEPTHS):
        sig, nonsig, diffs = [], [], {}
        for g in keep:
            if frac != 1.0:
                ok = all(np.isfinite(r[i]["sub_p"][g]) and
                         r[i]["sub_p"][g] < args.alpha for r in runs)
                (sig if ok else nonsig).append(g)
            diffs[g] = np.mean([r[i]["sub_auc"][g] - r[-1]["sub_auc"][g] for r in runs])
        if frac == 1.0:
            ext = "reference"
        else:
            gmax = min(diffs, key=diffs.get)
            gmin = min(diffs, key=lambda k: abs(diffs[k]))
            ext = f"{gmax} {diffs[gmax]:+.2f}, {gmin} {diffs[gmin]:+.2f}"
        t1.append(dict(depth=ov.loc[i, "depth"],
                       accuracy=f"{ov.loc[i,'acc']:.2f}%" +
                                ("*" if ov.loc[i, "acc_significant"] else ""),
                       gmb=f"{ov.loc[i,'gmb']:.2f}%",
                       significant=", ".join(sig) if sig else ("none" if frac != 1.0 else "-"),
                       not_significant=(("all" if len(nonsig) == len(keep) else
                                         ", ".join(nonsig)) if frac != 1.0 else "-"),
                       max_min_dAUC=ext))
    t1 = pd.DataFrame(t1)

    # ---- print ----
    print(t1.to_string(index=False))

    # ---- save ----
    d = Path(args.dir)
    ov.round(2).to_csv(d / f"{args.tag}-overall.csv", index=False)
    sub.round(2).to_csv(d / f"{args.tag}-subgroups.csv", index=False)
    t1.to_csv(d / f"{args.tag}-table1.csv", index=False)
    print(f"\nSaved {args.tag}-overall.csv, -subgroups.csv, -table1.csv in {d}")


if __name__ == "__main__":
    main()