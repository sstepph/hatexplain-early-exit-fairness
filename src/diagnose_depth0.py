"""
Why is 0% depth non-significant for most groups?

Prints, per seed: the predictions made at depth 0, each group's accuracy at
depth 0 and at depth 100%, the raw p-value, and the accuracy drop that would
be needed for significance at the chosen alpha.

Usage:
    python diagnose_depth0.py --tag bert-cls --depth 0
    python diagnose_depth0.py --tag bert-cls --depth 0.25
"""
import argparse, glob
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="bert-cls")
ap.add_argument("--dir", default="../outputs/probe")
ap.add_argument("--depth", type=float, default=0.0)
ap.add_argument("--alpha", type=float, default=0.01)
ap.add_argument("--min_n", type=int, default=50)
args = ap.parse_args()

files = sorted(glob.glob(f"{args.dir}/{args.tag}-scores-s*.npz"))
crit = stats.t.ppf(1 - args.alpha / 2, df=10**6)   # ~2.576

for f in files:
    z = np.load(f, allow_pickle=True)
    preds, y, g = z["preds"], z["labels"], z["groups"]
    L = int(z["n_layers"])
    l = int(round(args.depth * L))

    p_l, p_ref = preds[:, l], preds[:, L]
    print("=" * 74)
    print(f"{f.split('/')[-1]}   depth {args.depth*100:.0f}% = layer {l}")
    print(f"  predictions at layer {l}: "
          f"{dict(zip(*np.unique(p_l, return_counts=True)))}")
    if len(np.unique(p_l)) == 1:
        print("  --> CONSTANT: the head predicts one class for every post")
    print(f"  overall acc  L{l} {accuracy_score(y, p_l)*100:6.2f}%   "
          f"L{L} {accuracy_score(y, p_ref)*100:6.2f}%")
    print()
    print(f"  {'group':<12}{'n':>5}{'hate%':>7}{'acc L'+str(l):>9}"
          f"{'acc L'+str(L):>9}{'diff':>8}{'p':>10}{'needed':>9}{'sig':>5}")
    print("  " + "-" * 72)

    for grp in sorted(set(g)):
        if grp == "none":
            continue
        m = g == grp
        n = int(m.sum())
        if n < args.min_n or len(set(y[m])) < 2:
            continue
        a1 = accuracy_score(y[m], p_l[m])
        a2 = accuracy_score(y[m], p_ref[m])
        c, cref = (p_l[m] == y[m]).astype(float), (p_ref[m] == y[m]).astype(float)
        p = stats.ttest_rel(c, cref).pvalue if (c != cref).any() else 1.0
        need = crit * np.std(c - cref, ddof=1) / np.sqrt(n) if (c != cref).any() else np.nan
        print(f"  {grp:<12}{n:>5}{y[m].mean()*100:>6.0f}%"
              f"{a1*100:>8.1f}%{a2*100:>8.1f}%{(a1-a2)*100:>+7.1f}"
              f"{p:>10.4f}{need*100:>8.1f}{'  YES' if p < args.alpha else '   no':>5}")
    print()

print("'needed' = accuracy gap required for p < alpha, given that group's n.")
print("Small groups need a much larger gap to reach significance.")
