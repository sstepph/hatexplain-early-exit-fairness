"""
Experiment 2 — early exiting methods compared on a frozen model.

Reads outputs/probe/{tag}-scores-s*.npz (from Exp 1 + add_prototype_margins).
All three methods use the same per-layer heads for prediction; they differ
only in the exit criterion:

  entropy    exit at first layer where predictive entropy < threshold
  patience   exit at first layer where the last m predictions agree
  prototype  exit at first layer where |cos margin to prototypes| >= gap

Layer 0 (embedding output) is excluded: its CLS vector is identical for
every post. Exits range over layers 1..L.

Expected saving (Xin et al. 2020, DeeBERT, p.4):  1 - mean(exit layer) / L

Outputs in outputs/probe/:
  exp2-{tag}-sweep.csv    every parameter value, every method, mean over seeds
  exp2-{tag}-table2.csv   Table 2, one row per method at ~target saving
  exp2-{tag}-fig1.png     avg exit vs accuracy and macro F1
  exp2-{tag}-fig2.png     avg exit vs GMB

Usage:
  python exp2_early_exit.py --tag bert-cls
"""
import argparse, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

ENTROPIES  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.68, 0.69]
PATIENCES  = [2, 3, 4, 5, 6, 8, 10, 12]
PROTO_GAPS = [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075,
              0.01, 0.025, 0.05, 0.075, 0.1]


# ---------------------------------------------------------------- exit rules
def exit_entropy(scores, thr, L):
    p = np.clip(scores[:, 1:], 1e-12, 1 - 1e-12)
    H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    hit = H < thr
    return np.where(hit.any(1), hit.argmax(1) + 1, L)


def exit_patience(preds, m, L):
    P = preds[:, 1:]
    out = np.full(len(P), L)
    for n in range(len(P)):
        for j in range(m - 1, L):
            w = P[n, j - m + 1:j + 1]
            if (w == w[0]).all():
                out[n] = j + 1
                break
    return out


def exit_proto(margin, gap, L):
    hit = np.abs(margin[:, 1:]) >= gap
    return np.where(hit.any(1), hit.argmax(1) + 1, L)


# ---------------------------------------------------------------- metrics
def gmb(aucs, p=-5):
    a = np.asarray([x for x in aucs if np.isfinite(x)], float)
    return float(np.power(np.mean(np.power(a, p)), 1 / p)) if len(a) else np.nan


def perm_test(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    null = np.abs((rng.choice([-1., 1.], (n, len(d))) * d).mean(1))
    return float((null >= abs(d.mean())).mean())


def evaluate(exit_l, z, keep):
    n = np.arange(len(exit_l))
    L = int(z["n_layers"])
    y, g = z["labels"], z["groups"]
    pred = z["preds"][n, exit_l]
    score = z["scores"][n, exit_l]
    ref_pred = z["preds"][:, L]
    c, cref = (pred == y).astype(float), (ref_pred == y).astype(float)

    sub_auc, sub_p, a1, a2 = {}, {}, [], []
    for grp in keep:
        m = g == grp
        sub_auc[grp] = roc_auc_score(y[m], score[m])
        sub_p[grp] = stats.ttest_rel(c[m], cref[m]).pvalue if (c[m] != cref[m]).any() else 1.0
        a1.append(accuracy_score(y[m], pred[m]))
        a2.append(accuracy_score(y[m], ref_pred[m]))

    return dict(
        acc=accuracy_score(y, pred), f1=f1_score(y, pred, average="macro"),
        gmb=gmb(list(sub_auc.values())),
        avg_exit=float(exit_l.mean()), saving=1 - exit_l.mean() / L,
        p_acc=stats.ttest_rel(c, cref).pvalue if (c != cref).any() else 1.0,
        p_arr=stats.ttest_rel(a1, a2).pvalue, p_perm=perm_test(a1, a2),
        sub_auc=sub_auc, sub_p=sub_p)


def reference(z, keep):
    L = int(z["n_layers"])
    return evaluate(np.full(len(z["labels"]), L), z, keep)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bert-cls")
    ap.add_argument("--dir", default="../outputs/probe")
    ap.add_argument("--model_name", default="BERT")
    ap.add_argument("--target_saving", type=float, default=0.20)
    ap.add_argument("--min_n", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.dir}/{args.tag}-scores-s*.npz"))
    Z = [dict(np.load(f, allow_pickle=True)) for f in files]
    if not Z:
        raise SystemExit(f"No files for tag {args.tag}")
    if "proto_margin" not in Z[0]:
        raise SystemExit("proto_margin missing — run add_prototype_margins.py first")

    L = int(Z[0]["n_layers"])
    y, g = Z[0]["labels"], Z[0]["groups"]
    keep = [k for k in sorted(set(g)) if k != "none"
            and (g == k).sum() >= args.min_n and len(set(y[g == k])) == 2]
    print(f"Seeds {len(Z)} | layers {L} | subgroups {len(keep)}: {', '.join(keep)}\n")

    methods = {
        "Entropy-based":   (ENTROPIES,                  lambda z, v: exit_entropy(z["scores"], v, L)),
        "Patience-based":  (PATIENCES,                  lambda z, v: exit_patience(z["preds"], v, L)),
        "Prototype-based": (PROTO_GAPS,                 lambda z, v: exit_proto(z["proto_margin"], v, L)),
    }

    # ---- full sweep, for the figures ----
    sweep = []
    for name, (grid, fn) in methods.items():
        for v in grid:
            rs = [evaluate(fn(z, v), z, keep) for z in Z]
            sweep.append(dict(method=name, param=v,
                              avg_exit=np.mean([r["avg_exit"] for r in rs]),
                              saving=np.mean([r["saving"] for r in rs]) * 100,
                              acc=np.mean([r["acc"] for r in rs]) * 100,
                              f1=np.mean([r["f1"] for r in rs]) * 100,
                              gmb=np.mean([r["gmb"] for r in rs]) * 100))
    sweep = pd.DataFrame(sweep)
    print(sweep.round(2).to_string(index=False))

    # ---- pick parameter closest to the target saving ----
    # fine grids so every method can land near the target
    fine = {
        "Entropy-based":   np.round(np.arange(0.001, 0.694, 0.001), 3),
        "Patience-based":  list(range(2, L + 1)),
        "Prototype-based": np.unique(np.round(np.concatenate([
            np.quantile(np.abs(Z[0]["proto_margin"][:, 1:]), np.linspace(0, 1, 400)),
            PROTO_GAPS]), 6)),
    }
    chosen = {}
    for name, (_, fn) in methods.items():
        best, best_d = None, 1e9
        for v in fine[name]:
            s = np.mean([1 - fn(z, v).mean() / L for z in Z])
            if abs(s - args.target_saving) < best_d:
                best, best_d = v, abs(s - args.target_saving)
        chosen[name] = best

    print("\nCHOSEN PARAMETERS FOR TABLE 2:")
    for name, value in chosen.items():
        print(f"{name}: {value:.6f}")

    # ---- Table 2 ----
    refs = [reference(z, keep) for z in Z]
    rows = []
    for name, (_, fn) in methods.items():
        v = chosen[name]
        rs = [evaluate(fn(z, v), z, keep) for z in Z]
        sig_acc = all(r["p_acc"] < args.alpha for r in rs)
        sig, nonsig, diffs = [], [], {}
        for grp in keep:
            (sig if all(r["sub_p"][grp] < args.alpha for r in rs) else nonsig).append(grp)
            diffs[grp] = np.mean([r["sub_auc"][grp] - ref["sub_auc"][grp]
                                  for r, ref in zip(rs, refs)])
        gmax = min(diffs, key=diffs.get)
        gmin = min(diffs, key=lambda k: abs(diffs[k]))
        vlabel = f"{v:.6f}" if name != "Patience-based" else str(int(v))
        rows.append({
            "Early exiting": f"{name} {vlabel}",
            "Model": args.model_name,
            "Accuracy": f"{np.mean([r['acc'] for r in rs])*100:.2f}%" + ("*" if sig_acc else ""),
            "GMB": f"{np.mean([r['gmb'] for r in rs])*100:.2f}%",
            "Significant": ", ".join(sig) if sig else "none",
            "Not significant": "all" if len(nonsig) == len(keep) else ", ".join(nonsig),
            "Max / min dAUC": f"{gmax} {diffs[gmax]:+.3f}, {gmin} {diffs[gmin]:+.3f}",
            "Expected saving": f"{np.mean([r['saving'] for r in rs])*100:.2f}%",
            "Avg exit": f"{np.mean([r['avg_exit'] for r in rs]):.2f}",
        })
    t2 = pd.DataFrame(rows)

    ref_acc = np.mean([r["acc"] for r in refs]) * 100
    ref_gmb = np.mean([r["gmb"] for r in refs]) * 100
    print(f"\nReference (100% depth): accuracy {ref_acc:.2f}% | GMB {ref_gmb:.2f}%\n")
    print(t2.to_string(index=False))

    d = Path(args.dir)
    sweep.round(4).to_csv(d / f"exp2-{args.tag}-sweep.csv", index=False)
    t2.to_csv(d / f"exp2-{args.tag}-table2.csv", index=False)

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colours = {"Entropy-based": "#2a78d6", "Patience-based": "#eb6834", "Prototype-based": "#1baf7a"}

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for metric, ax, lab in [("acc", axes[0], "Accuracy (%)"),
                                ("f1", axes[1], "Macro F1 (%)")]:
            for name, grp in sweep.groupby("method"):
                grp = grp.sort_values("avg_exit")
                ax.plot(grp["avg_exit"], grp[metric], "o-", label=name,
                        color=colours[name], markersize=4)
            ax.set_xlabel("Average exit layer"); ax.set_ylabel(lab)
            ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(d / f"exp2-{args.tag}-fig1.png", dpi=150)
        fig.savefig(d / f"exp2-{args.tag}-fig1.pdf")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        for name, grp in sweep.groupby("method"):
            grp = grp.sort_values("avg_exit")
            ax.plot(grp["avg_exit"], grp["gmb"], "o-", label=name,
                    color=colours[name], markersize=4)
        ax.axhline(ref_gmb, ls="--", color="grey", lw=1, label="100% depth")
        ax.set_xlabel("Average exit layer"); ax.set_ylabel("GMB subgroup AUC (%)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(d / f"exp2-{args.tag}-fig2.png", dpi=150)
        fig.savefig(d / f"exp2-{args.tag}-fig2.pdf")
        print(f"\nFigures saved: exp2-{args.tag}-fig1/fig2 (.png and .pdf)")
    except ImportError:
        print("\nmatplotlib not installed — figures skipped")

    print(f"CSVs saved: exp2-{args.tag}-sweep.csv, exp2-{args.tag}-table2.csv")


if __name__ == "__main__":
    main()
