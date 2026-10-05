#!/usr/bin/env python3
import argparse
import glob
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.metrics import accuracy_score, roc_auc_score

DEPTHS = [0.0, 0.25, 0.50, 0.75]


def gmb(aucs, p=-5):
    a = np.asarray(aucs, dtype=float)
    return float(np.power(np.mean(np.power(a, p)), 1.0 / p))


def gmb_many(aucs, p=-5):
    a = np.asarray(aucs, dtype=float)
    return np.power(np.mean(np.power(a, p), axis=0), 1.0 / p)


def exact_mcnemar(y, pred_exit, pred_ref):
    y = np.asarray(y)
    a = np.asarray(pred_exit) == y
    b = np.asarray(pred_ref) == y

    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01

    p = 1.0 if n == 0 else float(
        binomtest(n10, n=n, p=0.5, alternative="two-sided").pvalue
    )
    return p, n10, n01


def compute_midrank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j

    out = np.empty(len(x), dtype=float)
    out[order] = ranks
    return out


def fast_delong(pred, n_pos):
    pred = np.asarray(pred, dtype=float)
    k = pred.shape[0]
    m = int(n_pos)
    n = pred.shape[1] - m

    pos = pred[:, :m]
    neg = pred[:, m:]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))

    for i in range(k):
        tx[i] = compute_midrank(pos[i])
        ty[i] = compute_midrank(neg[i])
        tz[i] = compute_midrank(pred[i])

    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.atleast_2d(np.cov(v01, bias=False))
    sy = np.atleast_2d(np.cov(v10, bias=False))
    cov = sx / m + sy / n
    return auc, cov


def paired_delong(y, score_exit, score_ref):
    y = np.asarray(y, dtype=int)
    s1 = np.asarray(score_exit, dtype=float)
    s2 = np.asarray(score_ref, dtype=float)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if n_pos < 2 or n_neg < 2:
        return (np.nan,) * 5

    order = np.argsort(-y)
    aucs, cov = fast_delong(np.vstack([s1, s2])[:, order], n_pos)

    auc_exit = float(aucs[0])
    auc_ref = float(aucs[1])
    delta = auc_exit - auc_ref
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])

    if not np.isfinite(var) or var < 0:
        return auc_exit, auc_ref, delta, np.nan, np.nan

    if var <= 1e-18:
        if abs(delta) <= 1e-15:
            return auc_exit, auc_ref, delta, 0.0, 1.0
        return auc_exit, auc_ref, delta, np.inf, 0.0

    z = delta / math.sqrt(var)
    p = float(2.0 * norm.sf(abs(z)))
    return auc_exit, auc_ref, delta, float(z), p


def bonferroni_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)

    m = int(np.sum(valid))
    if m:
        out[valid] = np.minimum(1.0, p[valid] * m)

    return out


def seed_from_filename(path):
    m = re.search(r"-s(\d+)\.npz$", str(path))
    if not m:
        raise ValueError(f"Could not parse seed from {path}")
    return int(m.group(1))


def eligible_groups(y, groups, min_n):
    return [
        g
        for g in sorted(set(groups))
        if g != "none"
        and int(np.sum(groups == g)) >= min_n
        and len(np.unique(y[groups == g])) == 2
    ]


def pairwise_auc_matrix(pos_scores, neg_scores):
    pos = np.asarray(pos_scores, dtype=float)[:, None]
    neg = np.asarray(neg_scores, dtype=float)[None, :]
    return (pos > neg).astype(float) + 0.5 * (pos == neg)


def bootstrap_auc_pair(pos1, neg1, pos2, neg2, n_boot, rng):
    n_pos = len(pos1)
    n_neg = len(neg1)

    pos_counts = rng.multinomial(
        n_pos, np.full(n_pos, 1.0 / n_pos), size=n_boot
    )
    neg_counts = rng.multinomial(
        n_neg, np.full(n_neg, 1.0 / n_neg), size=n_boot
    )

    mat1 = pairwise_auc_matrix(pos1, neg1)
    mat2 = pairwise_auc_matrix(pos2, neg2)

    auc1 = np.einsum(
        "bi,ij,bj->b", pos_counts, mat1, neg_counts, optimize=True
    ) / (n_pos * n_neg)

    auc2 = np.einsum(
        "bi,ij,bj->b", pos_counts, mat2, neg_counts, optimize=True
    ) / (n_pos * n_neg)

    return auc1, auc2


def bootstrap_gmb(y, groups, scores, layer, ref, keep, n_boot, alpha, seed):
    rng = np.random.default_rng(seed)

    observed_exit = []
    observed_ref = []
    boot_exit = []
    boot_ref = []

    for i, g in enumerate(keep, start=1):
        gm = np.flatnonzero(groups == g)
        pos = gm[y[gm] == 1]
        neg = gm[y[gm] == 0]

        observed_exit.append(roc_auc_score(y[gm], scores[gm, layer]))
        observed_ref.append(roc_auc_score(y[gm], scores[gm, ref]))

        a1, a2 = bootstrap_auc_pair(
            scores[pos, layer],
            scores[neg, layer],
            scores[pos, ref],
            scores[neg, ref],
            n_boot,
            rng,
        )

        boot_exit.append(a1)
        boot_ref.append(a2)
        print(f"    group {i}/{len(keep)}: {g}", flush=True)

    obs_exit = gmb(observed_exit)
    obs_ref = gmb(observed_ref)
    obs_delta = obs_exit - obs_ref

    boot_exit = np.vstack(boot_exit)
    boot_ref = np.vstack(boot_ref)
    deltas = gmb_many(boot_exit) - gmb_many(boot_ref)

    lo, hi = np.quantile(
        deltas,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    bonf_alpha = alpha / len(DEPTHS)
    lo_b, hi_b = np.quantile(
        deltas,
        [bonf_alpha / 2.0, 1.0 - bonf_alpha / 2.0],
    )

    return {
        "gmb_exit": obs_exit,
        "gmb_ref": obs_ref,
        "delta_gmb": obs_delta,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "ci_bonf_low": float(lo_b),
        "ci_bonf_high": float(hi_b),
        "significant_bonf": bool(lo_b > 0 or hi_b < 0),
    }


def analyse_one(path, min_n, n_boot, alpha, bootstrap_seed):
    z = np.load(path, allow_pickle=True)

    required = {"scores", "preds", "labels", "groups", "n_layers"}
    missing = required.difference(z.files)
    if missing:
        raise ValueError(f"{path}: missing keys: {sorted(missing)}")

    scores = np.asarray(z["scores"])
    preds = np.asarray(z["preds"])
    y = np.asarray(z["labels"], dtype=int)
    groups = np.asarray(z["groups"]).astype(str)
    L = int(z["n_layers"])
    seed = seed_from_filename(path)

    keep = eligible_groups(y, groups, min_n)
    ref = L

    rows = []
    gmb_rows = []

    for frac in DEPTHS:
        layer = max(1, int(round(frac * L)))
        depth = f"{frac * 100:.0f}%"

        for g in ["ALL"] + keep:
            mask = np.ones(len(y), dtype=bool) if g == "ALL" else groups == g
            yy = y[mask]

            pred_exit = preds[mask, layer]
            pred_ref = preds[mask, ref]
            score_exit = scores[mask, layer]
            score_ref = scores[mask, ref]

            acc_exit = accuracy_score(yy, pred_exit)
            acc_ref = accuracy_score(yy, pred_ref)

            p_mc, b, c = exact_mcnemar(yy, pred_exit, pred_ref)
            auc_exit, auc_ref, delta_auc, zval, p_d = paired_delong(
                yy, score_exit, score_ref
            )

            rows.append({
                "seed": seed,
                "depth": depth,
                "layer": layer,
                "ref_layer": ref,
                "group": g,
                "N": int(mask.sum()),
                "positives": int(np.sum(yy == 1)),
                "negatives": int(np.sum(yy == 0)),
                "acc_exit": float(acc_exit),
                "acc_ref": float(acc_ref),
                "delta_acc": float(acc_exit - acc_ref),
                "mcnemar_exit_correct_ref_wrong": b,
                "mcnemar_exit_wrong_ref_correct": c,
                "mcnemar_discordant": b + c,
                "p_mcnemar": p_mc,
                "auc_exit": auc_exit,
                "auc_ref": auc_ref,
                "delta_auc": delta_auc,
                "z_delong": zval,
                "p_delong": p_d,
            })

        print(
            f"seed {seed}, depth {depth}, layer {layer}: "
            f"GMB bootstrap ({n_boot})",
            flush=True,
        )

        boot = bootstrap_gmb(
            y,
            groups,
            scores,
            layer,
            ref,
            keep,
            n_boot,
            alpha,
            bootstrap_seed + seed * 1000 + layer,
        )

        boot.update({
            "seed": seed,
            "depth": depth,
            "layer": layer,
            "ref_layer": ref,
            "n_boot": n_boot,
        })
        gmb_rows.append(boot)

    return rows, gmb_rows, keep, L


def add_bonferroni(df):
    df["p_mcnemar_bonf"] = np.nan
    df["p_delong_bonf"] = np.nan

    for seed in sorted(df["seed"].unique()):
        seed_mask = df["seed"] == seed

        for test in ["mcnemar", "delong"]:
            pcol = f"p_{test}"
            outcol = f"p_{test}_bonf"

            overall = seed_mask & (df["group"] == "ALL")
            subgroup = seed_mask & (df["group"] != "ALL")

            df.loc[overall, outcol] = bonferroni_adjust(
                df.loc[overall, pcol].to_numpy()
            )
            df.loc[subgroup, outcol] = bonferroni_adjust(
                df.loc[subgroup, pcol].to_numpy()
            )

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bert-cls")
    ap.add_argument("--dir", default="../outputs/probe")
    ap.add_argument("--min_n", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=1234)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.dir}/{args.tag}-scores-s*.npz"))
    if not files:
        raise SystemExit(
            f"No NPZ files matching {args.dir}/{args.tag}-scores-s*.npz"
        )

    rows = []
    boot_rows = []
    all_counts = []
    keep = None
    n_layers = None

    for f in files:
        print(f"\nProcessing {f}", flush=True)

        r, b, keep, L = analyse_one(
            f,
            args.min_n,
            args.n_boot,
            args.alpha,
            args.bootstrap_seed,
        )

        rows.extend(r)
        boot_rows.extend(b)

        z = np.load(f, allow_pickle=True)
        y = np.asarray(z["labels"], dtype=int)
        groups = np.asarray(z["groups"]).astype(str)
        seed = seed_from_filename(f)

        for g in ["ALL"] + keep:
            mask = np.ones(len(y), dtype=bool) if g == "ALL" else groups == g
            yy = y[mask]
            all_counts.append({
                "seed": seed,
                "group": g,
                "N": int(mask.sum()),
                "positives": int(np.sum(yy == 1)),
                "negatives": int(np.sum(yy == 0)),
            })

        if n_layers is None:
            n_layers = L
        elif n_layers != L:
            raise ValueError("Runs have different numbers of layers.")

    df = add_bonferroni(pd.DataFrame(rows))
    boot = pd.DataFrame(boot_rows)

    for test in ["mcnemar", "delong"]:
        df[f"{test}_sig_raw"] = df[f"p_{test}"] < args.alpha
        df[f"{test}_sig_bonf"] = df[f"p_{test}_bonf"] < args.alpha

    out = Path(args.dir)

    per_seed_file = out / f"{args.tag}-paired-tests-per-seed.csv"
    summary_file = out / f"{args.tag}-paired-tests-summary.csv"
    counts_file = out / f"{args.tag}-paired-tests-group-counts.csv"
    gmb_file = out / f"{args.tag}-gmb-bootstrap.csv"
    gmb_summary_file = out / f"{args.tag}-gmb-bootstrap-summary.csv"

    df.to_csv(per_seed_file, index=False)
    boot.to_csv(gmb_file, index=False)

    counts = (
        pd.DataFrame(all_counts)
        .groupby("group", as_index=False)
        .agg(
            N=("N", "first"),
            positives=("positives", "first"),
            negatives=("negatives", "first"),
        )
    )
    counts.to_csv(counts_file, index=False)

    summary = (
        df.groupby(["depth", "layer", "ref_layer", "group"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_delta_acc=("delta_acc", "mean"),
            mcnemar_sig_seeds_raw=("mcnemar_sig_raw", "sum"),
            mcnemar_sig_seeds_bonf=("mcnemar_sig_bonf", "sum"),
            mean_delta_auc=("delta_auc", "mean"),
            delong_sig_seeds_raw=("delong_sig_raw", "sum"),
            delong_sig_seeds_bonf=("delong_sig_bonf", "sum"),
        )
    )
    summary.to_csv(summary_file, index=False)

    gmb_summary = (
        boot.groupby(["depth", "layer", "ref_layer"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_delta_gmb=("delta_gmb", "mean"),
            significant_seeds=("significant", "sum"),
            significant_seeds_bonf=("significant_bonf", "sum"),
        )
    )
    gmb_summary.to_csv(gmb_summary_file, index=False)

    print(
        f"\n{len(files)} seeds | {n_layers} layers | "
        f"groups: {', '.join(keep)}"
    )

    cols = [
        "seed",
        "depth",
        "layer",
        "group",
        "delta_acc",
        "p_mcnemar",
        "p_mcnemar_bonf",
        "delta_auc",
        "p_delong",
        "p_delong_bonf",
    ]

    print("\nPaired tests, raw p < alpha:")
    sig = df[
        (df["p_mcnemar"] < args.alpha)
        | (df["p_delong"] < args.alpha)
    ]
    print(sig[cols].to_string(index=False) if len(sig) else "None")

    print("\nPaired tests, Bonferroni p < alpha:")
    sig = df[
        (df["p_mcnemar_bonf"] < args.alpha)
        | (df["p_delong_bonf"] < args.alpha)
    ]
    print(sig[cols].to_string(index=False) if len(sig) else "None")

    print(f"\nGMB bootstrap ({100 * (1 - args.alpha):.1f}% CI):")
    print(
        boot[
            [
                "seed",
                "depth",
                "layer",
                "delta_gmb",
                "ci_low",
                "ci_high",
                "significant",
                "ci_bonf_low",
                "ci_bonf_high",
                "significant_bonf",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(per_seed_file)
    print(summary_file)
    print(counts_file)
    print(gmb_file)
    print(gmb_summary_file)


if __name__ == "__main__":
    main()
