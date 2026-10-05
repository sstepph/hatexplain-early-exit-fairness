#!/usr/bin/env python3
"""
Paired statistical tests for Experiment 2 (dynamic early exiting).

Reuses the tests from Irina's stat_tests.py, but gathers predictions at the
per-example exit layer chosen by each criterion rather than at one fixed layer.

Reference is full depth: every example exits at layer L.

    python stat_tests_exp2.py --tag bert-cls --n_boot 2000

Outputs in --dir:
    exp2-{tag}-paired-tests-per-seed.csv
    exp2-{tag}-paired-tests-summary.csv
    exp2-{tag}-gmb-bootstrap.csv
"""
import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from stat_tests import (exact_mcnemar, paired_delong, gmb, gmb_many,
                        bootstrap_auc_pair, eligible_groups,
                        bonferroni_adjust, seed_from_filename)


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


# ---------------------------------------------------------------- bootstrap
def bootstrap_gmb_vectors(y, groups, s_exit, s_ref, keep, n_boot, alpha,
                          seed, n_tests):
    """Stratified paired bootstrap on GMB, using pre-gathered score vectors."""
    rng = np.random.default_rng(seed)
    obs_e, obs_r, boot_e, boot_r = [], [], [], []

    for i, g in enumerate(keep, start=1):
        gm = np.flatnonzero(groups == g)
        pos, neg = gm[y[gm] == 1], gm[y[gm] == 0]

        obs_e.append(roc_auc_score(y[gm], s_exit[gm]))
        obs_r.append(roc_auc_score(y[gm], s_ref[gm]))

        a1, a2 = bootstrap_auc_pair(s_exit[pos], s_exit[neg],
                                    s_ref[pos], s_ref[neg], n_boot, rng)
        boot_e.append(a1)
        boot_r.append(a2)
        print(f"    group {i}/{len(keep)}: {g}", flush=True)

    g_e, g_r = gmb(obs_e), gmb(obs_r)
    deltas = gmb_many(np.vstack(boot_e)) - gmb_many(np.vstack(boot_r))

    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    ab = alpha / n_tests
    lo_b, hi_b = np.quantile(deltas, [ab / 2, 1 - ab / 2])

    return dict(gmb_exit=g_e, gmb_ref=g_r, delta_gmb=g_e - g_r,
                ci_low=float(lo), ci_high=float(hi),
                significant=bool(lo > 0 or hi < 0),
                ci_bonf_low=float(lo_b), ci_bonf_high=float(hi_b),
                significant_bonf=bool(lo_b > 0 or hi_b < 0))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bert-cls")
    ap.add_argument("--dir", default="../outputs/probe")
    ap.add_argument("--min_n", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=1234)
    ap.add_argument("--entropy", type=float, default=None,
                    help="entropy threshold; omit to search for --target_saving")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--proto", type=float, default=None)
    ap.add_argument("--target_saving", type=float, default=0.20)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.dir}/{args.tag}-scores-s*.npz"))
    if not files:
        raise SystemExit(f"No files for tag {args.tag} in {args.dir}")

    Z = [dict(np.load(f, allow_pickle=True)) for f in files]
    seeds = [seed_from_filename(f) for f in files]
    L = int(Z[0]["n_layers"])
    y0, g0 = np.asarray(Z[0]["labels"], int), np.asarray(Z[0]["groups"]).astype(str)
    keep = eligible_groups(y0, g0, args.min_n)

    if "proto_margin" not in Z[0]:
        raise SystemExit("proto_margin missing — run add_prototype_margins.py")

    rules = {
        "Entropy-based":   lambda z, v: exit_entropy(z["scores"], v, L),
        "Patience-based":  lambda z, v: exit_patience(z["preds"], int(v), L),
        "Prototype-based": lambda z, v: exit_proto(z["proto_margin"], v, L),
    }

    # pick parameters if not supplied
    given = {"Entropy-based": args.entropy, "Patience-based": args.patience,
             "Prototype-based": args.proto}
    grids = {
        "Entropy-based":   np.round(np.arange(0.001, 0.694, 0.001), 3),
        "Patience-based":  np.arange(2, L + 1),
        "Prototype-based": np.unique(np.round(np.quantile(
            np.abs(Z[0]["proto_margin"][:, 1:]), np.linspace(0, 1, 400)), 6)),
    }
    chosen = {}
    for name, fn in rules.items():
        if given[name] is not None:
            chosen[name] = given[name]
            continue
        best, bd = None, 1e9
        for v in grids[name]:
            s = np.mean([1 - fn(z, v).mean() / L for z in Z])
            if abs(s - args.target_saving) < bd:
                best, bd = v, abs(s - args.target_saving)
        chosen[name] = best

    print(f"{len(Z)} seeds | {L} layers | groups: {', '.join(keep)}")
    for k, v in chosen.items():
        print(f"  {k}: {v}")

    n_tests = len(rules)
    rows, boots = [], []

    for z, seed in zip(Z, seeds):
        y = np.asarray(z["labels"], int)
        groups = np.asarray(z["groups"]).astype(str)
        n = np.arange(len(y))
        pred_ref, score_ref = z["preds"][:, L], z["scores"][:, L]

        for name, fn in rules.items():
            e = fn(z, chosen[name])
            pred_e, score_e = z["preds"][n, e], z["scores"][n, e]

            for g in ["ALL"] + keep:
                m = np.ones(len(y), bool) if g == "ALL" else groups == g
                p_mc, b, c = exact_mcnemar(y[m], pred_e[m], pred_ref[m])
                a_e, a_r, d_auc, zval, p_d = paired_delong(
                    y[m], score_e[m], score_ref[m])
                rows.append(dict(
                    seed=seed, method=name, param=float(chosen[name]), group=g,
                    N=int(m.sum()),
                    avg_exit=float(e[m].mean()),
                    saving=float(1 - e[m].mean() / L),
                    acc_exit=accuracy_score(y[m], pred_e[m]),
                    acc_ref=accuracy_score(y[m], pred_ref[m]),
                    delta_acc=accuracy_score(y[m], pred_e[m])
                              - accuracy_score(y[m], pred_ref[m]),
                    p_mcnemar=p_mc, mcnemar_discordant=b + c,
                    auc_exit=a_e, auc_ref=a_r, delta_auc=d_auc,
                    z_delong=zval, p_delong=p_d))

            print(f"seed {seed}, {name}: GMB bootstrap ({args.n_boot})", flush=True)
            bt = bootstrap_gmb_vectors(
                y, groups, score_e, score_ref, keep, args.n_boot,
                args.alpha, args.bootstrap_seed + seed * 1000, n_tests)
            bt.update(seed=seed, method=name, param=float(chosen[name]),
                      avg_exit=float(e.mean()), saving=float(1 - e.mean() / L),
                      n_boot=args.n_boot)
            boots.append(bt)

    df = pd.DataFrame(rows)
    df["p_mcnemar_bonf"] = np.nan
    df["p_delong_bonf"] = np.nan
    for seed in df["seed"].unique():
        for test in ["mcnemar", "delong"]:
            for scope in [df["group"] == "ALL", df["group"] != "ALL"]:
                m = (df["seed"] == seed) & scope
                df.loc[m, f"p_{test}_bonf"] = bonferroni_adjust(
                    df.loc[m, f"p_{test}"].to_numpy())
    for test in ["mcnemar", "delong"]:
        df[f"{test}_sig_raw"] = df[f"p_{test}"] < args.alpha
        df[f"{test}_sig_bonf"] = df[f"p_{test}_bonf"] < args.alpha

    boot = pd.DataFrame(boots)
    out = Path(args.dir)
    df.to_csv(out / f"exp2-{args.tag}-paired-tests-per-seed.csv", index=False)
    boot.to_csv(out / f"exp2-{args.tag}-gmb-bootstrap.csv", index=False)

    summary = df.groupby(["method", "param", "group"], as_index=False).agg(
        seeds=("seed", "nunique"),
        mean_delta_acc=("delta_acc", "mean"),
        mcnemar_sig_raw=("mcnemar_sig_raw", "sum"),
        mcnemar_sig_bonf=("mcnemar_sig_bonf", "sum"),
        mean_delta_auc=("delta_auc", "mean"),
        delong_sig_raw=("delong_sig_raw", "sum"),
        delong_sig_bonf=("delong_sig_bonf", "sum"))
    summary.to_csv(out / f"exp2-{args.tag}-paired-tests-summary.csv", index=False)

    cols = ["seed", "method", "group", "delta_acc", "p_mcnemar",
            "p_mcnemar_bonf", "delta_auc", "p_delong", "p_delong_bonf"]
    sig = df[(df["p_mcnemar"] < args.alpha) | (df["p_delong"] < args.alpha)]
    print("\nRaw p < alpha:")
    print(sig[cols].to_string(index=False) if len(sig) else "None")

    sigb = df[(df["p_mcnemar_bonf"] < args.alpha) | (df["p_delong_bonf"] < args.alpha)]
    print("\nBonferroni p < alpha:")
    print(sigb[cols].to_string(index=False) if len(sigb) else "None")

    print(f"\nGMB bootstrap ({100*(1-args.alpha):.1f}% CI):")
    print(boot[["seed", "method", "avg_exit", "saving", "delta_gmb",
                "ci_low", "ci_high", "significant",
                "significant_bonf"]].to_string(index=False))


if __name__ == "__main__":
    main()