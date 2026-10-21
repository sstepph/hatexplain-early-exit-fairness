"""
Side-by-side comparison of CLS vs mean pooling.

Reads the CSVs written by analyze_forced_exit.py.

Usage:
    python compare_pooling.py
    python compare_pooling.py --a bert-cls --b bert-mean
"""
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="bert-cls")
ap.add_argument("--b", default="bert-mean")
ap.add_argument("--dir", default="../outputs/probe")
args = ap.parse_args()
d = Path(args.dir)

A = pd.read_csv(d / f"{args.a}-overall.csv")
B = pd.read_csv(d / f"{args.b}-overall.csv")

print(f"OVERALL — {args.a} vs {args.b}\n")
print(f"{'depth':>6}  {'acc '+args.a:>16} {'acc '+args.b:>16}   "
      f"{'GMB '+args.a:>16} {'GMB '+args.b:>16}")
print("-" * 80)
for i in range(len(A)):
    print(f"{A.loc[i,'depth']:>6}  "
          f"{A.loc[i,'acc']:>14.2f}% {B.loc[i,'acc']:>14.2f}%   "
          f"{A.loc[i,'gmb']:>14.2f}% {B.loc[i,'gmb']:>14.2f}%")

SA = pd.read_csv(d / f"{args.a}-subgroups.csv").set_index("group")
SB = pd.read_csv(d / f"{args.b}-subgroups.csv").set_index("group")
print(f"\nSUBGROUP AUC AT 100% DEPTH\n")
print(f"{'group':<12} {args.a:>12} {args.b:>12} {'diff':>8}")
print("-" * 48)
for g in SA.index:
    a, b = SA.loc[g, "auc_100"], SB.loc[g, "auc_100"]
    print(f"{g:<12} {a:>11.2f}% {b:>11.2f}% {b-a:>+7.2f}")

out = A.merge(B, on=["depth", "layer"], suffixes=(f"_{args.a}", f"_{args.b}"))
out.to_csv(d / f"compare-{args.a}-vs-{args.b}.csv", index=False)
print(f"\nSaved compare-{args.a}-vs-{args.b}.csv")
