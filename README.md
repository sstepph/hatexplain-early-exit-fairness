# Fairness of Early Exiting in Hate Speech Detection

LIRIS / École Centrale de Lyon — Stéphanie Ounanian, supervised by Julien Velcin.
With Irina Proskurina and Antoine Gourru.

**Question:** when a transformer stops early to save computation, does every
demographic subgroup pay the same price?

---

## Read this first

| If you want to… | Go to |
|---|---|
| understand the setup and what each experiment does | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| see the results without running anything | [`results/`](results/) |
| run the pipeline yourself | "Running it" below |
| look at the old thesis code | [`archive/thesis/`](archive/thesis/) — superseded, kept for reference |

**One thing to know before reading any number:** both encoders are **frozen**.
Nothing is fine-tuned on HateXplain. Only a linear classification head per
layer is trained. This matters for interpreting every result — see
[`EXPERIMENTS.md`](EXPERIMENTS.md#why-the-frozen-setup-matters).

---

## Setup

| | |
|---|---|
| Dataset | HateXplain (Mathew et al., 2021), **official** splits |
| Train / test | 17,305 (val merged in) / **1,924** |
| Models | `bert-base-cased` (12 layers), `answerdotai/ModernBERT-base` (22 layers) |
| Encoder | **frozen** — only per-layer linear heads are trained |
| Seeds | 3 per setting; the seed affects training only, never the split |
| Subgroups | target groups with at least **50** test posts (9 groups) — same threshold in every script |
| Significance | McNemar, DeLong, bootstrap, Bonferroni, α = 0.01 |

Data is not in the repo. Download `dataset.json` and `post_id_divisions.json`
from https://github.com/hate-alert/HateXplain/tree/master/Data into `data/`.

---

## Repository layout

```
src/              the current pipeline — run these
src/tools/        one-off diagnostics, not part of the pipeline
archive/thesis/   code from the Master's thesis; superseded, do not run
results/          committed CSVs and figures — browse without running anything
data/             not in the repo (see above)
outputs/          generated locally, gitignored
```

---

## Running it

Scores are cached after step 1, so steps 2–6 rerun in seconds without a GPU.

```bash
cd src

# 1 — train one linear head per layer on the frozen encoder
for s in 0 1 2; do python train_layer_heads.py --seed $s --tag bert-cls; done
for s in 0 1 2; do python train_layer_heads.py --seed $s \
      --model answerdotai/ModernBERT-base --tag modernbert-cls; done

# 2 — Experiment 1: forced exit at fixed depths
python analyze_forced_exit.py --tag bert-cls

# 3 — prototype margins (needed by Experiment 2)
for s in 0 1 2; do python add_prototype_margins.py --tag bert-cls --seed $s; done

# 4 — Experiment 2: dynamic exit at matched compute
python exp2_early_exit.py --tag bert-cls --saving 0.20
python exp2_early_exit.py --tag bert-cls --saving 0.50

# 5 — significance tests
python stat_tests.py      --tag bert-cls
python stat_tests_exp2.py --tag bert-cls

# 6 — per-subgroup AUC, distance from the mean, max shortfall
python subgroup_gap.py --tag bert-cls
python subgroup_gap.py --tag modernbert-cls
python plot_maxgap.py  --tag bert-cls            # same figure, no bands
```

Step 1 needs a GPU. Everything after it reads the cached
`outputs/probe/{tag}-scores-s{seed}.npz` files and runs on CPU.

---

## Current state of the results

- Overall performance barely changes until the network is cut very deep.
  Mid-depth matches full depth in both models.
- **No Δ GMB is statistically significant**, in either model, at any depth —
  0 of 3 seeds in Experiment 1, and all 18 bootstrap intervals contain zero in
  Experiment 2.
- Five per-subgroup **accuracy** shifts of 12–17 pp survive Bonferroni
  correction, all in ModernBERT. None of the corresponding AUC changes are
  significant.
- Patience-based exiting improves accuracy significantly (McNemar, 2 of 3
  seeds) while AUC is unchanged (DeLong, p = 0.20 and 0.46) — it shifts
  decisions across the boundary rather than ranking better.
- **Each model has one subgroup consistently at the bottom, and it is a
  different subgroup in each.** In BERT the largest shortfall is flat across
  depth (8.6–12.6 pp, clearing the noise reference at 3 of 12 layers), and the
  worst-placed subgroup is the same one at every layer. In ModernBERT the
  shortfall grows with depth (9.8 pp at layer 1 to 17.4 pp at layer 22,
  clearing the reference at 15 of 22 layers).
- The gap is already present at layer 1 in both models, so it is **not produced
  by early exiting**.

Details and interpretation: [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Open items

- Mean pooling has not been tested for ModernBERT (`src/tools/compare_pooling.py`).
- `stat_tests_exp2.py`: entropy can divide by zero on ModernBERT runs where
  float32 scores are exactly 0 or 1. Fix: cast to float64 and clip to
  `[1e-7, 1-1e-7]` before the log. Not yet applied.
- Second dataset not yet added (Measuring Hate Speech is the current candidate).

---

## Credits

`stat_tests.py` is adapted from code provided by Irina Proskurina
(https://github.com/upunaprosk/hate-prototypes).

## References

- Mathew et al. (2021), *HateXplain*
- Borkan et al. (2019), *Nuanced Metrics for Measuring Unintended Bias*
- Xin et al. (2020), *DeeBERT*
- Zhou et al. (2020), *PABEE*
- Proskurina, Carpentier & Velcin (2026), *HatePrototypes*, LREC 2026
  (https://aclanthology.org/2026.lrec-1.343/)
