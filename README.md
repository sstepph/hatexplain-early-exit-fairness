# Fairness of Early Exiting in Hate Speech Detection

Master's internship, LIRIS / École Centrale de Lyon.
Supervisor: Julien Velcin.

Does early exiting in transformer classifiers cost the same for every
demographic subgroup? Two experiments on HateXplain: forced exit at fixed
depths, and dynamic exit under three confidence rules at matched compute.

## Data

HateXplain (Mathew et al., 2021).
Download `dataset.json` and `post_id_divisions.json` from
https://github.com/hate-alert/HateXplain/tree/master/Data into `data/`.

All experiments use the **official** splits from `post_id_divisions.json`
(train 17,305 after merging val, test 1,924). The split is fixed; the seed
affects training only.

## Phase 1 — thesis

Fine-tuned BERT, prototype classification, early exiting, per-group fairness.

| Script | Purpose |
|---|---|
| `model_finetuning.py` | standard fine-tuning, loss at layer 12 |
| `model_finetuning_alllayer.py` | all-layer fine-tuning, loss at all 12 layers |
| `hate_prototypes_bert.py` | layer-wise prototypes, early exit, per-group metrics |
| `check_margins.py` | margin distribution per layer |
| `run_exp1.sh`, `run_exp2.sh` | controlled reruns after the defence |

## Phase 2 — current

**Frozen encoder**, one linear classification head per transformer layer.
Neither BERT nor ModernBERT is fine-tuned on HateXplain in this phase — only
the heads are trained. Models: `bert-base-cased` (12 layers) and
`answerdotai/ModernBERT-base` (22 layers), 3 seeds each.

| Script | Purpose |
|---|---|
| `train_layer_heads.py` | freeze encoder, train one head per layer, cache scores |
| `analyze_forced_exit.py` | Experiment 1 — forced exit at fixed depths, Table 1 |
| `add_prototype_margins.py` | add prototype margins to existing `.npz` (no retraining) |
| `exp2_early_exit.py` | Experiment 2 — entropy, patience, prototype; Table 2 |
| `stat_tests.py` | McNemar, DeLong, stratified bootstrap, Bonferroni (Exp 1) |
| `stat_tests_exp2.py` | same tests applied at each example's exit layer (Exp 2) |
| `compare_pooling.py` | [CLS] vs mean pooling |
| `diagnose_depth0.py` | diagnostic for the constant layer-0 embedding |

`stat_tests.py` is adapted from code provided by Irina Proskurina
(https://github.com/upunaprosk/hate-prototypes).

## Running it

```bash
cd src

# 1. train the per-layer heads (3 seeds)
for s in 0 1 2; do
  python train_layer_heads.py --seed $s --tag bert-cls
done

# ModernBERT
for s in 0 1 2; do
  python train_layer_heads.py --seed $s --model answerdotai/ModernBERT-base --tag modernbert
done

# 2. Experiment 1 — forced exit
python analyze_forced_exit.py --tag bert-cls

# 3. prototype margins, then Experiment 2
for s in 0 1 2; do python add_prototype_margins.py --tag bert-cls --seed $s; done
python exp2_early_exit.py --tag bert-cls --saving 0.20
python exp2_early_exit.py --tag bert-cls --saving 0.50

# 4. significance tests
python stat_tests.py       --tag bert-cls
python stat_tests_exp2.py  --tag bert-cls
```

Scores are cached in `outputs/probe/{tag}-scores-s{seed}.npz`, so steps 2–4
rerun without touching the GPU.

## Method notes

- The shallowest exit point is transformer layer 1, not the embedding output:
  `hidden_states[0]` at `[CLS]` is identical for every post and yields a
  constant prediction (AUC exactly 0.500). See `diagnose_depth0.py`.
- The baseline in every comparison is the **full-depth** model on the same
  test posts, so all tests are paired.
- GMB is the generalised power mean (p = −5) of per-subgroup AUCs
  (Borkan et al., 2019) — sensitive to the weakest subgroup.
- Significance at α = 0.01 with Bonferroni correction across settings.

## Known issues

- `stat_tests_exp2.py`: entropy can divide by zero on ModernBERT runs where
  float32 softmax scores are exactly 0 or 1. Fix is to cast to float64 and
  clip to `[1e-7, 1 - 1e-7]` before taking logs. Not yet applied.

## References

- Mathew et al. (2021), *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*
- Borkan et al. (2019), *Nuanced Metrics for Measuring Unintended Bias*
- Xin et al. (2020), *DeeBERT* — entropy-based early exiting
- Zhou et al. (2020), *PABEE* — patience-based early exiting
- Proskurina, Carpentier & Velcin (2026), *HatePrototypes: Interpretable and
  Transferable Representations for Implicit and Explicit Hate Speech Detection*,
  LREC 2026 — prototype-based early exiting
  (https://aclanthology.org/2026.lrec-1.343/)