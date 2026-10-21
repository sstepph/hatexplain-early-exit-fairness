# Archive — Master's thesis code

**Superseded. Kept for reference. Do not run these as part of the current
pipeline.**

This is the code from the Master's thesis phase, before the experiments were
rebuilt. It is kept because the thesis results came from it and because the
all-layer training experiment may be revisited.

It differs from the current pipeline in ways that make the numbers **not
comparable**:

| | thesis code (here) | current pipeline (`src/`) |
|---|---|---|
| Encoder | fine-tuned on HateXplain | **frozen** |
| Split | random 80/20, seeded | official HateXplain split |
| Test size | varies with the seed | fixed, 1,924 posts |
| Runs | single run per setting | 3 seeds, mean ± std |
| Significance | none | McNemar, DeLong, bootstrap, Bonferroni |
| Subgroup labels | majority vote (after fixing a `set()` ordering bug) | same |

Any `auc` reported by these scripts may be **PR-AUC**, not ROC-AUC. Check
before comparing to anything in `results/`.

## Files

| File | What it did |
|---|---|
| `model_finetuning.py` | standard fine-tuning, loss at the final layer only |
| `model_finetuning_alllayer.py` | all-layer fine-tuning, loss at all 12 layers (Antoine Gourru's approach) |
| `hate_prototypes_bert.py` | layer-wise prototypes, early exit, per-group metrics |
| `check_margins.py` | diagnostic: cosine margin distribution per layer |
| `run_exp1.sh`, `run_exp2.sh` | controlled reruns after the defence |
