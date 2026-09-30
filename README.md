# Fairness of Early Exiting in Hate Speech Detection

Master's internship, LIRIS / École Centrale de Lyon.
Supervisor: Julien Velcin.

## Data
HateXplain (Mathew et al., 2021)
Download `dataset.json` and `post_id_divisions.json` from
https://github.com/hate-alert/HateXplain/tree/master/Data into `data/`.

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
Frozen encoder with a classification head per layer. Compares three exit
criteria at matched compute.

| Script | Purpose |
|---|---|
| `train_layer_heads.py` | freeze encoder, train one head per layer |
| `analyze_forced_exit.py` | Experiment 1 — forced exit at fixed depths |
| `add_prototype_margins.py` | prototype margins for Experiment 2 |
| `exp2_early_exit.py` | Experiment 2 — entropy, patience, prototype |
| `compare_pooling.py` | [CLS] vs mean pooling |
| `diagnose_depth0.py` | per-group significance diagnostic |

## Environment
Python 3.6 for Phase 1 (`requirements-py36.txt`),
Python 3.11 for ModernBERT experiments (`requirements-py311.txt`).

## Note
This repository works with a hate speech corpus. Examples in outputs may
contain offensive language.
