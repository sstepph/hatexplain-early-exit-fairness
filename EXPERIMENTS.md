# Experiments — what is done, how, and how to read the numbers

This is the shared reference for the project. It explains the setup once, then
each experiment, then every metric and test used.

---

## 1. The setup, once

Both models are used **off the shelf with the encoder frozen**:

```python
enc = AutoModel.from_pretrained(args.model, output_hidden_states=True)
for p in enc.parameters():
    p.requires_grad = False
```

The only trained parameters are `L+1` linear heads, one per layer, each mapping
a pooled hidden state to two classes. Neither BERT nor ModernBERT is fine-tuned
on HateXplain.

| | |
|---|---|
| Pooling | first token (`[CLS]`); mean pooling available via `--pool mean` |
| Split | official HateXplain `post_id_divisions.json`, val merged into train |
| Train / test | 17,305 / 1,924 |
| Seeds | 0, 1, 2 — affect head initialisation and batch order only |
| Class weights | balanced, computed from the training labels |
| Layer 0 | the embedding output; identical at `[CLS]` for every post, so it is stored but never used. Analyses start at layer 1. |

### Why the frozen setup matters

In a fine-tuned model, the last layer is the best layer — that is what training
optimises. In a **frozen** model it is not: the last layer is specialised
towards the pre-training objective, not towards hate speech. Middle layers
classify better.

This is visible in accuracy alone:

| | best middle layer | last layer |
|---|---|---|
| BERT | 71.9% (layer 3) | 70.5% (layer 12) |
| ModernBERT | 69.5% (layer 1) | 68.2% (layer 22) |

**Consequence:** the full-depth model is the *weakest* reference point, not the
strongest. Any "improvement" from exiting early is partly just "this layer is
better than the last layer." Keep this in mind whenever a Δ against full depth
looks positive.

A second caveat for ModernBERT specifically: it has no next-sentence-prediction
objective, so its first token was never trained as a sentence summary the way
BERT's was. `[CLS]` pooling may therefore understate it. Untested — see open
items in the README.

---

## 2. Experiment 1 — forced exit

Every post stops at the same fixed depth: 0%, 25%, 50%, 75%, 100% of the
network. This isolates the effect of depth alone, with no confidence rule
involved.

`src/analyze_forced_exit.py`. Depth is clamped so the shallowest point is
layer 1, never the embedding layer.

Compared against the full-depth model on the same test posts, so every test is
paired.

---

## 3. Experiment 2 — dynamic exit

Each post stops when the model is confident enough. Three rules:

| Rule | Signal | Source |
|---|---|---|
| Entropy | Shannon entropy of the head's softmax falls below a threshold | DeeBERT (Xin et al., 2020) |
| Patience | the prediction stays the same for *n* consecutive layers | PABEE (Zhou et al., 2020) |
| Prototype | cosine margin to the two class centroids exceeds a threshold | HatePrototypes (Proskurina et al., 2026) |

Thresholds are tuned so that all three reach the **same compute saving**, at two
budgets: 20% and 50%. Saving is `1 − mean(exit layer) / L`.

Tuning to matched compute is what makes the three rules comparable — otherwise
a rule that exits later would look better simply by using more layers.

`src/exp2_early_exit.py`, then `src/stat_tests_exp2.py`, which gathers each
example's prediction at its own exit layer before testing.

---

## 4. Subgroup analysis

`src/subgroup_gap.py`. Reads the cached scores only — no retraining.

For every layer:

1. AUC for each subgroup with at least 50 test posts (9 groups)
2. the **simple mean** of those subgroup AUCs (not GMB)
3. for each subgroup, `mean AUC − subgroup AUC` — positive means behind the average
4. the **max** of (3) — the largest shortfall at that layer

Step 4 is the measure proposed by Julien. It answers "is one subgroup behind
the others," which GMB does not.

### Why GMB was not answering that question

GMB (generalised power mean, p = −5, Borkan et al. 2019) summarises per-subgroup
AUCs with the weakest groups weighted most heavily. It measures **how good the
weakest groups are**, not **how far behind** they are.

If every subgroup improves by 5 points, GMB rises although no gap has narrowed.
Combined with the frozen-baseline effect above, that is the most likely reason
Δ GMB came out positive for early exits. The max-gap measure does not have this
problem.

### The noise reference — the important part

The max over subgroups is a **noisy statistic**. Subgroup sizes range from 55 to
338 posts, so each subgroup AUC carries sampling error, and the largest of nine
noisy numbers is large by construction.

The threshold matters here. An earlier run used n ≥ 30, which added three groups
of 31–41 posts; those dominated the maximum and pushed the reference band up to
16–18 pp, hiding everything. At n ≥ 50 — the same threshold `stat_tests.py`
uses — the band falls to 11–12.5 pp and real gaps become visible.

To quantify this, `subgroup_gap.py` shuffles the subgroup labels at random 300
times and recomputes the max gap each time. With shuffled labels, subgroup
membership carries no information, so whatever gap remains is pure chance.

That reference comes out at **11–12.5 pp** at n ≥ 50.

**How to read the figures:** a point inside the grey band is consistent with no
real subgroup difference. Only points above it are worth interpreting.

**Caveat:** the layers are not independent tests — adjacent layers share the same
posts and nearly the same representations. Counts like "15 of 22 layers" describe
a pattern, not 15 separate results.

---

## 5. Metrics

| Metric | What it measures | Note |
|---|---|---|
| Accuracy | fraction of posts classified correctly | depends on the decision threshold |
| AUC (ROC) | probability a random hateful post is ranked above a random non-hateful one | threshold-free |
| GMB | power mean (p = −5) of per-subgroup AUCs | a **level**, not a gap |
| Max gap | largest `mean AUC − subgroup AUC` | a **gap**; needs the noise reference |

Accuracy and AUC can move independently. A rule that improves accuracy without
improving AUC is shifting predictions across the decision boundary rather than
ranking better — this is what patience does.

Thesis-era files reporting `auc` may contain **PR-AUC**, not ROC-AUC. Do not
compare those to current numbers without checking.

---

## 6. Significance testing

| Test | Used for | Why this one |
|---|---|---|
| Exact McNemar | accuracy, two settings | correct test for paired binary outcomes |
| Paired DeLong | AUC, two settings | compares two AUCs on the same examples |
| Stratified paired bootstrap | GMB difference | no closed form for GMB |
| Permutation (label shuffling) | max gap | gives the distribution under "no subgroup effect" |
| Bonferroni | all of the above | many settings are tested at once |

α = 0.01. Every comparison is paired — same posts, same labels, two settings.

### Statistical power

With 1,924 test posts and 50–350 per subgroup, only large effects can reach
significance. At α = 0.01, a paired binary test needs roughly a
`2.6 / √n` accuracy gap to detect a difference.

**A non-significant result here means "not detectable at this sample size," not
"no effect."** This is the main argument for adding a second dataset.

---

## 7. Current results

**Experiment 1 — forced exit.** Overall accuracy barely moves until the network
is cut very deep. Every Δ GMB is significant in **0 of 3 seeds**, for both
models, at every depth.

**Experiment 2 — dynamic exit.** All 18 bootstrap confidence intervals contain
zero. No subgroup-fairness change is detectable at either compute budget.
Patience improves accuracy significantly under McNemar in two of three seeds,
while DeLong on the same rows gives p = 0.20 and 0.46 — accuracy up, ranking
unchanged.

**Per-subgroup accuracy.** Five shifts of 12–17 pp survive Bonferroni
correction, all in ModernBERT: African −12.7 pp at 50% depth (seed 0),
Islam +14.2 pp and Jewish +16.7 pp at 50% (seed 1), Jewish +13.6 pp (seed 1)
and +13.0 pp (seed 2) at 75%. Every corresponding AUC test fails, so these are
changes in decisions, not in ranking. No comparable effect survives in BERT.

**Subgroup analysis (n ≥ 50).** Each model has one subgroup consistently at the
bottom, and it is a different subgroup in each.

| | BERT (12 layers) | ModernBERT (22 layers) |
|---|---|---|
| Layers above the noise reference | 3 of 12 | 15 of 22 |
| Worst-placed subgroup | Caucasian, at all 12 layers | Jewish, at 17 of 22 |
| Max gap | 8.6–12.6 pp, flat with depth | 9.8–17.4 pp, growing with depth |
| Mean subgroup AUC | 71–73%, stable | 70% → 68%, declining |

Two qualifications. The gap is already present at layer 1 in both models, so it
is not produced by early exiting — in ModernBERT, exiting earlier means a
*smaller* gap. And the bootstrap intervals are wide, so the existence of the gap
is better supported than its size.

---

## 8. Known inconsistencies

- ~~**Subgroup list.**~~ Resolved: every script now uses n ≥ 50 (9 subgroups).
- **Entropy underflow.** `stat_tests_exp2.py` can divide by zero on ModernBERT
  runs where float32 softmax scores are exactly 0 or 1. Fix identified
  (float64 + clip to `[1e-7, 1-1e-7]`), not yet applied.
- **Pooling.** Mean pooling has been run for BERT but not for ModernBERT.
