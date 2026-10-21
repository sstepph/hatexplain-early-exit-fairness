"""
hate_prototypes_bert.py
Steps 2-4 of the pipeline:
  - Extract CLS token embeddings at each of BERT's 12 layers
  - Build hate / not-hate prototypes per layer
  - Classify using last layer as baseline
  - Early exiting using layer-specific percentile thresholds
  - Per-group fairness analysis

Adapted from Irina Proskurina's hate_prototypes_bert.py
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Block 1 -- Load the fine-tuned model
# ---------------------------------------------------------------------------
def load_model(checkpoint_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModel.from_pretrained(
        checkpoint_dir,
        output_hidden_states=True,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Block 2 -- Load and split the data
# ---------------------------------------------------------------------------
def get_primary_group(targets_list):
    real = [t for t in targets_list if t != "None"]
    if not real:
        return "none"
    return Counter(real).most_common(1)[0][0]


def load_hatexplain(data_path, seed=0):
    with open(data_path) as f:
        data = json.load(f)
    rows = []
    for post_id, post_data in data.items():
        annotators = post_data["annotators"]
        labels = [a["label"] for a in annotators]
        majority_label = Counter(labels).most_common(1)[0][0]
        targets = []
        for a in annotators:
            targets.extend(a["target"])
        rows.append({
            "post_id":   post_id,
            "label_raw": majority_label,
            "targets":   targets,
            "post":      " ".join(post_data["post_tokens"]),
        })
    df = pd.DataFrame(rows)
    df["label"] = df["label_raw"].apply(
        lambda x: 1 if x in ["hatespeech", "offensive"] else 0
    )
    df["group"] = df["targets"].apply(get_primary_group)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Block 3 -- Extract CLS token at every layer
# ---------------------------------------------------------------------------
def extract_cls_embeddings(texts, tokenizer, model, device,
                           max_len=128, batch_size=32):
    all_hidden_states = [[] for _ in range(12)]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, truncation=True, padding=True,
            max_length=max_len, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        for layer_idx in range(12):
            cls = outputs.hidden_states[layer_idx + 1][:, 0, :]
            all_hidden_states[layer_idx].append(cls.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} posts")
    return [np.concatenate(l, axis=0) for l in all_hidden_states]


# ---------------------------------------------------------------------------
# Block 4 -- Build prototypes per layer
# ---------------------------------------------------------------------------
def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def build_prototypes(embeddings_per_layer, labels):
    labels = np.array(labels)
    hate_mask     = labels == 1
    not_hate_mask = labels == 0
    prototypes = []
    for layer_idx in range(12):
        emb = l2_normalize(embeddings_per_layer[layer_idx])
        hate_proto     = emb[hate_mask].mean(axis=0)
        not_hate_proto = emb[not_hate_mask].mean(axis=0)
        hate_proto     = hate_proto     / np.linalg.norm(hate_proto)
        not_hate_proto = not_hate_proto / np.linalg.norm(not_hate_proto)
        prototypes.append({"hate": hate_proto, "not_hate": not_hate_proto})
    return prototypes


# ---------------------------------------------------------------------------
# Block 5 -- Classify + metrics
# ---------------------------------------------------------------------------
def cosine_similarity(vectors, prototype):
    return vectors @ prototype


def classify_with_prototypes(embeddings, prototypes_at_layer):
    emb          = l2_normalize(embeddings)
    hate_sim     = cosine_similarity(emb, prototypes_at_layer["hate"])
    not_hate_sim = cosine_similarity(emb, prototypes_at_layer["not_hate"])
    preds        = (hate_sim > not_hate_sim).astype(int)
    scores       = hate_sim - not_hate_sim
    return preds, scores


def compute_metrics(labels, preds, scores):
    acc    = float(accuracy_score(labels, preds))
    f1_bin = float(f1_score(labels, preds, average="binary",  zero_division=0))
    f1_mac = float(f1_score(labels, preds, average="macro",   zero_division=0))
    auc    = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    return {"accuracy": acc, "f1_binary": f1_bin, "f1_macro": f1_mac, "auc": auc}


def evaluate_per_group(labels, preds, scores, groups, min_samples=10):
    labels = np.array(labels)
    preds  = np.array(preds)
    scores = np.array(scores)
    groups = np.array(groups)
    results = []
    for group in sorted(set(groups)):
        mask = groups == group
        if mask.sum() < min_samples:
            continue
        g_labels = labels[mask]
        g_preds  = preds[mask]
        g_scores = scores[mask]
        auc = round(float(roc_auc_score(g_labels, g_scores)), 3) \
              if len(set(g_labels)) > 1 else "N/A"
        recall = round(float(f1_score(g_labels, g_preds,
                                      average="binary", zero_division=0)), 3)
        not_hate_mask = g_labels == 0
        if not_hate_mask.sum() == 0:
            fpr = "N/A"
        else:
            fp  = ((g_preds == 1) & not_hate_mask).sum()
            fpr = round(float(fp / not_hate_mask.sum()), 3)
        results.append({
            "Group":  group,
            "N":      int(mask.sum()),
            "AUC":    auc,
            "Recall": recall,
            "FPR":    fpr,
        })
    return pd.DataFrame(results).set_index("Group")


# ---------------------------------------------------------------------------
# Block 6 -- Percentile-based early exiting
# ---------------------------------------------------------------------------
def compute_layer_thresholds(train_embeddings_per_layer, prototypes,
                              percentile=90):
    """
    For each layer, compute the margin threshold = top (100-percentile)%
    of absolute margins seen on the training set.

    percentile=90 means: exit when margin is in the top 10%.
    percentile=95 means: exit when margin is in the top 5%.

    We use the TRAINING set to calibrate thresholds so the test set
    stays unseen until final evaluation (no data leakage).
    """
    thresholds = []
    print(f"\nCalibrating layer thresholds (top {100-percentile}% of training margins):")
    print(f"{'Layer':>6} | {'Threshold':>12} | {'P50 margin':>12} | {'P95 margin':>12}")
    print("-" * 52)

    for layer_idx in range(12):
        emb          = l2_normalize(train_embeddings_per_layer[layer_idx])
        hate_sim     = cosine_similarity(emb, prototypes[layer_idx]["hate"])
        not_hate_sim = cosine_similarity(emb, prototypes[layer_idx]["not_hate"])
        margins      = np.abs(hate_sim - not_hate_sim)

        # threshold = the (percentile)th percentile of abs margins
        # e.g. percentile=90 → threshold = 90th percentile
        # meaning only the top 10% of margins exceed this and exit early
        threshold = np.percentile(margins, percentile)
        thresholds.append(threshold)

        print(f"  L{layer_idx+1:>2}  |  {threshold:>10.5f}  |"
              f"  {np.percentile(margins,50):>10.5f}  |"
              f"  {np.percentile(margins,95):>10.5f}")

    return thresholds


def early_exit_classify_percentile(embeddings_per_layer, prototypes,
                                    thresholds):
    """
    Classify posts using layer-specific percentile thresholds.

    At each layer, exit if abs(margin) >= threshold for that layer.
    If no layer is confident enough, use layer 12 (last layer).

    Args:
        thresholds: list of 12 floats from compute_layer_thresholds()

    Returns:
        preds:       predicted labels (0 or 1), shape (N,)
        scores:      margin at exit layer, shape (N,)
        exit_layers: which layer each post exited at (1-12), shape (N,)
    """
    N = embeddings_per_layer[0].shape[0]

    preds       = np.zeros(N, dtype=int)
    scores      = np.zeros(N, dtype=float)
    exit_layers = np.full(N, 12, dtype=int)
    remaining   = np.ones(N, dtype=bool)

    for layer_idx in range(12):
        if not remaining.any():
            break

        emb          = l2_normalize(embeddings_per_layer[layer_idx])
        hate_sim     = cosine_similarity(emb, prototypes[layer_idx]["hate"])
        not_hate_sim = cosine_similarity(emb, prototypes[layer_idx]["not_hate"])
        margin       = hate_sim - not_hate_sim

        # exit if this post's abs margin exceeds THIS layer's threshold
        confident = remaining & (np.abs(margin) >= thresholds[layer_idx])

        preds[confident]       = (margin[confident] > 0).astype(int)
        scores[confident]      = margin[confident]
        exit_layers[confident] = layer_idx + 1

        remaining[confident] = False

    # posts still remaining after all 12 layers → use layer 12
    if remaining.any():
        emb          = l2_normalize(embeddings_per_layer[11])
        hate_sim     = cosine_similarity(emb, prototypes[11]["hate"])
        not_hate_sim = cosine_similarity(emb, prototypes[11]["not_hate"])
        margin       = hate_sim[remaining] - not_hate_sim[remaining]
        preds[remaining]  = (margin > 0).astype(int)
        scores[remaining] = margin

    return preds, scores, exit_layers


def analyze_exit_layers(exit_layers, groups, labels, min_samples=10):
    """Where does each group exit? This is the core bias question."""
    groups      = np.array(groups)
    exit_layers = np.array(exit_layers)
    labels      = np.array(labels)

    results = []
    for group in sorted(set(groups)):
        mask = groups == group
        if mask.sum() < min_samples:
            continue
        g_exits  = exit_layers[mask]
        g_labels = labels[mask]
        results.append({
            "Group":          group,
            "N":              int(mask.sum()),
            "Avg exit layer": round(float(g_exits.mean()), 2),
            "Early exit %":   f"{(g_exits < 12).mean():.1%}",
            "Hate base rate": round(float((g_labels == 1).mean()), 3),
        })
    return pd.DataFrame(results).set_index("Group")


def sweep_percentiles(train_embeddings, test_embeddings, prototypes,
                       train_labels, test_labels, test_groups,
                       percentiles=[80, 85, 90, 95, 99]):
    """
    Sweep multiple percentile values and show the accuracy/speed tradeoff.
    Helps choose the best percentile before committing to per-group analysis.
    """
    test_labels_arr = np.array(test_labels)
    print(f"\n{'Pctile':>8} | {'Top%':>6} | {'Accuracy':>10} | {'F1':>8} | {'AUC':>8} | {'Avg exit L':>12} | {'Early exits':>12}")
    print("-" * 80)

    best = None
    for p in percentiles:
        thresholds = compute_layer_thresholds(
            train_embeddings, prototypes, percentile=p
        )
        preds, scores, exit_layers = early_exit_classify_percentile(
            test_embeddings, prototypes, thresholds
        )
        acc       = float(accuracy_score(test_labels_arr, preds))
        f1        = float(f1_score(test_labels_arr, preds, average="binary", zero_division=0))
        auc       = float(roc_auc_score(test_labels_arr, scores))
        avg_exit  = float(exit_layers.mean())
        early_pct = float((exit_layers < 12).mean())

        print(f"{p:>8} | {100-p:>5}% | {acc:>10.4f} | {f1:>8.4f} | {auc:>8.4f} | {avg_exit:>12.2f} | {early_pct:>11.1%}")

        if best is None:
            best = p
        if acc >= 0.74 and early_pct > 0.0:
            best = p

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path",      type=str,
                    default="../data/dataset.json")
    ap.add_argument("--checkpoint_dir", type=str,
                    default=None,
                    help="If not set, auto-resolves to hatexplain-bert-base-cased-s{seed}")
    ap.add_argument("--max_len",        type=int, default=128)
    ap.add_argument("--batch_size",     type=int, default=32)
    ap.add_argument("--seed",           type=int, default=0)
    ap.add_argument("--percentile",     type=int, default=90,
                    help="Percentile threshold. 90 = top 10%% exit early.")
    ap.add_argument("--sweep",          action="store_true",
                    help="Sweep multiple percentile values to find best tradeoff.")
    ap.add_argument("--out_dir",        type=str,
                    default="../outputs/predictions")
    ap.add_argument("--tag", type=str, default="std",
                    help="Prefix for output filenames.")
    ap.add_argument("--fixed_delta", type=float, default=None,
                    help="If set, use this fixed threshold at all layers "
                         "instead of per-layer percentiles.")
    ap.add_argument("--min_layer", type=int, default=1,
                    help="Earliest layer at which a post may exit.")
    args = ap.parse_args()

    # auto-resolve checkpoint path from seed if not explicitly set
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"../outputs/checkpoints/hatexplain-bert-base-cased-s{args.seed}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- load model ---
    print("\nLoading fine-tuned BERT...")
    tokenizer, model = load_model(args.checkpoint_dir, device)

    # --- load data ---
    print("\nLoading HateXplain data...")
    train_df, test_df = load_hatexplain(args.data_path, seed=args.seed)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # --- extract embeddings ---
    print("\nExtracting training embeddings...")
    train_embeddings = extract_cls_embeddings(
        train_df["post"].tolist(), tokenizer, model, device,
        max_len=args.max_len, batch_size=args.batch_size
    )
    print("\nExtracting test embeddings...")
    test_embeddings = extract_cls_embeddings(
        test_df["post"].tolist(), tokenizer, model, device,
        max_len=args.max_len, batch_size=args.batch_size
    )

    # --- build prototypes ---
    print("\nBuilding prototypes...")
    prototypes = build_prototypes(train_embeddings, train_df["label"].tolist())

    labels = test_df["label"].tolist()
    groups = test_df["group"].tolist()

    # --- BASELINE: layer 12, no early exit ---
    print("\n=== Baseline (layer 12, no early exit) ===")
    base_preds, base_scores = classify_with_prototypes(
        test_embeddings[11], prototypes[11]
    )
    base_metrics = compute_metrics(labels, base_preds, base_scores)
    for k, v in base_metrics.items():
        print(f"  {k}: {round(v,4)}")

    print("\n=== Baseline per-group ===")
    base_group = evaluate_per_group(labels, base_preds, base_scores, groups)
    print(base_group.to_string())

    # --- SWEEP (optional) ---
    if args.sweep:
        print("\n=== Percentile sweep ===")
        best_p = sweep_percentiles(
            train_embeddings, test_embeddings, prototypes,
            train_df["label"].tolist(), labels, groups,
            percentiles=[70, 75, 80, 85, 90, 95, 99]
        )
        print(f"\nRecommended percentile: {best_p}")

    # --- EARLY EXIT ---
    if args.fixed_delta is not None:
        print(f"\n=== Early exit (fixed delta={args.fixed_delta}, "
              f"min_layer={args.min_layer}) ===")
        thresholds = [float("inf")] * (args.min_layer - 1) + \
                     [args.fixed_delta] * (12 - args.min_layer + 1)
    else:
        print(f"\n=== Early exit (top {100-args.percentile}%, "
              f"percentile={args.percentile}) ===")
        thresholds = compute_layer_thresholds(
            train_embeddings, prototypes, percentile=args.percentile
        )
    ee_preds, ee_scores, exit_layers = early_exit_classify_percentile(
        test_embeddings, prototypes, thresholds
    )

    ee_metrics = compute_metrics(labels, ee_preds, ee_scores)
    print("\n=== Early exit overall results ===")
    for k, v in ee_metrics.items():
        print(f"  {k}: {round(v,4)}")

    avg_exit  = np.array(exit_layers).mean()
    early_pct = (np.array(exit_layers) < 12).mean()
    print(f"\n  Average exit layer : {avg_exit:.2f} / 12")
    print(f"  Posts exiting early: {early_pct:.1%}")

    # --- comparison ---
    print("\n=== Baseline vs Early exit ===")
    print(f"  {'Metric':<12} {'Baseline':>10} {'Early exit':>12} {'Diff':>10}")
    print("  " + "-" * 46)
    for k in ["accuracy", "f1_binary", "auc"]:
        b = base_metrics[k]
        e = ee_metrics[k]
        d = e - b
        print(f"  {k:<12} {b:>10.4f} {e:>12.4f} {d:>+10.4f}")

    # --- exit layer per group ---
    print("\n=== Exit layer analysis per group ===")
    exit_df = analyze_exit_layers(exit_layers, groups, labels)
    print(exit_df.to_string())

    # --- per-group fairness after early exit ---
    print("\n=== Per-group fairness (early exit) ===")
    ee_group = evaluate_per_group(labels, ee_preds, ee_scores, groups)
    print(ee_group.to_string())

    # --- per-group comparison: baseline vs early exit ---
    print("\n=== Per-group AUC: baseline vs early exit ===")
    print(f"  {'Group':<14} {'Base AUC':>10} {'EE AUC':>10} {'Diff':>8} {'Base FPR':>10} {'EE FPR':>10} {'Diff':>8}")
    print("  " + "-" * 76)
    for group in base_group.index:
        if group not in ee_group.index:
            continue
        b_auc = base_group.loc[group, "AUC"]
        e_auc = ee_group.loc[group, "AUC"]
        b_fpr = base_group.loc[group, "FPR"]
        e_fpr = ee_group.loc[group, "FPR"]
        if isinstance(b_auc, float) and isinstance(e_auc, float):
            d_auc = e_auc - b_auc
            d_fpr = e_fpr - b_fpr if isinstance(b_fpr, float) and isinstance(e_fpr, float) else "N/A"
            d_fpr_str = f"{d_fpr:>+8.3f}" if isinstance(d_fpr, float) else f"{'N/A':>8}"
            print(f"  {group:<14} {b_auc:>10.3f} {e_auc:>10.3f} {d_auc:>+8.3f} {str(b_fpr):>10} {str(e_fpr):>10} {d_fpr_str}")

    # --- save ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path("../outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / f"{args.tag}-baseline-metrics-s{args.seed}.json", "w") as f:
        json.dump({k: float(v) for k, v in base_metrics.items()
                   if isinstance(v, float)}, f, indent=2)
    with open(metrics_dir / f"{args.tag}-earlyexit-metrics-s{args.seed}.json", "w") as f:
        json.dump({k: float(v) for k, v in ee_metrics.items()
                   if isinstance(v, float)}, f, indent=2)


    base_group.to_csv(out_dir / f"{args.tag}-pergroup-baseline-s{args.seed}.csv")
    ee_group.to_csv(out_dir / f"{args.tag}-pergroup-earlyexit-s{args.seed}.csv")
    exit_df.to_csv(out_dir / f"{args.tag}-exit-layer-analysis-s{args.seed}.csv")
    print(f"\nAll results saved to {out_dir}/")

    
if __name__ == "__main__":
    main()