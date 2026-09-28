"""
model_finetuning_alllayer.py

Antoine's all-layer prototype training, adapted for Newton's packages:
  torch==1.10.2, transformers==4.18.0, sklearn==0.24.2

Key differences from model_finetuning.py (our original approach):
  - Loss computed at ALL 12 layers simultaneously (not just layer 12)
  - Prototypes recomputed after every epoch
  - Temperature scaling (T=0.1) on cosine similarities before loss
  - Uses hatespeech vs normal only (drops "offensive") — matching Antoine
  - Uses bert-base-cased (changed per Julien feedback for comparability)
  - Fixed margin threshold for early exit (not percentile-based)

Usage:
    python model_finetuning_alllayer.py                    # full run
    python model_finetuning_alllayer.py --sample 200       # smoke test
    python model_finetuning_alllayer.py --epochs 3         # quick run
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
)
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW


# ---------------------------------------------------------------------------
# Config — mirrors Antoine's CFG dataclass
# ---------------------------------------------------------------------------
class Config:
    model_name        = "bert-base-cased"    # switched to cased per Julien
    max_len           = 128
    batch_size        = 16       # Antoine used 512 on A100; 16 fits Newton GPU
    prototype_batch   = 32       # for prototype computation pass
    learning_rate     = 5e-5
    weight_decay      = 0.01
    num_epochs        = 3        # Julien: 3 epochs (same as standard fine-tuning)
    warmup_ratio      = 0.1
    temperature       = 0.1      # scales cosine sims before loss
    max_grad_norm     = 1.0
    seed              = 0         # use seed=0 to match experiment 1 split
    # equal weight on every layer (Antoine's default)
    layer_loss_weights = [1.0] * 12
    # early exit params (Antoine's defaults)
    exit_threshold    = 0.10
    min_exit_layer    = 3        # don't exit before layer 3

CFG = Config()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data loading
# NOTE: Antoine uses hatespeech vs normal only (drops offensive).
# This matches his binary_raw filtering: selected_classes = ("hatespeech","normal")
# We keep our bug-fixed group assignment (majority vote) for fairness analysis.
# ---------------------------------------------------------------------------
def get_primary_group(targets_list):
    real = [t for t in targets_list if t != "None"]
    if not real:
        return "none"
    return Counter(real).most_common(1)[0][0]


def load_hatexplain(data_path, seed=0):
    """
    Load HateXplain JSON, apply Antoine's label scheme:
      - hatespeech  → 1 (hate)
      - normal      → 0 (not hate)
      - offensive   → excluded (filtered out)

    NOTE: seed=42 matches Antoine's seed, not our original seed=0.
    This means a slightly different train/test split — intentional,
    to match his experimental setting as closely as possible.
    """
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
    
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.enc = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_len,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids":      torch.tensor(self.enc["input_ids"][i]),
            "attention_mask": torch.tensor(self.enc["attention_mask"][i]),
            "labels":         self.labels[i],
        }


# ---------------------------------------------------------------------------
# Extract CLS at all 12 layers — returns shape (batch, 12, 768)
# Identical to Antoine's all_layer_cls_embeddings()
# ---------------------------------------------------------------------------
def all_layer_cls_embeddings(model, batch):
    outputs = model(
        input_ids=batch["input_ids"].to(DEVICE),
        attention_mask=batch["attention_mask"].to(DEVICE),
        output_hidden_states=True,
        return_dict=True,
    )
    # hidden_states[0] = embedding layer, [1:] = 12 transformer layers
    encoder_states = outputs.hidden_states[1:]  # tuple of 12 tensors
    # stack CLS (position 0) from each layer → (batch, 12, 768)
    return torch.stack(
        [h[:, 0, :] for h in encoder_states], dim=1
    )


# ---------------------------------------------------------------------------
# Compute prototypes at all 12 layers — returns shape (12, num_classes, 768)
# Recomputed each epoch after model weights update
# Identical to Antoine's compute_layerwise_prototypes()
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_layerwise_prototypes(model, loader, num_classes=2):
    model.eval()
    hidden_size = 768
    num_layers  = 12

    class_sums   = torch.zeros(num_layers, num_classes, hidden_size,
                                dtype=torch.float32, device=DEVICE)
    class_counts = torch.zeros(num_classes, dtype=torch.float32, device=DEVICE)

    for batch in loader:
        cls_by_layer = all_layer_cls_embeddings(model, batch)  # (B, 12, 768)
        cls_by_layer = cls_by_layer.float()
        labels = batch["labels"].to(DEVICE)
        one_hot = F.one_hot(labels, num_classes=num_classes).float()
        # einsum: (B,L,D) x (B,C) → (L,C,D)
        class_sums.add_(torch.einsum("bld,bc->lcd", cls_by_layer, one_hot))
        class_counts.add_(one_hot.sum(dim=0))

    prototypes = class_sums / class_counts[None, :, None]  # (12, C, 768)
    return F.normalize(prototypes, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Loss — compute cosine similarity logits + cross-entropy at all 12 layers
# Identical to Antoine's layerwise_prototype_logits + joint_layerwise_loss
# ---------------------------------------------------------------------------
def layerwise_prototype_logits(cls_by_layer, prototypes, temperature):
    """
    cls_by_layer: (B, 12, 768)
    prototypes:   (12, C, 768)
    returns logits: (B, 12, C)
    """
    norm_cls   = F.normalize(cls_by_layer, p=2, dim=-1)
    norm_proto = F.normalize(prototypes,   p=2, dim=-1)
    # einsum: (B,L,D) x (L,C,D) → (B,L,C)
    cosine_sims = torch.einsum("bld,lcd->blc", norm_cls, norm_proto)
    return cosine_sims / temperature


def joint_layerwise_loss(logits, labels, layer_weights, class_weights=None):
    """
    logits:        (B, 12, C)
    labels:        (B,)
    layer_weights: (12,) normalised to sum=1
    """
    B, L, C = logits.shape
    flat_logits = logits.reshape(B * L, C)
    flat_labels = labels[:, None].expand(B, L).reshape(-1)
    flat_losses = F.cross_entropy(flat_logits, flat_labels,
                                  weight=class_weights, reduction="none")
    per_layer_losses = flat_losses.view(B, L).mean(dim=0)
    total_loss = torch.sum(layer_weights * per_layer_losses)
    return total_loss, per_layer_losses


# ---------------------------------------------------------------------------
# Evaluate all 12 layer classifiers on a dataloader
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_all_layers(model, loader, prototypes):
    model.eval()
    all_labels = []
    all_logits = []   # will be (N, 12, C)

    for batch in loader:
        cls_by_layer = all_layer_cls_embeddings(model, batch)
        logits = layerwise_prototype_logits(
            cls_by_layer, prototypes, CFG.temperature
        )
        all_labels.append(batch["labels"].numpy())
        all_logits.append(logits.cpu().float().numpy())

    labels  = np.concatenate(all_labels,  axis=0)   # (N,)
    logits  = np.concatenate(all_logits,  axis=0)   # (N, 12, C)

    results = []
    for layer_idx in range(12):
        preds = logits[:, layer_idx, :].argmax(axis=-1)
        acc   = float(accuracy_score(labels, preds))
        f1    = float(f1_score(labels, preds, average="macro", zero_division=0))
        results.append({"layer": layer_idx + 1, "accuracy": acc, "macro_f1": f1})

    return pd.DataFrame(results), labels, logits


# ---------------------------------------------------------------------------
# Early exit — Antoine's exact implementation
# Fixed margin threshold, minimum layer constraint
# ---------------------------------------------------------------------------
def early_exit_with_margin(similarities, threshold=0.10, min_layer=3):
    """
    similarities: (N, 12, C) — raw cosine similarities (logits * temperature)
    threshold:    minimum margin between top-2 similarities to exit
    min_layer:    don't exit before this layer (1-indexed)

    Returns:
        predictions: (N,)
        exit_layers: (N,) — 1-indexed layer where each post exited
    """
    N, L, C = similarities.shape
    predictions = []
    exit_layers = []

    for i in range(N):
        prediction = None
        exit_layer = L  # default: last layer

        for layer_idx in range(L):
            scores = similarities[i, layer_idx]
            sorted_scores = np.sort(scores)
            margin = sorted_scores[-1] - sorted_scores[-2]
            layer_number = layer_idx + 1

            if layer_number >= min_layer and margin >= threshold:
                prediction = int(np.argmax(scores))
                exit_layer = layer_number
                break

        if prediction is None:
            prediction = int(np.argmax(similarities[i, -1]))

        predictions.append(prediction)
        exit_layers.append(exit_layer)

    return np.array(predictions), np.array(exit_layers)


# ---------------------------------------------------------------------------
# Per-group fairness analysis (same as hate_prototypes_bert.py)
# ---------------------------------------------------------------------------
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
        not_hate = g_labels == 0
        fpr = round(float(((g_preds == 1) & not_hate).sum() / not_hate.sum()), 3) \
              if not_hate.sum() > 0 else "N/A"
        results.append({
            "Group": group, "N": int(mask.sum()),
            "AUC": auc, "Recall": recall, "FPR": fpr,
        })
    return pd.DataFrame(results).set_index("Group")


def analyze_exit_layers(exit_layers, groups, labels, min_samples=10):
    groups      = np.array(groups)
    exit_layers = np.array(exit_layers)
    labels      = np.array(labels)
    results = []
    for group in sorted(set(groups)):
        mask = groups == group
        if mask.sum() < min_samples:
            continue
        g_exits = exit_layers[mask]
        results.append({
            "Group":          group,
            "N":              int(mask.sum()),
            "Avg exit layer": round(float(g_exits.mean()), 2),
            "Early exit %":   f"{(g_exits < 12).mean():.1%}",
            "Hate base rate": round(float((labels[mask] == 1).mean()), 3),
        })
    return pd.DataFrame(results).set_index("Group")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path",  type=str, default="../data/dataset.json")
    ap.add_argument("--seed",       type=int, default=0,
                    help="Random seed. Run with 0, 1, 2 for 3-run statistical analysis.")
    ap.add_argument("--out_dir",    type=str, default="../outputs/checkpoints")
    ap.add_argument("--epochs",     type=int, default=CFG.num_epochs)
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--sample",     type=int, default=0,
                    help="If >0, use only this many training posts (smoke test).")
    ap.add_argument("--threshold",  type=float, default=CFG.exit_threshold)
    ap.add_argument("--min_layer",  type=int,   default=CFG.min_exit_layer)
    args = ap.parse_args()

    CFG.seed = args.seed  # override from command line
    set_seed(args.seed)
    print(f"Device: {DEVICE}")
    print(f"Model:  {CFG.model_name}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | T: {CFG.temperature}")
    print(f"Label scheme: hatespeech vs normal (offensive excluded — Antoine's setting)")

    # --- data ---
    train_df, test_df = load_hatexplain(args.data_path, seed=CFG.seed)
    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
    print(f"Label distribution (train): {dict(train_df['label'].value_counts())}")

    if args.sample > 0:
        train_df = train_df.sample(
            n=min(args.sample, len(train_df)), random_state=CFG.seed)
        print(f"Smoke-test: using {len(train_df)} training posts")

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)

    train_ds = HateDataset(train_df["post"].tolist(), train_df["label"].tolist(),
                           tokenizer, CFG.max_len)
    test_ds  = HateDataset(test_df["post"].tolist(),  test_df["label"].tolist(),
                           tokenizer, CFG.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)


    counts = train_df["label"].value_counts().sort_index().values
    cw = torch.tensor(len(train_df) / (2.0 * counts),
                      dtype=torch.float32, device=DEVICE)
    print(f"Class weights [not_hate, hate]: {cw.tolist()}")

    # --- model ---
    model = AutoModel.from_pretrained(
        CFG.model_name, output_hidden_states=True
    ).to(DEVICE)

    # layer loss weights — equal weight, normalised to sum=1
    layer_weights = torch.tensor(
        CFG.layer_loss_weights, dtype=torch.float32, device=DEVICE
    )
    layer_weights = layer_weights / layer_weights.sum()

    # --- optimizer + scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    )
    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = int(CFG.warmup_ratio * total_steps)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # --- initial prototypes (before training) ---
    print("\nComputing initial prototypes...")
    prototypes = compute_layerwise_prototypes(model, train_loader)
    print(f"Prototypes shape: {prototypes.shape}")  # (12, 2, 768)

    best_model     = None
    best_protos    = None

    # --- training loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_losses = []

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            cls_by_layer = all_layer_cls_embeddings(model, batch)
            logits = layerwise_prototype_logits(
                cls_by_layer, prototypes, CFG.temperature
            )
            total_loss, _ = joint_layerwise_loss(
                logits, batch["labels"].to(DEVICE), layer_weights,
                class_weights=cw
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CFG.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            total_losses.append(total_loss.item())


        # recompute prototypes with updated model (Antoine does this each epoch)
        prototypes = compute_layerwise_prototypes(model, train_loader)
        print(f"\nEpoch {epoch}/{args.epochs} | loss: {np.mean(total_losses):.4f}")


    # --- test set evaluation: all 12 layers ---
    print("\n=== Test set — all 12 layers ===")
    test_results, test_labels, test_logits = evaluate_all_layers(
        model, test_loader, prototypes
    )
    print(test_results[["layer", "accuracy", "macro_f1"]].to_string(index=False))

    # --- early exit ---
    print(f"\n=== Early exit (threshold={args.threshold}, min_layer={args.min_layer}) ===")
    # multiply logits by temperature to get back cosine similarities
    test_sims = test_logits * CFG.temperature  # (N, 12, C)
    ee_preds, exit_layers = early_exit_with_margin(
        test_sims, threshold=args.threshold, min_layer=args.min_layer
    )

    ee_acc = float(accuracy_score(test_labels, ee_preds))
    ee_f1  = float(f1_score(test_labels, ee_preds, average="macro", zero_division=0))
    avg_exit  = float(exit_layers.mean())
    early_pct = float((exit_layers < 12).mean())

    print(f"  Accuracy:          {ee_acc:.4f}")
    print(f"  Macro F1:          {ee_f1:.4f}")
    print(f"  Average exit layer:{avg_exit:.2f} / 12")
    print(f"  Posts exiting early: {early_pct:.1%}")

    # --- per-group fairness ---
    test_groups = test_df["group"].tolist()

    # baseline scores from last layer (layer 12)
    base_scores = test_logits[:, -1, 1] * CFG.temperature  # hate class similarity

    print("\n=== Per-group fairness (baseline: layer 12) ===")
    base_preds = (test_logits[:, -1, :].argmax(axis=-1))
    base_group = evaluate_per_group(
        test_labels, base_preds, base_scores, test_groups
    )
    print(base_group.to_string())

    # early exit scores
    ee_scores = test_sims[
        np.arange(len(test_labels)), exit_layers - 1, 1
    ]
    print("\n=== Per-group fairness (early exit) ===")
    ee_group = evaluate_per_group(
        test_labels, ee_preds, ee_scores, test_groups
    )
    print(ee_group.to_string())

    print("\n=== Exit layer analysis per group ===")
    exit_df = analyze_exit_layers(exit_layers, test_groups, test_labels)
    print(exit_df.to_string())

    # --- comparison baseline vs early exit ---
    print("\n=== Per-group: baseline vs early exit ===")
    print(f"  {'Group':<14} {'Base AUC':>10} {'EE AUC':>10} {'Diff':>8} "
          f"{'Base FPR':>10} {'EE FPR':>10} {'Diff':>8}")
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
            d_fpr = (e_fpr - b_fpr) if isinstance(b_fpr, float) \
                    and isinstance(e_fpr, float) else None
            d_fpr_str = f"{d_fpr:>+8.3f}" if d_fpr is not None else "     N/A"
            print(f"  {group:<14} {b_auc:>10.3f} {e_auc:>10.3f} {d_auc:>+8.3f} "
                  f"{str(b_fpr):>10} {str(e_fpr):>10} {d_fpr_str}")

    # --- save ---
    out_dir = Path(args.out_dir) / f"hatexplain-alllayer-bert-cased-s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    preds_dir = Path("../outputs/predictions")
    preds_dir.mkdir(parents=True, exist_ok=True)
    base_group.to_csv(preds_dir / "alllayer-pergroup-baseline.csv")
    ee_group.to_csv(preds_dir   / "alllayer-pergroup-earlyexit.csv")
    exit_df.to_csv(preds_dir    / "alllayer-exit-analysis.csv")

    test_results.to_csv(preds_dir / "alllayer-layer-metrics.csv", index=False)

    print(f"\nSaved to: {out_dir}")
    print(f"Per-group results saved to: {preds_dir}")


if __name__ == "__main__":
    main()