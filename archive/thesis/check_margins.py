"""
Diagnostic: what do the actual cosine margins look like across all 12 layers?
Processes in small batches to avoid GPU OOM.
"""
import json
import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import pandas as pd

CHECKPOINT = "../outputs/checkpoints/hatexplain-bert-base-cased-s0"
DATA_PATH  = "../data/dataset.json"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH      = 16   # small batch to avoid OOM

def get_primary_group(targets_list):
    real = [t for t in targets_list if t != "None"]
    if not real: return "none"
    return Counter(real).most_common(1)[0][0]

def load_hatexplain(data_path, seed=0):
    with open(data_path) as f: data = json.load(f)
    rows = []
    for post_id, post_data in data.items():
        annotators = post_data["annotators"]
        labels = [a["label"] for a in annotators]
        majority_label = Counter(labels).most_common(1)[0][0]
        targets = []
        for a in annotators: targets.extend(a["target"])
        rows.append({"post_id": post_id, "label_raw": majority_label,
                     "targets": targets, "post": " ".join(post_data["post_tokens"])})
    df = pd.DataFrame(rows)
    df["label"] = df["label_raw"].apply(
        lambda x: 1 if x in ["hatespeech","offensive"] else 0)
    df["group"] = df["targets"].apply(get_primary_group)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def l2_normalize(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0, 1, norms)

def extract_cls_batched(texts, tokenizer, model, device, batch=16):
    """Extract CLS at all 12 layers in small batches."""
    all_layers = [[] for _ in range(12)]
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        inputs = tokenizer(batch_texts, truncation=True, padding=True,
                           max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        for layer_idx in range(12):
            cls = outputs.hidden_states[layer_idx+1][:,0,:].cpu().numpy()
            all_layers[layer_idx].append(cls)
    return [np.concatenate(l, axis=0) for l in all_layers]

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
model = AutoModel.from_pretrained(
    CHECKPOINT, output_hidden_states=True,
    ignore_mismatched_sizes=True).to(DEVICE)
model.eval()
print("Done.")

print("Loading data...")
train_df, test_df = load_hatexplain(DATA_PATH)

# use 300 test posts and 500 training posts -- enough for the diagnostic
test_sample  = test_df.sample(300, random_state=42)
train_sample = train_df.sample(500, random_state=42)
print(f"Test sample: {len(test_sample)} | Train sample: {len(train_sample)}")

print("Extracting test embeddings (batched, batch=16)...")
te_layers = extract_cls_batched(
    test_sample["post"].tolist(), tokenizer, model, DEVICE, BATCH)

print("Extracting train embeddings (batched, batch=16)...")
tr_layers = extract_cls_batched(
    train_sample["post"].tolist(), tokenizer, model, DEVICE, BATCH)

train_labels = np.array(train_sample["label"].tolist())

print("\n=== Margin distribution per layer ===")
print(f"{'Layer':>6} | {'Min':>7} | {'Max':>7} | {'Mean|m|':>9} | {'P50|m|':>9} | {'P75|m|':>9} | {'P95|m|':>9}")
print("-" * 75)

for layer_idx in range(12):
    # build prototype from train sample
    tr_emb     = l2_normalize(tr_layers[layer_idx])
    hate_proto = tr_emb[train_labels==1].mean(axis=0)
    noha_proto = tr_emb[train_labels==0].mean(axis=0)
    hate_proto = hate_proto / np.linalg.norm(hate_proto)
    noha_proto = noha_proto / np.linalg.norm(noha_proto)

    # compute margins on test sample
    te_emb = l2_normalize(te_layers[layer_idx])
    margin = te_emb @ hate_proto - te_emb @ noha_proto
    abs_m  = np.abs(margin)

    print(f"{layer_idx+1:>6} | {margin.min():>7.4f} | {margin.max():>7.4f} | "
          f"{abs_m.mean():>9.4f} | {np.percentile(abs_m,50):>9.4f} | "
          f"{np.percentile(abs_m,75):>9.4f} | {np.percentile(abs_m,95):>9.4f}")

print("\n--- What this means ---")
print("P95 = 95% of posts have |margin| below this value.")
print("If P95 < 0.25 everywhere: margins are too small for a fixed threshold.")
print("If P95 grows layer by layer: representations sharpen with depth (expected).")
