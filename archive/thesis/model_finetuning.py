import os
import json
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EvalPrediction,
)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
            "post_id": post_id,
            "label_raw": majority_label,
            "targets": targets,
            "post": " ".join(post_data["post_tokens"]),
        })
    df = pd.DataFrame(rows)
    df["label"] = df["label_raw"].apply(lambda x: 1 if x in ["hatespeech", "offensive"] else 0)
    df["group"] = df["targets"].apply(get_primary_group)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

class BinaryHateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {
            "input_ids":      torch.tensor(self.enc["input_ids"][i]),
            "attention_mask": torch.tensor(self.enc["attention_mask"][i]),
            "labels":         self.labels[i],
        }

def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    logits = np.asarray(logits)
    labels = np.asarray(p.label_ids)
    preds  = logits.argmax(axis=-1)
    acc    = float(accuracy_score(labels, preds))
    f1_bin = float(f1_score(labels, preds, average="binary",  zero_division=0))
    f1_mac = float(f1_score(labels, preds, average="macro",   zero_division=0))
    if logits.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(labels, logits[:, 1])
        pr_auc = float(auc(recall, precision))
    else:
        pr_auc = float("nan")
    return {"accuracy": acc, "f1_binary": f1_bin, "f1_macro": f1_mac, "pr_auc": pr_auc}

class WeightedTrainer(Trainer):
    def __init__(self, loss_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weight = loss_weight
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_labels = model.module.config.num_labels if hasattr(model, "module") else model.config.num_labels
        loss = torch.nn.CrossEntropyLoss(weight=self.loss_weight.to(logits.device))(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True, collate_fn=default_data_collator)

def sklearn_class_weights(y_np, device="cpu"):
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_np)
    return torch.tensor(weights, dtype=torch.float, device=device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path",  type=str,   default="../data/dataset.json")
    ap.add_argument("--model_name", type=str,   default="bert-base-cased")
    ap.add_argument("--lr",         type=float, default=1e-5)
    ap.add_argument("--epochs",     type=int,   default=3)
    ap.add_argument("--batch_size", type=int,   default=16)
    ap.add_argument("--max_len",    type=int,   default=128)
    ap.add_argument("--seed",       type=int,   default=0)
    ap.add_argument("--sample",     type=int,   default=0)
    ap.add_argument("--out_dir",    type=str,   default="../outputs/checkpoints")
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"Device : {DEVICE}")

    train_df, test_df = load_hatexplain(args.data_path, seed=args.seed)
    print(f"Train  : {len(train_df)} | Test: {len(test_df)}")
    print(f"Label distribution (train): {dict(train_df['label'].value_counts())}")

    if args.sample > 0:
        train_df = train_df.sample(n=min(args.sample, len(train_df)), random_state=args.seed)
        print(f"Smoke-test: using {len(train_df)} posts")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    weights   = sklearn_class_weights(np.array(train_df["label"].tolist()), device=DEVICE)
    print(f"Class weights [not_hate, hate]: {weights.tolist()}")

    train_ds = BinaryHateDataset(train_df["post"].tolist(), train_df["label"].tolist(), tokenizer, args.max_len)
    test_ds  = BinaryHateDataset(test_df["post"].tolist(),  test_df["label"].tolist(),  tokenizer, args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2,
        id2label={0: "not_hate", 1: "hate"}, label2id={"not_hate": 0, "hate": 1},
    ).to(DEVICE)

    save_name = f"hatexplain-{args.model_name.split('/')[-1]}-s{args.seed}"
    out_dir   = Path(args.out_dir) / save_name

    training_args = TrainingArguments(
        output_dir=str(out_dir / "tmp"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        logging_strategy="steps",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )

    trainer = WeightedTrainer(
        loss_weight=weights, model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics, data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating...")
    results = trainer.evaluate()
    print("Results:", results)

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nModel saved to: {out_dir}")

    metrics_dir = Path("../outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"{save_name}-metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in results.items() if isinstance(v, (int, float))}, f, indent=2)
    print(f"Metrics saved to: ../outputs/metrics/{save_name}-metrics.json")

if __name__ == "__main__":
    main()
