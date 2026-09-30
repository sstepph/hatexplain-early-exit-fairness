"""
Experiment 1 — frozen encoder, one classification head per transformer layer.

Changes requested by Irina:
  * shallowest point is transformer layer 1, not the embedding output
    (hidden_states[0] is identical for every post and carries no signal)
  * official HateXplain train/val/test splits, so the split is fixed
  * the seed affects head initialisation and training only, never the data

Output: outputs/probe/{tag}-scores-s{seed}.npz
        scores/preds have L+1 columns; column 0 is the embedding layer and
        is kept only for reference. Analysis uses layers 1..L.

Usage:
    CUDA_VISIBLE_DEVICES=1 python train_layer_heads.py --seed 0 --tag bert-cls
"""
import json, argparse, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True


def get_primary_group(targets):
    real = [t for t in targets if t != "None"]
    return Counter(real).most_common(1)[0][0] if real else "none"


def load_hatexplain(data_path, divisions_path, merge_val_into_train=True):
    """Official HateXplain splits. Fixed — independent of the seed."""
    data = json.load(open(data_path))
    div = json.load(open(divisions_path))

    rows = {}
    for pid, p in data.items():
        ann = p["annotators"]
        maj = Counter([a["label"] for a in ann]).most_common(1)[0][0]
        tg = [t for a in ann for t in a["target"]]
        rows[pid] = {"post": " ".join(p["post_tokens"]),
                     "label": 1 if maj in ("hatespeech", "offensive") else 0,
                     "group": get_primary_group(tg)}

    def frame(ids):
        return pd.DataFrame([rows[i] for i in ids if i in rows])

    train = frame(div["train"])
    val   = frame(div["val"])
    test  = frame(div["test"])
    if merge_val_into_train:
        train = pd.concat([train, val], ignore_index=True)
        val = val.iloc[0:0]
    return train.reset_index(drop=True), val, test.reset_index(drop=True)


@torch.no_grad()
def extract_all_layers(texts, tok, model, max_len, batch_size, pool):
    """Return (N, L+1, H). Column 0 is the embedding layer."""
    model.eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt").to(DEV)
        out = model(**enc, output_hidden_states=True, return_dict=True)
        if pool == "cls":
            rep = torch.stack([h[:, 0, :] for h in out.hidden_states], dim=1)
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            rep = torch.stack([(h * m).sum(1) / m.sum(1).clamp(min=1)
                               for h in out.hidden_states], dim=1)
        chunks.append(rep.cpu())
        if (i // batch_size) % 20 == 0:
            print(f"    {min(i + batch_size, len(texts))}/{len(texts)}")
    return torch.cat(chunks, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="../data/dataset.json")
    ap.add_argument("--divisions", default="../data/post_id_divisions.json")
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--tag", default="bert-cls")
    ap.add_argument("--pool", choices=["cls", "mean"], default="cls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--head_batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--keep_val", action="store_true",
                    help="keep the official val split out of training")
    ap.add_argument("--out_dir", default="../outputs/probe")
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"Device {DEV} | {args.model} | pool {args.pool} | seed {args.seed}")

    train_df, val_df, test_df = load_hatexplain(
        args.data_path, args.divisions, merge_val_into_train=not args.keep_val)
    print(f"Official splits — train {len(train_df)} | val {len(val_df)} "
          f"| test {len(test_df)}  (split is fixed, seed affects training only)")
    print(f"Train labels {dict(train_df['label'].value_counts())}")

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = AutoModel.from_pretrained(args.model, output_hidden_states=True).to(DEV)
    for p in enc.parameters():
        p.requires_grad = False
    L, H = enc.config.num_hidden_layers, enc.config.hidden_size
    print(f"Layers {L} | hidden {H} | encoder frozen")

    print("\nExtracting train...")
    Xtr = extract_all_layers(train_df["post"].tolist(), tok, enc,
                             args.max_len, args.batch_size, args.pool)
    print("Extracting test...")
    Xte = extract_all_layers(test_df["post"].tolist(), tok, enc,
                             args.max_len, args.batch_size, args.pool)

    ytr = torch.tensor(train_df["label"].values, dtype=torch.long)
    yte = torch.tensor(test_df["label"].values, dtype=torch.long)

    counts = train_df["label"].value_counts().sort_index().values
    cw = torch.tensor(len(train_df) / (2.0 * counts),
                      dtype=torch.float32, device=DEV)
    print(f"Class weights {cw.tolist()}")

    heads = nn.ModuleList([nn.Linear(H, 2) for _ in range(L + 1)]).to(DEV)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(weight=cw)

    N = len(ytr)
    print(f"\nTraining {L+1} heads, {args.epochs} epochs...")
    for ep in range(1, args.epochs + 1):
        heads.train()
        perm = torch.randperm(N)
        for i in range(0, N, args.head_batch):
            idx = perm[i:i + args.head_batch]
            xb, yb = Xtr[idx].to(DEV), ytr[idx].to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = sum(lossf(heads[l](xb[:, l, :]), yb) for l in range(L + 1))
            loss.backward(); opt.step()

    heads.eval()
    scores = np.zeros((len(yte), L + 1), dtype=np.float32)
    preds  = np.zeros((len(yte), L + 1), dtype=np.int64)
    with torch.no_grad():
        for l in range(L + 1):
            o = heads[l](Xte[:, l, :].to(DEV))
            scores[:, l] = torch.softmax(o, -1)[:, 1].cpu().numpy()
            preds[:, l]  = o.argmax(-1).cpu().numpy()

    y = yte.numpy()
    print(f"\n{'layer':>6} {'acc':>9} {'macroF1':>9} {'AUC':>9}")
    for l in range(L + 1):
        note = "  (embeddings, not used)" if l == 0 else ""
        print(f"{l:>6} {accuracy_score(y, preds[:,l])*100:>8.2f}% "
              f"{f1_score(y, preds[:,l], average='macro')*100:>8.2f}% "
              f"{roc_auc_score(y, scores[:,l])*100:>8.2f}%{note}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fn = out / f"{args.tag}-scores-s{args.seed}.npz"
    np.savez_compressed(fn, scores=scores, preds=preds, labels=y,
                        groups=test_df["group"].values.astype(str),
                        n_layers=L, pool=args.pool)
    print(f"\nSaved {fn}")


if __name__ == "__main__":
    main()