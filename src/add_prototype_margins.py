"""
Add prototype margins to the Exp 1 .npz files, without retraining the heads.

Re-extracts frozen CLS representations, builds per-layer class prototypes
from the training set, and stores

    proto_margin (N_test, L+1) = cos(h, p_hate) - cos(h, p_nonhate)

in the same file. Heads, scores and predictions are left untouched, so
Exp 1 results stay exactly as they were.

Usage:
    CUDA_VISIBLE_DEVICES=1 python add_prototype_margins.py --seed 0
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from train_layer_heads import load_hatexplain, extract_all_layers, DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="../data/dataset.json")
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--tag", default="bert-cls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--dir", default="../outputs/probe")
    args = ap.parse_args()

    path = f"{args.dir}/{args.tag}-scores-s{args.seed}.npz"
    old = dict(np.load(path, allow_pickle=True))
    print(f"Loaded {path}")

    train_df, test_df = load_hatexplain(args.data_path, args.seed)
    assert np.array_equal(test_df["label"].values, old["labels"]), \
        "test split does not match the saved file — seed or data differ"

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = AutoModel.from_pretrained(args.model, output_hidden_states=True).to(DEV)
    for p in enc.parameters():
        p.requires_grad = False

    print("Extracting train...")
    Xtr = extract_all_layers(train_df["post"].tolist(), tok, enc,
                             args.max_len, args.batch_size, "cls")
    print("Extracting test...")
    Xte = extract_all_layers(test_df["post"].tolist(), tok, enc,
                             args.max_len, args.batch_size, "cls")

    ytr = torch.tensor(train_df["label"].values)
    Xtr_n = F.normalize(Xtr, dim=-1)
    Xte_n = F.normalize(Xte, dim=-1)

    p_hate = F.normalize(Xtr_n[ytr == 1].mean(0), dim=-1)   # (L+1, H)
    p_non  = F.normalize(Xtr_n[ytr == 0].mean(0), dim=-1)

    margin = ((Xte_n * p_hate).sum(-1) - (Xte_n * p_non).sum(-1)).numpy()

    old["proto_margin"] = margin.astype(np.float32)
    np.savez_compressed(path, **old)

    L = margin.shape[1] - 1
    print(f"\n{'layer':>6} {'P50 |m|':>10} {'P90 |m|':>10} {'max |m|':>10}")
    for l in range(L + 1):
        a = np.abs(margin[:, l])
        print(f"{l:>6} {np.percentile(a,50):>10.5f} {np.percentile(a,90):>10.5f} {a.max():>10.5f}")
    print(f"\nSaved proto_margin into {path}")


if __name__ == "__main__":
    main()
