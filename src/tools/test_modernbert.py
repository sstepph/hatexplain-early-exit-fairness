"""
Environment check for ModernBERT on Newton.
Run inside the python3.11 venv. Fails fast and tells you which step broke.
"""
import sys

print("=" * 60)
print("1. ENVIRONMENT")
print("=" * 60)
print(f"Python       : {sys.version.split()[0]}")

try:
    import torch, transformers
    print(f"torch        : {torch.__version__}")
    print(f"transformers : {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB")
    else:
        print("  !! No GPU visible from this venv — training would be CPU-only")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("2. LOAD MODEL")
print("=" * 60)
from transformers import AutoModel, AutoTokenizer
NAME = "answerdotai/ModernBERT-base"
try:
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModel.from_pretrained(NAME, output_hidden_states=True)
    print(f"layers       : {model.config.num_hidden_layers}")
    print(f"hidden size  : {model.config.hidden_size}")
    print(f"params       : {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    print("\nIf this is a network error, download the model elsewhere and")
    print("copy the folder to Newton, then load from the local path.")
    sys.exit(1)

print("\n" + "=" * 60)
print("3. FORWARD PASS + HIDDEN STATES")
print("=" * 60)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

texts = ["i hate all people from that group", "the community centre opens monday"]
enc = tok(texts, return_tensors="pt", padding=True,
          truncation=True, max_length=128).to(device)

try:
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    L = model.config.num_hidden_layers
    print(f"hidden_states length : {len(hs)}  (expect {L+1} = embedding + {L} layers)")
    print(f"shape per layer      : {tuple(hs[1].shape)}")
    cls = torch.stack([h[:, 0, :] for h in hs[1:]], dim=1)
    print(f"stacked CLS shape    : {tuple(cls.shape)}  (batch, layers, hidden)")
    assert cls.shape[1] == L, "layer count mismatch"
    print("OK — your extract_cls_embeddings logic will work with L from config")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("4. MEMORY UNDER TRAINING-LIKE LOAD")
print("=" * 60)
if torch.cuda.is_available():
    try:
        model.train()
        torch.cuda.reset_peak_memory_stats()
        big = tok(["i hate all people from that group"] * 16,
                  return_tensors="pt", padding="max_length",
                  truncation=True, max_length=128).to(device)
        out = model(**big, output_hidden_states=True, return_dict=True)
        loss = torch.stack([h[:, 0, :] for h in out.hidden_states[1:]]).pow(2).mean()
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"peak memory, batch 16 : {peak:.2f} GiB of {total:.1f} GiB")
        print("VERDICT: " + ("fits" if peak < total * 0.8
              else "TIGHT — reduce batch size or use gradient accumulation"))
    except torch.cuda.OutOfMemoryError:
        print("OUT OF MEMORY at batch 16. Try batch 8 or 4.")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
else:
    print("skipped (no GPU)")

print("\nAll checks passed." if torch.cuda.is_available() else "\nChecks passed (CPU only).")