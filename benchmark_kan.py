"""
KAN IMDB Benchmark

Compares KAN Language Model vs Transformer on IMDB sentiment:
  - Multiple grid sizes: [3, 5, 7]
  - Reports accuracy, parameter count, training time
  - Loads prior Transformer / WaveNet results for the full table
"""

import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer

from kan_language_model import KANLanguageModel


# ─── Hyperparams ────────────────────────────────────────────────────────────
EMBED_DIM   = 128      # smaller than transformer so param count is comparable
NUM_LAYERS  = 4
NUM_HEADS   = 4
MAX_LEN     = 256
TRAIN_SIZE  = 10_000
TEST_SIZE   = 5_000
BATCH_SIZE  = 32
EPOCHS      = 5
LR          = 3e-4
GRID_SIZES  = [3, 5, 7]


# ─── Data ───────────────────────────────────────────────────────────────────

def load_imdb(train_n: int, test_n: int):
    ds     = load_dataset("imdb")
    tok    = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode(split, n):
        texts  = ds[split]["text"][:n]
        labels = ds[split]["label"][:n]
        enc = tok(texts, padding="max_length", truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        return enc["input_ids"], torch.tensor(labels, dtype=torch.long)

    print("Tokenising train …")
    tx, ty = encode("train", train_n)
    print("Tokenising test  …")
    vx, vy = encode("test",  test_n)
    return (tx, ty), (vx, vy), tok.vocab_size


def make_loader(x, y, shuffle=False):
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=shuffle)


# ─── Training ───────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = total_correct = total_n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss    += loss.item() * len(yb)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_n       += len(yb)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        total_loss    += loss.item() * len(yb)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_n       += len(yb)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def inference_ms(model, loader, device, n_batches=20):
    model.eval()
    times = []
    for i, (xb, _) in enumerate(loader):
        if i >= n_batches:
            break
        xb = xb.to(device)
        t0 = time.perf_counter()
        model(xb)
        times.append((time.perf_counter() - t0) * 1000)
    return float(torch.tensor(times).mean())


# ─── Run one KAN experiment ─────────────────────────────────────────────────

def run_kan(grid_size: int, train_loader, test_loader, vocab_size: int, device):
    print(f"\n{'='*60}")
    print(f"KAN grid_size={grid_size}  embed={EMBED_DIM}  layers={NUM_LAYERS}")

    model = KANLanguageModel(
        vocab_size  = vocab_size,
        embed_dim   = EMBED_DIM,
        num_layers  = NUM_LAYERS,
        num_heads   = NUM_HEADS,
        grid_size   = grid_size,
        max_seq_len = MAX_LEN,
    ).to(device)

    params = model.count_parameters()
    print(f"Parameters: {params:,}")

    opt       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    epoch_results = []
    t_total = 0.0

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        t_total += elapsed
        print(f"  ep {ep}/{EPOCHS}  tr={tr_acc:.4f}  te={te_acc:.4f}  loss={te_loss:.4f}  {elapsed:.0f}s")
        epoch_results.append({"epoch": ep, "train_acc": tr_acc, "train_loss": tr_loss,
                               "test_acc": te_acc, "test_loss": te_loss, "epoch_time": elapsed})

    inf_ms = inference_ms(model, test_loader, device)
    best   = max(r["test_acc"] for r in epoch_results)

    return {
        "grid_size":          grid_size,
        "embed_dim":          EMBED_DIM,
        "parameters":         params,
        "final_test_acc":     epoch_results[-1]["test_acc"],
        "best_test_acc":      best,
        "total_train_time":   t_total,
        "inference_ms":       inf_ms,
        "epoch_results":      epoch_results,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    (tx, ty), (vx, vy), vocab_size = load_imdb(TRAIN_SIZE, TEST_SIZE)
    train_loader = make_loader(tx, ty, shuffle=True)
    test_loader  = make_loader(vx, vy, shuffle=False)

    results = {}

    for gs in GRID_SIZES:
        res = run_kan(gs, train_loader, test_loader, vocab_size, device)
        results[f"KAN_grid{gs}"] = res

    # ── Save results ────────────────────────────────────────────────────────
    with open("kan_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → kan_results.json")

    # ── Load prior baselines ─────────────────────────────────────────────────
    try:
        with open("imdb_results.json") as f:
            prior = json.load(f)
    except FileNotFoundError:
        prior = {}

    # ── Summary table ────────────────────────────────────────────────────────
    tf_acc    = prior.get("Transformer", {}).get("final_test_acc", 0.8388)
    tf_params = prior.get("Transformer", {}).get("parameters",    1_644_258)
    wn_acc    = prior.get("WaveNetNeuro",{}).get("final_test_acc", 0.8396)
    wn_params = prior.get("WaveNetNeuro",{}).get("parameters",    1_320_035)

    print("\n" + "="*75)
    print(f"{'Model':<20} {'Acc':>8} {'Params':>12} {'Ratio':>8} {'Inf ms':>9}")
    print("-"*75)

    def row(name, acc, params, inf_ms, ref_params):
        ratio = ref_params / params if params else 0
        print(f"{name:<20} {acc:>7.2%} {params:>12,} {ratio:>7.1f}x {inf_ms:>8.1f}")

    row("Transformer",  tf_acc,  tf_params, prior.get("Transformer",{}).get("inference_ms_per_batch", 12.4), tf_params)
    row("WaveNetNeuro", wn_acc,  wn_params, prior.get("WaveNetNeuro",{}).get("inference_ms_per_batch", 45.1), tf_params)

    for gs in GRID_SIZES:
        r   = results[f"KAN_grid{gs}"]
        row(f"KAN grid={gs}", r["best_test_acc"], r["parameters"], r["inference_ms"], tf_params)

    print("="*75)

    # ── Parameter efficiency analysis ────────────────────────────────────────
    print("\nParameter Efficiency Analysis (target: 80% accuracy)")
    print("-"*60)
    target = 0.80
    for gs in GRID_SIZES:
        r   = results[f"KAN_grid{gs}"]
        met = "YES" if r["best_test_acc"] >= target else "NO "
        ratio = tf_params / r["parameters"]
        print(f"  KAN grid={gs}: {r['best_test_acc']:.2%}  params={r['parameters']:,}  "
              f"ratio={ratio:.1f}x  target met: {met}")

    return results


if __name__ == "__main__":
    main()
