#!/usr/bin/env python3
"""
Benchmark: Causal Gated Network vs MLP vs Transformer

Part 1 — Synthetic causal dataset
  X₁ CAUSES Y, X₂ is spuriously correlated.
  Test: which model ignores X₂ when correlation shifts?

Part 2 — IMDB sentiment (10K train, 5K test)
  Compare against Transformer (80.5%) and PredCoding (82.5%).

Part 3 — Sample efficiency
  Train on 10%, 50%, 100% of data; measure accuracy curves.
"""

import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from causal_gated_network import CausalGatedMLP, CausalGatedNetwork


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Synthetic causal dataset
# ═══════════════════════════════════════════════════════════════════════════

def make_causal_dataset(n: int, spurious_corr: float = 0.8, noise_dims: int = 8,
                        seed: int = 42):
    """
    Create dataset where X₁ CAUSES Y and X₂ is spuriously correlated.

    Train: X₂ agrees with Y ~spurious_corr of the time (shortcut available)
    Test:  X₂ is RANDOM (shortcut fails)

    Extra noise dims are pure random to test robustness.
    """
    rng = np.random.RandomState(seed)

    # True causal feature
    x1 = rng.randint(0, 2, size=n).astype(np.float32)

    # Y is caused by X1 (with 5% noise)
    y = x1.copy()
    flip = rng.random(n) < 0.05
    y[flip] = 1 - y[flip]

    # Spurious feature: correlated with Y in training
    x2 = y.copy()
    flip_sp = rng.random(n) < (1 - spurious_corr)
    x2[flip_sp] = 1 - x2[flip_sp]

    # Noise features
    noise = rng.randn(n, noise_dims).astype(np.float32) * 0.3

    X = np.column_stack([x1, x2, noise])
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)


def make_causal_test_set(n: int, noise_dims: int = 8, seed: int = 99):
    """Test set where X₂ is RANDOM (no spurious correlation)."""
    rng = np.random.RandomState(seed)
    x1 = rng.randint(0, 2, size=n).astype(np.float32)
    y  = x1.copy()
    flip = rng.random(n) < 0.05
    y[flip] = 1 - y[flip]
    x2 = rng.randint(0, 2, size=n).astype(np.float32)  # RANDOM
    noise = rng.randn(n, noise_dims).astype(np.float32) * 0.3
    X = np.column_stack([x1, x2, noise])
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)


# --- Simple MLP baseline ---

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, num_layers=3, num_classes=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x), {}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Small Transformer for tabular data ---

class TabularTransformer(nn.Module):
    """Treat each feature as a 'token' and use attention."""
    def __init__(self, in_dim, embed_dim=32, num_heads=2, num_layers=2, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(1, embed_dim)
        self.pos  = nn.Parameter(torch.randn(1, in_dim, embed_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(embed_dim, num_heads,
                                                dim_feedforward=embed_dim*4,
                                                dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [batch, in_dim] → treat each feature as a token
        h = self.proj(x.unsqueeze(-1)) + self.pos  # [batch, in_dim, embed]
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.classifier(h), {}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_tabular(model, train_x, train_y, test_x, test_y, epochs=50, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True)

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits_te, _ = model(test_x)
            acc = (logits_te.argmax(1) == test_y).float().mean().item()
            logits_tr, _ = model(train_x)
            tr_acc = (logits_tr.argmax(1) == train_y).float().mean().item()
        history.append({"epoch": ep, "train_acc": tr_acc, "test_acc": acc})

    return history


def run_causal_experiment():
    print("=" * 70)
    print("PART 1: Synthetic Causal Dataset")
    print("  X₁ CAUSES Y, X₂ is spuriously correlated (80% in train)")
    print("  Test set: X₂ is RANDOM (shortcut fails)")
    print("=" * 70)

    noise_dims = 8
    in_dim = 2 + noise_dims

    train_x, train_y = make_causal_dataset(5000, spurious_corr=0.8,
                                            noise_dims=noise_dims)
    test_x,  test_y  = make_causal_test_set(2000, noise_dims=noise_dims)

    print(f"Train: {len(train_x)}, Test: {len(test_x)}, Features: {in_dim}")
    print(f"Train label balance: {train_y.float().mean():.2f}")
    print(f"Test  label balance: {test_y.float().mean():.2f}")

    results = {}

    # --- MLP ---
    print("\n--- MLP ---")
    mlp = SimpleMLP(in_dim, hidden=64, num_layers=3)
    print(f"  Params: {mlp.count_parameters():,}")
    hist = train_tabular(mlp, train_x, train_y, test_x, test_y, epochs=80)
    best = max(h["test_acc"] for h in hist)
    final = hist[-1]["test_acc"]
    print(f"  Final test acc: {final:.4f}  Best: {best:.4f}")
    results["MLP"] = {"params": mlp.count_parameters(), "best_acc": best,
                       "final_acc": final, "history": hist}

    # --- Tabular Transformer ---
    print("\n--- Transformer ---")
    tf = TabularTransformer(in_dim, embed_dim=32, num_heads=2, num_layers=2)
    print(f"  Params: {tf.count_parameters():,}")
    hist = train_tabular(tf, train_x, train_y, test_x, test_y, epochs=80)
    best = max(h["test_acc"] for h in hist)
    final = hist[-1]["test_acc"]
    print(f"  Final test acc: {final:.4f}  Best: {best:.4f}")
    results["Transformer"] = {"params": tf.count_parameters(), "best_acc": best,
                               "final_acc": final, "history": hist}

    # --- Causal Gated MLP ---
    print("\n--- Causal Gated Network ---")
    cgn = CausalGatedMLP(in_dim, hidden_dim=64, num_layers=3, bottleneck_dim=8)
    print(f"  Params: {cgn.count_parameters():,}")
    hist = train_tabular(cgn, train_x, train_y, test_x, test_y, epochs=80)
    best = max(h["test_acc"] for h in hist)
    final = hist[-1]["test_acc"]
    print(f"  Final test acc: {final:.4f}  Best: {best:.4f}")
    results["CausalGated"] = {"params": cgn.count_parameters(), "best_acc": best,
                               "final_acc": final, "history": hist}

    # --- Gate interpretability ---
    print("\n--- Gate Interpretability ---")
    cgn.eval()
    with torch.no_grad():
        h_test = torch.nn.functional.gelu(cgn.input_gate(test_x))
        gate_vals = torch.sigmoid(cgn.input_gate.gate(test_x))  # [N, hidden]

    # Check which input features the gate attends to
    gate_weight = cgn.input_gate.gate.weight  # [hidden, in_dim]
    # Average absolute gate weight per input feature
    feat_importance = gate_weight.abs().mean(dim=0)
    feat_names = ["X₁ (causal)", "X₂ (spurious)"] + [f"noise_{i}" for i in range(noise_dims)]
    print("  Feature importance (avg |gate weight|):")
    for name, imp in zip(feat_names, feat_importance):
        bar = "█" * int(imp.item() * 20)
        print(f"    {name:16s}: {imp.item():.4f} {bar}")

    # Mean gate activation per feature
    linear_weight = cgn.input_gate.linear.weight  # [hidden, in_dim]
    feat_contrib = (linear_weight.abs() * gate_weight.abs()).mean(dim=0)
    print("\n  Feature contribution (|W| * |G|):")
    for name, c in zip(feat_names, feat_contrib):
        bar = "█" * int(c.item() * 40)
        print(f"    {name:16s}: {c.item():.4f} {bar}")

    results["gate_analysis"] = {
        "feature_importance": {n: v.item() for n, v in zip(feat_names, feat_importance)},
        "feature_contribution": {n: v.item() for n, v in zip(feat_names, feat_contrib)},
    }

    # --- Sample efficiency ---
    print("\n--- Sample Efficiency ---")
    fractions = [0.1, 0.25, 0.5, 1.0]
    efficiency = {}
    for frac in fractions:
        n = int(5000 * frac)
        tx, ty = train_x[:n], train_y[:n]

        models_eff = {
            "MLP": SimpleMLP(in_dim, hidden=64, num_layers=3),
            "Transformer": TabularTransformer(in_dim, embed_dim=32, num_heads=2, num_layers=2),
            "CausalGated": CausalGatedMLP(in_dim, hidden_dim=64, num_layers=3, bottleneck_dim=8),
        }

        row = {}
        for name, m in models_eff.items():
            hist = train_tabular(m, tx, ty, test_x, test_y, epochs=80)
            best = max(h["test_acc"] for h in hist)
            row[name] = best

        efficiency[str(frac)] = row
        print(f"  {frac:.0%} data ({n:,} samples): "
              + "  ".join(f"{k}={v:.2%}" for k, v in row.items()))

    results["sample_efficiency"] = efficiency

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'Best Acc':>10} {'Params':>10}")
    print("-" * 70)
    for name in ["MLP", "Transformer", "CausalGated"]:
        r = results[name]
        print(f"{name:<20} {r['best_acc']:>9.2%} {r['params']:>10,}")
    print("=" * 70)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: IMDB Sentiment
# ═══════════════════════════════════════════════════════════════════════════

def load_imdb_data(train_n=10000, test_n=5000, max_len=256):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds  = load_dataset("imdb")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode(split, n):
        # IMPORTANT: IMDB is sorted by label — must shuffle first
        split_ds = ds[split].shuffle(seed=42)
        texts  = split_ds["text"][:n]
        labels = split_ds["label"][:n]
        enc = tok(texts, padding="max_length", truncation=True,
                  max_length=max_len, return_tensors="pt")
        return enc["input_ids"], torch.tensor(labels, dtype=torch.long)

    print("Tokenising IMDB train …")
    tx, ty = encode("train", train_n)
    print(f"  Label balance: {ty.float().mean():.3f}")
    print("Tokenising IMDB test  …")
    vx, vy = encode("test",  test_n)
    print(f"  Label balance: {vy.float().mean():.3f}")
    return (tx, ty), (vx, vy), tok.vocab_size


def train_epoch(model, loader, opt, criterion):
    model.train()
    total_loss = total_correct = total_n = 0
    for xb, yb in loader:
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
def eval_model(model, loader, criterion):
    model.eval()
    total_loss = total_correct = total_n = 0
    for xb, yb in loader:
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        total_loss    += loss.item() * len(yb)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_n       += len(yb)
    return total_loss / total_n, total_correct / total_n


def run_imdb_experiment():
    print("\n" + "=" * 70)
    print("PART 2: IMDB Sentiment (10K train, 5K test)")
    print("=" * 70)

    (tx, ty), (vx, vy), vocab_size = load_imdb_data()
    train_loader = DataLoader(TensorDataset(tx, ty), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(vx, vy), batch_size=32)

    # --- Causal Gated Network (sequence model) ---
    configs = [
        ("CausalGated_d256", dict(embed_dim=256, num_layers=4, bottleneck_ratio=0.25)),
        ("CausalGated_d128", dict(embed_dim=128, num_layers=4, bottleneck_ratio=0.25)),
    ]

    results = {}
    criterion = nn.CrossEntropyLoss()

    for name, cfg in configs:
        print(f"\n--- {name} ---")
        model = CausalGatedNetwork(
            vocab_size=vocab_size, num_classes=2, max_seq_len=256,
            use_temporal=True, **cfg
        )
        params = model.count_parameters()
        print(f"  Parameters: {params:,}")

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)

        epoch_results = []
        t_total = 0

        for ep in range(1, 9):  # 8 epochs
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, opt, criterion)
            te_loss, te_acc = eval_model(model, test_loader, criterion)
            sched.step()
            elapsed = time.time() - t0
            t_total += elapsed
            print(f"  ep {ep}/8  tr={tr_acc:.4f}  te={te_acc:.4f}  "
                  f"loss={te_loss:.4f}  {elapsed:.0f}s")
            epoch_results.append({
                "epoch": ep, "train_acc": tr_acc, "train_loss": tr_loss,
                "test_acc": te_acc, "test_loss": te_loss, "time": elapsed,
            })

        best = max(r["test_acc"] for r in epoch_results)

        # Gate sparsity report
        model.eval()
        xb_sample = tx[:32]
        sparsity = model.gate_sparsity_report(xb_sample)
        print(f"  Gate sparsity: {sparsity}")

        # Inference timing
        model.eval()
        times = []
        with torch.no_grad():
            for i, (xb, _) in enumerate(test_loader):
                if i >= 20:
                    break
                t0 = time.perf_counter()
                model(xb)
                times.append((time.perf_counter() - t0) * 1000)
        inf_ms = float(np.mean(times))

        results[name] = {
            "parameters": params,
            "best_test_acc": best,
            "final_test_acc": epoch_results[-1]["test_acc"],
            "total_train_time": t_total,
            "inference_ms": inf_ms,
            "gate_sparsity": sparsity,
            "epoch_results": epoch_results,
        }
        print(f"  BEST: {best:.4f}  Inference: {inf_ms:.1f} ms/batch")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    all_results = {}

    # Part 1: Causal dataset
    causal_results = run_causal_experiment()
    all_results["causal"] = causal_results

    # Part 2: IMDB
    imdb_results = run_imdb_experiment()
    all_results["imdb"] = imdb_results

    # Save
    # Convert non-serializable items
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open("causal_gated_results.json", "w") as f:
        json.dump(sanitize(all_results), f, indent=2)
    print("\nSaved → causal_gated_results.json")

    # ── Final comparison table ───────────────────────────────────────────
    print("\n" + "=" * 75)
    print("FINAL COMPARISON — IMDB Sentiment")
    print("=" * 75)

    # Prior results
    try:
        with open("imdb_results.json") as f:
            prior = json.load(f)
    except FileNotFoundError:
        prior = {}

    tf_acc  = prior.get("Transformer", {}).get("final_test_acc", 0.8388)
    tf_par  = prior.get("Transformer", {}).get("parameters",    1_644_258)
    wn_acc  = prior.get("WaveNetNeuro",{}).get("final_test_acc", 0.8396)
    wn_par  = prior.get("WaveNetNeuro",{}).get("parameters",    1_320_035)

    print(f"{'Model':<25} {'Acc':>8} {'Params':>12}")
    print("-" * 55)
    print(f"{'Transformer':<25} {tf_acc:>7.2%} {tf_par:>12,}")
    print(f"{'WaveNetNeuro':<25} {wn_acc:>7.2%} {wn_par:>12,}")
    print(f"{'PredCoding (prior)':<25} {'82.50%':>8} {'~1.5M':>12}")

    for name, r in imdb_results.items():
        print(f"{name:<25} {r['best_test_acc']:>7.2%} {r['parameters']:>12,}")
    print("=" * 55)


if __name__ == "__main__":
    main()
