"""
Extended training test: Confirm transformer keeps improving
while PredCoding stays flat on span extraction.

Uses 3K train, 500 test, 10 epochs to show learning curve divergence.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import os

from large_predcoding_qa import LargePredCodingQA, build_medium_transformer
from train_large_qa import SQuADv2Dataset, train_epoch, evaluate_squad


def main():
    print("=" * 80)
    print("EXTENDED QA TEST: Learning Curve Divergence")
    print("Does Transformer keep improving while PredCoding stays flat?")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_seq_len = 384
    batch_size = 8

    # Load data
    print("\n--- Loading SQuAD 2.0 ---")
    train_ds = SQuADv2Dataset("train", max_len=max_seq_len, max_samples=3000)
    test_ds = SQuADv2Dataset("validation", max_len=max_seq_len, max_samples=500)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    train_ans = sum(1 for ex in train_ds.examples if ex['is_answerable'])
    test_ans = sum(1 for ex in test_ds.examples if ex['is_answerable'])
    print(f"  Train: {len(train_ds)} ({train_ans} answerable)")
    print(f"  Test: {len(test_ds)} ({test_ans} answerable)")

    vocab_size = 30522
    num_epochs = 10

    # Models: PredCoding (PC-Pointer) vs Transformer
    models = {
        "PredCoding": LargePredCodingQA(
            vocab_size=vocab_size, embed_dim=256, hidden_dim=1024,
            num_layers=6, max_iterations=10, max_seq_len=max_seq_len, dropout=0.1,
        ),
        "Transformer": build_medium_transformer(vocab_size),
    }

    all_curves = {}

    for name, model in models.items():
        model = model.to(device)
        print(f"\n{'='*80}")
        print(f"TRAINING: {name} ({model.count_parameters():,} params)")
        print(f"{'='*80}")

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        curve = []
        for epoch in range(num_epochs):
            epoch_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, device)
            epoch_time = time.time() - epoch_start
            scheduler.step()

            metrics, _ = evaluate_squad(model, test_loader, device)

            print(f"  Epoch {epoch+1:>2}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Ans_F1: {metrics['answerable_f1']:.4f} | "
                  f"Ans_EM: {metrics['answerable_em']:.4f} | "
                  f"Time: {epoch_time:.1f}s")

            curve.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                'ans_f1': metrics['answerable_f1'],
                'ans_em': metrics['answerable_em'],
                'overall_em': metrics['overall_em'],
            })

        all_curves[name] = curve
        model = model.cpu()

    # Summary
    print(f"\n{'='*80}")
    print("LEARNING CURVE COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Epoch':>5} {'PC Loss':>8} {'PC F1':>8} {'TF Loss':>8} {'TF F1':>8} {'Gap':>8}")
    print("-" * 50)
    for epoch in range(num_epochs):
        pc = all_curves["PredCoding"][epoch]
        tf = all_curves["Transformer"][epoch]
        gap = tf['ans_f1'] - pc['ans_f1']
        print(f"{epoch+1:>5} {pc['loss']:>7.4f} {pc['ans_f1']:>7.4f} "
              f"{tf['loss']:>7.4f} {tf['ans_f1']:>7.4f} {gap:>+7.4f}")

    # Verdict
    pc_final = all_curves["PredCoding"][-1]['ans_f1']
    tf_final = all_curves["Transformer"][-1]['ans_f1']

    print(f"\n--- Verdict ---")
    print(f"  PredCoding final Ans_F1: {pc_final:.4f}")
    print(f"  Transformer final Ans_F1: {tf_final:.4f}")

    if pc_final < 0.01 and tf_final > 0.01:
        print(f"  CONFIRMED: Pooling prevents span learning")
        print(f"  Transformer shows learning; PredCoding flat at 0")
    elif pc_final < 0.01 and tf_final < 0.01:
        print(f"  INCONCLUSIVE: Both failed (need more data/epochs)")
    else:
        print(f"  SURPRISING: PredCoding learned! ({pc_final:.4f})")

    save_path = os.path.join(os.path.dirname(__file__), "extended_qa_curves.json")
    with open(save_path, "w") as f:
        json.dump(all_curves, f, indent=2, default=str)
    print(f"\nCurves saved to {save_path}")


if __name__ == "__main__":
    main()
