"""Fine-tune the FP32 baselines on BloodMNIST.

    python -m lowbit_lens.train                      # every model x every seed in the config
    python -m lowbit_lens.train --seeds 0            # one seed
    python -m lowbit_lens.train --models resnet18    # just one
    python -m lowbit_lens.train --epochs 1 --limit 512   # quick smoke test

Each run writes to runs/fp32/<model>_seed<seed>/ (runs/smoke/ when --epochs or
--limit is given, so a smoke test never overwrites a real checkpoint):
    best.pt        weights with the best validation accuracy (+ metadata)
    history.csv    per-epoch train/val loss and accuracy
    metrics.json   test-set results of best.pt, computed once after training
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from .config import DEFAULT_CONFIG, Config, load_config
from .data import Split, iterate_batches, load_split
from .models import build_model, count_parameters
from .seed import seed_everything


def run_dir(cfg: Config, model_name: str, seed: int, kind: str = "fp32") -> Path:
    """kind="smoke" keeps quick test runs away from the real checkpoints."""
    return cfg.paths.runs_dir / kind / f"{model_name}_seed{seed}"


@torch.no_grad()
def evaluate(
    model: nn.Module, split: Split, batch_size: int, device: torch.device, num_classes: int
) -> dict:
    """Loss, accuracy, confusion matrix and per-image logits (raw class scores).

    Always in full FP32 (no autocast).
    """
    model.eval()
    total_loss, all_logits = 0.0, []
    for x, y in iterate_batches(split, batch_size, device):
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        all_logits.append(logits.float().cpu())
    logits = torch.cat(all_logits)
    pred = logits.argmax(1)
    labels = split.labels
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    confusion.index_put_((labels, pred), torch.ones_like(labels), accumulate=True)
    return {
        "loss": total_loss / len(split),
        "acc": (pred == labels).float().mean().item(),
        "confusion": confusion.tolist(),  # rows = true class, columns = predicted
        "logits": logits,  # (N, num_classes), in split order
    }


def train_one(
    cfg: Config,
    model_name: str,
    train: Split,
    val: Split,
    test: Split,
    device: torch.device,
    *,
    seed: int,
    epochs: int,
    kind: str = "fp32",
) -> dict:
    out = run_dir(cfg, model_name, seed, kind)
    out.mkdir(parents=True, exist_ok=True)
    gen = seed_everything(seed)  # same seed -> same head init, batch order, augmentation

    model = build_model(model_name, cfg.data.num_classes, pretrained=True).to(device)
    tc = cfg.train
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    steps_per_epoch = -(-len(train) // tc.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tc.lr, total_steps=epochs * steps_per_epoch, pct_start=0.1
    )
    amp_dtype = getattr(torch, tc.amp_dtype)
    use_amp = device.type == "cuda" and amp_dtype != torch.float32
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    print(f"\n== {model_name} (seed {seed}): {count_parameters(model) / 1e6:.2f}M params ==")
    history, best_acc, best_epoch = [], -1.0, -1
    for epoch in range(1, epochs + 1):
        model.train()
        t0, seen, correct, loss_sum = time.perf_counter(), 0, 0, 0.0
        batches = iterate_batches(
            train, tc.batch_size, device, shuffle=True, augment=True, generator=gen
        )
        for x, y in batches:
            # Mixed precision: compute in bfloat16 where safe, keep FP32 master
            # weights. This is a training speed-up only; evaluation and every
            # later experiment use full FP32 as the reference.
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=tc.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            seen += y.numel()
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()

        v = evaluate(model, val, tc.eval_batch_size, device, cfg.data.num_classes)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_acc": correct / seen,
            "val_loss": v["loss"],
            "val_acc": v["acc"],
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        flag = ""
        if v["acc"] > best_acc:
            best_acc, best_epoch, flag = v["acc"], epoch, "  *best"
            torch.save(
                {
                    "model_name": model_name,
                    "num_classes": cfg.data.num_classes,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": v["acc"],
                    "seed": seed,
                    "config": cfg.to_dict(),
                },
                out / "best.pt",
            )
        print(
            f"epoch {epoch:2d}/{epochs}  train loss {row['train_loss']:.4f} "
            f"acc {row['train_acc']:.4f} | val loss {row['val_loss']:.4f} "
            f"acc {row['val_acc']:.4f} | {row['seconds']:.0f}s{flag}"
        )

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    # Test set: evaluated exactly once, on the checkpoint chosen by validation.
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    t = evaluate(model, test, tc.eval_batch_size, device, cfg.data.num_classes)
    metrics = {
        "model": model_name,
        "seed": seed,
        "params": count_parameters(model),
        "best_epoch": best_epoch,
        "epochs": epochs,
        "val_acc": best_acc,
        "test_acc": t["acc"],
        "test_loss": t["loss"],
        "test_confusion": t["confusion"],
        "train_size": len(train),
        "image_size": cfg.data.size,
        "train_minutes": sum(r["seconds"] for r in history) / 60,
        "env": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "python": platform.python_version(),
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Per-image test scores, reused for the clean test score and, later, to find
    # images whose prediction flips after quantization.
    np.save(out / "test_logits.npy", t["logits"].numpy())
    print(f"test acc {t['acc']:.4f} (best val {best_acc:.4f} at epoch {best_epoch})")
    return metrics


def subset(split: Split, n: int | None) -> Split:
    return split if n is None else Split(split.images[:n], split.labels[:n])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--models", nargs="+", help="subset of train.models in the config")
    p.add_argument("--seeds", type=int, nargs="+", help="subset of the config seeds")
    p.add_argument("--epochs", type=int, help="override for smoke tests only")
    p.add_argument("--limit", type=int, help="use only the first N images per split (smoke test)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    seeds = args.seeds or cfg.seeds
    seed_everything(seeds[0])  # sets cuBLAS determinism before any CUDA work
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available, training on CPU will be very slow.")

    d = cfg.data
    train, val, test = (
        subset(load_split(d.dataset, s, d.size, cfg.paths.data_dir), args.limit)
        for s in ("train", "val", "test")
    )
    print(f"BloodMNIST {d.size}px: train {len(train)}, val {len(val)}, test {len(test)}")

    kind = "smoke" if (args.epochs or args.limit) else "fp32"
    if kind == "smoke":
        print("Smoke-test settings: writing to runs/smoke/, not runs/fp32/.")
    for seed in seeds:
        for name in args.models or cfg.train.models:
            train_one(
                cfg, name, train, val, test, device,
                seed=seed, epochs=args.epochs or cfg.train.epochs, kind=kind,
            )


if __name__ == "__main__":
    main()
