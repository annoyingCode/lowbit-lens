"""Build results tables from finished runs (never from hand-typed numbers).

    python -m lowbit_lens.report
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics

import numpy as np

from .audit import load_excluded_test_indices
from .config import DEFAULT_CONFIG, Config, load_config
from .data import load_labels
from .train import run_dir


def load_metrics(cfg: Config) -> dict[tuple[str, int], dict]:
    out, missing = {}, []
    for name in cfg.train.models:
        for seed in cfg.seeds:
            path = run_dir(cfg, name, seed) / "metrics.json"
            if path.exists():
                out[name, seed] = json.loads(path.read_text())
            else:
                missing.append(f"{name} seed {seed}")
    if missing:
        raise FileNotFoundError(
            "missing finished runs: " + ", ".join(missing)
            + "\ntrain them with: python -m lowbit_lens.train"
        )
    return out


def add_clean_test_acc(cfg: Config, metrics: dict[tuple[str, int], dict]) -> int:
    """Accuracy on the test set without images that also occur in train/val.

    Returns how many test images were left out (see audit.py).
    """
    labels = load_labels(cfg.data.dataset, "test", cfg.data.size, cfg.paths.data_dir)
    keep = np.ones(len(labels), dtype=bool)
    excluded = load_excluded_test_indices(cfg)
    keep[excluded] = False
    for (name, seed), m in metrics.items():
        logits = np.load(run_dir(cfg, name, seed) / "test_logits.npy")
        pred = logits.argmax(1)
        # Compare counts of correct images: metrics.json stores accuracy computed
        # in float32, so comparing accuracies directly would flag rounding noise.
        correct = int((pred == labels).sum())
        if correct != round(m["test_acc"] * len(labels)):
            raise RuntimeError(f"{name} seed {seed}: test_logits.npy disagrees with metrics.json")
        m["test_acc_clean"] = float((pred[keep] == labels[keep]).mean())
    return len(excluded)


def mean_sd(xs: list[float], digits: int = 4) -> str:
    """'mean ± sample SD' over seeds (SD over 3 seeds is a rough spread, not a CI)."""
    if len(xs) == 1:
        return f"{xs[0]:.{digits}f}"
    return f"{statistics.mean(xs):.{digits}f} ± {statistics.stdev(xs):.{digits}f}"


def fp32_tables(cfg: Config) -> tuple[list[dict], list[dict], int]:
    metrics = load_metrics(cfg)
    n_excluded = add_clean_test_acc(cfg, metrics)
    per_seed = [
        {
            "model": name,
            "seed": seed,
            "params_M": round(m["params"] / 1e6, 2),
            "val_acc": f"{m['val_acc']:.4f}",
            "test_acc": f"{m['test_acc']:.4f}",
            "test_acc_clean": f"{m['test_acc_clean']:.4f}",
            "best_epoch": f"{m['best_epoch']}/{m['epochs']}",
            "train_min": round(m["train_minutes"], 1),
        }
        for (name, seed), m in metrics.items()
    ]
    summary = []
    for name in cfg.train.models:
        ms = [metrics[name, s] for s in cfg.seeds]
        summary.append(
            {
                "model": name,
                "params_M": round(ms[0]["params"] / 1e6, 2),
                "seeds": len(ms),
                "val_acc": mean_sd([m["val_acc"] for m in ms]),
                "test_acc": mean_sd([m["test_acc"] for m in ms]),
                "test_acc_clean": mean_sd([m["test_acc_clean"] for m in ms]),
            }
        )
    return summary, per_seed, n_excluded


def write_table(rows: list[dict], stem: str, cfg: Config, note: str) -> None:
    out = cfg.paths.results_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
        *("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows),
    ]
    (out / f"{stem}.md").write_text(f"{note}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / stem}.csv and .md")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    summary, per_seed, n_excluded = fp32_tables(cfg)
    base = (
        f"FP32 baselines on BloodMNIST {cfg.data.size}px, seeds {list(cfg.seeds)}. "
        "Test accuracy is for the checkpoint with the best validation accuracy; "
        "hyperparameters were not tuned (see docs/DECISIONS.md). "
        f"test_acc uses the official test set; test_acc_clean leaves out the {n_excluded} "
        "test images that are exact copies of training or validation images "
        "(results/tables/duplicate_images.csv)."
    )
    write_table(summary, "fp32_baselines", cfg, base + " Values are mean ± sample SD over seeds.")
    write_table(per_seed, "fp32_baselines_per_seed", cfg, base)


if __name__ == "__main__":
    main()
