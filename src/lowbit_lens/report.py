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


def int8_tables(cfg: Config) -> tuple[list[dict], list[dict]] | None:
    """Real INT8 post-training quantization vs FP32, both evaluated on the CPU."""
    from .quantize import int8_dir

    paths = {(m, s): int8_dir(cfg, m, s) / "metrics.json" for s in cfg.seeds for m in cfg.train.models}
    if not all(p.exists() for p in paths.values()):
        return None
    labels = load_labels(cfg.data.dataset, "test", cfg.data.size, cfg.paths.data_dir)
    keep = np.ones(len(labels), dtype=bool)
    keep[load_excluded_test_indices(cfg)] = False
    ms = {}
    for (name, seed), path in paths.items():
        m = json.loads(path.read_text())
        pred = np.load(int8_dir(cfg, name, seed) / "test_logits.npy").argmax(1)
        if int((pred == labels).sum()) != round(m["int8_test_acc"] * len(labels)):
            raise RuntimeError(f"{name} seed {seed}: INT8 logits disagree with metrics.json")
        m["int8_test_acc_clean"] = float((pred[keep] == labels[keep]).mean())
        ms[name, seed] = m

    per_seed = [
        {
            "model": name,
            "seed": seed,
            "fp32_acc": f"{m['fp32_cpu_test_acc']:.4f}",
            "int8_acc": f"{m['int8_test_acc']:.4f}",
            "int8_acc_clean": f"{m['int8_test_acc_clean']:.4f}",
            "same_prediction": f"{m['agreement_with_fp32']:.4f}",
            "changed": m["flips_total"],
            "right_to_wrong": m["flips_correct_to_wrong"],
            "wrong_to_right": m["flips_wrong_to_correct"],
            "fp32_MB": round(m["fp32_size_mb"], 1),
            "int8_MB": round(m["int8_size_mb"], 1),
        }
        for (name, seed), m in ms.items()
    ]
    summary = []
    for name in cfg.train.models:
        rs = [ms[name, s] for s in cfg.seeds]
        summary.append(
            {
                "model": name,
                "fp32_acc": mean_sd([r["fp32_cpu_test_acc"] for r in rs]),
                "int8_acc": mean_sd([r["int8_test_acc"] for r in rs]),
                "int8_acc_clean": mean_sd([r["int8_test_acc_clean"] for r in rs]),
                "same_prediction": mean_sd([r["agreement_with_fp32"] for r in rs]),
                "changed_of_3421": " / ".join(str(r["flips_total"]) for r in rs),
                "fp32_MB": round(rs[0]["fp32_size_mb"], 1),
                "int8_MB": round(rs[0]["int8_size_mb"], 1),
                "smaller_by": f"{rs[0]['fp32_size_mb'] / rs[0]['int8_size_mb']:.2f}x",
            }
        )
    return summary, per_seed


def latency_tables(cfg: Config) -> tuple[list[dict], list[dict], dict] | None:
    path = cfg.paths.runs_dir / "int8" / "latency.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if "rounds" not in data["setup"]:
        return None  # older format; rerun: python -m lowbit_lens.quantize --latency-only
    rows = data["rows"]

    def medians(kind, name, threads, precision):
        return [r["median_ms"] for r in rows if (r["kind"], r["model"], r["threads"],
                r["precision"]) == (kind, name, threads, precision)]

    main = []
    for name in cfg.train.models:
        f, q = medians("main", name, 1, "fp32"), medians("main", name, 1, "int8")
        ratios = [a / b for a, b in zip(f, q)]
        main.append({
            "model": name,
            "fp32_ms": f"{statistics.median(f):.1f} ({min(f):.1f}-{max(f):.1f})",
            "int8_ms": f"{statistics.median(q):.1f} ({min(q):.1f}-{max(q):.1f})",
            "int8_faster_by": f"{statistics.median(ratios):.1f}x ({min(ratios):.1f}-{max(ratios):.1f})",
        })
    diag = []
    for name in cfg.train.models:
        threads = sorted({r["threads"] for r in rows if r["kind"] == "diagnostic"})
        f1 = statistics.median(medians("main", name, 1, "fp32"))
        q1 = statistics.median(medians("main", name, 1, "int8"))
        for t in threads:
            f, q = medians("diagnostic", name, t, "fp32"), medians("diagnostic", name, t, "int8")
            diag.append({
                "model": name,
                "threads": t,
                "fp32_ms": f"{statistics.median(f):.1f} ({min(f):.1f}-{max(f):.1f})",
                "int8_ms": f"{statistics.median(q):.1f} ({min(q):.1f}-{max(q):.1f})",
                "fp32_vs_1_thread": f"{statistics.median(f) / f1:.2f}x",
                "int8_vs_1_thread": f"{statistics.median(q) / q1:.2f}x",
            })
    return main, diag, data["setup"]


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

    int8 = int8_tables(cfg)
    if int8 is None:
        print("INT8 results not complete yet; skipping INT8 tables")
    else:
        summary8, per_seed8 = int8
        note8 = (
            f"Real INT8 post-training quantization ({cfg.quant.calib_images} calibration images "
            f"from the training set, PyTorch's standard {cfg.quant.backend} recipe). FP32 and INT8 "
            "are both evaluated on the CPU on the official test set (3421 images); int8_acc_clean "
            f"leaves out the {n_excluded} duplicated test images. same_prediction = share of test "
            "images where INT8 predicts the same class as FP32; changed = number of test images "
            "whose prediction changed. Sizes are saved weights in MB (10^6 bytes)."
        )
        write_table(summary8, "int8_ptq", cfg, note8 + " Values are mean ± sample SD over seeds; "
                    "changed_of_3421 lists seeds " + ", ".join(map(str, cfg.seeds)) + ".")
        write_table(per_seed8, "int8_ptq_per_seed", cfg, note8)

    lat = latency_tables(cfg)
    if lat is None:
        print("latency results missing or outdated; skipping latency tables")
    else:
        main_rows, diag_rows, setup = lat
        where = (f"One laptop only ({setup['cpu']}, {setup['logical_cpus']} logical CPUs), "
                 f"PyTorch {setup['torch']}, {setup['backend']} INT8 engine, batch size 1, "
                 "seed-0 models. Times on other machines or engines can differ a lot.")
        write_table(
            main_rows, "cpu_latency", cfg,
            f"CPU time per image on 1 thread, in milliseconds. {setup['rounds']} rounds alternate "
            f"FP32 and INT8; each round's value is the median of {setup['runs_per_round']} runs "
            f"after {setup['warmup']} warm-up runs. Shown: median over rounds (lowest-highest "
            "round in brackets); int8_faster_by is the median of per-round FP32/INT8 ratios. "
            + where,
        )
        write_table(
            diag_rows, "cpu_latency_multithread", cfg,
            "Multi-thread check, reported as a limitation of this setup. *_vs_1_thread compares "
            "each median with the same model's 1-thread median from cpu_latency: below 1x means "
            "more threads helped, above 1x means they made it slower. Where INT8 gets slower with "
            "more threads while FP32 gets faster, the cause is most likely the threading of this "
            "PyTorch build on Windows (not investigated further), not INT8 as such. "
            "Median over rounds (lowest-highest round in brackets), milliseconds. " + where,
        )


if __name__ == "__main__":
    main()
