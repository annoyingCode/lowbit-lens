"""Milestone M1: real INT8 post-training quantization, size and CPU speed.

    python -m lowbit_lens.quantize              # quantize all models x seeds, then time them
    python -m lowbit_lens.quantize --force      # redo even if results exist

What happens to each trained FP32 model, in order:

1. Fuse. Conv -> BatchNorm -> ReLU are merged into one layer. They are
   computed together anyway, and merging means rounding happens once instead
   of three times (less error, faster).
2. Prepare. Small "observers" are attached to each layer. They do not change
   the output; they only record what values pass through.
3. Calibrate. 512 training images are passed through the model so the
   observers learn the typical range of every layer's activations (the
   in-between values). Weights do not need images: their range is fixed.
4. Convert. Every weight is rounded onto an 8-bit grid (256 levels), and each
   layer is replaced by an integer version that also rounds its outputs. The
   result is a real INT8 model that runs integer arithmetic on the CPU.

Rounding recipe (PyTorch's standard one for the onednn engine):
  - weights: one grid per output filter ("per-channel"), symmetric around 0,
    sized by that filter's smallest and largest weight;
  - activations: one grid per layer, its range picked from a histogram of the
    calibration values so a few extreme values do not stretch it.

Important limitation for later milestones: INT8 layers have no gradients, so
Grad-CAM cannot run on these models. M2 adds simulated ("fake") quantization
for that; M3 runs a gradient-free method on these real INT8 models.

Outputs per run (runs/int8/<model>_seed<seed>/):
    model_int8.pt       INT8 weights (reload with load_int8)
    test_logits.npy     per-image INT8 test scores
    metrics.json        accuracy, agreement with FP32, size, calibration info
and runs/fp32/<run>/test_logits_cpu.npy: the FP32 model's test scores on the
CPU, so FP32 and INT8 are compared on the same hardware. Timing results go to
runs/int8/latency.json.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.ao.quantization as tq
from torch import nn

from .config import DEFAULT_CONFIG, Config, load_config
from .data import Split, iterate_batches, load_split
from .models import build_model, load_checkpoint
from .train import run_dir

CPU = torch.device("cpu")
DEFAULT_THREADS = torch.get_num_threads()


def int8_dir(cfg: Config, model_name: str, seed: int) -> Path:
    return cfg.paths.runs_dir / "int8" / f"{model_name}_seed{seed}"


def calibration_indices(n_train: int, n: int, seed: int) -> torch.Tensor:
    """The same fixed random training images for every model."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_train, generator=g)[:n].sort().values


def _prepare(model: nn.Module, backend: str) -> nn.Module:
    torch.backends.quantized.engine = backend
    m = copy.deepcopy(model).to(CPU).eval()
    m.fuse_model()  # step 1: merge Conv+BN(+ReLU)
    m.qconfig = tq.get_default_qconfig(backend)  # the standard rounding recipe
    tq.prepare(m, inplace=True)  # step 2: attach observers
    return m


@torch.no_grad()
def quantize_int8(model: nn.Module, calib: Split, backend: str, batch_size: int) -> nn.Module:
    """FP32 model -> real INT8 model (the input model is not modified)."""
    m = _prepare(model, backend)
    for x, _ in iterate_batches(calib, batch_size, CPU):  # step 3: calibrate
        m(x)
    tq.convert(m, inplace=True)  # step 4: round and swap in integer layers
    return m


def int8_skeleton(model_name: str, num_classes: int, backend: str) -> nn.Module:
    """An uncalibrated INT8 model with the right structure, for loading weights."""
    m = _prepare(build_model(model_name, num_classes, pretrained=False), backend)
    return tq.convert(m, inplace=False)


def load_int8(path: Path, backend: str = "onednn") -> nn.Module:
    ckpt = torch.load(path, map_location=CPU, weights_only=False)
    m = int8_skeleton(ckpt["model_name"], ckpt["num_classes"], backend)
    m.load_state_dict(ckpt["state_dict"])
    return m.eval()


def state_dict_megabytes(model: nn.Module) -> float:
    """Size of the saved weights, as they would be shipped (MB = 10^6 bytes)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@torch.no_grad()
def cpu_logits(model: nn.Module, split: Split, batch_size: int) -> np.ndarray:
    model.eval()
    out = [model(x).float() for x, _ in iterate_batches(split, batch_size, CPU)]
    return torch.cat(out).numpy()


@torch.no_grad()
def time_model(model: nn.Module, x: torch.Tensor, threads: int, runs: int, warmup: int) -> dict:
    """Milliseconds per single-image forward pass: median and middle 50% range."""
    torch.set_num_threads(threads)
    try:
        model.eval()
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1e3)
    finally:
        torch.set_num_threads(DEFAULT_THREADS)
    q = statistics.quantiles(times, n=4)
    return {"median_ms": statistics.median(times), "p25_ms": q[0], "p75_ms": q[2]}


def quantize_run(cfg: Config, name: str, seed: int, calib: Split, calib_hash: str,
                 test: Split, labels: np.ndarray) -> dict:
    qc = cfg.quant
    fp32_dir = run_dir(cfg, name, seed)
    fp32, _ = load_checkpoint(fp32_dir / "best.pt", map_location=CPU)

    # FP32 reference on the CPU: the only difference to INT8 is then quantization.
    cpu_path = fp32_dir / "test_logits_cpu.npy"
    if not cpu_path.exists():
        np.save(cpu_path, cpu_logits(fp32, test, qc.batch_size))
    fp32_cpu = np.load(cpu_path)
    fp32_gpu = np.load(fp32_dir / "test_logits.npy")

    t0 = time.perf_counter()
    int8 = quantize_int8(fp32, calib, qc.backend, qc.batch_size)
    quant_seconds = time.perf_counter() - t0
    int8_logits = cpu_logits(int8, test, qc.batch_size)

    p32, p8 = fp32_cpu.argmax(1), int8_logits.argmax(1)
    ok32, ok8 = p32 == labels, p8 == labels
    out = int8_dir(cfg, name, seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_name": name, "num_classes": cfg.data.num_classes,
         "state_dict": int8.state_dict(), "backend": qc.backend},
        out / "model_int8.pt",
    )
    np.save(out / "test_logits.npy", int8_logits)
    metrics = {
        "model": name,
        "seed": seed,
        "backend": qc.backend,
        "calibration": {"images": len(calib), "seed": qc.calib_seed, "sha1_of_indices": calib_hash},
        "fp32_cpu_test_acc": float(ok32.mean()),
        "int8_test_acc": float(ok8.mean()),
        "agreement_with_fp32": float((p32 == p8).mean()),
        "flips_total": int((p32 != p8).sum()),
        "flips_correct_to_wrong": int((ok32 & ~ok8).sum()),
        "flips_wrong_to_correct": int((~ok32 & ok8).sum()),
        "fp32_gpu_vs_cpu_agreement": float((fp32_gpu.argmax(1) == p32).mean()),
        "fp32_size_mb": state_dict_megabytes(fp32),
        "int8_size_mb": state_dict_megabytes(int8),
        "quantize_seconds": quant_seconds,
        "torch": torch.__version__,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"{name} seed {seed}: FP32 {metrics['fp32_cpu_test_acc']:.4f} -> INT8 "
        f"{metrics['int8_test_acc']:.4f}, {metrics['flips_total']} predictions changed, "
        f"{metrics['fp32_size_mb']:.1f} MB -> {metrics['int8_size_mb']:.1f} MB"
    )
    return metrics


def measure_latency(cfg: Config, test: Split) -> dict:
    """Single-thread FP32 vs INT8 timing in alternating rounds, plus a multi-thread check.

    Timing on a laptop drifts (heat, background load, and on hybrid CPUs whether
    Windows runs the job on a fast or a slow core). Alternating FP32 and INT8
    within each round and repeating rounds exposes that drift instead of hiding it.
    """
    lc = cfg.latency
    x, _ = next(iterate_batches(Split(test.images[:1], test.labels[:1]), 1, CPU))
    models = {}
    for seed in lc.seeds:
        for name in cfg.train.models:
            fp32, _ = load_checkpoint(run_dir(cfg, name, seed) / "best.pt", map_location=CPU)
            int8 = load_int8(int8_dir(cfg, name, seed) / "model_int8.pt", cfg.quant.backend)
            models[name, seed] = {"fp32": fp32, "int8": int8}

    plan = [("main", 1, lc.rounds, lc.runs)] + [
        ("diagnostic", t, lc.diagnostic_rounds, lc.diagnostic_runs) for t in lc.diagnostic_threads
    ]
    rows = []
    for kind, threads, rounds, runs in plan:
        for rnd in range(rounds):
            order = ("fp32", "int8") if rnd % 2 == 0 else ("int8", "fp32")
            for (name, seed), pair in models.items():
                for precision in order:
                    r = time_model(pair[precision], x, threads, runs, lc.warmup)
                    rows.append({"kind": kind, "model": name, "seed": seed, "threads": threads,
                                 "round": rnd, "precision": precision, **r})
            print(f"{kind} timing, {threads} thread(s), round {rnd + 1}/{rounds} done")
    result = {
        "rows": rows,
        "setup": {
            "batch_size": 1,
            "runs_per_round": lc.runs,
            "rounds": lc.rounds,
            "warmup": lc.warmup,
            "backend": cfg.quant.backend,
            "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "torch": torch.__version__,
            "note": "single laptop; times depend on power mode, temperature, background load "
                    "and core placement",
        },
    }
    path = cfg.paths.runs_dir / "int8" / "latency.json"
    path.write_text(json.dumps(result, indent=2))
    return result


def cpu_name() -> str:
    """Readable CPU model name where the OS offers one, else Python's description."""
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--force", action="store_true", help="redo runs that already have results")
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--latency-only", action="store_true", help="only (re)run the CPU timing")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    d, qc = cfg.data, cfg.quant

    todo = [
        (m, s) for s in cfg.seeds for m in cfg.train.models
        if args.force or not (int8_dir(cfg, m, s) / "metrics.json").exists()
    ]
    if args.latency_only:
        todo = []
    if todo:
        train = load_split(d.dataset, "train", d.size, cfg.paths.data_dir)
        idx = calibration_indices(len(train), qc.calib_images, qc.calib_seed)
        calib = Split(train.images[idx].clone(), train.labels[idx].clone())
        calib_hash = hashlib.sha1(idx.numpy().tobytes()).hexdigest()
        del train
        test = load_split(d.dataset, "test", d.size, cfg.paths.data_dir)
        labels = test.labels.numpy()
        for name, seed in todo:
            quantize_run(cfg, name, seed, calib, calib_hash, test, labels)
    else:
        print("all INT8 runs present (use --force to redo)")
        test = None

    latency_path = cfg.paths.runs_dir / "int8" / "latency.json"
    if not args.skip_latency and (args.force or args.latency_only or todo or not latency_path.exists()):
        if test is None:
            test = load_split(d.dataset, "test", d.size, cfg.paths.data_dir)
        measure_latency(cfg, test)


if __name__ == "__main__":
    main()
