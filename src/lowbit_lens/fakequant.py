"""Milestone M2: simulated ("fake") quantization at 8, 6, 4 and 2 bits.

    python -m lowbit_lens.fakequant            # both sweeps, all models x seeds
    python -m lowbit_lens.fakequant --force    # redo existing results

PyTorch can only *run* real 8-bit models. To study fewer bits we simulate them:
every weight and activation stays a float, but is snapped to the nearest level
of a b-bit grid (2^b levels) before it is used. The model then makes the same
rounding errors as a real b-bit model would, without being smaller or faster.

Two more reasons this is useful:
  - Grad-CAM needs gradients. Rounding itself has no useful gradient, so, as
    PyTorch's fake quantization always does, gradients pass through the rounding
    step unchanged (the "straight-through" rule) and are zeroed only where a
    value was clipped at the edge of the grid. Real INT8 models give no
    gradients at all.
  - At 8 bits the simulation uses the same recipe as the real INT8 models of M1,
    so comparing the two checks that the simulator is faithful.

Recipe (docs/DECISIONS.md, D15-D18):
  - the same steps as M1: fuse Conv+BN(+ReLU), the same 512 calibration images,
    per-channel symmetric weights, histogram-based activation ranges;
  - first and last layer (and the input) stay at 8 bits;
  - sweep "both":    weights and activations at b bits      (W8A8, W6A6, W4A4, W2A2)
  - sweep "weights": weights at b bits, activations at 8   (W6A8, W4A8, W2A8)
  - activation ranges come from PyTorch's standard HistogramObserver. Its range
    search always assumes 256 levels; a "levels-aware" variant was tested and
    gave equal or lower accuracy, so the standard one is used (D16, and
    `python -m lowbit_lens.fakequant --observer-check`).

Calibration runs on the CPU, as in M1. The calibrated model is then evaluated on
the GPU. Results go to runs/fakequant/<model>_seed<seed>/W<w>A<a>/.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.ao.quantization as tq
from torch import nn
from torch.ao.quantization import FakeQuantize, HistogramObserver, PerChannelMinMaxObserver, QConfig

from .config import DEFAULT_CONFIG, Config, load_config
from .data import Split, iterate_batches, load_split
from .models import load_checkpoint
from .quantize import calibration_indices, int8_dir
from .train import evaluate, run_dir

CPU = torch.device("cpu")

# Layers kept at 8 bits in every setting: the first layer (sees raw pixels) and
# the last one (makes the decision); "quant" is the input-quantization point.
EDGE_LAYERS = {
    "resnet18": ("quant", "conv1", "fc"),
    "mobilenet_v2": ("quant", "features.0.0", "classifier.1"),
    "googlenet": ("quant", "conv1.conv", "fc"),
}


class LevelsAwareHistogramObserver(HistogramObserver):
    """Variant of PyTorch's histogram range-finder that is told the real number of levels.

    The stock HistogramObserver optimises its clipping range for 256 levels
    (2^8) whatever quant_min/quant_max say. This variant optimises for the
    actual grid (e.g. 16 levels at 4 bits), which clips large activations more.
    It was tested against the stock observer and was not better (it lowered
    ResNet-18 W4A4 accuracy), so it is NOT used for the results; it exists only
    for the recorded comparison (--observer-check). Identical to stock at 8 bits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dst_nbins = int(self.quant_max - self.quant_min + 1)


def make_qconfig(weight_bits: int, act_bits: int, observer=HistogramObserver) -> QConfig:
    act = FakeQuantize.with_args(
        observer=observer, quant_min=0, quant_max=2**act_bits - 1,
        dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
    )
    weight = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        quant_min=-(2 ** (weight_bits - 1)), quant_max=2 ** (weight_bits - 1) - 1,
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0,
    )
    return QConfig(activation=act, weight=weight)


@torch.no_grad()
def build_fake_quant(
    fp32: nn.Module, model_name: str, calib: Split, weight_bits: int, act_bits: int,
    batch_size: int, edge_bits: int = 8, observer=HistogramObserver,
) -> nn.Module:
    """FP32 model -> calibrated fake-quantized model (on the CPU, in eval mode)."""
    m = copy.deepcopy(fp32).to(CPU).eval()
    m.fuse_model()  # same fusion as real INT8 (BatchNorm folded into the conv)
    m.qconfig = make_qconfig(weight_bits, act_bits, observer)
    for name in EDGE_LAYERS[model_name]:
        m.get_submodule(name).qconfig = make_qconfig(edge_bits, edge_bits, observer)
    m.train()  # prepare_qat requires training mode; it only swaps in fake-quant layers
    tq.prepare_qat(m, inplace=True)
    m.eval()

    # Calibrate like real post-training quantization: observers watch the
    # *unrounded* values, then rounding is switched on.
    m.apply(tq.disable_fake_quant)
    m.apply(tq.enable_observer)
    for x, _ in iterate_batches(calib, batch_size, CPU):
        m(x)
    m.apply(tq.disable_observer)
    m.apply(tq.enable_fake_quant)
    return m


def settings(cfg: Config) -> list[tuple[str, int, int]]:
    """(sweep, weight bits, activation bits); W8A8 is shared by both sweeps."""
    fq = cfg.fakequant
    out = []
    for b in fq.bits:
        if "both" in fq.sweeps:
            out.append(("both", b, b))
        if "weights" in fq.sweeps and b != 8:
            out.append(("weights", b, 8))
    return out


def fq_dir(cfg: Config, model_name: str, seed: int, w: int, a: int) -> Path:
    return cfg.paths.runs_dir / "fakequant" / f"{model_name}_seed{seed}" / f"W{w}A{a}"


def run_setting(cfg, name, seed, sweep, w, a, fp32, calib, test, labels, device) -> dict:
    t0 = time.perf_counter()
    model = build_fake_quant(fp32, name, calib, w, a, cfg.quant.batch_size, cfg.fakequant.edge_bits)
    calib_s = time.perf_counter() - t0
    res = evaluate(model.to(device), test, cfg.train.eval_batch_size, device, cfg.data.num_classes)
    logits = res["logits"].numpy()

    ref = np.load(run_dir(cfg, name, seed) / "test_logits.npy")  # FP32 on the GPU
    p32, pq = ref.argmax(1), logits.argmax(1)
    ok32, okq = p32 == labels, pq == labels
    metrics = {
        "model": name, "seed": seed, "sweep": sweep, "weight_bits": w, "act_bits": a,
        "edge_bits": cfg.fakequant.edge_bits,
        "test_acc": float(okq.mean()),
        "fp32_test_acc": float(ok32.mean()),
        "agreement_with_fp32": float((p32 == pq).mean()),
        "flips_total": int((p32 != pq).sum()),
        "flips_correct_to_wrong": int((ok32 & ~okq).sum()),
        "flips_wrong_to_correct": int((~ok32 & okq).sum()),
        "calibration_seconds": calib_s,
    }
    if (w, a) == (8, 8):  # faithfulness check against the real INT8 model from M1
        real = np.load(int8_dir(cfg, name, seed) / "test_logits.npy").argmax(1)
        metrics["agreement_with_real_int8"] = float((real == pq).mean())
        metrics["differs_from_real_int8"] = int((real != pq).sum())

    out = fq_dir(cfg, name, seed, w, a)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "test_logits.npy", logits)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    extra = f", differs from real INT8 on {metrics['differs_from_real_int8']}" if (w, a) == (8, 8) else ""
    print(f"{name} seed {seed} W{w}A{a}: acc {metrics['test_acc']:.4f}, "
          f"{metrics['flips_total']} predictions changed vs FP32{extra}")
    return metrics


def observer_check(cfg: Config, calib: Split, test: Split, device: torch.device) -> list[dict]:
    """Stock vs levels-aware activation range search, seed-0 ResNet-18 and MobileNetV2."""
    rows = []
    for name in ("resnet18", "mobilenet_v2"):
        fp32, _ = load_checkpoint(run_dir(cfg, name, 0) / "best.pt", map_location=CPU)
        for w, a in ((4, 4), (2, 8), (2, 2)):
            row = {"model": name, "seed": 0, "setting": f"W{w}A{a}"}
            for label, obs in (("stock", HistogramObserver), ("levels_aware", LevelsAwareHistogramObserver)):
                fq = build_fake_quant(fp32, name, calib, w, a, cfg.quant.batch_size,
                                      cfg.fakequant.edge_bits, observer=obs).to(device)
                row[f"{label}_acc"] = evaluate(fq, test, cfg.train.eval_batch_size, device,
                                               cfg.data.num_classes)["acc"]
            rows.append(row)
            print(f"{name} W{w}A{a}: stock {row['stock_acc']:.4f} | levels-aware {row['levels_aware_acc']:.4f}")
    out = cfg.paths.runs_dir / "fakequant" / "observer_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--force", action="store_true")
    p.add_argument("--observer-check", action="store_true",
                   help="also run the stock vs levels-aware range-search comparison (D16)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    d, qc = cfg.data, cfg.quant

    todo = [
        (name, seed, sweep, w, a)
        for seed in cfg.seeds for name in cfg.train.models for sweep, w, a in settings(cfg)
        if args.force or not (fq_dir(cfg, name, seed, w, a) / "metrics.json").exists()
    ]
    check_path = cfg.paths.runs_dir / "fakequant" / "observer_check.json"
    run_check = args.observer_check and (args.force or not check_path.exists())
    if not todo and not run_check:
        print("all fake-quantization results present (use --force to redo)")
        return

    train = load_split(d.dataset, "train", d.size, cfg.paths.data_dir)
    idx = calibration_indices(len(train), qc.calib_images, qc.calib_seed)  # same images as M1
    calib = Split(train.images[idx].clone(), train.labels[idx].clone())
    del train
    test = load_split(d.dataset, "test", d.size, cfg.paths.data_dir)
    labels = test.labels.numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if run_check:
        observer_check(cfg, calib, test, device)
    loaded = {}
    for name, seed, sweep, w, a in todo:
        if (name, seed) not in loaded:
            loaded.clear()
            loaded[name, seed] = load_checkpoint(run_dir(cfg, name, seed) / "best.pt", map_location=CPU)[0]
        run_setting(cfg, name, seed, sweep, w, a, loaded[name, seed], calib, test, labels, device)


if __name__ == "__main__":
    main()
