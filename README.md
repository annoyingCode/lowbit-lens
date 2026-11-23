# lowbit-lens

**When a CNN is quantized, its accuracy often barely changes. Does it still base
its decisions on the same evidence?**

This project measures how Grad-CAM explanations of image classifiers change as
numeric precision drops from 32-bit floats to 8, 6, 4 and 2 bits, on blood-cell
microscopy images (BloodMNIST).

**Scope of the claims.** The dataset has no ground-truth regions, so the project
tests whether quantized models stay *consistent* with the FP32 model's
explanations and whether those explanations stay *faithful* to the model. It does
not test whether any model looks at the medically correct evidence. Each design
choice and its alternatives are recorded in [docs/DECISIONS.md](docs/DECISIONS.md).

> **Status: work in progress.** Done: FP32 baselines (M0) and real INT8
> quantization (M1). Next: the bit-width sweep (M2), then the explanations.
> Every number here comes from a table the code generates; see
> [docs/MILESTONES.md](docs/MILESTONES.md) for the plan.

## Study design

| | |
|---|---|
| **Data** | [BloodMNIST](https://medmnist.com/) (MedMNIST v2): 17,092 images, 8 blood-cell classes, official train/val/test split, 224 px |
| **Models** | ResNet-18, MobileNetV2, GoogLeNet: ImageNet-pretrained (torchvision), fine-tuned in FP32 with 3 seeds each |
| **Quantization** | Real INT8 post-training static quantization; simulated ("fake") quantization at 8/6/4/2 bits; optionally INT8 quantization-aware training |
| **Explanations** | Grad-CAM on FP32 and fake-quantized models; Score-CAM (gradient-free) on FP32, fake and real INT8 models |
| **Metrics** | Accuracy, model size, CPU latency; heatmap similarity to FP32 (Spearman rank correlation, top-k% region IoU, SSIM); faithfulness (deletion/insertion AUC); prediction-flipped vs unchanged images analysed separately |
| **Noise floor** | Heatmap similarity between FP32 models that differ only in their training seed: quantization effects are judged against it |
| **Sanity check** | Model-parameter randomization test (Adebayo et al., 2018) |

**A technical trap the design works around:** real INT8 PyTorch models run on
integer kernels that have no autograd, so gradient-based Grad-CAM cannot run on
them. Grad-CAM therefore runs on fake-quantized models (float tensors rounded to
the low-bit grid, which keeps gradients), and a gradient-free method (Score-CAM)
runs on the true INT8 models to check that the fake-quantized picture holds.

## Results

Explanation results come in later milestones. So far (all tables are generated
by the code):

- **FP32 baselines:** [fp32_baselines.md](results/tables/fp32_baselines.md):
  test accuracy for 3 models x 3 seeds, standard untuned training recipe.
- **Real INT8 post-training quantization:** [int8_ptq.md](results/tables/int8_ptq.md):
  accuracy, agreement with FP32 predictions and model size. On this dataset
  INT8 changes very few predictions; whether it changes the explanations is
  what the next milestones test.
- **CPU speed (one laptop only):** [cpu_latency.md](results/tables/cpu_latency.md)
  for 1 thread, measured in alternating rounds, and
  [cpu_latency_multithread.md](results/tables/cpu_latency_multithread.md), a
  limitation of this setup (D13 in the decision log). Speed numbers from one
  laptop should not be read as general speed-ups.

**Data audit.** The official BloodMNIST split contains a few byte-identical
images, including 6 test images that also appear in the training or validation
set (one with a conflicting label). Results use the official split and also
report a clean test accuracy without those 6 images; they are also excluded from
the heatmap analysis. Details: [docs/DECISIONS.md](docs/DECISIONS.md) (D10) and
[results/tables/duplicate_images.csv](results/tables/duplicate_images.csv).

## Reproduce

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU for
training (developed on an RTX 4050 Laptop, 6 GB). The lockfile pins PyTorch
2.14.1 built for CUDA 13.0, which needs a recent NVIDIA driver; no separate CUDA
toolkit is required.

```bash
uv sync                                  # create .venv from uv.lock
uv run pytest                            # fast CPU checks
uv run python -m lowbit_lens.reproduce   # download data, train, regenerate tables
```

All settings live in [configs/default.toml](configs/default.toml); seeds are fixed
and cuDNN runs in deterministic mode. A few GPU training kernels have no
deterministic version, so retraining can differ very slightly from run to run;
that variation is much smaller than the seed-to-seed spread the results report. BloodMNIST (a 1.5 GB file at 224 px) downloads
to `data/` on first run. If Zenodo is slow, `bash scripts/download_data.sh` fetches
the same file over parallel resumable connections and checks its MD5.

Useful pieces on their own:

```bash
uv run python -m lowbit_lens.train --models resnet18 --seeds 0   # one run
uv run python -m lowbit_lens.train --epochs 1 --limit 512     # smoke test (writes to runs/smoke/)
uv run python -m lowbit_lens.report                           # rebuild tables from finished runs
```

## Repository layout

```
configs/default.toml     every experiment setting
src/lowbit_lens/
  config.py              typed config loading
  seed.py                seeding and determinism
  data.py                BloodMNIST loading, GPU-side augmentation
  models.py              quantizable ResNet-18 / MobileNetV2 / GoogLeNet
  train.py               FP32 fine-tuning and test evaluation
  audit.py               duplicate-image check across splits
  predict.py             per-image test predictions for finished runs
  quantize.py            real INT8 quantization, size and CPU timing (M1)
  report.py              tables built from run outputs
  reproduce.py           the one command that regenerates everything
tests/                   CPU unit tests
results/                 generated tables and figures (committed)
runs/, data/             checkpoints and downloads (not committed)
scripts/download_data.sh fallback dataset download
docs/MILESTONES.md       plan
docs/DECISIONS.md        every design choice, with reasons and alternatives
```

## References

- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Wang et al., *Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks*, CVPR Workshops 2020.
- Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018.
- Petsiuk et al., *RISE: Randomized Input Sampling for Explanation of Black-box Models*, BMVC 2018 (deletion/insertion metrics).
- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018.
- Yang et al., *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification*, Scientific Data 2023.
- Acevedo et al., *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems*, Data in Brief 2020 (source of BloodMNIST).

## License

Code: [MIT](LICENSE). Data: BloodMNIST is distributed by MedMNIST under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); it is downloaded at run
time and not included in this repository.
