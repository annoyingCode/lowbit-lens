# Decision log

Every design choice, the reason for it and the alternatives considered. Status is
**agreed** (discussed and confirmed), **proposed** (in use, still open for
discussion) or **revised** (changed; the entry says what changed and why).

## D1. Research question and claims (agreed)

- **Question:** when a CNN is quantized, do its Grad-CAM explanations stay
  consistent with those of the FP32 model, and do they stay faithful to the
  model's own behaviour?
- **What we do not claim:** BloodMNIST has no ground-truth regions, and each image
  is mostly one centred cell. So we cannot say whether a model looks at the
  *correct* evidence, only whether quantized models look where the FP32 model
  looks (consistency) and whether the heatmaps track what drives the prediction
  (faithfulness, via deletion/insertion).
- **Novelty:** no claim of being first. A literature check comes before any
  framing in the README or blog post (scheduled in M4).
- *Alternative considered:* adding a dataset with segmentation masks to test
  "right evidence". Rejected for now (1–2 weeks of extra work); possible later
  extension.

## D2. Dataset resolution: 224 px (agreed)

Grad-CAM's map has the resolution of the last conv layer, which is input size ÷ 32:
2×2 at 64 px, 4×4 at 128 px, 7×7 at 224 px. Top-k% IoU and SSIM on a 4×4 grid
are too coarse to mean much. 224 px is also the ImageNet pretraining size and a
native MedMNIST+ size (no resampling by us).
- *Cost:* a 1.5 GB download, about 2.6 GB of RAM for all splits, slower training.
- *Alternatives:* 128 px (4×4 maps); 128 px upscaled to 224 (7×7 maps, but no
  real detail added and the same compute as 224).

## D3. Three training seeds per model (agreed)

Seeds 0, 1, 2. They give a spread (mean ± SD) for every result and a **noise
floor**: the heatmap similarity between two FP32 models that differ only in
their seed. A quantization effect is meaningful only relative to that floor.
- *Cost:* 3× training time.
- *Alternatives:* 1 seed (cannot separate quantization effects from retraining
  variation); 2 seeds (noise floor but no spread).

## D4. Training recipe: standard defaults, not tuned (agreed)

AdamW, peak learning rate 5e-4 with a one-cycle schedule (10% warm-up), weight
decay 1e-4, 15 epochs, batch 64, bfloat16 mixed precision for training speed
only. Augmentation: random rotations by 90° and flips (the 8 symmetries of the
square; cells have no canonical orientation). The checkpoint with the best
validation accuracy is kept, and the test set is evaluated exactly once.
- The same recipe is used for all three models and was not tuned. The study
  compares each model with its own quantized versions, so the baselines need to
  be solid, not state of the art. This is stated next to every accuracy table.
- *Alternative:* a small learning-rate search on the validation set.

## D5. Quantizable torchvision model variants (agreed)

`torchvision.models.quantization` versions of ResNet-18, MobileNetV2 and GoogLeNet,
loaded with the standard ImageNet weights. In FP32 they compute the same function
as the standard models, and they contain what real INT8 conversion needs (quant
stubs, fusable Conv+BN+ReLU blocks, quantization-safe residual adds). Fine-tuning
these variants means the exact same weights are later quantized.
- *Alternative:* standard models with FX graph-mode quantization, which inserts
  the hooks automatically. Rejected: less transparent, and FX mode is being
  phased out in favour of torchao.

## D6. Software versions (agreed)

PyTorch 2.14.1 + torchvision 0.29.1 (CUDA 13.0 builds), Python 3.12, managed with
uv. `uv.lock` pins every package; `requirements.txt` is exported from it for pip
users.
- Checked before pinning: eager-mode quantization (`torch.ao.quantization`) and
  `torchvision.models.quantization` both exist in these versions.
- Consequence: the Windows build offers only the `onednn` quantized engine. INT8
  CPU latency will be reported as "this laptop, onednn backend, N threads" and
  not as a general speed-up figure. Accuracy and heatmap results do not depend
  on the engine.
- *Alternative:* an older PyTorch whose Windows build has the x86/fbgemm engine.
  Rejected: unverified, and it would mean an older, soon-unsupported stack.

## D7. Reproducibility mechanics (agreed)

One TOML config; fixed seeds; cuDNN deterministic mode with
`torch.use_deterministic_algorithms(warn_only=True)` (a few CUDA kernels have no
deterministic version, so GPU runs are repeatable up to those ops); results
tables are generated only from run outputs, never typed by hand;
`python -m lowbit_lens.reproduce` reruns everything.
- Steps after training (quantization, explanations, metrics) run in eval mode or
  on CPU and are to be checked for bit-exact repeatability when they are built.
- *Alternative:* strict determinism, which errors on any non-deterministic op.
  Rejected: may force model changes and slower kernels, and the variation it
  removes is far smaller than the seed-to-seed variation already measured (D3).

## D8. Data source and integrity (agreed)

BloodMNIST 224 px from the official MedMNIST+ Zenodo record (10519652), checked
against the MD5 published in the `medmnist` package. `scripts/download_data.sh`
exists because Zenodo was very slow per connection during setup; it uses parallel
resumable ranges of the same official file.
- Default path: `medmnist` downloads on first run and verifies the same MD5
  (through torchvision's `download_url`). The script is a documented fallback;
  both paths end with a checksum-verified official file.

## D9. License and authorship (agreed)

Code: MIT license, author Abdul Wahab Khan. Data: BloodMNIST is CC BY 4.0. The
README credits MedMNIST v2 and the original dataset (Acevedo et al., 2020), and
the data is never committed to the repository (`data/` is gitignored).
- *Alternatives:* Apache-2.0 (adds a patent grant; more formal); no license
  (blocks reuse; unusual for research code).

## D10. Duplicate images in the official split (agreed)

An audit (`python -m lowbit_lens.audit`) found 17 groups of byte-identical images
in BloodMNIST 224 px: 8 within train, 5 train/test, 3 train/val and 1 val/test.
One train/test pair has conflicting labels (class 1 in train, class 6 in test).
The full list is in `results/tables/duplicate_images.csv`.
- **Decision:** keep the official split so results stay comparable with other
  work, and also report `test_acc_clean`, which leaves out the 6 test images that
  are copies of training or validation images. The same 6 images are left out of
  the heatmap analysis (M3–M4), so no conclusion rests on a copied image.
- **What the clean score showed:** all 9 FP32 models classify the 5 consistently
  labelled copies correctly, and predict the training label (class 1) for the
  conflicting image. The clean score is therefore slightly higher (+0.03
  percentage points in every run), so the copies did not inflate the results.
- **Limitation:** only exact copies are detected, not near-duplicates.
- *Alternatives:* note it and change nothing (no clean score); remove the copies
  from the dataset (cleanest, but not comparable with published numbers).

## D11. INT8 calibration set: 512 training images (agreed)

Activation ranges are measured on 512 training images, drawn at random with a
fixed seed (`quant.calib_seed`) so every model is calibrated on the same images.
Test images are never used for calibration. Images are not augmented.
- *Alternatives:* all ~12,000 training images (slower, usually barely different;
  possible later sensitivity check); 128 images (risk of an unrepresentative range).

## D12. Rounding recipe: PyTorch's standard onednn recipe (agreed)

Eager-mode post-training static quantization: fuse Conv+BN(+ReLU), attach the
default `onednn` observers, calibrate, convert. As shipped in PyTorch 2.14.1:
weights are per-channel symmetric INT8 (`PerChannelMinMaxObserver`); activations
are per-tensor 8-bit with a range picked from a histogram (`HistogramObserver`,
no reduced range). The point is to measure what practitioners actually get.
- *Alternative:* a simplest-possible recipe (per-layer weights, min/max
  ranges): easier to explain but worse than anything used in practice.
- FP32 and INT8 are compared on the **CPU**, so quantization is the only
  difference. FP32 GPU-vs-CPU prediction agreement is recorded as a check.

## D13. CPU speed measurement (revised, agreed)

**Final method:** batch size 1 (one image at a time, like an app) on **1 thread**.
Each of 5 rounds times every model in FP32 and INT8, alternating which goes
first; a round's value is the median of 100 runs after 10 warm-up runs. Reported:
the median over rounds, the lowest and highest round, and the median per-round
speed-up. Labelled "this laptop, onednn engine, PyTorch 2.14.1" only.

**Why it was revised.** The first plan (1 thread and all threads, 200 runs,
one pass) gave numbers that did not hold up:
- ResNet-18 in INT8 with 2, 4 or 8 threads was several times *slower* than on
  1 thread (repeated over rounds), while its FP32 version sped up normally with
  more threads. GoogLeNet in INT8 showed this only mildly and MobileNetV2 not at all. It looks
  like a threading problem in this PyTorch build on Windows, not a property of
  INT8. It is reported per model in a separate table as a limitation
  (`cpu_latency_multithread`), never as a speed result.
- FP32 single-thread time differed between sessions (about 79 vs 122 ms for
  ResNet-18). The CPU (Intel i5-13420H) mixes fast and slow cores, and Windows
  decides which one runs the job. Alternating rounds make that drift visible
  in the reported range instead of hiding it in one number.
- *Alternatives considered:* spend time fixing multi-threaded INT8 (uncertain,
  and speed is not the project's main question); drop speed entirely (loses a
  practical, honestly measurable result).

## D14. Which models are quantized and timed (agreed)

All 9 models (3 architectures x 3 seeds) are quantized and evaluated, which the
noise-floor comparison needs. Speed is measured for the seed-0 model of each
architecture only, because all seeds of an architecture have the same structure
and therefore the same amount of computation.
