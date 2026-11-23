"""Fast checks that run on CPU without downloading data or weights."""

import pytest
import torch

from lowbit_lens.config import load_config
from lowbit_lens.data import Split, iterate_batches, random_dihedral
from lowbit_lens.models import MODEL_NAMES, build_model


def test_config_loads():
    cfg = load_config()
    assert cfg.data.num_classes == 8
    assert set(cfg.train.models) <= set(MODEL_NAMES)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_output_shape(name):
    model = build_model(name, num_classes=8, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 8)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_state_dict_roundtrip(name):
    a = build_model(name, num_classes=8, pretrained=False)
    b = build_model(name, num_classes=8, pretrained=False)
    b.load_state_dict(a.state_dict())  # strict: architectures must match exactly


def test_dihedral_only_permutes_pixels():
    x = torch.rand(16, 3, 8, 8)
    y = random_dihedral(x, torch.Generator().manual_seed(0))
    for a, b in zip(x, y):
        assert torch.equal(a.flatten().sort().values, b.flatten().sort().values)


def test_batches_are_reproducible():
    split = Split(torch.randint(0, 256, (10, 3, 4, 4), dtype=torch.uint8), torch.arange(10))

    def run():
        g = torch.Generator().manual_seed(1)
        return [y for _, y in iterate_batches(split, 4, torch.device("cpu"), shuffle=True, generator=g)]

    assert all(torch.equal(a, b) for a, b in zip(run(), run()))


def test_duplicate_audit_finds_cross_split_copies():
    from lowbit_lens.audit import excluded_test_indices, find_duplicate_groups

    a, b, c = (torch.full((3, 2, 2), v, dtype=torch.uint8) for v in (1, 2, 3))
    splits = {
        "train": Split(torch.stack([a, b, b]), torch.tensor([0, 1, 1])),
        "val": Split(torch.stack([c]), torch.tensor([2])),
        "test": Split(torch.stack([c, a, torch.zeros_like(a)]), torch.tensor([2, 5, 0])),
    }
    groups = find_duplicate_groups(splits)
    assert len(groups) == 3  # a: train/test, b: within train, c: val/test
    assert excluded_test_indices(groups) == [0, 1]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_int8_quantization_runs_and_repeats(name):
    from lowbit_lens.quantize import int8_skeleton, quantize_int8

    torch.manual_seed(0)
    model = build_model(name, num_classes=8, pretrained=False).eval()
    calib = Split(torch.randint(0, 256, (8, 3, 64, 64), dtype=torch.uint8), torch.zeros(8, dtype=torch.long))
    x = torch.randn(2, 3, 64, 64)
    a = quantize_int8(model, calib, "onednn", 4)
    b = quantize_int8(model, calib, "onednn", 4)
    with torch.no_grad():
        assert a(x).shape == (2, 8)
        assert torch.equal(a(x), b(x))  # calibration + conversion are deterministic
    skeleton = int8_skeleton(name, 8, "onednn")
    skeleton.load_state_dict(a.state_dict())  # saved INT8 weights load into the skeleton
    with torch.no_grad():
        assert torch.equal(skeleton(x), a(x))


def test_levels_aware_observer_uses_real_number_of_levels():
    from lowbit_lens.fakequant import LevelsAwareHistogramObserver

    assert LevelsAwareHistogramObserver(quant_min=0, quant_max=15).dst_nbins == 16
    assert LevelsAwareHistogramObserver().dst_nbins == 256  # 8 bits: same as PyTorch's default


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fake_quant_grid_and_gradients(name):
    from lowbit_lens.fakequant import EDGE_LAYERS, build_fake_quant

    torch.manual_seed(0)
    model = build_model(name, num_classes=8, pretrained=False).eval()
    calib = Split(torch.randint(0, 256, (8, 3, 64, 64), dtype=torch.uint8), torch.zeros(8, dtype=torch.long))
    fq = build_fake_quant(model, name, calib, weight_bits=4, act_bits=4, batch_size=4)
    for layer in EDGE_LAYERS[name]:
        fq.get_submodule(layer)  # edge layers exist under these names
    # a 4-bit layer's weights sit on at most 16 levels per output channel
    convs = [m for n, m in fq.named_modules() if hasattr(m, "weight_fake_quant") and n not in EDGE_LAYERS[name]]
    w = convs[len(convs) // 2]
    wq = w.weight_fake_quant(w.weight).detach()
    assert max(len(torch.unique(wq[c])) for c in range(wq.shape[0])) <= 16
    # Gradients reach the input (needed for Grad-CAM). The input must look like the
    # calibration images, since values clipped at the grid's edge pass no gradient.
    # A randomly initialised MobileNetV2 has a vanishing gradient even in FP32
    # (~1e-8), which rounding then removes, so it is skipped here; the trained
    # MobileNetV2 checkpoints do have gradients at 8, 4 and 2 bits.
    if name != "mobilenet_v2":
        x, _ = next(iterate_batches(calib, 1, torch.device("cpu")))
        x.requires_grad_(True)
        fq(x)[0].max().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0
