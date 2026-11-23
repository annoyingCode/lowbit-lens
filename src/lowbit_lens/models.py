"""The three CNNs under study, built so the same weights can later be quantized.

Quantization in one paragraph: a trained network stores each weight and
activation as a 32-bit float (FP32). Quantization maps those values onto a small
grid of integers, for example the 256 levels of INT8, plus a scale (and zero
point) per tensor or per channel that converts back: real ~= scale * (int - zero_point).
The model gets about 4x smaller and integer arithmetic is faster on CPUs, at the
cost of rounding error. This project asks whether that rounding error changes
*where the model looks*, even when it barely changes accuracy.

Why `torchvision.models.quantization`: real INT8 conversion in PyTorch's eager
mode needs a few things in the model definition, and these variants already
have them:
  - `QuantStub` / `DeQuantStub` mark where float input becomes INT8 and back;
  - residual additions use `FloatFunctional`, because adding two INT8 tensors
    with different scales needs a quantization-aware add;
  - `fuse_model()` merges Conv+BatchNorm(+ReLU) into one op before quantizing.
In FP32 mode the stubs do nothing, so these models compute exactly what the
standard torchvision models compute and load the same ImageNet weights.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import GoogLeNet_Weights, MobileNet_V2_Weights, ResNet18_Weights
from torchvision.models import quantization as qmodels

MODEL_NAMES = ("resnet18", "mobilenet_v2", "googlenet")


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """A quantizable FP32 model with a fresh `num_classes`-way classifier head."""
    if name == "resnet18":
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = qmodels.resnet18(weights=w, quantize=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenet_v2":
        w = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = qmodels.mobilenet_v2(weights=w, quantize=False)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif name == "googlenet":
        # With pretrained weights torchvision drops the auxiliary classifiers and
        # sets transform_input=True (re-normalises ImageNet-normalised input the
        # way the original GoogLeNet weights expect). Without weights we ask for
        # the same architecture explicitly so checkpoints load into it.
        if pretrained:
            m = qmodels.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1, quantize=False)
        else:
            m = qmodels.googlenet(
                weights=None, quantize=False, aux_logits=False, transform_input=True,
                init_weights=False,
            )
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return m


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path, map_location="cpu") -> tuple[nn.Module, dict]:
    """Rebuild a fine-tuned model from a checkpoint written by `train.py`."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
