"""Load configs/*.toml into typed, read-only dataclasses."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.toml"


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    runs_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class DataCfg:
    dataset: str
    size: int
    num_classes: int


@dataclass(frozen=True)
class TrainCfg:
    models: tuple[str, ...]
    epochs: int
    batch_size: int
    eval_batch_size: int
    lr: float
    weight_decay: float
    label_smoothing: float
    amp_dtype: str


@dataclass(frozen=True)
class QuantCfg:
    backend: str
    calib_images: int
    calib_seed: int
    batch_size: int


@dataclass(frozen=True)
class LatencyCfg:
    runs: int
    warmup: int
    rounds: int
    seeds: tuple[int, ...]
    diagnostic_threads: tuple[int, ...]
    diagnostic_rounds: int
    diagnostic_runs: int


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...]
    paths: Paths
    data: DataCfg
    train: TrainCfg
    quant: QuantCfg
    latency: LatencyCfg

    def to_dict(self) -> dict:
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        return d


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Relative paths are relative to the project root, not the current directory.
    paths = Paths(**{k: (PROJECT_ROOT / v).resolve() for k, v in raw["paths"].items()})
    data = DataCfg(**raw["data"])
    train = TrainCfg(**{**raw["train"], "models": tuple(raw["train"]["models"])})

    if data.size not in (28, 64, 128, 224):
        raise ValueError(f"data.size must be 28, 64, 128 or 224, got {data.size}")
    if train.amp_dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"unknown train.amp_dtype {train.amp_dtype!r}")
    seeds = tuple(raw["seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be a non-empty list of distinct ints, got {seeds}")
    quant = QuantCfg(**raw["quant"])
    lat = raw["latency"]
    latency = LatencyCfg(
        runs=lat["runs"], warmup=lat["warmup"], rounds=lat["rounds"],
        seeds=tuple(lat["seeds"]), diagnostic_threads=tuple(lat["diagnostic_threads"]),
        diagnostic_rounds=lat["diagnostic_rounds"], diagnostic_runs=lat["diagnostic_runs"],
    )
    return Config(seeds=seeds, paths=paths, data=data, train=train, quant=quant, latency=latency)
