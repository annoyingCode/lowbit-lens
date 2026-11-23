"""Data audit: find images that appear more than once in the dataset.

    python -m lowbit_lens.audit

The official BloodMNIST release contains a few exact duplicates, some of them
across splits: a test image that is also a training image is not really
"unseen". We keep the official split (so results stay comparable with other
work) and additionally report a *clean* test score that leaves out every test
image that also appears in the training or validation split. The same images
are left out of the later heatmap analysis (see docs/DECISIONS.md, D10).

"Exact duplicate" means byte-for-byte identical pixels. Near-duplicates (the
same cell slightly shifted or re-compressed) are not detected by this check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, Config, load_config
from .data import Split, load_split

SPLITS = ("train", "val", "test")


def image_hashes(split: Split) -> list[str]:
    return [hashlib.sha1(img.numpy().tobytes()).hexdigest() for img in split.images]


def find_duplicate_groups(splits: dict[str, Split]) -> list[list[dict]]:
    """Groups of identical images: [[{split, index, label}, ...], ...]."""
    by_hash: dict[str, list[dict]] = {}
    for name, split in splits.items():
        for i, h in enumerate(image_hashes(split)):
            by_hash.setdefault(h, []).append(
                {"split": name, "index": i, "label": int(split.labels[i])}
            )
    return [g for g in by_hash.values() if len(g) > 1]


def excluded_test_indices(groups: list[list[dict]]) -> list[int]:
    """Test images that also appear in the train or validation split."""
    out = set()
    for g in groups:
        if any(m["split"] in ("train", "val") for m in g):
            out.update(m["index"] for m in g if m["split"] == "test")
    return sorted(out)


def audit_path(cfg: Config) -> Path:
    return cfg.paths.results_dir / "tables" / "duplicate_images.json"


def run_audit(cfg: Config) -> dict:
    d = cfg.data
    splits = {s: load_split(d.dataset, s, d.size, cfg.paths.data_dir) for s in SPLITS}
    groups = find_duplicate_groups(splits)
    excluded = excluded_test_indices(groups)
    result = {
        "dataset": d.dataset,
        "size": d.size,
        "split_sizes": {s: len(splits[s]) for s in SPLITS},
        "duplicate_groups": groups,
        "label_conflicts": [g for g in groups if len({m["label"] for m in g}) > 1],
        "excluded_test_indices": excluded,
    }
    out = audit_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "split", "index", "label"])
        for gi, g in enumerate(groups):
            for m in g:
                w.writerow([gi, m["split"], m["index"], m["label"]])

    pairs: dict[str, int] = {}
    for g in groups:
        names = sorted({m["split"] for m in g}, key=SPLITS.index)
        key = "/".join(names) if len(names) > 1 else f"within {names[0]}"
        pairs[key] = pairs.get(key, 0) + 1
    print(f"{len(groups)} groups of identical images: {pairs}")
    print(f"{len(result['label_conflicts'])} group(s) with conflicting labels")
    print(f"clean test set leaves out {len(excluded)} of {len(splits['test'])} test images")
    print(f"wrote {out} and .csv")
    return result


def load_excluded_test_indices(cfg: Config) -> list[int]:
    path = audit_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python -m lowbit_lens.audit")
    return json.loads(path.read_text())["excluded_test_indices"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = p.parse_args(argv)
    run_audit(load_config(args.config))


if __name__ == "__main__":
    main()
