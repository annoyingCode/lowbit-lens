"""Regenerate every result in the repo with one command.

    python -m lowbit_lens.reproduce            # reuse finished runs, fill in missing ones
    python -m lowbit_lens.reproduce --force    # retrain everything from scratch

Stages so far (more are added as milestones land, see docs/MILESTONES.md):
    1. fine-tune FP32 baselines (every model x seed) -> runs/fp32/
    2. data audit: duplicate images  -> results/tables/duplicate_images.{json,csv}
    3. per-image test predictions    -> runs/fp32/*/test_logits.npy (if missing)
    4. real INT8 quantization + CPU timing -> runs/int8/  (milestone M1)
    5. tables                        -> results/tables/*.{csv,md}
"""

from __future__ import annotations

import argparse

from . import audit, predict, quantize, report, train
from .config import DEFAULT_CONFIG, load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--force", action="store_true", help="retrain even if a finished run exists")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    todo = [
        (m, s) for s in cfg.seeds for m in cfg.train.models
        if args.force or not (train.run_dir(cfg, m, s) / "metrics.json").exists()
    ]
    if todo:
        print("[1/5] training: " + ", ".join(f"{m} seed {s}" for m, s in todo))
        for s in dict.fromkeys(s for _, s in todo):  # one call per seed, in order
            models = [m for m, s2 in todo if s2 == s]
            train.main(["--config", args.config, "--models", *models, "--seeds", str(s)])
    else:
        print("[1/5] training: all FP32 runs present, skipping (use --force to retrain)")

    print("[2/5] data audit")
    audit.main(["--config", args.config])
    print("[3/5] test predictions")
    predict.main(["--config", args.config])
    print("[4/5] INT8 quantization and CPU timing")
    quantize.main(["--config", args.config] + (["--force"] if args.force else []))
    print("[5/5] tables")
    report.main(["--config", args.config])


if __name__ == "__main__":
    main()
