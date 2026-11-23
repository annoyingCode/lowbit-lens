"""Save per-image test predictions for finished runs that do not have them yet.

    python -m lowbit_lens.predict

Runs trained after this was added already write test_logits.npy themselves.
For each checkpoint this recomputes the test set and checks that the accuracy
matches the number recorded at training time, so stored results and
checkpoints cannot silently disagree.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .config import DEFAULT_CONFIG, load_config
from .data import load_split
from .models import load_checkpoint
from .train import evaluate, run_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    todo = [
        run_dir(cfg, m, s)
        for s in cfg.seeds for m in cfg.train.models
        if not (run_dir(cfg, m, s) / "test_logits.npy").exists()
    ]
    if not todo:
        print("all runs already have test_logits.npy")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = cfg.data
    test = load_split(d.dataset, "test", d.size, cfg.paths.data_dir)
    for out in todo:
        model, _ = load_checkpoint(out / "best.pt", map_location=device)
        t = evaluate(model.to(device), test, cfg.train.eval_batch_size, device, d.num_classes)
        recorded = json.loads((out / "metrics.json").read_text())["test_acc"]
        if round(t["acc"] * len(test)) != round(recorded * len(test)):  # compare counts
            raise RuntimeError(f"{out.name}: recomputed test acc {t['acc']} != recorded {recorded}")
        np.save(out / "test_logits.npy", t["logits"].numpy())
        print(f"{out.name}: test acc {t['acc']:.4f} matches metrics.json; saved test_logits.npy")


if __name__ == "__main__":
    main()
