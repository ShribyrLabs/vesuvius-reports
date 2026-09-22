#!/usr/bin/env python3
"""Write an ink_9um fine-tune config from a checkpoint's embedded config plus the
pseudo-labelled segments (built by build_pseudo_labels.py / build_pseudo_0172.py).

One --dataset SCROLL=DIR per label root; each becomes its own dataset entry so the
scroll-balanced sampler draws from every scroll evenly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="SCROLL=DIR",
        help="label root per scroll (default: 0139=data/ink_pseudo)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config-out", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--exclude", nargs="*", default=[], help="segment names to leave out entirely")
    # Each loader worker keeps its own copy of per-segment caches (forkserver start, no sharing);
    # at 79 segments that is 4+ GB per worker, and 12 of them OOM-killed the desktop 3x.
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    for k in (
        "data_provenance",
        "fixed_scroll_prior",
        "patch_cache_filename",
        "volume_cache_dir",
        "volume_cache_max_gb",
        "sampling_audit_every",
    ):
        cfg.pop(k, None)

    roots = [d.split("=", 1) for d in (a.dataset or ["0139=data/ink_pseudo"])]
    cfg["datasets"], held, total = [], [], 0
    for scroll, root in roots:
        labels = Path(root)
        names = sorted(
            p.parent.name
            for p in labels.glob("*/build.json")
            if p.parent.name not in set(a.exclude)
        )
        if not names:
            raise SystemExit(f"no built segments under {labels}")
        cfg["datasets"].append(
            {
                "segments_path": str(labels.resolve()),
                "segments": names,
                "surface_volume_paths": {
                    n: str((labels / n / "surface-volume.zarr").resolve()) for n in names
                },
                "volume_scale": 0,
                "sampling_scroll": scroll,
                "sampling_physical_segment_keys": {n: f"{scroll}:{n}" for n in names},
                "sampling_representation_keys": {
                    n: f"native_9p362_level0_pseudo:{n}" for n in names
                },
            }
        )
        held += [
            f"{scroll}:{n}" for n in names if (labels / n / f"{n}_validation_mask.zarr").exists()
        ]
        total += len(names)
    cfg.update(
        {
            "checkpoint": str(a.checkpoint.resolve()),
            "weights_only": True,
            "sampling_strategy": "scroll_segment_balanced",
            "num_iterations": a.iterations,
            "learning_rate": a.lr,
            "warmup_steps": a.warmup,
            "batch_size": a.batch_size,
            "val_every": a.val_every,
            "save_every": a.save_every,
            "dataloader_workers": a.workers,
            "out_dir": str(a.out_dir.resolve()),
            "patch_cache_filename": str((a.out_dir / "patch_cache.pkl").resolve()),
            "description": "Fine-tune of ink_9um hybrid_3d2d on dense pseudo-labels at native 9.362um "
            f"({', '.join(s for s, _ in roots)}); held-out segments carry validation "
            "masks only.",
        }
    )
    a.config_out.parent.mkdir(parents=True, exist_ok=True)
    a.config_out.write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"{total} segments ({len(held)} validation-only: {held}) -> {a.config_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
