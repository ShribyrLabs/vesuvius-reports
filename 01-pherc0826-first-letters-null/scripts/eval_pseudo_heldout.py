#!/usr/bin/env python3
"""Score ink predictions on a pseudo-labelled segment against its projected 2.4 um key.

Reports Pearson r (prediction vs key, over valid papyrus), precision/recall at p>0.5
against key>=64, and optional seed-to-seed agreement. Writes a ds4 side-by-side PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_pred(path: Path, shape) -> np.ndarray:
    p = tifffile.imread(path).astype(np.float32)
    if p.shape != tuple(shape):
        p = p[: shape[0], : shape[1]]
    return p


def score(pred: np.ndarray, key: np.ndarray, valid: np.ndarray) -> dict:
    a, b = pred[valid], key[valid]
    r = float(np.corrcoef(a, b)[0, 1])
    ink = b >= 64
    pos = a > 128
    tp = float((pos & ink).sum())
    return {
        "r": r,
        "precision": tp / max(pos.sum(), 1),
        "recall": tp / max(ink.sum(), 1),
        "pos_frac": float(pos.mean()),
        "key_ink_frac": float(ink.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("segment_dir", type=Path, help="data/ink_pseudo/<name>")
    ap.add_argument(
        "--pred", nargs="+", type=Path, required=True, help="prediction TIFFs (name=path)"
    )
    ap.add_argument("--png", type=Path)
    ap.add_argument("--mask", type=Path, help="bool .npy; score only where it is True")
    a = ap.parse_args()

    key = np.load(a.segment_dir / "key_native.npy").astype(np.float32)
    valid = key >= 0
    if a.mask:
        valid &= np.load(a.mask)
    preds = {}
    for item in a.pred:
        name, path = str(item).split("=", 1)
        preds[name] = load_pred(Path(path), key.shape)
    out = {n: score(p, key, valid) for n, p in preds.items()}
    names = list(preds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out[f"agree({names[i]},{names[j]})"] = float(
                np.corrcoef(preds[names[i]][valid], preds[names[j]][valid])[0, 1]
            )
    print(json.dumps(out, indent=1))
    if a.png:
        tiles = [np.clip(key, 0, 255)] + [preds[n] for n in names]
        ds = [Image.fromarray(t.astype(np.uint8)).reduce(4) for t in tiles]
        w, h = ds[0].size
        canvas = Image.new("L", (w * len(ds) + 8 * (len(ds) - 1), h), 128)
        for i, im in enumerate(ds):
            canvas.paste(im, (i * (w + 8), 0))
        canvas.save(a.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
