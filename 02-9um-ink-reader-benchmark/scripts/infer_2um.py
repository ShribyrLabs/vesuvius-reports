#!/usr/bin/env python3
"""Run the canonical 2 um ink model (scrollprize/ink_canonical_2um, ResNet3D-152 + 3D decoder,
recipe new_canon_autoresearch_recipe) on a local surface-volume zarr.

Re-implements the tiling of villa ink-detection/optimized_inference/inference.py (which needs
albumentations) without changing its maths: layers [start, end) of a (C, H, W) uint8 volume,
256 px tiles at the given stride, clip to [0, 200] and divide by 200, sigmoid, Hann-weighted
overlap-add over pixels where any layer is non-zero. The model code itself is imported from
villa unchanged. Output: uint8 TIFF (p * 255), same H, W as the input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VILLA_INF = ROOT / "upstream/villa/ink-detection/optimized_inference"


def grid_1d(n: int, tile: int, stride: int) -> list[int]:
    """Tile origins covering [0, n), last tile flush with the border (villa's _grid_1d)."""
    xs = list(range(0, max(1, n - tile + 1), stride))
    end = max(0, n - tile)
    if not xs or xs[-1] != end:
        xs.append(end)
    return xs


def hann2d(h: int, w: int) -> np.ndarray:
    k = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    return k / k.sum()


def main() -> int:
    import tifffile
    import torch
    import torch.nn.functional as F
    import zarr

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zarr", help="surface volume (group with '0', or array), shape (C, H, W)")
    ap.add_argument("checkpoint")
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=63)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reverse", action="store_true", help="reverse layer order")
    ap.add_argument(
        "--up",
        type=int,
        default=1,
        help="coarse input: read tile/UP px tiles and upsample them UP x in-plane to the model's "
        "2.4 um scale (4 for 9.362 um input); the prediction is written on the input grid",
    )
    ap.add_argument(
        "--model-layers",
        type=int,
        default=None,
        help="with --up, resample the layer window to this many layers (62 for the canonical model)",
    )
    ap.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    a = ap.parse_args()

    sys.path.insert(0, str(VILLA_INF))
    from model_resnet3d_3d_decoder import load_model

    root = zarr.open(a.zarr, mode="r")
    vol = root["0"] if isinstance(root, zarr.Group) else root
    c, h, w = vol.shape
    y0, y1, x0, x1 = a.crop if a.crop else (0, h, 0, w)
    stack = np.asarray(vol[a.start : a.end, y0:y1, x0:x1])  # (C, h, w) uint8
    if a.reverse:
        stack = stack[::-1]
    nc, hh, ww = stack.shape
    print(f"input {vol.shape} -> layers [{a.start},{a.end}) crop {hh}x{ww}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tin = a.tile // a.up  # tile size on the input grid
    nm = a.model_layers or nc
    model = load_model(a.checkpoint, dev, num_frames=nm)
    win = torch.from_numpy(hann2d(tin, tin)).to(dev)
    pred = np.zeros((hh, ww), np.float32)
    cnt = np.zeros((hh, ww), np.float32)
    tiles = [(ty, tx) for ty in grid_1d(hh, tin, a.stride) for tx in grid_1d(ww, tin, a.stride)]
    with torch.inference_mode():
        for i in range(0, len(tiles), a.batch):
            batch = tiles[i : i + a.batch]
            arr = np.zeros((len(batch), 1, nc, tin, tin), np.float32)
            valid = np.zeros((len(batch), tin, tin), np.float32)
            for j, (ty, tx) in enumerate(batch):
                t = stack[:, ty : ty + tin, tx : tx + tin]
                ph, pw = tin - t.shape[1], tin - t.shape[2]
                if ph or pw:
                    t = np.pad(t, ((0, 0), (0, ph), (0, pw)))
                valid[j] = (t != 0).any(0)
                arr[j, 0] = np.clip(t, 0, 200).astype(np.float32) / 200.0
            if not valid.any():
                continue
            x = torch.from_numpy(arr).to(dev)
            if a.up != 1 or nm != nc:
                x = F.interpolate(
                    x, size=(nm, a.tile, a.tile), mode="trilinear", align_corners=False
                )
            with torch.autocast(device_type=dev.type, enabled=True):
                y = model.forward(x)
            y = torch.sigmoid(y).float()
            y = F.interpolate(y, size=(tin, tin), mode="bilinear", align_corners=False)
            yw = (y[:, 0] * win).cpu().numpy()
            wn = win.cpu().numpy()
            for j, (ty, tx) in enumerate(batch):
                sh, sw = min(tin, hh - ty), min(tin, ww - tx)
                v = valid[j, :sh, :sw]
                pred[ty : ty + sh, tx : tx + sw] += yw[j, :sh, :sw] * v
                cnt[ty : ty + sh, tx : tx + sw] += wn[:sh, :sw] * v
            if (i // a.batch) % 50 == 0:
                print(f"  tiles {i + len(batch)}/{len(tiles)}", flush=True)
    p = np.divide(pred, cnt, out=np.zeros_like(pred), where=cnt > 0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(a.out, (np.clip(p, 0, 1) * 255).astype(np.uint8))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
