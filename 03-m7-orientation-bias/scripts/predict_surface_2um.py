#!/usr/bin/env python3
"""Run staff's published surface model (scrollprize/surface_recto_320) on a local CT crop.

The published 2 µm surface maps (e.g. PHercParis4 `…surface-recto-2um-ps256…`) come from this
nnU-Net ResidualEncoderUNet. No scroll we care about has one at 2.4 µm, so we make our own: the
architecture is rebuilt from the checkpoint's own `plans.json` and the weights loaded from
`network_weights`, which avoids nnU-Net's environment-variable setup entirely.

Conventions taken from the checkpoint, not assumed: 256³ patches, Z-score normalisation per
patch, one input channel, two labels (background / fiber) — we write the fiber probability.

usage: predict_surface_2um.py CROP_ZARR OUT_ZARR [--checkpoint DIR] [--weights-file F] [--tile 256] [--overlap 64]

The normalisation is read from the plans (ZScore for surface_recto_320, CTNormalization for
surface_m7_nnunet: `--checkpoint data/checkpoints/surface_m7 --weights-file fold_0/checkpoint_best.pth
--tile 192`). CROP_ZARR may be a level-0 array URL on the open-data bucket.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "data/checkpoints/surface_2um"


def build_network(plans: dict, config: str = "3d_fullres", num_classes: int = 2):
    """Instantiate the exact architecture the checkpoint was trained with."""
    import importlib

    import torch.nn as nn  # noqa: F401  (referenced by the kwargs below)

    arch = plans["configurations"][config]["architecture"]
    mod_name, cls_name = arch["network_class_name"].rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    kwargs = dict(arch["arch_kwargs"])
    for key in arch.get("_kw_requires_import", []):
        val = kwargs.get(key)
        if isinstance(val, str):
            m, c = val.rsplit(".", 1)
            kwargs[key] = getattr(importlib.import_module(m), c)
    return cls(input_channels=1, num_classes=num_classes, deep_supervision=False, **kwargs)


def tile_starts(n: int, tile: int, overlap: int) -> list[int]:
    """Tile origins covering [0, n) with the given overlap; the last tile is flush to the end."""
    if n <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, n - tile + 1, step))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


def ct_normalize(x: np.ndarray, props: dict) -> np.ndarray:
    """nnU-Net CTNormalization: clip to the dataset's 0.5/99.5 percentiles, then dataset mean/std."""
    x = np.clip(x.astype(np.float32), props["percentile_00_5"], props["percentile_99_5"])
    return (x - props["mean"]) / max(props["std"], 1e-8)


def zscore(x: np.ndarray) -> np.ndarray:
    """nnU-Net ZScoreNormalization, per patch."""
    x = x.astype(np.float32)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-8 else 1.0)


def main() -> int:
    import torch
    import zarr
    from numcodecs import Blosc

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("crop", help="local CT zarr (array or group with '0')")
    ap.add_argument("out", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument(
        "--weights-file", default="checkpoint_final.pth", help="relative to --checkpoint"
    )
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--bbox", type=int, nargs=6, metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"))
    ap.add_argument(
        "--full-shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="write into a full-volume-shaped store at the bbox offset, appending to any existing "
        "one. predict3d --pred-dt validates that the map is a pyramid LEVEL of the volume, so a "
        "bare crop is rejected; this makes a sparse full-shape store instead.",
    )
    a = ap.parse_args()

    plans = json.loads((a.checkpoint / "plans.json").read_text())
    ck = torch.load(a.checkpoint / a.weights_file, map_location="cpu", weights_only=False)
    cfg = plans["configurations"]["3d_fullres"]
    norm_scheme = cfg["normalization_schemes"][0]
    props = plans.get("foreground_intensity_properties_per_channel", {}).get("0")
    if norm_scheme == "CTNormalization":
        normalize = lambda p: ct_normalize(p, props)  # noqa: E731
    elif norm_scheme == "ZScoreNormalization":
        normalize = zscore
    else:
        raise SystemExit(f"unsupported normalization {norm_scheme}")
    print(f"normalization {norm_scheme}, patch {cfg['patch_size']}", flush=True)
    net = build_network(plans)
    net.load_state_dict(ck["network_weights"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(dev).eval()

    if str(a.crop).startswith("http"):
        src = zarr.open_array(str(a.crop), mode="r", zarr_format=2)
    else:
        src = zarr.open(str(a.crop), mode="r")
    src = src["0"] if hasattr(src, "keys") and "0" in list(src.keys()) else src
    z0, z1, y0, y1, x0, x1 = (
        a.bbox if a.bbox else (0, src.shape[0], 0, src.shape[1], 0, src.shape[2])
    )
    vol = np.asarray(src[z0:z1, y0:y1, x0:x1])
    print(f"crop {vol.shape} nonzero {float((vol > 0).mean()):.3f}", flush=True)

    # Accumulate in z-slabs and flush each one: holding float32 acc+wgt for the whole crop needs
    # ~8 bytes/voxel on top of the uint8 volume, which OOM-killed a 3.7 G-voxel run at a 40 GB cap
    # even though every tile had already been computed.
    t = a.tile
    # a cosine window keeps tile seams from showing in the probability map
    w1 = np.hanning(t + 2)[1:-1].astype(np.float32)
    win = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]

    zs = tile_starts(vol.shape[0], t, a.overlap)
    ys = tile_starts(vol.shape[1], t, a.overlap)
    xs = tile_starts(vol.shape[2], t, a.overlap)
    total = len(zs) * len(ys) * len(xs)

    store = zarr.storage.LocalStore(str(a.out))
    if a.full_shape and (a.out / "0" / ".zarray").exists():
        root = zarr.open_group(store=store, mode="a", zarr_format=2)
        arr = root["0"]
        print(f"appending into existing {a.out} {arr.shape}", flush=True)
    else:
        root = zarr.open_group(store=store, mode="w", zarr_format=2)
        arr = root.create_array(
            "0",
            shape=tuple(a.full_shape) if a.full_shape else vol.shape,
            chunks=(128, 128, 128),
            dtype="u1",
            compressors=Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE),
        )
    # where this crop's rows land in the output store
    oz, oy, ox = (z0, y0, x0) if a.full_shape else (0, 0, 0)

    done = 0
    lit = 0
    with torch.no_grad():
        for zi in zs:
            sz = min(t, vol.shape[0] - zi)
            acc = np.zeros((sz, vol.shape[1], vol.shape[2]), np.float32)
            wgt = np.zeros_like(acc)
            for yi in ys:
                for xi in xs:
                    patch = vol[zi : zi + t, yi : yi + t, xi : xi + t]
                    pad = [(0, t - s) for s in patch.shape]
                    if any(p[1] for p in pad):
                        patch = np.pad(patch, pad)
                    inp = torch.from_numpy(normalize(patch))[None, None].to(dev)
                    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                        out = net(inp)
                    prob = torch.softmax(out.float(), dim=1)[0, 1].cpu().numpy()
                    sy = min(t, vol.shape[1] - yi)
                    sx = min(t, vol.shape[2] - xi)
                    acc[:sz, yi : yi + sy, xi : xi + sx] += (prob * win)[:sz, :sy, :sx]
                    wgt[:sz, yi : yi + sy, xi : xi + sx] += win[:sz, :sy, :sx]
                    done += 1
                    if done % 20 == 0 or done == total:
                        print(f"  {done}/{total} tiles", flush=True)
            # in place and in y-blocks: whole-slab float64 temporaries peaked near 58 G on a
            # 640x4360x2280 band and were OOM-killed at a 40 G cap
            np.maximum(wgt, 1e-6, out=wgt)
            np.divide(acc, wgt, out=acc)
            del wgt
            # tiles overlap in z too, so only the rows this slab owns outright are final
            lo = zi if zi == zs[0] else zi + a.overlap // 2
            hi = zi + sz
            for yb in range(0, vol.shape[1], 512):
                part = acc[lo - zi : hi - zi, yb : yb + 512]
                arr[
                    oz + lo : oz + hi, oy + yb : oy + yb + part.shape[1], ox : ox + vol.shape[2]
                ] = np.clip(part * np.float32(255.0), 0, 255).astype(np.uint8)
                lit += int((part > 0.5).sum())
            del acc

    root.attrs["source"] = str(a.crop)
    root.attrs["bbox_zyx"] = [z0, z1, y0, y1, x0, x1]
    print(f"wrote {a.out}  surface>0.5 in {lit / vol.size:.3f} of voxels", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
