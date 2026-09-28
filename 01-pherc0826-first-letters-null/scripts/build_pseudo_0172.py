#!/usr/bin/env python3
"""Build ink_9um-style pseudo-labels for one PHerc0172 segment from its native-resolution maps.

PHerc0172 (Scroll 5) segments ship a 7.91 um surface volume [33,H,W] and two timesformer
ink maps on the SAME canvas (july_retreat, november19), so no cross-scan registration is
needed. Everything is resampled to 9.362 um voxels (in-plane 7.91/9.362, depth 33 -> 28,
which keeps the 261 um slab thickness) so the data matches the 0139/0500P2 sets.

Labels: ink where BOTH maps >= --ink-thr, background where BOTH < --bg-thr; anything the two
maps disagree on is left unsupervised. Papyrus must be present (native mid-planes > 0).

Output layout (same as build_pseudo_labels.py):
  <out>/<name>/surface-volume.zarr/0, <name>_inklabels.zarr/0, <name>_supervision_mask.zarr/0,
  <name>_validation_mask.zarr/0 (with --held-out), key_native.npy (mean of both maps),
  build.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import s3fs
import tifffile
import zarr
from numcodecs import Blosc
from scipy.ndimage import map_coordinates

BUCKET = "vesuvius-challenge-open-data"
SCROLL = "PHerc0172"
NATIVE_VOL = "7.91um-53keV-volume-20241024131838"
SRC_UM, DST_UM = 7.91, 9.362
DST_DEPTH = 28
ROOT = Path(__file__).resolve().parents[1]
FETCHER = ROOT / "scripts" / "fetch_control_crop.py"
sys.path.insert(0, str(ROOT / "scripts"))
from build_pseudo_labels import write_label_zarr  # noqa: E402


def resample_plane(plane: np.ndarray, H2: int, W2: int, y2: int = 0, y0: int = 0) -> np.ndarray:
    """Sample a 7.91 um plane at 9.362 um pixel centres: output j -> source j * 9.362/7.91.

    plane is source rows [y0:], the result is output rows [y2:y2+H2] and columns [0:W2].
    """
    s = SRC_UM / DST_UM
    ys = (np.arange(y2, y2 + H2, dtype=np.float32) / s) - y0
    xs = np.arange(W2, dtype=np.float32) / s
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    return map_coordinates(plane, [Y, X], order=1, mode="constant", cval=0.0)


def resample_volume(src: zarr.Array, out: Path, band: int = 1024) -> zarr.Array:
    """[33,H,W] at 7.91 um -> [28,H',W'] at 9.362 um, processed in y-bands (src is ~6 GB).

    Depth: output plane k sits at k*(D-1)/(DST_DEPTH-1) in source planes (both slabs span
    ~253 um), linear between the two neighbours. In-plane: resample_plane (same rule as the
    maps, so labels and voxels stay on one grid).
    """
    D, H, W = src.shape
    s = SRC_UM / DST_UM
    H2, W2 = int(np.floor((H - 1) * s)) + 1, int(np.floor((W - 1) * s)) + 1
    g = zarr.open_group(str(out), mode="w", zarr_format=2)
    dst = g.create_array(
        "0",
        shape=(DST_DEPTH, H2, W2),
        chunks=(DST_DEPTH, 128, 128),
        dtype="u1",
        compressors=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        config={"write_empty_chunks": False},
    )
    zsrc = np.arange(DST_DEPTH) * (D - 1) / (DST_DEPTH - 1)
    for y2 in range(0, H2, band):
        y2e = min(y2 + band, H2)
        y0 = max(int(np.floor(y2 / s)) - 2, 0)
        y1 = min(int(np.ceil((y2e - 1) / s)) + 3, H)
        blk = src[:, y0:y1, :].astype(np.float32)
        if not blk.any():
            continue
        outb = np.zeros((DST_DEPTH, y2e - y2, W2), np.float32)
        for k, zf in enumerate(zsrc):
            z0 = min(int(np.floor(zf)), D - 2)
            t = zf - z0
            plane = blk[z0] * (1 - t) + blk[z0 + 1] * t if t > 1e-6 else blk[z0]
            outb[k] = resample_plane(plane, y2e - y2, W2, y2, y0)
        dst[:, y2:y2e, :] = np.clip(np.rint(outb), 0, 255).astype(np.uint8)
        print(f"  band {y2}/{H2}", flush=True)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("segment", help="S3 segment directory name (…_flatboi)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ink_pseudo_0172")
    ap.add_argument("--ink-thr", type=float, default=128)
    ap.add_argument("--bg-thr", type=float, default=64)
    ap.add_argument("--held-out", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument(
        "--keep-native", action="store_true", help="keep the 7.91 um zarr after resampling"
    )
    a = ap.parse_args()

    d = a.out / a.name
    d.mkdir(parents=True, exist_ok=True)
    seg = f"{SCROLL}/segments/{a.segment}"
    fs = s3fs.S3FileSystem(anon=True)
    src_zarr = d / "native_7p91.zarr"
    maps = {}
    for tag, token in (("july", "july_retreat"), ("nov19", "november19")):
        p = d / f"map_{tag}.tif"
        if not p.exists():
            hits = [
                k
                for k in fs.ls(f"{BUCKET}/{seg}/ink-detection")
                if token in k and NATIVE_VOL in k and k.endswith(".tif")
            ]
            if not hits:
                print(f"{a.name}: no {token} map", file=sys.stderr)
                return 1
            fs.get(hits[0], str(p))
        maps[tag] = tifffile.imread(p)
    # A killed fetch leaves a valid-looking zarr with missing chunks, so the presence of the
    # zarr is not enough: rerun the fetcher (it skips complete chunks) until it has finished once.
    fetched = d / "fetch.done"
    if not a.skip_fetch and not fetched.exists():
        subprocess.run(
            [
                sys.executable,
                str(FETCHER),
                "--zarr-prefix",
                f"{seg}/surface-volumes/{NATIVE_VOL}.zarr",
                "--level",
                "0",
                "--full",
                "--out",
                str(src_zarr),
            ],
            check=True,
        )
        fetched.touch()
    src = zarr.open(str(src_zarr), mode="r")["0"]
    D, H, W = src.shape
    for tag, m in maps.items():
        if m.shape != (H, W):
            print(f"{a.name}: map {tag} {m.shape} != volume {(H, W)}", file=sys.stderr)
            return 1

    sv = d / "surface-volume.zarr"
    if not (sv / "0" / ".zarray").exists():
        print(f"{a.name}: resampling {src.shape} -> 9.362 um", flush=True)
        resample_volume(src, sv)
    native = zarr.open(str(sv), mode="r")["0"]
    depth, H2, W2 = native.shape
    mj = resample_plane(maps["july"].astype(np.float32), H2, W2)
    mn = resample_plane(maps["nov19"].astype(np.float32), H2, W2)
    zc = depth // 2
    valid = native[zc - 3 : zc + 4].astype(np.float32).mean(0) > 0
    ink = valid & (mj >= a.ink_thr) & (mn >= a.ink_thr)
    bg = valid & (mj < a.bg_thr) & (mn < a.bg_thr)
    sup = ink | bg
    write_label_zarr(d / f"{a.name}_inklabels.zarr", ink.astype(np.uint8) * 255, depth)
    write_label_zarr(d / f"{a.name}_supervision_mask.zarr", sup.astype(np.uint8) * 255, depth)
    if a.held_out:
        write_label_zarr(d / f"{a.name}_validation_mask.zarr", sup.astype(np.uint8) * 255, depth)
    key = np.where(valid, (mj + mn) / 2, -1).astype(np.float16)
    np.save(d / "key_native.npy", key)
    info = {
        "segment": a.segment,
        "shape": list(native.shape),
        "src_shape": [D, H, W],
        "valid_frac": float(valid.mean()),
        "ink_frac_of_valid": float(ink.sum() / max(valid.sum(), 1)),
        "supervised_frac_of_valid": float(sup.sum() / max(valid.sum(), 1)),
        "ink_thr": a.ink_thr,
        "bg_thr": a.bg_thr,
        "held_out": a.held_out,
    }
    (d / "build.json").write_text(json.dumps(info, indent=1))
    print(f"{a.name}: {info}")
    if not a.keep_native:
        shutil.rmtree(src_zarr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
