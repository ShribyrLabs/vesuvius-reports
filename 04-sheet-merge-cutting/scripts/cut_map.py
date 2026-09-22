"""Cut a binary surface map by the zero-shot winding field (GATE vesuvius.md LOG 2026-09-15 07:03).

Tiles of 256^3 (stride 192) over the map's bounding box, CT and map fetched once per 256-deep
z-slab: RGT-Est field per tile from the CT with the tile's own stacking axis, components split at
histogram valleys (b64 / d0.3), and every voxel whose 26-neighbourhood holds a different instance
is flagged (`winding_split.contact_voxels`, zero-padded — the first version wrapped around and
flagged every tile face; fixed 2026-09-15 15:50). The output map is the input minus
all flagged voxels (a one-voxel gap where sheets were fused), in the same zarr v2 layout the track
extractor reads (uint8, 0/255, 128^3 chunks).

usage: cut_map.py CT_SOURCE MAP_IN.zarr MAP_OUT.zarr [--box Z0 Z1 Y0 Y1 X0 X1]
  CT_SOURCE: .npy (whole box, zyx) or a zarr array URL/path with the same coordinates as MAP_IN.
"""

import os
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from numcodecs import Blosc
from scipy import ndimage as ndi

ROOT = Path(os.environ.get("VESUVIUS_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts"))
from winding_split import contact_voxels, rgt_field, split_by_field, stacking_axis  # noqa: E402

TILE, STRIDE = 256, 192


def starts(n: int) -> list[int]:
    s = list(range(0, max(n - TILE, 0) + 1, STRIDE))
    if s[-1] + TILE < n:
        s.append(n - TILE)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ct")
    ap.add_argument("map_in")
    ap.add_argument("map_out")
    ap.add_argument(
        "--box", type=int, nargs=6, help="z0 z1 y0 y1 x0 x1 of the region to process"
    )
    ap.add_argument(
        "--mask-by-ct",
        action="store_true",
        help="also drop map voxels where the (masked) CT is 0, i.e. outside the scroll body",
    )
    ap.add_argument(
        "--weights",
        default=str(ROOT / "data/checkpoints/rgt_est/RGT-Est_CIG-Benchmark.pt"),
    )
    a = ap.parse_args()
    src = zarr.open_array(a.map_in, mode="r", zarr_format=2)
    if a.ct.endswith(".npy"):
        ct_all = np.load(a.ct, mmap_mode="r")
        ct_get = lambda sl: np.asarray(ct_all[sl])  # noqa: E731
    else:
        ct_arr = zarr.open_array(a.ct, mode="r", zarr_format=2)
        ct_get = lambda sl: np.asarray(ct_arr[sl])  # noqa: E731
    z0, z1, y0, y1, x0, x1 = (
        a.box if a.box else (0, src.shape[0], 0, src.shape[1], 0, src.shape[2])
    )
    model = torch.jit.load(a.weights).cuda().eval()
    out = zarr.create_array(
        store=a.map_out,
        shape=src.shape,
        chunks=(128, 128, 128),
        dtype="uint8",
        zarr_format=2,
        compressors=Blosc(cname="zstd", clevel=1),
        overwrite=True,
    )
    cut = np.zeros((z1 - z0, y1 - y0, x1 - x0), bool)
    n_tiles = 0
    t0 = time.time()
    for tz in starts(z1 - z0):
        zs = slice(z0 + tz, z0 + tz + TILE)
        t_io = time.time()
        mask_slab = np.asarray(src[zs, y0:y1, x0:x1]) > 0
        if mask_slab.mean() < 0.0005:
            continue
        ct_slab = ct_get((zs, slice(y0, y1), slice(x0, x1)))
        print(f"slab z{z0 + tz}: fetched in {time.time() - t_io:.0f}s", flush=True)
        for ty in starts(y1 - y0):
            for tx in starts(x1 - x0):
                box = (slice(0, TILE), slice(ty, ty + TILE), slice(tx, tx + TILE))
                mask = mask_slab[box]
                if mask.mean() < 0.002:
                    continue
                ct = ct_slab[box]
                if ct.shape != mask.shape:
                    continue
                if a.mask_by_ct:
                    cut[tz : tz + TILE, ty : ty + TILE, tx : tx + TILE] |= mask & (ct == 0)
                field = rgt_field(model, ct, stacking_axis(mask))
                inst = split_by_field(mask, field, bins=64, depth=0.3)
                c = contact_voxels(inst)
                cut[tz : tz + TILE, ty : ty + TILE, tx : tx + TILE] |= c
                n_tiles += 1
                if n_tiles % 20 == 0:
                    print(
                        f"{n_tiles} tiles, cut voxels so far {int(cut.sum())}, {time.time() - t0:.0f}s",
                        flush=True,
                    )
        del ct_slab, mask_slab
    # write only the processed box; the output keeps the source's full shape (unwritten chunks
    # read as 0), so it drops into any consumer at the same coordinates
    zb0 = z0 - z0 % 128
    for z in range(zb0, z1, 128):
        za, zbnd = max(z, z0), min(z + 128, z1)
        blk = np.asarray(src[za:zbnd, y0:y1, x0:x1]) > 0
        blk &= ~cut[za - z0 : zbnd - z0]
        out[za:zbnd, y0:y1, x0:x1] = blk.astype(np.uint8) * 255
    # final stats, chunked (a whole-box read here was OOM-killed once)
    box_lit = 0
    for z in range(z0, z1, 128):
        box_lit += int((np.asarray(src[z : min(z + 128, z1), y0:y1, x0:x1]) > 0).sum())
    print(
        f"CUT_DONE tiles {n_tiles} cut voxels {int(cut.sum())} "
        f"({100 * cut.sum() / max(box_lit, 1):.2f} % of box map voxels) {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
