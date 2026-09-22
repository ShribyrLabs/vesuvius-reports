#!/usr/bin/env python3
"""Cut sheet merges out of a thick surface map along the ridges of its probability.

A surface model's thick binary map fuses neighbouring sheets wherever they come within a voxel
or two. Its probability still peaks once per sheet across the sheet, so non-maximum suppression
along the local sheet normal (structure tensor of the smoothed probability) leaves one ridge per
sheet. Every map voxel is then assigned to the nearest ridge component, and the voxels where two
assignments touch are removed. The map keeps its thickness, which the spiral fitter's track
extraction needs; only the bridges go (vesuvius.md LOG 2026-09-16 20:13: 60 Scroll-4 cubes merges
1.62 -> 0.38 on the tolerant scorer; 0139 spiral fit w042 0.605 vs 0.571 uncut). No model, no
GPU.

Library:
    sheet_normals(p)                    unit normal per voxel (zyx components)
    nms_ridge(p, mask)                  ridge voxels of p inside mask
    ridge_guided_cut(mask, ridge)       mask with the contact voxels between ridge instances removed
    cut_probability(p, threshold)       both steps on one probability block
    merge_and_split_tol(comp, lab)      merges / fragments with a 3-voxel tolerance
CLI (tile-wise; OME-Zarr in, OME-Zarr out laid out like the hosted m7 maps: 6 levels, 192^3 chunks):
    ridge_cut.py PROB.zarr OUT.zarr --threshold 73 [--box Z0 Z1 Y0 Y1 X0 X1] [--shape Z Y X]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from winding_split import contact_voxels  # noqa: E402

S26 = np.ones((3, 3, 3), bool)
TILE, PAD = 192, 24


def sheet_normals(p: np.ndarray) -> np.ndarray:
    """Dominant structure-tensor eigenvector of `p` (sigma 1 gradient, sigma 2 window)."""
    ps = ndi.gaussian_filter(p.astype(np.float32), 1.0)
    g = np.stack(np.gradient(ps), -1)
    st = np.empty(p.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            st[..., i, j] = st[..., j, i] = ndi.gaussian_filter(g[..., i] * g[..., j], 2.0)
    _, v = np.linalg.eigh(st.reshape(-1, 3, 3))
    return v[:, :, -1].reshape(p.shape + (3,))


def nms_ridge(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Voxels of `mask` where `p` is not exceeded at +-1 and +-2 voxels along the sheet normal."""
    n = sheet_normals(p)
    idx = np.argwhere(mask)
    base = p[tuple(idx.T)]
    keep = np.ones(len(idx), bool)
    for step in (1.0, -1.0, 2.0, -2.0):
        q = idx + step * n[tuple(idx.T)]
        keep &= base >= ndi.map_coordinates(p, q.T, order=1, mode="nearest")
    out = np.zeros(mask.shape, bool)
    out[tuple(idx[keep].T)] = True
    return out


def ridge_guided_cut(mask: np.ndarray, ridge: np.ndarray, tol: float = 3.0) -> np.ndarray:
    """`mask` minus the voxels where two nearest-ridge-component assignments touch.

    Voxels farther than `tol` from any ridge are dropped too."""
    rl, _ = ndi.label(ridge, structure=S26)
    d, idx = ndi.distance_transform_edt(rl == 0, return_indices=True)
    inst = rl[tuple(idx)].astype(np.int32)
    inst[(d > tol) | ~mask] = 0
    return mask & (inst > 0) & ~contact_voxels(inst)


def cut_probability(p: np.ndarray, threshold: float, tol: float = 3.0) -> np.ndarray:
    """Threshold `p`, find its ridges and return the cut binary map."""
    p = p.astype(np.float32)
    mask = p >= threshold
    return ridge_guided_cut(mask, nms_ridge(p, mask), tol)


def merge_and_split_tol(
    comp: np.ndarray, lab: np.ndarray, min_vox: int = 500, tol: float = 3.0
) -> tuple[int, int]:
    """(merge pairs, fragments) of instance map `comp` against binary label `lab`.

    As `winding_split.merge_pairs` / `fragments`, but each label voxel counts for the nearest
    predicted instance within `tol` voxels, so a thin or slightly displaced map can still cover
    its sheet. Numbers from this scorer are not comparable with the strict one."""
    lc, ln = ndi.label(lab, structure=S26)
    sizes = np.bincount(lc.ravel(), minlength=ln + 1)
    d, idx = ndi.distance_transform_edt(comp == 0, return_indices=True)
    near = comp[tuple(idx)]
    near[d > tol] = 0
    pn = int(comp.max())
    key = lc[lab].astype(np.int64) * (pn + 1) + near[lab]
    u, cnt = np.unique(key, return_counts=True)
    li, pi = u // (pn + 1), u % (pn + 1)
    big = (pi > 0) & (sizes[li] >= min_vox)
    owner = np.bincount(pi[big & (cnt >= 0.3 * sizes[li])], minlength=pn + 1)
    pieces = np.bincount(li[big & (cnt >= 0.05 * sizes[li])], minlength=ln + 1)
    return int((owner * (owner - 1) // 2).sum()), int(np.clip(pieces[1:] - 1, 0, None).sum())


def tiles(box: tuple[int, ...], tile: int = TILE) -> Iterator[tuple[int, int, int]]:
    z0, z1, y0, y1, x0, x1 = box
    for z in range(z0, z1, tile):
        for y in range(y0, y1, tile):
            for x in range(x0, x1, tile):
                yield z, y, x


def cut_tile(
    prob,
    corner,
    box,
    threshold: float,
    tile: int = TILE,
    pad: int = PAD,
    scale: float = 255.0,
) -> np.ndarray | None:
    """Cut map for the tile at `corner`, computed on a `pad`-voxel margin clipped to the array.

    `prob` is anything sliceable (numpy or zarr) holding probabilities times `scale`.
    Returns None for an (almost) empty tile."""
    lo = [max(c - pad, 0) for c in corner]
    hi = [min(c + tile + pad, n) for c, n in zip(corner, prob.shape, strict=True)]
    p = np.asarray(prob[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]).astype(np.float32) / scale
    if (p >= threshold).mean() < 0.001:
        return None
    cut = cut_probability(p, threshold)
    end = [min(c + tile, b) for c, b in zip(corner, box[1::2], strict=True)]
    return cut[tuple(slice(c - a, e - a) for c, a, e in zip(corner, lo, end, strict=True))]


def _worker(args):
    path, corner, box, threshold = args
    import zarr

    return corner, cut_tile(_level0(zarr.open(path, mode="r")), corner, box, threshold)


def _level0(node):
    return node["0"] if hasattr(node, "keys") and "0" in node else node


def level_shapes(shape: tuple[int, ...], levels: int) -> list[tuple[int, ...]]:
    """OME-Zarr pyramid shapes: each level halves the previous one, rounding up."""
    out = [tuple(shape)]
    for _ in range(levels - 1):
        out.append(tuple(-(-n // 2) for n in out[-1]))
    return out


def downsample_max(a: np.ndarray) -> np.ndarray:
    """2x max-pool with the odd trailing voxel kept, so no lit voxel vanishes from a level."""
    pad = [(0, n % 2) for n in a.shape]
    a = np.pad(a, pad)
    z, y, x = (n // 2 for n in a.shape)
    return a.reshape(z, 2, y, 2, x, 2).max(axis=(1, 3, 5))


def build_pyramid(root, box: tuple[int, ...], levels: int, slab: int = 64) -> None:
    """Fill levels 1..levels-1 of `root` from level 0 over `box` (level-0 coordinates)."""
    for lv in range(1, levels):
        src, dst = root[str(lv - 1)], root[str(lv)]
        f = 2 ** (lv - 1)
        lo = [b // f // 2 * 2 for b in box[0::2]]
        hi = [min(-(-b // f), n) for b, n in zip(box[1::2], src.shape, strict=True)]
        for z in range(lo[0], hi[0], slab):
            blk = np.asarray(src[z : min(z + slab, hi[0]), lo[1] : hi[1], lo[2] : hi[2]])
            d = downsample_max(blk)
            dst[
                z // 2 : z // 2 + d.shape[0],
                lo[1] // 2 : lo[1] // 2 + d.shape[1],
                lo[2] // 2 : lo[2] // 2 + d.shape[2],
            ] = d


def create_store(path: str, shape: tuple[int, ...], levels: int, attrs: dict):
    """Empty OME-Zarr v2 group laid out like the hosted surface predictions (192^3 chunks)."""
    import zarr
    from numcodecs import Blosc

    root = zarr.open_group(path, mode="w", zarr_format=2)
    for lv, sh in enumerate(level_shapes(shape, levels)):
        root.create_array(
            str(lv),
            shape=sh,
            chunks=(192, 192, 192),
            dtype="uint8",
            fill_value=0,
            compressors=Blosc(cname="zstd", clevel=1),
        )
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "ridge_cut",
            "axes": [{"name": n, "type": "space"} for n in "zyx"],
            "datasets": [
                {
                    "path": str(lv),
                    "coordinateTransformations": [{"type": "scale", "scale": [2.0**lv] * 3}],
                }
                for lv in range(levels)
            ],
            "metadata": {"downsampling_method": "max"},
        }
    ]
    root.attrs["ridge_cut"] = attrs
    return root


def main() -> None:
    import zarr

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("prob", help="OME-Zarr group (level '0') or array of uint8 probabilities")
    ap.add_argument("out", help="output OME-Zarr group; level '0' is the cut map (0/255)")
    ap.add_argument("--threshold", type=int, default=73, help="mask threshold on the 0-255 scale")
    ap.add_argument("--box", type=int, nargs=6, help="Z0 Z1 Y0 Y1 X0 X1 (default: whole volume)")
    ap.add_argument(
        "--shape",
        type=int,
        nargs=3,
        help="output level-0 shape (default: input's);"
        " set it to the scan's shape when the input is chunk-padded",
    )
    ap.add_argument("--levels", type=int, default=6, help="pyramid levels (hosted maps have 6)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    src = _level0(zarr.open(a.prob, mode="r"))
    shape = tuple(a.shape) if a.shape else src.shape
    box = tuple(a.box) if a.box else (0, shape[0], 0, shape[1], 0, shape[2])
    box = tuple(min(b, shape[k // 2]) for k, b in enumerate(box))
    attrs = {"source": str(a.prob), "threshold": a.threshold, "box": list(box)}
    root = create_store(a.out, shape, a.levels, attrs)
    out = root["0"]
    jobs = [(a.prob, c, box, a.threshold / 255.0) for c in tiles(box)]
    print(f"{len(jobs)} tiles", flush=True)
    t0, lit = time.time(), 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (c, core) in enumerate(ex.map(_worker, jobs, chunksize=2)):
            if core is not None:
                sl = tuple(slice(o, o + s) for o, s in zip(c, core.shape, strict=True))
                out[sl] = core.astype(np.uint8) * 255
                lit += int(core.sum())
            if k % 100 == 0:
                print(f"{k}/{len(jobs)} tiles, lit {lit}, {time.time() - t0:.0f}s", flush=True)
    build_pyramid(root, box, a.levels)
    print(f"RIDGE_CUT_DONE lit voxels {lit} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
