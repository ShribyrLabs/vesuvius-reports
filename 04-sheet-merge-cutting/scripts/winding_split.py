#!/usr/bin/env python3
"""Split merged sheets in a papyrus surface map with a zero-shot winding-number field.

Sheets in a rolled scroll are nested layers, so locally they are level sets of one smooth
"layer index" field. A relative-geologic-time network from seismic interpretation (RGT-Est,
Dou et al. 2026, arXiv 2605.01273; code MIT, weights CC-BY 4.0) produces such a field from raw
CT with no training on scrolls (vesuvius.md LOG 2026-09-15 00:14). Cutting each connected
component of a binary surface map where its field values split into separate modes removes
sheet merges without touching the segmenter (LOG 00:26: -45 % merges for +6 % fragments at the
default cut, on 60 held-out Scroll-4 cubes).

Library:
    stacking_axis(mask)                 axis along which the sheets are stacked
    rgt_field(model, ct, axis)          the field on the CT crop (zyx), same shape
    split_by_field(mask, field, ...)    instance ids after cutting at histogram valleys
    merge_pairs(inst, lab), fragments(inst, lab)   the metrics
CLI:
    winding_split.py CT.tif MASK.tif OUT.tif [--weights PT] [--bins 64] [--depth 0.3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

S26 = np.ones((3, 3, 3), bool)
MODEL_SHAPE = (
    400,
    512,
    512,
)  # the only inference size the released RGT-Est checkpoint accepts


def stacking_axis(mask: np.ndarray, step: int = 2) -> int:
    """Axis along which the sheets in `mask` are stacked (dominant gradient direction).

    Computed on the mask subsampled by `step` (smoothing scaled to match); the answer is the
    same as at full resolution on every cube and tile tried, at a fraction of the cost.
    """
    sub = mask[::step, ::step, ::step] if step > 1 else mask
    sm = ndi.gaussian_filter(sub.astype(np.float32), 3.0 / step)
    cov = np.zeros((3, 3))
    grads = np.gradient(sm)
    for i in range(3):
        for j in range(i, 3):
            cov[i, j] = cov[j, i] = float((grads[i] * grads[j]).sum())
    return int(np.argmax(np.linalg.eigh(cov)[1][:, -1] ** 2))


def z_score_clip(x: np.ndarray, clp: float = 2.0) -> np.ndarray:
    """RGT-Est input normalisation: z-score, clip, then map to [-1, 1]."""
    z = np.clip((x - x.mean()) / (x.std() + 1e-8), -clp, clp)
    return ((z - z.min()) / (z.max() - z.min() + 1e-6) * 2 - 1).astype(np.float32)


def rgt_field(model, ct: np.ndarray, axis: int) -> np.ndarray:
    """RGT-Est field for a zyx CT crop, with `axis` as the model's 'time' axis."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    perm = [axis] + [a for a in range(3) if a != axis]
    x = np.ascontiguousarray(z_score_clip(ct.astype(np.float32)).transpose(perm))
    t = torch.from_numpy(x)[None, None]
    xi = F.interpolate(t, MODEL_SHAPE, mode="trilinear")
    xi = torch.cat([xi, torch.zeros_like(xi), torch.zeros_like(xi)], 1).cuda()
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        r = model(nn.ReplicationPad3d(8)(xi))[:, :, 8:-8, 8:-8, 8:-8]
    field = F.interpolate(r.float(), x.shape, mode="trilinear").cpu().numpy()[0, 0]
    return np.ascontiguousarray(field.transpose(np.argsort(perm)))


def split_by_field(
    mask: np.ndarray,
    field: np.ndarray,
    bins: int = 64,
    depth: float = 0.3,
    min_vox: int = 200,
) -> np.ndarray:
    """Instance ids: each component of `mask` is cut where the histogram of its field values has
    a valley lower than `depth` times the smaller neighbouring peak.

    Every component is processed inside its own bounding box (``ndi.find_objects``), which is
    what makes this fast on tiles with hundreds of components.
    """
    comp, n = ndi.label(mask, structure=S26)
    out = np.zeros(mask.shape, np.int32)
    nxt = 1
    for i, box in enumerate(ndi.find_objects(comp), start=1):
        if box is None:
            continue
        m = comp[box] == i
        v = field[box][m]
        if len(v) < min_vox:
            out[box][m] = nxt
            nxt += 1
            continue
        h, edges = np.histogram(v, bins=bins)
        hs = ndi.gaussian_filter1d(h.astype(float), 1.0)
        # zero-pad so modes at the ends of the range count as peaks (a two-sheet merge has
        # exactly those)
        hp = np.concatenate([[0.0], hs, [0.0]])
        peaks, _ = find_peaks(hp, prominence=depth * hs.max() * 0.2)
        peaks = peaks - 1
        cuts = []
        for a, b in zip(peaks[:-1], peaks[1:], strict=True):
            j = a + int(np.argmin(hs[a : b + 1]))
            if hs[j] < depth * min(hs[a], hs[b]):
                cuts.append(edges[j + 1])
        if not cuts:
            out[box][m] = nxt
            nxt += 1
            continue
        band = np.searchsorted(np.array(cuts), v)
        bl = np.zeros(m.shape, np.int32)
        bl[m] = band + 1
        view = out[box]
        for b in range(1, len(cuts) + 2):
            sub, k = ndi.label(bl == b, structure=S26)
            if k:
                sel = sub > 0
                view[sel] = sub[sel] + nxt - 1
                nxt += k
    return out


def contact_voxels(inst: np.ndarray) -> np.ndarray:
    """Voxels of `inst` whose 26-neighbourhood holds a different non-zero instance id.

    Two rank filters instead of 26 shifted copies: a foreground voxel touches another instance
    iff the largest or the smallest non-zero id in its 3x3x3 neighbourhood differs from its own.
    """
    fg = inst > 0
    big = np.iinfo(inst.dtype).max
    mx = ndi.maximum_filter(inst, size=3, mode="constant", cval=0)
    mn = ndi.minimum_filter(np.where(fg, inst, big), size=3, mode="constant", cval=big)
    return fg & ((mx != inst) | (mn != inst))


def _cover(inst: np.ndarray, lab: np.ndarray, min_vox: int, frac: float):
    lc, ln = ndi.label(lab, structure=S26)
    sizes = np.bincount(lc.ravel(), minlength=ln + 1)
    pn = int(inst.max())
    key = lc[lab].astype(np.int64) * (pn + 1) + inst[lab]
    u, cnt = np.unique(key, return_counts=True)
    li, pi = u // (pn + 1), u % (pn + 1)
    ok = (pi > 0) & (sizes[li] >= min_vox) & (cnt >= frac * sizes[li])
    return li[ok], pi[ok], ln, pn


def merge_pairs(inst: np.ndarray, lab: np.ndarray, min_vox: int = 500) -> int:
    """Pairs of label sheets (>= min_vox) covered >= 30 % by the same predicted instance."""
    _, pi, _, pn = _cover(inst, lab, min_vox, 0.3)
    owner = np.bincount(pi, minlength=pn + 1)
    return int((owner * (owner - 1) // 2).sum())


def fragments(inst: np.ndarray, lab: np.ndarray, min_vox: int = 500) -> int:
    """Extra predicted pieces (each >= 5 % of the sheet) per label sheet, summed."""
    li, _, ln, _ = _cover(inst, lab, min_vox, 0.05)
    pieces = np.bincount(li, minlength=ln + 1)
    return int(np.clip(pieces[1:] - 1, 0, None).sum())


def main() -> None:
    import tifffile
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ct", type=Path)
    ap.add_argument("mask", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path("data/checkpoints/rgt_est/RGT-Est_CIG-Benchmark.pt"),
    )
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--depth", type=float, default=0.3)
    a = ap.parse_args()
    ct = tifffile.imread(a.ct)
    mask = tifffile.imread(a.mask) > 0
    model = torch.jit.load(str(a.weights)).cuda().eval()
    field = rgt_field(model, ct, stacking_axis(mask))
    inst = split_by_field(mask, field, a.bins, a.depth)
    tifffile.imwrite(a.out, inst.astype(np.uint32), compression="zlib")
    print(f"{ndi.label(mask, structure=S26)[1]} components -> {int(inst.max())} instances")


if __name__ == "__main__":
    main()
