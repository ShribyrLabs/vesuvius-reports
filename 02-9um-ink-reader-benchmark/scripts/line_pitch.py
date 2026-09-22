#!/usr/bin/env python3
"""Is there a repeating line structure along the scroll axis in an ink map?

Text lines run around the scroll, so successive lines are stacked along the scroll axis (z).
Every map pixel gets its z from the tifxyz mesh (bilinear, grid step = 1/scale pixels); the
map is averaged in z bins (default 8 voxels), high-passed (minus a 1200-voxel running mean) and
autocorrelated. "rise" = highest autocorrelation after the first minimum minus that minimum,
within lags 200-1000 voxels (1.9-9.4 mm at 9.362 um); text gives a clear peak at the line
pitch, noise gives none. This generalises the vertical-autocorrelation test that separated
w042 forward (rise 0.108) from depth-reversed (0.000) on 2026-09-08 to meshes of any
orientation. Prints JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def z_profile(pred: np.ndarray, zpix: np.ndarray, bin_vox: float, min_count: int = 200):
    """(bin centres, mean prediction per z bin) over pixels with valid z and prediction."""
    v = np.isfinite(zpix) & (pred > 0)
    zb = np.floor(zpix[v] / bin_vox).astype(np.int64)
    lo = zb.min()
    idx = zb - lo
    s = np.bincount(idx, weights=pred[v].astype(np.float64))
    n = np.bincount(idx)
    prof = np.where(n >= min_count, s / np.maximum(n, 1), np.nan)
    return (np.arange(len(prof)) + lo + 0.5) * bin_vox, prof


def rise(
    prof: np.ndarray,
    bin_vox: float,
    lag_lo: float = 200,
    lag_hi: float = 1000,
    smooth_vox: float = 1200,
) -> dict:
    """Autocorrelation rise after the first minimum, over lags [lag_lo, lag_hi] voxels."""
    p = prof.copy()
    ok = np.isfinite(p)
    if ok.sum() < 3 * lag_hi / bin_vox:
        return {"rise": None, "lag_vox": None, "n_bins": int(ok.sum())}
    p[~ok] = np.nanmean(p)
    k = max(1, int(smooth_vox / bin_vox))
    hp = p - np.convolve(p, np.ones(k) / k, mode="same")
    hp = hp[k // 2 : len(hp) - k // 2]  # drop edges where the running mean is truncated
    hp = (hp - hp.mean()) / (hp.std() + 1e-9)
    n = len(hp)
    lags = np.arange(int(lag_lo / bin_vox), min(int(lag_hi / bin_vox), n // 2))
    ac = np.array([np.mean(hp[: n - L] * hp[L:]) for L in lags])
    i_min = int(np.argmin(ac[: max(1, len(ac) // 2)]))
    j = i_min + int(np.argmax(ac[i_min:]))
    return {
        "rise": float(ac[j] - ac[i_min]),
        "lag_vox": float(lags[j] * bin_vox),
        "n_bins": int(ok.sum()),
    }


def mesh_z_pixels(tifxyz: Path, shape: tuple[int, int]) -> np.ndarray:
    """z (voxels) for every map pixel by bilinear upsampling of the mesh z grid; NaN = invalid."""
    import tifffile
    from scipy.ndimage import map_coordinates

    z = tifffile.imread(tifxyz / "z.tif").astype(np.float64)
    meta = json.loads((tifxyz / "meta.json").read_text())
    step = 1.0 / float(meta["scale"][0])
    z[z == -1] = np.nan
    ys = (np.arange(shape[0]) + 0.5) / step - 0.5
    xs = (np.arange(shape[1]) + 0.5) / step - 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    valid = np.isfinite(z).astype(np.float64)
    zf = np.nan_to_num(z)
    zz = map_coordinates(zf, [gy, gx], order=1, mode="nearest")
    vv = map_coordinates(valid, [gy, gx], order=1, mode="nearest")
    zz[vv < 0.999] = np.nan  # any contribution from an invalid vertex -> invalid
    return zz


def main() -> int:
    import tifffile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pred", type=Path, help="ink map tif (H, W), same grid as the mesh render")
    ap.add_argument("mesh", type=Path, help="tifxyz dir the map was rendered from")
    ap.add_argument("--bin", type=float, default=8.0)
    a = ap.parse_args()
    pred = tifffile.imread(a.pred).astype(np.float32)
    zpix = mesh_z_pixels(a.mesh, pred.shape)
    _, prof = z_profile(pred, zpix, a.bin)
    out = rise(prof, a.bin)
    out["z_span_vox"] = float(np.nanmax(zpix) - np.nanmin(zpix))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
