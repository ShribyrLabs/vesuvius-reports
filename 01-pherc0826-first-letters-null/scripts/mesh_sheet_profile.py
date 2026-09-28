"""Measure whether a tifxyz spiral mesh sits on the papyrus sheets.

On one axial slice, every mesh column is cut at ``z`` and the raw CT is sampled
along the in-plane normal over plus/minus half a sheet pitch. For each profile,

    contrast = (on-mesh - off-mesh) / (peak - trough)

where on-mesh is the value at the mesh point, off-mesh the mean beyond a quarter
pitch on either side. Random placement gives ~0; a mesh on the sheets gives a
clearly positive number. The "snapped" control moves each point to the
brightest voxel inside the same window and reports the ceiling this metric can
reach on this slice, so the two numbers are comparable across scrolls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy.ndimage import map_coordinates

AIR_LEVEL = 20  # uint8 CT value below which a whole profile counts as air


def load_mesh(tifxyz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return x, y, z (rows x cols, level-0 voxels, -1 = invalid) and meta.json."""
    arrays = [tifffile.imread(tifxyz / f"{n}.tif").astype(np.float32) for n in "xyz"]
    with open(tifxyz / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return arrays[0], arrays[1], arrays[2], meta


def cut_at_z(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, z_target: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate each mesh column at z_target -> (cols, x, y) for columns that reach it."""
    cols, xs, ys = [], [], []
    for col in range(x.shape[1]):
        valid = z[:, col] != -1.0
        if valid.sum() < 2:
            continue
        cz, cx, cy = z[valid, col], x[valid, col], y[valid, col]
        order = np.argsort(cz)
        cz, cx, cy = cz[order], cx[order], cy[order]
        if cz[0] <= z_target <= cz[-1]:
            cols.append(col)
            xs.append(np.interp(z_target, cz, cx))
            ys.append(np.interp(z_target, cz, cy))
    return np.asarray(cols), np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def in_plane_normals(cols: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Unit normals (nx, ny) from the tangent between neighbouring mesh columns."""
    t = np.zeros((len(cols), 2))
    for i in range(len(cols)):
        lo = i - 1 if i > 0 and cols[i] - cols[i - 1] == 1 else i
        hi = i + 1 if i + 1 < len(cols) and cols[i + 1] - cols[i] == 1 else i
        if lo == hi:
            continue
        t[i] = (xs[hi] - xs[lo], ys[hi] - ys[lo])
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    length = np.linalg.norm(n, axis=1)
    keep = length > 0
    n[keep] /= length[keep, None]
    return np.where(keep[:, None], n, np.nan)


def read_slice(
    path: str, z_index: int, crop: tuple[int, int, int, int], level: str = "0"
) -> np.ndarray:
    """Read one plane of pyramid ``level`` inside crop (x0, y0, x1, y1) as float32.

    Coordinates are in that level's own voxels; the default level 0 is the working
    frame for a native 9 µm scroll, level 2 for Paris 4 (2.4 µm base).
    """
    opts = {"anon": True} if path.startswith("s3://") else None
    group = zarr.open_group(path, mode="r", storage_options=opts)
    x0, y0, x1, y1 = crop
    return np.asarray(group[level][z_index, y0:y1, x0:x1]).astype(np.float32)


def profiles(
    plane: np.ndarray, xs: np.ndarray, ys: np.ndarray, normals: np.ndarray, half: int
) -> np.ndarray:
    """Sample plane along each normal at integer offsets -half..half -> (N, 2*half+1)."""
    t = np.arange(-half, half + 1, dtype=np.float64)
    px = xs[:, None] + t[None, :] * normals[:, 0:1]
    py = ys[:, None] + t[None, :] * normals[:, 1:2]
    return map_coordinates(plane, [py.ravel(), px.ravel()], order=1, mode="nearest").reshape(
        px.shape
    )


def contrast(prof: np.ndarray, quarter: int) -> np.ndarray:
    """Per-profile (on - off) / (peak - trough); the centre column is the mesh point."""
    mid = prof.shape[1] // 2
    on = prof[:, mid]
    off = np.concatenate([prof[:, : mid - quarter + 1], prof[:, mid + quarter :]], axis=1).mean(
        axis=1
    )
    span = prof.max(axis=1) - prof.min(axis=1)
    return (on - off) / np.maximum(span, 1e-6)


def measure(
    plane: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    normals: np.ndarray,
    pitch: float,
) -> dict:
    """Contrast for the mesh as placed and for the snapped-to-brightest control."""
    half = max(2, int(round(pitch / 2)))
    quarter = max(1, int(round(pitch / 4)))
    prof = profiles(plane, xs, ys, normals, half)
    papyrus = prof.max(axis=1) >= AIR_LEVEL
    shift = prof.argmax(axis=1) - half
    snapped = profiles(plane, xs + shift * normals[:, 0], ys + shift * normals[:, 1], normals, half)
    c_mesh = contrast(prof[papyrus], quarter)
    c_snap = contrast(snapped[papyrus], quarter)
    return {
        "points": int(len(xs)),
        "in_air_fraction": float(1 - papyrus.mean()) if len(xs) else float("nan"),
        "contrast_mesh": float(c_mesh.mean()) if len(c_mesh) else float("nan"),
        "contrast_snapped": float(c_snap.mean()) if len(c_snap) else float("nan"),
        "median_abs_shift_vox": (
            float(np.median(np.abs(shift[papyrus]))) if papyrus.any() else float("nan")
        ),
        "pitch": pitch,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--tifxyz", type=Path, required=True, help="Directory with x/y/z.tif + meta.json"
    )
    ap.add_argument(
        "--zarr", required=True, help="CT OME-Zarr root (local or s3://), level 0 is used"
    )
    ap.add_argument("--z", type=float, required=True, help="Axial slice (level-0 voxels)")
    ap.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        required=True,
        help="Level-0 window to read and measure in",
    )
    ap.add_argument("--pitch", type=float, default=17.0, help="Sheet pitch in voxels")
    args = ap.parse_args(argv)

    x, y, z, _meta = load_mesh(args.tifxyz)
    cols, xs, ys = cut_at_z(x, y, z, args.z)
    x0, y0, x1, y1 = args.crop
    margin = args.pitch
    inside = (xs >= x0 + margin) & (xs < x1 - margin) & (ys >= y0 + margin) & (ys < y1 - margin)
    normals = in_plane_normals(cols, xs, ys)
    inside &= ~np.isnan(normals[:, 0])
    if not inside.any():
        print(json.dumps({"points": 0}))
        return 2
    plane = read_slice(args.zarr, int(round(args.z)), (x0, y0, x1, y1))
    result = measure(plane, xs[inside] - x0, ys[inside] - y0, normals[inside], args.pitch)
    result.update({"z": args.z, "crop": list(args.crop)})
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
