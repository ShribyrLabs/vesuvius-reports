"""Orientation gate for a fitted spiral: does the mesh run *along* the papyrus layers?

Intensity gates fail on delaminated papyrus (a sheet at 9.6 µm is a bundle of strands with
air gaps, so a point on the sheet's mid-plane is in a gap half the time — see the Paris 4
check of 2026-09-08). This gate instead compares the mesh's in-plane normal at each cut
point with the local layer normal from the CT structure tensor. A mesh that follows the
layers scores near 1; a mesh cutting across them scores like the rotated control.

Usage: gate_orientation.py --meshes DIR --zarr ROOT --z Z --crop X0 Y0 X1 Y1 [--level 0]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_sheet_profile import cut_at_z, in_plane_normals, load_mesh, read_slice  # noqa: E402

COHERENCE_MIN = 0.3
MATERIAL_MIN = 50.0


def layer_normals(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Structure tensor -> (nx, ny) of the dominant gradient direction, coherence, material."""
    gx = ndi.gaussian_filter(plane, 1, order=(0, 1))
    gy = ndi.gaussian_filter(plane, 1, order=(1, 0))
    jxx = ndi.gaussian_filter(gx * gx, 3)
    jyy = ndi.gaussian_filter(gy * gy, 3)
    jxy = ndi.gaussian_filter(gx * gy, 3)
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy)  # direction of the dominant gradient
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / np.maximum(jxx + jyy, 1e-6)
    mat = ndi.gaussian_filter(plane, 1) > MATERIAL_MIN
    return np.stack([np.cos(ang), np.sin(ang)], -1), coh, mat


def score(plane, xs, ys, normals) -> dict:
    ln, coh, mat = layer_normals(plane)
    coords = [ys, xs]
    lx = map_coordinates(ln[..., 0], coords, order=1, mode="nearest")
    ly = map_coordinates(ln[..., 1], coords, order=1, mode="nearest")
    c = map_coordinates(coh, coords, order=1, mode="nearest")
    m = map_coordinates(mat.astype(np.float32), coords, order=1, mode="nearest") > 0.5
    ok = (c >= COHERENCE_MIN) & m
    cos = np.abs(lx * normals[:, 0] + ly * normals[:, 1])
    rot = np.abs(lx * -normals[:, 1] + ly * normals[:, 0])  # 90° rotated control
    return {
        "points": int(len(xs)),
        "usable_fraction": float(ok.mean()) if len(xs) else float("nan"),
        "along_layers": float((cos[ok] > 0.8).mean()) if ok.any() else float("nan"),
        "along_layers_rotated_control": float((rot[ok] > 0.8).mean()) if ok.any() else float("nan"),
        "median_cos": float(np.median(cos[ok])) if ok.any() else float("nan"),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--meshes", required=True, help="fitted/ dir with w### subdirs, or one tifxyz")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--z", type=float, required=True)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), required=True)
    ap.add_argument("--level", default="0")
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    x0, y0, x1, y1 = args.crop
    meshes = Path(args.meshes)
    dirs = sorted(d for d in glob.glob(str(meshes / "w*")) if re.search(r"w\d+$", d))
    if not dirs:
        dirs = [str(meshes)]
    plane = read_slice(args.zarr, int(round(args.z)), (x0, y0, x1, y1), args.level)
    per, XS, YS, NN = [], [], [], []
    for d in dirs:
        x, y, z, _ = load_mesh(Path(d))
        cols, xs, ys = cut_at_z(x, y, z, args.z)
        if len(xs) < 5:
            continue
        nn = in_plane_normals(cols, xs, ys)
        keep = (
            (xs >= x0 + args.margin)
            & (xs < x1 - args.margin)
            & (ys >= y0 + args.margin)
            & (ys < y1 - args.margin)
            & ~np.isnan(nn[:, 0])
        )
        if keep.sum() < 5:
            continue
        r = score(plane, xs[keep] - x0, ys[keep] - y0, nn[keep])
        per.append((Path(d).name, r))
        XS.append(xs[keep] - x0)
        YS.append(ys[keep] - y0)
        NN.append(nn[keep])
    if not XS:
        print(json.dumps({"points": 0}))
        return 2
    pooled = score(plane, np.concatenate(XS), np.concatenate(YS), np.concatenate(NN))
    pooled.update({"z": args.z, "windings": len(per)})
    print(json.dumps(pooled))
    if not args.json:
        print("winding  pts  usable  along  control")
        for name, r in per:
            print(
                f"{name:7s} {r['points']:5d}  {r['usable_fraction']:.2f}   "
                f"{r['along_layers']:.2f}   {r['along_layers_rotated_control']:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
