"""Build an outer-shell tifxyz for the spiral fitter from a surface prediction and (optionally) raw CT.

For each z plane the outermost material radius from the umbilicus is taken per angular bin:
the surface-prediction mask (saturated inference blocks and specks removed) and the raw CT
above a threshold (specks removed) are both used and the larger radius wins. The result is
written as a tifxyz grid with z along rows and angle along columns, in the working-frame
coordinates of the surface store, so `load_tifxyz` / `ShellPolarMap` read it unchanged.

usage: build_outer_shell.py --surface ZARR --umbilicus JSON --z-begin Z --z-end Z --out DIR
       [--ct ZARR --ct-scale 2 --ct-thr 50] [--z-step 8] [--theta-bins 720] [--margin 5]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy import ndimage


def umbilicus_interp(path: Path):
    pts = json.loads(Path(path).read_text())["control_points"]
    z = np.array([p["z"] for p in pts], dtype=np.float64)
    o = np.argsort(z)
    x = np.array([p["x"] for p in pts], dtype=np.float64)[o]
    y = np.array([p["y"] for p in pts], dtype=np.float64)[o]
    z = z[o]
    return lambda zz: (float(np.interp(zz, z, x)), float(np.interp(zz, z, y)))


def drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return np.concatenate([[False], sizes >= min_px])[lab]


def clean_surface_plane(plane: np.ndarray, min_cc: int) -> np.ndarray:
    """Remove the solid saturated blocks the inference writes at the mask edge, then specks."""
    fg = plane > 0
    # solid blocks are >=192 px wide; find them on a 4x downsampled mask (opening 4x4 ~ 16 px)
    ds = fg[::4, ::4]
    solid = ndimage.binary_opening(ds, structure=np.ones((4, 4), bool))
    solid = ndimage.binary_dilation(solid, structure=np.ones((3, 3), bool))
    solid_full = np.repeat(np.repeat(solid, 4, axis=0), 4, axis=1)[: fg.shape[0], : fg.shape[1]]
    fg &= ~solid_full
    return drop_small(fg, min_cc)


def polar_max(mask: np.ndarray, scale: float, cx: float, cy: float, n_bins: int) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    yy = yy * scale + scale / 2
    xx = xx * scale + scale / 2
    th = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)
    rr = np.hypot(xx - cx, yy - cy)
    bins = (th / (2 * np.pi) * n_bins).astype(np.int64) % n_bins
    out = np.full(n_bins, -np.inf, dtype=np.float32)
    if rr.size:
        np.maximum.at(out, bins, rr.astype(np.float32))
    out[~np.isfinite(out)] = np.nan
    return out


def read_plane(arr, idx: int, tries: int = 10):
    """Read one plane, retrying transport errors.

    The stores are remote: a momentary DNS failure threw away a 75-plane run mid-band
    (2026-09-11), because one raised exception aborts the whole build.
    """
    for k in range(tries):
        try:
            return arr[int(idx)]
        except Exception as exc:  # noqa: BLE001 - any transport error is worth retrying
            if k == tries - 1:
                raise
            print(f"  read z={idx} failed ({type(exc).__name__}), retry {k + 1}", flush=True)
            time.sleep(min(2**k, 30))
    raise AssertionError("unreachable")


def write_tifxyz(out: Path, x, y, z, a, n_rows: int, nb: int) -> int:
    """Write the tifxyz grid (and meta.json). Called periodically so a crash keeps its work."""
    out.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out / "x.tif", x)
    tifffile.imwrite(out / "y.tif", y)
    tifffile.imwrite(out / "z.tif", z)
    valid = z >= 0
    if not valid.any():
        return 0
    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    meta = {
        "format": "tifxyz",
        "type": "seg",
        "uuid": a.uuid,
        "scale": [1.0 / a.z_step, 1.0 / a.z_step],
        "bbox": [pts.min(axis=0).tolist(), pts.max(axis=0).tolist()],
        "area_vx2": float(valid.sum() * a.z_step**2),
        "source": {"surface": a.surface, "ct": a.ct, "ct_thr": a.ct_thr, "margin": a.margin},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return int(valid.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--surface", required=True, help="surface-prediction zarr array in working coords"
    )
    ap.add_argument("--umbilicus", required=True, type=Path)
    ap.add_argument("--z-begin", type=int, required=True)
    ap.add_argument("--z-end", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ct", default=None, help="raw CT zarr array (any pyramid level)")
    ap.add_argument("--ct-scale", type=float, default=1.0, help="working voxels per CT voxel")
    ap.add_argument("--ct-thr", type=float, default=50.0)
    ap.add_argument("--z-step", type=int, default=8)
    ap.add_argument("--theta-bins", type=int, default=720)
    ap.add_argument("--margin", type=float, default=5.0, help="radius added outward (vox)")
    ap.add_argument("--min-cc", type=int, default=100)
    ap.add_argument("--uuid", default="outer_shell_ours")
    ap.add_argument("--resume", action="store_true", help="continue from the tifs already in --out")
    ap.add_argument("--retries", type=int, default=10, help="retries per remote plane read")
    ap.add_argument("--save-every", type=int, default=10, help="checkpoint every N planes")
    a = ap.parse_args()

    umb = umbilicus_interp(a.umbilicus)
    surf = zarr.open(a.surface, mode="r")
    ct = zarr.open(a.ct, mode="r") if a.ct else None
    zs = np.arange(a.z_begin, a.z_end + 1, a.z_step)
    nb = a.theta_bins
    theta = (np.arange(nb) + 0.5) / nb * 2 * np.pi
    x = np.full((len(zs), nb), -1.0, np.float32)
    y = np.full((len(zs), nb), -1.0, np.float32)
    z = np.full((len(zs), nb), -1.0, np.float32)
    start = 0
    if a.resume and (a.out / "z.tif").exists():
        prev = tifffile.imread(a.out / "z.tif")
        if prev.shape == z.shape:
            x[:] = tifffile.imread(a.out / "x.tif")
            y[:] = tifffile.imread(a.out / "y.tif")
            z[:] = prev
            done = np.flatnonzero((z >= 0).any(axis=1))
            start = int(done.max()) + 1 if done.size else 0
            print(f"resume: {start}/{len(zs)} rows already built", flush=True)
        else:
            print(f"resume ignored: existing grid {prev.shape} != {z.shape}", flush=True)

    for i, z0 in enumerate(zs):
        if i < start:
            continue
        cx, cy = umb(z0)
        fg = clean_surface_plane(read_plane(surf, z0, a.retries), a.min_cc)
        r = polar_max(fg, 1.0, cx, cy, nb)
        if ct is not None:
            raw = read_plane(ct, z0 // a.ct_scale, a.retries).astype(np.float32)
            pl = ndimage.gaussian_filter(raw, 1.0)
            m = drop_small(pl > a.ct_thr, 50)
            r = np.fmax(r, polar_max(m, a.ct_scale, cx, cy, nb))
        ok = np.isfinite(r)
        r = r + a.margin
        x[i, ok] = cx + r[ok] * np.cos(theta[ok])
        y[i, ok] = cy + r[ok] * np.sin(theta[ok])
        z[i, ok] = z0
        print(f"z={z0}: {ok.sum()}/{nb} bins, r median {np.nanmedian(r):.0f}", flush=True)
        if a.save_every and (i + 1 - start) % a.save_every == 0:
            write_tifxyz(a.out, x, y, z, a, len(zs), nb)

    n = write_tifxyz(a.out, x, y, z, a, len(zs), nb)
    print(f"wrote {a.out}: {n} valid vertices on a {len(zs)}x{nb} grid")


if __name__ == "__main__":
    main()
