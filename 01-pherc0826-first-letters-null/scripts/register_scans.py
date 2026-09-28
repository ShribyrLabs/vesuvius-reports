#!/usr/bin/env python3
"""Register a high-resolution band scan of a scroll to its eligible 9 um scan, automatically.

Output is the open-data catalogue convention: rows (x, y, z), p_fixed = M @ [p_moving, 1] in
level-0 voxel indices of each volume (the form PHerc0139/0814/1667/Paris4 publish).

Method (after Ben Black's derive/refine_transform.py for PHerc1203, MIT, which assumed no
re-mount; this version also recovers a rotation about the scan axis and a mirror):
  1. z offset and direction: correlate the per-slice cross-section area (mm^2) of both scans.
  2. rotation + xy shift: on three band slices, search angle x mirror with FFT correlation of
     smoothed material masks at ~300 um, refine at ~75 um with intensity, refine z per slice.
  3. 3D refinement: at ~15 probe blocks, resample the moving scan into the fixed frame with the
     current transform (~37 um), measure the residual shift by phase correlation, and fit a
     similarity (Umeyama) to all block correspondences. Two rounds.
Volumes are read over HTTPS from the open-data bucket (uint8, raw 128^3 chunks; chunks cached).

Usage: register_scans.py FIXED_PREFIX MOVING_PREFIX --out transform.json
       [--reference published.json]  (reports the distance between the two transforms)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import threading
import time
from pathlib import Path

import numpy as np

HOST = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache_zarr"


# ---------------------------------------------------------------- pure geometry (unit-tested)
def umeyama(
    src: np.ndarray, dst: np.ndarray, scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity dst ~ s * R @ src + t for (N, 3) point sets. R may be improper
    (a mirror) only if the data demand it."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.ones(3)
    rot = u @ np.diag(d) @ vt
    s = float((sig * d).sum() / (a**2).sum() * len(src)) if scale else 1.0
    return s, rot, mu_d - s * rot @ mu_s


def xcorr_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[np.ndarray, float]:
    """Integer shift d with ref(p) ~ mov(p - d), by FFT cross-correlation of zero-mean arrays;
    also the NCC of the overlap at that shift."""
    a, b = ref - ref.mean(), mov - mov.mean()
    c = np.fft.ifftn(np.fft.fftn(a) * np.conj(np.fft.fftn(b))).real
    d = np.array(np.unravel_index(np.argmax(c), c.shape))
    d = np.where(d > np.array(c.shape) // 2, d - np.array(c.shape), d)
    return d, ncc_at(ref, mov, d)


def ncc_at(ref: np.ndarray, mov: np.ndarray, d: np.ndarray) -> float:
    sa, sb = [], []
    for k, dk in enumerate(d):
        n = ref.shape[k]
        sa.append(slice(max(dk, 0), n + min(dk, 0)))
        sb.append(slice(max(-dk, 0), n - max(dk, 0)))
    x, y = ref[tuple(sa)].ravel(), mov[tuple(sb)].ravel()
    if x.size < 16 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rot2(theta_deg: float, mirror: bool) -> np.ndarray:
    """2D linear map on (y, x): optional mirror of x, then rotation."""
    t = np.deg2rad(theta_deg)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return r @ np.diag([1.0, -1.0 if mirror else 1.0])


def warp2(img: np.ndarray, lin: np.ndarray, t: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Image W on a `shape` grid with W(q) = img(lin^-1 (q - t)); (y, x) pixel coords."""
    from scipy.ndimage import affine_transform

    inv = np.linalg.inv(lin)
    return affine_transform(img, inv, offset=-inv @ t, output_shape=shape, order=1)


def search_rotation(fixed: np.ndarray, moving: np.ndarray, angles, mirrors=(False, True)):
    """Best (ncc, angle, mirror, t) such that fixed(q) ~ moving(R^-1 (q - t)); t in pixels."""
    cf_, cm = np.array(fixed.shape) / 2, np.array(moving.shape) / 2
    best = (-2.0, 0.0, False, np.zeros(2))
    for mir in mirrors:
        for ang in angles:
            lin = rot2(ang, mir)
            t0 = cf_ - lin @ cm
            w = warp2(moving, lin, t0, fixed.shape)
            d, r = xcorr_shift(fixed, w)
            if r > best[0]:
                best = (r, float(ang), mir, t0 + d)
    return best


def to_catalogue(
    s: float, rot_zyx: np.ndarray, t_zyx_um: np.ndarray, v_fixed: float, v_mov: float
) -> list:
    """Similarity in um (zyx) -> 3x4 voxel matrix, rows (x, y, z), cols (x, y, z, 1)."""
    lin = s * rot_zyx * v_mov / v_fixed
    t = t_zyx_um / v_fixed
    perm = [2, 1, 0]
    m = np.zeros((3, 4))
    m[:, :3] = lin[np.ix_(perm, perm)]
    m[:, 3] = t[perm]
    return m.round(8).tolist()


def apply_catalogue(m: np.ndarray, p_zyx: np.ndarray) -> np.ndarray:
    xyz = p_zyx[:, ::-1]
    return (np.asarray(m)[:, :3] @ xyz.T).T[:, ::-1] + np.asarray(m)[:, 3][::-1]


# ---------------------------------------------------------------- bucket access
_local = threading.local()


def _session():
    """One requests.Session per thread (Session is not thread-safe)."""
    if not hasattr(_local, "s"):
        import requests

        _local.s = requests.Session()
    return _local.s


class Vol:
    def __init__(self, prefix: str, level: int):
        self.prefix, self.level = prefix, level
        meta = json.loads(_session().get(f"{HOST}/{prefix}/{level}/.zarray", timeout=60).text)
        assert meta["compressor"] is None and meta["dtype"] == "|u1", meta
        self.shape = tuple(meta["shape"])
        self.v0 = float(prefix.split("/")[-1].split("-")[1].replace("um", ""))
        self.um = self.v0 * 2**level

    # The um frame is level-0 voxel index * v0; a level-L voxel i is centred on level-0 index
    # (i + 0.5) * 2^L - 0.5, so blocks read at different levels agree to the sub-voxel.
    def to_um(self, i):
        return (np.asarray(i, dtype=np.float64) + 0.5) * self.um - 0.5 * self.v0

    def to_idx(self, u):
        return (np.asarray(u, dtype=np.float64) + 0.5 * self.v0) / self.um - 0.5

    def _get(self, key: str, rng: tuple[int, int] | None = None) -> bytes:
        last = ""
        for attempt in range(10):
            try:
                h = {"Range": f"bytes={rng[0]}-{rng[1] - 1}"} if rng else {}
                r = _session().get(f"{HOST}/{key}", headers=h, timeout=120)
                if r.status_code in (403, 404):
                    return b""
                if r.status_code in (200, 206):
                    return (
                        r.content
                        if (rng is None or r.status_code == 206)
                        else r.content[rng[0] : rng[1]]
                    )
                last = f"HTTP {r.status_code}"
            except Exception as e:  # network blips: retry with backoff
                last = repr(e)[:120]
            time.sleep(min(2.0 * (attempt + 1), 20))
        raise OSError(f"failed {key}: {last}")

    def chunk(self, i: int, j: int, k: int) -> np.ndarray:
        key = f"{self.prefix}/{self.level}/{i}/{j}/{k}"
        fn = CACHE / key
        if fn.exists():
            buf = fn.read_bytes()
        else:
            buf = self._get(key)
            fn.parent.mkdir(parents=True, exist_ok=True)
            fn.write_bytes(buf)
        if len(buf) != 128**3:
            return np.zeros((128, 128, 128), np.uint8)
        return np.frombuffer(buf, np.uint8).reshape(128, 128, 128)

    def read(self, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Block [z0:z1, y0:y1, x0:x1]; out-of-range parts are zero."""
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        lo = [max(v, 0) for v in (z0, y0, x0)]
        hi = [min(v, s) for v, s in zip((z1, y1, x1), self.shape, strict=True)]
        if any(a >= b for a, b in zip(lo, hi, strict=True)):
            return out
        jobs = [
            (i, j, k)
            for i in range(lo[0] // 128, (hi[0] - 1) // 128 + 1)
            for j in range(lo[1] // 128, (hi[1] - 1) // 128 + 1)
            for k in range(lo[2] // 128, (hi[2] - 1) // 128 + 1)
        ]
        with cf.ThreadPoolExecutor(16) as ex:
            for (i, j, k), blk in zip(jobs, ex.map(lambda t: self.chunk(*t), jobs), strict=True):
                a = [max(lo[0], i * 128), max(lo[1], j * 128), max(lo[2], k * 128)]
                b = [
                    min(hi[0], i * 128 + 128),
                    min(hi[1], j * 128 + 128),
                    min(hi[2], k * 128 + 128),
                ]
                out[a[0] - z0 : b[0] - z0, a[1] - y0 : b[1] - y0, a[2] - x0 : b[2] - x0] = blk[
                    a[0] - i * 128 : b[0] - i * 128,
                    a[1] - j * 128 : b[1] - j * 128,
                    a[2] - k * 128 : b[2] - k * 128,
                ]
        return out

    def plane(self, z: int) -> np.ndarray:
        """One z plane by range reads (16 KB per chunk), uncached."""
        zc, zo = divmod(z, 128)
        ny, nx = -(-self.shape[1] // 128), -(-self.shape[2] // 128)
        out = np.zeros((ny * 128, nx * 128), np.uint8)
        off = zo * 128 * 128

        def get(jk):
            buf = self._get(f"{self.prefix}/{self.level}/{zc}/{jk[0]}/{jk[1]}", (off, off + 16384))
            if len(buf) == 16384:
                out[jk[0] * 128 : jk[0] * 128 + 128, jk[1] * 128 : jk[1] * 128 + 128] = (
                    np.frombuffer(buf, np.uint8).reshape(128, 128)
                )

        with cf.ThreadPoolExecutor(24) as ex:
            list(ex.map(get, [(j, k) for j in range(ny) for k in range(nx)]))
        return out[: self.shape[1], : self.shape[2]]


# ---------------------------------------------------------------- pipeline
def area_curve(vol: Vol, step: int) -> tuple[np.ndarray, np.ndarray]:
    """(z_um, cross-section area mm^2) sampled every `step` planes of `vol`'s level."""
    zs = np.arange(0, vol.shape[0], step)
    with cf.ThreadPoolExecutor(2) as ex:
        planes = list(ex.map(vol.plane, zs.tolist()))
    area = np.array([(p > 0).sum() * (vol.um / 1000) ** 2 for p in planes])
    return vol.to_um(zs), area


def match_area_curves(zf, af, zm, am, grid: float = 300.0) -> tuple[float, int, float]:
    """(dz, sign, ncc) with z_fixed ~ sign * z_moving + dz, by sliding the moving scan's
    cross-section-area curve (trimmed to its material) along the fixed one, both directions."""
    gf = np.arange(zf[0], zf[-1], grid)
    f = np.interp(gf, zf, af)
    gm = np.arange(zm[0], zm[-1], grid)
    m = np.interp(gm, zm, am)
    keep = np.nonzero(m > 0.05 * m.max())[0]
    m, gm = m[keep[0] : keep[-1] + 1], gm[keep[0] : keep[-1] + 1]
    best = (-2.0, 0.0, 1)
    for sign in (1, -1):
        ms, zs = (m, gm) if sign == 1 else (m[::-1], gm[::-1])
        for off in range(0, len(f) - len(ms) + 1):
            r = np.corrcoef(f[off : off + len(ms)], ms)[0, 1]
            if r > best[0]:
                best = (float(r), gf[off] - sign * zs[0], sign)
    return best[1], best[2], best[0]


def coarse_z(fix: Vol, mov: Vol) -> tuple[float, int, float]:
    """(dz_um, sign, ncc): z_fixed_um ~ sign * z_mov_um + dz_um, from area curves at ~300 um."""
    zf, af = area_curve(fix, max(1, round(300 / fix.um)))
    zm, am = area_curve(mov, max(1, round(300 / mov.um)))
    return match_area_curves(zf, af, zm, am)


def match_slice(fix: Vol, mov: Vol, z_mov: int, z_fix_guess: int, zwin: int, coarse_angle=None):
    """Rotation/shift of one moving plane vs fixed planes near the guess. Returns dict in um."""
    from scipy.ndimage import gaussian_filter, zoom

    pm = mov.plane(z_mov).astype(np.float32)
    pm = zoom(pm, mov.um / fix.um, order=1)  # onto the fixed pixel size
    f0 = fix.plane(z_fix_guess).astype(np.float32)
    k = 4  # ~300 um search grid
    small = lambda a: gaussian_filter((a > 0).astype(np.float32), 2)[::k, ::k]  # noqa: E731
    if coarse_angle is None:
        r, ang, mir, _ = search_rotation(small(f0), small(pm), np.arange(0, 360, 2.0))
    else:
        ang, mir = coarse_angle
    best = None
    for zf in range(z_fix_guess - zwin, z_fix_guess + zwin + 1, max(1, zwin // 4) if zwin else 1):
        f = fix.plane(zf).astype(np.float32)
        r, a2, m2, t = search_rotation(f, pm, np.arange(ang - 3, ang + 3.01, 0.25), mirrors=(mir,))
        if best is None or r > best["ncc"]:
            best = {"ncc": r, "angle": a2, "mirror": m2, "t_px": t, "z_fix": zf}
    best["t_um"] = (best["t_px"] * fix.um).tolist()
    best["z_mov_um"] = float(mov.to_um(z_mov))
    best["z_fix_um"] = float(fix.to_um(best["z_fix"]))
    best.pop("t_px")
    return best


def slice_to_similarity(matches: list, sign: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Initial 3D similarity (um, zyx) from per-slice 2D rigid matches (mean angle)."""
    ang = float(np.mean([m["angle"] for m in matches]))
    mir = matches[0]["mirror"]
    r2 = rot2(ang, mir)
    rot = np.eye(3)
    rot[0, 0] = sign
    rot[1:, 1:] = r2
    src = np.array([[m["z_mov_um"], 0, 0] for m in matches])
    dst = np.array([[m["z_fix_um"], *m["t_um"]] for m in matches])
    t = (dst - src @ rot.T).mean(0)
    return 1.0, rot, t


def refine_blocks(fix: Vol, mov: Vol, s, rot, t, probes_um: np.ndarray, n: int = 96):
    """Correspondences (src_um, dst_um, ncc) from 3D phase correlation at each probe."""
    from scipy.ndimage import affine_transform

    rows = []
    inv = np.linalg.inv(s * rot)
    for p in probes_um:
        q = s * rot @ p + t  # predicted fixed-frame um
        c0 = np.round(fix.to_idx(q)).astype(int) - n // 2
        a = fix.read(c0[0], c0[0] + n, c0[1], c0[1] + n, c0[2], c0[2] + n)
        if (a > 0).mean() < 0.5:
            continue
        # moving block large enough to cover the rotated fixed block
        h = int(np.ceil(n * fix.um / mov.um * 0.9)) + 4
        m0 = np.round(mov.to_idx(p)).astype(int) - h
        b = mov.read(m0[0], m0[0] + 2 * h, m0[1], m0[1] + 2 * h, m0[2], m0[2] + 2 * h).astype(
            np.float32
        )
        # fixed block voxel i -> fixed um -> moving um -> moving block voxel j (affine_transform
        # wants j = lin @ i + off)
        lin = inv * fix.um / mov.um
        off = mov.to_idx(inv @ (fix.to_um(c0) - t)) - m0
        w = affine_transform(b, lin, offset=off, output_shape=(n, n, n), order=1)
        d, r = xcorr_shift(a.astype(np.float32), w)
        if r < 0.3:
            continue
        rows.append((p, q + d * fix.um, r))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "fixed", help="bucket prefix of the eligible scan, e.g. PHerc0846A/volumes/2025...zarr"
    )
    ap.add_argument("moving", help="bucket prefix of the high-resolution scan")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, help="published transform json to compare against")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)  # noqa: E731

    f3, m5 = Vol(a.fixed, 3), Vol(a.moving, 5)
    dz, sign, rz = coarse_z(Vol(a.fixed, 5), Vol(a.moving, 5))
    log(f"coarse z: z_fix_um = {sign:+d} * z_mov_um + {dz:.0f}  (area-curve ncc {rz:.3f})")

    # three band slices with material
    zm = [int(f * m5.shape[0]) for f in (0.25, 0.5, 0.75)]
    matches, coarse = [], None
    for z in zm:
        zf_guess = int(round((sign * (z + 0.5) * m5.um + dz) / f3.um))
        mt = match_slice(f3, m5, z, zf_guess, zwin=8, coarse_angle=coarse)
        coarse = coarse or (mt["angle"], mt["mirror"])
        matches.append(mt)
        log(
            f"slice z_mov={z}: angle {mt['angle']:.2f} mirror {mt['mirror']} ncc {mt['ncc']:.3f} "
            f"z_fix {mt['z_fix']} (guess {zf_guess})"
        )
    s, rot, t = slice_to_similarity(matches, sign)

    # probe blocks spread through the band at the material's lateral positions
    fix2, mov4 = Vol(a.fixed, 2), Vol(a.moving, 4)
    rng = np.random.default_rng(0)
    probes = []
    for zf in np.linspace(0.08, 0.92, 7):
        z = int(zf * m5.shape[0])
        pl = m5.plane(z)
        ys, xs = np.nonzero(pl > 0)
        if len(ys) < 1000:
            continue
        for i in rng.choice(len(ys), 3, replace=False):
            probes.append(m5.to_um([z, ys[i], xs[i]]))
    probes = np.array(probes)
    for rnd in (1, 2):
        rows = refine_blocks(fix2, mov4, s, rot, t, probes)
        if len(rows) < 6:
            log(f"round {rnd}: only {len(rows)} blocks registered — stopping")
            break
        src = np.array([r[0] for r in rows])
        dst = np.array([r[1] for r in rows])
        s, rot, t = umeyama(src, dst)
        res = dst - (s * (rot @ src.T).T + t)
        log(
            f"round {rnd}: {len(rows)}/{len(probes)} blocks (ncc {min(r[2] for r in rows):.2f}-"
            f"{max(r[2] for r in rows):.2f}); scale {s:.5f}; residual rms zyx "
            f"{np.round(np.sqrt((res**2).mean(0)), 1)} um"
        )

    m = to_catalogue(s, rot, t, f3.um / 8, m5.um / 32)
    out = {
        "_comment": "DERIVED by scripts/register_scans.py (not published by the open-data catalogue).",
        "from_volume": a.moving,
        "to_volume": a.fixed,
        "axis_order": "xyz",
        "transformation_matrix": m,
        "scale": s,
        "rotation_zyx": rot.tolist(),
        "coarse_z": {"dz_um": dz, "sign": sign, "ncc": rz},
        "slice_matches": matches,
        "n_blocks": len(rows),
        "residual_rms_um_zyx": np.sqrt((res**2).mean(0)).tolist() if len(rows) >= 6 else None,
    }
    if a.reference:
        ref = np.array(json.loads(a.reference.read_text())["transformation_matrix"])
        mshape = Vol(a.moving, 0).shape
        grid = np.array(
            [
                [z, y, x]
                for z in np.linspace(0, mshape[0], 5)
                for y in np.linspace(0.2, 0.8, 3) * mshape[1]
                for x in np.linspace(0.2, 0.8, 3) * mshape[2]
            ]
        )
        dist = np.linalg.norm(
            apply_catalogue(np.array(m), grid) - apply_catalogue(ref, grid), axis=1
        )
        dist_um = dist * f3.um / 8
        out["vs_reference_um"] = {"median": float(np.median(dist_um)), "max": float(dist_um.max())}
        log(f"vs reference: median {np.median(dist_um):.0f} um, max {dist_um.max():.0f} um")
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
