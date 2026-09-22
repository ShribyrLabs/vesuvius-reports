"""Bigger-patch letter-scale test: pool existing 2 um tile reads of one winding, map every sampled
pixel to the w035 key (nearest point on the densified staff surface), then hp r at (0,0), a
+-200 px shift search, and the same search at far key offsets (null).

usage: big.py HIRES_WINDING_DIR [--tol 24] [--step 3] [--sense rev]
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
KEY = ROOT / "data/ink_pseudo/w035/key_2p4um.tif"
W35 = ROOT / "data/xform_test/w035_to2um"
SIG = 20.0
CELL = 20  # key px per w035_to2um grid cell
T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def xyz(p):
    return np.stack([tifffile.imread(p / f"{n}.tif").astype(np.float32) for n in "xyz"], -1)


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 100 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wdir", type=Path)
    ap.add_argument("--tol", type=float, default=24.0)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--sense", default="rev")
    ap.add_argument("--split-only", action="store_true")
    ap.add_argument("--save-kxy", type=Path, default=None, help="save matched key px (y, x) as .npy")
    ap.add_argument("--key", type=Path, default=KEY, help=".tif or .npy key on the ref grid")
    ap.add_argument("--ref", type=Path, default=W35, help="reference tifxyz the key lives on")
    ap.add_argument("--pxscale", type=float, default=1.0,
                    help="key px size / 2.4 um; spatial constants are divided by it (9.36 um: 3.9)")
    ap.add_argument("--glob", default=None, help="tile glob inside wdir (default by sense)")
    ap.add_argument("--mesh", type=Path, default=None, help="mesh_2um dir (default wdir/mesh_2um)")
    a = ap.parse_args()

    global SIG
    SIG = SIG / a.pxscale
    PS = a.pxscale
    w35 = xyz(a.ref)
    w35_ok = w35[..., 2] >= 0
    vr, vc = np.where(w35_ok)
    coarse = cKDTree(w35[vr, vc], balanced_tree=False, compact_nodes=False)
    mdir = a.mesh or a.wdir / "mesh_2um"
    meshes = {}

    def mesh_for(t):
        """Per-tile mesh if the tile's folder has its own mesh_2um, else the shared one."""
        md = t.parent / "mesh_2um" if (t.parent / "mesh_2um").exists() else mdir
        if md not in meshes:
            g = xyz(md)
            pp = 1.0 / float(__import__("json").loads((md / "meta.json").read_text())["scale"][0])
            meshes[md] = (g, (g[..., 2] >= 0).astype(np.float32), pp)
        return meshes[md]
    suffix = "_rev_pred.tif" if a.sense == "rev" else "_pred.tif"
    if a.glob:
        tiles = sorted(a.wdir.glob(a.glob))
    else:
        tiles = sorted(p for p in a.wdir.glob(f"t*_s*{suffix}") if (a.sense == "rev") == ("_rev_" in p.name))
    tick(f"{a.wdir}: {len(tiles)} tiles")

    P_all, V_all, Vraw_all = [], [], []
    for t in tiles:
        ty, tx, _ = map(int, re.match(r"t(\d+)_(\d+)_s(\d+)", t.name).groups())
        mesh, mesh_ok, per_px = mesh_for(t)
        pred = tifffile.imread(t).astype(np.float32)
        ph = pred - gaussian_filter(pred, SIG)
        ii, jj = np.mgrid[0 : pred.shape[0] : a.step, 0 : pred.shape[1] : a.step]
        gi, gj = (ii + ty) / per_px, (jj + tx) / per_px
        ok = map_coordinates(mesh_ok, [gi, gj], order=1) > 0.999
        # keep away from tile edges, where the high-pass is distorted by the zero border
        ok &= (ii > 2 * SIG) & (jj > 2 * SIG) & (ii < pred.shape[0] - 2 * SIG) & (jj < pred.shape[1] - 2 * SIG)
        gi, gj, ii, jj = gi[ok], gj[ok], ii[ok], jj[ok]
        P = np.stack([map_coordinates(mesh[..., k], [gi, gj], order=1) for k in range(3)], -1)
        P_all.append(P)
        V_all.append(ph[ii, jj])
        Vraw_all.append(pred[ii, jj])
    P = np.concatenate(P_all)
    V = np.concatenate(V_all)
    Vraw = np.concatenate(Vraw_all)
    d0, vid = coarse.query(P, distance_upper_bound=a.tol + 40, workers=-1)
    m = np.isfinite(d0)
    tick(f"{len(P):,} sampled px, {m.mean():.1%} within {a.tol + 40:.0f} vox of a w035 vertex")
    P, V, Vraw, vid = P[m], V[m], Vraw[m], vid[m]

    # densify only the w035 cells our pixels land near (dilated by one cell)
    cells = set()
    for rr, cc in zip(vr[vid], vc[vid]):
        for dr in (-1, 0):
            for dc in (-1, 0):
                cells.add((rr + dr, cc + dc))
    cells = np.array(sorted(cells))
    H35, W35n = w35_ok.shape
    cells = cells[(cells[:, 0] >= 0) & (cells[:, 1] >= 0) & (cells[:, 0] < H35 - 1) & (cells[:, 1] < W35n - 1)]
    full = w35_ok[cells[:, 0], cells[:, 1]] & w35_ok[cells[:, 0] + 1, cells[:, 1]] \
        & w35_ok[cells[:, 0], cells[:, 1] + 1] & w35_ok[cells[:, 0] + 1, cells[:, 1] + 1]
    cells = cells[full]
    u = (np.arange(CELL) / CELL).astype(np.float32)
    fy, fx = np.meshgrid(u, u, indexing="ij")
    fy, fx = fy.ravel(), fx.ravel()
    pts, kys, kxs = [], [], []
    for chunk in np.array_split(cells, max(1, len(cells) // 20000)):
        rr, cc = chunk[:, 0:1], chunk[:, 1:2]
        c00, c10 = w35[rr, cc], w35[rr + 1, cc]
        c01, c11 = w35[rr, cc + 1], w35[rr + 1, cc + 1]
        wy, wx = fy[None, :, None], fx[None, :, None]
        pts.append(((1 - wy) * (1 - wx) * c00 + wy * (1 - wx) * c10 + (1 - wy) * wx * c01
                    + wy * wx * c11).reshape(-1, 3))
        kys.append((rr * CELL + (fy * CELL).astype(int)[None, :]).ravel())
        kxs.append((cc * CELL + (fx * CELL).astype(int)[None, :]).ravel())
    pts = np.concatenate(pts)
    kys = np.concatenate(kys)
    kxs = np.concatenate(kxs)
    tick(f"densified {len(cells):,} w035 cells -> {len(pts):,} key px")
    fine = cKDTree(pts, balanced_tree=False, compact_nodes=False)
    d, idx = fine.query(P, distance_upper_bound=a.tol, workers=-1)
    m = np.isfinite(d)
    ky, kx, V, Vraw, d = kys[idx[m]], kxs[idx[m]], V[m], Vraw[m], d[m]
    tick(f"{len(ky):,} px within {a.tol:.0f} vox of w035 (median {np.median(d):.1f}); "
         f"distinct key px {len(np.unique(ky.astype(np.int64) * 100000 + kx)):,}")

    M = int(1800 / PS)
    y0, x0 = max(0, ky.min() - M), max(0, kx.min() - M)
    keyf = np.load(a.key) if a.key.suffix == ".npy" else tifffile.imread(a.key)
    key = keyf[y0 : ky.max() + M, x0 : kx.max() + M].astype(np.float32)
    del keyf
    kvalid = key >= 0
    key = np.where(kvalid, key, 0).astype(np.float32)
    kh = key - gaussian_filter(key, SIG)
    ky, kx = ky - y0, kx - x0
    keep = kvalid[ky, kx]
    ky, kx, V, Vraw = ky[keep], kx[keep], V[keep], Vraw[keep]
    if a.save_kxy:
        np.save(a.save_kxy, np.stack([ky + y0, kx + x0]))
        np.save(a.save_kxy.with_name(a.save_kxy.stem + "_vals.npy"), Vraw)
    H, W = kh.shape
    tick(f"key crop {key.shape}, ink frac under our px {(key[ky, kx] > 128).mean():.3f}; "
         f"raw r(0,0) {r(Vraw, key[ky, kx]):+.3f}  hp r(0,0) {r(V, kh[ky, kx]):+.3f}")

    ks = gaussian_filter(key, 40.0 / PS)
    kv = ks[ky, kx]
    for name, sel in [("inked (smoothed key > 60)", kv > 60), ("mid (20-60)", (kv >= 20) & (kv <= 60)),
                      ("blank (smoothed key < 20)", kv < 20)]:
        yy, xx = np.clip(ky[sel] + int(1200 / PS), 0, H - 1), np.clip(kx[sel], 0, W - 1)
        nl = []
        for oy, ox in [(1200, 0), (-1200, 0), (0, 1200), (0, -1200), (900, 900), (-900, -900),
                       (900, -900), (-900, 900), (600, 0), (-600, 0), (0, 600), (0, -600)]:
            oy, ox = int(oy / PS), int(ox / PS)
            y2, x2 = ky[sel] + oy, kx[sel] + ox
            ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
            if sel.sum() and ok.mean() > 0.95:
                nl.append(r(V[sel][ok], kh[y2[ok], x2[ok]]))
        if nl:
            tick(f"  {name}: null r(0,0) at far offsets: " + " ".join(f"{v:+.3f}" for v in nl))
        tick(f"  {name}: n={sel.sum():,}  hp r {r(V[sel], kh[ky[sel], kx[sel]]):+.3f}  "
             f"(key shifted 1200: {r(V[sel], kh[yy, xx]):+.3f})  pred hp sd {V[sel].std():.1f} "
             f"key hp sd {kh[ky[sel], kx[sel]].std():.1f}")
    if a.split_only:
        return
    S = np.arange(-200, 201, 8) / PS
    S = np.unique(np.round(S).astype(int))

    def search(oy, ox):
        best, grid = (-9.0, 0, 0), np.full((len(S), len(S)), np.nan)
        for i, dy in enumerate(S):
            for j, dx in enumerate(S):
                yy, xx = ky + oy + dy, kx + ox + dx
                ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                if ok.mean() < 0.95:
                    continue
                v = r(V[ok], kh[yy[ok], xx[ok]])
                grid[i, j] = v
                if v > best[0]:
                    best = (v, int(dy), int(dx))
        return best, grid

    (b, dy, dx), grid = search(0, 0)
    c = len(S) // 2
    np.save(a.wdir / f"bigshift_{a.sense}.npy", grid)
    tick(f"REAL: hp r(0,0) {grid[c, c]:+.3f}; best {b:+.3f} at ({dy},{dx}); "
         f"grid median {np.nanmedian(grid):+.3f}")
    nulls = []
    for oy, ox in [(1200, 0), (-1200, 0), (0, 1200), (0, -1200), (900, 900), (-900, -900),
                   (900, -900), (-900, 900)]:
        (nb, _, _), ng = search(int(oy / PS), int(ox / PS))
        if np.isfinite(ng).mean() > 0.5:
            nulls.append(nb)
            tick(f"null@({oy},{ox}) best {nb:+.3f}")
    if nulls:
        tick(f"SUMMARY: real best {b:+.3f} at ({dy},{dx}) vs null best max {max(nulls):+.3f} "
             f"median {np.median(nulls):+.3f} ({len(nulls)})")


if __name__ == "__main__":
    main()
