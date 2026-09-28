#!/usr/bin/env python3
"""Render a tifxyz surface from an OME-Zarr CT volume, sampling on the GPU.

Same job as villa's vc_render_tifxyz (CPU-only, and its fetches retry for only ~1.5 s, which our
router's DNS outages outlast at high parallelism): for every output pixel, interpolate the mesh
position, take the surface normal, and sample the volume at num_slices points along it.

Here the interpolation and sampling run in torch on the GPU, and the volume is fetched by a
threaded reader with retries and connection reuse (scripts/register_scans.py Vol). The output
is written in the same layout vc_render_tifxyz produces (group, "0" = (slices, H, W) uint8,
128-chunked, blosc), so the rest of the chain is unchanged.

Validate against a CPU render before trusting it: --compare <zarr> reports per-slice
correlation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def level_geometry(scale: float, level: int, out_scale: float) -> tuple[float, float]:
    """(effective tifxyz scale, coordinate divisor) for rendering from pyramid level `level`.

    Mesh coordinates are level-0 voxels; level L voxels are 2**L of them, so coordinates are divided by
    2**L. `out_scale` < 1 renders fewer output pixels than the grid's full resolution: output pixel p
    samples grid index p * scale / out_scale.
    """
    return scale / out_scale, float(2**level)


def out_shape(grid_hw: tuple[int, int], scale: float) -> tuple[int, int]:
    """Output pixels for a tifxyz grid: villa uses round(n / scale) (scale 0.05 -> 20x)."""
    return int(round(grid_hw[0] / scale)), int(round(grid_hw[1] / scale))


def block_ranges(n: int, block: int) -> list[tuple[int, int]]:
    return [(i, min(i + block, n)) for i in range(0, n, block)]


def split_block(b: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Split an output block (y0, y1, x0, x1) into up to four non-empty quadrants."""
    y0, y1, x0, x1 = b
    ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
    ys = [(y0, ym), (ym, y1)] if y1 - y0 > 1 else [(y0, y1)]
    xs = [(x0, xm), (xm, x1)] if x1 - x0 > 1 else [(x0, x1)]
    return [(a, b_, c, d) for a, b_ in ys for c, d in xs]


def prefetch_chunks(vol, need: set, workers: int) -> int:
    """Warm the on-disk chunk cache for every chunk in `need`; returns the count.

    The fetched arrays are dropped as they arrive. Collecting them (list(ex.map(...))) held
    every chunk in RAM until the end: 10,713 x 2 MB = 21 GB on a raw 1447 mesh (2026-09-25).
    Submissions are bounded too, so pending results cannot pile up either.
    """
    import concurrent.futures as cf

    todo = sorted(need)
    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        window = max(1, 4 * workers)
        for s in range(0, len(todo), window):
            futs = [ex.submit(lambda c: vol.chunk(*c) is None, c) for c in todo[s : s + window]]
            for f in futs:
                f.result()
                done += 1
    return done


def chunk_ids_for_bbox(
    zlo: float,
    zhi: float,
    ylo: float,
    yhi: float,
    xlo: float,
    xhi: float,
    chunk: int = 128,
) -> list[tuple[int, int, int]]:
    """Every (z, y, x) chunk index a sample bounding box touches.

    Kept separate from the GPU path so the prefetch set can be computed from a block's
    bounds alone: retaining each block's sample points instead costs (slices x 3 x h x w)
    floats for the WHOLE output, which is 121 GiB on a 1660x174840 spiral ribbon.
    """
    return [
        (i, j, k)
        for i in range(int(zlo) // chunk, int(zhi) // chunk + 1)
        for j in range(int(ylo) // chunk, int(yhi) // chunk + 1)
        for k in range(int(xlo) // chunk, int(xhi) // chunk + 1)
    ]


def normal_ok(valid: np.ndarray) -> np.ndarray:
    """Cells of a tifxyz grid (or pixels of a block) whose normal the renderer can compute.

    The normal comes from differences to the +-1 row and column neighbours (torch.gradient:
    central inside, one-sided on the array border). A difference taken across an invalid (-1)
    cell mixes that sentinel into the tangent and tilts the normal: 22-24 degrees on a flat test
    sheet, a median of 24-63 degrees at the hole edges of real meshes (vesuvius.md LOG 2026-09-28).
    So a cell counts only if it and every neighbour the difference uses are valid, as in
    vc_render_tifxyz, whose grid_normal_int returns NaN next to an invalid cell and so renders
    nothing there.
    """
    v = np.asarray(valid, dtype=bool)
    ok = v.copy()
    ok[:, 1:] &= v[:, :-1]
    ok[:, :-1] &= v[:, 1:]
    ok[1:, :] &= v[:-1, :]
    ok[:-1, :] &= v[1:, :]
    return ok


def grid_normals(grid, ok, smooth: int = 0):
    """Unit normals (1, 3, gh, gw) of a tifxyz grid (3, gh, gw); meaningful only where `ok`.

    `smooth` box-averages the normals over `ok` cells only, so no normal from beside a hole
    leaks into its neighbours.
    """
    import torch
    import torch.nn.functional as F

    gn = torch.cross(torch.gradient(grid, dim=2)[0], torch.gradient(grid, dim=1)[0], dim=0)
    n = (gn / (gn.norm(dim=0, keepdim=True) + 1e-6))[None]
    if smooth:
        w = ok.to(n.dtype)[None, None]
        num = F.avg_pool2d(n * w, smooth, stride=1, padding=smooth // 2, count_include_pad=False)
        den = F.avg_pool2d(w, smooth, stride=1, padding=smooth // 2, count_include_pad=False)
        n = num / den.clamp(min=1e-6)
        n = n / (n.norm(dim=1, keepdim=True) + 1e-6)
    return n


def main() -> int:
    import torch
    import torch.nn.functional as F
    import zarr
    from numcodecs import Blosc
    from register_scans import Vol

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mesh", type=Path, help="tifxyz dir")
    ap.add_argument("volume", help="bucket prefix of the CT volume (level 0 is used)")
    ap.add_argument("out", type=Path, help="output .zarr")
    ap.add_argument("--num-slices", type=int, default=28)
    ap.add_argument(
        "--max-slab-voxels",
        type=int,
        default=200_000_000,
        help="split a block whose CT bounding box exceeds this (0.2 GB RAM, 0.8 GB GPU float)",
    )
    ap.add_argument("--slice-step", type=float, default=1.0, help="in voxels of --level")
    ap.add_argument(
        "--level", type=int, default=0, help="pyramid level to sample (coords / 2**level)"
    )
    ap.add_argument(
        "--out-scale",
        type=float,
        default=1.0,
        help="output pixels per full-resolution pixel (e.g. 0.125 with --level 3)",
    )
    ap.add_argument("--flip-normals", action="store_true")
    ap.add_argument("--block", type=int, default=1024, help="output block side in pixels")
    ap.add_argument("--normal-source", choices=["pixel", "grid"], default="grid")
    ap.add_argument("--normal-smooth", type=int, default=0, help="box-smooth the mesh normals")
    ap.add_argument("--offset-shift", type=float, default=0.0)
    ap.add_argument("--prefetch-workers", type=int, default=64, help="0 disables prefetch")
    ap.add_argument(
        "--keep-geometry",
        action="store_true",
        help="hold every block's sample points on the GPU (only for small outputs)",
    )
    ap.add_argument("--compare", type=Path, help="existing render to correlate against")
    a = ap.parse_args()

    import tifffile

    xyz = np.stack([tifffile.imread(a.mesh / f"{n}.tif").astype(np.float32) for n in "xyz"], 0)
    meta = json.loads((a.mesh / "meta.json").read_text())
    scale, cdiv = level_geometry(float(meta["scale"][0]), a.level, a.out_scale)
    gh, gw = xyz.shape[1:]
    oh, ow = out_shape((gh, gw), scale)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_np = xyz[2] != -1
    grid = torch.from_numpy(np.nan_to_num(xyz) / cdiv).to(dev)[None]  # (1,3,gh,gw), level voxels
    gnrm = None
    if a.normal_source == "grid":  # normals from the coarse mesh, then interpolated
        # a cell beside a hole has no usable normal, so it is not rendered (normal_ok)
        valid_np = normal_ok(valid_np)
        gnrm = grid_normals(grid[0], torch.from_numpy(valid_np).to(dev), a.normal_smooth)
    valid = torch.from_numpy(valid_np.astype(np.float32)).to(dev)[None, None]
    ns = a.num_slices
    offs = (
        torch.arange(ns, device=dev, dtype=torch.float32) - (ns - 1) / 2 + a.offset_shift
    ) * a.slice_step

    vol = Vol(a.volume, a.level)
    store = zarr.storage.LocalStore(str(a.out))
    root = zarr.open_group(store=store, mode="w", zarr_format=2)
    arr = root.create_array(
        "0",
        shape=(ns, oh, ow),
        chunks=(ns, 128, 128),
        dtype="u1",
        compressors=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
    )
    print(f"{a.mesh.name}: grid {gh}x{gw} -> {ns}x{oh}x{ow} on {dev}", flush=True)

    def block_geometry(y0, y1, x0, x1):
        """(sample points, valid mask) for one output block, or None if nothing valid."""
        bh, bw = y1 - y0, x1 - x0
        if True:
            # villa maps output pixel p to mesh coordinate p * scale (not the pixel centre):
            # using centres put our render half a grid cell (10 px) off, at ncc 0.999.
            gy = torch.arange(y0, y1, device=dev, dtype=torch.float32) * scale
            gx = torch.arange(x0, x1, device=dev, dtype=torch.float32) * scale
            ny = gy * 2 / (gh - 1) - 1
            nx = gx * 2 / (gw - 1) - 1
            gg = torch.stack(torch.meshgrid(ny, nx, indexing="ij")[::-1], -1)[None]  # (1,bh,bw,2)
            pos = F.grid_sample(grid, gg, mode="bilinear", align_corners=True)[0]  # (3,bh,bw)
            vmask = F.grid_sample(valid, gg, mode="bilinear", align_corners=True)[0, 0] > 0.999
            if not bool(vmask.any()):
                return None
            # normals from the sampled surface (grid spacing is one output pixel)
            if gnrm is not None:
                nrm = F.grid_sample(gnrm, gg, mode="bilinear", align_corners=True)[0]
            else:
                dv = torch.gradient(pos, dim=1)[0]
                du = torch.gradient(pos, dim=2)[0]
                nrm = torch.cross(du, dv, dim=0)
                # differences reach the neighbouring pixels: next to the mask edge those hold
                # positions interpolated from invalid cells, so keep only pixels whose
                # neighbours are valid too (same rule as the grid path, normal_ok)
                vmask = torch.from_numpy(normal_ok(vmask.cpu().numpy())).to(dev)
                if not bool(vmask.any()):
                    return None
            nrm = nrm / (nrm.norm(dim=0, keepdim=True) + 1e-6)
            if a.flip_normals:
                nrm = -nrm
            pts = pos[None] + offs[:, None, None, None] * nrm[None]  # (ns,3,bh,bw) in x,y,z
            return pts, vmask[None].expand(ns, bh, bw)

    blocks = [
        (y0, y1, x0, x1)
        for y0, y1 in block_ranges(oh, a.block)
        for x0, x1 in block_ranges(ow, a.block)
    ]
    # Pass 1 collects the chunks to prefetch. The sample points are DROPPED as soon as each
    # block's bounds are read: holding them all needs (ns x 3 x oh x ow) floats, which is
    # 121 GiB for a 1660x174840 ribbon and OOMs a 32 GB card at every block size (total
    # geometry scales with output area, not block size). Pass 2 recomputes per block, which
    # is a cheap grid_sample. --keep-geometry restores the old single-pass behaviour.
    geom = {}
    need = set()
    live = []
    for b in blocks:
        g = block_geometry(*b)
        if g is None:
            continue
        live.append(b)
        pts, m = g
        pz, py, px = pts[:, 2], pts[:, 1], pts[:, 0]
        need.update(
            chunk_ids_for_bbox(
                float(pz[m].min()),
                float(pz[m].max()),
                float(py[m].min()),
                float(py[m].max()),
                float(px[m].min()),
                float(px[m].max()),
            )
        )
        if a.keep_geometry:
            geom[b] = g
        else:
            del pts, m, g
    if a.prefetch_workers and need:
        import time as _t

        t0 = _t.time()
        n = prefetch_chunks(vol, need, a.prefetch_workers)
        print(f"  prefetched {n} chunks in {_t.time() - t0:.0f}s", flush=True)

    # A curved or steep mesh can make one block's bounding box huge (a 13.7 GiB request on a
    # raw 1447 mesh, 2026-09-25), so blocks over the voxel cap are split into quadrants.
    work = list(reversed(live))
    while work:
        b = work.pop()
        g = geom.pop(b, None) if a.keep_geometry else None
        if g is None:  # not kept, or a quadrant of a split block
            g = block_geometry(*b)
        if g is not None:
            y0, y1, x0, x1 = b
            pts, m = g
            pz, py, px = pts[:, 2], pts[:, 1], pts[:, 0]
            z0 = int(torch.floor(pz[m].min()).item()) - 1
            z1b = int(torch.ceil(pz[m].max()).item()) + 2
            yy0 = int(torch.floor(py[m].min()).item()) - 1
            yy1 = int(torch.ceil(py[m].max()).item()) + 2
            xx0 = int(torch.floor(px[m].min()).item()) - 1
            xx1 = int(torch.ceil(px[m].max()).item()) + 2
            vox = (z1b - z0) * (yy1 - yy0) * (xx1 - xx0)
            if vox > a.max_slab_voxels and (y1 - y0 > 1 or x1 - x0 > 1):
                del pts, m, g
                work.extend(reversed(split_block(b)))
                continue
            slab = vol.read(z0, z1b, yy0, yy1, xx0, xx1)
            t = torch.from_numpy(slab).to(dev).float()[None, None]
            del slab
            dz, dy, dx = t.shape[2:]
            gz = ((pz - z0) * 2 + 1) / dz - 1
            gyy = ((py - yy0) * 2 + 1) / dy - 1
            gxx = ((px - xx0) * 2 + 1) / dx - 1
            samp = F.grid_sample(
                t,
                torch.stack([gxx, gyy, gz], -1)[None],
                mode="bilinear",
                align_corners=False,
                padding_mode="zeros",
            )[0, 0]  # (ns,bh,bw)
            samp = torch.where(m, samp, torch.zeros_like(samp))
            arr[:, y0:y1, x0:x1] = samp.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
            del pts, m, g, t, samp
    print(f"  {len(live)}/{len(blocks)} blocks rendered", flush=True)

    if a.compare:
        ref = zarr.open(str(a.compare), mode="r")
        ref = ref["0"] if isinstance(ref, zarr.Group) else ref
        ours = arr[:]
        r = np.asarray(ref[:])
        rs = []
        for i in range(0, ns, max(1, ns // 6)):
            m2 = (r[i] > 0) & (ours[i] > 0)
            rs.append(
                round(float(np.corrcoef(r[i][m2], ours[i][m2])[0, 1]), 4)
                if m2.sum() > 1000
                else None
            )
        print(
            f"compare {a.compare.name}: shapes {r.shape} vs {ours.shape}; per-slice r {rs}",
            flush=True,
        )
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
