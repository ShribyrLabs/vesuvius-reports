#!/usr/bin/env python3
"""Render a tifxyz surface from an OME-Zarr CT volume, sampling on the GPU.

Same job as villa's vc_render_tifxyz (which is CPU-only and, at high parallelism, aborts on
the first DNS hiccup because it has no retry): for every output pixel, interpolate the mesh
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


def out_shape(grid_hw: tuple[int, int], scale: float) -> tuple[int, int]:
    """Output pixels for a tifxyz grid: villa uses round(n / scale) (scale 0.05 -> 20x)."""
    return int(round(grid_hw[0] / scale)), int(round(grid_hw[1] / scale))


def block_ranges(n: int, block: int) -> list[tuple[int, int]]:
    return [(i, min(i + block, n)) for i in range(0, n, block)]


def chunk_ids_for_bbox(
    zlo: float, zhi: float, ylo: float, yhi: float, xlo: float, xhi: float, chunk: int = 128
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
    ap.add_argument("--slice-step", type=float, default=1.0)
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
    scale = float(meta["scale"][0])
    gh, gw = xyz.shape[1:]
    oh, ow = out_shape((gh, gw), scale)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid = torch.from_numpy((xyz[2] != -1).astype(np.float32)).to(dev)[None, None]
    grid = torch.from_numpy(np.nan_to_num(xyz)).to(dev)[None]  # (1,3,gh,gw)
    gnrm = None
    if a.normal_source == "grid":  # normals from the coarse mesh, then interpolated
        g = grid[0]
        gn = torch.cross(torch.gradient(g, dim=2)[0], torch.gradient(g, dim=1)[0], dim=0)
        gnrm = (gn / (gn.norm(dim=0, keepdim=True) + 1e-6))[None]
        if a.normal_smooth:
            k = a.normal_smooth
            gnrm = F.avg_pool2d(gnrm, k, stride=1, padding=k // 2, count_include_pad=False)
            gnrm = gnrm / (gnrm.norm(dim=1, keepdim=True) + 1e-6)
    ns = a.num_slices
    offs = (
        torch.arange(ns, device=dev, dtype=torch.float32) - (ns - 1) / 2 + a.offset_shift
    ) * a.slice_step

    vol = Vol(a.volume, 0)
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
        import concurrent.futures as cf
        import time as _t

        t0 = _t.time()
        with cf.ThreadPoolExecutor(a.prefetch_workers) as ex:
            list(ex.map(lambda c: vol.chunk(*c), sorted(need)))
        print(f"  prefetched {len(need)} chunks in {_t.time() - t0:.0f}s", flush=True)

    for b in live:
        g = geom[b] if a.keep_geometry else block_geometry(*b)
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
            slab = vol.read(z0, z1b, yy0, yy1, xx0, xx1)
            t = torch.from_numpy(slab.astype(np.float32)).to(dev)[None, None]
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
