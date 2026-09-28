#!/usr/bin/env python3
"""Two-colour the recto sheet labels of each cube: odd sheets -> 1, even sheets -> 2.

Sheet merging is a precision error that no published topology loss targets (vesuvius.md LOG
2026-09-14 21:20). Colouring alternate sheets differently inside each training cube turns "two
adjacent sheets fused into one" into a class-exclusion violation that a loss can see (Topological
Interaction loss, Gupta et al. ECCV 2022). The colouring only has to be consistent within a cube.

Method: 26-connected components of the binary label are the sheet pieces. Two pieces are adjacent
when the Voronoi regions of their voxels touch within `--dist` voxels of both (one distance
transform per cube). The adjacency graph is 2-coloured by BFS; when a cycle is odd the edge with
the largest gap in that cycle is dropped (counted and reported). Components below `--min-vox`
voxels take the colour of their nearest large neighbour.

usage: m7_parity_labels.py LABELS_DIR OUT_DIR [--dist 14] [--min-vox 500] [--workers 8]
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

STRUCT = np.ones((3, 3, 3), bool)


def adjacency(comp: np.ndarray, ncomp: int, dist: float) -> dict[tuple[int, int], float]:
    """Pairs of component ids whose Voronoi cells meet within `dist` voxels of both, with the gap."""
    d, idx = ndi.distance_transform_edt(comp == 0, return_indices=True)
    nearest = comp[idx[0], idx[1], idx[2]]
    close = d <= dist / 2
    edges: dict[tuple[int, int], float] = {}
    for ax in range(3):
        a = np.take(nearest, np.arange(0, comp.shape[ax] - 1), axis=ax)
        b = np.take(nearest, np.arange(1, comp.shape[ax]), axis=ax)
        ca = np.take(close, np.arange(0, comp.shape[ax] - 1), axis=ax)
        cb = np.take(close, np.arange(1, comp.shape[ax]), axis=ax)
        da = np.take(d, np.arange(0, comp.shape[ax] - 1), axis=ax)
        db = np.take(d, np.arange(1, comp.shape[ax]), axis=ax)
        sel = (a != b) & ca & cb
        pa, pb, gap = a[sel], b[sel], da[sel] + db[sel]
        lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
        key = lo.astype(np.int64) * (ncomp + 1) + hi
        for k in np.unique(key):
            m = key == k
            g = float(np.median(gap[m]))
            pair = (int(k // (ncomp + 1)), int(k % (ncomp + 1)))
            edges[pair] = min(edges.get(pair, 1e9), g)
    return edges


def two_colour(n: int, edges: dict[tuple[int, int], float]) -> tuple[np.ndarray, int]:
    """BFS 2-colouring; conflicting edges (odd cycles) are dropped largest-gap first."""
    nb: dict[int, set[int]] = {i: set() for i in range(1, n + 1)}
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    colour = np.zeros(n + 1, np.int8)
    dropped = 0
    for s in range(1, n + 1):
        if colour[s]:
            continue
        colour[s] = 1
        q = deque([s])
        while q:
            u = q.popleft()
            for v in sorted(nb[u], key=lambda v: edges[(min(u, v), max(u, v))]):
                if not colour[v]:
                    colour[v] = 3 - colour[u]
                    q.append(v)
                elif colour[v] == colour[u]:
                    nb[u].discard(v)
                    nb[v].discard(u)
                    dropped += 1
    return colour, dropped


def colour_cube(args: tuple[Path, Path, float, int]) -> str:
    src, dst, dist, min_vox = args
    lab = tifffile.imread(src) > 0
    comp, n = ndi.label(lab, structure=STRUCT)
    if n == 0:
        tifffile.imwrite(dst, np.zeros(lab.shape, np.uint8), compression="zlib")
        return f"{src.name}: empty"
    sizes = np.bincount(comp.ravel())
    small = np.nonzero(sizes < min_vox)[0]
    small = small[small > 0]
    edges = adjacency(comp, n, dist)
    big_edges = {
        e: g for e, g in edges.items() if sizes[e[0]] >= min_vox and sizes[e[1]] >= min_vox
    }
    colour, dropped = two_colour(n, big_edges)
    # small pieces: colour of the nearest big component they are adjacent to, else odd
    for s in small:
        cands = [(g, e[0] if e[1] == s else e[1]) for e, g in edges.items() if s in e]
        cands = [(g, o) for g, o in cands if sizes[o] >= min_vox]
        colour[s] = 3 - colour[min(cands)[1]] if cands else 1
    out = colour[comp].astype(np.uint8)
    tifffile.imwrite(dst, out, compression="zlib")
    return f"{src.name}: {n} pieces, {len(big_edges)} edges, {dropped} dropped"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--dist",
        type=float,
        default=14.0,
        help="max gap (voxels) that counts as adjacent",
    )
    ap.add_argument("--min-vox", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (p, a.out / p.name, a.dist, a.min_vox)
        for p in sorted(a.labels.glob("*.tif"))
        if not (a.out / p.name).exists()
    ]
    print(f"{len(jobs)} cubes to colour", flush=True)
    dropped = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, msg in enumerate(ex.map(colour_cube, jobs, chunksize=4)):
            dropped += int(msg.rsplit(" ", 2)[-2]) if "dropped" in msg else 0
            if i % 100 == 0:
                print(i, msg, flush=True)
    print(f"done; {dropped} edges dropped for odd cycles", flush=True)


if __name__ == "__main__":
    sys.exit(main())
