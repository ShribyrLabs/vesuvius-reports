#!/usr/bin/env python3
"""Build self-distillation cubes for an orientation-robust m7.

For each scan, sample 192^3 cubes fully inside the papyrus, run m7 once on the cube as scanned
and once each with z<->y and z<->x swapped, and keep the cube only when the as-scanned run is
clearly the most decisive (fewest voxels with 0.1 < p < 0.9). Those cubes have sheets m7 already
reads; training on random axis permutations of them, with the teacher map permuted the same way,
teaches the student every orientation without changing what "surface" means. The saved
teacher map for a kept cube is the as-scanned run averaged with three single-axis mirrors.

usage: m7_distill_data.py VOLUMES_TXT OUT_DIR [--per-volume 40] [--seed 0]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

ORIENTS = ((0, 1, 2), (1, 0, 2), (2, 1, 0))
HOST = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
CUBE = 192
spec = importlib.util.spec_from_file_location("mo", Path(__file__).with_name("m7_orient.py"))
mo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mo)


def uncertain_fraction(p: np.ndarray) -> float:
    """Share of voxels the model is undecided about."""
    return float(((p > 0.1) & (p < 0.9)).mean())


def keep_cube(
    u_native: float, u_swaps: list[float], max_u: float = 0.3, margin: float = 0.85
) -> bool:
    """True when the as-scanned run is decisive and clearly more decisive than every swapped run."""
    return u_native < max_u and all(u_native < margin * u for u in u_swaps)


def read_retry(fn, tries: int = 6, wait: float = 5.0):
    """Call fn, retrying transient network failures with a growing pause."""
    import time

    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # aiohttp/zarr raise several unrelated types for dropped links
            if i == tries - 1:
                raise
            print(f"  read failed ({type(exc).__name__}), retry {i + 1}", flush=True)
            time.sleep(wait * (i + 1))


def tta_probs(net, vol: np.ndarray, props: dict) -> np.ndarray:
    """m7 probability averaged over the identity and three single-axis mirrors."""
    acc = mo.predict_probs(net, vol, props)
    for ax in range(3):
        acc += np.flip(mo.predict_probs(net, np.ascontiguousarray(np.flip(vol, ax)), props), ax)
    return acc / 4.0


def main() -> None:
    import zarr

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("volumes")
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-volume", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--best-orientation",
        action="store_true",
        help="teacher from whichever of the three orientations is clearly most decisive (not only "
        "the as-scanned one), so crushed regions contribute rotated-teacher targets",
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    net, props = mo.load_net()
    for line in Path(a.volumes).read_text().split():
        vol_url = HOST + line.rstrip("/")
        name = line.split("/")[0]
        l3 = read_retry(
            lambda u=vol_url: np.asarray(zarr.open_array(u + "/5", mode="r", zarr_format=2)[:])
        )
        inside = np.argwhere(l3 > 0)
        l0 = zarr.open_array(vol_url + "/0", mode="r", zarr_format=2)
        kept = len(list(a.out.glob(f"{name}_z*.npz")))  # resume: cubes already built count
        tried = 0
        while kept < a.per_volume and tried < a.per_volume * 6:
            tried += 1
            cz, cy, cx = inside[rng.integers(len(inside))] * 32
            z0, y0, x0 = (max(0, int(c) - CUBE // 2) for c in (cz, cy, cx))
            if any(s + CUBE > n for s, n in zip((z0, y0, x0), l0.shape, strict=True)):
                continue
            # cheap pre-check on the 32x-downsampled level before streaming 192^3 at level 0
            s3 = l3[
                z0 // 32 : -(-(z0 + CUBE) // 32),
                y0 // 32 : -(-(y0 + CUBE) // 32),
                x0 // 32 : -(-(x0 + CUBE) // 32),
            ]
            if (s3 == 0).mean() > 0.0:
                continue
            cube = read_retry(
                lambda a=l0, z=z0, y=y0, x=x0: np.asarray(
                    a[z : z + CUBE, y : y + CUBE, x : x + CUBE]
                )
            )
            if (cube == 0).mean() > 0.02:
                continue
            us = [uncertain_fraction(mo.predict_probs(net, cube, props, pm)) for pm in ORIENTS]
            best = int(np.argmin(us)) if a.best_orientation else 0
            u_nat, u_swaps = us[best], [u for i, u in enumerate(us) if i != best]
            ok = keep_cube(u_nat, u_swaps)
            print(
                f"{name} z{z0} y{y0} x{x0} u {us[0]:.3f} zy {us[1]:.3f} zx {us[2]:.3f} best {best} "
                f"{'KEEP' if ok else 'skip'}",
                flush=True,
            )
            if ok:
                pm = ORIENTS[best]
                p_native = tta_probs(net, np.ascontiguousarray(cube.transpose(pm)), props)
                p_native = p_native.transpose(np.argsort(pm))
                np.savez_compressed(
                    a.out / f"{name}_z{z0}_y{y0}_x{x0}.npz",
                    img=cube,
                    teacher=(p_native * 255).astype(np.uint8),
                    orientation=np.array(ORIENTS[best]),
                )
                kept += 1
        print(f"== {name}: kept {kept} of {tried}", flush=True)


if __name__ == "__main__":
    main()
