#!/usr/bin/env python3
"""Build dense pseudo-labels for one PHerc0139 segment at native 9.362 um.

Steps: fetch the native surface volume (level 0) and the published 2.399 um ink
prediction TIFF; register the native render to level 2 of the 2.399 um surface volume
(scale = canvas ratio, shift = median of tile cross-correlation peaks); project the key
onto the native canvas; write ink_9um-style label zarrs (uint8 0/255 on plane depth//2).

Output layout (what vesuvius.ink_detection.training expects):
  <out>/<name>/surface-volume.zarr/0            native [28,H,W]
  <out>/<name>/<name>_inklabels.zarr/0          key >= --ink-thr
  <out>/<name>/<name>_supervision_mask.zarr/0   papyrus present and key outside the
                                                ambiguous band [--bg-thr, --ink-thr)
  <out>/<name>/<name>_validation_mask.zarr/0    (only with --held-out) = supervision mask
  <out>/<name>/reg.json, build.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import s3fs
import tifffile
import zarr
from numcodecs import Blosc
from numpy.fft import fft2, ifft2
from scipy.ndimage import map_coordinates, zoom

HOST = "vesuvius-challenge-open-data.s3.amazonaws.com"
BUCKET = "vesuvius-challenge-open-data"
SCROLL = "PHerc0139"
NATIVE_VOL = "9.362um-1.2m-113keV-volume-20250728140407"
KEY_VOL = "2.399um-0.22m-78keV-volume-20260102150214"  # gitleaks:allow (volume name, not a key)
ROOT = Path(__file__).resolve().parents[1]
FETCHER = ROOT / "scripts" / "fetch_control_crop.py"


def get(key: str) -> bytes:
    for attempt in range(8):
        try:
            if attempt:
                time.sleep(min(2**attempt, 30))
            c = http.client.HTTPSConnection(HOST, timeout=60)
            c.request("GET", "/" + key)
            r = c.getresponse()
            b = r.read()
            c.close()
            if r.status == 200:
                return b
            if r.status == 404:
                return b""
        except Exception as e:  # noqa: BLE001
            print(f"retry {attempt+1} {key}: {e}", file=sys.stderr)
    raise RuntimeError(key)


def ncc_shift(a: np.ndarray, b: np.ndarray, maxs: int) -> tuple[int, int, float]:
    """(dy, dx, peak r) such that a[y, x] ~ b[y - dy, x - dx]."""
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    cc = np.fft.fftshift(np.real(ifft2(fft2(a * w) * np.conj(fft2(b * w)))))
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    win = cc[cy - maxs : cy + maxs + 1, cx - maxs : cx + maxs + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    return int(iy - maxs), int(ix - maxs), float(win[iy, ix] / w.sum())


def register(native, sv24_prefix: str, n: int = 4, margin: int = 96) -> dict:
    """Fit native = ratio * level2 + shift from tile cross-correlations."""
    H, W = native.shape[1:]
    za = json.loads(get(f"{sv24_prefix}/2/.zarray"))
    D, H2, W2 = za["shape"]
    ch = za["chunks"]
    sy, sx = H / H2, W / W2
    z24 = (D // 2 - 8, D // 2 + 8)
    zn = (native.shape[0] // 2 - 3, native.shape[0] // 2 + 4)
    # Candidate 2x2-chunk tiles ranked by native coverage (segments can be ragged patches
    # on a mostly empty canvas, so a fixed grid finds nothing); keep the n*n fullest.
    mid = native[native.shape[0] // 2] > 0
    cand = []
    for cy in range(1, H2 // ch[1] - 2):
        for cx in range(1, W2 // ch[2] - 2):
            y0, x0 = int(cy * ch[1] * sy), int(cx * ch[2] * sx)
            y1, x1 = int((cy + 2) * ch[1] * sy), int((cx + 2) * ch[2] * sx)
            cov = float(mid[y0:y1, x0:x1].mean()) if y1 <= H and x1 <= W else 0.0
            if cov >= 0.9:
                cand.append((cov, cy, cx))
    cand.sort(reverse=True)
    tiles = []
    drop = {"key_fill": 0, "bounds": 0, "native_fill": 0}
    for _, cy, cx in cand[: n * n]:
        tile = np.zeros((D, 2 * ch[1], 2 * ch[2]), np.uint8)
        for dy in range(2):
            for dx in range(2):
                b = get(f"{sv24_prefix}/2/0/{cy+dy}/{cx+dx}")
                if b:
                    tile[:, dy * ch[1] : (dy + 1) * ch[1], dx * ch[2] : (dx + 1) * ch[2]] = (
                        np.frombuffer(b, np.uint8).reshape(ch)
                    )
        t2 = tile[z24[0] : z24[1]].mean(0)
        if (t2 > 0).mean() < 0.9:
            drop["key_fill"] += 1
            continue
        y2c, x2c = (cy + 1) * ch[1], (cx + 1) * ch[2]
        t2z = zoom(t2, (sy, sx), order=1)
        hh, ww = t2z.shape
        ync, xnc = y2c * sy, x2c * sx
        y0 = int(round(ync - hh / 2)) - margin
        x0 = int(round(xnc - ww / 2)) - margin
        if y0 < 0 or x0 < 0 or y0 + hh + 2 * margin > H or x0 + ww + 2 * margin > W:
            drop["bounds"] += 1
            continue
        a = native[zn[0] : zn[1], y0 : y0 + hh + 2 * margin, x0 : x0 + ww + 2 * margin]
        a = a.astype(np.float32).mean(0)
        if (a > 0).mean() < 0.9:
            drop["native_fill"] += 1
            continue
        bpad = np.zeros_like(a)
        bpad[margin : margin + hh, margin : margin + ww] = t2z
        dy, dx, r = ncc_shift(a, bpad, margin - 4)
        tiles.append([int(cy), int(cx), dy, dx, r])
    print(f"register: {len(cand)} candidates, {len(tiles)} correlated, dropped {drop}", flush=True)
    return consensus({"ay": sy, "ax": sx, "H": H, "W": W, "H2": H2, "W2": W2, "tiles": tiles})


def consensus(reg: dict, min_r: float = 0.12, tol: float = 3.0, min_tiles: int = 4) -> dict:
    """Shift = median of agreeing tiles; ok when at least min_tiles agree within tol px."""
    cand = [t for t in reg["tiles"] if t[4] >= min_r]
    good: list = []
    if len(cand) >= min_tiles:
        my = float(np.median([t[2] for t in cand]))
        mx = float(np.median([t[3] for t in cand]))
        good = [t for t in cand if abs(t[2] - my) <= tol and abs(t[3] - mx) <= tol]
    if len(good) >= min_tiles:
        by = float(np.median([t[2] for t in good]))
        bx = float(np.median([t[3] for t in good]))
        spread = max(max(abs(t[2] - by) for t in good), max(abs(t[3] - bx) for t in good))
    else:
        by = bx = 0.0
        spread = float("inf")
    reg.update(
        {
            "by": by,
            "bx": bx,
            "n_good": len(good),
            "spread_px": spread,
            "ok": bool(len(good) >= min_tiles),
        }
    )
    return reg


def project_key(key: np.ndarray, reg: dict) -> np.ndarray:
    """Average the key 4x4 to the level-2 canvas, then resample to the native canvas."""
    Hk, Wk = (key.shape[0] // 4) * 4, (key.shape[1] // 4) * 4
    k2 = key[:Hk, :Wk].reshape(Hk // 4, 4, Wk // 4, 4).mean((1, 3), dtype=np.float32)
    H, W = reg["H"], reg["W"]
    y2 = (np.arange(H, dtype=np.float32) - reg["by"]) / reg["ay"]
    x2 = (np.arange(W, dtype=np.float32) - reg["bx"]) / reg["ax"]
    out = np.zeros((H, W), np.float32)
    for y0 in range(0, H, 512):
        gy, gx = np.meshgrid(y2[y0 : y0 + 512], x2, indexing="ij")
        out[y0 : y0 + 512] = map_coordinates(k2, [gy, gx], order=1, mode="constant", cval=-1.0)
    return out  # -1 where the key canvas does not cover


def write_label_zarr(path: Path, plane: np.ndarray, depth: int) -> None:
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    arr = g.create_array(
        "0",
        shape=(depth, *plane.shape),
        chunks=(depth, 128, 128),
        dtype="u1",
        compressors=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        config={"write_empty_chunks": False},
    )
    arr[depth // 2] = plane


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "segment", help="S3 segment directory name, e.g. 20260317000000-w035_2026031718"
    )
    ap.add_argument("--name", help="short name (default: the wNNN token)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ink_pseudo")
    ap.add_argument("--ink-thr", type=float, default=64.0)
    ap.add_argument("--bg-thr", type=float, default=48.0)
    ap.add_argument(
        "--held-out", action="store_true", help="also write a validation mask = supervision"
    )
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--scroll", default=SCROLL, help="bucket scroll dir (default PHerc0139)")
    ap.add_argument(
        "--native-vol", default=NATIVE_VOL, help="native surface-volume name (no .zarr)"
    )
    ap.add_argument(
        "--key-vol", default=KEY_VOL, help="high-res key surface-volume name (no .zarr)"
    )
    a = ap.parse_args()

    name = a.name or next(t for t in a.segment.replace("-", "_").split("_") if t.startswith("w"))
    d = a.out / name
    d.mkdir(parents=True, exist_ok=True)
    seg = f"{a.scroll}/segments/{a.segment}"
    sv_native = f"{seg}/surface-volumes/{a.native_vol}.zarr"
    sv_key = f"{seg}/surface-volumes/{a.key_vol}.zarr"
    fs = s3fs.S3FileSystem(anon=True)
    keys = [
        p for p in fs.ls(f"{BUCKET}/{seg}/ink-detection") if a.key_vol in p and p.endswith(".tif")
    ]
    if (
        len(keys) > 1
    ):  # some segments carry several models' maps; keep the recipe used everywhere else
        keys = [p for p in keys if "new_canon_autoresearch" in p] or keys[:1]
    if len(keys) != 1:
        raise SystemExit(f"{name}: expected one 2.4um key, found {keys}")
    key_tif = d / "key_2p4um.tif"
    sv_local = d / "surface-volume.zarr"
    if not a.skip_fetch:
        subprocess.run(
            [
                sys.executable,
                str(FETCHER),
                "--zarr-prefix",
                sv_native,
                "--level",
                "0",
                "--full",
                "--out",
                str(sv_local),
            ],
            check=True,
        )
        if not key_tif.exists():
            fs.get(keys[0], str(key_tif))
    native = zarr.open(str(sv_local), mode="r")["0"]
    depth = native.shape[0]
    if (d / "reg.json").exists():
        reg = consensus(json.loads((d / "reg.json").read_text()))
    else:
        reg = register(native, sv_key)
    (d / "reg.json").write_text(json.dumps(reg, indent=1))
    print(
        f"{name}: registration ok={reg['ok']} good tiles={reg['n_good']} "
        f"shift=({reg['by']:.1f},{reg['bx']:.1f}) spread={reg['spread_px']:.1f}px"
    )
    if not reg["ok"]:
        return 2
    key = tifffile.imread(key_tif)
    proj = project_key(key, reg)
    zc = native.shape[0] // 2
    valid = native[zc - 3 : zc + 4].astype(np.float32).mean(0) > 0
    valid &= proj >= 0
    ink = valid & (proj >= a.ink_thr)
    sup = valid & ((proj < a.bg_thr) | (proj >= a.ink_thr))
    write_label_zarr(d / f"{name}_inklabels.zarr", ink.astype(np.uint8) * 255, depth)
    write_label_zarr(d / f"{name}_supervision_mask.zarr", sup.astype(np.uint8) * 255, depth)
    if a.held_out:
        write_label_zarr(d / f"{name}_validation_mask.zarr", sup.astype(np.uint8) * 255, depth)
    np.save(d / "key_native.npy", proj.astype(np.float16))
    info = {
        "segment": a.segment,
        "key": keys[0],
        "shape": list(native.shape),
        "valid_frac": float(valid.mean()),
        "ink_frac_of_valid": float(ink.sum() / max(valid.sum(), 1)),
        "supervised_frac_of_valid": float(sup.sum() / max(valid.sum(), 1)),
        "ink_thr": a.ink_thr,
        "bg_thr": a.bg_thr,
        "held_out": a.held_out,
    }
    (d / "build.json").write_text(json.dumps(info, indent=1))
    print(f"{name}: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
