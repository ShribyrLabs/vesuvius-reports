#!/usr/bin/env python3
"""Orientation-aware inference for staff's m7 surface model (scrollprize/surface_m7_nnunet).

m7 segments sheets well only when their normal lies roughly in the scan's xy plane; where crushing
has laid sheets flat (normal along z) it outputs blobs. Running it again on the volume with z and y
swapped turns those flat sheets into ones it can see. The two probability maps are blended per
voxel by how much of the local sheet normal points along z, measured from the CT's structure
tensor, so regions m7 already handles keep the original prediction.

Library use: `predict_probs(net, vol, props)` and `orientation_weight(vol)`; `merge(p0, p1, w)`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "data/checkpoints/surface_m7"
SWAP_ZY = (1, 0, 2)


def ct_normalize(x: np.ndarray, props: dict) -> np.ndarray:
    """nnU-Net CTNormalization with the dataset's intensity properties."""
    x = np.clip(x.astype(np.float32), props["percentile_00_5"], props["percentile_99_5"])
    return (x - props["mean"]) / max(props["std"], 1e-8)


def tile_starts(n: int, tile: int, overlap: int) -> list[int]:
    """Tile origins covering [0, n); the last tile is flush with the end."""
    if n <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, n - tile + 1, step))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


def orientation_weight(vol: np.ndarray, grad_sigma: float = 1.5, rho: float = 6.0) -> np.ndarray:
    """Per-voxel n_z^2 of the dominant structure-tensor eigenvector (the sheet normal), in [0, 1]."""
    v = vol.astype(np.float32)
    g = [ndi.gaussian_filter(v, grad_sigma, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    j = np.empty(v.shape + (3, 3), np.float32)
    for a in range(3):
        for b in range(a, 3):
            j[..., a, b] = j[..., b, a] = ndi.gaussian_filter(g[a] * g[b], rho)
    _, vecs = np.linalg.eigh(j)  # ascending eigenvalues; the last column is the normal
    return vecs[..., 0, -1] ** 2  # component along axis 0 (z)


def merge(p_orig: np.ndarray, p_swap: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Blend the two predictions: weight w on the z<->y-swapped run."""
    return (1.0 - w) * p_orig + w * p_swap


def load_net(ckpt: Path = CKPT, num_classes: int = 2):
    """The m7 network and its intensity properties, rebuilt from plans.json.

    `num_classes` > 2 widens the final 1x1 head (foreground weights copied to every extra class,
    bias shifted so the foreground probability is unchanged) for parity-coloured training.
    """
    import importlib

    import torch

    plans = json.loads((ckpt / "plans.json").read_text())
    arch = plans["configurations"]["3d_fullres"]["architecture"]
    mod, cls = arch["network_class_name"].rsplit(".", 1)
    kw = dict(arch["arch_kwargs"])
    for key in arch.get("_kw_requires_import", []):
        if isinstance(kw.get(key), str):
            m, c = kw[key].rsplit(".", 1)
            kw[key] = getattr(importlib.import_module(m), c)
    net = getattr(importlib.import_module(mod), cls)(
        input_channels=1, num_classes=num_classes, deep_supervision=False, **kw
    )
    ck = torch.load(ckpt / "fold_0/checkpoint_best.pth", map_location="cpu", weights_only=False)
    net.load_state_dict(widen_head(ck["network_weights"], num_classes))
    return net.cuda().eval(), plans["foreground_intensity_properties_per_channel"]["0"]


def head_classes(weights: dict) -> int:
    """Number of output classes stored in an nnU-Net state dict."""
    key = sorted(k for k in weights if ".seg_layers." in k and k.endswith(".weight"))[-1]
    return int(weights[key].shape[0])


def widen_head(weights: dict, num_classes: int) -> dict:
    """Copy a 2-class state dict into `num_classes` outputs (fg -> every extra class)."""
    import math

    import torch

    have = head_classes(weights)
    if have == num_classes:
        return weights
    assert have == 2 and num_classes > 2, (have, num_classes)
    out = dict(weights)
    for k, v in weights.items():
        if ".seg_layers." not in k:
            continue
        extra = num_classes - 1
        if k.endswith(".weight"):
            out[k] = torch.cat([v[:1]] + [v[1:2]] * extra, 0)
        elif k.endswith(".bias"):
            out[k] = torch.cat([v[:1]] + [v[1:2] - math.log(extra)] * extra, 0)
    return out


def load_variant(path):
    """Network + props for a fine-tuned checkpoint, whatever its head width."""
    import torch

    w = torch.load(path, map_location="cpu")["network_weights"]
    net, props = load_net(num_classes=head_classes(w))
    net.load_state_dict(w)
    return net.eval(), props


def predict_probs(
    net,
    vol: np.ndarray,
    props: dict,
    perm: tuple[int, int, int] = (0, 1, 2),
    tile: int = 192,
) -> np.ndarray:
    """Surface probability for vol (zyx), running the network on vol.transpose(perm).

    Volumes smaller than a tile are reflect-padded so the network never sees zero padding.
    """
    import torch

    x = vol.transpose(perm)
    pad = [(0, max(0, tile - s)) for s in x.shape]
    xp = np.pad(x, pad, mode="reflect") if any(p[1] for p in pad) else x
    ov = tile // 2
    w1 = np.hanning(tile + 2)[1:-1].astype(np.float32)
    win = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
    acc = np.zeros(xp.shape, np.float32)
    wgt = np.zeros_like(acc)
    with torch.no_grad():
        for a in tile_starts(xp.shape[0], tile, ov):
            for b in tile_starts(xp.shape[1], tile, ov):
                for c in tile_starts(xp.shape[2], tile, ov):
                    p = xp[a : a + tile, b : b + tile, c : c + tile]
                    t = torch.from_numpy(ct_normalize(p, props))[None, None].cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prob = (1.0 - torch.softmax(net(t).float(), 1)[0, 0]).cpu().numpy()
                    acc[a : a + tile, b : b + tile, c : c + tile] += prob * win
                    wgt[a : a + tile, b : b + tile, c : c + tile] += win
    out = (acc / np.maximum(wgt, 1e-6))[: x.shape[0], : x.shape[1], : x.shape[2]]
    return out.transpose(np.argsort(perm))
