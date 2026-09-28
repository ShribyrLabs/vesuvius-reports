"""Surface-Dice loss for thin-sheet segmentation (soft skeletons via min/max pooling).

Adapted from MemBrain-seg's `surface_dice.py` (teamtomo/membrain-seg, MIT), itself a modification
of clDice (Paetzold & Shit, https://github.com/jocpae/clDice, MIT). Both prediction and target are
reduced to ~1-voxel-thick soft skeletons, and the loss scores skeleton precision/recall, so training
rewards sheet connectivity (continuous, unbroken sheets) and is indifferent to thickness — thickness
is left to the CE/Dice terms. Our label fine-tunes produced dashed sheets at matched volume
(vesuvius.md LOG 2026-09-14 00:20), which is what this term targets.

Tensors are (B, 1, D, H, W). `prob` is a probability in [0, 1]; `target` is binary; `mask` marks
voxels that count.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    return -F.max_pool3d(-img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iterations: int) -> torch.Tensor:
    """Differentiable skeleton: what each round of opening removes is thin structure."""
    skel = F.relu(img - soft_open(img))
    for _ in range(iterations):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


_KERNELS: dict[tuple[int, float, str], torch.Tensor] = {}


def gaussian_blur(x: torch.Tensor, size: int = 15, sigma: float = 2.0) -> torch.Tensor:
    key = (size, sigma, str(x.device))
    if key not in _KERNELS:
        g = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
        k = torch.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma**2))
        _KERNELS[key] = (k / k.sum()).view(1, 1, size, size, size).to(x.device)
    return F.conv3d(x, _KERNELS[key], padding=size // 2)


def target_skeleton(target: torch.Tensor, iterations: int) -> torch.Tensor:
    """Skeleton of a smoothed binary target (the raw binary skeleton is patchy)."""
    return soft_skel(gaussian_blur(target.float()) * 1.5, iterations)


def surface_dice_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    iterations: int = 5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """1 - surface Dice, averaged over the batch.

    `iterations` erosions skeletonise sheets up to ~2*iterations+1 voxels thick; thicker predictions
    lose their skeleton and the precision term must not then reward them, so `smooth` stays small."""
    if mask is None:
        mask = torch.ones_like(prob)
    skel_pred = soft_skel(prob, iterations) * mask
    skel_true = target_skeleton(target, iterations) * mask
    t = target.float() * mask
    p = prob * mask
    dims = (1, 2, 3, 4)
    tprec = ((skel_pred * t).sum(dims) + smooth) / (skel_pred.sum(dims) + smooth)
    tsens = ((skel_true * p).sum(dims) + smooth) / (skel_true.sum(dims) + smooth)
    return (1.0 - 2.0 * tprec * tsens / (tprec + tsens)).mean()


def random_rotation_matrix(rng, max_deg: float = 180.0) -> torch.Tensor:
    """Uniform random axis, uniform angle in [-max_deg, max_deg]; 3x3 float tensor."""
    axis = rng.normal(size=3)
    axis /= max(float((axis**2).sum() ** 0.5), 1e-8)
    a = math.radians(rng.uniform(-max_deg, max_deg))
    kx, ky, kz = axis
    k = torch.tensor([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=torch.float32)
    return torch.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


def rotate_batch(x: torch.Tensor, rot: torch.Tensor, nearest: bool) -> torch.Tensor:
    """Rotate a (B, C, D, H, W) tensor about its centre by `rot` (3x3), reflection padding."""
    theta = torch.cat([rot, torch.zeros(3, 1)], 1)[None].to(x.device).expand(x.shape[0], 3, 4)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(
        x,
        grid,
        mode="nearest" if nearest else "bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
