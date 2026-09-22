#!/usr/bin/env python3
"""Fine-tune m7 on the seg-derived recto surface labels with random axis permutations and flips.

m7 was trained with nnU-Net's default augmentation (no 90-degree axis swaps), and it misses sheets
that lie across the scan axis. Here each step draws a labelled cube, takes a random 192^3 crop
(reflect-padding smaller cubes, with the padding excluded from the loss), permutes and flips the
axes, and trains on cross-entropy + Dice. Cubes that overlap any held-out evaluation cube are
dropped by bounding box before training.

usage: m7_label_train.py DATA_DIR OUT_DIR --holdout NAMES_TXT --holdout-dir DIR [--steps 6000]
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mo = _load("m7_orient")
dt = _load("m7_distill_train")
sd = _load("surface_dice")
CROP = 192
ROT_CROP = 224  # rotate a larger crop on the GPU, then centre-crop to CROP
NAME = re.compile(r"(s\d+)_z(\d+)_y(\d+)_x(\d+)")


def cube_box(name: str, shape: tuple[int, int, int]) -> tuple[str, np.ndarray, np.ndarray]:
    """(scroll, lower corner zyx, upper corner zyx) from a dataset file name and its array shape."""
    m = NAME.search(name)
    lo = np.array([int(m.group(2)), int(m.group(3)), int(m.group(4))])
    return m.group(1), lo, lo + np.array(shape)


def overlaps(a: tuple[str, np.ndarray, np.ndarray], b: tuple[str, np.ndarray, np.ndarray]) -> bool:
    """True when two cube boxes are on the same scroll and share any voxel."""
    return a[0] == b[0] and bool(np.all(a[1] < b[2]) and np.all(b[1] < a[2]))


def tif_shape(path: Path) -> tuple[int, int, int]:
    with tifffile.TiffFile(path) as t:
        return tuple(t.series[0].shape)


def crop_pad(
    img: np.ndarray, lab: np.ndarray, rng: np.random.Generator, crop: int = CROP
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random CROP^3 window; axes shorter than CROP are reflect-padded and masked out of the loss."""
    mask = np.ones(img.shape, np.float32)
    pads = [(0, max(0, crop - s)) for s in img.shape]
    if any(p[1] for p in pads):
        img = np.pad(img, pads, mode="reflect")
        lab = np.pad(lab, pads)
        mask = np.pad(mask, pads)
    o = [rng.integers(0, s - crop + 1) for s in img.shape]
    sl = tuple(slice(k, k + crop) for k in o)
    return img[sl], lab[sl], mask[sl]


class LabelStream(torch.utils.data.IterableDataset):
    def __init__(self, pairs: list[tuple[str, str]], props: dict, seed: int, crop: int = CROP):
        self.pairs, self.props, self.seed, self.crop = pairs, props, seed, crop

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 1000 + int(time.time()))
        while True:
            ip, lp = self.pairs[rng.integers(len(self.pairs))]
            img, lab, mask = crop_pad(tifffile.imread(ip), tifffile.imread(lp), rng, self.crop)
            perm = dt.PERMS[rng.integers(len(dt.PERMS))]
            img, lab, mask = (a.transpose(perm) for a in (img, lab, mask))
            for ax in range(3):
                if rng.random() < 0.5:
                    img, lab, mask = (np.flip(a, ax) for a in (img, lab, mask))
            x = torch.from_numpy(mo.ct_normalize(np.ascontiguousarray(img), self.props))[None]
            yield (
                x,
                torch.from_numpy(np.ascontiguousarray(lab).astype(np.float32)),
                torch.from_numpy(np.ascontiguousarray(mask)),
            )


def ti_loss(
    logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor, d: int
) -> tuple[torch.Tensor, float]:
    """Topological Interaction loss (Gupta et al., ECCV 2022) for the odd/even sheet classes.

    Critical voxels are where the hard odd prediction lies within `d` voxels of the hard even
    prediction (or vice versa); the loss is the cross-entropy restricted to them, so the network is
    pushed to resolve every place two differently coloured sheets touch. Returns (loss, share of
    valid voxels that are critical)."""
    import torch.nn.functional as F

    hard = logits.argmax(1)
    odd, even = (hard == 1).float()[:, None], (hard == 2).float()[:, None]
    k = 2 * d + 1
    crit = (F.max_pool3d(odd, k, 1, d) * even + F.max_pool3d(even, k, 1, d) * odd).clamp(max=1)
    crit = crit[:, 0] * m
    n = crit.sum()
    if n < 1:
        return logits.sum() * 0.0, 0.0
    ce = -torch.log_softmax(logits, 1).gather(1, y[:, None])[:, 0]
    return (ce * crit).sum() / n, float(n / m.sum().clamp(min=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument(
        "--holdout-dir", type=Path, required=True, help="labelsTr of the held-out cubes"
    )
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument(
        "--exclude-scroll", action="append", default=[], help="e.g. s4: leave a scroll out"
    )
    ap.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="probability of an extra arbitrary-angle 3D rotation per batch (MemBrain-seg recipe)",
    )
    ap.add_argument(
        "--surface-dice",
        type=float,
        default=0.0,
        help="weight of the Surface-Dice (soft-skeleton) loss term",
    )
    ap.add_argument(
        "--parity",
        type=Path,
        default=None,
        help="dir of parity-coloured labels (0 bg, 1 odd sheets, 2 even; scripts/m7_parity_labels.py): "
        "train a 3-class head so adjacent sheets are different classes",
    )
    ap.add_argument(
        "--ti",
        type=float,
        default=0.0,
        help="weight of the Topological Interaction (odd/even exclusion) loss",
    )
    ap.add_argument("--ti-d", type=int, default=2, help="exclusion distance in voxels for --ti")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    held = [cube_box(n, tif_shape(a.holdout_dir / n)) for n in a.holdout.read_text().split()]
    pairs, dropped = [], 0
    for lp in sorted((a.data / "labelsTr").glob("*.tif")):
        ip = a.data / "imagesTr" / (lp.stem + "_0000.tif")
        if not ip.exists():
            continue
        if lp.name.split("_")[0] in a.exclude_scroll:
            continue
        box = cube_box(lp.name, tif_shape(lp))
        if any(overlaps(box, h) for h in held):
            dropped += 1
            continue
        if a.parity:
            lp = a.parity / lp.name
            if not lp.exists():
                continue
        pairs.append((str(ip), str(lp)))
    print(
        f"{len(pairs)} training cubes, {dropped} dropped for overlapping held-out cubes",
        flush=True,
    )
    net, props = mo.load_net(num_classes=3 if a.parity else 2)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    loader = iter(
        torch.utils.data.DataLoader(
            LabelStream(pairs, props, a.seed, ROT_CROP if a.rotate > 0 else CROP),
            batch_size=a.batch,
            num_workers=6,
            pin_memory=True,
        )
    )
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = dt.lr_at(step, a.steps, a.lr)
        x, y, m = (v.cuda(non_blocking=True) for v in next(loader))
        if a.rotate > 0:
            if rng.random() < a.rotate:
                rot = sd.random_rotation_matrix(rng)
                x = sd.rotate_batch(x, rot, nearest=False)
                y = sd.rotate_batch(y[:, None], rot, nearest=True)[:, 0]
                m = sd.rotate_batch(m[:, None], rot, nearest=True)[:, 0]
            o = (ROT_CROP - CROP) // 2
            x, y, m = (t[..., o : o + CROP, o : o + CROP, o : o + CROP] for t in (x, y, m))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(x)
        logp = torch.log_softmax(logits.float(), 1)
        if a.parity:
            yi = y.long()
            fg = (y > 0).float()
            ce = -(logp.gather(1, yi[:, None])[:, 0] * m).sum() / m.sum().clamp(min=1)
            p = (1 - logp[:, 0].exp()) * m
        else:
            fg = y
            ce = -((y * logp[:, 1] + (1 - y) * logp[:, 0]) * m).sum() / m.sum().clamp(min=1)
            p = logp[:, 1].exp() * m
        dice = 1 - (2 * (p * fg).sum() + 1) / (p.sum() + (fg * m).sum() + 1)
        loss = ce + dice
        if a.parity and a.ti > 0:
            ti, crit_frac = ti_loss(logits.float(), yi, m, a.ti_d)
            loss = loss + a.ti * ti
        if a.surface_dice > 0:
            sdl = sd.surface_dice_loss(p[:, None], fg[:, None], m[:, None])
            loss = loss + a.surface_dice * sdl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12.0)
        opt.step()
        if step % 50 == 0:
            print(
                f"step {step} loss {loss.item():.4f} ce {ce.item():.4f} dice {dice.item():.4f} "
                + (f"sdice {sdl.item():.4f} " if a.surface_dice > 0 else "")
                + (f"ti {ti.item():.4f} crit {crit_frac:.2e} " if a.parity and a.ti > 0 else "")
                + f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s",
                flush=True,
            )
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            torch.save(
                {"network_weights": net.state_dict()},
                a.out / f"step_{step + 1:06d}.pth",
            )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
