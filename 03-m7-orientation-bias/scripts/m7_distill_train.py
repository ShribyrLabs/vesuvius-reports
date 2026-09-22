#!/usr/bin/env python3
"""Fine-tune m7 to be orientation-robust by distilling its own as-scanned predictions.

Each step takes a cube from `m7_distill_data.py`, applies one random axis permutation (of six) and
random flips to both the CT and the teacher map, and trains the student (initialised from m7) to
reproduce the transformed teacher: soft cross-entropy + soft Dice on the surface class.

Checkpoints keep nnU-Net's `network_weights` key, so they drop into anything that loads m7.

usage: m7_distill_train.py CUBES_DIR OUT_DIR [--steps 6000] [--lr 2e-5] [--batch 2]
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("mo", Path(__file__).with_name("m7_orient.py"))
mo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mo)
PERMS = list(itertools.permutations(range(3)))


def augment(
    img: np.ndarray, teacher: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """The same random axis permutation and flips applied to the CT cube and its teacher map."""
    perm = PERMS[rng.integers(len(PERMS))]
    img, teacher = img.transpose(perm), teacher.transpose(perm)
    for ax in range(3):
        if rng.random() < 0.5:
            img, teacher = np.flip(img, ax), np.flip(teacher, ax)
    return np.ascontiguousarray(img), np.ascontiguousarray(teacher)


def lr_at(step: int, total: int, base: float, warmup: int = 200) -> float:
    """Linear warmup, then cosine decay to zero."""
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


class CubeStream(torch.utils.data.IterableDataset):
    """Endless random augmented (CT, teacher) pairs; module level so worker processes can pickle it."""

    def __init__(self, files: list[str], props: dict, seed: int):
        self.files, self.props, self.seed = files, props, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed + wid * 1000 + int(time.time()))
        while True:
            d = np.load(self.files[rng.integers(len(self.files))])
            img, t = augment(d["img"], d["teacher"], rng)
            x = torch.from_numpy(mo.ct_normalize(img, self.props))[None]
            yield x, torch.from_numpy(t.astype(np.float32) / 255.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cubes")
    ap.add_argument("out", type=Path)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{a.cubes}/*.npz"))
    print(f"{len(files)} cubes", flush=True)
    net, props = mo.load_net()
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)

    loader = iter(
        torch.utils.data.DataLoader(
            CubeStream(files, props, a.seed), batch_size=a.batch, num_workers=4, pin_memory=True
        )
    )
    t0 = time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr)
        x, t = next(loader)
        x, t = x.cuda(non_blocking=True), t.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(x)
        logp = torch.log_softmax(logits.float(), 1)
        ce = -(t * logp[:, 1] + (1 - t) * logp[:, 0]).mean()
        p = logp[:, 1].exp()
        dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
        loss = ce + dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12.0)
        opt.step()
        if step % 50 == 0:
            print(
                f"step {step} loss {loss.item():.4f} ce {ce.item():.4f} dice {dice.item():.4f} "
                f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s",
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
