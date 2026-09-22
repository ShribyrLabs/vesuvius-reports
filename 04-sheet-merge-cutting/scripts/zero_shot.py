"""RGT-Est zero-shot on scroll CT cubes (GATE: vesuvius.md LOG 2026-09-14 22:05).

Known-first: the authors' Poseidon survey must give a field monotone along time. Then the 20
held-out labelled cubes: "time" = the axis along which the label sheets are stacked; score each
field (RGT-Est, linear ramp, smoothed CT) by how well its level sets separate the label sheets.

usage: zero_shot.py OUT.json
"""

import os
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import spearmanr

ROOT = Path(os.environ.get("VESUVIUS_ROOT", Path(__file__).resolve().parents[2]))
CK = ROOT / "data/checkpoints/rgt_est/RGT-Est_CIG-Benchmark.pt"
SHAPE = (400, 512, 512)
dev = "cuda"


def z_score_clip(x: np.ndarray, clp: float = 3.0) -> np.ndarray:
    z = (x - x.mean()) / (x.std() + 1e-8)
    z = np.clip(z, -clp, clp)
    return (z - z.min()) / (z.max() - z.min() + 1e-6)


def run_model(model, vol_thw: np.ndarray) -> np.ndarray:
    """vol_thw: float32 (T, H, W) in [-1, 1]; returns the field at the input shape."""
    t = torch.from_numpy(vol_thw)[None, None]
    x = F.interpolate(t, SHAPE, mode="trilinear")
    x = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], 1).to(dev)
    with torch.no_grad(), torch.autocast(device_type=dev):
        r = model(nn.ReplicationPad3d(8)(x))[:, :, 8:-8, 8:-8, 8:-8]
    return F.interpolate(r.float(), vol_thw.shape, mode="trilinear").cpu().numpy()[0, 0]


def stacking_axis(lab: np.ndarray) -> int:
    """Axis along which the label sheets are stacked (dominant gradient direction of the label)."""
    sm = ndi.gaussian_filter(lab.astype(np.float32), 3)
    g = np.stack(np.gradient(sm), -1).reshape(-1, 3)
    cov = g.T @ g
    return int(np.argmax(np.linalg.eigh(cov)[1][:, -1] ** 2))


def sheet_score(field: np.ndarray, lab: np.ndarray, min_vox: int = 500) -> dict:
    """median sigma_in/Delta over sheets and the merged fraction (see the GATE line)."""
    comp, n = ndi.label(lab, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(comp.ravel())
    ids = [i for i in range(1, n + 1) if sizes[i] >= min_vox]
    if len(ids) < 2:
        return {"n_sheets": len(ids)}
    med = np.array([np.median(field[comp == i]) for i in ids])
    # does the field ORDER the sheets as they are stacked along axis 0 (the stacking axis)?
    zc = np.array([np.argwhere(comp == i)[:, 0].mean() for i in ids])
    order_rho = float(abs(spearmanr(med, zc).correlation)) if len(ids) > 2 else float("nan")
    std = np.array([field[comp == i].std() for i in ids])
    order = np.argsort(med)
    ratios, merged = [], []
    for k, i in enumerate(ids):
        pos = int(np.nonzero(order == k)[0][0])
        nb = [order[pos - 1]] if pos > 0 else []
        nb += [order[pos + 1]] if pos + 1 < len(order) else []
        delta = min(abs(med[k] - med[j]) for j in nb)
        ratios.append(std[k] / max(delta, 1e-9))
        v = field[comp == i]
        merged.append(
            float(np.mean([np.mean(np.abs(v - med[j]) < delta / 4) for j in nb]))
        )
    return {
        "n_sheets": len(ids),
        "ratio": float(np.median(ratios)),
        "merged": float(np.mean(merged)),
        "order_rho": order_rho,
    }


def main() -> None:
    out = Path(sys.argv[1])
    model = torch.jit.load(str(CK)).to(dev).eval()
    res = {}
    # --- known-first: the authors' field survey ---
    seis = np.load(ROOT / "data/checkpoints/rgt_est/Poseidon_part1.npy").astype(
        np.float32
    )
    seis = z_score_clip(seis, 2).transpose(1, 2, 0) * 2 - 1  # (H, W, T) as in the demo
    thw = np.ascontiguousarray(seis.transpose(2, 0, 1))
    r = run_model(model, thw)
    tt = np.broadcast_to(np.arange(r.shape[0])[:, None, None], r.shape)
    sub = (slice(None, None, 4),) * 3
    rho = spearmanr(r[sub].ravel(), tt[sub].ravel()).correlation
    res["poseidon"] = {"shape": list(r.shape), "spearman_time": float(rho)}
    print(f"Poseidon: field shape {r.shape}, Spearman with time {rho:.3f}", flush=True)
    # --- the 20 held-out cubes ---
    names = (ROOT / "data/m7check/labels/pick.txt").read_text().split()
    cubes = []
    for nme in names:
        lab = tifffile.imread(ROOT / "data/m7check/labels/labelsTr" / nme) > 0
        img = tifffile.imread(
            ROOT / "data/m7check/labels/imagesTr" / (nme[:-4] + "_0000.tif")
        )
        ax = stacking_axis(lab)
        perm = [ax] + [a for a in range(3) if a != ax]
        img_t, lab_t = img.transpose(perm), lab.transpose(perm)
        x = (z_score_clip(img_t.astype(np.float32), 2) * 2 - 1).astype(np.float32)
        rgt = run_model(model, np.ascontiguousarray(x))
        ramp = np.broadcast_to(
            np.linspace(0, 1, x.shape[0], dtype=np.float32)[:, None, None], x.shape
        )
        smooth = ndi.gaussian_filter(img_t.astype(np.float32), 3)
        row = {"axis": ax}
        for tag, f in (("rgt", rgt), ("ramp", ramp), ("smooth_ct", smooth)):
            row[tag] = sheet_score(np.ascontiguousarray(f), lab_t)
        cubes.append({"cube": nme, **row})
        print(
            f"{nme:28s} axis {ax}  rgt {row['rgt'].get('ratio', float('nan')):.3f}/{row['rgt'].get('merged', float('nan')):.2f}"
            f"  ramp {row['ramp'].get('ratio', float('nan')):.3f}/{row['ramp'].get('merged', float('nan')):.2f}"
            f"  ct {row['smooth_ct'].get('ratio', float('nan')):.3f}",
            flush=True,
        )
    ok = [c for c in cubes if "ratio" in c["rgt"] and "ratio" in c["ramp"]]
    wins = sum(c["rgt"]["ratio"] < c["ramp"]["ratio"] for c in ok)
    med = float(np.median([c["rgt"]["ratio"] for c in ok]))
    m_rgt = float(np.mean([c["rgt"]["merged"] for c in ok]))
    m_ramp = float(np.mean([c["ramp"]["merged"] for c in ok]))
    rho = float(np.median([c["rgt"]["order_rho"] for c in ok]))
    res["cubes"] = cubes
    res["summary"] = {
        "wins_vs_ramp": wins,
        "n": len(ok),
        "median_ratio": med,
        "merged_rgt": m_rgt,
        "merged_ramp": m_ramp,
    }
    verdict = wins >= 14 and med < 0.3 and m_rgt < m_ramp
    print(
        f"SUMMARY rgt beats ramp {wins}/{len(ok)}  median ratio {med:.3f}  merged rgt {m_rgt:.2f} ramp {m_ramp:.2f}  order rho {rho:.2f}  "
        f"GATE {'PASS' if verdict else 'FAIL'}",
        flush=True,
    )
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
