"""Pre-registered gate for an m7 variant on the 20 held-out labelled cubes (vesuvius.md LOG 2026-09-14):
as scanned Dice >= base - 0.01 AND z<->y Dice >= CT echo + 0.10, both with the variant's threshold set
so its volume matches base@0.2 on that view.

usage: gate_eval.py CHECKPOINT
"""

import importlib.util
import sys

import numpy as np
import tifffile
import torch
from scipy import ndimage as ndi

spec = importlib.util.spec_from_file_location("mo", "scripts/m7_orient.py")
mo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mo)
base, props = mo.load_net()
var, _ = mo.load_variant(sys.argv[1])
cc = (slice(16, -16),) * 3
import glob

# default: the 20 held-out cubes; optional 2nd arg = glob of label tifs (e.g. a left-out scroll)
if len(sys.argv) > 2:
    rng = np.random.default_rng(0)
    pool = sorted(glob.glob(sys.argv[2]))
    names = [pool[i] for i in rng.choice(len(pool), min(60, len(pool)), replace=False)]
else:
    names = [f"data/m7check/labels/labelsTr/{n}" for n in open("data/m7check/labels/pick.txt").read().split()]
S26 = np.ones((3, 3, 3), bool)


def merge_pairs(pred: np.ndarray, lab: np.ndarray, min_vox: int = 500) -> int:
    """Number of pairs of label sheets (components >= min_vox) that one predicted component covers.

    A label sheet is 'covered' by the predicted component holding >= 30 % of its voxels. Two sheets
    covered by the same predicted component are a merge (sheet switch waiting to happen)."""
    lc, ln = ndi.label(lab, structure=S26)
    pc, pn = ndi.label(pred, structure=S26)
    sizes = np.bincount(lc.ravel(), minlength=ln + 1)
    key = lc[lab].astype(np.int64) * (pn + 1) + pc[lab]
    u, cnt = np.unique(key, return_counts=True)
    li, pi = u // (pn + 1), u % (pn + 1)
    ok = (pi > 0) & (sizes[li] >= min_vox) & (cnt >= 0.3 * sizes[li])
    owner = np.bincount(pi[ok], minlength=pn + 1)
    return int((owner * (owner - 1) // 2).sum())


acc = {}
merges = {}
for n in names:
    img0 = tifffile.imread(n.replace("labelsTr", "imagesTr")[:-4] + "_0000.tif")
    lab0 = tifffile.imread(n) > 0
    for view, perm in (("asis", (0, 1, 2)), ("zy", (1, 0, 2))):
        img, lab = np.ascontiguousarray(img0.transpose(perm)), lab0.transpose(perm)
        dil = ndi.binary_dilation(lab, iterations=1)[cc]
        pb = mo.predict_probs(base, img, props)[cc]
        pv = mo.predict_probs(var, img, props)[cc]
        vol = (pb >= 0.2).mean()
        th = np.quantile(
            pv, 1 - vol
        )  # variant threshold giving base's volume on this cube
        echo = img[cc].astype(np.float32)
        for meth, pr in (
            ("base@0.2", pb >= 0.2),
            ("variant@0.2", pv >= 0.2),
            ("variant@matched", pv >= th),
            ("ct_echo", echo >= np.quantile(echo, 1 - vol)),
        ):
            d = 2 * (pr & dil).sum() / max(pr.sum() + dil.sum(), 1)
            acc.setdefault((view, meth), []).append(float(d))
            if meth != "ct_echo":
                merges.setdefault((view, meth), []).append(merge_pairs(pr, lab[cc]))
        merges.setdefault((view, "label_dil2"), []).append(
            merge_pairs(ndi.binary_dilation(lab, iterations=2)[cc], lab[cc])
        )
for k, v in acc.items():
    print(f"{k[0]:4s} {k[1]:16s} Dice {np.mean(v):.3f}")
for k, v in merges.items():
    print(f"{k[0]:4s} {k[1]:16s} merge pairs/cube {np.mean(v):.2f}  (cubes with any: {np.mean(np.array(v) > 0):.2f})")
a_ok = (
    np.mean(acc[("asis", "variant@matched")])
    >= np.mean(acc[("asis", "base@0.2")]) - 0.01
)
r_ok = np.mean(acc[("zy", "variant@matched")]) >= np.mean(acc[("zy", "ct_echo")]) + 0.10
print(
    f"GATE as-scanned {'PASS' if a_ok else 'FAIL'}  rotated {'PASS' if r_ok else 'FAIL'}",
    flush=True,
)
