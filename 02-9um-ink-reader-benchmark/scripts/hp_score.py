"""Letter-scale (hp r) scores for held-out exam reads on their own grid (pred and key_native share it).

usage: score.py TAG [TAG ...]   (pred files are <exam>/preds/<TAG>.tif; ft_s42 is the r1 baseline)

Pixels where a pred is exactly 0 count as not covered and are skipped. A model that writes hard zeros
inside the render is therefore scored on fewer pixels; compare on matched masks (see the 2026-09-24 addendum).
"""
import sys

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

SIG, SMO = 20 / 3.9, 40 / 3.9  # 48 um high-pass, 100 um "inked" smoothing, at 9.362 um px
EXAMS = {"w042 held-out 0139": "data/ink_pseudo/w042", "0814 legible anchor": "data/ink_pseudo_0814/p46527",
         "0500P2 known-bad": "data/ink_pseudo_0500P2/front"}


def hp(img, m):
    w = gaussian_filter(m.astype(np.float32), SIG)
    return img - gaussian_filter(img * m, SIG) / np.maximum(w, 1e-3)


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


for name, seg in EXAMS.items():
    key = np.load(f"{seg}/key_native.npy").astype(np.float32)
    valid = key >= 0
    key = np.where(valid, key, 0)
    kh, inked = hp(key, valid), gaussian_filter(key, SMO) > 60
    core = valid & (gaussian_filter(valid.astype(np.float32), 2 * SIG) > 0.999)
    for tag in sys.argv[1:]:
        try:
            p = tifffile.imread(f"{seg}/preds/{tag}.tif").astype(np.float32)[: key.shape[0], : key.shape[1]]
        except FileNotFoundError:
            continue
        m = core & (p > 0)
        ph = hp(p, m)
        null = max(abs(r(ph[m & np.roll(m, s, 0)], np.roll(kh, s, 0)[m & np.roll(m, s, 0)])) for s in (150, 300, -300))
        print(f"{name:20s} {tag:10s} raw r {r(p[m], key[m]):+.3f}  hp r {r(ph[m], kh[m]):+.3f}  "
              f"inked hp r {r(ph[m & inked], kh[m & inked]):+.3f}  null|max| {null:.3f}", flush=True)
