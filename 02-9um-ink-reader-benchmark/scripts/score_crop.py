"""hp-r of a read against its key inside one crop, on either grid.

usage: score_crop.py PRED.tif KEY(.npy|.tif) Y0 Y1 X0 X1 [--px 9.362|2.399]
Same maths as data/ink_jitter/score.py (48 um high-pass, normalised convolution, inked = key
smoothed 100 um > 60, nulls from rolled keys), restricted to the crop and to pred > 0.
"""

import sys

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

pred_p, key_p = sys.argv[1], sys.argv[2]
y0, y1, x0, x1 = (int(v) for v in sys.argv[3:7])
px = float(sys.argv[8]) if len(sys.argv) > 8 else 9.362
SIG, SMO = 48 / px, 96 / px
key = (np.load(key_p) if key_p.endswith(".npy") else tifffile.imread(key_p)).astype(
    np.float32
)
pred = tifffile.imread(pred_p).astype(np.float32)
if pred.shape != key.shape:  # a bare crop: embed at the crop origin
    canvas = np.zeros(key.shape, np.float32)
    canvas[y0 : y0 + pred.shape[0], x0 : x0 + pred.shape[1]] = pred
    pred = canvas
pad = int(4 * SIG)
sl = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
key, pred = key[sl], pred[sl]
valid = key >= 0
key = np.where(valid, key, 0)


def hp(img, m):
    w = gaussian_filter(m.astype(np.float32), SIG)
    return img - gaussian_filter(img * m, SIG) / np.maximum(w, 1e-3)


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


kh, inked = hp(key, valid), gaussian_filter(key, SMO) > 60
core = valid & (gaussian_filter(valid.astype(np.float32), 2 * SIG) > 0.999)
inner = np.zeros_like(valid)
inner[pad : pad + (y1 - y0), pad : pad + (x1 - x0)] = True
m = core & (pred > 0) & inner
ph = hp(pred, pred > 0)
shifts = [int(round(s * 9.362 / px)) for s in (150, 300, -300)]
null = max(
    abs(r(ph[m & np.roll(m, s, 0)], np.roll(kh, s, 0)[m & np.roll(m, s, 0)]))
    for s in shifts
)
print(
    f"{pred_p}: px {m.sum()/1e6:.2f} M  raw r {r(pred[m], key[m]):+.3f}  hp r {r(ph[m], kh[m]):+.3f}  "
    f"inked hp r {r(ph[m & inked], kh[m & inked]):+.3f}  null|max| {null:.3f}"
)
