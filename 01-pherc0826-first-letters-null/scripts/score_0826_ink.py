"""Per-winding seed agreement + ink fraction for two prediction prefixes on the 0826 renders.

usage: score_0826_ink.py IN_DIR OUT_DIR A_PREFIX_FMT B_PREFIX_FMT
  prefix fmt has {w} and {d}: e.g. 's42_{w}_step-075000_{d}' and 'ft42_{w}_ckpt_012000_{d}'
Prints one row per winding/direction: frac>128, p99, r(seed42,seed43) for both prefixes,
and writes a ds4 montage (seed mean) for the best windings.
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

IN, OUT = sys.argv[1], sys.argv[2]
pairs = {
    "base": (sys.argv[3], sys.argv[3].replace("s42", "s43")),
    "ft": (sys.argv[4], sys.argv[4].replace("ft42", "ft43")),
}
os.makedirs(OUT, exist_ok=True)


def load(wdir, w, fmt, d):
    fs = sorted(glob.glob(f"{wdir}/preds/{fmt.format(w=w, d=d)}_*.tif"))
    return tifffile.imread(fs[0]).astype(np.float32) if fs else None


rows, best = [], []
for wdir in sorted(glob.glob(f"{IN}/w*")):
    w = os.path.basename(wdir)
    for d in ("forward", "reverse"):
        row = {"w": w, "dir": d}
        for tag, (fa, fb) in pairs.items():
            a, b = load(wdir, w, fa, d), load(wdir, w, fb, d)
            if a is None or b is None:
                continue
            m = (a > 0) & (b > 0)
            if m.sum() < 1000:
                continue
            r = float(np.corrcoef(a[m], b[m])[0, 1])
            mean = (a + b) / 2
            row[tag] = {
                "r": round(r, 3),
                "frac128": round(float((mean[m] > 128).mean()), 4),
                "p99": float(np.percentile(mean[m], 99)),
            }
            if tag == "ft":
                best.append((r, w, d, mean[::4, ::4]))
        rows.append(row)

print("winding dir      base r  base>128   ft r   ft>128  ft p99")
for r in rows:
    b, f = r.get("base", {}), r.get("ft", {})
    print(
        "%s %-8s %6s  %8s  %6s  %7s  %5s"
        % (
            r["w"],
            r["dir"],
            b.get("r", "-"),
            b.get("frac128", "-"),
            f.get("r", "-"),
            f.get("frac128", "-"),
            round(f["p99"]) if f else "-",
        )
    )
Path(f"{OUT}/scores.json").write_text(json.dumps(rows, indent=1))
for tag in pairs:
    rs = [r[tag]["r"] for r in rows if tag in r]
    fr = [r[tag]["frac128"] for r in rows if tag in r]
    if rs:
        print(
            f"{tag}: r median {np.median(rs):.3f} (min {min(rs):.2f} max {max(rs):.2f}); frac>128 median {np.median(fr):.3f} max {max(fr):.3f}"
        )
best.sort(key=lambda t: -t[0])
for r, w, d, im in best[:8]:
    Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).save(f"{OUT}/{w}_{d}_r{r:.2f}_ds4.png")
