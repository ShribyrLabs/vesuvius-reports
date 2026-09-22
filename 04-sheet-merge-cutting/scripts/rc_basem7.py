#!/usr/bin/env python3
"""GATE (vesuvius.md LOG 2026-09-22): the ridge cut on base m7 probabilities, same 60 cubes as rc_variants.py."""
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path("/path/to/vesuvius")
TH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
sys.path.insert(0, str(ROOT / "scripts"))
import ridge_cut as rc  # noqa: E402


def main(out_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("sm", ROOT / "data/rgt/split_merges.py")
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    base, props = sm.mo.load_net()
    rng = np.random.default_rng(0)
    pool = sorted(glob.glob("data/m7label/labelsTr/s4_*.tif"))
    picks = [pool[i] for i in rng.choice(len(pool), min(60, len(pool)), replace=False)]
    tot = {"mask": [0, 0, 0], "cut": [0, 0, 0]}
    rows = []
    for n in picks:
        img = tifffile.imread(n.replace("labelsTr", "imagesTr")[:-4] + "_0000.tif")
        lab = (tifffile.imread(n) > 0)[sm.CC]
        pb = sm.mo.predict_probs(base, img, props)[sm.CC].astype(np.float32)
        mask = pb >= TH
        ridge = rc.nms_ridge(pb, mask)
        cut = rc.ridge_guided_cut(mask, ridge)
        row = {"cube": Path(n).name}
        for k, m in (("mask", mask), ("cut", cut)):
            mg, fr = rc.merge_and_split_tol(ndi.label(m, structure=rc.S26)[0], lab)
            tot[k][0] += mg
            tot[k][1] += fr
            tot[k][2] += int(m.sum())
            row[k] = {"merges": mg, "frags": fr, "vox": int(m.sum())}
        rows.append(row)
        print(f"{row['cube']:26s} mask m{row['mask']['merges']} f{row['mask']['frags']}  cut m{row['cut']['merges']} f{row['cut']['frags']}", flush=True)
    nc = len(picks)
    for k in tot:
        print(f"SUMMARY {k:5s} merges {tot[k][0] / nc:.2f} frags {tot[k][1] / nc:.2f} vox {tot[k][2]}", flush=True)
    kept = tot["cut"][2] / max(tot["mask"][2], 1)
    ok = (tot["cut"][0] <= 0.5 * tot["mask"][0]) and (tot["cut"][1] / nc <= tot["mask"][1] / nc + 0.4) and kept >= 0.9
    print(f"GATE base-m7 ridge cut: {'PASS' if ok else 'FAIL'} (kept {kept:.3f}x)", flush=True)
    json.dump({"cubes": rows, "totals": tot, "kept": kept, "pass": bool(ok)}, open(out_path, "w"), indent=1)
    print("BASEM7_DONE", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
