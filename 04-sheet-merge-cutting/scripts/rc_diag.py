"""Are the surviving merges already present in the ridge? (GATE vesuvius.md LOG 2026-09-17 19:25)

usage: rc_diag.py OUT.json
"""

import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path("/path/to/vesuvius")
sys.path.insert(0, str(ROOT / "scripts"))
import ridge_cut as rc  # noqa: E402

S26 = rc.S26


def merged_pairs(comp: np.ndarray, lc: np.ndarray, sizes, min_vox=500, tol=3.0):
    """Set of label-sheet id pairs that one component covers >= 30 % of, tolerant assignment."""
    lab = lc > 0
    d, idx = ndi.distance_transform_edt(comp == 0, return_indices=True)
    near = comp[tuple(idx)]
    near[d > tol] = 0
    pn = int(comp.max())
    if pn == 0:
        return set()
    key = lc[lab].astype(np.int64) * (pn + 1) + near[lab]
    u, cnt = np.unique(key, return_counts=True)
    li, pi = u // (pn + 1), u % (pn + 1)
    ok = (pi > 0) & (sizes[li] >= min_vox) & (cnt >= 0.3 * sizes[li])
    pairs = set()
    for comp_id in np.unique(pi[ok]):
        sheets = sorted(li[ok][pi[ok] == comp_id].tolist())
        for a in range(len(sheets)):
            for b in range(a + 1, len(sheets)):
                pairs.add((sheets[a], sheets[b]))
    return pairs


def main(out_path: Path) -> None:
    sm = importlib.util.module_from_spec(
        spec := importlib.util.spec_from_file_location("sm", ROOT / "data/rgt/split_merges.py")
    )
    spec.loader.exec_module(sm)
    base, props = sm.mo.load_net()
    var, _ = sm.mo.load_variant("data/m7label/run_nos4/step_006000.pth")
    rng = np.random.default_rng(0)
    pool = sorted(glob.glob("data/m7label/labelsTr/s4_*.tif"))
    picks = [pool[i] for i in rng.choice(len(pool), min(60, len(pool)), replace=False)]
    tot = {"mask": 0, "cut": 0, "ridge": 0, "cut_and_ridge": 0, "cut_only": 0}
    rows = []
    for n in picks:
        img = tifffile.imread(n.replace("labelsTr", "imagesTr")[:-4] + "_0000.tif")
        lab = (tifffile.imread(n) > 0)[sm.CC]
        lc, ln = ndi.label(lab, structure=S26)
        sizes = np.bincount(lc.ravel(), minlength=ln + 1)
        pb = sm.mo.predict_probs(base, img, props)[sm.CC]
        pv = sm.mo.predict_probs(var, img, props)[sm.CC].astype(np.float32)
        th = float(np.quantile(pv, 1 - (pb >= 0.2).mean()))
        mask = pv >= th
        ridge = rc.nms_ridge(pv, mask)
        cut = rc.ridge_guided_cut(mask, ridge)
        p_mask = merged_pairs(ndi.label(mask, structure=S26)[0], lc, sizes)
        p_cut = merged_pairs(ndi.label(cut, structure=S26)[0], lc, sizes)
        p_ridge = merged_pairs(ndi.label(ridge, structure=S26)[0], lc, sizes)
        both = p_cut & p_ridge
        tot["mask"] += len(p_mask)
        tot["cut"] += len(p_cut)
        tot["ridge"] += len(p_ridge)
        tot["cut_and_ridge"] += len(both)
        tot["cut_only"] += len(p_cut - p_ridge)
        rows.append({"cube": Path(n).name, "mask": len(p_mask), "cut": len(p_cut),
                     "ridge": len(p_ridge), "both": len(both)})
        print(f"{Path(n).name:26s} mask {len(p_mask)} cut {len(p_cut)} ridge {len(p_ridge)} both {len(both)}", flush=True)
    share = tot["cut_and_ridge"] / max(tot["cut"], 1)
    print(f"TOTALS {tot}", flush=True)
    print(f"RIDGE-LEVEL SHARE {share:.2f} of {tot['cut']} surviving merges "
          f"(cheat check: mask pairs {tot['mask']} > cut pairs {tot['cut']} = {tot['mask'] > tot['cut']})", flush=True)
    verdict = "ceiling: ridge is the limit" if share >= 0.7 else ("assignment loses separations" if share <= 0.3 else "mixed, no claim")
    print(f"GATE READING: {verdict}", flush=True)
    json.dump({"cubes": rows, "totals": tot}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
