"""Is there a CT gap where the ridge merges two sheets? (GATE vesuvius.md LOG 2026-09-17 20:05)

usage: ct_gap.py OUT.json
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
NEAR, FAR, MAXPTS = 8.0, 12.0, 4000


def segments(a_mask: np.ndarray, b_mask: np.ndarray, near: float, rng):
    """Sample points along the segment from each near-B voxel of A to its nearest B voxel."""
    d, idx = ndi.distance_transform_edt(~b_mask, return_indices=True)
    sel = a_mask & (d <= near) & (d >= 2)
    pts = np.argwhere(sel)
    if len(pts) == 0:
        return None
    if len(pts) > MAXPTS:
        pts = pts[rng.choice(len(pts), MAXPTS, replace=False)]
    nb = np.stack([idx[k][tuple(pts.T)] for k in range(3)], -1)
    fr = np.linspace(0.2, 0.8, 7)[:, None, None]
    return pts[None] * (1 - fr) + nb[None] * fr  # (7, n, 3)


def ratios(vol: np.ndarray, seg: np.ndarray, sheet_vals: np.ndarray) -> float:
    """Darkest point along each segment, relative to the sheets' own value."""
    v = ndi.map_coordinates(vol.astype(np.float32), seg.reshape(-1, 3).T, order=1, mode="nearest")
    lowest = v.reshape(seg.shape[0], seg.shape[1]).min(axis=0)
    return float(np.median(lowest) / max(np.median(sheet_vals), 1e-6))


def main(out_path: Path) -> None:
    sm = importlib.util.module_from_spec(
        spec := importlib.util.spec_from_file_location("sm", ROOT / "data/rgt/split_merges.py")
    )
    spec.loader.exec_module(sm)
    sys.path.insert(0, str(ROOT / "data/rgt"))
    diag = importlib.util.module_from_spec(
        spec2 := importlib.util.spec_from_file_location("diag", ROOT / "data/rgt/rc_diag.py")
    )
    spec2.loader.exec_module(diag)
    base, props = sm.mo.load_net()
    var, _ = sm.mo.load_variant("data/m7label/run_nos4/step_006000.pth")
    rng = np.random.default_rng(0)
    pool = sorted(glob.glob("data/m7label/labelsTr/s4_*.tif"))
    picks = [pool[i] for i in rng.choice(len(pool), min(60, len(pool)), replace=False)]
    groups = {k: {"ct": [], "prob": []} for k in ("stuck", "control", "far", "within")}
    for n in picks:
        img = tifffile.imread(n.replace("labelsTr", "imagesTr")[:-4] + "_0000.tif")[sm.CC]
        lab = (tifffile.imread(n) > 0)[sm.CC]
        lc, ln = ndi.label(lab, structure=S26)
        sizes = np.bincount(lc.ravel(), minlength=ln + 1)
        pb = sm.mo.predict_probs(base, tifffile.imread(n.replace("labelsTr", "imagesTr")[:-4] + "_0000.tif"), props)[sm.CC]
        pv = sm.mo.predict_probs(var, tifffile.imread(n.replace("labelsTr", "imagesTr")[:-4] + "_0000.tif"), props)[sm.CC].astype(np.float32)
        th = float(np.quantile(pv, 1 - (pb >= 0.2).mean()))
        mask = pv >= th
        ridge = rc.nms_ridge(pv, mask)
        stuck = diag.merged_pairs(ndi.label(ridge, structure=S26)[0], lc, sizes)
        big = [i for i in range(1, ln + 1) if sizes[i] >= 500]
        for ai in range(len(big)):
            for bi in range(ai + 1, len(big)):
                a, b = big[ai], big[bi]
                A, B = lc == a, lc == b
                dmin = ndi.distance_transform_edt(~B)[A].min()
                if dmin > FAR:
                    continue
                grp = "stuck" if (a, b) in stuck else ("control" if dmin <= NEAR else "far")
                mids = segments(A, B, FAR if grp == "far" else NEAR, rng)
                if mids is None or mids.shape[1] < 50:
                    continue
                sheet_ct = img[A | B]
                sheet_pv = pv[A | B]
                groups[grp]["ct"].append(ratios(img, mids, sheet_ct))
                groups[grp]["prob"].append(ratios(pv, mids, sheet_pv))
        for a in big[:2]:  # cheating baseline: midpoints inside one sheet
            A = lc == a
            pts = np.argwhere(A)
            if len(pts) < 100:
                continue
            pts = pts[rng.choice(len(pts), min(MAXPTS, len(pts)), replace=False)]
            other = pts[rng.permutation(len(pts))]
            keep = np.linalg.norm(other - pts, axis=1) <= 12
            if keep.sum() < 50:
                continue
            fr = np.linspace(0.2, 0.8, 7)[:, None, None]
            seg = pts[keep][None] * (1 - fr) + other[keep][None] * fr
            groups["within"]["ct"].append(ratios(img, seg, img[A]))
            groups["within"]["prob"].append(ratios(pv, seg, pv[A]))
        print(f"{Path(n).name:26s} " + " ".join(f"{k} {len(groups[k]['ct'])}" for k in groups), flush=True)
    out = {}
    for k, v in groups.items():
        out[k] = {"n": len(v["ct"]),
                  "ct_median": float(np.median(v["ct"])) if v["ct"] else None,
                  "prob_median": float(np.median(v["prob"])) if v["prob"] else None}
        print(f"GROUP {k:8s} n {out[k]['n']:4d} CT dip {out[k]['ct_median']} prob dip {out[k]['prob_median']}", flush=True)
    far_ok = out["far"]["ct_median"] is not None and out["far"]["ct_median"] <= 0.85
    print(f"VALIDITY far-pair dip <= 0.85: {far_ok} ({out['far']['ct_median']})", flush=True)
    if not far_ok:
        print("GATE READING: probe invalid, no claim", flush=True)
        json.dump(out, open(out_path, "w"), indent=1)
        return
    s, c = out["stuck"]["ct_median"], out["control"]["ct_median"]
    if s is not None and c is not None:
        reading = ("CT keeps the gap - build a CT-aided ridge" if s <= 0.85 and abs(s - c) <= 0.05
                   else "fused in the scan - close the line" if s >= 0.95 else "mixed, no claim")
        print(f"GATE READING: {reading} (stuck {s:.2f}, control {c:.2f})", flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
