"""Sweep the ridge cut's knobs on the 60 Scroll-4 cubes (GATE vesuvius.md LOG 2026-09-17 15:55).

variants: base (tol 3, contact 3^3), tol2, contact5, tol2+contact5, second pass.
usage: rc_variants.py OUT.json
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


def contact(inst: np.ndarray, size: int = 3) -> np.ndarray:
    fg = inst > 0
    big = np.iinfo(inst.dtype).max
    mx = ndi.maximum_filter(inst, size=size, mode="constant", cval=0)
    mn = ndi.minimum_filter(np.where(fg, inst, big), size=size, mode="constant", cval=big)
    return fg & ((mx != inst) | (mn != inst))


def instances(mask: np.ndarray, ridge: np.ndarray, tol: float) -> np.ndarray:
    rl, _ = ndi.label(ridge, structure=S26)
    d, idx = ndi.distance_transform_edt(rl == 0, return_indices=True)
    inst = rl[tuple(idx)].astype(np.int32)
    inst[(d > tol) | ~mask] = 0
    return inst


def cut(mask, ridge, tol=3.0, csize=3):
    inst = instances(mask, ridge, tol)
    return mask & (inst > 0) & ~contact(inst, csize)


def second_pass(p, mask, cut1, min_vox=20000):
    """Recompute the ridge inside each surviving big instance with a tighter tensor, cut again."""
    lab, n = ndi.label(cut1, structure=S26)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    boxes = ndi.find_objects(lab)
    out = cut1.copy()
    for i in np.flatnonzero(sizes >= min_vox):
        if i == 0:
            continue
        box = boxes[int(i) - 1]
        sub = lab[box] == i
        ps = ndi.gaussian_filter(p[box].astype(np.float32), 0.5)
        g = np.stack(np.gradient(ps), -1)
        st = np.empty(ps.shape + (3, 3), np.float32)
        for a in range(3):
            for b in range(a, 3):
                st[..., a, b] = st[..., b, a] = ndi.gaussian_filter(g[..., a] * g[..., b], 1.0)
        nrm = np.linalg.eigh(st.reshape(-1, 3, 3))[1][:, :, -1].reshape(ps.shape + (3,))
        idx = np.argwhere(sub)
        base = ps[tuple(idx.T)]
        keep = np.ones(len(idx), bool)
        for step in (1.0, -1.0, 2.0, -2.0):
            q = idx + step * nrm[tuple(idx.T)]
            keep &= base >= ndi.map_coordinates(ps, q.T, order=1, mode="nearest")
        ridge2 = np.zeros(sub.shape, bool)
        ridge2[tuple(idx[keep].T)] = True
        out[box] &= ~sub | cut(sub, ridge2)
    return out


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
    names = ["mask", "rc_base", "rc_tol2", "rc_c5", "rc_tol2_c5", "rc_pass2"]
    tot = {k: [0, 0, 0] for k in names}
    rows = []
    for n in picks:
        img = tifffile.imread(n.replace("labelsTr", "imagesTr")[:-4] + "_0000.tif")
        lab = (tifffile.imread(n) > 0)[sm.CC]
        pb = sm.mo.predict_probs(base, img, props)[sm.CC]
        pv = sm.mo.predict_probs(var, img, props)[sm.CC].astype(np.float32)
        th = float(np.quantile(pv, 1 - (pb >= 0.2).mean()))
        mask = pv >= th
        ridge = rc.nms_ridge(pv, mask)
        c = {
            "mask": mask,
            "rc_base": cut(mask, ridge),
            "rc_tol2": cut(mask, ridge, tol=2.0),
            "rc_c5": cut(mask, ridge, csize=5),
            "rc_tol2_c5": cut(mask, ridge, tol=2.0, csize=5),
        }
        c["rc_pass2"] = second_pass(pv, mask, c["rc_base"])
        row = {"cube": Path(n).name}
        for k, m in c.items():
            mg, fr = rc.merge_and_split_tol(ndi.label(m, structure=S26)[0], lab)
            tot[k][0] += mg
            tot[k][1] += fr
            tot[k][2] += int(m.sum())
            row[k] = {"merges": mg, "frags": fr, "vox": int(m.sum())}
        rows.append(row)
        print(f"{Path(n).name:26s} " + "  ".join(f"{k} m{row[k]['merges']} f{row[k]['frags']}" for k in names), flush=True)
    nc = len(picks)
    for k in names:
        kept = tot[k][2] / max(tot["rc_base"][2], 1)
        print(f"SUMMARY {k:11s} merges {tot[k][0] / nc:.2f} frags {tot[k][1] / nc:.2f} kept {kept:.2f}x base", flush=True)
    ok = [k for k in names[2:] if tot[k][0] / nc <= 0.25 and tot[k][1] / nc <= 1.10 and tot[k][2] >= 0.9 * tot["rc_base"][2]]
    print(f"GATE variants passing: {ok if ok else 'NONE'}", flush=True)
    json.dump({"cubes": rows, "totals": tot}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
