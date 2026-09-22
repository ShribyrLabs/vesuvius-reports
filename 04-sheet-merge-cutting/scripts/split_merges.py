"""Split merged sheets in a surface map by the zero-shot RGT field (GATE LOG 2026-09-15 00:16).

On the 60 Scroll-4 gate cubes: nos4 mask at matched volume -> components -> each component split
into field-level bands (histogram valleys) -> components again. Same with the linear ramp along
the predicted mask's stacking axis (cheating baseline). Scored by merge pairs and split count
against the label sheets. usage: split_merges.py OUT.json
"""

import os
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path(os.environ.get("VESUVIUS_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "data/rgt"))
import zero_shot as zs  # noqa: E402

spec = importlib.util.spec_from_file_location("mo", ROOT / "scripts/m7_orient.py")
mo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mo)
S26 = np.ones((3, 3, 3), bool)
CC = (slice(16, -16),) * 3


def merge_and_split(
    inst: np.ndarray, lab: np.ndarray, min_vox: int = 500
) -> tuple[int, int, int]:
    """(merge pairs, split count) of an instance map against the label sheets."""
    lc, ln = ndi.label(lab, structure=S26)
    sizes = np.bincount(lc.ravel(), minlength=ln + 1)
    pn = int(inst.max())
    key = lc[lab].astype(np.int64) * (pn + 1) + inst[lab]
    u, cnt = np.unique(key, return_counts=True)
    li, pi = u // (pn + 1), u % (pn + 1)
    ok = (pi > 0) & (sizes[li] >= min_vox) & (cnt >= 0.3 * sizes[li])
    owner = np.bincount(pi[ok], minlength=pn + 1)
    merges = int((owner * (owner - 1) // 2).sum())
    covered = np.bincount(li[ok], minlength=ln + 1)
    splits = int((covered >= 2).sum())
    ok5 = (pi > 0) & (sizes[li] >= min_vox) & (cnt >= 0.05 * sizes[li])
    pieces = np.bincount(li[ok5], minlength=li.max() + 2 if len(li) else ln + 1)[: ln + 1]
    frags = int(np.clip(pieces[1:] - 1, 0, None).sum())  # extra pieces (>= 5 %) per label sheet
    return merges, splits, frags


sys.path.insert(0, str(ROOT / "scripts"))
from winding_split import split_by_field  # noqa: E402


def stacking_axis_of(mask: np.ndarray) -> int:
    return zs.stacking_axis(mask)


def main() -> None:
    out = Path(sys.argv[1])
    model = __import__("torch").jit.load(str(zs.CK)).to("cuda").eval()
    base, props = mo.load_net()
    var, _ = mo.load_variant("data/m7label/run_nos4/step_006000.pth")
    rng = np.random.default_rng(0)
    pool = sorted(glob.glob("data/m7label/labelsTr/s4_*.tif"))
    names = [pool[i] for i in rng.choice(len(pool), min(60, len(pool)), replace=False)]
    rows = []
    for n in names:
        img = tifffile.imread(n.replace("labelsTr", "imagesTr")[:-4] + "_0000.tif")
        lab = (tifffile.imread(n) > 0)[CC]
        pb = mo.predict_probs(base, img, props)[CC]
        pv = mo.predict_probs(var, img, props)[CC]
        vol = (pb >= 0.2).mean()
        mask = pv >= np.quantile(pv, 1 - vol)
        ax = stacking_axis_of(mask)
        perm = [ax] + [a for a in range(3) if a != ax]
        inv = np.argsort(perm)
        x = (
            zs.z_score_clip(img[CC].transpose(perm).astype(np.float32), 2) * 2 - 1
        ).astype(np.float32)
        field = zs.run_model(model, np.ascontiguousarray(x)).transpose(inv)
        ramp = np.broadcast_to(
            np.linspace(0, 1, x.shape[0], dtype=np.float32)[:, None, None], x.shape
        ).transpose(inv)
        row = {"cube": Path(n).name, "axis": ax}
        nosplit = ndi.label(mask, structure=S26)[0]
        settings = [
            ("nosplit", None, 64, 0.5),
            ("rgt", field, 64, 0.5),
            ("ramp", np.ascontiguousarray(ramp), 64, 0.5),
        ] + [
            (f"rgt_b{b}_d{d}", field, b, d)
            for b in (32, 64, 128)
            for d in (0.3, 0.5, 0.7)
            if not (b == 64 and d == 0.5)
        ]
        for tag, f, b, d in settings:
            inst = nosplit if f is None else split_by_field(mask, f, bins=b, depth=d)
            mg, sp, fr = merge_and_split(inst, lab)
            row[tag] = {"merges": mg, "splits": sp, "frags": fr, "n_inst": int(inst.max())}
        rows.append(row)
        print(
            f"{Path(n).name:26s} ax {ax}  "
            + "  ".join(
                f"{t} m{row[t]['merges']} s{row[t]['splits']} n{row[t]['n_inst']}"
                for t in ("nosplit", "rgt", "ramp")
            ),
            flush=True,
        )
    tags = [t for t in rows[0] if t not in ("cube", "axis")]
    tot = {
        t: {k: sum(r[t][k] for r in rows) for k in ("merges", "splits", "frags", "n_inst")}
        for t in tags
    }
    nc = len(rows)
    for t in tags:
        print(
            f"SWEEP {t:14s} merges {tot[t]['merges'] / nc:.2f}  frags {tot[t]['frags'] / nc:.2f}  "
            f"inst {tot[t]['n_inst'] / nc:.1f}",
            flush=True,
        )
    print(
        "SUMMARY per cube: "
        + "  ".join(
            f"{t} merges {tot[t]['merges'] / nc:.2f} splits {tot[t]['splits'] / nc:.2f}"
            for t in tot
        )
    )
    cut = 1 - tot["rgt"]["merges"] / max(tot["nosplit"]["merges"], 1)
    ok = (
        cut >= 0.30
        and tot["rgt"]["merges"] < tot["ramp"]["merges"]
        and tot["rgt"]["splits"] <= tot["ramp"]["splits"]
        and tot["rgt"]["splits"] < 1.2 * max(tot["nosplit"]["splits"], 1)
    )
    print(f"GATE merge cut {cut:.2f}  {'PASS' if ok else 'FAIL'}", flush=True)
    json.dump({"cubes": rows, "totals": tot}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
