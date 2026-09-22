"""Track statistics per map on the crushed 0490A box, split by family, and how well track tangents
lie in the CT's own sheets (|tangent . CT normal|, 0 = in-sheet; random = 0.5), overall and where
the CT sheets lie flat (normal within 45 deg of z)."""
import dbm, pickle
import numpy as np
from scipy import ndimage as ndi
ct = np.load("data/m7tracks/ct.npy")
c2 = ct[::2, ::2, ::2].astype(np.float32)
g = [ndi.gaussian_filter(c2, 1.0, order=o) for o in ((1,0,0),(0,1,0),(0,0,1))]
j = np.empty(c2.shape + (3, 3), np.float32)
for a in range(3):
    for b in range(a, 3):
        j[..., a, b] = j[..., b, a] = ndi.gaussian_filter(g[a] * g[b], 3.0)
del g
n = np.linalg.eigh(j)[1][..., -1]; del j
print(f"CT flat-sheet share (n_z^2 > 0.5): {float((n[..., 0] ** 2 > 0.5).mean()):.3f}", flush=True)
for m in ("hosted", "base02", "run2_matched", "nos4_matched", "nos4_cut", "echo_matched"):
    fam = {}
    with dbm.open(f"data/m7tracks/{m}.dbm", "r") as db:
        for k in db.keys():
            f = k.decode().split(":")[0]
            fam.setdefault(f, []).extend(pickle.loads(db[k]))
    tot_tr = sum(len(v) for v in fam.values()); tot_pts = sum(len(t) for v in fam.values() for t in v)
    dots, flat_dots, flat_pts = [], [], 0
    for v in fam.values():
        for t in v:
            t = t.astype(np.float32)
            if len(t) < 5:
                continue
            tan = t[4:] - t[:-4]; tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-6
            p = np.clip((t[2:-2] // 2).astype(int), 0, np.array(n.shape[:3]) - 1)
            nn = n[p[:, 0], p[:, 1], p[:, 2]]
            d = np.abs((tan * nn).sum(1)); dots.append(d)
            fl = nn[:, 0] ** 2 > 0.5; flat_pts += int(fl.sum()); flat_dots.append(d[fl])
    dots = np.concatenate(dots); flat_dots = np.concatenate(flat_dots)
    per = " ".join(f"{f}:{len(v)}tr/{sum(len(t) for t in v)}pt" for f, v in sorted(fam.items()))
    print(f"{m:13s} tracks {tot_tr:6d} points {tot_pts:8d} mean len {tot_pts/max(tot_tr,1):6.1f} | {per} | "
          f"|t.n| all {dots.mean():.3f} | flat-region points {flat_pts:7d} |t.n| {flat_dots.mean() if len(flat_dots) else float('nan'):.3f}", flush=True)
