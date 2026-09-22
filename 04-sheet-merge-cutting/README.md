# Cutting merged sheets out of a surface map, and where that stops working

**In one sentence:** A post-processing cut that separates sheets a surface model has fused, using the model's own probability ridges; it improves a spiral fit on PHerc0139, but mostly because it fixes merges that my own fine-tune creates, not ones the released m7 makes.

**One real example:** PHerc0139 text band, probability map from the fine-tune in report 03. `scripts/ridge_cut.py` removes 2.4 % of the voxels (53 min on 6 CPU cores, no GPU). Same track extraction, same spiral fit, 4 seeds each: held-out winding score 0.571 → 0.605, every cut seed above every uncut seed.

**Before:** 1.62 merged sheet pairs per cube on 60 labelled PHerc1667 cubes; in a 384³ box of the band, one connected piece held 72 % of the map.

**After:** 0.38 merged pairs per cube; largest piece 38 %. `figures/0139_ridge_cut_before_after_384.png`. The cut box is in `results/` as a VC3D-loadable zarr (5 MB) if you want to look at it.

**How:** the probability still peaks once per sheet where the thresholded map has fused two. Non-maximum suppression along the sheet normal leaves one ridge per sheet; every map voxel goes to its nearest ridge; voxels between two ridges are removed. The map keeps its thickness, which track extraction needs. An earlier version used a seismic "relative geologic time" network zero-shot on the CT as the winding signal (`scripts/winding_split.py`); it cuts fewer merges and does nothing inside tight stacks, but its level sets order the sheets, which may be useful on its own.

**Where it stops:** of the merges that survive, 78 % have no probability dip between the two sheets at all, so there is nothing to cut along; knob sweeps and adding the raw CT to the signal made it worse. And on the released m7 checkpoint's own probabilities, the cut changes 3–5 of 60 cubes (−12 % to −24 % of a small count). Base m7 merges half as often as my fine-tune on these cubes, and its merges are the kind without a dip. The published surface maps are thresholded, so the tool can't even take them as input. So: useful if your model merges through thickness, close to a no-op for m7 as released. Details and every negative result in [DETAILS.md](DETAILS.md).

**Proof:** figures above; `scripts/rc_variants.py`, `scripts/rc_diag.py`, `scripts/rc_basem7.py`; raw outputs in `results/*.json`. 8 tests in `scripts/test_ridge_cut.py`.

**Why / where this is useful:** Open problem #3. Also the merge metric itself (label-sheet pairs covered by one predicted piece, with a dilated-label baseline). It was the thing that showed the fine-tune's cost in report 03.

- [x] I verified the example and proof above on the stated data.

