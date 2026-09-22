# Cutting sheet merges out of a thick surface map: a zero-shot winding field, a ridge-guided cut, a measured ceiling: and why it does little for the released m7

**In one sentence:** Two post-processing cuts remove the bridges where a surface model has fused neighbouring
sheets: one from a seismic "relative geologic time" network run zero-shot on scroll CT, one from non-maximum
suppression of the model's own probability along the sheet normal: and the second improves a spiral fit on
PHerc0139 (held-out winding score 0.571 → 0.605); but the merges it removes are mostly the ones our rotation
fine-tune creates, and on the released m7 checkpoint's own probabilities the cut changes 3–5 of 60 cubes.

**One real example:** Starting with the probability map of a rotation-fine-tuned m7 (report 03) on the PHerc0139
text band (z 6980–7620, y 840–5200, x 760–5040, threshold 73/255), `scripts/ridge_cut.py` removed 2.4 % of the lit
voxels in 53 min on 6 CPU cores, no GPU. Unchanged track extraction and spiral fitting, 4 seeds each: w042 score
0.571 ± 0.004 uncut → **0.605 ± 0.004** cut, every cut seed above every uncut seed; track count within 3 % of uncut.

**Before:** On 60 labelled PHerc1667 cubes the fine-tune's map had 1.62 merged sheet pairs per cube; in a 384³ box of
the band the largest connected piece held 72 % of the map.

**After this report:** 0.38 merged pairs per cube (fragments 0.78 → 1.10); largest piece 38 %. A VC3D-loadable cut
map of that box is in `results/PHerc0139_ridgecut_z7108_y2440_x2360.zarr.zip` (5 MB, the hosted maps' layout; runs
through `vc_gen_normalgrids` unchanged).

**Proof:** `figures/0139_ridge_cut_before_after_384.png` (CT | map before, colours = 3-D pieces, largest in red = 72 %
| after, 38 %); `figures/0139_ridge_cut_before_after_256.png`; `figures/0139_merge_site_example_winding_field.png`
(a merge site with the zero-shot field's level sets). Numbers below from `scripts/split_merges.py`,
`scripts/rc_variants.py`, `scripts/rc_diag.py`, `scripts/rc_basem7.py`; raw outputs in `results/*.json`.

**Why / where this is useful:** Open problem #3 (mergers between layers). For anyone whose surface model merges
sheets through thickness: which the rotation fine-tune does: the ridge cut is a free 9-minute-per-band fix before
track extraction. For the released m7 as published, it is not (see below), which is itself worth knowing.

- [x] I personally verified that the example and proof above were produced on the stated data.

## Details

### 1. A merge metric, with its cheating baseline

Count pairs of label sheets (≥ 500 vox each) where one predicted connected component covers ≥ 30 % of both; the
"tolerant" version assigns each label voxel to the nearest predicted piece within 3 vox so it also works on thin
maps. The label itself dilated by 2 vox scores 0.40 pairs/cube: the floor thickness alone charges. Base m7 scores 0.60
as scanned on the 60 cubes (strict scorer); our rotation fine-tune 0.97.

### 2. A winding-number field for free (`scripts/winding_split.py`, `scripts/zero_shot.py`)

RGT-Est (Dou et al. 2026, arXiv 2605.01273; MIT code, CC-BY 4.0 weights, 90 MB), a seismic network trained on
synthetic layered volumes, fed raw 9 µm scroll CT with no scroll training: on 20 held-out labelled cubes its level
sets are the sheets (within-sheet spread over between-sheet step 0.22 vs 1.35 for a linear ramp; beats the ramp on
20/20; sheet order Spearman ≥ 0.98 on every cube). Cutting each predicted component at the valleys of its field
histogram: **0.97 → 0.53 merged pairs (−45 %)** at +6 % fragments on the fine-tune's cubes; the ramp instead cuts 12 %
and shreds (2.55 fragments). Limit: inside tight sheet stacks the field has no valleys, so on the dense 0139 band the
3-D connectivity barely changes; the fit gain there was +0.02 (0.571 → 0.592). ~13 s per 256³ tile on a 5090.

### 3. The ridge-guided cut (`scripts/ridge_cut.py`, 8 tests)

The probability still peaks once per sheet across the sheet where the thresholded map has fused two sheets.
Non-maximum suppression along the structure-tensor normal (±1, ±2 vox) leaves one ridge per sheet; ridge voxels are
labelled 26-connected; every map voxel is assigned to the nearest ridge component within 3 vox; voxels whose 3×3×3
neighbourhood holds two assignments are removed. The map keeps its thickness (a one-voxel ridge map as fitter input
collapses the winding count: tracks need thickness). Tile-wise, 192³ tiles, 24-vox pad; the library version
reproduces the validated band cut to 1 % of the removed voxels.

Cubes, tolerant scorer: mask 1.62 merges / 0.78 frags → **ridge cut 0.38 / 1.10** (field cut 1.02). Dense 0139
windows: largest-component share 0.74 → 0.51. Fit: 0.605 (above the field cut's 0.592, all seeds).

### 4. The ceiling, measured

Of the 23 merged pairs that survive the cut on the 60 cubes, **18 (78 %) are merged in the bare ridge itself**: the
probability has no dip between the two sheets, so there is nothing to cut along. Pre-registered reading: ≥ 70 % means
the remaining merges are not a cut problem. A knob sweep (assignment radius 2, contact band 5³, both, a second pass
inside big pieces) leaves merges at 0.38. Adding the raw CT to the ridge signal made it worse (0.50 / 1.40; CT alone
0.33 merges at 6 fragments: shreds), although the CT itself still shows a gap at those sites (0.73 of the sheets' own
value; control 0.76; within-sheet 0.88): the between-sheet dip is 1–3 vox wide and any blur that kills the CT speckle
kills the dip too.

### 5. On the released m7 checkpoint, the cut is nearly a no-op

Pre-registered before writing this report: same 60 cubes, base m7's own probabilities. At the published threshold
(0.2): merged pairs 51 → 39 (−24 %), 5 cubes improved, none worse, 0.4 % of voxels removed. At the shipped threshold
73/255: 43 → 38 (−12 %), 3 cubes changed. Bar was −50 %: **FAIL**. Base m7 merges half as often as the fine-tune on
these cubes, and its remaining merges mostly have no probability dip. Two caveats cut both ways: PHerc1667 is in base
m7's training set and was held out from the fine-tune, so this comparison flatters base; and the published surface
maps on the data server are already thresholded (0/255), so the tool cannot take them as input at all: it needs the
probabilities from the inference run.

### Things that did not help (all pre-registered, all on the same band or cubes)

Splitting tracks where the field jumps (same as random cuts); capping track length (hurts at every setting: the
fitter needs long tracks); a thin ridge map as drop-in fitter input (winding count collapses); dilating the ridge
(reconnects the bridges); a scroll-native winding regressor trained on the recto labels (orders sheets but wanders
within them, 4× the field's noise); label snapping to the CT face (agreement 0.37 → 0.43, bar 0.55; the ±3-vox
wiggle in the labels is estimator noise); signing sheet normals from the field without a spiral fit (0.74 vs 0.97
for a plain radial rule on 498 k fitted-winding points).

### Costs

Ridge cut: 53 min on 6 CPU workers for the 12 G-voxel band, no GPU. Field cut: 4 h 26 min GPU for the same band.
Each 4-seed fit comparison: ~1 h on the 5090.

## Notes

<!-- CHRIS: your words. The honest headline is section 5. -->
