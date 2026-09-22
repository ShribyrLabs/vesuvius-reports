# The m7 surface model misses sheets lying across the scan axis; a rotation fine-tune fixes that on an unseen scroll, and what it costs

**In one sentence:** `scrollprize/surface_m7_nnunet` scores below a plain CT threshold on sheets that lie across the
scan axis, because it never saw one in training; fine-tuning it with random axis permutations restores those sheets on
a held-out scroll without losing the as-scanned score — but the fine-tune merges neighbouring sheets more often, and
the two obvious recipe fixes for that both failed their gates.

**One real example:** Starting from the proofread recto cubes (`seg-derived-recto-surfaces`), I swapped the z and y
axes of 20 held-out cubes and ran the released m7: Dice fell from 0.567 to 0.266, below thresholding the CT at the
same volume (0.330). Eight-way mirror test-time augmentation does not help (0.234 → 0.239), because mirrors never swap
axes. After fine-tuning 6 000 steps with random permutations and flips (PHerc1667 held out entirely), the swapped-cube
Dice on 60 unseen 1667 cubes is 0.516 (from 0.290) and the as-scanned Dice is unchanged (0.549 → 0.555).

**Before:** In crushed regions where sheets lie flat, m7 draws vertical columns where the CT shows layers
(`figures/0490A_crushed_crop_ct_m7_transposed.png`, PHerc0490A: CT | m7 | m7 on the z↔y-transposed crop | x↔z). Tracks
extracted from the hosted map cut across the layers.

**After this report:** The fine-tune's tracks on a held-out crushed PHerc0490A box (73 % of it flat-lying sheets)
follow the layers: mean |track tangent · CT sheet normal| 0.377 hosted → 0.188 fine-tune, with more tracks than base
(59 k vs 54 k). A thresholded CT scores 0.189 but gives 27 % fewer tracks. `figures/0490A_tracks_hosted_finetune_ct.png`.

**Proof:** the two figures above; `figures/0139_transpose_known_bad.png` (the same transpose on PHerc0139, where sheets
run along z, turns clean sheets into blobs — so the transpose is a diagnostic, not a fix); numbers below from
`scripts/gate_eval.py` (volume-matched Dice on held-out cubes, as scanned and swapped, with the CT-threshold and
shifted-mesh cheating baselines) and `scripts/track_metrics.py`.

**Why / where this is useful:** Anyone training or fine-tuning a 9 µm surface model. The root cause is one line: nnU-Net's
default augmentation rotates by ±30° only (`configure_rotation_dummyDA_mirroring_and_inital_patch_size`), so a
sheet more than 30° from its training pose is out of distribution. The fix is a training-recipe change, not an
architecture change. Scripts: `scripts/m7_label_train.py` (the fine-tune), `scripts/m7_orient.py` (transpose test),
`scripts/predict_surface_2um.py` (inference reading the normalisation from the plans). Checkpoint (820 MB) on request.

- [x] I personally verified that the example and proof above were produced on the stated data.

## Details

### Also found: an air-ring artefact in the hosted maps

The hosted `m7-L0-th0.2` maps carry a blocky stair-stepped band of "surface" in the air outside the scroll — 42–56 %
of all lit voxels on PHerc0846A planes. Rerunning the same checkpoint on a PHerc0490A edge crop lights 0.1 % of air
voxels vs 8.8 % hosted, so it is the hosting pipeline, not the model. Published tracks are barely affected (0.3–0.8 %
of points outside the scroll). `figures/0490A_hosted_map_air_ring.png`.

### The fine-tune, and its gate

`scripts/m7_label_train.py`: each step draws a labelled cube, a random 192³ crop (reflect-padded, padding excluded
from the loss), a random axis permutation and flips, cross-entropy + Dice, 6 000 steps from the released weights. Gate
(pre-registered): as-scanned Dice ≥ base − 0.01 AND swapped Dice ≥ CT-echo + 0.10, volume-matched, on cubes of a scroll
not in training.

Leave-PHerc1667-out, 60 unseen cubes: as scanned 0.549 → 0.555; swapped 0.290 → **0.516** (CT threshold 0.325).
A first run trained with 1667 in the set looked better as scanned (0.686) — that was same-scroll leakage, and it is
why the held-out version is the one reported.

### Spiral fits on PHerc0139

Text band, same fitter config, 4 seeds each, hosted tracks vs fine-tune tracks from the same box, scored as the share
of the fitted winding within 6 voxels of the hand-traced sheet: w042 0.562 vs 0.571 (tie); w035 0.334 ± 0.124 vs
0.486 ± 0.061 (one-sided permutation p 0.04, rank test 0.057 — better on average, borderline).

### The cost: more merged sheets

Measured after posting the fix, with a merge metric on the same 60 cubes (label-sheet pairs ≥ 500 vox both covered
≥ 30 % by one predicted connected component; the label itself dilated by 2 vox scores 0.40, the floor thickness alone
charges): base m7 **0.60** merged pairs per cube as scanned, the fine-tune **0.97**; swapped 1.72 vs 2.20. Recall was
bought with merges.

Two recipe fixes, both pre-registered, both FAILED: (a) the MemBrain-seg recipe (arbitrary rotations + Surface-Dice
loss) cuts merges 72–81 % and gains +0.04 as scanned, but thickens sheets (volume/label 1.68–1.85 vs 1.55) and loses
swapped Dice on one seed; (b) a parity two-colouring + Topological Interaction loss changes merges by −18 % on one
seed and +15 % on the other and loses Dice on one. Ablating the recipe (rotation-only; Surface-Dice-only) fails the
same way. The merges are handled downstream instead (report 04).

### Limitations

- The swap test turns non-crushed cubes 90°; that is not real crushing. There is no ground truth in a crushed 9 µm
  region: an attempt to make one from the 2.4 µm scans with the 2 µm surface model failed its own known-good (that
  model self-agrees only 0.35 under axis swap on crushed sheets). Only human tracing can settle whether the fine-tune
  helps where it matters most.
- One fitted band on one scroll for the downstream test; the w035 gain is inside that sheet's usual seed spread.
- The fine-tune's extra merges are real and cost a spiral fit if not cut (report 04).

## Notes

<!-- CHRIS: your words. This is the one you posted in #robots on 09-14; say what you'd say now. -->
