# m7 misses sheets that lie across the scan axis

**In one sentence:** The m7 surface model never saw a sheet more than 30° from its training pose, so it misses sheets lying across the scan axis; fine-tuning with random axis swaps fixes that on a held-out scroll, at the price of more merged sheets.

**One real example:** Take 20 proofread recto cubes, swap the z and y axes, run the released m7: Dice drops from 0.567 to 0.266, below simply thresholding the CT (0.330). Mirror TTA doesn't help — mirrors never swap axes. In crushed regions where sheets lie flat, that is what the vertical-column blobs are (`figures/0490A_crushed_crop_ct_m7_transposed.png`).

**Before:** Tracks from the hosted map on a crushed PHerc0490A box cut across the layers: mean |track tangent · sheet normal| 0.38.

**After:** Fine-tune 6 000 steps with random permutations and flips, PHerc1667 held out entirely. On 60 unseen 1667 cubes: swapped Dice 0.29 → 0.52, as-scanned unchanged (0.549 → 0.555). Tracks on the 0490A box now follow the layers (0.19) with more tracks than base. `figures/0490A_tracks_hosted_finetune_ct.png`.

**Proof:** the two figures, `scripts/gate_eval.py` (volume-matched Dice, with CT-threshold and shifted-mesh baselines), `scripts/track_metrics.py`. Root cause is one line: nnU-Net's default augmentation rotates ±30° only. All numbers in [DETAILS.md](DETAILS.md).

**Why / where this is useful:** Anyone training a 9 µm surface model. It is a training-recipe change, not an architecture change. Checkpoint (820 MB) on request.

**The cost, measured afterwards:** the fine-tune merges neighbouring sheets more often — 0.60 → 0.97 merged label-sheet pairs per cube on the same 60 cubes. Two recipe fixes for that (the MemBrain rotations + Surface-Dice recipe; a parity/topological-interaction loss) both failed their pre-registered gates. Report 04 deals with the merges downstream.

**Limits:** a 90° swap of clean cubes is not real crushing, and there is no ground truth in a crushed 9 µm region — I tried to make one from the 2.4 µm scans and it failed its own check. Only human tracing can settle whether this helps where it matters most. Also found: the hosted maps carry a ring of "surface" in the air around the scroll (up to half the lit voxels on some 0846A planes); that is the hosting pipeline, not the model.

- [x] I verified the example and proof above on the stated data.

<!-- CHRIS: your words, or delete. -->
