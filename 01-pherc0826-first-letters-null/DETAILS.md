# PHerc0826: the First Letters workflow end to end, with a reader validated on two unseen scrolls: no letters

**In one sentence:** Three spiral-fitted bands and 21 published body patches of PHerc0826 (9.362 µm, 113 keV) were
rendered and read with the released `ink_9um` checkpoints and with a reader fine-tuned to read native 9 µm text;
every stage was first reproduced on a known scroll; the result is blobs, no rows, no glyphs, in both depth
directions.

**One real example:** Starting from the team's published tracks, umbilicus and lasagna inputs for PHerc0826, I
fitted the spiral on z 3200–4200 (30 k steps, 12 min on one RTX 5090), rendered windings w020–w090 (28 slices
each, ~1.3 min per winding) and ran the fine-tuned reader in both depth directions (8 s per winding per direction).
The mesh passes the on-layer gate at 0.87–0.89 (known-good 0.955, known-bad 0.70). The reader output is dense
round blobs with no line structure on all 71 windings.

**Before:** The only published PHerc0826 attempt (millerandmuller, z 10 000–11 000, windings 10–65) was a
pre-registered null on about 6 % of the roll's height, with the ink threshold calibrated on the released checkpoint's
own training letters.

**After this report:** Three more bands (z 2200–3200, 3200–4200, 4300–5300: 2.4 to 32 mm from one end of the
134 mm roll) and 24 GrowPatch body patches (z 6100–14 600, ~170 cm², Ben Black's release) are also null, this time
with a reader that draws letters on native 9 µm scans it never trained on. Anyone planning a PHerc0826 attempt can
skip these regions and this reader, and knows which stages of the workflow silently fail.

**Proof:** `figures/0826_low_band_w030-w080_reader_montage.png` (six windings of the best band, fine-tuned reader,
forward direction: blobs); `figures/0826_low_band_w041_hottest_two_seeds.png` (the single hottest patch on the
band, both seeds agree on the blobs); `figures/0826_mid_band_outer_windings.png`. For what a real read looks like
with the same reader: `figures/control_PHerc0172_w070_unseen_scroll_read.png` (PHerc0172, never in training:
Greek lines) and `figures/control_Paris4_pooled_9um_vs_published.png` (Paris 4 control, r 0.85 vs the published
2.4 µm map).

**Why / where this is useful:** Anyone following the team's First Letters workflow guide on a 9 µm prize scroll.
The gates and failure list below are the part I would have wanted before starting.

- [x] I personally verified that the example and proof above were produced on the stated data.

## Details

### Pipeline, and what each stage was checked against before touching 0826

1. **Spiral fit.** villa `spiral-fitting/fit_spiral.py`, 30 000 steps, tracks on, patches off, `dense_spacing_mode
   grad_mag`, winding count sized so the fitted shell reaches the scroll's outer shell (`shell_outer` loss ≈ 0 at
   the end; if it is not, the winding count is wrong). Checked on PHercParis4: the published pins reproduce the staff
   mesh. Cost: 12–16 min per 1000-slice band on the 5090.
2. **On-layer gate.** `scripts/gate_orientation.py`: structure-tensor layer normal of the CT vs the mesh normal at each
   cut point, score = fraction of usable points with |cos| > 0.8, control = the same normals rotated 90°. Known-good
   (Paris 4 staff mesh) 0.955; known-bad 0.70; rotated control ≤ 0.2. Bands: top 0.84–0.88, low 0.87–0.89, mid
   0.75–0.79.
3. **Render.** `vc_render_tifxyz` (28 slices) or the GPU port `scripts/render_gpu.py` (validated against it). Checked:
   our render of the published Paris 4 mesh matches the published surface volume slice for slice at r 0.93, and the
   published PHerc0139 w035 volume at r 1.000: **only with `--flip-normals`** for staff meshes.
4. **Reader.** Released `scrollprize/ink_9um` (seeds 42/43, steps 20 k and 75 k), then a fine-tune (below). Both
   depth directions always.
5. **Score.** `scripts/score_0826_ink.py`: fraction of sheet above 128, 99th percentile, seed-42-vs-43 agreement,
   line-pitch autocorrelation; pictures at text scale.

### The reader had to be fixed first: the released checkpoint memorises on native 9 µm data

On PHerc0139 w035 at native 9.362 µm (a training segment of `ink_9um`), the released model's two seeds agree at
0.948 inside the few chunks that carried training labels and draw crisp letters there; on the rest of the same sheet
- dense text under every patch according to the 2.4 µm key: agreement drops to 0.60 and the output is blobs. So on
native 9 µm data the released model reproduces its labels rather than reading, and a null from it means nothing.

Fix: project PHerc0139's published 2.4 µm ink predictions onto native 9 µm renders of 35 segments (registration
verified on the hand-labelled w035: r 0.588, precision 0.70 / recall 0.64 against the hand letters), fine-tune
`ink_9um` 12 000 steps (36–40 min per seed). Held-out exam, segments never in any staff list:

- 0139 w042 (dense): r vs key 0.47 → **0.65**, precision 0.48 → 0.68, recall 0.31 → 0.47.
- 0139 w050 (small, faint): r 0.34 → 0.48.
- **Unseen scroll PHerc0172 w070** (7.91 µm / 53 keV, resampled to 9.36 µm): r **0.55** against the published
  map, the same Greek lines in the same places (figure). Native 7.91 µm scored lower (0.46): the reader's home scale
  is 9.36 µm.
- **Unseen scroll PHerc0814 p46527** (9.362 µm / 113 keV: the 0826 setting): "ΙΟΝΤ" legible by eye, r 0.66.

Scripts: `scripts/build_pseudo_labels.py`, `scripts/make_finetune_config.py`, `scripts/eval_pseudo_heldout.py`.

### PHerc0826 results

| region | geometry | gate | reader output |
| --- | --- | --- | --- |
| z 2200–3200, w020–w090 | spiral fit, 100 windings | 0.84–0.88 | blobs; base-model seed agreement 0.39 (Paris 4 letters: 0.93) |
| z 3200–4200, w020–w090 | spiral fit, 100 windings | **0.87–0.89** | blobs; sheet fraction > 128 median 0.070, max 0.33 |
| z 4300–5300, w020–w090 | spiral fit | 0.75–0.79 | blobs, incl. the outer windings the gate passes |
| z 6100–14 600, 21 patches | Ben Black's GrowPatch release | 12 solid at 3 heights, 8 drifting, 1 off | blobs on all 21, both directions |

Whole-chain wall time per band (fit → render → read → score): 2 h 34 min unattended. Label build 1.5 h, two
fine-tunes 1.3 h. No cloud cost; ~17 GB of CT crop per band on local disk.

### Failures, in the order they were found

1. **The first exported mesh was not on the sheets.** `input_use_tracks` defaults to false and the patch inputs were
   empty, so the fit had pitch but no phase: on-mesh CT contrast 0.02 (random placement) vs 0.45 when the same
   points are snapped to the nearest bright voxel. Rendering it would have produced papyrus texture unrelated to
   any single sheet. Fixed by turning tracks on and checking the overlay before rendering.
2. **The point-intensity "on-sheet" gate measured nothing.** On the *validated* Paris 4 fit it gave 0.014 vs 0.51
   snapped: the same "not on sheets" verdict as the bad mesh: because a 9.6 µm "sheet" is a bundle of strands with
   air gaps, so a mid-plane point sits in a gap half the time. Replaced by the orientation gate above.
3. **Seed agreement is not evidence after fine-tuning.** The two fine-tuned seeds agree at 0.94 on a known-bad (the
   held-out w042 read with depth reversed, r 0.18). Every discriminator must be re-checked on a known-bad for every
   new model.
4. **`--flip-normals` is required for staff meshes**; without it the depth order is reversed and the read collapses
   (r 0.80 → 0.23 on Paris 4).
5. **Winding sense cannot be copied between scrolls.** 0826 is CW; the same setting on PHerc0139 was wrong (ACW),
   and the fitter does not warn. Fit both ways and keep the higher `satisfied_tracks`.
6. **Three out-of-memory crashes** killed the desktop: 12 DataLoader workers on Python 3.14 (forkserver) each copy the
   2.8 GB patch list. Use 4 workers and run every heavy step under `systemd-run --scope -p MemoryMax=`.
7. **The spiral fitter's default winding spacing (16 vox) was 40 % too small** for PHerc0139 and it does not learn
   (init 16 → 17.0; init 27.8 → 27.8): measure the sheet spacing from the CT in the band you fit and set
   `model_initial_dr_per_winding` from it. The fine deformation field is nearly dormant at the default lr scale 0.2;
   1.0 engages it.

### Limitations: what this null does and does not say

- The bands were not chosen for text; z 2200–5300 is the first 32 mm of a 134 mm roll and could be margin. The
  body of the roll (z ≳ 6000) is crushed flat and the spiral fitter degrades there; the 21 body patches sample 24
  spots on 2–3 windings each, not a continuous unroll.
- On a known scroll our automatic spiral fits carry 30–60 % of a hand-traced mesh's letter-scale signal (report 02),
  so a null on our geometry is weaker than a null on a hand-traced mesh.
- The reader transfers to two unseen scrolls, one at the 0826 setting; it has not been shown to transfer to 0826's own
  material, and on ordinary carbon ink at 9 µm it reads a few letters at best (report 02). If 0826's ink is not
  denser than PHerc0139's, no current 9 µm reader will show letters on it.
- Nothing on this scroll has been claimed as text by anyone.

## Notes

