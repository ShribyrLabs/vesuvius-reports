# PHerc0826: First Letters workflow, three bands, no letters

**In one sentence:** I ran the team's First Letters workflow on three bands of PHerc0826 and on Ben Black's 21 body patches, with an ink reader that was first shown to read native 9 µm text on two scrolls it never trained on. Nothing.

**One real example:** Band z 3200–4200, spiral fit from the published tracks (12 min on a 5090), 71 windings rendered, both depth directions read. The mesh sits on the layers (gate 0.87–0.89; a staff mesh scores 0.955). The reader draws blobs on every winding. Same on z 2200–3200, z 4300–5300, and all 21 body patches.

**Before:** One published null on this scroll, z 10 000–11 000 (millerandmuller), with the ink threshold set on the released model's own training letters.

**After:** Four more regions are null (2 to 32 mm from one end of the roll, plus 24 spots through the body), with a reader you can trust more. On PHerc0172 and PHerc0814, which it never saw, it draws the same Greek lines as the published maps. Whoever tries 0826 next can skip these regions and this reader.

**Proof:** `figures/control_PHerc0172_w070_unseen_scroll_read.png` is what a real read looks like with this reader. `figures/0826_low_band_w030-w080_reader_montage.png` is 0826. Every number is in [DETAILS.md](DETAILS.md).

**Why / where this is useful:** If you follow the workflow guide on a 9 µm scroll, these are the things that failed silently for me, in the order I hit them:

- the first exported mesh was not on the sheets (`input_use_tracks` defaults to false; the fit had a pitch but no phase);
- the released `ink_9um` memorises its training labels on native 9 µm data, so a null from it means nothing. I had to fine-tune it first;
- after fine-tuning, the two seeds agree at 0.94 on a known-bad, so seed agreement is not evidence either;
- `--flip-normals` is required for staff meshes, and winding sense must be fitted both ways, not copied between scrolls.

**Limits:** the bands were not chosen for text and could be margin; the body of the roll is crushed and my spiral fits carry only 30–60 % of a hand-traced mesh's letter signal on a known scroll. If 0826's ink is no denser than PHerc0139's, no current 9 µm reader will show it (see report 02).

Cost: one RTX 5090, about 2.5 hours per band unattended, no cloud.

- [x] I verified the example and proof above on the stated data.

<!-- CHRIS: two or three sentences of your own here, or delete this line. -->
