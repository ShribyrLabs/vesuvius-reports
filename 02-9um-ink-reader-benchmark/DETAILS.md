# A letter-scale benchmark for 9 µm ink readers, six levers that did not move it, and the measured reason

**In one sentence:** A number that tracks letters rather than ink regions (correlation after removing a 48 µm blur),
calibrated on three native 9.362 µm exams with a legibility anchor and a known-bad, used to test six ideas from the
channel for improving 9 µm ink reading — none cleared the noise bar — plus a measurement of how much letter signal
9 µm sampling alone destroys.

**One real example:** Starting from the released `ink_9um` and a fine-tune of it (report 01), I read the hand-traced
held-out PHerc0139 w042 mesh at native 9.362 µm and scored the read against the published 2.4 µm ink prediction
resampled to the 9 µm grid. Raw correlation is 0.64 — and stays at ~0.44 with the key shifted 150 px, so it cannot
tell letters from smudge. After a 48 µm high-pass on both, the fine-tuned reader scores **0.111**, the released
model 0.035, and a rolled key ≤ 0.004.

**Before:** Reader changes were compared by raw correlation or by eye at 1:1, where a 1 mm Greek letter fills the
frame and reads as a blob.

**After this report:** A scorer with a calibrated scale — released model 0.035, current best 0.11–0.14, a read where
a person made out four letters 0.076, blobs 0.04, null ≤ 0.01 — and a power rule (≥ ~1 M on-sheet pixels, or the
null is not distinguishable). Six community ideas have numbers against it. And one number says why the lane is hard:
the canonical 2.4 µm model scores 0.95 on native 2.4 µm layers and **0.11 on the same layers degraded to 9 µm
sampling**.

**Proof:** `figures/0814_p46527_key_vs_reader_IONT.png` (key | fine-tuned reader | round-2 reader: "ΙΟΝΤ" legible in
the reads, hp r 0.076); `figures/0500P2_front_key_vs_reader_blobs.png` (six Greek lines in the key, blobs in the
reads, hp r 0.04); `figures/0139_w042_text_scale_key_vs_reads.png` (about 10 lines of the key vs the hand-traced
mesh read and our fit's read, reduced 3×: both reads are texture fields, not legible); `figures/0139_w042_1to1_view.png`
(why 1:1 viewing misleads).

**Why / where this is useful:** Anyone claiming a 9 µm reader improvement, or deciding whether a 9 µm prize scroll
is worth a First Letters attempt. Scripts: `scripts/hp_score.py` (same-grid pred vs key), `scripts/hp_score_big.py`
(pooled tile reads mapped to the key through the reference mesh, with shift search and far-offset nulls),
`scripts/line_pitch.py` (text-line pitch autocorrelation).

- [x] I personally verified that the example and proof above were produced on the stated data.

## Details

### The metric

hp r = Pearson correlation between the read and the answer key after both are high-passed with a Gaussian of
σ ≈ 48 µm (20 px at 2.4 µm, 5 px at 9.36 µm), normalised convolution so invalid borders do not leak. "Inked hp r"
restricts to the key's inked area (key smoothed σ 96 µm > 60). Nulls from eight rolled keys. Planted shifts are
recovered 12/12 on good reads and 0/12 on weak ones at 256-px blocks.

### Calibration (all native 9.362 µm, 113 keV, forward depth order)

| exam | what it is | released `ink_9um` | fine-tuned reader | null |
| --- | --- | --- | --- | --- |
| PHerc0139 w042, hand-traced mesh, held out | dense text | 0.035 | **0.111** (inked 0.142) | ≤ 0.004 |
| PHerc0814 p46527, unseen scroll | "ΙΟΝΤ" legible by eye | 0.035 | **0.076** (inked 0.104) — the legibility anchor | ≤ 0.011 |
| PHerc0500P2 front, unseen scroll | six lines in the key, blobs in every read | — | 0.041–0.045 — the known-bad | |

Run-to-run noise for the same recipe (two trainings, same seed policy): 0.005. That is the bar.

Our own automatic spiral fits on w042 (report 01's geometry) score 0.044–0.072 on the same key pixels where the
hand-traced mesh scores 0.120–0.145: **30–60 % of the letter-scale signal**, so geometry costs signal, but the
hand-traced read is itself not legible at text scale (figure). The reader is the binding limit before the geometry.

### Six levers, all against a same-recipe control (w042 / 0814 / 0500P2 hp r)

- Window jitter ±0 / ±1 / ±2 layers (retrained): 0.108 / 0.108 / 0.106 — spread ≤ 0.002, under the noise bar.
- 9-slice window instead of 17 (retrained): 0.101 vs 0.106. The default centred window is the best single depth;
  ±2-layer shifts lower hp r on w042 and on 0814.
- Danilo Lapegna's geometry-surface-consistency post-processing, on 2-D reads and on a 5-depth stack: hp r falls on
  all three exams (w042 0.106 → 0.092–0.098); the inked-area gain appears on the known-bad too, i.e. smoothing.
- Staff's DINO-guided 3-D ink model (`scrollprize/ink_3d_dino_guided`) on 9.6 µm-pooled Paris 4: faithful at native
  2.4 µm (r 0.87 vs the published volume) but hp r 0.085 on 9.6 µm input against its own native output — the input
  scale, not the setup.
- Relabelling (drop the three lowest-agreement training keys): +0.007 on one seed, +0.002 on the other, no gain on the
  legible anchor. Marginal, not a result.
- Canonical 2.4 µm r152 model fine-tuned at 9 µm (inputs upsampled 4×): w042 0.107 vs 0.106, **known-bad 0.074 vs
  0.042** — the known-bad rises most and the "ΙΟΝΤ" letters are lost. Ensembles behave the same way: every average
  raises the known-bad roughly in proportion to w042.

### Why: 9 µm sampling alone removes most of the letter signal

Same w042 crop, same scorer, canonical 2.4 µm model: native 2.4 µm layers **0.951** (circular ceiling, the key is
this model's output); the same layers degraded to 9 µm sampling **0.113**; the real 9 µm scan **0.031**. So sampling
removes ~88 % of what the model can use, and the real scan most of the rest. The degraded volume averages noise down,
so 0.113 is an optimistic ceiling for any super-resolution or domain-translation route — exactly where the fine-tuned
9 µm reader already sits. Pre-registered rule: ≤ 0.06 closes the lane, ≥ 0.20 opens it; 0.113 is reported and no lane
opened.

A split test on PHerc0500P2 (train on half the labelled lines, test on the other half, 256-px gap) shows the same
thing from the data side: PHerc0172's ink transfers between lines (r 0.72 → 0.88), 0500P2's does not (0.33 → 0.32,
predicted-ink fraction collapsing). At 9.362 µm / 113 keV that patch carries no signal the model can generalise from.

### Costs

Each retrain: 12 000 steps ≈ 36–40 min on one RTX 5090 (4 DataLoader workers; 12 workers OOM-killed the machine
three times); patch-finding cache 44 min CPU, reused across seeds. Each exam ≈ 1 min. Total for the six levers: about
one GPU-day.

### Limitations

- Three exams, one of them a single small patch; hp r depends on how dense and clean each key is, so compare within an
  exam, never across.
- The keys are model outputs (the 2.4 µm canonical model), not human labels, except that w035's key was checked
  against staff's hand labels (r 0.59, which is roughly the ceiling for agreeing with such a key).
- Staff's newer `hecate` 9.6 µm distillation was not scored; it is the obvious next known-first on this bench.
- "Legible" calls are calibration, not evidence; the anchor is one read of four letters.

## Notes

<!-- CHRIS: your words. -->
