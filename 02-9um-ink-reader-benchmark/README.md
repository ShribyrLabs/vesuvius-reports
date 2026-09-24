# A letter-scale score for 9 µm ink reads, and six ideas that didn't move it

**In one sentence:** A score that tracks letters instead of ink blobs, calibrated on three native 9.362 µm exams, and the result of running six community ideas for 9 µm ink models through it.

**One real example:** Held-out PHerc0139 w042, hand-traced mesh, read at 9.362 µm and scored against the published 2.4 µm prediction. Plain correlation is 0.64, and it stays around 0.44 with the key shifted 150 px, so it cannot tell letters from smudge. Remove a 48 µm blur from both first and correlate what is left: the released `ink_9um` scores 0.035, my fine-tune 0.11, a rolled key 0.004.

**Before:** Reader changes got compared by plain correlation, or by eye at 1:1 where a 1 mm letter fills the screen and looks like a blob.

**After:** A scale you can place things on: released model 0.035, best current reads 0.11–0.14, a read where a person made out four letters ("ΙΟΝΤ", PHerc0814) 0.076, blobs 0.04, noise 0.01. Run-to-run noise for the same recipe is 0.005, so that is the bar.

Six ideas from the channel, each retrained against a control: window jitter, a 9-slice window, the DINO-guided 3D model on 9.6 µm input, geometry-surface-consistency cleanup, dropping the worst training keys, and the canonical 2.4 µm model fine-tuned at 9 µm. None clears 0.005. The ones that "gain" also raise the known-bad, which is smoothing, not letters.

**Proof:** `figures/0814_p46527_key_vs_reader_IONT.png` (key, reader, the letters are there), `figures/0500P2_front_key_vs_reader_blobs.png` (six lines in the key, blobs in the read), `figures/0139_w042_text_scale_key_vs_reads.png`. Scripts: `scripts/hp_score.py`. Numbers: [DETAILS.md](DETAILS.md).

**Why / where this is useful:** Before claiming a 9 µm reader improvement, or before spending weeks on a 9 µm prize scroll. One measurement explains a lot: the canonical 2.4 µm model scores 0.95 on native 2.4 µm layers and 0.11 on the same layers downsampled to 9 µm. Sampling alone removes most of the letter signal. Happy to run anyone's 9 µm checkpoint through the three exams.

**Limits:** three exams, one of them small; the keys are model outputs, not hand labels. Staff's newer `hecate` 9.6 µm model was scored on 2026-09-24, see the addendum below.

- [x] I verified the example and proof above on the stated data.

## Addendum, 2026-09-24

When I wrote this I said staff's new hecate model wasn't scored yet. It is now. They put a 9.6 µm checkpoint up on 2026-09-15, so I ran it through the same three exams as everything else here.

One wrinkle first: hecate refuses renders that aren't within 2 % of 9.6 µm, and ours are 9.362. I resampled each render to 9.6 and mapped the output back to the key grid. Just telling it the render was 9.6 also works but scores a bit lower, so I'm quoting the resampled numbers. Forward depth order on all three; reversed it scores nothing, which is how you know the input was fine.

The numbers: 0.067 on PHerc0139 w042, 0.041 on PHerc0814, 0.039 on PHerc0500P2. For comparison my fine-tuned reader gets 0.111, 0.076 and 0.041, and the released ink_9um gets 0.035, 0.035 and 0.014. The CT-copy baseline and the rolled-key nulls all stay under 0.015.

Hecate leaves a lot of the render at exactly zero and the scorer skips zeros, so I also scored both readers on identical pixels. That widens the gap rather than closing it: hecate 0.062 / 0.027 / 0.036 on my reader's pixels, my reader 0.118 / 0.122 / 0.049 on hecate's.

I tried to make it do better. Shifting the depth window four planes up or down roughly halved its score on 0814. Taking the max or mean of its 3D output instead of its 2D map was worse still. So the numbers above are its best setting, not a bad one.

Reading it plainly: hecate finds where the ink is about as well as the released model does, but it isn't pulling letters out of 9 µm data either. The ceiling I described above holds for staff's newest model too.
