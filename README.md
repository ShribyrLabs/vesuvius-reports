# Vesuvius Challenge: experiment reports, September 2026

Four short reports on real scroll data, with the numbers, the commands, the figures and the things that did not
work. Each folder is one report in the villa pull-request format (one sentence, one real example, before, after,
proof, why, details, limitations). Nothing here claims text on any scroll.

| # | Report | One line |
| --- | --- | --- |
| 01 | [PHerc0826 end-to-end First Letters run](01-pherc0826-first-letters-null/) | The published workflow run on three bands and 21 body patches of PHerc0826 with a reader validated on two unseen scrolls: no letters. Costs, gates and every failure listed. |
| 02 | [A letter-scale benchmark for 9 µm ink readers](02-9um-ink-reader-benchmark/) | A metric that tracks letters rather than blobs, calibrated on three exams; six reader levers tested against it (all null); the measured reason 9 µm reads fail. |
| 03 | [The m7 surface model misses sheets lying across the scan axis](03-m7-orientation-bias/) | Proven on proofread cubes; a rotation fine-tune fixes it on an unseen scroll but merges more sheets; what the trade-off costs. |
| 04 | [Cutting sheet merges out of surface maps](04-sheet-merge-cutting/) | A zero-shot winding field and a ridge-guided cut that remove merges from a thick surface map and improve a spiral fit: and the honest finding that on the released m7 checkpoint's own probabilities the cut is nearly a no-op. |

**Author:** Chris Scheirer (Shribyr Labs). Discord: Chris Scheirer.

**How this was built (AI assistance disclosure).** The experiments were designed, run and checked by Claude in an
agent setup I direct, on my own machine (one RTX 5090), against real scroll data from the Vesuvius Challenge open
data bucket. I reviewed the results and the text. Every number in these reports comes from a run whose command
and output are named in the report; the pre-registered pass/fail rules were written down before each run.

**What is and is not here.** Scripts are the experiment copies that produced the numbers, MIT-licensed; some carry
paths from my machine and are provided to show exactly what ran, not as a package. Figures and the one derived
data file (the 0139 ridge-cut box in report 04) come from the open data and carry its CC BY-NC 4.0 terms. Model checkpoints (the 9 µm reader fine-tune, ~0.8 GB; the m7 rotation
fine-tune, 820 MB) are not in this repository: ask and I will put them on the data server or Hugging Face.

**Machine.** One RTX 5090 (32 GB), 61 GB RAM, everything streamed from S3; no cloud compute was rented for any
of this.

**Data.** All scan data is from the Vesuvius Challenge open data bucket (`vesuvius-challenge-open-data`, CC BY-NC 4.0).
Citation, as the data page asks: Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes
Rudolph, Paul Tafforeau, Alessandro Mirone, Paul Henderson, Hendrik Schilling, Forrest McDonald, David Josey, Youssef
Nader, C. Seth Parker, W. Brent Seales. *Vesuvius Challenge - CT Scans of Herculaneum Papyri*. Vesuvius Challenge.
