# Digantara AI/ML Data Annotation Assessment — Written Response

**Dataset:** 10 raw FITS frames, 9568 × 6380, 16-bit unsigned, single band.
**Method:** scripted annotation pipeline (source in the accompanying repository).
**Result:** 700 tiles of 1024 × 1024 with YOLO-Ultralytics segmentation labels,
two classes — `0 = blob` (star), `1 = streak` (space object).

Annotation was scripted rather than drawn by hand because the frames contain
**3 400 to 23 600 sources each**, roughly 100 000 objects in total. At that
density a manual pass is not slow, it is infeasible; the engineering question is
therefore how to make an automatic decision *defensible*, which is what the four
answers below address.

---

## (a) Pre-processing steps applied

Inspection first. The ten frames are **not one homogeneous set**, and the split
drives every later choice:

| | 8 × UUID-named | 2 × `CAM_B_*` |
|---|---|---|
| Background median | 1–3 ADU | 45–46 ADU |
| Maximum value | 4095 (12-bit, clipped) | 65535 (full 16-bit) |
| Header | full observatory metadata, `static_staring` | minimal, has `GAIN`, `FRAMENO` |

Four measurements then determined the pipeline:

**1. The faint frames are Poisson counts at λ ≈ 1.** In `1a600998`, 99 % of
pixels are 0, 1 or 2 ADU. The **median absolute deviation of the raw image is
exactly 0** on five of the ten frames, so the textbook `median + k·MAD` threshold
divides by zero. Noise cannot be estimated on raw pixels here.

**2. The background is not flat.** The frame centre runs **1.88× to 3.03×**
brighter than the corners in all ten frames. A single global threshold would
over-detect the centre and go blind at the edges.

**3. There are no hot pixels.** Stacking all ten frames, **zero** pixels fire
above threshold in ≥ 8 of them. No bad-pixel map is needed — this is reported
because it was checked, not assumed.

**4. The point spread is stable**, 3.5–5.2 px across frames and cameras.

The resulting steps, in order:

1. **2D background model** — block means on a 64 px mesh, median-filtered across
   blocks to reject blocks contaminated by bright sources, bilinearly
   interpolated back to full resolution, and subtracted. This answers (2).
2. **PSF-matched filter** — convolution with a Gaussian of σ = 1.1 px matching
   the measured point-source width. This is the load-bearing step: it is both the
   optimal filter for point sources *and* the thing that makes the quantised data
   continuous again, because averaging ~15 pixels restores a measurable noise
   distribution. It answers (1).
3. **Robust noise scale** — estimated from the **lower half** of the filtered
   histogram (`median − P15.87`). Sources only ever add flux, so they live in the
   upper half and cannot bias this statistic. It is deliberately treated as a
   *scale*, not a Gaussian σ: the filtered Poisson distribution is not Gaussian,
   so the operating threshold built on it is validated empirically in (d) rather
   than by assuming a tail probability.
4. **Display rendition** — `asinh` stretch, computed once per frame from global
   statistics so all 70 tiles share one transfer function and seams are
   invisible. `asinh` rather than linear so faint stars and saturated streaks are
   both visible in one image.

No flat-field or dark frame was supplied, so neither was applied; the 2D
background model absorbs the large-scale component of both.

---

## (b) Handling image boundaries when tiling to exact 1024 × 1024

9568 / 1024 = 9.34 and 6380 / 1024 = 6.23, so an exact grid does not fit.
Padding invents edges the annotator must label around; dropping the remainder
loses data, which the question forbids.

**The grid used is non-uniform and overlapping.** The last tile on each axis is
pulled flush to the far edge and the rest are spread evenly between:

| axis | tiles | offsets | strides | overlap |
|---|---|---|---|---|
| x (9568) | 10 | 0, 949, 1899, … 8544 | 949 / 950 | 74–75 px |
| y (6380) | 7 | 0, 893, 1785, … 5356 | 892 / 893 | 131–132 px |

**70 tiles per frame, every tile exactly 1024 × 1024, every tile fully
in-bounds, zero padding, minimum coverage 1.** `src/tiling.py` asserts all four
properties and additionally verifies that `repatch(cut(x)) == x` **bit-exactly**,
on the real frame shape plus three edge cases.

The more important decision is **ordering: detection runs once on the whole
frame, and the finished masks are then cut into tiles.** The frame is only
61 Mpx and fits in memory, so this costs nothing — and it makes boundary handling
a non-problem rather than a problem to solve. An object lying across a cut was
segmented once at its true full extent; each tile simply records the part visible
inside it. Detecting per tile instead would see two truncated fragments, measure
both as shorter than the streak threshold, and silently relabel a streak as two
blobs. Re-patching is likewise exact, because overlapping pixels come from one
source frame.

---

## (c) Distinguishing short/fat streaks from blobs

**The rule.** An object is a streak when, measured on its 5σ **seed core**, the
major axis exceeds **3 × the frame's own PSF width** *and* the elongation exceeds
**2.5**. The PSF is re-measured per frame as the median major axis of
near-circular cores (3.46–5.18 px across this dataset), so both thresholds are
expressed in units of that frame's point-source size rather than as pixel counts
tuned on one camera.

Three things had to be fixed before that rule behaved, each found by measurement:

**Geometry must be measured on the seed core, not the exported mask.** The 2.5σ
isophote used for mask extent traces ragged edges on faint sources, and
raggedness reads as elongation. Classifying on it produced **1 649 "streaks"** in
one frame whose true count is near 10. Measuring on the high-SNR 5σ core and
exporting the 2.5σ extent separates "what shape is it" from "where does it end".

**The object unit must be one connected core.** Two stars sharing a single
isophote blob give a bimodal pixel set whose second moments are long and thin —
the pair reports as a streak. Making the seed core the unit of annotation, and
assigning each core its isophote territory by watershed, removes this failure
mode entirely.

**A deblend split is believed only if its pieces are round.** Bright close pairs
saturate and bridge at the seed threshold, arriving as one dumbbell, and must be
split. But **tumbling objects modulate in brightness along a streak**, which
produces exactly the same two-peaks-with-a-saddle signature, and splitting those
destroyed real streaks (longest surviving streak fell from 62.9 px to 33.6 px).
The discriminator is the *result* of the split: a star pair breaks into round
pieces, a modulated streak into pieces still elongated along the same axis. Any
shape test applied *before* splitting is circular — it protects precisely the
merged pairs it is supposed to resolve, which was confirmed by trying it.

**Two hypotheses were tested and rejected**, recorded here because the negative
results constrain the problem:

- *Streaks in a frame share an orientation.* They do not. Across a full frame
  strong streaks spread over ±20–30°, and marginal candidates aligned with them
  12 % of the time against 11 % expected by chance. Unusable.
- *The residual elongated bright sources are comatic* (the optics are f/1.3).
  They are not. Comatic flares point radially from the optical axis; the measured
  deviation from radial was 37–70°, indistinguishable from the control
  population of bright round blobs.

**Known limit, reported rather than hidden.** Objects in the **3–5 × PSF** band a
few pixels thick cannot be separated from bright stars with asymmetric wings on
single-frame morphology alone. These are labelled as streaks but carry a
`marginal` flag in `summary.csv`. Resolving them properly requires cross-frame
association — a source that moves between exposures is an object, one that does
not is a star. That was attempted on the two `CAM_B` frames (11 minutes apart):
the fields do overlap (915 of 3 821 bright sources match within 3 px, against ~39
expected by chance) but share no common translation, so the field has rotated, as
an alt-az mount would. Aligning them needs an astrometric fit, and these headers
carry no WCS. It is the right next step, not a step taken here.

---

## (d) Criteria for annotating faint or small blobs as the star class

A faintness floor is a false-alarm decision, so it was **measured, not chosen**.

Sources only add positive flux. Therefore every detection made in the **negated**
background-subtracted image is, by construction, a false alarm. Sweeping the
threshold over both signs of the image gives the false-alarm count directly, with
no assumption about the noise distribution — which matters here precisely because
the noise is Poisson and not Gaussian:

| threshold | real detections (+) | **false alarms (−)** |
|---|---|---|
| 3 σ | 11 818 – 88 604 | **622 – 8 794** |
| 4 σ | 4 641 – 41 603 | **0 – 79** |
| **5 σ** | 3 501 – 22 211 | **0 in all 10 frames** |
| 6 σ | 2 822 – 13 268 | 0 |

**The criterion is therefore: the faintest object annotated as a star is the
faintest one at which the measured whole-frame false-alarm count is zero** — a
connected group of at least **5 pixels** above **5 σ** in the PSF-matched-filtered
image. Both halves matter: the minimum-area requirement is what removes isolated
single-pixel noise spikes, and it is the standard `DETECT_MINAREA` criterion.

There is no natural "knee" in the detection-count curve to cut at — counts fall
smoothly because the field is genuinely crowded with faint stars — so a
false-alarm budget is the only defensible basis for the cut, and 6σ was rejected
as discarding real sources for no measured gain in purity.

---

## Results

| frame | PSF px | blobs | streaks | marginal | longest streak |
|---|---|---|---|---|---|
| 1a600998 | 5.18 | 3 405 | 6 | 0 | 62.9 |
| 3e6beceb | 4.00 | 9 396 | 7 | 2 | 53.8 |
| 4a8e6cd4 | 5.17 | 6 016 | 1 | 0 | 62.8 |
| 4e2cdd92 | 5.13 | 6 011 | 3 | 0 | 70.3 |
| 66ed4268 | 5.13 | 17 847 | 8 | 6 | 38.3 |
| 7030c2ac | 4.21 | 9 720 | 10 | 3 | 54.9 |
| 7d7fefcb | 5.18 | 3 500 | 8 | 0 | 67.9 |
| 7e5f3e1c | 5.14 | 6 008 | 10 | 0 | 61.7 |
| CAM_B…207 | 3.46 | 23 635 | 18 | 11 | 62.3 |
| CAM_B…276 | 3.46 | 22 249 | 13 | 9 | 59.1 |

**84 streaks and 107 787 blobs over 10 frames; 31 streaks flagged marginal.**
These become **130 847 polygons** across the 700 tiles: an object lying across a
tile cut is recorded in each tile it touches, which is correct for per-tile labels.
Every streak in every frame was inspected individually on the per-frame contact
sheets in `outputs/qa/`. The marginal flag reproduces that independent visual
judgement: frames whose streaks read as unambiguous report 0 marginal, while
`66ed4268` — where almost every candidate looked doubtful by eye — reports 6 of 8.

## Assumptions

- Sources are positive excursions above a smooth background.
- No flat-field or dark calibration frames were provided; the 2D background model
  absorbs their large-scale component only.
- Frames are treated independently; no cross-frame association is performed.
- Saturated cores are annotated at their measured extent, which overstates the
  true angular size of the brightest objects.
- The `marginal` flag is advisory. If the downstream use favours precision over
  recall on the streak class, those 31 objects are the ones to drop first.
