"""Detection, deblending and blob/streak classification.

The object unit is the *seed core*: a connected group of pixels above the 5
sigma detection threshold. Each core is one annotated object, and the 2.5 sigma
isophote around it is that object's exported extent. Where several cores sit
inside one isophote blob -- two stars close enough to touch at the low threshold
-- the blob is partitioned between them by watershed.

Two things here were arrived at by measurement rather than assumption, and both
cost a rewrite:

* Class geometry is measured on the SEED core, never on the grown mask. The low
  isophote traces ragged edges on faint sources and raggedness reads as
  elongation; classifying on it produced 1649 "streaks" in one frame against a
  true count near 10.

* Geometry must come from a single connected core, not from all seed pixels
  inside a grown blob. Two star cores sharing one isophote blob give a bimodal
  pixel set whose second moments are long and thin -- the stars get reported as
  a streak. Partitioning by core first is what removes that whole failure mode.

An orientation-consensus test was tried and rejected: streaks within a frame
spread over +/-20 to 30 degrees, so angle agreement carries no usable signal.

Run this module directly to print a per-frame detection summary.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage import measure
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

DET_K = 5.0  # seed threshold, in units of the robust noise scale
GROW_K = 2.5  # isophote defining exported mask extent
MIN_SEED = 5  # connected seed pixels required (SExtractor DETECT_MINAREA)
DEBLEND_H = 3.0  # minimum saddle depth in noise units ...
SADDLE_FRAC = 0.20  # ... or this fraction of the source peak, whichever is larger
STREAK_LEN = 3.0  # streak if major axis exceeds this many PSF widths ...
STREAK_ELONG = 2.5  # ... and is at least this elongated
# A streak comfortably past both thresholds is unambiguous. One sitting just
# inside them is flagged `marginal`: at this plate scale a 3-5 x PSF object a
# few pixels thick cannot be separated from a bright star with asymmetric wings
# on single-frame morphology alone. The flag is reported, not silently resolved.
CLEAR_LEN = 4.0
CLEAR_ELONG = 3.0

BLOB, STREAK = 0, 1
CLASS_NAMES = {BLOB: "blob", STREAK: "streak"}


@dataclass
class Detection:
    label: int
    cls: int
    length: float  # major axis of the seed core, px
    elong: float
    area: int  # exported mask area, px
    centroid: tuple
    peak_snr: float
    marginal: bool = False


def _seed_mask(mf, sigma):
    """Pixels above DET_K sigma, in groups of at least MIN_SEED."""
    seeds = (mf > DET_K * sigma).astype(np.uint8)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(seeds, 8)
    too_small = np.where(stats[:, cv2.CC_STAT_AREA] < MIN_SEED)[0]
    seeds[np.isin(lab, too_small)] = 0
    return seeds.astype(bool)


def _grown_mask(mf, sigma, seeds):
    """GROW_K isophote blobs that contain at least one seed."""
    lab = measure.label(mf > GROW_K * sigma, connectivity=2)
    keep = np.unique(lab[seeds])
    return np.isin(lab, keep[keep > 0])


def split_cores(mf, sigma, seeds, psf):
    """Split seed cores that hold two peaks separated by a deep saddle.

    Bright close pairs saturate and bridge at the 5 sigma threshold, arriving as
    one dumbbell-shaped core; by second moments that is long and thin and reads
    as a streak. In one frame 16 of 17 reported streaks were such pairs, with
    peak SNR up to 1951.

    The saddle depth required scales with the source's own peak rather than
    being a fixed multiple of sigma. A fixed 3 sigma works for faint pairs but
    shreds bright streaks, whose ridges ripple by far more than that along their
    length; requiring the dip to be a fraction of the peak keeps a uniform ridge
    intact while still splitting a genuine pair. Unequal pairs where the fainter
    star sits below the saddle fraction are a known miss, noted in the report.
    """
    lab = measure.label(seeds, connectivity=2)
    out = lab.copy()
    nxt = int(lab.max()) + 1
    min_core = max(MIN_SEED, int(np.ceil(np.pi * (psf / 2) ** 2)))

    for p in measure.regionprops(lab):
        if p.area < 2 * min_core:
            continue
        y0, x0, y1, x1 = p.bbox
        patch = mf[y0:y1, x0:x1] * p.image
        h = max(DEBLEND_H * sigma, SADDLE_FRAC * float(patch.max()))
        peaks = measure.label((h_maxima(patch, h) > 0) & p.image, connectivity=2)
        if peaks.max() < 2:
            continue
        ws = watershed(-patch, peaks, mask=p.image)
        # Believe the split only if it produced stars. Two merged stars break
        # into two round pieces; a streak whose brightness is modulated along
        # its length -- a tumbling object glinting, which is common here --
        # breaks into pieces that are each still elongated along the same axis,
        # and that is the signature used to reject the split and keep it whole.
        if any(
            b.axis_major_length / max(b.axis_minor_length, 1e-3) >= STREAK_ELONG
            for b in measure.regionprops(ws)
        ):
            continue
        region = out[y0:y1, x0:x1]
        for k in range(2, int(peaks.max()) + 1):  # basin 1 keeps the old id
            region[(ws == k) & p.image] = nxt
            nxt += 1
    return out


def measure_psf(seeds):
    """Self-calibrating PSF width: median major axis of near-circular cores.

    Measured per frame, so every geometric threshold below is expressed in units
    of that frame's own point-source size rather than a pixel count tuned on one
    camera. Across these 10 frames it ranges 3.5 to 5.2 px.
    """
    props = measure.regionprops(measure.label(seeds, connectivity=2))
    widths = [
        p.axis_major_length
        for p in props
        if p.area >= MIN_SEED
        and p.axis_major_length / max(p.axis_minor_length, 1e-3) < 1.3
    ]
    if not widths:
        raise ValueError("no compact sources found; cannot calibrate PSF")
    return float(np.median(widths))


def partition(mf, cores_lab, grown):
    """One label per seed core; each core owns the isophote territory around it.

    Returns (labels, cores) where `cores` holds the same ids restricted to seed
    pixels, so class geometry can be read off a single connected core.
    """
    blobs = measure.label(grown, connectivity=2)
    labels = np.zeros_like(blobs)
    next_label = 1

    for p in measure.regionprops(blobs):
        y0, x0, y1, x1 = p.bbox
        comp = p.image
        # cores_lab already carries one id per core, splits included; renumber
        # the ids present in this blob to 1..n so they can seed the watershed
        local = np.where(comp, cores_lab[y0:y1, x0:x1], 0)
        ids = np.unique(local)
        ids = ids[ids > 0]
        core = np.zeros_like(local)
        for i, cid in enumerate(ids, start=1):
            core[local == cid] = i
        n = len(ids)
        region = labels[y0:y1, x0:x1]
        if n == 0:
            continue
        if n == 1:
            region[comp] = next_label
            next_label += 1
            continue
        # several cores share this isophote blob: split it between them
        ws = watershed(-mf[y0:y1, x0:x1], core, mask=comp)
        for k in range(1, n + 1):
            region[(ws == k) & comp] = next_label
            next_label += 1

    return labels, np.where(cores_lab > 0, labels, 0)


def detect(frame):
    """Full-frame detection. Returns (label_image, [Detection], psf_width)."""
    mf, sigma = frame.mf, frame.sigma
    seeds = _seed_mask(mf, sigma)
    psf = measure_psf(seeds)
    cores_lab = split_cores(mf, sigma, seeds, psf)
    labels, cores = partition(mf, cores_lab, _grown_mask(mf, sigma, seeds))

    geom = {p.label: p for p in measure.regionprops(cores)}
    dets = []
    for p in measure.regionprops(labels, intensity_image=frame.sub):
        c = geom.get(p.label)
        if c is None or c.area < MIN_SEED:
            continue
        length = float(c.axis_major_length)
        elong = length / max(float(c.axis_minor_length), 1e-3)
        cls = STREAK if (length > STREAK_LEN * psf and elong > STREAK_ELONG) else BLOB
        marginal = cls == STREAK and (length < CLEAR_LEN * psf or elong < CLEAR_ELONG)
        dets.append(
            Detection(
                p.label,
                cls,
                length,
                elong,
                int(p.area),
                (float(p.centroid[0]), float(p.centroid[1])),
                float(p.intensity_max / sigma),
                marginal,
            )
        )
    return labels, dets, psf


def _summary():
    import glob
    from pathlib import Path

    import preprocess

    src = Path(__file__).resolve().parents[2] / "Datasets_Assessment"
    print(
        f"{'frame':26s} {'PSF':>5} {'total':>7} {'blobs':>7} {'streaks':>8} "
        f"{'marginal':>9} {'longest':>8} {'medSNR':>7}"
    )
    for f in sorted(glob.glob(str(src / "*.fits"))):
        fr = preprocess.load_frame(f)
        labels, dets, psf = detect(fr)
        st = [d for d in dets if d.cls == STREAK]
        longest = max((d.length for d in st), default=0.0)
        print(
            f"{fr.name[:26]:26s} {psf:5.2f} {len(dets):7d} {len(dets) - len(st):7d} "
            f"{len(st):8d} {sum(d.marginal for d in st):9d} {longest:8.1f} "
            f"{np.median([d.peak_snr for d in dets]):7.1f}"
        )
        assert len(dets) > 0, f"{fr.name}: no detections"
        assert all(d.area > 0 for d in dets), f"{fr.name}: empty mask region"
        # every label carries exactly one connected core, by construction
        assert len(dets) == len(np.unique(labels)) - 1, (
            f"{fr.name}: label/detection mismatch"
        )
        del fr, labels


if __name__ == "__main__":
    _summary()
