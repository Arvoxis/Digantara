"""Pre-processing for Digantara SSA frames (9568 x 6380, uint16, single band).

Three facts about this data drive the design:

1. The faint frames are Poisson counts at lambda ~ 1 ADU: 99% of pixels are 0, 1
   or 2, and the median absolute deviation of the raw image is exactly 0. Any
   threshold of the form `median + k * MAD` divides by zero. Noise must not be
   estimated on raw pixels.
2. The sky background is not flat. The frame centre runs 2-3x brighter than the
   corners in all 10 images, so one global threshold over-detects the centre and
   goes blind at the corners. A 2D background model is required.
3. Point sources are ~5 px across in every frame. Convolving with a matched
   Gaussian of that width is both the optimal filter for them and the thing that
   makes the noise continuous again, which fixes (1) as a side effect.

Run this module directly for a per-frame summary table.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from astropy.io import fits

PSF_SIGMA = 1.1  # Gaussian sigma matching the ~5 px measured point-source width
BG_BOX = 64  # background mesh: 64 px blocks -> 149 x 99 grid


@dataclass
class Frame:
    name: str
    header: fits.Header
    sub: np.ndarray  # background-subtracted, float32, ADU
    mf: np.ndarray  # matched-filtered and median-centred, float32
    sigma: float  # robust noise scale of `mf`
    bg_range: tuple  # (min, max) of the background model, for the report


def background(img, box=BG_BOX):
    """Coarse 2D background: block means, spatially median-filtered, interpolated.

    The block mean is biased by any bright source inside the block, but the 5x5
    median across blocks rejects those contaminated blocks, so the surviving
    surface tracks sky + vignetting rather than the sources sitting on it.
    """
    h, w = img.shape
    small = cv2.resize(img, (w // box, h // box), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def noise_scale(mf):
    """Robust noise scale from the negative half of the histogram.

    Sources only ever add flux, so they live in the upper half and cannot bias a
    statistic built from the lower half. This is a scale estimate, not a Gaussian
    sigma -- the filtered Poisson distribution is not Gaussian. The operating
    threshold built on it is validated empirically in calibrate.py, by counting
    detections in the negated image where every hit is a false alarm.
    """
    s = mf[::3, ::3]
    med = float(np.median(s))
    return med, float(med - np.percentile(s, 15.865))


def load_frame(path, psf_sigma=PSF_SIGMA, bg_box=BG_BOX):
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        d = hdul[0].data.astype(np.float32)

    bg = background(d, bg_box)
    sub = d - bg
    del d
    bg_range = (float(bg.min()), float(bg.max()))
    del bg

    mf = cv2.GaussianBlur(sub, (0, 0), psf_sigma)
    med, sigma = noise_scale(mf)
    mf -= med

    from pathlib import Path

    return Frame(Path(path).stem, header, sub, mf, sigma, bg_range)


def to_display(frame, lo_pct=5.0, hi_pct=99.95, soft=4.0):
    """8-bit asinh rendition of the background-subtracted frame.

    asinh rather than linear so that faint stars and bright saturated streaks are
    both visible in one image; linear at the noise, logarithmic at the bright end.
    The transfer function is computed once per frame from global statistics, so
    all 70 tiles of a frame share it and tile seams are invisible.
    """
    s = frame.sub[::7, ::7]
    lo = float(np.percentile(s, lo_pct))
    hi = float(np.percentile(s, hi_pct))
    # per-pixel noise, propagated back through the known filter gain
    sigma_px = frame.sigma * 2.0 * PSF_SIGMA * np.sqrt(np.pi)
    a = max(soft * sigma_px, 1e-6)
    den = np.arcsinh((hi - lo) / a)
    out = np.arcsinh((frame.sub - lo) / a) / den
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _summary():
    import glob
    import os
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "Datasets_Assessment"
    print(
        f"{'frame':26s} {'exp':>5} {'bg min':>8} {'bg max':>8} {'ratio':>6} "
        f"{'sigma_mf':>9} {'raw MAD':>8}"
    )
    for f in sorted(glob.glob(str(src / "*.fits"))):
        fr = load_frame(f)
        raw = fits.getdata(f)[::7, ::7].astype(np.float32)
        mad = float(np.median(np.abs(raw - np.median(raw))))
        lo, hi = fr.bg_range
        print(
            f"{fr.name[:26]:26s} {fr.header.get('EXPOSURE', 0):5.1f} "
            f"{lo:8.2f} {hi:8.2f} {hi / max(lo, 1e-3):6.2f} "
            f"{fr.sigma:9.4f} {mad:8.2f}"
        )
        assert fr.sigma > 0, f"{fr.name}: non-positive noise scale"
        assert fr.sub.shape == (6380, 9568), f"{fr.name}: unexpected shape"
        del fr
    print(
        "\nraw MAD == 0 on the faint frames is the point: the matched filter is "
        "what makes a noise scale measurable at all."
    )


if __name__ == "__main__":
    _summary()
