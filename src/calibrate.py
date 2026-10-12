"""Measure the detection false-alarm rate, and with it the faintness floor.

Sources only ever add flux, so every detection made in the *negated*
background-subtracted image is a false alarm by construction. Sweeping the
threshold over both signs of the same image therefore measures the false-alarm
count directly, with no assumption about the shape of the noise distribution --
which matters here because the noise is Poisson at lambda ~ 1 and decidedly not
Gaussian, so a nominal "5 sigma" tail probability would mean nothing.

This is what sets DET_K and MIN_SEED in segment.py, and it is the evidence
behind answer (d) in RESPONSE.md. Run it to reproduce that table:

    python src/calibrate.py
"""

import glob
from pathlib import Path

import cv2
import numpy as np

import preprocess
from segment import DET_K, MIN_SEED

KS = (3.0, 4.0, 5.0, 6.0, 8.0)


def count_above(img, thr, min_area):
    """Connected groups of at least `min_area` pixels above `thr`."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((img > thr).astype(np.uint8), 8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= min_area).sum())


def calibrate_frame(frame, ks=KS, min_area=MIN_SEED):
    """[(k, real, false)] for one frame, where `false` counts negative detections."""
    x, sigma = frame.mf, frame.sigma
    return [
        (k, count_above(x, k * sigma, min_area), count_above(-x, k * sigma, min_area))
        for k in ks
    ]


def main():
    src = Path(__file__).resolve().parents[2] / "Datasets_Assessment"
    files = sorted(glob.glob(str(src / "*.fits")))
    if not files:
        raise SystemExit(f"no .fits files under {src}")

    table = {k: [] for k in KS}
    for f in files:
        frame = preprocess.load_frame(f)
        for k, real, false in calibrate_frame(frame):
            table[k].append((real, false))
        print(
            f"  {frame.name[:34]:34s} "
            + "  ".join(
                f"{k:g}s:{r}/{fa}"
                for k, (r, fa) in zip(KS, table_row(table, len(table[KS[0]]) - 1))
            )
        )
        del frame

    print(f"\nminimum connected area: {MIN_SEED} px   frames: {len(files)}\n")
    print(f"| threshold | real detections (+) | false alarms (-) |")
    print(f"|---|---|---|")
    for k in KS:
        reals = [r for r, _ in table[k]]
        falses = [f for _, f in table[k]]
        rng = f"{min(reals):,} - {max(reals):,}"
        fr = (
            "0 in all frames"
            if max(falses) == 0
            else f"{min(falses):,} - {max(falses):,}"
        )
        mark = "  <-- operating point" if k == DET_K else ""
        print(f"| {k:g} sigma | {rng} | {fr} |{mark}")

    worst = max(f for _, f in table[DET_K])
    print(
        f"\noperating point {DET_K:g} sigma, >= {MIN_SEED} px: "
        f"worst-frame false-alarm count = {worst}"
    )
    assert worst == 0, f"false alarms at the operating point: {worst}"
    print("PASS: zero measured false alarms at the operating point, every frame")


def table_row(table, i):
    return [table[k][i] for k in KS]


if __name__ == "__main__":
    main()
