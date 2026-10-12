"""Visual QA: full-frame overlays and a contact sheet of every streak.

The streak class is rare (tens per frame against tens of thousands of blobs), so
it is cheap to inspect every single one by eye and expensive to trust a summary
statistic about them. The contact sheet exists to make that inspection a
one-image job.
"""

from pathlib import Path

import cv2
import numpy as np

from segment import BLOB, STREAK

BLOB_BGR = (255, 140, 0)  # orange
STREAK_BGR = (0, 0, 255)  # red


def overlay(display, labels, dets, alpha=0.55):
    """Colour-coded mask overlay on the 8-bit display image."""
    rgb = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    blob_ids = np.array([d.label for d in dets if d.cls == BLOB])
    streak_ids = np.array([d.label for d in dets if d.cls == STREAK])
    paint = rgb.copy()
    if blob_ids.size:
        paint[np.isin(labels, blob_ids)] = BLOB_BGR
    if streak_ids.size:
        paint[np.isin(labels, streak_ids)] = STREAK_BGR
    return cv2.addWeighted(paint, alpha, rgb, 1 - alpha, 0)


def streak_sheet(sub, dets, psf, box=96, cols=8, scale=2):
    """Contact sheet of every streak detection, brightest-stretched per cutout."""
    streaks = sorted((d for d in dets if d.cls == STREAK), key=lambda d: -d.length)
    if not streaks:
        return None
    h, w = sub.shape
    cells = []
    for d in streaks:
        cy, cx = d.centroid
        y0 = int(np.clip(cy - box // 2, 0, h - box))
        x0 = int(np.clip(cx - box // 2, 0, w - box))
        c = sub[y0 : y0 + box, x0 : x0 + box]
        lo, hi = np.percentile(c, [20, 99.5])
        g = (np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        g = cv2.cvtColor(
            cv2.resize(g, (box * scale, box * scale), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )
        cv2.putText(
            g,
            f"L{d.length:.0f} E{d.elong:.1f}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            g,
            f"{d.length / psf:.1f}xPSF snr{d.peak_snr:.0f}",
            (4, box * scale - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 200, 255),
            1,
        )
        cells.append(g)
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i : i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def write_qa(frame, display, labels, dets, psf, out_dir, shrink=6):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The full-size annotated render is produced by repatch.py instead, which
    # rebuilds it from the exported tiles and labels rather than from memory, so
    # the deliverable is demonstrably the artefact that was shipped.
    ov = overlay(display, labels, dets)
    small = cv2.resize(
        ov, (ov.shape[1] // shrink, ov.shape[0] // shrink), interpolation=cv2.INTER_AREA
    )
    # JPEG: at this size the annotation lines are visually identical to PNG
    # (checked side by side) while the set drops from 32.5 MB to 5.5 MB, which
    # is the difference between fitting in a submission upload and not.
    cv2.imwrite(str(out_dir / f"{frame.name}_overlay_small.jpg"), small,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    sheet = streak_sheet(frame.sub, dets, psf)
    if sheet is not None:
        cv2.imwrite(str(out_dir / f"{frame.name}_streaks.png"), sheet)
    return ov
