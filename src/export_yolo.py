"""Export full-frame segmentation as per-tile YOLO Ultralytics segmentation labels.

Detection runs once on the whole 9568 x 6380 frame and the finished masks are
then cut into tiles, rather than detecting inside each tile separately. That
ordering is what makes boundary handling a non-problem: a streak lying across a
tile edge was segmented as one object with its true full extent, and each tile
simply records the part of that object visible inside it. Detecting per tile
would instead see two truncated fragments, measure both as too short, and
classify them as blobs.

Label format is one .txt per tile, one line per object:

    <class> <x1> <y1> <x2> <y2> ... <xn> <yn>

with polygon vertices normalised to [0, 1] against the 1024 x 1024 tile.
Class 0 = blob (star), 1 = streak (space object).
"""

from pathlib import Path

import cv2
import numpy as np

from segment import BLOB, CLASS_NAMES, STREAK
from tiling import TILE, tile_grid

MIN_POLY_AREA = 3.0  # px^2; a polygon smaller than this cannot be drawn meaningfully
APPROX_EPS = 0.4  # px tolerance for contour simplification


def mask_to_polygons(mask, eps=APPROX_EPS):
    """Outer contours of a binary mask as float arrays of (x, y) vertices."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys = []
    for c in contours:
        if cv2.contourArea(c) < MIN_POLY_AREA:
            continue
        c = cv2.approxPolyDP(c, eps, True)
        if len(c) < 3:  # not a polygon; drop it
            continue
        polys.append(c.reshape(-1, 2).astype(np.float32))
    return polys


def tile_labels(labels, cls_of, y0, x0, tile=TILE):
    """YOLO-seg label lines for one tile cut from the full-frame label image."""
    sub = labels[y0 : y0 + tile, x0 : x0 + tile]
    lines = []
    for lab in np.unique(sub):
        if lab == 0:
            continue
        cls = cls_of.get(int(lab))
        if cls is None:
            continue
        for poly in mask_to_polygons(sub == lab):
            # +0.5 puts the vertex at the pixel centre; clip guards the far edge
            norm = np.clip((poly + 0.5) / tile, 0.0, 1.0)
            # 4 dp is 1e-4 of a 1024 px tile, so 0.1 px: finer than the mask
            # boundary is meaningful. 6 dp claimed 0.001 px and cost 25% more
            # bytes across 700 files for precision the raster cannot carry.
            coords = " ".join(f"{v:.4f}" for v in norm.reshape(-1))
            lines.append(f"{cls} {coords}")
    return lines


def export_frame(frame, labels, dets, display, out_dir, tile=TILE, write_images=True):
    """Write every tile's image and label file for one frame. Returns a stats dict."""
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    cls_of = {d.label: d.cls for d in dets}
    counts = {BLOB: 0, STREAK: 0}
    empty_tiles = 0

    for y0, x0 in tile_grid(labels.shape, tile):
        stem = f"{frame.name}_y{y0:05d}_x{x0:05d}"
        lines = tile_labels(labels, cls_of, y0, x0, tile)
        for ln in lines:
            counts[int(ln.split(" ", 1)[0])] += 1
        if not lines:
            empty_tiles += 1
        (out_dir / "labels" / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        if write_images:
            cv2.imwrite(
                str(out_dir / "images" / f"{stem}.png"),
                display[y0 : y0 + tile, x0 : x0 + tile],
            )

    return {
        "tiles": len(tile_grid(labels.shape, tile)),
        "polygons": counts[BLOB] + counts[STREAK],
        "blob_polys": counts[BLOB],
        "streak_polys": counts[STREAK],
        "empty_tiles": empty_tiles,
    }


def write_data_yaml(out_dir, path_root="."):
    (Path(out_dir) / "data.yaml").write_text(
        f"path: {path_root}\ntrain: images\nval: images\n\n"
        f"names:\n  {BLOB}: {CLASS_NAMES[BLOB]}\n  {STREAK}: {CLASS_NAMES[STREAK]}\n"
    )


def _self_check():
    """Round-trip a synthetic mask through polygon export and back."""
    lab = np.zeros((TILE, TILE), np.int32)
    cv2.circle(lab, (300, 300), 9, 1, -1)  # a blob
    cv2.line(lab, (600, 400), (680, 450), 2, 3)  # a streak
    cls_of = {1: BLOB, 2: STREAK}
    lines = tile_labels(lab, cls_of, 0, 0)
    assert len(lines) == 2, f"expected 2 objects, got {len(lines)}"
    assert {int(l[0]) for l in lines} == {BLOB, STREAK}, "classes not preserved"

    for ln in lines:
        parts = ln.split()
        vals = np.array([float(v) for v in parts[1:]])
        assert len(vals) % 2 == 0 and len(vals) >= 6, "polygon needs >=3 vertices"
        assert (vals >= 0).all() and (vals <= 1).all(), "coords outside [0,1]"
        poly = (vals.reshape(-1, 2) * TILE).astype(np.int32)
        back = np.zeros((TILE, TILE), np.uint8)
        cv2.fillPoly(back, [poly], 1)
        orig = lab == (BLOB if int(parts[0]) == BLOB else 0) + (
            1 if int(parts[0]) == BLOB else 2
        )
        iou = (back & orig).sum() / max((back | orig).sum(), 1)
        assert iou > 0.75, f"polygon round-trip IoU {iou:.2f} too low"
    print("OK: polygons export, normalise into [0,1], and re-rasterise to IoU > 0.75")


def _check_exported(tile_dir="outputs/tiles", n=40, seed=0):
    """Real-data pass: every exported polygon must sit on an actual source.

    The test is on the PEAK inside the polygon, not the mean. Masks are grown to
    the 2.5 sigma isophote, so a faint source's outer pixels legitimately pull
    the mean down to the local background; the peak is what says a source is
    there. Testing the mean instead flags ~0.5% of perfectly good polygons.
    """
    import glob
    import os
    import random

    labels = sorted(glob.glob(os.path.join(tile_dir, "labels", "*.txt")))
    if not labels:
        print(f"no exported labels under {tile_dir}; skipping real-data check")
        return
    random.seed(seed)
    checked = failed = 0
    for lp in random.sample(labels, min(n, len(labels))):
        stem = Path(lp).stem
        img = cv2.imread(os.path.join(tile_dir, "images", stem + ".png"),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for line in Path(lp).read_text().splitlines():
            if not line.strip():
                continue
            pts = np.array([float(v) for v in line.split()[1:]], np.float32)
            poly = (pts.reshape(-1, 2) * TILE).astype(np.int32)
            m = np.zeros(img.shape, np.uint8)
            cv2.fillPoly(m, [poly], 1)
            inside = m.astype(bool)
            if not inside.any():
                failed += 1
                continue
            ring = cv2.dilate(m, np.ones((15, 15), np.uint8)).astype(bool) & ~inside
            local = np.median(img[ring]) if ring.any() else np.median(img)
            if img[inside].max() <= local:
                failed += 1
            checked += 1
    print(f"OK: {checked} exported polygons across {min(n, len(labels))} random tiles, "
          f"{failed} not sitting on a source")
    assert failed == 0, f"{failed} polygons do not sit on a source"


if __name__ == "__main__":
    _self_check()
    _check_exported()
