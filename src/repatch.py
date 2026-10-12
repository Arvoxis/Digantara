"""Rebuild full-size frames from the exported tiles, with their labels drawn on.

This is the "repatched full-size imagery for visual inspection" deliverable, and
it is built from the exported artefacts rather than from anything still in
memory: it reads the 1024 x 1024 tile PNGs and their YOLO-seg .txt files back
off disk and reassembles them. So it doubles as an end-to-end check that the
tiles and labels actually represent the frame they claim to.

    python src/repatch.py --tiles outputs/tiles --out outputs/repatched
    python src/repatch.py --verify ../Datasets_Assessment   # also compare to source

With --verify the reassembled image is compared pixel-for-pixel against the
pre-processed frame rendered straight from the FITS; they must match exactly.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from segment import STREAK
from tiling import TILE, repatch

BLOB_BGR = (255, 140, 0)
STREAK_BGR = (0, 0, 255)
STEM_RE = re.compile(r"^(?P<frame>.+)_y(?P<y>\d{5})_x(?P<x>\d{5})$")


def group_tiles(tile_dir):
    """{frame_name: [(y, x, image_path), ...]} from the exported tile file names."""
    frames = defaultdict(list)
    for p in sorted(Path(tile_dir, "images").glob("*.png")):
        m = STEM_RE.match(p.stem)
        if not m:
            raise ValueError(f"unexpected tile name: {p.name}")
        frames[m["frame"]].append((int(m["y"]), int(m["x"]), p))
    return frames


def draw_labels(canvas, label_path, y0, x0):
    """Draw one tile's polygons onto the full-frame canvas at its offset."""
    if not label_path.exists():
        return 0, 0
    nb = ns = 0
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        poly = np.array([float(v) for v in parts[1:]], np.float32).reshape(-1, 2)
        poly = (poly * TILE).astype(np.int32) + (x0, y0)
        cv2.polylines(
            canvas, [poly], True, STREAK_BGR if cls == STREAK else BLOB_BGR, 1
        )
        ns, nb = (ns + 1, nb) if cls == STREAK else (ns, nb + 1)
    return nb, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="outputs/tiles")
    ap.add_argument("--out", default="outputs/repatched")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument(
        "--verify",
        metavar="FITS_DIR",
        help="also compare the result against the source FITS",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = group_tiles(args.tiles)
    if not frames:
        raise SystemExit(f"no tiles found under {args.tiles}")

    for name, pieces in frames.items():
        grey = repatch(
            [(y, x, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for y, x, p in pieces],
            (max(y for y, _, _ in pieces) + TILE, max(x for _, x, _ in pieces) + TILE),
        )

        if args.verify:
            import preprocess

            src = Path(args.verify) / f"{name}.fits"
            ref = preprocess.to_display(preprocess.load_frame(src))
            if not np.array_equal(ref, grey):
                diff = int((ref != grey).sum())
                raise AssertionError(
                    f"{name}: repatched image differs from source in {diff} px"
                )
            print(f"  {name[:34]:34s} verified bit-exact against source FITS")

        canvas = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        nb = ns = 0
        for y, x, p in pieces:
            b, s = draw_labels(
                canvas, Path(args.tiles, "labels", p.stem + ".txt"), y, x
            )
            nb += b
            ns += s
        dst = out / f"{name}_repatched.jpg"
        cv2.imwrite(str(dst), canvas, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        print(
            f"{name[:34]:34s} {len(pieces):3d} tiles -> {grey.shape[1]}x{grey.shape[0]}  "
            f"{nb} blob + {ns} streak polygons  ({dst.stat().st_size / 1e6:.1f} MB)"
        )

    print(f"\nwrote {len(frames)} repatched frames to {out}")


if __name__ == "__main__":
    main()
