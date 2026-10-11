"""Run the whole annotation pipeline over a folder of FITS frames.

    python src/run.py --src Datasets_Assessment --out outputs

Produces, under --out:
    tiles/images/   1024x1024 PNG tiles of the pre-processed frames
    tiles/labels/   matching YOLO-seg .txt label files
    tiles/data.yaml Ultralytics dataset descriptor
    qa/             full-frame overlays plus a contact sheet of every streak
    summary.csv     per-frame counts and calibration values
"""

import argparse
import csv
import glob
import time
from pathlib import Path

import cv2

import preprocess
import qa
from export_yolo import export_frame, write_data_yaml
from segment import BLOB, STREAK, detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Datasets_Assessment")
    ap.add_argument("--out", default="outputs")
    ap.add_argument(
        "--limit", type=int, default=0, help="process only the first N frames"
    )
    ap.add_argument(
        "--no-tile-images",
        action="store_true",
        help="write label files but skip the ~500 MB of tile PNGs",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.src) / "*.fits")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .fits files under {args.src}")

    out = Path(args.out)
    rows = []
    for i, f in enumerate(files, 1):
        t0 = time.time()
        frame = preprocess.load_frame(f)
        labels, dets, psf = detect(frame)
        display = preprocess.to_display(frame)

        stats = export_frame(
            frame,
            labels,
            dets,
            display,
            out / "tiles",
            write_images=not args.no_tile_images,
        )
        qa.write_qa(frame, display, labels, dets, psf, out / "qa")
        cv2.imwrite(str(out / "qa" / f"{frame.name}_display_full.png"), display)

        n_blob = sum(d.cls == BLOB for d in dets)
        n_streak = sum(d.cls == STREAK for d in dets)
        longest = max((d.length for d in dets if d.cls == STREAK), default=0.0)
        rows.append(
            {
                "frame": frame.name,
                "psf_px": round(psf, 2),
                "noise_scale": round(frame.sigma, 4),
                "bg_min": round(frame.bg_range[0], 2),
                "bg_max": round(frame.bg_range[1], 2),
                "blobs": n_blob,
                "streaks": n_streak,
                "longest_streak_px": round(longest, 1),
                "tiles": stats["tiles"],
                "polygons": stats["polygons"],
                "empty_tiles": stats["empty_tiles"],
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(
            f"[{i}/{len(files)}] {frame.name[:28]:28s} psf={psf:.2f} "
            f"blobs={n_blob:6d} streaks={n_streak:3d} longest={longest:5.1f} "
            f"polys={stats['polygons']:6d} ({rows[-1]['seconds']}s)"
        )
        del frame, labels, display

    write_data_yaml(out / "tiles")
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(
        f"\nwrote {out / 'summary.csv'}  ({sum(r['polygons'] for r in rows)} polygons, "
        f"{sum(r['streaks'] for r in rows)} streaks over {len(rows)} frames)"
    )


if __name__ == "__main__":
    main()
