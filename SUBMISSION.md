# Submission contents and how to use them

**Assessment:** AI/ML Data Annotation Intern — blob and streak annotation of SSA imagery.
**Repository:** all source code, in `src/`. **Written response:** `RESPONSE.pdf` (3 pages).

## Start here

| file | what it is |
|---|---|
| `RESPONSE.pdf` | the written response — answers Q2 (a)–(d) with the measurements behind each decision |
| `README.md` | what the pipeline is, how to run it, how to verify it |
| `outputs/summary.csv` | per-frame results: PSF, noise scale, background range, blob/streak/marginal counts |
| `outputs/qa/*_overlay_small.png` | quickest way to see the annotations — one downsampled view per frame |
| `outputs/qa/*_streaks.png` | contact sheet of **every** streak in a frame, for individual inspection |

## The annotations

`outputs/tiles/labels/` holds **700 label files**, one per 1024 × 1024 tile
(70 tiles × 10 frames), in YOLO Ultralytics segmentation format:

```
<class> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

Polygon vertices are normalised to the tile. **`0` = blob (star), `1` = streak
(space object)**, as declared in `outputs/tiles/data.yaml`.

File names encode the tile's origin in the full frame:
`<frame>_y<row>_x<col>.txt`, so `..._y00893_x01899.txt` is the tile whose
top-left corner is at (x = 1899, y = 893).

Totals: **130 847 polygons — 130 740 blob, 107 streak.** The polygon count
exceeds the object count (107 787 blobs, 84 streaks) because tiles overlap, and
an object lying across a cut is recorded in each tile it touches, which is
correct for per-tile labels.

## The imagery

Two renditions, both generated from the same pre-processed frame:

- `outputs/tiles/images/` — the 700 tiles themselves, lossless PNG. This plus
  `labels/` is a directly loadable Ultralytics segmentation dataset.
- `outputs/repatched/*_repatched.jpg` — the **repatched full-size** 9568 × 6380
  frames with annotations drawn on, for visual inspection. These are rebuilt
  *from the exported tiles and label files*, not from memory, and each one was
  verified bit-identical to the frame rendered straight from its FITS.

Both are regenerated from the FITS by one command (about 6 minutes for all ten
frames), so if either is absent from the archive you received, run:

```bash
pip install -r requirements.txt
python src/run.py     --src <folder with the 10 .fits files> --out outputs
python src/repatch.py --tiles outputs/tiles --out outputs/repatched
```

## Reproducing and verifying

Each of these asserts its own result and exits non-zero on failure:

```bash
python src/tiling.py       # tiling is in-bounds, gapless, bit-exact round-trip
python src/export_yolo.py  # polygons normalise to [0,1], re-rasterise to IoU > 0.75
python src/calibrate.py    # false-alarm sweep; asserts zero at the operating point
python src/segment.py      # per-frame detection summary across the dataset
python src/repatch.py --verify <fits folder>   # tiles rebuild the frame exactly
python make_pdf.py         # rebuilds RESPONSE.pdf from RESPONSE.md
```

## A note on size

The full output is about 880 MB, almost all of it imagery: the frames are
61 megapixels of shot noise, which neither PNG nor JPEG can compress much
(a single full-size annotated frame is ~43 MB).

If this archive was sent without `outputs/tiles/images/` and
`outputs/repatched/`, that was to keep it mailable — everything
needed to regenerate them byte-for-byte is included, via the command above.
