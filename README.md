<div align="center">

# SSA Blob & Streak Annotation

**Pixel-level annotation of stars and space objects in raw SSA sky imagery**

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Astropy](https://img.shields.io/badge/astropy-8.0-FF5252?logo=astropy&logoColor=white)](https://www.astropy.org/)
[![OpenCV](https://img.shields.io/badge/opencv-4.10-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.26-F7931E)](https://scikit-image.org/)
[![Format](https://img.shields.io/badge/labels-YOLO--seg-00FFFF)](https://docs.ultralytics.com/datasets/segment/)

</div>

---

## What this is

An automated annotation pipeline for the Digantara AI/ML Data Annotation assessment.
It reads raw 9568 x 6380 16-bit FITS frames, pre-processes them, segments every
source down to the noise floor, classifies each as **blob** (star) or **streak**
(space object), and exports 1024 x 1024 tiles with matching YOLO Ultralytics
segmentation labels plus full-frame overlays for visual inspection.

Annotation is scripted rather than drawn by hand because the frames contain
**3 400 to 23 600 sources each** — roughly 100 000 objects across the ten images.
At that density a manual pass is not a scheduling problem, it is impossible.

## What it needs

Python 3.11 with four packages:

```bash
pip install -r requirements.txt
```

No GPU, no network. Peak memory is about 2 GB per frame; frames are processed one
at a time, so that is the whole requirement.

## How to run it

```bash
python src/run.py     --src ../Datasets_Assessment --out outputs
python src/repatch.py --tiles outputs/tiles --out outputs/repatched                       --verify ../Datasets_Assessment
```

The first command segments, tiles and labels (about 35 s per frame, ~6 minutes
for all ten). The second rebuilds the full-size annotated frames **from the
exported tiles and label files**, and with `--verify` asserts the result is
bit-identical to the frame rendered straight from the FITS.

| flag | effect |
|---|---|
| `--src` | folder of `.fits` frames (default `Datasets_Assessment`) |
| `--out` | output folder (default `outputs`) |
| `--limit N` | process only the first N frames |
| `--no-tile-images` | write labels but skip the ~500 MB of tile PNGs |

### Output layout

```
outputs/
  tiles/images/   700 PNG tiles, 1024 x 1024, pre-processed
  tiles/labels/   700 YOLO-seg .txt files, one per tile
  tiles/data.yaml Ultralytics dataset descriptor (0 = blob, 1 = streak)
  repatched/      *_repatched.jpg      full-size annotated frame, rebuilt from tiles
  qa/             *_overlay_small.png  downsampled overview
                  *_streaks.png        contact sheet of every streak in the frame
  summary.csv     per-frame counts and calibration values
```

### Label format

One `.txt` per tile, one line per object, polygon vertices normalised to the tile:

```
<class> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

`0` = blob (star), `1` = streak (space object).

## Verifying it

Three modules carry runnable self-checks. Each exits non-zero on failure:

```bash
python src/tiling.py       # grid is in-bounds, gapless, bit-exact round-trip
python src/export_yolo.py  # polygons normalise to [0,1] and re-rasterise to IoU > 0.75
python src/calibrate.py    # false-alarm sweep; asserts zero at the operating point
python src/segment.py      # per-frame detection summary over the dataset
python src/repatch.py --verify ../Datasets_Assessment   # tiles rebuild the frame exactly
```

The first three assert and exit non-zero on failure, so they work as a test
suite. `calibrate.py` is what sets the detection threshold, and `repatch.py
--verify` proves the tiling loses nothing on the real data rather than only on
the synthetic case.

## How it works

Five stages, one module each:

| module | role |
|---|---|
| [`preprocess.py`](src/preprocess.py) | 2D background model, PSF-matched filter, robust noise scale |
| [`tiling.py`](src/tiling.py) | lossless non-uniform 1024 x 1024 grid |
| [`segment.py`](src/segment.py) | detect, deblend, classify blob vs streak |
| [`export_yolo.py`](src/export_yolo.py) | masks to normalised polygons, per tile |
| [`qa.py`](src/qa.py) | overlays and streak contact sheets |
| [`calibrate.py`](src/calibrate.py) | false-alarm measurement that sets the detection threshold |
| [`repatch.py`](src/repatch.py) | rebuilds full-size frames from the exported tiles |

Detection runs **once on the whole frame**, and the finished masks are then cut
into tiles. That ordering is what makes tile boundaries a non-issue: an object
lying across a cut was measured at its true full extent, and each tile simply
records the part visible inside it. Detecting per tile instead would see two
truncated fragments, measure both as too short, and misclassify them.

The reasoning behind each threshold — and the two hypotheses that were tested
and rejected — is in `RESPONSE.md` and in the module docstrings.

## Assumptions

- Sources are positive excursions above a smooth background; the noise scale is
  estimated from the lower half of the histogram, which sources cannot bias.
- The point-source width is measured per frame rather than assumed, so every
  geometric threshold is expressed in units of that frame's own PSF.
- Frames are treated independently. No cross-frame association is attempted; the
  consequence is recorded in `RESPONSE.md`.
