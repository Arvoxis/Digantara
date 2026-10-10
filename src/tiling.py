"""Lossless 1024x1024 tiling for 9568 x 6380 SSA frames.

9568 / 1024 = 9.34 and 6380 / 1024 = 6.23, so an exact grid does not fit.
Padding invents edges the annotator would have to label around; dropping the
remainder loses data. Instead the last tile in each axis is pulled flush to the
far edge and the rest are spread evenly between, giving a non-uniform grid of
tiles that are each exactly 1024x1024 and entirely inside the frame. Adjacent
tiles overlap slightly (75 px in x, 132 px in y) and every source pixel is
covered at least once.

Run this module directly to verify the round-trip is bit-exact.
"""

import numpy as np

TILE = 1024


def tile_offsets(total, tile=TILE):
    """Start offsets along one axis. Every tile is in-bounds, union covers all."""
    if tile > total:
        raise ValueError(f"tile {tile} larger than axis {total}")
    if total == tile:
        return [0]
    n = int(np.ceil(total / tile))  # tiles needed to cover
    raw = np.linspace(0, total - tile, n)  # last one flush to edge
    return sorted({int(round(v)) for v in raw})


def tile_grid(shape, tile=TILE):
    """[(y0, x0), ...] in row-major order for an image of the given shape."""
    h, w = shape[:2]
    return [(y, x) for y in tile_offsets(h, tile) for x in tile_offsets(w, tile)]


def cut(img, tile=TILE):
    """Yield (y0, x0, view) for each tile. Views, not copies."""
    for y, x in tile_grid(img.shape, tile):
        yield y, x, img[y : y + tile, x : x + tile]


def repatch(pieces, shape, tile=TILE):
    """Reassemble (y0, x0, array) pieces into a full frame.

    Overlaps are overwritten rather than blended: tiles are cut from one source
    frame, so overlapping pixels are identical and last-write-wins is exact.
    Blending would only matter if tiles had been modified independently.
    """
    pieces = list(pieces)
    out = np.zeros(shape, dtype=pieces[0][2].dtype)
    seen = np.zeros(shape[:2], bool)
    for y, x, p in pieces:
        out[y : y + tile, x : x + tile] = p
        seen[y : y + tile, x : x + tile] = True
    if not seen.all():
        raise AssertionError(f"{(~seen).sum()} pixels uncovered - tiling lost data")
    return out


def _self_check():
    h, w = 6380, 9568
    oy, ox = tile_offsets(h), tile_offsets(w)
    print(f"grid {len(ox)} x {len(oy)} = {len(ox) * len(oy)} tiles")
    print(
        f"  x offsets {ox}\n  x strides {np.diff(ox)}  overlap {TILE - np.diff(ox).max()}..{TILE - np.diff(ox).min()}"
    )
    print(
        f"  y offsets {oy}\n  y strides {np.diff(oy)}  overlap {TILE - np.diff(oy).max()}..{TILE - np.diff(oy).min()}"
    )

    for axis, offs in ((w, ox), (h, oy)):
        assert all(0 <= o and o + TILE <= axis for o in offs), "tile out of bounds"
        cov = np.zeros(axis, int)
        for o in offs:
            cov[o : o + TILE] += 1
        assert cov.min() >= 1, "gap in coverage - data loss"

    rng = np.random.default_rng(0)
    src = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    back = repatch(list(cut(src)), src.shape)
    assert back.dtype == src.dtype and np.array_equal(back, src), (
        "round-trip not bit-exact"
    )
    assert all(p.shape == (TILE, TILE) for _, _, p in cut(src)), "tile not 1024x1024"

    # odd shapes: exact multiple, smaller-than-tile axis, off-by-one
    for shape in ((2048, 1024), (1024, 1024), (1025, 3000), (6380, 9568)):
        s = rng.integers(0, 255, shape, dtype=np.uint8)
        assert np.array_equal(repatch(list(cut(s)), shape), s), (
            f"round-trip failed {shape}"
        )

    print("OK: every tile 1024x1024 and in-bounds, full coverage, bit-exact round-trip")


if __name__ == "__main__":
    _self_check()
