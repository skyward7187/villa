#!/usr/bin/env python3
"""Write a slice of a vc_render_tifxyz zarr out as a BigTIFF for GIMP.

Level 0 of this render is 133,940 x 12,820 -- 1.7 gigapixels. That fits in RAM
as uint8 once (1.7 GB), so the output plane is preallocated and filled a strip
at a time rather than read whole.

Written as a tiled, deflate-compressed BigTIFF. The render is ~90% background,
so it packs down hard despite the pixel count.
"""

import argparse
import time

import numpy as np
import tifffile
import zarr


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("zarr")
    ap.add_argument("out")
    ap.add_argument("--level", default="0")
    ap.add_argument("--slice", type=int, default=None,
                    help="z index to extract; default is the middle slice")
    ap.add_argument("--strip-rows", type=int, default=1024)
    args = ap.parse_args()

    arr = zarr.open(args.zarr, mode="r")[args.level]
    depth, height, width = arr.shape
    mid = depth // 2 if args.slice is None else args.slice
    log(f"level {args.level}: {arr.shape} {arr.dtype}")
    if not (0 <= mid < depth):
        raise SystemExit(f"slice {mid} out of range for depth {depth}")
    log(f"extracting slice {mid} of {depth}")

    out = np.zeros((height, width), np.uint8)
    t = time.time()
    for y0 in range(0, height, args.strip_rows):
        y1 = min(y0 + args.strip_rows, height)
        out[y0:y1] = np.asarray(arr[mid, y0:y1, :])
        if y0 % (args.strip_rows * 4) == 0:
            log(f"  rows {y0}/{height}")
    log(f"read in {time.time() - t:.0f}s; {100 * (out > 0).mean():.1f}% non-empty")

    t = time.time()
    tifffile.imwrite(args.out, out, bigtiff=True, tile=(256, 256),
                     compression="deflate", metadata={"axes": "YX"})
    log(f"wrote {args.out} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
