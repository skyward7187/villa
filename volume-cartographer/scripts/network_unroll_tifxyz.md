# Merging a fiber network's patches into one unrolled tifxyz

The tool is `volume-cartographer/scripts/network_unroll_tifxyz.py` (untracked).
It resamples the patches of one fiber network onto the unrolled UV that
`fiber_patch_unroll.py` draws, and writes them as a single sparse tifxyz that
can be rendered directly.

Read `fiber_patch_unroll.md` first — the placement, erosion and consistency
filtering all come from there.

## Run it

```bash
cd volume-cartographer/scripts
/home/djosey/venv/bin/python network_unroll_tifxyz.py
```

Writes `/media/djosey/nvme2/network_unrolled/network01_unrolled.tifxyz`. Takes
about 10 s for 1605 patches.

Flags: `--trust-tier` (default `corroborated`), `--step` (UV pitch in ds2
voxels, default 20 — the patch grid pitch), `--oversample` (bilinear samples per
quad edge, default 3), `--erode-cells`, `--component`, `--out`, `--uuid`, plus
the usual path overrides.

## Why not `vc_merge_tifxyz`

That app does an N-surface *global* merge and derives its own parameterisation,
which is a re-flattening by another name and can disagree with the winding
assignment the consistency audit validated. Here the UV is the umbilicus unroll
itself: `u = unwrapped angle x reference radius`, `v = z`. Every patch is
already placed on a verified whole turn, so resampling is a pure change of
sampling grid and nothing is re-solved.

## Feed it the consistent set

`--trust-tier corroborated` is the default for a reason. The unfiltered set
still has ~940 overlapping pairs that disagree about the winding by a whole
turn; fusing those averages together voxels a full wrap apart. `placed` is
available for comparison, not for use.

## What it does

1. Place the network's fibers, then every patch of the chosen tier, exactly as
   `fiber_patch_unroll` does.
2. Bilinearly sample the interior of each fully valid grid quad
   (`--oversample` per edge). Scattering raw vertices instead leaves holes: a
   patch's 20-voxel vertex step is measured *along the surface*, and the unroll
   stretches it by `r_ref / r`, so a stretched patch skips UV cells.
3. Scatter the samples into a `--step`-pitch UV grid, averaging where patches
   overlap. Averaging is safe only because the tier is winding-consistent.
4. Write `zyx` with `-1` in uncovered cells via `save_tifxyz`.

Current output: **871 x 7004 grid, 134.4 x 16.7 cm, 8.9% of cells filled**,
from 1605 patches plus 119 fiber strips. The strips add ~124,000 filled cells
(+30%) and widen the extent, since fibers run beyond where patches were grown.

Sparsity is inherent — these are grown patches and thin ribbons, not a
continuous segmentation.

## Fiber strips

Each fiber's surface strip is the same ribbon the line-annotation GUI shows in
its top view: `LineViewSurfaces::lineSurface`, already a `QuadSurface`. Export
them geometry-only with the flag added for this:

```bash
vc_lasagna_line_probe <manifest.lasagna.json> --fiber <fiber.json> \
    --tifxyz-output-dir /media/djosey/nvme2/network_unrolled/strips
```

`--obj-output-dir` also emits the strip but bakes a rendered texture and so
needs a texture volume; `--tifxyz-output-dir` calls `QuadSurface::save()` and
needs no volume. Do **not** route the geometry through `vc_obj2tifxyz` — it
reparameterises onto a flattened grid, which is exactly what this pipeline
avoids.

Strips come out in *annotation* coordinates (2.4 um), unlike patches which are
ds2, so they load with scale 1.0 rather than `--patch-scale`.

The whole-turn offset is resolved **per column**, not per strip. A patch spans
far less than one turn so a single offset covers it, but a horizontal fiber
winds through many turns. The strip's centre row lies on the fiber by
construction, so each column takes the placed angle of the nearest fiber line
point. Applying one offset to a whole strip would be wrong by whole turns along
its length.

Pass `--no-strips` for patches only.

## Orientation

The grid rows are reversed on write so the tifxyz — and anything rendered from
it — reads the same way round as `network_NN_patches.png`: **z increasing
upward, unrolled length increasing to the right**. Matplotlib's y axis points
up while grid row 0 is the top of an image, so without the flip the render is a
vertical mirror of the map. No rotation is needed; both are wide.

One consequence: reversing the rows reverses the surface's handedness, so the
normal direction flips. That is invisible for a single-slice render but matters
if you ever render multiple slices along the normal — `vc_render_tifxyz` has
`--flip-normals` for that.

## Rendering it, without re-flattening

`--flatten` is off by default in `vc_render_tifxyz`, so plain rendering never
re-flattens:

```bash
build/bin/vc_render_tifxyz \
    -v /media/djosey/nvme2/PHercParis4.volpkg/volumes/s1_2um_ds2.zarr \
    --group-idx 0 --scale 1.0 \
    -s /media/djosey/nvme2/network_unrolled/network01_unrolled.tifxyz \
    --zarr-output /media/djosey/nvme2/network_unrolled/network01_render.zarr \
    --cache-gb 12
```

`--group-idx` and `--scale` are both required even though they look optional in
`--help`.

Full render size is `grid / scale` = **140,080 x 17,420** — 2.4 gigapixels, so
use `--zarr-output`, not `--tif-output`. For a quick look, render a crop at full
scale instead:

```bash
... --crop-x 24000 --crop-y 10000 --crop-width 2000 --crop-height 2000 \
    --tif-output crop.tif
```

**Do not use `--scale-segmentation` to shrink the output.** At 0.1 it reported a
normal-looking `13394x1282` render and produced an entirely black image, while
the same surface cropped at full scale rendered 82% non-empty. Whatever it does,
it is not a safe downscale for a sparse surface.

## Gotcha

`scripts/spiral/tifxyz.py` imports `einops` at module scope for functions this
tool never calls, and `/home/djosey/venv` does not have it even though the
spiral pyproject declares it. The script stubs the import rather than
reimplement `save_tifxyz`. `pip install einops` in that venv removes the stub's
reason to exist.
