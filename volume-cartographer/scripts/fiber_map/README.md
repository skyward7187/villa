# Fiber map pipeline

Tools for turning hand-traced papyrus fiber annotations into a flattened map of
the sheet they lie on, checking that map against the grown surface patches, and
merging the result into a single renderable surface.

Everything here concerns **one scroll, PHerc Paris 4 (Scroll 1)**, and one
*fiber network* at a time — a connected component of the fiber link graph.

---

## The idea in one paragraph

Annotators trace horizontal and vertical fibers through the scroll volume and
link the crossings where two fibers meet on the same point of the sheet. A
connected component of those links is a piece of papyrus weave. Unroll it about
the scroll's umbilicus — `x = unwrapped angle × reference radius`, `y = z` — and
the network lies flat, with every crossing coinciding by construction. That
unrolled frame is a global surface parameterisation, so anything else known to
sit on the same sheet (the grown `graph_patches`, the fiber ribbons themselves)
can be placed on it and merged into one surface without ever running a
flattening solver.

---

## What each script does

Run order matters; this is also dependency order.

| script | reads | writes |
|---|---|---|
| `prepare_inputs.py` | fibers, patch `meta.json` | fiber point cache, patch bboxes, candidate prefilter |
| `fiber_patch_overlap.py` | that cache, patch surfaces | `networkNN_patches_to_fibers.json` |
| `patch_neighbours.py` | the overlap report | `networkNN_patch_neighbours.json` |
| `patch_consistency.py` | both reports | `networkNN_consistency.json` |
| `patch_trust.py` | the audit | `networkNN_patch_trust.json` + `.csv` |
| `fiber_network_unroll.py` | fibers, umbilicus | `network_NN.png`, `top_networks.png` |
| `fiber_patch_unroll.py` | + patch reports | `network_NN_patches.png` |
| `network_unroll_tifxyz.py` | + fiber strips | `networkNN_unrolled.tifxyz` |
| `zarr_to_tif.py` | a render zarr | a BigTIFF for GIMP |

`common.py` holds every path and the native-module import. Nothing else
hardcodes a location.

---

## Quick start

```bash
cd volume-cartographer/scripts/fiber_map

# 1. all the analysis reports, in order (~3 min)
./run_pipeline.sh

# 2. the plots
/home/djosey/venv/bin/python fiber_network_unroll.py     # fibers alone
/home/djosey/venv/bin/python fiber_patch_unroll.py       # fibers + patches

# 3. merge to one surface and render it
/home/djosey/venv/bin/python network_unroll_tifxyz.py
```

Use `/home/djosey/venv/bin/python`, not the system python — the system one has
no torch, matplotlib or tifffile. Override with `FIBER_MAP_PYTHON`.

### Configuration

All via environment variables, read in `common.py`:

| variable | default | meaning |
|---|---|---|
| `FIBER_MAP_FIBERS` | `…/fibers/PHercParis4.volpkg.json` | fiber JSON directory |
| `FIBER_MAP_VOLPKG` | `…/PHercParis4.volpkg` | holds `umbilicus.json` |
| `FIBER_MAP_PATCHES` | `…/graph_patches` | tifxyz patches |
| `FIBER_MAP_REPORTS` | `…/fiber_patch_overlap` | where reports land |
| `FIBER_MAP_PLOTS` | `…/PHercParis4_network_2d` | where PNGs land |
| `FIBER_MAP_MERGED` | `…/network_unrolled` | merged tifxyz + renders |
| `FIBER_MAP_WORK` | `<reports>/work` | caches and intermediates |
| `FIBER_MAP_NETWORK` | `0` | which network (0 = largest) |
| `PATCH_ERODE_CELLS` | `1` | border erosion, in grid rings |

---

## Prerequisites

**`vc.surface_index`** must be importable — it is the exact point-to-quad
projection every geometric test uses.

```bash
pip install nanobind
cmake -S volume-cartographer -B volume-cartographer/build -DVC_BUILD_PYTHON=ON
ninja -C volume-cartographer/build vc_surface_index
```

The build directory is prone to being reconfigured with `VC_BUILD_PYTHON=OFF`,
which deletes `build/python/`. `<reports>/vc_ext/` holds a fallback copy with
its libraries; `common.import_surface_index()` finds either, and tells you what
to run if neither is present.

**Fiber strips** (optional, for the merge) come from a VC3D app:

```bash
build/bin/vc_lasagna_line_probe <manifest.lasagna.json> \
    --fiber <fiber.json> --tifxyz-output-dir <merged>/strips
```

---

## The three frames

Getting these wrong produces plausible-looking nonsense, so they are worth
memorising:

| thing | frame | to ds2 |
|---|---|---|
| fiber annotations | full-res, 2.4 µm/voxel | × 0.25 |
| umbilicus JSON | 4× coarser than annotations | × 4 → annotation |
| graph patches | ds2, 9.6 µm/voxel | — |
| merged tifxyz | ds2 | — |

The volume to render against is `<volpkg>/volumes/s1_2um_ds2.zarr`.

---

## Reading the consistency reports

`patch_trust.py` sorts patches into nested tiers. These are **necessary
conditions, not proof**: a patch in the core is one that nothing contradicts.

```
placed                every patch a network fiber runs along
unstraddled           minus patches whose own fibers disagree about the winding
no_B                  minus patches a fiber crosses in 2D but misses in 3D
A_consistent          minus enough patches to break every winding disagreement
corroborated          minus patches with no second fiber and no agreeing neighbour
corroborated_C_clean  minus patches with any footprint clash
```

**Use `corroborated`.** It is the default for the merge, and for good reason:
the unfiltered set still contains ~940 pairs of physically overlapping patches
placed a whole turn apart, and merging those averages together voxels a full
wrap apart.

`networkNN_patch_trust.csv` gives one row per patch with its verdict and each
violation count — start there when a patch looks wrong on the map.

---

## Things that will bite you

**Run the stages in order.** `patch_trust.py` reads whatever
`patch_consistency.py` last wrote. Running the audit for one configuration and
the tiers for another silently mixes them; this cost real debugging time once,
producing 16 phantom conflicts. `run_pipeline.sh` exists to prevent it.

**Regenerate after adding fibers.** Every report is keyed to a fiber set. New
fibers are simply absent, and the map will under-report coverage while looking
completely healthy.

**Two definitions of a network.** `all_components()` in `fiber_network_unroll`
follows *every* branch including `pending` ones; `prepare_inputs.py` follows the
same rule, but `spiral_helpers.resolve_fiber_links` (used by `fit_spiral`)
excludes pending links. The same component can differ by a few fibers depending
on which rule applied.

**Whole-turn offsets must be voted by mode, never averaged.** Each anchor votes
for a multiple of 2π; a median over an even number of disagreeing anchors
returns a half-turn — a placement no anchor asked for, landing the patch ~4 cm
off with nothing obviously wrong in the numbers. `dominant_turn()` exists for
exactly this.

**Anchors must come from the reports.** Picking them by bounding-box test looks
equivalent and is not: a fiber winding past on the neighbouring wrap also enters
the box and votes a full turn out.

**A patch spans much less than a turn; a fiber strip does not.** Patches take one
offset; strips resolve one per column.

**Do not use `--scale-segmentation` to shrink a render.** At 0.1 it reports a
normal-looking size and produces an entirely black image for a sparse surface.
Render crops at full scale, or subsample the output zarr.

**`--group-idx` and `--scale` are required** by `vc_render_tifxyz` even though
`--help` presents them as optional.

---

## Deeper reading

Each stage has its own document with the reasoning, the rejected alternatives
and the measured numbers:

- `fiber_network_unroll.md` — the unroll itself, and why the intrinsic/geodesic
  predecessor was abandoned
- `fiber_patch_unroll.md` — patch placement, border erosion with its measured
  cost/benefit, and the consistency tiers
- `network_unroll_tifxyz.md` — the merge, fiber strips, orientation, rendering
