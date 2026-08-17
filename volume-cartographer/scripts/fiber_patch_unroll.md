# Mapping graph patches onto the fiber unroll

Companion to `fiber_network_unroll.md`. The tool is
`volume-cartographer/scripts/fiber_patch_unroll.py` (untracked). It draws the
surface patches a fiber network runs through in the *same* unrolled frame as the
fibers, so a network reads as sheet coverage instead of bare curves.

**Read `fiber_network_unroll.md` first.** Everything about the fiber layout is
that script's and is imported, not copied: components, whole-turn placement,
smoothing, labels, legend, scale conventions. This tool adds two things only —
reading tifxyz patches, and placing them on the same frame.

## Run it

```bash
cd volume-cartographer/scripts
/home/djosey/venv/bin/python fiber_patch_unroll.py
```

Same interpreter as the unroll script. This one additionally needs **tifffile**
(present in that venv) on top of numpy + matplotlib.

Writes one PNG per run: `network_NN_patches.png`, alongside the unroll script's
`network_NN.png` in the same output directory. Different filename, so the two
tools never overwrite each other and the stale-orphan hazard in
`fiber_network_unroll.md` does not apply here — but the network *numbering* is
shared, so a link-graph change renumbers both.

## Inputs and outputs

| | path |
|---|---|
| fiber annotations (in) | `/media/djosey/nvme2/fibers/PHercParis4.volpkg.json/*.json` |
| umbilicus (in) | `/media/djosey/nvme2/PHercParis4.volpkg/umbilicus.json` |
| patches (in) | `/media/djosey/nvme2/graph_patches/*.tifxyz` |
| patch→fiber report (in) | `/media/djosey/nvme2/fiber_patch_overlap/network01_patches_to_fibers.json` |
| patch→patch report (in) | `/media/djosey/nvme2/fiber_patch_overlap/network01_patch_neighbours.json` |
| plot (out) | `/media/djosey/nvme2/fibers/PHercParis4_network_2d/network_NN_patches.png` |

All overridable: `--fibers-dir`, `--volpkg`, `--patches-dir`, `--overlap`,
`--neighbours`, `--out`. Other flags: `--component N` (default 0 = largest
network; unlike the unroll script this draws one network per run),
`--no-neighbours`, `--patch-alpha`, `--neighbour-alpha`, `--patch-edges`,
`--fit all|fibers`, `--erode-cells`, `--patch-scale`, `--trust-tier`, plus the inherited `--theme`, `--scale`,
`--smooth-mm`, `--suspect-turns`, `--dpi`.

A missing neighbour report is not an error: the run says so and draws the
fiber-carrying patches only.

## Two levels of patch

**Level 0** — patches at least one network fiber runs along, from `--overlap`.
**Level 1** — patches that overlap a level-0 patch but carry no network fiber
themselves, from `--neighbours`: one step further out along the sheet.

Level 1 draws underneath level 0, so where the two coincide the level-0 fill
wins and what reads as violet is the *extra reach* — sheet that the network
touches only indirectly.

## Where the reports come from

`--overlap` is the JSON written by `fiber_patch_overlap.py` (kept in
`/media/djosey/nvme2/fiber_patch_overlap/`). It is a patch-keyed map of which
network fibers run along which patch, measured with
`vc.surface_index.SurfacePatchIndex` — exact point-to-quad projection, the same
engine `fit_spiral`'s `link_points_to_patches` uses. Criteria baked into the
report: a fiber point counts as on the surface within **4 ds2 voxels**, and a
pair counts only with a continuous run of **18 ds2 voxels** or more.

That report is also what decides *which* patches get drawn: patches with no
fiber from this network are not in the figure at all. Regenerate it whenever the
fiber set changes materially, or the map silently under-reports coverage.

`--neighbours` is the JSON written by `patch_neighbours.py`, kept in the same
directory. It takes the level-0 patches and finds every other graph patch whose
surface lies within **2 ds2 voxels** of at least **16** of a level-0 patch's
vertices — the same method and the same defaults as
`scripts/spiral/connect_overlapping_patches.py`. It must be regenerated *after*
the patch→fiber report it reads, since level 0 defines level 1.

**`vc.surface_index` must be built** to regenerate either report: it needs
`nanobind` installed and the build configured with `-DVC_BUILD_PYTHON=ON`, then
`ninja -C build vc_surface_index`.

## How a patch gets placed

A patch is not in the link graph, so it has no whole-turn offset of its own. It
takes one from the fibers that overlap it, by the same snapping rule the link
graph uses:

1. **Self-consistent angles.** A patch spans far less than one turn — a few
   hundred ds2 voxels against a circumference of ~8000 — so raw `arctan2` angles
   are wrapped to within pi of the patch's own *circular* median. That removes
   the theta=0 seam with no unwrapping order dependence. Do not swap in
   `np.unwrap` here; a grid has no single traversal order.
2. **Anchors come from the report.** Each `(patch, fiber)` record carries the
   fiber points the surface index found *on that patch's surface*. Such a point
   is on both, so the fiber's placed angle there is the patch's placed angle
   there.
3. **Offset** = the most-voted nearest whole turn across the anchors.

A patch with no anchors in the report is skipped and named in the run output.

Two traps here, both hit once already:

- **Do not pick anchors by bounding box.** Testing which of a fiber's points
  fall inside the patch's 3D bbox looks equivalent and is not: a fiber winding
  past on a neighbouring wrap also enters that box, and votes a full turn out.
  That is why `fiber_patch_overlap.py` records anchors at all — the map cannot
  rediscover them without the index.
- **The vote must be a mode, never a mean or median.** Each anchor votes for a
  whole multiple of 2pi. `np.median` over an even number of anchors that
  disagree by one turn returns the midpoint — half a turn, a placement no anchor
  asked for, and the patch lands ~4 cm off with nothing obviously wrong in the
  numbers. `dominant_turn()` exists for this; keep using it.

## Patches that cross a wrap

The run reports how many patches reach fibers on **more than one winding**
(216 of 2115 as of 2026-08-17). This is not a placement failure. Wrap spacing at
this radius is a few hundred microns, so two fibers a millimetre apart in 3D are
several wraps apart and correctly place a turn apart; a patch touching both has
crossed a wrap. The majority winding is drawn and the minority fibers then read
as sitting a clean turn away from their patch.

Treat the count as a band-grower quality signal, not as something to tune away.

## How a neighbour gets placed

A level-1 patch carries no fiber, so there is nothing to anchor it to directly.
It borrows its whole turn from the level-0 patch it overlaps. `patch_neighbours.py`
records, for each neighbour, the level-0 vertices that lie on the neighbour's
surface — points on *both* patches. At such a point the level-0 patch's placed
angle is the neighbour's placed angle, and the offset follows by the same
nearest-whole-turn rule, median over the anchors.

This is why the neighbour report stores anchors and a `via` patch rather than
just a list of names: without the shared points the map has no way to place the
outer level.

## Border erosion (`--erode-cells`, default 1)

Every patch is shrunk by one grid ring before use — 20 ds2 voxels, 192 µm —
because band grower output is least reliable at the border, where a patch can
lip onto the neighbouring wrap. `erode_valid()` in `fiber_patch_unroll.py` is
the single implementation; the three report generators import it through
`load_patch_surface()` so the maps and the reports never disagree about what a
patch is. They honour `PATCH_ERODE_CELLS` in the environment (default 1), which
is how the comparison below was produced.

It costs 14.8% of grid vertices and empties no patch. Measured against an
otherwise identical run (same 119 fibers, erosion the only difference):

| | erode 0 | erode 1 |
|---|---|---|
| patches with fiber contact | 2345 | 2115 |
| straddle a wrap | 12.2% | **10.2%** |
| survive B | 82.1% | **84.3%** |
| trusted core (share) | 73.7% | **75.9%** |
| trusted core (count) | **1728** | 1605 |

So erosion buys about two points of purity on every measure and cuts wrap
straddling by a sixth — the border hypothesis is real. But it is a blunt
instrument at this grid pitch: 230 patches lose fiber contact altogether, and
**152 of them had passed every consistency test**. Erosion therefore *shrinks*
the absolute trusted core even as it raises the trusted fraction.

Read that trade honestly before changing the default. Those 152 touched a fiber
only inside the ring erosion removes, so their support came entirely from the
least reliable part of the surface — demoting them is defensible, losing them is
still a real cost. Run `PATCH_ERODE_CELLS=0` for maximum coverage.

Most inconsistency is *not* a border effect: the great majority of straddles, B
violations and winding disagreements survive erosion, which points at whole
patches being wrong rather than edges being sloppy.

## Consistency filtering (`--trust-tier`)

`patch_consistency.py` and `patch_trust.py` (both in
`/media/djosey/nvme2/fiber_patch_overlap/`) test the placed map against three
constraints that must hold if a patch is a genuine single-sheet surface on the
right winding, and reduce the patches to a mutually consistent core.

| | constraint | violation means |
|---|---|---|
| **A** | patches overlapping in 3D are the same sheet, so their independently derived windings must agree | one of them is on the wrong winding |
| **B** | a fiber running through a patch's footprint is on that sheet there, so it must touch it in 3D | patch or fiber is misplaced |
| **C** | footprints that clash in 2D claim the same sheet, so the surfaces must touch in 3D | as A, but weakest evidence |

3D "not touching" is judged at **8 ds2 voxels (77 µm)**. That threshold is safe
because the measured adjacent-winding spacing in this network is **349 µm
median, 121 µm at the 10th percentile** — the audit prints it. Do not raise it
much without re-measuring: past ~120 µm the test stops distinguishing "different
wrap" from "same sheet, different fit".

Tiers, each strictly inside the last (counts as of 2026-08-17):

```
0 placed                2115
1 unstraddled           1899   -216  own fibers disagree about the winding
2 no_B                  1783   -116  a fiber crosses them in 2D and misses in 3D
3 A_consistent          1755    -28  greedy removal until no overlap disagrees
4 corroborated          1605   -150  no second fiber and no agreeing neighbour
  corroborated_C_clean  1125   -480  also free of any footprint clash
```

`--trust-tier corroborated` draws only that core and writes
`network_NN_patches_corroborated.png`, so the unfiltered map is never
overwritten. Neighbours whose level-0 patch was filtered out are dropped with it.

**These are necessary conditions, not proof.** A patch in the core is one that
nothing contradicts, which is not the same as a patch that is right. Treat
`corroborated` as the working set and `corroborated_C_clean` as the cautious one.

## Silhouettes, not meshes

Each patch draws as **one polygon**: walk the grid along whichever axis has more
occupied lines, take each line's first valid vertex down one side and its last
valid vertex back up the other. At ~2000 patches per figure a per-quad mesh is
not an option, and a convex hull badly over-fills a curved band.

Accepted limitation: a grid line whose valid vertices are not contiguous has its
gap filled. This is a silhouette, not a validity mask — a patch with a real hole
in it will read as solid.

## Flat fill; depth is stacking

Every patch draws at the same colour and the same transparency
(`--patch-alpha`, default 0.3), one flat fill per level — green for level 0,
violet for level 1. Both hues sit apart from the unroll's blue horizontals and
orange verticals rather than competing with either.

Nothing is encoded in the shade. Where the fill looks denser, more patches are
stacked there: alpha compositing does that on its own, which is the whole point
of an equal-weight fill. An earlier version ramped the colour by fiber count and
was replaced — do not put it back, it made stacking and fiber count
indistinguishable.

## Hooks added to `fiber_network_unroll.py`

Rather than forking 900 lines, `draw()` there gained three generic, default-off
parameters and this script supplies them:

- `extra_bounds` — a box merged into the data bounds before the figure is sized
  (`--fit all`; `--fit fibers` passes nothing and keeps the original framing).
- `underlay(ax, shift)` — called after the axes exist, with the same left-edge
  shift applied to the fibers. Patches draw at `zorder=1`, under the fibers.
- `legend_extra` / `subtitle_extra` — prepend legend handles, append subtitle text.

With all four unset, `draw()` behaves exactly as before. Keep it that way.

## Expected output (2026-08-17, 450 fibers)

```
network 1: 117 fibers, 137 links, r ≈ 1.26 cm
  2115 patches on fibers (3255 patch–fiber pairs) from network01_patches_to_fibers.json
  1275 patches overlapping those, from network01_patch_neighbours.json
  216 patches reach fibers on more than one winding (286 fibers off the majority turn); drawn on the majority winding
  9 fibers carry no patch: dj-21, kb-178, kb-204, kb-206, kb-217, kb-227, kb-229, lt-547, lt-684
```

Sanity checks:

- **Patches must hug the fibers.** Green sitting exactly under the blue/orange
  curves is the proof that the offset snapping worked. Green offset by a clean
  whole turn from its fibers means the anchor logic failed for that patch.
- **Violet must fringe the green, not float free.** A level-1 patch overlaps a
  level-0 patch by definition, so every violet region has to touch green
  somewhere. An isolated violet island means its `via` patch was placed on the
  wrong turn.
- **The bare-fiber list is a real result, not noise.** Some of those fibers run
  where no patch was grown. But the last few names are usually just fibers newer
  than the overlap report — regenerate it before drawing conclusions.
- Unrolled patch width tracks `r_ref / r`: measured against 3D in-plane width the
  ratio is 0.96 median, 0.65–1.27 at the 5th/95th. That is the reference-radius
  approximation already documented for the fibers, nothing more.

## Gotchas

**Three different frames, three different scales.** Fiber annotations are the
finest (2.4 um/voxel). The umbilicus json is 4x coarser (`--umbilicus-scale`).
Patches are ds2, also 4x coarser (`--patch-scale`). Both scale *up* into
annotation coordinates. Getting `--patch-scale` wrong does not produce a warning
— it produces patches in the wrong place, or none placed at all.

**The overlap report and the network can disagree on membership.**
`all_components` in the unroll script follows *every* branch including
`pending` ones; `fiber_patch_overlap.py` follows the non-pending links only,
matching `spiral_helpers.resolve_fiber_links`. Network 1 is 117 fibers by the
first rule and 114 by the second. The report supplied here was generated for the
114-fiber set, which is why the run above lists more bare fibers than are truly
uncovered.
