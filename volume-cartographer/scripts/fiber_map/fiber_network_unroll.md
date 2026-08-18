# Regenerating the fiber network plots

Handoff spec for a fresh session. The tool is
`volume-cartographer/scripts/fiber_network_unroll.py` (untracked). Read this
before changing anything: most of what looks arbitrary in the script is a
settled decision, and several obvious "improvements" were tried and rejected.

## Run it

```bash
cd volume-cartographer/scripts
/home/djosey/venv/bin/python fiber_network_unroll.py
```

Use that interpreter, not `python3` — the system Python lacks the deps. The
script itself needs only numpy + matplotlib (no zarr, no scipy).

**Clear the output directory first** whenever the link graph may have changed:

```bash
rm -f /media/djosey/nvme2/fibers/PHercParis4_network_2d/network_*.png \
      /media/djosey/nvme2/fibers/PHercParis4_network_2d/top_networks.png
```

Network numbering is by descending size, so a single new link that merges two
networks renumbers everything below it and leaves a stale, orphaned
`network_NN.png` with no owner. This has already happened once.

## Inputs and outputs

| | path |
|---|---|
| fiber annotations (in) | `/media/djosey/nvme2/fibers/PHercParis4.volpkg.json/*.json` |
| umbilicus (in) | `/media/djosey/nvme2/PHercParis4.volpkg/umbilicus.json` |
| plots (out) | `/media/djosey/nvme2/fibers/PHercParis4_network_2d/network_NN.png` + `top_networks.png` |

All overridable: `--fibers-dir`, `--volpkg`, `--out`. Other flags:
`--component N` (one network; also suppresses `top_networks.png`),
`--min-fibers` (default 3), `--theme light|dark`, `--scale` (inches per data cm),
`--smooth-mm`, `--suspect-turns`, `--dpi`.

PNGs only — **do not add `.txt` sidecars**, they were explicitly removed as clutter.

## Expected output (as of 2026-08-12, 426 fibers)

```
426 fibers, 21 networks with >= 3 fibers
  network_01: 79 fibers, 94 links, worst link residual 0.00 turns
  network_02: 13 fibers, 13 links, worst link residual 0.00 turns
  network_03: 9 fibers, 10 links, worst link residual 0.00 turns
  ...
  top_networks.png: network 1 (r 1.24 cm) < network 2 (r 1.28 cm) < network 3 (r 1.46 cm)
```

Sanity checks:

- **Every link residual should be 0.00 turns.** A nonzero one is flagged red on
  the plot as winding-suspect and printed. That means a link was annotated on the
  wrong winding (or the umbilicus frame is wrong) — investigate, don't tune the
  `--suspect-turns` threshold to hide it.
- Network 1 dominates (~40% of all crossings). Fiber counts only grow as
  annotation continues; a sudden *drop* means fibers failed to load.

## How the layout works

x = unwrapped angle about the umbilicus × the network's median radius; y = z.
Per-fiber angles each carry an arbitrary multiple of 2π, so offsets are snapped
to the nearest whole turn of an already-placed neighbour by walking the link
graph (Prim-style, best-agreeing links first, so one bad link can't misplace a
fiber). Crossings then coincide and loops close **by construction** — there is no
solver and no accumulated drift.

**Do not reintroduce an intrinsic/geodesic layout.** The predecessor
(`fiber_network_2d.py`, deleted 2026-08-06) developed each fiber by
parallel-transporting a frame along lasagna normals and welded the network with
sparse least squares. Turning-angle noise integrated into heading drift, rotations
random-walked through the link graph, and networks came out tangled. Roughly 2600
lines of shrinkage, detrending, angle-reliability weighting and loop-closure
diagnostics did not fix it. The extrinsic unroll is ~30 lines and exact.

Accepted limitation: unrolling at one reference radius is metrically approximate —
x distances are exact only at `r_ref`, so wraps inside/outside it are slightly
compressed/stretched. Connectivity, ordering and winding assignment (the actual
review targets) are unaffected.

## Gotchas

**Umbilicus scale is ×4** (`DEFAULT_UMBILICUS_SCALE`). `umbilicus.json` is in a
downsampled frame (raw z 641–18144); ×4 puts it at 2564–72576, which covers the
annotation z range. Verified 2026-08-12 — do not "fix" it. 0.56% of fiber points
fall outside that z span and `np.interp` clamps them to the end center, which is
acceptable at that rate but would silently distort a fiber annotated far outside
the umbilicus. See the `umbilicus-frame-mismatch` memory.

**`line_points` overshoot the outermost control points** by over a cm on many
fibers. Those tails carry no segment metadata, so they are clipped out of the
geometry in `network_geometry()`. Everything — drawn curve, label anchors, plot
extents — must be computed from the clipped geometry, or labels float off the
visible ends (this was a real bug).

**Traced vs interpolated** is per segment: `segment_to_next.interp_mode == "trace"`
(no `failure_code`) draws solid/vivid/thick, anything else draws
faded/dashed/thin. Matches VC3D's `interp` column (`deriveTraceState()`).

## Settled design conventions

Change these only if asked:

- One fixed physical scale for every plot (`--scale`, 0.25 in per data cm), equal
  on both axes, figure sized from the data extent. Never size the figure to a
  fixed width — that silently rescales the data per network.
- Axes start at 0 on the **left**; windings numbered 0, 1, 2… left to right. No
  negative lengths or ±winding numbering.
- Titles: y = `scroll z-axis (cm)`, bottom x = `unrolled length (cm)`, top x =
  `windings`, both x titles centered on the axes.
- Winding numbers sit **above** the gridlines, outside the axes.
- Legend: `hz fiber`, `vt fiber`, `old interpolation`, `links` — no counts (they
  are in the subtitle), no gridline entry.
- Fibers are lines only: no control-point dots. Dashes are `(0, (5, 2.2))` with
  butt caps; shorter round-cap dashes fragment into dot-confetti on curved lines.
- Labels are placed as physical rectangles at whichever fiber end is nearer its
  plot edge, first candidate that is inside the axes and clear of other chips.
- `top_networks.png`: top 3 side by side, shared z axis, ordered inner→outer by
  median umbilicus radius, subtle dotted break between panels, unrolled length
  and winding count continuous across panels on one 5 cm tick grid
  (`X_TICK_CM`), spacing between networks arbitrary and labelled as such.

## Verify

Open `network_01.png` and `top_networks.png` and actually look at them. Horizontal
fibers should read horizontal, verticals vertical, crossings should sit exactly
where two fibers meet, and no label should overlap another or leave the axes.
