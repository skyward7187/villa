#!/usr/bin/env python3
"""Merge a fiber network's patches into one sparse tifxyz on the unrolled UV.

The unroll that ``fiber_patch_unroll.py`` draws is already a flattening: every
patch sits at a verified whole-turn offset, so (unwrapped angle x reference
radius, z) is a global surface parameterisation of the network's sheet.  This
resamples the patches onto that grid and writes the result as a single tifxyz.

Nothing here re-flattens.  The UV comes straight from the umbilicus unroll, so
the merged surface inherits exactly the winding assignment the consistency audit
validated -- which is the whole reason to build it this way rather than with
``vc_merge_tifxyz``, whose global merge derives its own parameterisation.

Feed it the *consistent* patch set.  ``--trust-tier corroborated`` is the
default: the unfiltered set still has ~950 overlapping pairs that disagree about
the winding by a whole turn, and fusing those would average together voxels a
full wrap apart.

Rendering the result, without re-flattening (``--flatten`` is off by default):

    vc_render_tifxyz -v /media/djosey/nvme2/PHercParis4.volpkg/volumes/s1_2um_ds2.zarr \\
        -s <out>/network01_unrolled.tifxyz --tif-output render.tif

Needs numpy, tifffile, Pillow and torch (the last only because it imports
scripts/spiral/tifxyz.py for save_tifxyz).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "spiral"))

from fiber_network_unroll import (
    DEFAULT_UMBILICUS_SCALE,
    DEFAULT_VOXEL_UM,
    TWO_PI,
    umbilicus_center,
    all_components,
    build_fiber,
    collect_links,
    component_xy,
    load_fibers,
    load_umbilicus,
    place_component,
)
from fiber_patch_unroll import (
    DEFAULT_PATCH_SCALE,
    load_overlap,
    load_patch_grid,
    load_trust,
    patch_angles,
    patch_offset,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strip_theta(xyz, valid, fb, umb, patch_scale):
    """Placed angle for every vertex of one fiber's surface strip.

    A patch spans far less than a turn, so one whole-turn offset covers it.  A
    fiber strip does not: a horizontal fiber winds through many turns, so the
    offset has to be resolved per column.  The strip's centre row lies on the
    fiber by construction, so each column takes the placed angle of the fiber
    line point nearest its centre vertex, and the whole column inherits that
    turn -- a column spans a few hundred microns across the sheet, far less than
    the angle where the choice of turn could differ.
    """
    rows, cols = valid.shape
    mid = rows // 2
    ctr = umbilicus_center(umb, xyz[..., 2].reshape(-1))
    rel = xyz[..., :2].reshape(-1, 2) - ctr
    raw = np.arctan2(rel[:, 1], rel[:, 0]).reshape(rows, cols)

    centre = xyz[mid]
    ok = valid[mid]
    theta = np.full((rows, cols), np.nan)
    # nearest fiber point per column, vectorised in blocks to bound memory
    idx = np.empty(cols, np.int64)
    for a in range(0, cols, 256):
        b = min(a + 256, cols)
        d = np.linalg.norm(fb.line[None, :, :] - centre[a:b, None, :], axis=2)
        idx[a:b] = d.argmin(axis=1)
    placed = fb.theta_line[idx] + fb.offset
    turn = np.round((placed - raw[mid]) / TWO_PI) * TWO_PI
    theta[:, ok] = raw[:, ok] + turn[ok]
    return theta


def quad_samples(xyz, valid, theta, k):
    """Bilinear samples inside every fully valid grid quad.

    Scattering the raw vertices leaves holes: a patch vertex step is 20 ds2
    voxels *along the surface*, which the unroll stretches by r_ref/r, so a
    stretched patch skips UV cells.  Sampling the quad interiors instead keeps
    the resampled surface gap-free wherever the patch itself is.

    Returns (theta, z, xyz) for every sample.
    """
    ok = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    if not ok.any():
        return None
    r, c = np.nonzero(ok)
    # corner stacks: (n, 3) each
    p00, p10 = xyz[r, c], xyz[r + 1, c]
    p01, p11 = xyz[r, c + 1], xyz[r + 1, c + 1]
    t00, t10 = theta[r, c], theta[r + 1, c]
    t01, t11 = theta[r, c + 1], theta[r + 1, c + 1]
    f = (np.arange(k) + 0.5) / k
    a, b = np.meshgrid(f, f, indexing="ij")
    a, b = a.ravel()[None, :, None], b.ravel()[None, :, None]
    pts = ((1 - a) * (1 - b) * p00[:, None, :] + a * (1 - b) * p10[:, None, :]
           + (1 - a) * b * p01[:, None, :] + a * b * p11[:, None, :])
    a2, b2 = a[..., 0], b[..., 0]
    th = ((1 - a2) * (1 - b2) * t00[:, None] + a2 * (1 - b2) * t10[:, None]
          + (1 - a2) * b2 * t01[:, None] + a2 * b2 * t11[:, None])
    pts = pts.reshape(-1, 3)
    return th.reshape(-1), pts[:, 2], pts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fibers-dir", type=Path,
                    default=Path("/media/djosey/nvme2/fibers/PHercParis4.volpkg.json"))
    ap.add_argument("--volpkg", type=Path,
                    default=Path("/media/djosey/nvme2/PHercParis4.volpkg"))
    ap.add_argument("--patches-dir", type=Path,
                    default=Path("/media/djosey/nvme2/graph_patches"))
    ap.add_argument("--overlap", type=Path,
                    default=Path("/media/djosey/nvme2/fiber_patch_overlap/"
                                 "network01_patches_to_fibers.json"))
    ap.add_argument("--trust", type=Path,
                    default=Path("/media/djosey/nvme2/fiber_patch_overlap/"
                                 "network01_patch_trust.json"))
    ap.add_argument("--trust-tier", default="corroborated",
                    help="patch set to merge; 'placed' takes everything, "
                         "including the winding disagreements")
    ap.add_argument("--out", type=Path,
                    default=Path("/media/djosey/nvme2/network_unrolled"))
    ap.add_argument("--uuid", default=None)
    ap.add_argument("--component", type=int, default=0)
    ap.add_argument("--step", type=float, default=20.0,
                    help="UV grid pitch in ds2 voxels; 20 matches the patch grid")
    ap.add_argument("--oversample", type=int, default=3,
                    help="bilinear samples per quad edge when resampling")
    ap.add_argument("--strips-dir", type=Path,
                    default=Path("/media/djosey/nvme2/network_unrolled/strips"),
                    help="fiber surface strips exported by "
                         "vc_lasagna_line_probe --tifxyz-output-dir")
    ap.add_argument("--no-strips", action="store_true",
                    help="merge patches only")
    ap.add_argument("--erode-cells", type=int, default=1)
    ap.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    ap.add_argument("--umbilicus-scale", type=float, default=DEFAULT_UMBILICUS_SCALE)
    ap.add_argument("--patch-scale", type=float, default=DEFAULT_PATCH_SCALE)
    args = ap.parse_args()

    D = load_fibers(args.fibers_dir)
    umb = load_umbilicus(args.volpkg / "umbilicus.json", args.umbilicus_scale)
    to_cm = args.voxel_um / 10000.0
    comp = all_components(D)[args.component]
    fibers = {f: build_fiber(f, D[f], umb) for f in comp}
    links = collect_links(D, comp, fibers)
    place_component(comp, fibers, links)
    _xy, r_ref = component_xy(comp, fibers, to_cm)
    # r_ref comes back in annotation voxels; the merged surface lives in ds2.
    r_ref_ds2 = r_ref / args.patch_scale
    log(f"network {args.component + 1}: {len(comp)} fibers, "
        f"r_ref {r_ref_ds2 * 9.6 / 10000:.2f} cm")

    overlap = load_overlap(args.overlap)
    wanted = load_trust(args.trust, args.trust_tier) if args.trust_tier else None
    comp_set = set(comp)

    # pass 1: place every patch and learn the UV extent
    placed = []
    for name in sorted(overlap):
        if wanted is not None and name not in wanted:
            continue
        entries = [e for e in overlap[name] if e[0] in comp_set]
        if not entries:
            continue
        xyz, valid = load_patch_grid(args.patches_dir / name, args.patch_scale,
                                     args.erode_cells)
        if not valid.any():
            continue
        theta, ref = patch_angles(xyz, valid, umb)
        off, _straddled = patch_offset(ref, entries, fibers, umb, args.patch_scale)
        if off is None:
            continue
        placed.append((name, xyz, valid, theta + off))
    log(f"{len(placed)} patches placed from tier '{args.trust_tier}'")

    # Fiber strips: already in annotation coordinates, so scale 1.0. Each is a
    # thin ribbon along one fiber, filling sheet the grown patches missed.
    strips = []
    if not args.no_strips and args.strips_dir.is_dir():
        for name in sorted(comp):
            sdir = args.strips_dir / (Path(name).stem + ".tifxyz")
            if not (sdir / "x.tif").is_file():
                continue
            sxyz, svalid = load_patch_grid(sdir, 1.0, 0)
            if not svalid.any():
                continue
            sth = strip_theta(sxyz, svalid, fibers[name], umb, args.patch_scale)
            svalid &= np.isfinite(sth)
            if svalid.any():
                strips.append((sdir.name, sxyz, svalid, sth))
        log(f"{len(strips)} fiber strips loaded from {args.strips_dir.name}")
    placed += strips
    if not placed:
        raise SystemExit("nothing to merge")

    u_lo = min(float(np.nanmin(t[v])) for _n, _x, v, t in placed) * r_ref_ds2
    u_hi = max(float(np.nanmax(t[v])) for _n, _x, v, t in placed) * r_ref_ds2
    z_lo = min(float(x[v][:, 2].min()) for _n, x, v, _t in placed) / args.patch_scale
    z_hi = max(float(x[v][:, 2].max()) for _n, x, v, _t in placed) / args.patch_scale
    cols = int(np.ceil((u_hi - u_lo) / args.step)) + 1
    rows = int(np.ceil((z_hi - z_lo) / args.step)) + 1
    log(f"UV grid {rows} rows x {cols} cols at {args.step:g} ds2 vox "
        f"({(u_hi - u_lo) * 9.6 / 10000:.1f} x {(z_hi - z_lo) * 9.6 / 10000:.1f} cm)")

    acc = np.zeros((rows, cols, 3), np.float64)
    cnt = np.zeros((rows, cols), np.int32)

    # pass 2: resample each patch's quads into the grid
    t0 = time.time()
    for k, (name, xyz, valid, th) in enumerate(placed):
        s = quad_samples(xyz, valid, th, args.oversample)
        if s is None:
            continue
        theta_s, z_s, pts = s
        col = np.rint((theta_s * r_ref_ds2 - u_lo) / args.step).astype(np.int64)
        row = np.rint((z_s / args.patch_scale - z_lo) / args.step).astype(np.int64)
        m = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
        if not m.any():
            continue
        flat = row[m] * cols + col[m]
        p = pts[m] / args.patch_scale          # annotation -> ds2
        for ch in range(3):
            np.add.at(acc.reshape(-1, 3)[:, ch], flat, p[:, ch])
        np.add.at(cnt.reshape(-1), flat, 1)
        if (k + 1) % 400 == 0:
            log(f"  {k + 1}/{len(placed)} patches")
    log(f"resampled in {time.time() - t0:.1f}s")

    filled = cnt > 0
    zyxs = np.full((rows, cols, 3), -1.0, np.float32)
    xyz_mean = acc[filled] / cnt[filled][:, None]
    zyxs[filled] = xyz_mean[:, ::-1]           # xyz -> zyx
    log(f"filled {filled.sum():,} of {rows * cols:,} cells ({filled.mean():.1%}); "
        f"mean {cnt[filled].mean():.1f} samples per filled cell")

    # fiber_patch_unroll draws with matplotlib, whose y axis points up, while
    # row 0 of a grid is the top of the image. Reverse the rows so the tifxyz
    # and everything rendered from it read the same way round as the map:
    # z increasing upward, unrolled length increasing to the right.
    zyxs = zyxs[::-1].copy()

    # scripts/spiral/tifxyz.py imports einops at module scope for functions we
    # do not call. /home/djosey/venv lacks it even though the spiral pyproject
    # declares it, so stub it rather than reimplement save_tifxyz or touch the
    # interpreter. `pip install einops` in that venv removes the need for this.
    if "einops" not in sys.modules:
        try:
            import einops  # noqa: F401
        except ModuleNotFoundError:
            import types
            stub = types.ModuleType("einops")
            stub.rearrange = None
            sys.modules["einops"] = stub
            log("einops missing; stubbed (only save_tifxyz is used from tifxyz.py)")
    from tifxyz import save_tifxyz
    uuid = args.uuid or f"network{args.component + 1:02d}_unrolled.tifxyz"
    args.out.mkdir(parents=True, exist_ok=True)
    save_tifxyz(zyxs, str(args.out), uuid, args.step, 9.6,
                f"unrolled merge of {len(placed)} graph patches "
                f"(tier {args.trust_tier}, erode {args.erode_cells})")
    log(f"wrote {args.out / uuid}")


if __name__ == "__main__":
    main()
