#!/usr/bin/env python3
"""Draw graph patches and the fiber network they carry on one unrolled map.

``fiber_network_unroll.py`` flattens a linked H/V fiber network by unrolling it
about the umbilicus (x = unwrapped angle x reference radius, y = z).  This tool
puts the *surface patches* those fibers lie on into the same frame, so a network
can be read as sheet coverage rather than as bare curves: where the patches are,
which ones a fiber crosses, and which stretches of a fiber no patch covers.

Everything about the fiber layout -- component decomposition, whole-turn
placement, smoothing, labels, legend, physical scale -- is
``fiber_network_unroll``'s, imported rather than reimplemented.  This module adds
exactly two things: reading tifxyz patches, and placing them on the same frame.

Placing a patch
---------------
A patch is not in the link graph, so it has no whole-turn offset of its own.  It
gets one from the fibers that run along it, by the same snapping rule the link
graph uses:

  * a patch spans far less than one turn (a few hundred ds2 voxels against a
    circumference of ~8000), so its vertex angles are made self-consistent by
    wrapping them to within pi of the patch's own circular median;
  * every fiber point that overlaps the patch is a point whose placed angle is
    already known, so the patch's offset is the whole turn that best agrees with
    those anchors.

Which fibers overlap which patch comes from the overlap report written by
``fiber_patch_overlap.py`` (``--overlap``), keyed by patch directory name.

Frames
------
Fiber annotations are the finest frame (2.4 um/voxel).  The umbilicus json is 4x
coarser and is scaled up by ``--umbilicus-scale`` (see the unroll script's
gotchas).  Patches are ds2, also 4x coarser, so patch vertices are scaled by
``--patch-scale`` (4.0) into annotation coordinates.

Needs numpy, matplotlib and tifffile.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from fiber_network_unroll import (
    DEFAULT_UMBILICUS_SCALE,
    DEFAULT_VOXEL_UM,
    THEME,
    TWO_PI,
    all_components,
    build_fiber,
    collect_links,
    component_xy,
    draw,
    load_fibers,
    load_umbilicus,
    place_component,
    tint,
    umbilicus_center,
)

# Patch tifxyz coordinates are ds2; fiber annotations are 4x finer.
DEFAULT_PATCH_SCALE = 4.0

# tifxyz marks an invalid grid vertex with -1 in every channel.
INVALID = -1.0

# One flat fill per level, drawn semi-transparent: every patch is the same
# weight, so where the sheet is covered by several patches the fill composites
# darker on its own.  Depth of colour is patch stacking, nothing else.
#
# Green for the fiber-carrying patches and violet for their neighbours, both
# chosen to sit apart from the unroll's blue horizontals and orange verticals
# rather than compete with either.
PATCH_FILL = {"light": "#3d8b60", "dark": "#3a9260"}
NEIGHBOUR_FILL = {"light": "#7a6bab", "dark": "#6f63a8"}


# --------------------------------------------------------------------------- #
# patches
# --------------------------------------------------------------------------- #

def load_overlap(path: Path) -> dict:
    """patch name -> [(fiber json name, (k, 3) on-surface anchors)] from the report."""
    doc = json.loads(path.read_text())
    out = {}
    for name, rec in doc["patches"].items():
        out[name] = [(f["fiber"], np.asarray(f.get("anchors") or [], float))
                     for f in rec["fibers"]]
    return out


def load_trust(path: Path, tier: str) -> set:
    """Patch names in one tier of the consistency report from patch_trust.py."""
    doc = json.loads(path.read_text())
    tiers = doc["tiers"]
    if tier not in tiers:
        raise SystemExit(f"unknown trust tier {tier!r}; have {sorted(tiers)}")
    return set(tiers[tier])


def load_neighbours(path: Path) -> dict:
    """patch name -> {'via': level-0 patch, 'anchors': (k, 3) shared points}."""
    doc = json.loads(path.read_text())
    return {name: {"via": rec["via"],
                   "anchors": np.asarray(rec["anchors"], float)}
            for name, rec in doc["patches"].items()}


def erode_valid(valid: np.ndarray, cells: int) -> np.ndarray:
    """Drop ``cells`` rings of grid vertices from the border of the valid region.

    One ring is one grid cell -- 20 ds2 voxels, 192 um at 9.6 um/voxel. Band
    grower output is least reliable at the border, where a patch can lip onto
    the neighbouring wrap; eroding costs a little coverage and removes the part
    of each surface most likely to be wrong.

    8-connected, and the array edge counts as border: a vertex survives only if
    its whole 3x3 grid neighbourhood is valid and present.
    """
    for _ in range(max(0, cells)):
        v = valid
        keep = v.copy()
        keep[:-1] &= v[1:]
        keep[1:] &= v[:-1]
        keep[:, :-1] &= v[:, 1:]
        keep[:, 1:] &= v[:, :-1]
        keep[:-1, :-1] &= v[1:, 1:]
        keep[1:, 1:] &= v[:-1, :-1]
        keep[:-1, 1:] &= v[1:, :-1]
        keep[1:, :-1] &= v[:-1, 1:]
        keep[0] = keep[-1] = False
        keep[:, 0] = keep[:, -1] = False
        valid = keep
    return valid


def load_patch_grid(patch_dir: Path, scale: float, erode_cells: int = 0):
    """(R, C, 3) xyz grid in annotation coords plus its validity mask."""
    import tifffile
    xyz = np.stack([tifffile.imread(patch_dir / f"{c}.tif") for c in "xyz"],
                   axis=-1).astype(np.float64)
    valid = np.all(xyz != INVALID, axis=-1)
    if erode_cells:
        valid = erode_valid(valid, erode_cells)
    return xyz * scale, valid


def load_patch_surface(patch_dir: Path, erode_cells: int = 0):
    """ds2 zyx grid ready for vc.surface_index.QuadSurface, eroded vertices cut.

    QuadSurface reads the -1 sentinel as "no vertex", so eroding is a matter of
    writing the sentinel back over the ring this loader dropped.
    """
    xyz, valid = load_patch_grid(patch_dir, 1.0, erode_cells)
    xyz[~valid] = INVALID
    return np.ascontiguousarray(xyz[..., ::-1].astype(np.float32)), valid


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + math.pi) % TWO_PI - math.pi


def dominant_turn(offsets: np.ndarray) -> float:
    """The most-voted whole-turn offset.

    Must be a *mode*, never a mean or median: each anchor votes for some whole
    multiple of 2*pi, and averaging two anchors that disagree by one turn lands
    the patch half a turn off -- a placement no single anchor asked for.  With
    an even number of anchors np.median does exactly that.
    """
    values, counts = np.unique(np.round(offsets / TWO_PI).astype(np.int64),
                               return_counts=True)
    return float(values[int(np.argmax(counts))]) * TWO_PI


def patch_angles(xyz: np.ndarray, valid: np.ndarray, umb: np.ndarray):
    """Self-consistent (not yet globally placed) angle and z for every vertex.

    Raw arctan2 angles jump by 2*pi across the theta=0 seam.  A patch is small
    compared with a turn, so wrapping every angle to within pi of the patch's
    circular median removes the seam without any unwrapping order dependence.
    """
    flat = xyz[valid]
    ctr = umbilicus_center(umb, flat[:, 2])
    rel = flat[:, :2] - ctr
    raw = np.arctan2(rel[:, 1], rel[:, 0])
    ref = math.atan2(float(np.sin(raw).mean()), float(np.cos(raw).mean()))
    theta = np.full(valid.shape, np.nan)
    theta[valid] = ref + wrap_to_pi(raw - ref)
    return theta, ref


def patch_offset(ref, entries, fibers, umb, patch_scale):
    """Whole-turn offset aligning the patch with the fibers that overlap it.

    ``entries`` is [(fiber name, anchors)] from the overlap report; an anchor is
    a fiber point the surface index found *on this patch's surface*, so its
    placed angle is the patch's placed angle there and the offset is the nearest
    whole turn between the two.

    The anchors have to come from the report.  Selecting them here by testing
    which of a fiber's points fall inside the patch's bounding box does not
    work: a fiber winding past on the neighbouring wrap also enters that box,
    and votes a full turn out.  Matching each anchor back to its own fiber's
    polyline (nearest point, so the right pass of a fiber that re-enters) keeps
    the vote on the pass that is actually on the patch.

    Returns ``(offset, straddled)``.  ``straddled`` counts overlapping fibers
    that voted for a different turn than the winner: the patch reaches fibers on
    more than one winding.  Wrap spacing here is a few hundred microns, so two
    fibers a millimetre apart in 3D are several wraps apart and *should* place a
    turn apart -- such a patch crosses a wrap.  The majority winding is drawn
    and the count is reported; those patches are worth a look.
    """
    offsets, per_fiber = [], []
    for name, anchors in entries:
        fb = fibers.get(name)
        if fb is None or anchors.ndim != 2 or not len(anchors):
            continue
        pts = anchors * patch_scale
        idx = np.array([int(np.argmin(np.linalg.norm(fb.line - p, axis=1)))
                        for p in pts])
        ctr = umbilicus_center(umb, pts[:, 2])
        rel = pts[:, :2] - ctr
        local = ref + wrap_to_pi(np.arctan2(rel[:, 1], rel[:, 0]) - ref)
        placed = fb.theta_line[idx] + fb.offset
        offsets.append(placed - local)
        per_fiber.append(dominant_turn(placed - local))
    if not offsets:
        return None, 0
    off = dominant_turn(np.concatenate(offsets))
    straddled = sum(1 for v in per_fiber if abs(v - off) > math.pi)
    return off, straddled


def patch_ribbon(xyz, valid, theta, offset, r_ref, to_cm):
    """Silhouette polygon of the patch in unrolled cm.

    Walked as a ribbon along whichever grid axis has more occupied lines: down
    one side taking each line's first valid vertex, back up the other taking its
    last.  That tracks a curved band far better than a convex hull, and unlike a
    per-quad mesh it stays one polygon per patch, which matters at ~2000 patches
    per figure.  A line whose valid vertices are not contiguous has its gap
    filled -- this is a silhouette, not a mask.
    """
    x = (theta + offset) * r_ref * to_cm
    y = xyz[..., 2] * to_cm
    rows_used = int(np.count_nonzero(valid.any(axis=1)))
    cols_used = int(np.count_nonzero(valid.any(axis=0)))
    if cols_used > rows_used:
        x, y, valid = x.T, y.T, valid.T
    left, right = [], []
    for r in range(valid.shape[0]):
        cols = np.flatnonzero(valid[r])
        if not len(cols):
            continue
        a, b = int(cols[0]), int(cols[-1])
        left.append((x[r, a], y[r, a]))
        right.append((x[r, b], y[r, b]))
    if len(left) < 2:
        return None
    return np.array(left + right[::-1], float)


def build_patches(patch_dirs, overlap, comp_set, fibers, umb, r_ref, to_cm,
                  patch_scale, erode_cells=0):
    """Ribbon polygon for every patch this network's fibers run through.

    ``ref`` and ``offset`` are kept per patch: a neighbouring patch carries no
    fiber of its own and is placed from these (see build_neighbours).
    """
    out, unplaced = [], []
    for name in sorted(overlap):
        entries = [e for e in overlap[name] if e[0] in comp_set]
        if not entries:
            continue
        pdir = patch_dirs / name
        if not (pdir / "x.tif").is_file():
            unplaced.append((name, "missing tifxyz"))
            continue
        xyz, valid = load_patch_grid(pdir, patch_scale, erode_cells)
        if not valid.any():
            unplaced.append((name, "no valid vertices"))
            continue
        theta, ref = patch_angles(xyz, valid, umb)
        off, straddled = patch_offset(ref, entries, fibers, umb, patch_scale)
        if off is None:
            unplaced.append((name, "no anchors in the overlap report"))
            continue
        poly = patch_ribbon(xyz, valid, theta, off, r_ref, to_cm)
        if poly is None:
            unplaced.append((name, "degenerate silhouette"))
            continue
        out.append(dict(name=name, poly=poly, ref=ref, offset=off,
                        straddled=straddled,
                        fibers=[e[0] for e in entries]))
    return out, unplaced


def build_neighbours(patch_dirs, neighbours, placed, umb, r_ref, to_cm,
                     patch_scale, erode_cells=0):
    """Ribbon polygons for the patches one overlap step outside the fiber set.

    A neighbour has no fiber, so it borrows its whole turn from the level-0
    patch it overlaps.  The report's anchors are points on *both* surfaces, so
    the level-0 patch's placed angle there is the neighbour's placed angle
    there, and the offset follows by the same nearest-whole-turn rule used
    everywhere else.
    """
    by_name = {p["name"]: p for p in placed}
    out, unplaced = [], []
    for name in sorted(neighbours):
        rec = neighbours[name]
        via = by_name.get(rec["via"])
        if via is None:
            unplaced.append((name, f"level-0 patch {rec['via']} not placed"))
            continue
        anchors = rec["anchors"] * patch_scale
        if anchors.ndim != 2 or not len(anchors):
            unplaced.append((name, "no anchors"))
            continue
        pdir = patch_dirs / name
        if not (pdir / "x.tif").is_file():
            unplaced.append((name, "missing tifxyz"))
            continue
        xyz, valid = load_patch_grid(pdir, patch_scale, erode_cells)
        if not valid.any():
            unplaced.append((name, "no valid vertices"))
            continue
        theta, ref = patch_angles(xyz, valid, umb)

        ctr = umbilicus_center(umb, anchors[:, 2])
        rel = anchors[:, :2] - ctr
        raw = np.arctan2(rel[:, 1], rel[:, 0])
        placed_here = via["ref"] + wrap_to_pi(raw - via["ref"]) + via["offset"]
        local = ref + wrap_to_pi(raw - ref)
        off = dominant_turn(placed_here - local)

        poly = patch_ribbon(xyz, valid, theta, off, r_ref, to_cm)
        if poly is None:
            unplaced.append((name, "degenerate silhouette"))
            continue
        out.append(dict(name=name, poly=poly, via=rec["via"]))
    return out, unplaced


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #

def patch_bounds(*groups):
    polys = [p["poly"] for g in groups for p in g]
    if not polys:
        return None
    allxy = np.concatenate(polys)
    return allxy.min(0), allxy.max(0)


def make_underlay(patches, neighbours, C, theme, alpha, nb_alpha, edges):
    """Return an ``underlay(ax, shift)`` for fiber_network_unroll.draw.

    Each level is one PolyCollection of equally weighted, equally transparent
    silhouettes, so overlapping patches composite into a denser fill by
    themselves.  Neighbours go underneath the fiber-carrying patches.
    """
    edge = tint(C["ink_soft"], C["surface"], 0.55) if edges else "none"

    def layer(ax, group, color, a, z):
        from matplotlib.collections import PolyCollection
        if not group:
            return
        ax.add_collection(PolyCollection(
            [p["poly"] + np.array([shift_holder[0], 0.0]) for p in group],
            facecolors=color, alpha=a, edgecolors=edge,
            linewidths=0.35 if edges else 0.0, zorder=z))

    shift_holder = [0.0]

    def underlay(ax, shift):
        shift_holder[0] = shift
        layer(ax, neighbours, NEIGHBOUR_FILL[theme], nb_alpha, 0.8)
        layer(ax, patches, PATCH_FILL[theme], alpha, 1.0)
    return underlay


def patch_legend(theme, alpha, nb_alpha, with_neighbours):
    from matplotlib.patches import Patch
    out = [Patch(facecolor=PATCH_FILL[theme], alpha=alpha, edgecolor="none",
                 label="patch on a fiber")]
    if with_neighbours:
        out.append(Patch(facecolor=NEIGHBOUR_FILL[theme], alpha=nb_alpha,
                         edgecolor="none", label="patch overlapping those"))
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

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
                                 "network01_patches_to_fibers.json"),
                    help="patch->fiber report from fiber_patch_overlap.py")
    ap.add_argument("--neighbours", type=Path,
                    default=Path("/media/djosey/nvme2/fiber_patch_overlap/"
                                 "network01_patch_neighbours.json"),
                    help="patch->patch report from patch_neighbours.py; the "
                         "level of overlap one step outside the fiber network")
    ap.add_argument("--no-neighbours", action="store_true",
                    help="draw only the patches the fibers themselves run along")
    ap.add_argument("--out", type=Path,
                    default=Path("/media/djosey/nvme2/fibers/PHercParis4_network_2d"))
    ap.add_argument("--component", type=int, default=0,
                    help="network index in the size-sorted list (0 = largest)")
    ap.add_argument("--theme", choices=("light", "dark"), default="dark")
    ap.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    ap.add_argument("--umbilicus-scale", type=float, default=DEFAULT_UMBILICUS_SCALE)
    ap.add_argument("--patch-scale", type=float, default=DEFAULT_PATCH_SCALE,
                    help="ds2 patch coords -> annotation coords")
    ap.add_argument("--patch-alpha", type=float, default=0.3,
                    help="one transparency for every patch; stacked patches "
                         "composite darker on their own")
    ap.add_argument("--neighbour-alpha", type=float, default=None,
                    help="transparency for the outer level (default: same)")
    ap.add_argument("--patch-edges", action="store_true",
                    help="outline every patch silhouette")
    ap.add_argument("--erode-cells", type=int, default=1,
                    help="shrink every patch by this many grid rings before "
                         "use; one ring is 20 ds2 voxels (192 um), where band "
                         "grower output is least reliable")
    ap.add_argument("--trust", type=Path,
                    default=Path("/media/djosey/nvme2/fiber_patch_overlap/"
                                 "network01_patch_trust.json"),
                    help="consistency report from patch_trust.py")
    ap.add_argument("--trust-tier", default=None,
                    help="draw only patches in this tier of --trust "
                         "(placed, unstraddled, no_B, A_consistent, "
                         "corroborated, corroborated_C_clean); default: all")
    ap.add_argument("--fit", choices=("all", "fibers"), default="all",
                    help="frame the patches too, or keep the fiber-only extent")
    ap.add_argument("--suspect-turns", type=float, default=0.25)
    ap.add_argument("--smooth-mm", type=float, default=1.2)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    D = load_fibers(args.fibers_dir)
    umb = load_umbilicus(args.volpkg / "umbilicus.json", args.umbilicus_scale)
    to_cm = args.voxel_um / 10000.0
    args.out.mkdir(parents=True, exist_ok=True)

    comp = all_components(D)[args.component]
    fibers = {f: build_fiber(f, D[f], umb) for f in comp}
    links = collect_links(D, comp, fibers)
    if not links:
        raise SystemExit(f"network {args.component + 1} has no links")
    place_component(comp, fibers, links)
    xy, r_ref = component_xy(comp, fibers, to_cm)
    r_ref_cm = r_ref * to_cm

    overlap = load_overlap(args.overlap)
    patches, unplaced = build_patches(
        args.patches_dir, overlap, set(comp), fibers, umb, r_ref, to_cm,
        args.patch_scale, args.erode_cells)

    dropped = 0
    if args.trust_tier:
        trusted = load_trust(args.trust, args.trust_tier)
        before = len(patches)
        patches = [p for p in patches if p["name"] in trusted]
        dropped = before - len(patches)

    neighbours, nb_unplaced = [], []
    if not args.no_neighbours and args.neighbours.is_file():
        neighbours, nb_unplaced = build_neighbours(
            args.patches_dir, load_neighbours(args.neighbours), patches, umb,
            r_ref, to_cm, args.patch_scale, args.erode_cells)
    elif not args.no_neighbours:
        print(f"  no neighbour report at {args.neighbours}; "
              f"drawing the fiber-carrying patches only")

    covered = {f for p in patches for f in p["fibers"]}
    bare = sorted(set(comp) - covered)
    n_pairs = sum(len(p["fibers"]) for p in patches)
    print(f"network {args.component + 1}: {len(comp)} fibers, {len(links)} links, "
          f"r ≈ {r_ref_cm:.2f} cm")
    print(f"  {len(patches)} patches on fibers ({n_pairs} patch–fiber pairs) "
          f"from {args.overlap.name}")
    if args.trust_tier:
        print(f"  trust tier '{args.trust_tier}' from {args.trust.name}: "
              f"{dropped} inconsistent patches dropped")
    if neighbours:
        print(f"  {len(neighbours)} patches overlapping those, "
              f"from {args.neighbours.name}")
    straddle = [p for p in patches if p["straddled"]]
    if straddle:
        print(f"  {len(straddle)} patches reach fibers on more than one winding "
              f"({sum(p['straddled'] for p in straddle)} fibers off the majority "
              f"turn); drawn on the majority winding")
    if bare:
        print(f"  {len(bare)} fibers carry no patch: "
              + ", ".join(fibers[f].label for f in bare))
    # One line per reason, not per patch: filtering by trust tier orphans a few
    # hundred neighbours at once and the names are in the reports anyway.
    reasons = {}
    for name, why in unplaced + nb_unplaced:
        key = "level-0 patch not placed" if why.endswith("not placed") else why
        reasons.setdefault(key, []).append(name)
    for why, names_ in sorted(reasons.items()):
        detail = f": {names_[0]}" if len(names_) == 1 else ""
        print(f"  skipped {len(names_)} — {why}{detail}")

    C = THEME[args.theme]
    nb_alpha = (args.patch_alpha if args.neighbour_alpha is None
                else args.neighbour_alpha)
    suffix = f"_{args.trust_tier}" if args.trust_tier else ""
    out_path = args.out / f"network_{args.component + 1:02d}_patches{suffix}.png"
    subtitle = f"  ·  {len(patches)} patches on fibers"
    if neighbours:
        subtitle += f"  ·  {len(neighbours)} patches overlapping those"
    n_suspect = draw(
        comp, fibers, links, xy, r_ref_cm, out_path, args.theme,
        f"Fiber network {args.component + 1} on graph patches",
        args.suspect_turns, args.smooth_mm, args.scale, args.dpi,
        underlay=make_underlay(patches, neighbours, C, args.theme,
                               args.patch_alpha, nb_alpha, args.patch_edges),
        extra_bounds=(patch_bounds(patches, neighbours)
                      if args.fit == "all" else None),
        legend_extra=patch_legend(args.theme, args.patch_alpha, nb_alpha,
                                  bool(neighbours)),
        subtitle_extra=subtitle)
    print(f"  wrote {out_path}"
          + (f"  ({n_suspect} winding-suspect links)" if n_suspect else ""))


if __name__ == "__main__":
    main()
