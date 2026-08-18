#!/usr/bin/env python3
"""Flatten manually-linked H/V fiber networks by unrolling about the umbilicus.

Fiber annotations (``vc3d_fiber`` JSON) trace horizontal and vertical papyrus
fibers through a scroll volume, and ``branches`` records link H/V crossings that
an annotator judged to be the same point on the sheet.  A connected component of
that link graph is a patch of papyrus weave; this tool lays each component out in
2D so the linked structure can be reviewed: which fibers connect to which, and in
what order along each fiber.

The mapping is *extrinsic* and deliberately simple:

    x = unwrapped angle about the umbilicus  x  reference radius   (circumference)
    y = z                                                          (scroll axis)

np.unwrap gives each fiber an angle sequence carrying its own arbitrary multiple
of 2*pi, so per-fiber angles are made mutually consistent by walking the link
graph: two linked crossings are the same physical point, so each fiber's offset
is snapped to the nearest whole turn of its already-placed neighbour.  With that,
crossings coincide by construction, horizontal fibers run horizontally (they wind
around the scroll), vertical fibers run vertically (they follow z), and closed
loops close -- there is nothing to relax and no drift to accumulate.

The one thing that CAN disagree is a link whose two endpoints, after the whole
component is placed, still sit a large fraction of a turn apart: that link was
annotated on the wrong winding (or connects through a tear).  Those are the links
worth reviewing, so they are flagged in red on the plot and listed in the report.

Requires numpy and matplotlib only.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import common

EPS = 1e-12
TWO_PI = 2.0 * math.pi

# Fiber/annotation coordinates are 4x finer than volumes/s1_2um_ds2.zarr, whose
# meta.json declares voxelsize 9.60 um -- so one annotation voxel is 2.4 um.
DEFAULT_VOXEL_UM = 2.4

# The umbilicus json lives in volume-grid coordinates; annotations are 4x finer.
DEFAULT_UMBILICUS_SCALE = 4.0

# dataviz categorical slots 1 and 2, validated all-pairs in both modes.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_soft": "#52514e",
        "grid": "#e2e1dd",
        "H": "#2a78d6",
        "V": "#eb6834",
        "crossing": "#6b5f3f",
        "suspect": "#c92a2a",
        "winding": "#8e8b80",
        "chip_H": "#d5e5f8",
        "chip_V": "#fadcc8",
        "chip_ink": "#0b0b0b",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_soft": "#c3c2b7",
        "grid": "#333330",
        "H": "#3987e5",
        "V": "#d95926",
        "crossing": "#e6d8b8",
        "suspect": "#ff6b6b",
        "winding": "#9a978c",
        "chip_H": "#9cc5f0",
        "chip_V": "#f2ab7a",
        "chip_ink": "#0b0b0b",
    },
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_fibers(fibers_dir: Path) -> dict:
    out = {}
    for p in sorted(fibers_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict) or d.get("type") != "vc3d_fiber":
            continue
        if not d.get("control_points") or not d.get("line_points"):
            continue
        # vc3d_fiber v3 stores each control point as {"position": [x,y,z], ...}
        # alongside per-segment tracer metadata; v1 stored a bare [x,y,z].
        # Normalise to the bare triple, keeping the segment metadata aside.
        cp = d["control_points"]
        if cp and isinstance(cp[0], dict):
            try:
                d["_seg_meta"] = [q.get("segment_to_next") or {} for q in cp]
                d["control_points"] = [q["position"] for q in cp]
            except (KeyError, TypeError):
                continue
        lp = d["line_points"]
        if lp and isinstance(lp[0], dict):
            try:
                d["line_points"] = [q["position"] for q in lp]
            except (KeyError, TypeError):
                continue
        out[p.name] = d
    if not out:
        raise SystemExit(f"no vc3d_fiber JSON found under {fibers_dir}")
    return out


def hv_tag(d: dict) -> str:
    hv = d.get("hv_classification") or {}
    tag = (hv.get("manual_tag") or "").strip() or (hv.get("automatic_tag") or "").strip()
    return tag.upper() or "?"


def branch_records(D: dict, name: str):
    for br in D[name].get("branches") or []:
        tgt = br.get("branch_file")
        if tgt in D:
            yield br, tgt


def link_key(fa, ia, fb, ib):
    return tuple(sorted([(fa, ia), (fb, ib)]))


def all_components(D: dict) -> list[list[str]]:
    par = {f: f for f in D}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for f in D:
        for _, tgt in branch_records(D, f):
            a, b = find(f), find(tgt)
            if a != b:
                par[a] = b
    comp = collections.defaultdict(list)
    for f in D:
        comp[find(f)].append(f)
    return sorted((sorted(v) for v in comp.values()), key=lambda c: (-len(c), c[0]))


def load_umbilicus(path: Path, scale: float) -> np.ndarray:
    """(M,3) umbilicus points sorted by z, scaled into annotation coords."""
    raw = json.loads(path.read_text())
    pts = raw["control_points"] if isinstance(raw, dict) else raw
    arr = np.array([[p["x"], p["y"], p["z"]] for p in pts], float) * scale
    return arr[np.argsort(arr[:, 2])]


def umbilicus_center(umb: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(z, umb[:, 2], umb[:, 0]), np.interp(z, umb[:, 2], umb[:, 1])], axis=-1
    )


def fiber_label(name: str, d: dict) -> str:
    """Short unique label: "<file prefix>-<sequence>", e.g. kb-604.

    The user part comes from the FILENAME, not the `username` field -- for some
    fibers the two disagree, and the filename is what the VC3D list shows.
    Neither component alone is unique; the pair is.
    """
    prefix = name.split("_", 1)[0] if "_" in name else name
    return f"{prefix}-{d.get('sequence', '?')}"


def tint(hex_color: str, toward: str, amount: float) -> str:
    def parts(h):
        h = h.lstrip("#")
        return [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    a, b = parts(hex_color), parts(toward)
    return "#" + "".join(f"{round(x + (y - x) * amount):02x}" for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# unrolling
# --------------------------------------------------------------------------- #

@dataclass
class Fiber:
    name: str
    label: str
    tag: str
    cp: np.ndarray            # (n, 3) control points, annotation vx
    line: np.ndarray          # (m, 3) dense line points
    cp_line_idx: np.ndarray   # (n,) index of each control point in line
    theta_line: np.ndarray    # (m,) unwrapped angle, fiber-local until offset applied
    radius: np.ndarray        # (m,) distance from umbilicus in the z-slice plane
    traced: np.ndarray = None  # (n-1,) segment k was fiber-model traced, not interpolated
    offset: float = 0.0       # whole-turn offset making theta globally consistent

    @property
    def theta_cp(self) -> np.ndarray:
        return self.theta_line[self.cp_line_idx] + self.offset


@dataclass
class Link:
    fa: str
    ia: int
    fb: str
    ib: int
    sep3d: float
    turn_err: float = 0.0     # |theta_a - theta_b| after placement, in turns
    on_tree: bool = False


def control_line_indices(cp: np.ndarray, lp: np.ndarray) -> np.ndarray:
    """Index into line_points for each control point (they are an exact subset)."""
    idx = np.empty(len(cp), np.int64)
    for i, p in enumerate(cp):
        idx[i] = int(np.argmin(np.linalg.norm(lp - p, axis=1)))
    return idx


def trusted_segments(seg_meta, n: int) -> np.ndarray:
    """Which control-point segments were fiber-model traced (vs interpolated)?

    v3 records how each segment between control points was produced.  'trace'
    segments come from native_fiber_trace3d following the fiber prediction
    volume; anything else (lasagna interpolation, or a trace that recorded a
    failure_code) is only an interpolation and its line points wander.
    """
    ok = np.zeros(n, bool)
    if not seg_meta:
        return ok
    for k in range(min(n, len(seg_meta))):
        m = seg_meta[k] or {}
        if m.get("interp_mode") == "trace" and not m.get("failure_code"):
            ok[k] = True
    return ok


def build_fiber(name: str, d: dict, umb: np.ndarray) -> Fiber:
    cp = np.array(d["control_points"], float)
    lp = np.array(d["line_points"], float)
    ctr = umbilicus_center(umb, lp[:, 2])
    rel = lp[:, :2] - ctr
    theta = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    return Fiber(
        name=name,
        label=fiber_label(name, d),
        tag=hv_tag(d),
        cp=cp,
        line=lp,
        cp_line_idx=control_line_indices(cp, lp),
        theta_line=theta,
        radius=np.linalg.norm(rel, axis=1),
        traced=trusted_segments(d.get("_seg_meta"), max(0, len(cp) - 1)),
    )


def collect_links(D: dict, comp: list[str], fibers: dict[str, Fiber]) -> list[Link]:
    inset = set(comp)
    seen = {}
    for fa in comp:
        na = len(fibers[fa].cp)
        for br, fb in branch_records(D, fa):
            if fb not in inset:
                continue
            ia = int(br["control_point_index"])
            ib = int(br["branch_control_point_index"])
            if not (0 <= ia < na) or not (0 <= ib < len(fibers[fb].cp)):
                print(f"  warn: link {fa}[{ia}] -> {fb}[{ib}] out of range; skipped")
                continue
            key = link_key(fa, ia, fb, ib)
            if key in seen:
                continue
            pa = np.array(br["control_point_position"], float)
            pb = np.array(br["branch_control_point_position"], float)
            seen[key] = Link(fa, ia, fb, ib, float(np.linalg.norm(pa - pb)))
    return sorted(seen.values(), key=lambda l: (l.fa, l.ia, l.fb, l.ib))


def place_component(comp: list[str], fibers: dict[str, Fiber], links: list[Link]):
    """Snap each fiber's whole-turn offset to its neighbours via BFS on the links.

    The BFS uses the best-agreeing links first (smallest fractional-turn residual
    against an already-placed fiber) so that one wrong-winding link cannot decide
    a fiber's offset when a clean link to the same fiber exists.
    """
    adj = collections.defaultdict(list)
    for li, l in enumerate(links):
        adj[l.fa].append((l.fb, li))
        adj[l.fb].append((l.fa, li))

    def theta_at(f: str, i: int) -> float:
        fb = fibers[f]
        return float(fb.theta_line[fb.cp_line_idx[i]])

    placed: dict[str, bool] = {}
    root = max(comp, key=lambda f: (len(adj[f]), f))
    fibers[root].offset = 0.0
    placed[root] = True

    # Prim-style growth: always attach next the fiber whose connecting link has
    # the smallest fractional-turn disagreement.
    import heapq
    heap = []

    def push_edges(f):
        for g, li in adj[f]:
            if g in placed:
                continue
            l = links[li]
            i, j = (l.ia, l.ib) if l.fa == f else (l.ib, l.ia)
            th_f = theta_at(f, i) + fibers[f].offset
            th_g = theta_at(g, j)
            off = round((th_f - th_g) / TWO_PI) * TWO_PI
            frac = abs(th_f - (th_g + off)) / TWO_PI
            heapq.heappush(heap, (frac, li, f, g, off))

    push_edges(root)
    while heap:
        frac, li, f, g, off = heapq.heappop(heap)
        if g in placed:
            continue
        fibers[g].offset = off
        placed[g] = True
        links[li].on_tree = True
        push_edges(g)

    missing = [f for f in comp if f not in placed]
    if missing:
        # Disconnected within the component cannot happen (components come from
        # the same link graph), but be safe.
        for f in missing:
            fibers[f].offset = 0.0

    for l in links:
        ta = theta_at(l.fa, l.ia) + fibers[l.fa].offset
        tb = theta_at(l.fb, l.ib) + fibers[l.fb].offset
        l.turn_err = abs(ta - tb) / TWO_PI

    # Centre the component: put the median angle near turn 0 so gridline labels
    # stay small.
    med = float(np.median(np.concatenate([fibers[f].theta_cp for f in comp])))
    shift = round(med / TWO_PI) * TWO_PI
    for f in comp:
        fibers[f].offset -= shift


def component_xy(comp: list[str], fibers: dict[str, Fiber], to_mm: float):
    """Return per-fiber dense 2D polylines (mm) plus the reference radius (mm)."""
    r_ref = float(np.median(np.concatenate(
        [fibers[f].radius[fibers[f].cp_line_idx] for f in comp])))
    xy = {}
    for f in comp:
        fb = fibers[f]
        x = (fb.theta_line + fb.offset) * r_ref * to_mm
        y = fb.line[:, 2] * to_mm
        xy[f] = np.stack([x, y], axis=1)
    return xy, r_ref


# --------------------------------------------------------------------------- #
# smoothing
# --------------------------------------------------------------------------- #

def smooth_polyline(P: np.ndarray, sigma: float, step: float):
    """Resample a polyline at uniform arclength and Gaussian-smooth it.

    ``sigma`` and ``step`` are in the polyline's own units.  The line points
    wander around the fiber's true run (placement noise plus interpolated
    stretches), which reads as bumpiness at plot scale.  Smoothing in
    *arclength* keeps the low-frequency shape and is independent of the very
    uneven raw point spacing.

    Returns (s, Q): arclength grid and smoothed points, so positions at any raw
    arclength (control points, crossings) can be read back with np.interp and
    land exactly on the drawn curve.
    """
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s_raw = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_raw[-1])
    if total < 2.0 * step or len(P) < 3:
        return s_raw, P.copy()
    s = np.arange(0.0, total + 0.5 * step, step)
    s[-1] = total
    Q = np.stack([np.interp(s, s_raw, P[:, 0]), np.interp(s, s_raw, P[:, 1])], axis=1)
    if sigma <= 0:
        return s, Q
    rad = max(1, int(round(3.0 * sigma / step)))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) * step / sigma) ** 2)
    k /= k.sum()
    # reflect-pad so the ends do not shrink toward the interior
    ext = np.concatenate([2 * Q[:1] - Q[rad:0:-1], Q, 2 * Q[-1:] - Q[-2:-rad - 2:-1]])
    for c in (0, 1):
        Q[:, c] = np.convolve(ext[:, c], k, mode="valid")
    return s, Q


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #

def network_geometry(comp, fibers, xy, smooth_mm):
    """Smooth each fiber once; marker positions are read off the smoothed curve.

    line_points can overshoot the outermost control points by over a cm; those
    tails carry no segment metadata and are not drawn, so they are clipped out
    of the geometry entirely -- otherwise label anchors and the plot extent
    would be computed from invisible curve.
    """
    geom = {}
    for f in comp:
        fb = fibers[f]
        P = xy[f]
        s_raw = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        s, Q = smooth_polyline(P, smooth_mm / 10.0, step=0.025)
        s_cp = s_raw[fb.cp_line_idx]
        cp_pts = np.stack(
            [np.interp(s_cp, s, Q[:, 0]), np.interp(s_cp, s, Q[:, 1])], axis=1)
        a = int(np.searchsorted(s, s_cp[0], side="left"))
        b = int(np.searchsorted(s, s_cp[-1], side="right"))
        s, Q = s[max(0, a - 1): b + 1], Q[max(0, a - 1): b + 1]
        geom[f] = (s, Q, s_cp, cp_pts)
    return geom


def network_bounds(comp, geom, scale):
    """Padded data bounds for one network.

    The minimum pad leaves room for a few rows of label chips on every side,
    and a minimum canvas guarantees titles, ticks and the y label fit even for
    a tiny network -- the scale never changes, small networks just get more
    empty sheet around them.
    """
    allxy = np.concatenate([geom[f][1] for f in comp])
    lo, hi = allxy.min(0), allxy.max(0)
    pad = np.maximum(0.05 * (hi - lo), [2.2, 1.6])
    lo, hi = lo - pad, hi + pad
    min_ax_in = (3.4, 2.8)
    for c in (0, 1):
        short = min_ax_in[c] / scale - float(hi[c] - lo[c])
        if short > 0:
            lo[c] -= 0.5 * short
            hi[c] += 0.5 * short
    return lo, hi


def render_network(ax, C, comp, fibers, links, geom, lo, hi, r_ref_cm,
                   scale, suspect_turns, windings=None):
    """Draw one network into ``ax`` with data bounds lo/hi.

    ``windings`` overrides the winding gridlines as (x, label) pairs -- the
    combined figure numbers them continuously across panels.  By default they
    sit at whole local turns, numbered from this network's own winding 0.

    Returns (any_interp, n_suspect) so the caller can build the legend.
    """
    ax.set_facecolor(C["surface"])

    # winding gridlines: with a fixed reference radius a turn is a fixed width
    if windings is None:
        circ = TWO_PI * r_ref_cm
        windings = [(k * circ, f"{k:+d}" if k else "0")
                    for k in range(math.ceil(lo[0] / circ),
                                   math.floor(hi[0] / circ) + 1)]
    for xline, lab in windings:
        ax.axvline(xline, color=C["winding"], lw=0.7, ls=(0, (1, 4)),
                   alpha=0.55, zorder=0)
        ax.annotate(lab, xy=(xline, hi[1]),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6,
                    color=C["winding"], annotation_clip=False, zorder=0)

    # Traced (fiber-model optimised) runs draw solid, thick and vivid; segments
    # that are only interpolations draw thin, dashed and faded toward the
    # surface -- "dashed = not real trace data" at a glance.
    any_interp = False
    for f in comp:
        fb = fibers[f]
        col = C.get(fb.tag, C["ink_soft"])
        faint = tint(col, C["surface"], 0.45)
        s, Q, s_cp, cp_pts = geom[f]
        tr = fb.traced
        nseg = max(0, len(fb.cp) - 1)
        if tr is None or len(tr) != nseg or nseg == 0:
            ax.plot(Q[:, 0], Q[:, 1], color=col, lw=2.0,
                    solid_capstyle="round", zorder=2)
        else:
            k = 0
            while k < nseg:
                j = k
                while j + 1 < nseg and bool(tr[j + 1]) == bool(tr[k]):
                    j += 1
                a = int(np.searchsorted(s, s_cp[k], side="left"))
                b = int(np.searchsorted(s, s_cp[j + 1], side="right"))
                seg = Q[max(0, a - 1): min(len(Q), b + 1)]
                if len(seg) > 1:
                    if tr[k]:
                        ax.plot(seg[:, 0], seg[:, 1], color=col, lw=2.2,
                                solid_capstyle="round", zorder=2)
                    else:
                        any_interp = True
                        ax.plot(seg[:, 0], seg[:, 1], color=faint, lw=1.4,
                                ls=(0, (5, 2.2)), dash_capstyle="butt", zorder=2)
                k = j + 1

    # crossings
    n_suspect = 0
    for l in links:
        pa = geom[l.fa][3][l.ia]
        pb = geom[l.fb][3][l.ib]
        if l.turn_err > suspect_turns:
            n_suspect += 1
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=C["suspect"],
                    lw=1.0, ls="--", zorder=4)
            for p in (pa, pb):
                ax.scatter([p[0]], [p[1]], s=34, facecolor="none",
                           edgecolor=C["suspect"], linewidths=1.4, zorder=5)
            mid = 0.5 * (pa + pb)
            ax.annotate(f"{l.turn_err:+.1f} turn", xy=(mid[0], mid[1]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=6.5, color=C["suspect"], zorder=5)
        else:
            mid = 0.5 * (pa + pb)
            ax.scatter([mid[0]], [mid[1]], s=26, color=C["crossing"],
                       alpha=0.55, zorder=4, linewidths=0)

    # labels: chip at whichever fiber end sits nearest its plot edge (H fibers:
    # left vs right, V fibers: bottom vs top).  Chips are laid out greedily as
    # physical rectangles: offsets staggered away from the fiber, either side of
    # it, first candidate that stays inside the axes and clear of already placed
    # chips wins.  Falls back to inside-but-overlapping rather than ever leaving
    # the axes (which would collide with the title or legend).
    placed_rects = []
    chip_h_in = 0.19

    def chip_rect(anchor, dx_pt, dy_pt, ha, w_in):
        cx = anchor[0] + dx_pt / 72.0 / scale
        cy = anchor[1] + dy_pt / 72.0 / scale
        w, h = w_in / scale, chip_h_in / scale
        x0 = cx if ha == "left" else cx - w
        return (x0, x0 + w, cy - 0.5 * h, cy + 0.5 * h)

    def clear_of_others(r):
        return all(r[1] <= q[0] or r[0] >= q[1] or r[3] <= q[2] or r[2] >= q[3]
                   for q in placed_rects)

    def in_bounds(r):
        return (lo[0] <= r[0] and r[1] <= hi[0]
                and lo[1] <= r[2] and r[3] <= hi[1])

    order = sorted(comp, key=lambda f: (fibers[f].tag, fibers[f].label))
    for f in order:
        fb = fibers[f]
        P = geom[f][1]
        w_in = 0.10 + 0.058 * len(fb.label)      # ~7 pt text plus chip padding
        if fb.tag == "V":
            p_lo, p_hi = P[np.argmin(P[:, 1])], P[np.argmax(P[:, 1])]
            at_top = (hi[1] - p_hi[1]) < (p_lo[1] - lo[1])
            p = p_hi if at_top else p_lo
            sgn = 1 if at_top else -1
            cands = []
            for s in range(9):
                dy = sgn * (10 + 11 * s)
                cands += [(8, dy, "left"), (-8, dy, "right")]
        else:
            p_lo, p_hi = P[np.argmin(P[:, 0])], P[np.argmax(P[:, 0])]
            at_right = (hi[0] - p_hi[0]) < (p_lo[0] - lo[0])
            p = p_hi if at_right else p_lo
            dx, ha = (10, "left") if at_right else (-10, "right")
            dys = [-2]
            for k in range(1, 6):
                dys += [-2 + 11 * k, -2 - 11 * k]
            cands = [(dx, dy, ha) for dy in dys]
            cands += [(-dx, dy, "right" if ha == "left" else "left") for dy in dys]
        chosen = next(
            ((d, r) for d in cands
             if in_bounds(r := chip_rect(p, *d, w_in)) and clear_of_others(r)),
            None)
        if chosen is None:
            chosen = next(
                ((d, r) for d in cands if in_bounds(r := chip_rect(p, *d, w_in))),
                (cands[0], chip_rect(p, *cands[0], w_in)))
        (dx_pt, dy_pt, ha), r = chosen
        placed_rects.append(r)
        chip = C["chip_V"] if fb.tag == "V" else C["chip_H"]
        ax.annotate(fb.label, xy=(p[0], p[1]), xytext=(dx_pt, dy_pt),
                    textcoords="offset points", ha=ha,
                    va="center", fontsize=7, color=C["chip_ink"], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=chip,
                              edgecolor="none"))

    from matplotlib.ticker import MultipleLocator
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.xaxis.set_major_locator(MultipleLocator(X_TICK_CM))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=C["ink_soft"], labelsize=8, length=0)
    ax.grid(True, color=C["grid"], lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    return any_interp, n_suspect


# figure margins shared by every layout, in inches
MARGINS_IN = dict(left=0.85, right=0.3, top=1.35, bot=1.25)

# one x-tick interval for every plot and panel, in cm
X_TICK_CM = 5.0

# smallest coordinate gap between panels of the combined figure, in cm
MIN_GAP_CM = 1.0


def legend_handles(C, any_interp, n_suspect, extra=None):
    from matplotlib.lines import Line2D
    handles = list(extra or [])
    handles += [
        Line2D([], [], color=C["H"], lw=2.2, label="hz fiber"),
        Line2D([], [], color=C["V"], lw=2.2, label="vt fiber"),
        Line2D([], [], marker="o", color="none", markerfacecolor=C["crossing"],
               markersize=6, label="links"),
    ]
    if any_interp:
        handles.insert(len(handles) - 1, Line2D(
            [], [], color=tint(C["ink_soft"], C["surface"], 0.35), lw=1.4,
            ls=(0, (5, 2.2)), dash_capstyle="butt", label="old interpolation"))
    if n_suspect:
        handles.append(Line2D([], [], color=C["suspect"], lw=1.2, ls="--",
                              label="suspect link"))
    return handles


def add_legend(fig, C, fig_w, fig_h, any_interp, n_suspect, extra=None):
    leg = fig.legend(handles=legend_handles(C, any_interp, n_suspect, extra),
                     loc="lower center",
                     ncol=5 if fig_w >= 10 else 2,
                     frameon=False, bbox_to_anchor=(0.5, 0.2 / fig_h))
    for t in leg.get_texts():
        t.set_color(C["ink_soft"])
        t.set_fontsize(9.5)


def draw(comp, fibers, links, xy, r_ref_cm, out_path: Path, theme: str,
         title: str, suspect_turns: float, smooth_mm: float = 1.2,
         scale: float = 0.25, dpi: int = 160,
         underlay=None, extra_bounds=None, legend_extra=None,
         subtitle_extra: str = ""):
    """Draw one network.

    The three optional hooks let another tool put its own geometry on the same
    unrolled frame without forking this function (see fiber_patch_unroll.py):
    ``extra_bounds`` is an ((xlo, ylo), (xhi, yhi)) box merged into the data
    bounds before the figure is sized, ``underlay`` is called as
    ``underlay(ax, shift)`` with the same left-edge shift applied to the fibers,
    and ``legend_extra`` prepends handles to the legend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = THEME[theme]
    geom = network_geometry(comp, fibers, xy, smooth_mm)
    lo, hi = network_bounds(comp, geom, scale)
    if extra_bounds is not None:
        lo = np.minimum(lo, np.asarray(extra_bounds[0], float))
        hi = np.maximum(hi, np.asarray(extra_bounds[1], float))
    # start the frame at the left edge: unrolled length runs from 0 and the
    # winding gridlines are numbered 0, 1, 2, ... left to right
    shift = -float(lo[0])
    for f in comp:
        geom[f][1][:, 0] += shift
        geom[f][3][:, 0] += shift
    circ = TWO_PI * r_ref_cm
    marks = [m * circ + shift
             for m in range(math.ceil(float(lo[0]) / circ),
                            math.floor(float(hi[0]) / circ) + 1)]
    windings = [(x, str(i)) for i, x in enumerate(marks)]
    lo[0], hi[0] = 0.0, float(hi[0]) + shift
    span = hi - lo

    # One fixed physical scale for every network: `scale` inches of figure per
    # cm of papyrus, identical on both axes.  The figure is sized from the data
    # extent plus fixed margins, so line widths, fonts and label chips (all in
    # points) mean the same thing on every plot.
    ax_w_in = float(span[0]) * scale
    ax_h_in = float(span[1]) * scale
    M = MARGINS_IN
    fig_w = max(ax_w_in + M["left"] + M["right"], 7.0)
    fig_h = ax_h_in + M["top"] + M["bot"]
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(C["surface"])
    ax = fig.add_axes([M["left"] / fig_w, M["bot"] / fig_h,
                       ax_w_in / fig_w, ax_h_in / fig_h])
    if underlay is not None:
        underlay(ax, shift)

    any_interp, n_suspect = render_network(
        ax, C, comp, fibers, links, geom, lo, hi, r_ref_cm, scale,
        suspect_turns, windings=windings)

    ax.set_ylabel("scroll z-axis (cm)", color=C["ink_soft"], fontsize=10)
    ax.set_xlabel("unrolled length (cm)", color=C["ink_soft"], fontsize=10)
    ax.xaxis.set_label_coords(0.5, -0.42 / ax_h_in)
    ax.text(0.5, 1.0 + 0.24 / ax_h_in, "windings", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10, color=C["ink_soft"])

    nH = sum(1 for f in comp if fibers[f].tag == "H")
    nV = sum(1 for f in comp if fibers[f].tag == "V")
    n_cycles = max(0, len(links) - len(comp) + 1)
    subtitle = (f"{len(comp)} linked fibers  ·  {nH} horizontal, {nV} vertical"
                f"  ·  {len(links)} crossings  ·  {n_cycles} independent cycles")
    if n_suspect:
        subtitle += f"  ·  {n_suspect} winding-suspect link{'s' if n_suspect > 1 else ''}"
    subtitle += subtitle_extra
    tx = 0.5 / fig_w
    fig.text(tx, 1.0 - 0.35 / fig_h, title, fontsize=15, color=C["ink"],
             ha="left", va="top")
    fig.text(tx, 1.0 - 0.68 / fig_h, subtitle, fontsize=10,
             color=C["ink_soft"], ha="left", va="top")
    add_legend(fig, C, fig_w, fig_h, any_interp, n_suspect, legend_extra)

    fig.savefig(out_path, dpi=dpi, facecolor=C["surface"])
    plt.close(fig)
    return n_suspect


def draw_top(items, out_path: Path, theme: str, suspect_turns: float,
             smooth_mm: float = 1.2, scale: float = 0.25, dpi: int = 160):
    """The largest networks side by side in one figure, sharing the z axis.

    Panels are ordered by each network's median distance from the umbilicus
    (innermost first) and separated by a subtle dotted break.  Unrolled length
    starts at 0 on the left and runs continuously through every panel, and the
    winding gridlines are numbered 0, 1, 2, ... straight across -- knowing full
    well the spacing from one network to the next is arbitrary.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    C = THEME[theme]
    geoms = [network_geometry(it["comp"], it["fibers"], it["xy"], smooth_mm)
             for it in items]
    bnds = [network_bounds(it["comp"], g, scale) for it, g in zip(items, geoms)]
    y_lo = min(float(lo[1]) for lo, _ in bnds)
    y_hi = max(float(hi[1]) for _, hi in bnds)

    widths_cm = [float(hi[0] - lo[0]) for lo, hi in bnds]
    ax_h_in = (y_hi - y_lo) * scale
    M = MARGINS_IN

    # Panel offsets in the shared frame.  Each next panel starts on the global
    # X_TICK_CM grid, so every panel shows the same labeling interval; the gap
    # between networks is whatever that snap requires (at least MIN_GAP_CM) and
    # is real coordinate space -- the x axis runs continuously through it.
    offs, gaps_cm = [], []
    x_off = 0.0
    for k, w in enumerate(widths_cm):
        offs.append(x_off)
        if k < len(widths_cm) - 1:
            nxt = X_TICK_CM * math.ceil((x_off + w + MIN_GAP_CM) / X_TICK_CM)
            gaps_cm.append(nxt - (x_off + w))
            x_off = nxt
    fig_w = M["left"] + (sum(widths_cm) + sum(gaps_cm)) * scale + M["right"]
    fig_h = ax_h_in + M["top"] + M["bot"]
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(C["surface"])

    any_interp, n_suspect = False, 0
    n_wind = 0         # continuous winding count across the panels
    for k, (it, g, (lo, hi)) in enumerate(zip(items, geoms, bnds)):
        x_in = M["left"] + offs[k] * scale
        ax = fig.add_axes([x_in / fig_w, M["bot"] / fig_h,
                           widths_cm[k] * scale / fig_w, ax_h_in / fig_h])
        # shift this network into the shared frame: its left edge lands at
        # offs[k], so the unrolled length runs from 0 straight through every
        # panel
        shift = offs[k] - float(lo[0])
        for f in it["comp"]:
            g[f][1][:, 0] += shift
            g[f][3][:, 0] += shift
        circ = TWO_PI * it["r_ref_cm"]
        marks = [m * circ + shift
                 for m in range(math.ceil(float(lo[0]) / circ),
                                math.floor(float(hi[0]) / circ) + 1)]
        windings = [(x, str(n_wind + i)) for i, x in enumerate(marks)]
        n_wind += len(marks)
        plo = np.array([offs[k], y_lo])
        phi = np.array([offs[k] + widths_cm[k], y_hi])
        ai, ns = render_network(ax, C, it["comp"], it["fibers"], it["links"],
                                g, plo, phi, it["r_ref_cm"], scale,
                                suspect_turns, windings=windings)
        any_interp = any_interp or ai
        n_suspect += ns
        if k == 0:
            ax.set_ylabel("scroll z-axis (cm)", color=C["ink_soft"],
                          fontsize=10)
        else:
            ax.tick_params(labelleft=False)
            # the subtle break between panels
            bx = (x_in - 0.5 * gaps_cm[k - 1] * scale) / fig_w
            fig.add_artist(Line2D(
                [bx, bx], [M["bot"] / fig_h, (M["bot"] + ax_h_in) / fig_h],
                transform=fig.transFigure, color=C["winding"], lw=1.0,
                ls=(0, (2, 3)), alpha=0.8))
        ax.text(0.5, 1.0 + 0.24 / ax_h_in,
                f"network {it['ci'] + 1}  ·  r ≈ {it['r_ref_cm']:.2f} cm",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, color=C["ink_soft"])

    nfib = sum(len(it["comp"]) for it in items)
    nlink = sum(len(it["links"]) for it in items)
    subtitle = (f"{nfib} linked fibers  ·  {nlink} crossings  ·  "
                f"panels ordered inner → outer by median distance from the "
                f"umbilicus  ·  unrolled length and winding count run "
                f"continuously; spacing between networks is arbitrary")
    if n_suspect:
        subtitle += f"  ·  {n_suspect} winding-suspect links"
    tx = 0.5 / fig_w
    fig.text(tx, 1.0 - 0.35 / fig_h,
             "Linked papyrus fiber networks — largest three",
             fontsize=15, color=C["ink"], ha="left", va="top")
    fig.text(tx, 1.0 - 0.68 / fig_h, subtitle, fontsize=10,
             color=C["ink_soft"], ha="left", va="top")
    fig.text(0.5, (M["bot"] - 0.42) / fig_h, "unrolled length (cm)",
             ha="center", va="center", fontsize=10, color=C["ink_soft"])
    add_legend(fig, C, fig_w, fig_h, any_interp, n_suspect)

    fig.savefig(out_path, dpi=dpi, facecolor=C["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fibers-dir", type=Path,
                    default=common.FIBERS_DIR)
    ap.add_argument("--volpkg", type=Path,
                    default=common.VOLPKG)
    ap.add_argument("--out", type=Path,
                    default=common.PLOTS)
    ap.add_argument("--theme", choices=("light", "dark"), default="dark")
    ap.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    ap.add_argument("--umbilicus-scale", type=float, default=DEFAULT_UMBILICUS_SCALE)
    ap.add_argument("--min-fibers", type=int, default=3)
    ap.add_argument("--component", default=None,
                    help="only this network (index into the size-sorted list)")
    ap.add_argument("--suspect-turns", type=float, default=0.25,
                    help="flag links whose endpoints disagree by more than this "
                         "fraction of a turn after placement")
    ap.add_argument("--smooth-mm", type=float, default=1.2,
                    help="Gaussian arclength sigma for de-bumping the drawn "
                         "fibers (0 disables)")
    ap.add_argument("--scale", type=float, default=0.25,
                    help="figure inches per data cm, same for every network "
                         "and both axes")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    D = load_fibers(args.fibers_dir)
    umb = load_umbilicus(args.volpkg / "umbilicus.json", args.umbilicus_scale)
    to_cm = args.voxel_um / 10000.0
    args.out.mkdir(parents=True, exist_ok=True)

    comps = [c for c in all_components(D) if len(c) >= args.min_fibers]
    which = range(len(comps)) if args.component is None else [int(args.component)]
    print(f"{len(D)} fibers, {len(comps)} networks with >= {args.min_fibers} fibers")

    total_suspect = 0
    top_items = []
    for ci in which:
        comp = comps[ci]
        fibers = {f: build_fiber(f, D[f], umb) for f in comp}
        links = collect_links(D, comp, fibers)
        if not links:
            continue
        place_component(comp, fibers, links)
        xy, r_ref = component_xy(comp, fibers, to_cm)
        r_ref_cm = r_ref * to_cm
        if args.component is None and ci < 3:
            top_items.append(dict(ci=ci, comp=comp, fibers=fibers, links=links,
                                  xy=xy, r_ref_cm=r_ref_cm))

        name = f"network_{ci + 1:02d}"
        n_suspect = draw(comp, fibers, links, xy, r_ref_cm,
                         args.out / f"{name}.png", args.theme,
                         f"Linked papyrus fiber network {ci + 1}",
                         args.suspect_turns, args.smooth_mm, args.scale,
                         args.dpi)
        total_suspect += n_suspect

        worst = max((l.turn_err for l in links), default=0.0)
        flag = f"  ({n_suspect} winding-suspect)" if n_suspect else ""
        print(f"  {name}: {len(comp)} fibers, {len(links)} links, "
              f"worst link residual {worst:.2f} turns{flag}")
        for l in links:
            if l.turn_err > args.suspect_turns:
                print(f"    suspect: {fibers[l.fa].label}[{l.ia}] x "
                      f"{fibers[l.fb].label}[{l.ib}]  "
                      f"off by {l.turn_err:.2f} turns  "
                      f"(3D sep {l.sep3d * args.voxel_um / 1000.0:.2f} mm)")

    if len(top_items) >= 2:
        top_items.sort(key=lambda it: it["r_ref_cm"])
        draw_top(top_items, args.out / "top_networks.png", args.theme,
                 args.suspect_turns, args.smooth_mm, args.scale, args.dpi)
        order = "  <  ".join(
            f"network {it['ci'] + 1} (r {it['r_ref_cm']:.2f} cm)"
            for it in top_items)
        print(f"  top_networks.png: {order}")

    if total_suspect:
        print(f"\n{total_suspect} winding-suspect links across all networks -- "
              f"these are the links to review first")


if __name__ == "__main__":
    main()
