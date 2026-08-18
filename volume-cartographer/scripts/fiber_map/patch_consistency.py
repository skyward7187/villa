#!/usr/bin/env python3
"""Consistency audit, strict version.

Same three constraints, but each flag now has to survive a generous reading:
3D "touching" is tested at 8 ds2 voxels (~77 um, far beyond the 2-4 used to
*find* overlaps), and a 2D footprint clash has to be real area, not a slice of
silhouette slop.  A surviving flag therefore means the two things claim the same
(theta, z) on the same winding while sitting genuinely far apart in 3D.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

import common  # noqa: E402

common.add_scripts_to_path()
si = common.import_surface_index()
import fiber_patch_unroll as M  # noqa: E402
from fiber_network_unroll import (  # noqa: E402
    TWO_PI, all_components, build_fiber, collect_links, component_xy,
    load_fibers, load_umbilicus, place_component, umbilicus_center,
)

GP = common.PATCHES_DIR
REP = common.REPORTS
SC = common.WORK

FIND_TOL = 2.0      # ds2 voxels: what counts as an overlapping pair
FAR_TOL = 8.0       # ds2 voxels: beyond this, "they do not touch" is not slop
MIN_SHARED = 16
MIN_INSIDE_CM = 0.05    # fiber arclength inside a footprint
MIN_AREA_FRAC = 0.10    # footprint clash as a fraction of the smaller footprint
ERODE_CELLS = common.ERODE_CELLS         # one grid ring, matching the reports this audits


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


D = load_fibers(common.FIBERS_DIR)
umb = load_umbilicus(common.VOLPKG / 'umbilicus.json', 4.0)
to_cm = 2.4 / 10000.0
comp = all_components(D)[common.NETWORK]
fibers = {f: build_fiber(f, D[f], umb) for f in comp}
links = collect_links(D, comp, fibers)
place_component(comp, fibers, links)
xy, r_ref = component_xy(comp, fibers, to_cm)
ov = M.load_overlap(common.overlap_report())
patches, _ = M.build_patches(GP, ov, set(comp), fibers, umb, r_ref, to_cm, 4.0,
                            ERODE_CELLS)
by = {p['name']: p for p in patches}
names = [p['name'] for p in patches]
log(f'{len(patches)} patches, {len(comp)} fibers')

fcurve = {f: np.stack([(fibers[f].theta_line + fibers[f].offset) * r_ref * to_cm,
                       fibers[f].line[:, 2] * to_cm], axis=1) for f in comp}

grids = {}


def load_surface(name):
    zyx, valid = M.load_patch_surface(GP / name, ERODE_CELLS)
    grids[name] = (np.ascontiguousarray(zyx[..., ::-1]), valid)
    return si.QuadSurface(name, zyx, 0.05, 0.05)


with ThreadPoolExecutor(32) as ex:
    surfaces = list(ex.map(load_surface, names))
index = si.SurfacePatchIndex()
index.rebuild(surfaces, FAR_TOL, 1)
ids = index.surface_ids()
idx_of = {n: i for i, n in enumerate(ids)}
log('index built')


def scan(src):
    xyz, valid = grids[src]
    pts = xyz[valid][::2]
    off, sidx, dist, _ij = index.locate_all_xyz_batch(
        np.ascontiguousarray(pts.astype(np.float32)), FAR_TOL)
    if not len(sidx):
        return []
    qpt = np.repeat(np.arange(len(pts)), np.diff(off))
    out = []
    for s in np.unique(sidx):
        dst = ids[int(s)]
        if dst == src:
            continue
        sel = sidx == s
        mine, dm = qpt[sel], dist[sel]
        near = mine[dm <= FIND_TOL]
        pick = (near if len(near) >= MIN_SHARED else mine)
        if len(pick) < MIN_SHARED:
            continue
        chosen = pick[np.linspace(0, len(pick) - 1, min(12, len(pick))).astype(int)]
        out.append((src, dst, pts[chosen], len(near) >= MIN_SHARED))
    return out


t = time.time()
res = []
with ThreadPoolExecutor(32) as ex:
    for r in ex.map(scan, names):
        res.extend(r)
log(f'{len(res)} directed near/overlap pairs in {time.time() - t:.1f}s')

touch_far = {tuple(sorted((a, b))) for a, b, _, _ in res}
overlap = {}
for a, b, shared, close in res:
    if close:
        overlap.setdefault(tuple(sorted((a, b))), shared)
log(f'{len(overlap)} undirected pairs overlap within {FIND_TOL} ds2 vox; '
    f'{len(touch_far)} come within {FAR_TOL}')

# --- A: overlapping patches must agree on the winding
disagree, agree = {}, 0
for (a, b), shared in overlap.items():
    p, q = by[a], by[b]
    s = shared * 4.0
    ctr = umbilicus_center(umb, s[:, 2])
    raw = np.arctan2((s[:, :2] - ctr)[:, 1], (s[:, :2] - ctr)[:, 0])
    tp = p['ref'] + M.wrap_to_pi(raw - p['ref']) + p['offset']
    tq = q['ref'] + M.wrap_to_pi(raw - q['ref']) + q['offset']
    turns = float(np.median((tp - tq) / TWO_PI))
    if abs(turns) > 0.5:
        disagree.setdefault(a, []).append(b)
        disagree.setdefault(b, []).append(a)
    else:
        agree += 1
log(f'A: {agree} agree / {sum(len(v) for v in disagree.values()) // 2} disagree '
    f'({len(disagree)} patches)')

paths, boxes, areas = {}, {}, {}
for p in patches:
    poly = p['poly']
    paths[p['name']] = MplPath(poly)
    boxes[p['name']] = (poly.min(0), poly.max(0))
    x, y = poly[:, 0], poly[:, 1]
    areas[p['name']] = abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2


# --- B: fiber runs through the footprint but stays far from the surface in 3D
def check_b(p):
    n = p['name']
    lo, hi = boxes[n]
    claimed = {f for f, _ in ov[n]}
    hits = []
    for f in comp:
        if f in claimed:
            continue
        C2 = fcurve[f]
        m = ((C2[:, 0] >= lo[0]) & (C2[:, 0] <= hi[0]) &
             (C2[:, 1] >= lo[1]) & (C2[:, 1] <= hi[1]))
        if not m.any():
            continue
        sub = C2[m]
        ins = paths[n].contains_points(sub)
        if not ins.any():
            continue
        seg = np.linalg.norm(np.diff(sub[ins], axis=0), axis=1)
        if float(seg[seg < 0.05].sum()) < MIN_INSIDE_CM:
            continue
        # exact 3D check: are those fiber points anywhere near this surface?
        raw3d = fibers[f].line[np.flatnonzero(m)[ins]] / 4.0    # -> ds2
        q = np.ascontiguousarray(raw3d.astype(np.float32))
        sidx, dist, _ij = index.locate_xyz_nearest_batch(q, FAR_TOL)
        if not np.any(sidx == idx_of[n]):
            hits.append(f)
    return n, hits


t = time.time()
Bviol = {}
with ThreadPoolExecutor(16) as ex:
    for n, hits in ex.map(check_b, patches):
        if hits:
            Bviol[n] = hits
log(f'B: {len(Bviol)} patches crossed by a fiber that stays >{FAR_TOL} ds2 vox away '
    f'({sum(len(v) for v in Bviol.values())} cases) in {time.time() - t:.1f}s')


# --- C: footprints share real area but the surfaces never come close
def clash_area(a, b):
    la, ha = boxes[a]
    lb, hb = boxes[b]
    lo = np.maximum(la, lb)
    hi = np.minimum(ha, hb)
    if np.any(hi <= lo):
        return 0.0
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 24), np.linspace(lo[1], hi[1], 24))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    both = paths[a].contains_points(pts) & paths[b].contains_points(pts)
    cell = (hi[0] - lo[0]) * (hi[1] - lo[1]) / both.size
    return float(both.sum()) * cell


order = sorted(names, key=lambda n: boxes[n][0][0])
los = np.array([boxes[n][0] for n in order])
his = np.array([boxes[n][1] for n in order])
Cviol = {}
t = time.time()
for i, n in enumerate(order):
    j = i + 1
    while j < len(order) and los[j][0] <= his[i][0]:
        m = order[j]
        if (los[j][1] <= his[i][1] and los[i][1] <= his[j][1]
                and tuple(sorted((n, m))) not in touch_far):
            ar = clash_area(n, m)
            if ar >= MIN_AREA_FRAC * min(areas[n], areas[m]):
                Cviol.setdefault(n, []).append(m)
                Cviol.setdefault(m, []).append(n)
        j += 1
log(f'C: {len(Cviol)} patches with a real footprint clash against a patch they '
    f'never come within {FAR_TOL} ds2 vox of, in {time.time() - t:.1f}s')

json.dump({
    'params': {'find_tol': FIND_TOL, 'far_tol': FAR_TOL,
               'min_inside_cm': MIN_INSIDE_CM, 'min_area_frac': MIN_AREA_FRAC,
               'erode_cells': ERODE_CELLS},
    'all_patches': names,
    'fiber_count': {p['name']: len(p['fibers']) for p in patches},
    'straddled': {p['name']: p['straddled'] for p in patches if p['straddled']},
    'overlap_pairs': [list(k) for k in overlap],
    'A_disagree': disagree,
    'B_violations': Bviol,
    'C_violations': Cviol,
}, open(common.consistency_report(), 'w'))
log(f'wrote {common.consistency_report().name}')
