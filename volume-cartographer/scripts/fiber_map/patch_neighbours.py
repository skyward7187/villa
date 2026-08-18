#!/usr/bin/env python3
"""One level outward: patches that overlap the fiber-carrying patches.

Reads the patch->fiber report from fiber_patch_overlap.py, takes those patches as
level 0, and finds every other graph patch whose surface lies within
``TOLERANCE`` of a level-0 patch's vertices.  Same method and defaults as
scripts/spiral/connect_overlapping_patches.py: SurfacePatchIndex over the
candidate patches, each level-0 patch's valid vertices queried against it in a
batch, a neighbour counted when enough of those vertices land on it.

For each neighbour it also records anchor points -- level-0 vertices that lie on
the neighbour's surface.  Those are shared 3D points, which is what
fiber_patch_unroll.py needs to give a neighbour the same whole-turn offset as
the level-0 patch it touches (a neighbour carries no fiber of its own, so it has
no other way to be placed).
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pathlib import Path

import common  # noqa: E402

common.add_scripts_to_path()
si = common.import_surface_index()
from fiber_patch_unroll import load_patch_surface  # noqa: E402

GP = str(common.PATCHES_DIR)
OUT = str(common.REPORTS)
LEVEL0 = str(common.overlap_report())

TOLERANCE = 2.0          # ds2 voxels, connect_overlapping_patches.py default
MIN_OVERLAP_POINTS = 16  # ditto
QUERY_STRIDE = 2         # every other level-0 vertex is plenty at 20-vox spacing
MAX_ANCHORS = 12
CELL = 64.0
ERODE_CELLS = common.ERODE_CELLS          # one grid ring, matching fiber_patch_overlap.py
THREADS = 32


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def read_meta(d):
    try:
        m = json.load(open(f'{GP}/{d}/meta.json'))
        b = m.get('bbox')
        return (d, b[0], b[1]) if b else None
    except Exception:
        return None


def load_grid(d):
    zyx, valid = load_patch_surface(Path(GP) / d, ERODE_CELLS)
    return zyx[..., ::-1], valid


level0 = sorted(json.load(open(LEVEL0))['patches'])
log(f'{len(level0)} level-0 (fiber-carrying) patches')

dirs = sorted(os.listdir(GP))
with ThreadPoolExecutor(THREADS) as ex:
    metas = [r for r in ex.map(read_meta, dirs) if r]
names = [m[0] for m in metas]
lo = np.array([m[1] for m in metas], np.float32)
hi = np.array([m[2] for m in metas], np.float32)
pos = {n: i for i, n in enumerate(names)}
log(f'{len(names)} patches on disk')

# cells covered by the level-0 bounding boxes, padded by the tolerance
cells = set()
for n in level0:
    i = pos[n]
    a = np.floor((lo[i] - TOLERANCE) / CELL).astype(np.int64)
    b = np.floor((hi[i] + TOLERANCE) / CELL).astype(np.int64)
    for cx in range(a[0], b[0] + 1):
        for cy in range(a[1], b[1] + 1):
            for cz in range(a[2], b[2] + 1):
                cells.add((cx, cy, cz))
log(f'{len(cells)} occupied cells')

keep = []
for i, n in enumerate(names):
    a = np.floor((lo[i] - TOLERANCE) / CELL).astype(np.int64)
    b = np.floor((hi[i] + TOLERANCE) / CELL).astype(np.int64)
    hit = False
    for cx in range(a[0], b[0] + 1):
        for cy in range(a[1], b[1] + 1):
            for cz in range(a[2], b[2] + 1):
                if (cx, cy, cz) in cells:
                    hit = True
                    break
            if hit:
                break
        if hit:
            break
    if hit:
        keep.append(n)
log(f'{len(keep)} candidate patches after the bbox prefilter')

grids = {}


def load_surface(n):
    zyx, valid = load_patch_surface(Path(GP) / n, ERODE_CELLS)
    grids[n] = (np.ascontiguousarray(zyx[..., ::-1]), valid)
    return si.QuadSurface(n, zyx, 0.05, 0.05)


t = time.time()
with ThreadPoolExecutor(THREADS) as ex:
    surfaces = list(ex.map(load_surface, keep))
log(f'loaded in {time.time() - t:.1f}s')

index = si.SurfacePatchIndex()
index.rebuild(surfaces, TOLERANCE, 1)
ids = index.surface_ids()
log(f'index: {index.surface_count()} surfaces, {index.patch_count()} bvh patches')

level0_set = set(level0)
found = {}   # neighbour -> {'points': int, 'via': str, 'anchors': [[x,y,z], ...]}
lock_free = []


def scan(src):
    xyz, valid = grids[src]
    pts = xyz[valid][::QUERY_STRIDE]
    if not len(pts):
        return None
    q = np.ascontiguousarray(pts.astype(np.float32))
    off, sidx, _dist, _ij = index.locate_all_xyz_batch(q, TOLERANCE)
    if not len(sidx):
        return None
    qpt = np.repeat(np.arange(len(pts)), np.diff(off))
    out = []
    for s in np.unique(sidx):
        name = ids[int(s)]
        if name == src or name in level0_set:
            continue
        mine = qpt[sidx == s]
        if len(mine) < MIN_OVERLAP_POINTS:
            continue
        pick = mine[np.linspace(0, len(mine) - 1, min(MAX_ANCHORS, len(mine))).astype(int)]
        out.append((name, int(len(mine)), src, pts[pick].astype(float).tolist()))
    return out


t = time.time()
with ThreadPoolExecutor(THREADS) as ex:
    for res in ex.map(scan, level0):
        for name, npts, src, anchors in (res or []):
            cur = found.get(name)
            if cur is None or npts > cur['points']:
                found[name] = {'points': npts, 'via': src, 'anchors': anchors}
log(f'scanned {len(level0)} level-0 patches in {time.time() - t:.1f}s')
log(f'{len(found)} level-1 neighbours')

json.dump({
    'description': 'graph patches overlapping the fiber-carrying (level-0) patches',
    'source': os.path.basename(LEVEL0),
    'criteria': {
        'tolerance_ds2_voxels': TOLERANCE,
        'min_overlap_points': MIN_OVERLAP_POINTS,
        'query_stride': QUERY_STRIDE,
        'erode_cells': ERODE_CELLS,
        'geometry': 'vc.surface_index.SurfacePatchIndex exact point-to-quad projection',
        'anchors': 'level-0 vertices lying on the neighbour, ds2 coords',
    },
    'level0_count': len(level0),
    'level1_count': len(found),
    'patches': dict(sorted(found.items())),
}, open(common.neighbour_report(), 'w'), indent=1)
log(f'wrote {common.neighbour_report().name}')
