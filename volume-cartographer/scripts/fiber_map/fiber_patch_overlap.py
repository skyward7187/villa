#!/usr/bin/env python3
"""Report which graph_patches each fiber of the largest fiber-link network overlaps.

Geometry comes from vc.surface_index.SurfacePatchIndex (exact point-to-quad
projection), the same engine fit_spiral's link_points_to_patches and
connect_overlapping_patches.py use.

Frames: fiber JSON control/line points are VC3D full-res xyz; patches (and the
spiral tracks store they were grown from) are ds2. Fiber coords are scaled by
0.25, matching spiral_helpers.load_fiber_point_collection(coordinate_scale=0.25).
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

SC = str(common.WORK)
GP = str(common.PATCHES_DIR)

TOL = 4.0          # ds2 voxels: fiber point counts as on-surface within this
# A run must be a genuine stretch along the patch, not a graze. 98 of the 111
# fibers are traced at 2.0 ds2-voxel point spacing but 13 are at ~8.0, so the
# threshold is arclength (18 vox == 10 consecutive points at 2.0 spacing) rather
# than a point count, which would be 4x stricter on the coarse fibers.
MIN_RUN_ARC = 18.0
MIN_RUN_PTS = 3
MAX_ANCHORS = 12   # on-surface fiber points recorded per (patch, fiber)
ERODE_CELLS = common.ERODE_CELLS    # shrink every patch by one grid ring (20 ds2 vox, 192 um);
                   # band grower borders are the least reliable part of a patch
QUERY_CHUNK = 4096
THREADS = 32


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------- fiber points
fiber_meta = json.load(open(f'{SC}/fiber_pts_meta.json'))
fiber_names = list(fiber_meta)
fiber_pts = [np.load(f'{SC}/fp_{m}.npy') for m in fiber_names]
counts = np.array([len(p) for p in fiber_pts])
starts = np.concatenate([[0], np.cumsum(counts)])
points = np.ascontiguousarray(np.concatenate(fiber_pts).astype(np.float32))
n_points = len(points)
# per-point owning fiber and index within that fiber
owner = np.repeat(np.arange(len(fiber_names)), counts)
local = np.concatenate([np.arange(c) for c in counts])
# cumulative arclength within each fiber (ds2 voxels)
arc = np.concatenate([
    np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p.astype(np.float64), axis=0), axis=1))])
    for p in fiber_pts
])
log(f'{len(fiber_names)} fibers, {n_points} points')

# ------------------------------------------------------------------- patches
names = json.load(open(f'{SC}/patch_names.json'))
areas = json.load(open(f'{SC}/patch_areas.json'))
keep = json.load(open(f'{SC}/patch_keep_idx.json'))
log(f'loading {len(keep)} candidate patches')


def load(i):
    d = names[i]
    zyx, _valid = load_patch_surface(Path(GP) / d, ERODE_CELLS)
    return si.QuadSurface(d, zyx, 0.05, 0.05)


t = time.time()
with ThreadPoolExecutor(THREADS) as ex:
    surfaces = list(ex.map(load, keep))
log(f'loaded in {time.time() - t:.1f}s')

index = si.SurfacePatchIndex()
t = time.time()
index.rebuild(surfaces, TOL, 1)
log(f'index rebuilt in {time.time() - t:.1f}s: '
    f'{index.surface_count()} surfaces, {index.patch_count()} bvh patches')
surface_ids = index.surface_ids()

# --------------------------------------------------------------------- query
bounds = list(range(0, n_points, QUERY_CHUNK))


def query(begin):
    end = min(begin + QUERY_CHUNK, n_points)
    off, sidx, dist, ij = index.locate_all_xyz_batch(
        np.ascontiguousarray(points[begin:end]), TOL)
    if len(sidx) == 0:
        return None
    pt = begin + np.repeat(np.arange(end - begin), np.diff(off))
    return pt, sidx.copy(), dist.copy()


t = time.time()
with ThreadPoolExecutor(THREADS) as ex:
    chunks = [c for c in ex.map(query, bounds) if c is not None]
log(f'queried {n_points} points in {time.time() - t:.1f}s')

if not chunks:
    log('no hits at all')
    raise SystemExit(1)

hit_pt = np.concatenate([c[0] for c in chunks])
hit_surf = np.concatenate([c[1] for c in chunks])
hit_dist = np.concatenate([c[2] for c in chunks])
log(f'{len(hit_pt)} raw (point, patch) hits within {TOL} vox')

# ----------------------------------------------------------------- aggregate
# group by (patch, fiber), then find runs of consecutive local indices
key = hit_surf.astype(np.int64) * len(fiber_names) + owner[hit_pt]
order = np.lexsort((local[hit_pt], key))
key_s = key[order]
pt_s = hit_pt[order]
dist_s = hit_dist[order]

edges = np.flatnonzero(np.diff(key_s)) + 1
groups = np.concatenate([[0], edges, [len(key_s)]])

records = []
for g in range(len(groups) - 1):
    a, b = groups[g], groups[g + 1]
    sid = int(key_s[a] // len(fiber_names))
    fid = int(key_s[a] % len(fiber_names))
    idxs = local[pt_s[a:b]]
    dists = dist_s[a:b]
    arcs = arc[pt_s[a:b]]
    # consecutive runs in the fiber's own point order
    brk = np.flatnonzero(np.diff(idxs) != 1) + 1
    run_bounds = np.concatenate([[0], brk, [len(idxs)]])
    runs = []
    for r in range(len(run_bounds) - 1):
        ra, rb = run_bounds[r], run_bounds[r + 1]
        runs.append((int(rb - ra), float(arcs[rb - 1] - arcs[ra]),
                     int(idxs[ra]), int(idxs[rb - 1])))
    longest = max(runs, key=lambda r: r[1])
    qualifying = [r for r in runs if r[1] >= MIN_RUN_ARC and r[0] >= MIN_RUN_PTS]
    # Anchors: fiber points that are genuinely ON this patch's surface (within
    # TOL of it), sampled across the qualifying runs.  Consumers that need to
    # tie the patch to the fiber cannot rediscover these without the index --
    # a bounding-box test is not a substitute, because a fiber winding past on
    # the neighbouring wrap also falls inside the box.
    anchor_rows = [pt_s[a:b][(idxs >= lo) & (idxs <= hi)]
                   for _, _, lo, hi in qualifying]
    anchor_pts = np.concatenate(anchor_rows) if anchor_rows else np.empty(0, int)
    if len(anchor_pts) > MAX_ANCHORS:
        anchor_pts = anchor_pts[
            np.linspace(0, len(anchor_pts) - 1, MAX_ANCHORS).astype(int)]
    records.append({
        'patch': surface_ids[sid],
        'fiber': fiber_names[fid],
        'points_on_patch': int(b - a),
        'longest_run_points': longest[0],
        'longest_run_arclength_ds2': round(longest[1], 1),
        'total_run_arclength_ds2': round(sum(r[1] for r in qualifying), 1),
        'num_runs': len(runs),
        'num_qualifying_runs': len(qualifying),
        'median_distance': round(float(np.median(dists)), 3),
        'min_distance': round(float(dists.min()), 3),
        'fiber_point_span': [int(idxs.min()), int(idxs.max())],
        'anchors': np.round(points[anchor_pts], 2).tolist(),
        'qualifies': len(qualifying) > 0,
    })

log(f'{len(records)} (patch, fiber) pairs with >=1 hit; '
    f'{sum(r["qualifies"] for r in records)} qualify '
    f'(run >= {MIN_RUN_ARC} vox and >= {MIN_RUN_PTS} pts)')

json.dump({
    'tolerance_ds2_voxels': TOL,
    'min_run_arclength_ds2': MIN_RUN_ARC,
    'min_run_points': MIN_RUN_PTS,
    'fiber_coordinate_scale': 0.25,
    'erode_cells': ERODE_CELLS,
    'network_fiber_count': len(fiber_names),
    'candidate_patches': len(keep),
    'total_patches': len(names),
    'records': records,
}, open(f'{SC}/overlap_records.json', 'w'), indent=1)
log('wrote overlap_records.json')

# ------------------------------------------------- the per-network report
# Grouped by patch; this is what fiber_patch_unroll, patch_neighbours and
# network_unroll_tifxyz all read.
from collections import defaultdict  # noqa: E402

qualifying = [r for r in records if r['qualifies']]
area_by_name = dict(zip(names, areas))
by_patch = defaultdict(list)
for r in qualifying:
    by_patch[r['patch']].append(r)

grouped = {
    'description': f'graph patches overlapped by network {common.NETWORK + 1} '
                   f'of fiber_network_unroll.py',
    'criteria': {
        'tolerance_ds2_voxels': TOL,
        'min_run_arclength_ds2_voxels': MIN_RUN_ARC,
        'min_run_points': MIN_RUN_PTS,
        'fiber_coordinate_scale': 0.25,
        'erode_cells': ERODE_CELLS,
        'anchors': 'fiber points lying on the patch surface, ds2 coords',
        'geometry': 'vc.surface_index.SurfacePatchIndex exact point-to-quad projection',
    },
    'network_fiber_count': len(fiber_names),
    'patches_total_scanned': len(names),
    'patches_with_overlap': len(by_patch),
    'pairs': len(qualifying),
    'patches': {
        p: {
            'area_cm2': round(area_by_name.get(p, 0.0), 4),
            'fiber_count': len(v),
            'fibers': [{
                'fiber': r['fiber'],
                'overlap_arclength_ds2vox': r['total_run_arclength_ds2'],
                'longest_run_ds2vox': r['longest_run_arclength_ds2'],
                'points_on_patch': r['points_on_patch'],
                'median_distance_ds2vox': r['median_distance'],
                'anchors': r['anchors'],
            } for r in v],
        }
        for p, v in sorted(by_patch.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    },
}
out = common.overlap_report()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(grouped, indent=1))
log(f'wrote {out.name}: {len(by_patch)} patches, {len(qualifying)} pairs, '
    f'{len({r["fiber"] for r in qualifying})} fibers hit')
