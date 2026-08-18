#!/usr/bin/env python3
"""Cache the inputs the overlap stage needs: fiber points and patch bboxes.

Splitting this out keeps fiber_patch_overlap.py from re-reading 66k meta.json
files and re-parsing every fiber on each run. Rerun it whenever the fiber set
changes; the patch cache only changes when patches are added.

Writes into common.WORK:
    fiber_pts_meta.json   per-fiber trimmed point counts
    fp_<fiber>.npy        trimmed dense polyline, scaled into ds2
    patch_names.json      every patch directory name
    patch_areas.json      area_cm2, parallel to patch_names
    patch_lo/hi.npy       patch bounding boxes, parallel to patch_names
    patch_keep_idx.json   patches whose bbox is near this network's fibers
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common

common.add_scripts_to_path()
import sys
sys.path.insert(0, str(common.VC_ROOT.parent / "vesuvius" / "src"))

from fiber_network_unroll import all_components, load_fibers  # noqa: E402
from vc3d_fiber_format import parse_vc3d_fiber_format  # noqa: E402

FIBER_SCALE = 0.25      # annotation (2.4 um) -> ds2, as load_fiber_point_collection
CELL = 64.0             # spatial hash cell for the bbox prefilter, ds2 voxels
PAD = 8.0               # prefilter pad; must exceed the overlap tolerance


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    common.WORK.mkdir(parents=True, exist_ok=True)

    D = load_fibers(common.FIBERS_DIR)
    comp = all_components(D)[common.NETWORK]
    log(f"network {common.NETWORK + 1}: {len(comp)} fibers of {len(D)}")

    # Trim each fiber to its first-to-last control point: the tracer extends the
    # polyline ~150 points past both ends, and those tails would manufacture
    # overlaps with patches the fiber never really reaches.
    meta = {}
    for name in comp:
        d = json.loads((common.FIBERS_DIR / name).read_text())
        p = parse_vc3d_fiber_format(d, path=name)
        cp = np.asarray(p.control_points_xyz, float)
        lp = np.asarray(p.line_points_xyz, float)
        if lp.ndim != 2 or len(lp) < 2 or not len(cp):
            log(f"  skipping {name}: no usable polyline")
            continue
        a = int(np.linalg.norm(lp - cp[0], axis=1).argmin())
        b = int(np.linalg.norm(lp - cp[-1], axis=1).argmin())
        lo, hi = sorted((a, b))
        trimmed = lp[lo:hi + 1]
        meta[name] = {"n_line": len(lp), "n_trim": len(trimmed), "n_cp": len(cp)}
        np.save(common.WORK / f"fp_{name}.npy",
                (trimmed * FIBER_SCALE).astype(np.float32))
    (common.WORK / "fiber_pts_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"cached {len(meta)} fibers, {sum(v['n_trim'] for v in meta.values())} points")

    def read_meta(name):
        try:
            m = json.loads((common.PATCHES_DIR / name / "meta.json").read_text())
            box = m.get("bbox")
            return (name, box[0], box[1], float(m.get("area_cm2", 0))) if box else None
        except Exception:
            return None

    with ThreadPoolExecutor(32) as ex:
        metas = [r for r in ex.map(read_meta, sorted(os.listdir(common.PATCHES_DIR))) if r]
    lo = np.array([m[1] for m in metas], np.float32)
    hi = np.array([m[2] for m in metas], np.float32)
    np.save(common.WORK / "patch_lo.npy", lo)
    np.save(common.WORK / "patch_hi.npy", hi)
    (common.WORK / "patch_names.json").write_text(json.dumps([m[0] for m in metas]))
    (common.WORK / "patch_areas.json").write_text(json.dumps([m[3] for m in metas]))
    log(f"read {len(metas)} patch bboxes")

    # Prefilter: keep patches whose padded bbox touches a cell holding a fiber
    # point. Conservative -- a patch within tolerance of the surface is
    # necessarily within tolerance of the bbox.
    pts = np.concatenate([np.load(common.WORK / f"fp_{m}.npy") for m in meta]).astype(np.float64)
    cells = set(map(tuple, np.floor(pts / CELL).astype(np.int64)))
    clo = np.floor((lo - PAD) / CELL).astype(np.int64)
    chi = np.floor((hi + PAD) / CELL).astype(np.int64)
    keep = []
    for i in range(len(metas)):
        a, b = clo[i], chi[i]
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
            keep.append(i)
    (common.WORK / "patch_keep_idx.json").write_text(json.dumps(keep))
    log(f"{len(keep)} of {len(metas)} patches are candidates")


if __name__ == "__main__":
    main()
