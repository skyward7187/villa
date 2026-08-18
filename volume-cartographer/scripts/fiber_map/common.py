#!/usr/bin/env python3
"""Shared paths and imports for the fiber-map pipeline.

Every script in this folder goes through here for three things that were
previously copy-pasted with absolute paths baked in: where the data lives, where
intermediates go, and how to import the native surface index.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
VC_ROOT = SCRIPTS.parent

# ---------------------------------------------------------------- data layout
# Override any of these with the matching environment variable.
FIBERS_DIR = Path(os.environ.get(
    "FIBER_MAP_FIBERS", "/media/djosey/nvme2/fibers/PHercParis4.volpkg.json"))
VOLPKG = Path(os.environ.get(
    "FIBER_MAP_VOLPKG", "/media/djosey/nvme2/PHercParis4.volpkg"))
PATCHES_DIR = Path(os.environ.get(
    "FIBER_MAP_PATCHES", "/media/djosey/nvme2/graph_patches"))
REPORTS = Path(os.environ.get(
    "FIBER_MAP_REPORTS", "/media/djosey/nvme2/fiber_patch_overlap"))
PLOTS = Path(os.environ.get(
    "FIBER_MAP_PLOTS", "/media/djosey/nvme2/fibers/PHercParis4_network_2d"))
MERGED = Path(os.environ.get(
    "FIBER_MAP_MERGED", "/media/djosey/nvme2/network_unrolled"))

# Intermediates (fiber point caches, patch bboxes, the prefilter). Durable by
# default so a rerun does not depend on whatever /tmp happened to hold.
WORK = Path(os.environ.get("FIBER_MAP_WORK", REPORTS / "work"))

# One grid ring of border erosion, honoured by every stage so the reports and
# the maps never disagree about what a patch is. See fiber_patch_unroll.md.
ERODE_CELLS = int(os.environ.get("PATCH_ERODE_CELLS", "1"))

NETWORK = int(os.environ.get("FIBER_MAP_NETWORK", "0"))          # 0 = largest


def tag(component: int | None = None) -> str:
    """Filename stem for a network's reports, e.g. 'network01'."""
    return f"network{(NETWORK if component is None else component) + 1:02d}"


def overlap_report(component: int | None = None) -> Path:
    return REPORTS / f"{tag(component)}_patches_to_fibers.json"


def neighbour_report(component: int | None = None) -> Path:
    return REPORTS / f"{tag(component)}_patch_neighbours.json"


def trust_report(component: int | None = None) -> Path:
    return REPORTS / f"{tag(component)}_patch_trust.json"


def consistency_report(component: int | None = None) -> Path:
    return WORK / f"{tag(component)}_consistency.json"


# ------------------------------------------------------------- native imports
def import_surface_index():
    """vc.surface_index, from the build tree or the stashed fallback.

    The build directory keeps getting reconfigured with VC_BUILD_PYTHON=OFF,
    which deletes build/python/. REPORTS/vc_ext holds a copy plus the VC
    libraries it links against; that copy needs LD_LIBRARY_PATH pointing at it,
    because the module's RUNPATH refers to the build tree. See its README.
    """
    for candidate in (VC_ROOT / "build" / "python" / "vc", REPORTS / "vc_ext"):
        if (candidate).is_dir() and any(candidate.glob("surface_index*.so")):
            sys.path.insert(0, str(candidate))
            try:
                import surface_index  # noqa: F401
                return surface_index
            except ImportError:
                sys.path.pop(0)
    raise SystemExit(
        "vc.surface_index not importable.\n"
        f"  build it:  cmake -S {VC_ROOT} -B {VC_ROOT}/build -DVC_BUILD_PYTHON=ON "
        f"&& ninja -C {VC_ROOT}/build vc_surface_index\n"
        f"  or run with: LD_LIBRARY_PATH={REPORTS / 'vc_ext'}")


def add_scripts_to_path():
    """Make this folder importable (fiber_patch_unroll, fiber_network_unroll)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
