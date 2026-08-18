#!/bin/bash
# Regenerate every report for one fiber network, in dependency order.
#
#   ./run_pipeline.sh            # network 0 (largest), one ring of erosion
#   FIBER_MAP_NETWORK=1 ./run_pipeline.sh
#   PATCH_ERODE_CELLS=0 ./run_pipeline.sh
#
# Stages 2-4 each read what the previous one wrote, so running them out of
# order silently mixes stale data -- that is why this exists as one script.
set -euo pipefail
cd "$(dirname "$0")"
PY=${FIBER_MAP_PYTHON:-/home/djosey/venv/bin/python}

echo "=== 1/5 prepare inputs (fiber points, patch bboxes, prefilter)"
"$PY" prepare_inputs.py
echo "=== 2/5 fiber <-> patch overlap"
"$PY" fiber_patch_overlap.py
echo "=== 3/5 patch <-> patch neighbours (one level out)"
"$PY" patch_neighbours.py
echo "=== 4/5 consistency audit"
"$PY" patch_consistency.py
echo "=== 5/5 trust tiers"
"$PY" patch_trust.py
echo "=== done"
