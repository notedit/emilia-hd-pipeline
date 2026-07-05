#!/usr/bin/env bash
# Launch N self-healing Phase-1 worker loops, one per GPU, fully detached.
#
#   scripts/run_full.sh configs/full.yaml 8
#
# Each loop: run the worker on its hash partition -> guard (quarantine
# poison shards, print remaining pending) -> exit when this partition is
# empty AND the downloader wrote DOWNLOAD_DONE; otherwise sleep and re-run
# (picks up crashes AND newly downloaded tars). Survives SSH disconnect
# (setsid); safe to re-run at any time (done markers make workers idempotent).
set -euo pipefail

CFG=${1:?usage: run_full.sh CONFIG NUM_GPUS}
N=${2:?usage: run_full.sh CONFIG NUM_GPUS}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

DATA_ROOT=$("$PY" - "$CFG" <<'EOF'
import sys; sys.path.insert(0, ".")
from emilia_pipeline.common.config import load_config
print(load_config(sys.argv[1]).paths.source.parent)
EOF
)
LOGDIR="$DATA_ROOT/logs"
mkdir -p "$LOGDIR"

for g in $(seq 0 $((N - 1))); do
  setsid nohup bash -c "
    cd '$ROOT'
    # Cap BLAS/OMP threads: without this every worker (and each of its pool
    # children) spawns nproc-wide thread teams -> 8-way oversubscription that
    # made the first full-scale pass ~5x slower per shard.
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    while true; do
      CUDA_VISIBLE_DEVICES=$g '$PY' -m emilia_pipeline.phase1.worker \
        --config '$CFG' --num-workers $N --worker-index $g \
        >> '$LOGDIR/worker-$g.log' 2>&1 || true
      PENDING=\$('$PY' scripts/phase1_guard.py --config '$CFG' \
        --num-workers $N --worker-index $g 2>>'$LOGDIR/worker-$g.log' | tail -1)
      echo \"[loop-$g] pass done, pending=\$PENDING\" >> '$LOGDIR/worker-$g.log'
      if [ \"\$PENDING\" = 0 ] && [ -e '$DATA_ROOT/DOWNLOAD_DONE' ]; then
        echo \"[loop-$g] ALL DONE\" >> '$LOGDIR/worker-$g.log'
        break
      fi
      sleep 30
    done
  " >/dev/null 2>&1 < /dev/null &
  echo "launched loop for GPU $g (partition $g/$N)"
done
echo "logs: $LOGDIR/worker-{0..$((N - 1))}.log"
