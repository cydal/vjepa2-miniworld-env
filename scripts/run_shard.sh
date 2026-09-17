#!/bin/bash
# Generate one shard's full episode count in small batches, each a fresh
# process -- works around a real Panda3D memory leak (RSS grows ~linearly
# with reset() count; see the note in generate_carnav.py) by letting the OS
# reclaim everything on process exit instead of relying on a long-running
# process's memory staying bounded.
#
# Usage: run_shard.sh <shard_id> <seed_start> <total_episodes> <out_dir> [batch_size]
set -e
SHARD_ID="$1"
SEED_START="$2"
TOTAL="$3"
OUT_DIR="$4"
BATCH="${5:-50}"

PY=/home/ubuntu/miniconda3/envs/carnav/bin/python
cd "$(dirname "$0")/.."

done_count=0
while [ "$done_count" -lt "$TOTAL" ]; do
  remaining=$((TOTAL - done_count))
  n=$((remaining < BATCH ? remaining : BATCH))
  seed=$((SEED_START + done_count))
  echo "[shard $SHARD_ID] batch: seed-start=$seed n=$n (done $done_count/$TOTAL)"
  "$PY" scripts/generate_carnav.py --n-episodes "$n" --seed-start "$seed" \
      --shard-id "$SHARD_ID" --out-dir "$OUT_DIR" --max-steps 300 --append
  done_count=$((done_count + n))
done
echo "[shard $SHARD_ID] complete: $done_count/$TOTAL episodes"
