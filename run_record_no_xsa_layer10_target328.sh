#!/usr/bin/env bash
set -euo pipefail

cd /root/xsa_gate_ablation
source .venv/bin/activate

run_id=record-no-xsa-layer10-target328
export DATA_PATH=/root/xsa_gate_ablation
export LOG_DIR=/root/xsa_gate_ablation/logs
export CHECKPOINT_DIR=/root/xsa_gate_ablation/checkpoints
export TORCHINDUCTOR_CACHE_DIR=/root/xsa_gate_ablation/torchinductor/${run_id}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "RUN_START ${run_id} $(date -Is)"
torchrun --standalone --nproc_per_node=1 train_gpt.py \
  --xsa-mode record \
  --xsa-disable-layers 10 \
  --num-extension-iterations 75 \
  --max-train-steps 1450 \
  --val-loss-every 250 \
  --val-every-after-step 1385 \
  --stop-val-loss-below 3.28 \
  --save-checkpoint \
  --run-id "$run_id" \
  2>&1 | tee "/root/xsa_gate_ablation/logs/${run_id}.console.log"
echo "RUN_DONE ${run_id} $(date -Is)"
