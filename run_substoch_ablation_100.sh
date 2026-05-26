#!/usr/bin/env bash
set -euo pipefail

cd /root/xsa_gate_ablation
source .venv/bin/activate

export DATA_PATH=/root/xsa_gate_ablation
export LOG_DIR=/root/xsa_gate_ablation/logs
export CHECKPOINT_DIR=/root/xsa_gate_ablation/checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XSA_INTERP=0

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" /root/xsa_gate_ablation/torchinductor

for mode in record substoch substoch-delta; do
  run_id=substoch-ablation-${mode}-100
  export TORCHINDUCTOR_CACHE_DIR=/root/xsa_gate_ablation/torchinductor/${run_id}
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
  echo "RUN_START ${run_id} $(date -Is)"
  torchrun --standalone --nproc_per_node=1 train_gpt.py \
    --max-train-steps 100 \
    --skip-warmup \
    --val-loss-every 100 \
    --xsa-mode "$mode" \
    --run-id "$run_id"
  echo "RUN_DONE ${run_id} $(date -Is)"
done
