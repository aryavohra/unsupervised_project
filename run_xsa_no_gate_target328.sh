#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/modded-nanogpt/results/codex_xsa_no_gate_target328_$(date -u +%Y%m%dT%H%M%SZ)"
CODE_DIR="$ROOT_DIR/code"
mkdir -p "$CODE_DIR" "$ROOT_DIR/logs" "$ROOT_DIR/checkpoints" "$ROOT_DIR/torchinductor"
cp /root/modded-nanogpt/train_gpt.py /root/modded-nanogpt/triton_kernels.py /root/modded-nanogpt/xsa_interp.py "$CODE_DIR/"

cd "$CODE_DIR"

export DATA_PATH=/root/modded-nanogpt
export LOG_DIR="$ROOT_DIR/logs"
export CHECKPOINT_DIR="$ROOT_DIR/checkpoints"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/torchinductor/xsa-no-gate-record"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

{
  echo "RUN_DIR=$ROOT_DIR"
  echo "CODE_DIR=$CODE_DIR"
  date -u
  /opt/conda/envs/nanogpt210/bin/python - <<'PY'
import torch, triton, kernels
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("triton", triton.__version__)
print("kernels", kernels.__file__)
PY
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "===== RUN xsa record, sparse attention gate disabled ====="
} | tee "$ROOT_DIR/launch.log"

/opt/conda/envs/nanogpt210/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  train_gpt.py \
  --xsa-mode record \
  --disable-attn-gate \
  --num-extension-iterations 1125 \
  --max-train-steps 2500 \
  --val-loss-every 250 \
  --val-every-after-step 1385 \
  --stop-val-loss-below 3.28 \
  --run-id codex-xsa-no-gate-target328 \
  2>&1 | tee "$ROOT_DIR/console.log"

echo "RUN_DIR=$ROOT_DIR" | tee -a "$ROOT_DIR/launch.log"
tail -n 120 "$ROOT_DIR/logs/codex-xsa-no-gate-target328.txt" | tee "$ROOT_DIR/final_tail.txt"
