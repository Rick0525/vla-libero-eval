#!/usr/bin/env bash
# EGL-spread diagnostic: is the async-b10 vector step bottlenecked by 10
# worker EGL contexts time-slicing a single GPU?
# Background (2026-07-28 async validation): asyncb10 bought only ~1.3-2x over
# serial sync-b1 even on an idle machine, and robosuite's EGL fallback pins
# every worker's rendering to the inference GPU (CUDA_VISIBLE_DEVICES=0).
# Three legs, all otherwise the official protocol on the same checkpoint
# (n_action_steps=1, 10 eps/task, seed 1000, batch 10, async on, GPU0
# inference):
#   egl0_r1   - all worker rendering pinned to GPU0 (replicates yesterday)
#   egl12_r1  - rendering spread over GPU1+GPU2, GPU0 free for inference
#   egl012_r1 - rendering spread over all three GPUs
# Verdict rule: materially higher per-step throughput with spreading confirms
# EGL serialization as the bottleneck; flat timing moves suspicion to
# CPU-side/IPC costs. Timing via eval_info.json eval_s plus in-log it/s;
# success rates are recorded but individual episodes are not comparable
# (stack is not episode-deterministic).
# GPU utilization is sampled machine-wide for the whole session so the render
# load's location is directly visible per leg.
set -uo pipefail  # no -e: a failed leg must not kill the rest

source "$HOME/vla_lab/scripts/env.sh"
source /mnt/hdd16t/rick/vla_lab/vla-eval-harness/scripts/env.local.sh
cd /mnt/hdd16t/rick/vla_lab/vla-eval-harness
export CUDA_VISIBLE_DEVICES=0

RUN=smolvla_spatial_exp_e_ctrl_20260728
STAMP=20260729

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader -l 2 \
  > "$HOME/vla_lab/logs/egl_spread_gpusample_${STAMP}.csv" &
SMI_PID=$!

for leg in "egl0:0" "egl12:1,2" "egl012:0,1,2"; do
  tag="${leg%%:*}"
  devs="${leg#*:}"
  echo "=== ${tag} (EGL devices ${devs}) start $(date) ==="
  TRAIN_RUN="$RUN" CKPT_STEP=030000 EVAL_BATCH_SIZE=10 EVAL_USE_ASYNC=true \
    EVAL_EGL_DEVICES="$devs" EVAL_ENTRY="python scripts/lerobot_eval_eglspread.py" \
    EVAL_TAG="${tag}_r1" bash scripts/eval_checkpoint_spatial.sh \
    2>&1 | tee "$HOME/vla_lab/logs/eval_${RUN}_${tag}_r1_${STAMP}.log"
  echo "=== ${tag} end $(date) ==="
done

kill "$SMI_PID" 2>/dev/null
echo "EGL_SPREAD_VALIDATION_DONE"
