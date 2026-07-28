#!/usr/bin/env bash
# Async/parallel-env validation (GPU0): does batch_size=10 change eval RESULTS,
# and how much wall-clock does it buy, under the official protocol?
# Discovery behind this run: in this lerobot the async switch defaults to TRUE
# and batch_size=1 is what forces SyncVectorEnv — so the speed knob is batch
# size, not the async flag. W1's "b10 gives no speedup" was measured on the
# old conda stack and doesn't bind here.
# Legs (all: ctrl arm ckpt 030000, n_action_steps=1, 10 eps/task, seed 1000):
#   asyncb10_r1-r3 — batch 10, async on (3 runs = same-seed noise band)
#   syncb10_r1     — batch 10, async off (isolates batch vs async axes)
# Reference leg already on disk: tonight's official sync-b1 x3 (87/84/81).
# Verdict rule (preregistered): asyncb10 overall & per-task must sit inside
# the sync-b1 three-run noise band; wall-clock compared via eval_s.
set -uo pipefail  # no -e: a failed leg must not kill the rest

source "$HOME/vla_lab/scripts/env.sh"
source /mnt/hdd16t/rick/vla_lab/vla-eval-harness/scripts/env.local.sh
cd /mnt/hdd16t/rick/vla_lab/vla-eval-harness
export CUDA_VISIBLE_DEVICES=0

# GPU0 belongs to the ctrl-arm curve daemon until it exits.
while tmux has-session -t exp_e_curve_ctrl 2>/dev/null; do sleep 60; done

RUN=smolvla_spatial_exp_e_ctrl_20260728

for r in r1 r2 r3; do
  echo "=== asyncb10 ${r} start $(date) ==="
  TRAIN_RUN="$RUN" CKPT_STEP=030000 EVAL_BATCH_SIZE=10 EVAL_USE_ASYNC=true \
    EVAL_TAG="asyncb10_${r}" bash scripts/eval_checkpoint_spatial.sh \
    2>&1 | tee "$HOME/vla_lab/logs/eval_${RUN}_asyncb10_${r}_20260728.log"
  echo "=== asyncb10 ${r} end $(date) ==="
done

echo "=== syncb10 r1 start $(date) ==="
TRAIN_RUN="$RUN" CKPT_STEP=030000 EVAL_BATCH_SIZE=10 EVAL_USE_ASYNC=false \
  EVAL_TAG="syncb10_r1" bash scripts/eval_checkpoint_spatial.sh \
  2>&1 | tee "$HOME/vla_lab/logs/eval_${RUN}_syncb10_r1_20260728.log"
echo "=== syncb10 r1 end $(date) ==="

echo "ASYNC_VALIDATION_DONE"
