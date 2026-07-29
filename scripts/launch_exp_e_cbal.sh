#!/usr/bin/env bash
# Exp E-C launcher (counterbalanced boost arm, dataset local/libero_task5boost_k4_bal).
# `arm` starts the WHOLE chain as three tmux sessions at once — probe-gated
# training (GPU0+1), an eval chain that waits for the final checkpoint (GPU2),
# and a curve daemon that waits for training to end (GPU0). No human trigger
# points (7/28 lesson: every "after X do Y" needs a process already waiting).
# Usage: launch_exp_e_cbal.sh arm|train|eval|curve
set -uo pipefail
LEG="${1:?usage: launch_exp_e_cbal.sh arm|train|eval|curve}"

source "$HOME/vla_lab/scripts/env.sh"
source /mnt/hdd16t/rick/vla_lab/vla-eval-harness/scripts/env.local.sh
cd /mnt/hdd16t/rick/vla_lab/vla-eval-harness

RUN=smolvla_spatial_exp_e_cbal_20260729
DS=local/libero_task5boost_k4_bal
FINAL_CK="${VLA_TRAIN_OUTPUT_DIR}/${RUN}/checkpoints/030000/pretrained_model/model.safetensors"
STAMP=20260729

case "$LEG" in
  arm)
    tmux new-session -d -s exp_e_cbal_train \
      "bash $PWD/scripts/launch_exp_e_cbal.sh train 2>&1 | tee $HOME/vla_lab/logs/train_exp_e_cbal_${STAMP}.log"
    tmux new-session -d -s exp_e_cbal_eval \
      "bash $PWD/scripts/launch_exp_e_cbal.sh eval 2>&1 | tee $HOME/vla_lab/logs/eval_exp_e_cbal_chain_${STAMP}.log"
    tmux new-session -d -s exp_e_cbal_curve \
      "bash $PWD/scripts/launch_exp_e_cbal.sh curve 2>&1 | tee $HOME/vla_lab/logs/curve_daemon_exp_e_cbal_${STAMP}.log"
    tmux ls
    ;;
  train)
    export CUDA_VISIBLE_DEVICES=0,1
    # Probe gate (house rule): 300-step full-pipeline check on the new dataset.
    export RUN_TAG=exp_e_cbal_probe DATASET_REPO="$DS"
    export TRAIN_STEPS=300 SAVE_FREQ=300 LOG_FREQ=50
    rm -rf "${VLA_TRAIN_OUTPUT_DIR}/smolvla_spatial_${RUN_TAG}"
    bash scripts/train_smolvla_spatial_b64.sh
    if [ ! -d "${VLA_TRAIN_OUTPUT_DIR}/smolvla_spatial_exp_e_cbal_probe/checkpoints/000300" ]; then
      echo "CBAL_PROBE_FAILED — real training NOT started"; exit 1
    fi
    echo "CBAL_PROBE_OK — launching real training"
    unset TRAIN_STEPS SAVE_FREQ LOG_FREQ
    export RUN_TAG=exp_e_cbal_20260729
    bash scripts/train_smolvla_spatial_b64.sh || { echo "CBAL_TRAIN_FAILED"; exit 1; }
    echo "CBAL_TRAIN_DONE"
    ;;
  eval)
    export CUDA_VISIBLE_DEVICES=2
    until [ -f "$FINAL_CK" ]; do
      if ! tmux has-session -t exp_e_cbal_train 2>/dev/null && [ ! -f "$FINAL_CK" ]; then
        echo "CBAL_EVAL_ABORT — training session gone without final checkpoint"; exit 1
      fi
      sleep 120
    done
    sleep 60  # let the checkpoint finish writing
    for r in r1 r2 r3; do
      TRAIN_RUN="$RUN" CKPT_STEP=030000 EVAL_TAG="expe_${r}" \
        bash scripts/eval_checkpoint_spatial.sh \
        2>&1 | tee "$HOME/vla_lab/logs/eval_${RUN}_${r}_${STAMP}.log"
    done
    python scripts/attribution_probe.py --mode rollout --task-id 5 --init-index 3 \
      --base-seed 2000 --n-rollouts 10 --train-run "$RUN" --ckpt-step 030000 \
      --out-dir "${VLA_EVAL_OUTPUT_DIR}/${RUN}_init3_probe" \
      2>&1 | tee "$HOME/vla_lab/logs/probe_init3_${RUN}_${STAMP}.log"
    echo "EXP_E_CBAL_EVAL_DONE"
    ;;
  curve)
    # Wait for training to end (GPU0 freed), then score the checkpoint grid.
    while tmux has-session -t exp_e_cbal_train 2>/dev/null; do sleep 120; done
    if [ ! -f "$FINAL_CK" ]; then
      echo "CBAL_CURVE_ABORT — no final checkpoint"; exit 1
    fi
    # A DONE marker must never fire on failure (7/29 lesson: the first run of
    # this leg printed DONE after a file-not-found, and the monitor believed it).
    bash scripts/launch_exp_e_curves.sh "$RUN" 0 || { echo "CBAL_CURVE_FAILED"; exit 1; }
    echo "CBAL_CURVE_DONE"
    ;;
esac
