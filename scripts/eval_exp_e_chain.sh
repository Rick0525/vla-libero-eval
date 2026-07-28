#!/usr/bin/env bash
# Exp E evaluation chain (GPU2, overnight): the preregistered protocol for
# both retrain arms, in sequence.
# Per arm: 3x official 100-episode evals (seed 1000, n_action_steps=1 — the
# exact baseline protocol behind task5 7/30) + init3 probe x10 (rollout mode,
# base-seed 2000 = the Exp F seed batch; baseline 0/10 directly comparable).
# The chain waits for the boost arm's final checkpoint before scoring it, so
# a single tmux session covers the whole night.
set -uo pipefail  # no -e: a failed eval run must not silently kill the rest

source "$HOME/vla_lab/scripts/env.sh"
source /mnt/hdd16t/rick/vla_lab/vla-eval-harness/scripts/env.local.sh
cd /mnt/hdd16t/rick/vla_lab/vla-eval-harness
export CUDA_VISIBLE_DEVICES=2

run_arm() {
  local run="$1"
  for r in r1 r2 r3; do
    TRAIN_RUN="$run" CKPT_STEP=030000 EVAL_TAG="expe_${r}" \
      bash scripts/eval_checkpoint_spatial.sh \
      2>&1 | tee "$HOME/vla_lab/logs/eval_${run}_${r}_20260728.log"
  done
  python scripts/attribution_probe.py --mode rollout --task-id 5 --init-index 3 \
    --base-seed 2000 --n-rollouts 10 --train-run "$run" --ckpt-step 030000 \
    --out-dir "${VLA_EVAL_OUTPUT_DIR}/${run}_init3_probe" \
    2>&1 | tee "$HOME/vla_lab/logs/probe_init3_${run}_20260728.log"
}

run_arm smolvla_spatial_exp_e_ctrl_20260728

BOOST_CK=/mnt/hdd16t/rick/vla_lab/train_runs/smolvla_spatial_exp_e_boost_k4_20260728/checkpoints/030000/pretrained_model/model.safetensors
until [ -f "$BOOST_CK" ]; do sleep 120; done
sleep 60  # let the checkpoint finish writing

run_arm smolvla_spatial_exp_e_boost_k4_20260728
echo "EXP_E_EVAL_CHAIN_DONE"
